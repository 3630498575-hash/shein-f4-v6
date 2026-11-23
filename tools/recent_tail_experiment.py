"""Compare visible deadline outcomes and order-entry waiting times for F4."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from feature_groups_experiment import predict as province_predict
from prospective_policy import (COLD, FEATURES, OUT as PREVIOUS, ROOT, WINDOWS,
                                apply_actions, fit_actions, fold_inputs, forecasts, history, prepare)
from run_province_release import group_metrics
from score import score


OUT = ROOT / 'output/recent_tail'
CENTERS = ['half_center', 'joint_center', 'joint_center_protect']
METHODS = ['marginal_12', 'recent_half', 'recent_protect', 'wait', 'wait_protect'] + CENTERS
HOURS = np.arange(0., 721., 12.)
KEYS = ['lgs_company_nm', 'shpp_chnl']
METRICS = ['A', 'B', 'ME', 'pen_ontime', 'pen_severe']


def visible_sample(data, cutoff):
    # The oldest needed issue is 30 days (720h) + 14 days before cutoff.
    known = data.loc[(data.issue_time >= cutoff - pd.Timedelta(days=44))
                     & (data.issue_time < cutoff) & (data.pay_time < cutoff) & (data.elapsed >= 0),
                     KEYS + ['issue_time', 'sign_time']]
    known = known.sample(n=min(len(known), 420000), random_state=42).copy()
    known['age'] = (cutoff - known.issue_time).dt.total_seconds() / 3600
    observed = known.sign_time.where(known.sign_time < cutoff)
    known['duration'] = ((observed - known.issue_time).dt.total_seconds() / 3600).fillna(np.inf)
    return known.loc[known.duration >= 0]


def recent_delta(data, train, joined, cutoff, horizons):
    recent = visible_sample(data, cutoff)
    old_groups = train.groupby(KEYS, observed=True).indices
    recent_groups = recent.groupby(KEYS, observed=True).indices
    delta = np.zeros(horizons.shape, dtype='float32')
    for key, rows in joined.groupby(KEYS, observed=True).indices.items():
        old_idx = old_groups.get(key)
        new_idx = recent_groups.get(key)
        if old_idx is None or new_idx is None or min(len(old_idx), len(new_idx)) < 100:
            continue
        old = np.sort(train.remaining.to_numpy()[old_idx])
        old_rates = 1 - np.searchsorted(old, HOURS, side='right') / len(old)
        age = recent.age.to_numpy()[new_idx]
        duration = recent.duration.to_numpy()[new_idx]
        changes = np.zeros(len(HOURS))
        for j, h in enumerate(HOURS):
            due = (age > h) & (age <= h + 14 * 24)
            count = int(due.sum())
            if count >= 100:
                changes[j] = ((duration[due] > h).sum() - count * old_rates[j]) / (count + 2000)
        delta[rows] = np.interp(horizons[rows], np.append(HOURS, 744.), np.append(changes, 0.))
    return delta


def wait_risks(train, joined, horizons, prior):
    def strata(frame):
        return frame.groupby(KEYS + ['province', np.digitize(frame.elapsed, [6, 12, 24, 48])],
                             observed=True, sort=False).indices
    old = strata(train)
    result = prior.copy()
    for key, rows in strata(joined).items():
        idx = old.get(key)
        if idx is None or len(idx) < 500:
            continue
        values = np.sort(train.remaining.to_numpy()[idx])
        risk = 1 - np.searchsorted(values, horizons[rows], side='right') / len(values)
        weight = len(values) / (len(values) + 2000)
        result[rows] = weight * risk + (1 - weight) * prior[rows]
    return result


def adjust(risk, delta, horizons, upward_only):
    change = np.maximum(delta, 0.) if upward_only else delta
    corrected = np.clip(risk + .5 * change, 0, 1)
    order = np.argsort(horizons, axis=1, kind='stable')
    ordered = np.take_along_axis(corrected, order, axis=1)
    ordered = np.maximum.accumulate(ordered[:, ::-1], axis=1)[:, ::-1]
    np.put_along_axis(corrected, order, ordered, axis=1)
    return corrected


def joint_wait_center(train, joined, prior):
    def with_key(frame):
        band = np.digitize(frame.elapsed, [6, 12, 24, 48], right=True)
        key = frame.province.str.len().astype(str) + ':' + frame.province + band.astype(str)
        return frame.assign(province_wait=key)
    joint, _ = province_predict(with_key(train), with_key(joined), prior, 'province_wait')
    return joint


def predict_fold(data, start, orders, methods):
    train, joined, n = fold_inputs(data, start, orders)
    promises, base, expected = forecasts(train, joined)
    horizons = np.column_stack([promises, promises + 24]) - joined.elapsed.clip(lower=0).to_numpy()[:, None]
    if any(m in methods for m in ['recent_half', 'recent_protect', 'wait_protect']):
        delta = recent_delta(data, train, joined, pd.Timestamp(start), horizons)
    if any(m in methods for m in ['wait', 'wait_protect']):
        waiting = wait_risks(train, joined, horizons, base)
    carrier = joined.lgs_company_nm.to_numpy()
    found = {}
    for method in methods:
        if method in CENTERS:
            continue
        risk = waiting if method.startswith('wait') else base
        if method in ['recent_half', 'recent_protect', 'wait_protect']:
            risk = adjust(risk, delta, horizons, upward_only=method.endswith('protect'))
        rules = fit_actions(carrier[:n], promises[:n], risk[:n], expected[:n], 'marginal', 12)
        found[method] = apply_actions(carrier[n:], risk[n:], promises[n:], rules, 'marginal')
    if any(m in CENTERS for m in methods):
        prior = pd.Series(promises[:, 2], index=joined.index)
        joint = joint_wait_center(train, joined, prior)
        for label, center in [('half_center', (prior + joint) / 2), ('joint_center', joint)]:
            selected = [m for m in methods if m == label or m == label + '_protect']
            if not selected:
                continue
            choices, risk, expectation = forecasts(train, joined, center)
            for method in selected:
                prob = risk
                if method.endswith('_protect'):
                    h = np.column_stack([choices, choices + 24]) - joined.elapsed.clip(lower=0).to_numpy()[:, None]
                    prob = adjust(risk, recent_delta(data, train, joined, pd.Timestamp(start), h), h, True)
                rules = fit_actions(carrier[:n], choices[:n], prob[:n], expectation[:n], 'marginal', 12)
                found[method] = apply_actions(carrier[n:], prob[n:], choices[n:], rules, 'marginal')
    return found, len(train)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['replay', 'candidate'])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    data = history()
    if args.stage == 'candidate':
        name = json.loads((OUT / 'selection.json').read_text())['winner']
        if name == 'marginal_12':
            print('Previous candidate retained.')
            return
        test = prepare(pd.read_csv(ROOT / 'data/test_202609.csv', usecols=['pkg_no'] + FEATURES))
        pred, train_rows = predict_fold(data, '2026-09-21', test, [name])
        p = pred[name]
        if (len(test) != 911415 or test.pkg_no.isna().any() or not test.pkg_no.is_unique
                or not np.isfinite(p).all() or (p < np.maximum(1., test.elapsed)).any()):
            raise ValueError('Test predictions failed ID or bounds checks.')
        dest = OUT / 'candidate.csv'
        pd.DataFrame({'pkg_no': test.pkg_no, 'pred_hours': p}).to_csv(dest, index=False)
        meta = dict(method=name, train_rows=train_rows, rows=len(test), mean_hours=float(p.mean()),
                    sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        (OUT / 'candidate.json').write_text(json.dumps(meta, indent=2))
        print(meta)
        return
    scores, groups = [], []
    for start, end in [*WINDOWS, COLD]:
        valid = data.loc[(data.pay_time >= start) & (data.pay_time < end)]
        results, train_rows = predict_fold(data, start, valid, METHODS)
        for name, p in results.items():
            if not np.isfinite(p).all() or (p < np.maximum(1., valid.elapsed)).any():
                raise ValueError('Invalid predictions')
            metrics = score(valid.lgs_company_nm, valid.y_total_hours, p)
            scores.append(dict(cutoff=start, method=name, valid_rows=len(p), train_rows=train_rows, **metrics))
            groups.append(group_metrics(valid, pd.Series(p, index=valid.index), start, name))
            print(start, name, round(metrics['A'], 4), flush=True)
    scores = pd.DataFrame(scores)
    prev = pd.concat([pd.read_csv(PREVIOUS / f'{s}_scores.csv') for s in ['development', 'cold']])
    a = scores.query("method == 'marginal_12'").set_index('cutoff').sort_index()
    b = prev.query("method == 'marginal_12'").set_index('cutoff').sort_index()
    np.testing.assert_allclose(a[METRICS], b[METRICS], rtol=0, atol=1e-8)
    panel = scores.pivot(index='cutoff', columns='method', values='A')
    means = panel.drop(COLD[0]).mean()
    eligible = means[panel.loc[COLD[0]] <= panel.loc[COLD[0], 'marginal_12']]
    winner = eligible.idxmin()
    scores.to_csv(OUT / 'scores.csv', index=False)
    pd.concat(groups).to_csv(OUT / 'companies.csv', index=False)
    (OUT / 'selection.json').write_text(json.dumps(dict(winner=winner, means=means.to_dict(),
                                                        cold=panel.loc[COLD[0]].to_dict()), indent=2))
    print('Selected:', winner, flush=True)


if __name__ == '__main__':
    main()
