"""Freeze carrier and risk-band actions on the previous week's order mix."""
import argparse
import hashlib
import itertools
import json

import numpy as np
import pandas as pd

from distribution_promise_experiment import WINDOWS, empirical_cdf
from feature_groups_experiment import predict as province_predict
from remaining_experiment import weighted_remaining_predict
from run_province_release import ROOT, group_metrics
from score import score


OUT = ROOT / 'output/prospective_policy'
FEATURES = ['lgs_company_nm', 'shpp_chnl', 'shpp_prvn_nm', 'route_cat', 'pay_time', 'issue_time']
OFFSETS = np.array([-24, -12, 0, 12, 24])
CONFIGS = {'carrier_12': ('carrier', 12), 'tail_12': ('tail', 12),
           'marginal_12': ('marginal', 12), 'marginal_24': ('marginal', 24)}
COLD = ('2026-08-23', '2026-08-25')


def prepare(frame):
    frame = frame.copy()
    for col in ['pay_time', 'issue_time']:
        frame[col] = pd.to_datetime(frame[col], errors='raise')
    frame['elapsed'] = (frame.issue_time - frame.pay_time).dt.total_seconds() / 3600
    frame['province'] = frame.shpp_prvn_nm.astype('string').fillna('MISSING')
    return frame


def history():
    frame = prepare(pd.read_parquet(ROOT / 'data/train_slim.parquet',
                                    columns=FEATURES + ['sign_time', 'y_total_hours']))
    frame['sign_time'] = pd.to_datetime(frame.sign_time, errors='raise')
    frame['remaining'] = frame.y_total_hours - frame.elapsed
    return frame


def forecasts(train, orders, base=None):
    if base is None:
        base, _ = province_predict(train, orders, weighted_remaining_predict(train, orders), 'province')
    elapsed = orders.elapsed.to_numpy().clip(min=0)
    promises = np.maximum(base.to_numpy()[:, None] + OFFSETS, np.maximum(1, elapsed)[:, None])
    horizons = np.column_stack((promises, promises + 24)) - elapsed[:, None]
    cdf, mean = empirical_cdf(train, orders, horizons, include_route=True)
    return promises, 1 - cdf, mean + elapsed


def risk_band(risks, kind):
    if kind == 'carrier':
        return np.zeros(len(risks), dtype=int)
    if kind == 'tail':
        return np.digitize(risks[:, 7], [.02, .04])
    # Percentage of additional late and severely late events between -12h and +12h.
    sensitivity = (risks[:, 1] - risks[:, 3]) + (risks[:, 6] - risks[:, 8])
    return np.digitize(sensitivity, [.02, .05])


def fit_actions(companies, promises, risks, expected, kind, cap):
    indices = np.flatnonzero(abs(OFFSETS) <= cap)
    # This certificate makes the absolute mean term linear for every candidate.
    # Stop if a different dataset would require a coupled optimization.
    if promises[:, indices[0]].mean() < expected.mean():
        raise ValueError('Mean-gap sign is not certified on the reference pool.')
    bands = risk_band(risks, kind)
    rules = {}
    for company, rows in pd.Series(companies).groupby(companies, observed=True).indices.items():
        groups = bands[rows]
        n = len(rows)
        costs = np.array([promises[rows[groups == k]].sum(axis=0) / len(bands) for k in range(3)])
        freq = np.array([risks[rows[groups == k]].sum(axis=0, dtype=float) / n for k in range(3)])
        def objective(choice):
            g = np.arange(3)
            return (costs[g, choice].sum()
                    + 500 * max(0, freq[g, choice].sum() - .10)
                    + 500 * max(0, freq[g, choice + len(OFFSETS)].sum() - .03))
        uniform = min(indices, key=lambda i: (objective(np.repeat(i, 3)), abs(OFFSETS[i])))
        counts = np.bincount(groups, minlength=3)
        options = [indices if count >= 500 else [uniform] for count in counts]
        choices = (np.array(t) for t in itertools.product(*options))
        best = min(choices, key=lambda c: (objective(c), np.abs(OFFSETS[c]).sum()))
        rules[company] = best
    return rules


def apply_actions(companies, risks, promises, rules, kind):
    bands = risk_band(risks, kind)
    actions = np.full(len(bands), 2)  # zero offset for unseen carriers
    for company, rows in pd.Series(companies).groupby(companies, observed=True).indices.items():
        if company in rules:
            actions[rows] = rules[company][bands[rows]]
    return promises[np.arange(len(bands)), actions]


def fold_inputs(data, start, orders):
    cut = pd.Timestamp(start)
    if orders.issue_time.isna().any() or (orders.issue_time < cut).any():
        raise ValueError('Frozen rules must be available when each order is issued.')
    train = data.loc[(data.pay_time < cut - pd.Timedelta(days=21)) & (data.sign_time < cut)
                     & (data.issue_time < cut) & (data.elapsed >= 0) & (data.remaining >= 0)]
    ref = data.loc[(data.issue_time < cut) & (data.issue_time >= cut - pd.Timedelta(days=7))
                   & (data.pay_time < cut), FEATURES + ['province', 'elapsed']]
    ref = ref.sample(n=min(160000, len(ref)), random_state=42)
    orders = orders[FEATURES + ['province', 'elapsed']]
    joined = pd.concat([ref, orders], ignore_index=True)
    return train, joined, len(ref)


def run_fold(data, start, orders, methods):
    train, joined, n = fold_inputs(data, start, orders)
    promises, risks, expected = forecasts(train, joined)
    carrier = joined.lgs_company_nm.to_numpy()
    found = {'v5': promises[n:, 2]}
    rules_log = []
    for method in methods:
        kind, cap = CONFIGS[method]
        rules = fit_actions(carrier[:n], promises[:n], risks[:n], expected[:n], kind, cap)
        found[method] = apply_actions(carrier[n:], risks[n:], promises[n:], rules, kind)
        for company, actions in rules.items():
            for band, action in enumerate(actions):
                rules_log.append(dict(cutoff=start, method=method, company=company, band=band,
                                      offset=int(OFFSETS[action])))
    return found, rules_log, len(train), n


def replay(data, windows, methods):
    scores, groups, rules = [], [], []
    for start, end in windows:
        valid = data.loc[(data.pay_time >= pd.Timestamp(start)) & (data.pay_time < pd.Timestamp(end))]
        pred, logs, n, ref = run_fold(data, start, valid, methods)
        rules.extend(logs)
        for name, p in pred.items():
            if not np.isfinite(p).all() or (p < np.maximum(1, valid.elapsed)).any():
                raise ValueError('Invalid prediction')
            result = score(valid.lgs_company_nm, valid.y_total_hours, p)
            scores.append(dict(cutoff=start, method=name, train_rows=n, reference_rows=ref,
                               valid_rows=len(valid), **result))
            groups.append(group_metrics(valid, pd.Series(p, index=valid.index), start, name))
            print(start, name, round(result['A'], 4), flush=True)
    return pd.DataFrame(scores), pd.concat(groups), pd.DataFrame(rules)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['development', 'cold', 'candidate'])
    stage = parser.parse_args().stage
    OUT.mkdir(exist_ok=True)
    data = history()
    lock = OUT / 'selection.json'
    if stage == 'candidate':
        name = json.loads(lock.read_text())['winner']
        test = prepare(pd.read_csv(ROOT / 'data/test_202609.csv', usecols=['pkg_no'] + FEATURES))
        pred, rules, train_rows, ref_rows = run_fold(data, '2026-09-21', test, [] if name == 'v5' else [name])
        p = pred[name]
        if (len(test) != 911415 or test.pkg_no.isna().any() or not test.pkg_no.is_unique
                or not np.isfinite(p).all() or (p < np.maximum(1, test.elapsed)).any()):
            raise ValueError('Candidate failed row, ID or prediction checks')
        dest = OUT / 'candidate.csv'
        pd.DataFrame({'pkg_no': test.pkg_no, 'pred_hours': p}).to_csv(dest, index=False)
        pd.DataFrame(rules).to_csv(OUT / 'candidate_rules.csv', index=False)
        meta = dict(method=name, train_rows=train_rows, reference_rows=ref_rows, rows=len(test),
                    mean_hours=float(p.mean()), sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        (OUT / 'candidate.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        print(meta, flush=True)
        return
    if stage == 'development':
        windows, methods = WINDOWS, list(CONFIGS)
    else:
        winner = json.loads(lock.read_text())['winner']
        windows, methods = [COLD], list(dict.fromkeys(['carrier_12', winner]))
        methods = [name for name in methods if name != 'v5']
    scores, groups, rules = replay(data, windows, methods)
    scores.to_csv(OUT / f'{stage}_scores.csv', index=False)
    groups.to_csv(OUT / f'{stage}_companies.csv', index=False)
    rules.to_csv(OUT / f'{stage}_rules.csv', index=False)
    if stage == 'development':
        baseline = pd.read_csv(ROOT / 'output/rolling_risk_scores.csv')
        reference = baseline.loc[baseline.method == 'v5'].set_index('cutoff').sort_index()
        current = scores.loc[scores.method == 'v5'].set_index('cutoff').sort_index()
        metrics = ['ME', 'pen_ontime', 'pen_severe', 'A', 'B']
        np.testing.assert_allclose(current[metrics], reference[metrics], atol=1e-8, rtol=0)
        ranking = scores.groupby('method').A.mean().sort_values()
        lock.write_text(json.dumps(dict(winner=ranking.index[0], metric='mean four-window A',
                                        development=ranking.to_dict(), cold_window=COLD), indent=2))
        print(ranking)


if __name__ == '__main__':
    main()
