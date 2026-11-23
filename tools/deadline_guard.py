"""Re-evaluate the six-hour candidate using deadlines visible at the fold cutoff."""
import argparse
import hashlib
import itertools
import json

import numpy as np
import pandas as pd

from carrier_allocation import OUT as PREVIOUS, allocation_predictions, check_bounds
from distribution_promise_experiment import empirical_cdf
from prospective_policy import COLD, FEATURES, ROOT, WINDOWS, fold_inputs, history, prepare
from recent_tail_experiment import METRICS, adjust, recent_delta
from run_province_release import group_metrics
from score import score


OUT = ROOT / 'output/deadline_guard'
BASE = 'price_half'
OFFSETS = np.array([0., -6., 6., 12.])
METHODS = [BASE, 'extend', 'local', 'local_half', 'early_extend', 'early_local', 'early_half']
N_BANDS = 4


def calendar_predictions(train, joined, cutoff, base):
    # Use complete Monday-Sunday issue weeks before the maturity cutoff.
    columns = ['lgs_company_nm', 'shpp_chnl', 'issue_time', 'remaining']
    d = train.loc[train.issue_time < cutoff - pd.Timedelta(days=21), columns].copy()
    day = d.issue_time.dt.normalize()
    d['weekday'] = day.dt.day_of_week
    d['week'] = day - pd.to_timedelta(d.weekday, unit='D')
    d['weekend'] = (d.weekday >= 5).astype(int)
    keys = columns[:2]
    for key in keys:
        d[key] = d[key].astype('string')
    week_keys = keys + ['week']
    weekly = d.groupby(week_keys, observed=True)
    complete = weekly.weekday.nunique().eq(7)
    full = complete[complete].index
    reference = weekly.remaining.quantile(.97).reindex(full).rename('week_tail')
    result = {}
    for period, minimum in [('weekday', 100), ('weekend', 200)]:
        grouped = d.groupby(week_keys + [period], observed=True).remaining
        stats = pd.concat([grouped.size().rename('count'), grouped.quantile(.97).rename('tail')], axis=1)
        stats = stats.join(reference, on=week_keys).dropna(subset=['week_tail'])
        stats = stats.loc[stats['count'] >= minimum].reset_index()
        stats['change'] = (stats['tail'] - stats['week_tail']) * stats['count'] / (stats['count'] + 500)
        summary = stats.groupby(keys + [period], observed=True).change.agg(['size', 'min', 'max', 'median'])
        consistent = (summary['size'] >= 2) & ((summary['min'] > 0) | (summary['max'] < 0))
        effects = summary.loc[consistent, 'median'].clip(-12, 12)
        dow = joined.issue_time.dt.day_of_week
        lookup = pd.MultiIndex.from_arrays([joined[k].astype('string') for k in keys]
                                           + [dow if period == 'weekday' else (dow >= 5).astype(int)])
        change = effects.reindex(lookup).fillna(0.).to_numpy()
        floor = np.maximum(1, joined.elapsed.to_numpy())
        result[period + '_half'] = np.maximum(floor, base + .5 * change)
        result[period + '_extend'] = base + np.maximum(0, change)
    return result


def schedule(orders, base):
    elapsed = orders.elapsed.to_numpy().clip(min=0)
    choices = np.maximum(base[:, None] + OFFSETS, np.maximum(1, elapsed)[:, None])
    bands = np.digitize(base - elapsed, [168, 240, 336])
    return choices, bands


def observed_events(cal, cutoff, base):
    choices, bands = schedule(cal, base)
    age = (cutoff - cal.pay_time).dt.total_seconds().to_numpy() / 3600
    end = cal.sign_time.where(cal.sign_time < cutoff)
    duration = (end - cal.pay_time).dt.total_seconds().fillna(np.inf).to_numpy() / 3600
    # Common eligibility gives every action exactly the same mature cohort.
    last_deadline = choices.max(axis=1) + 24
    mature = (age > last_deadline) & (age <= last_deadline + 14 * 24)
    checkpoint = np.maximum(base - 72, cal.elapsed.clip(lower=0).to_numpy())
    early = (age > checkpoint) & (age <= checkpoint + 7 * 24)
    still_waiting = duration > checkpoint
    late = duration[:, None] > np.column_stack([choices, choices + 24])
    return bands, mature, early, still_waiting, late


def choose_offsets(count, cost, risk, support, allow_shortening):
    active = [0, 1, 2, 3] if allow_shortening else [0, 2, 3]
    options = [active if n and seen >= 500 else [0] for n, seen in zip(count, support)]
    bands = np.arange(N_BANDS)
    weight = count / count.sum()
    def loss(choices):
        late = weight @ risk[bands, choices]
        severe = weight @ risk[bands, choices + len(OFFSETS)]
        return cost[bands, choices].sum() + 500 * (max(0., late - .10) + max(0., severe - .03))
    candidates = (np.array(c) for c in itertools.product(*options))
    return min(candidates, key=lambda c: (loss(c), np.abs(OFFSETS[c]).sum()))


def predict(data, start, orders, include_blends=False):
    cut = pd.Timestamp(start)
    train, joined, n = fold_inputs(data, start, orders)
    # Disjoint payment periods for maturity-filtered training and calibration.
    # The counterfactual historical predictions here are made with today's model.
    cal = data.loc[(data.pay_time >= cut - pd.Timedelta(days=21)) & (data.pay_time < cut)
                   & (data.issue_time < cut) & (data.elapsed >= 0)]
    cal = cal.sample(min(300000, len(cal)), random_state=45)
    combined = pd.concat([joined, cal[joined.columns]], ignore_index=True)
    parent_predictions = allocation_predictions(data, start, train, combined, n)
    parent = parent_predictions[BASE]
    ref = joined.iloc[:n]
    choices, bands = schedule(joined, parent[:len(joined)])
    ref_horizons = np.column_stack([choices[:n], choices[:n] + 24]) - ref.elapsed.clip(lower=0).to_numpy()[:, None]
    cdf, mean = empirical_cdf(train, ref, ref_horizons, include_route=True)
    model_risk = adjust(1 - cdf, recent_delta(data, train, ref, cut, ref_horizons), ref_horizons, True)
    if choices[:n].min(axis=1).mean() < (mean + ref.elapsed.clip(lower=0)).mean():
        raise ValueError('Reference mean-gap sign uncertified.')
    cb, mature, early, waiting, actual = observed_events(cal, cut, parent[len(joined):])
    cgroups = cal.groupby('lgs_company_nm', observed=True).indices
    found = {name: parent[:len(joined)].copy() for name in METHODS}
    log = []
    for company, rows in joined.groupby('lgs_company_nm', observed=True).indices.items():
        cr = cgroups.get(company)
        reference = rows[rows < n]
        if cr is None or len(reference) < 500:
            continue
        count = np.zeros(N_BANDS, dtype=int)
        support = count.copy()
        cost = np.zeros((N_BANDS, len(OFFSETS)))
        risk = np.zeros((N_BANDS, 2 * len(OFFSETS)))
        trend_risk = risk.copy()
        for k in range(N_BANDS):
            r = reference[bands[reference] == k]
            count[k] = len(r)
            if not len(r):
                continue
            c = cr[cb[cr] == k]
            m, e = c[mature[c]], c[early[c]]
            support[k] = len(m)
            prior = model_risk[r].mean(axis=0, dtype=float)
            risk[k] = (actual[m].sum(axis=0) + 2000 * prior) / (len(m) + 2000)
            ratio = 1.
            if min(len(m), len(e)) >= 500 and waiting[m].sum() > 0:
                earlier = waiting[m].mean()
                recent = (waiting[e].sum() + 2000 * earlier) / (len(e) + 2000)
                ratio = np.clip(recent / earlier, 1., 1.5)
            trend_risk[k] = np.minimum(1., risk[k] * ratio)
            cost[k] = (choices[r] - choices[r, :1]).sum(axis=0) / n
            log.append(dict(cutoff=start, company=company, band=k, mature=len(m), recent=len(e),
                            reference=len(r), multiplier=ratio, model_severe=prior[4],
                            estimated_severe=risk[k, 4], early_estimated_severe=trend_risk[k, 4]))
        for name, estimates, shorten in [('extend', risk, False), ('local', risk, True),
                                          ('early_extend', trend_risk, False), ('early_local', trend_risk, True)]:
            action = choose_offsets(count, cost, estimates, support, shorten)
            found[name][rows] = choices[rows, action[bands[rows]]]
    for half, full in [('local_half', 'local'), ('early_half', 'early_local')]:
        found[half] = (found[BASE] + found[full]) / 2
    found.update(calendar_predictions(train, joined, cut, found[BASE]))
    if include_blends:
        full = parent_predictions['price'][:len(joined)]
        found['blend_full'] = (found['local_half'] + full) / 2
        found['blend_base'] = (found['local_half'] + found[BASE]) / 2
        found['blend_limited'] = (found['blend_full'] + found['blend_base']) / 2
    return {method: pred[n:] for method, pred in found.items()}, log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['replay', 'candidate'])
    stage = parser.parse_args().stage
    OUT.mkdir(exist_ok=True)
    data = history()
    if stage == 'candidate':
        name = json.loads((OUT / 'selection.json').read_text())['winner']
        if name == BASE:
            print('Incumbent retained:', PREVIOUS / 'candidate.csv')
            return
        orders = prepare(pd.read_csv(ROOT / 'data/test_202609.csv', usecols=['pkg_no'] + FEATURES))
        found, log = predict(data, '2026-09-21', orders)
        prev = pd.read_csv(PREVIOUS / 'candidate.csv')
        if (len(orders) != 911415 or orders.pkg_no.hasnans or not orders.pkg_no.is_unique
                or not orders.pkg_no.equals(prev.pkg_no)):
            raise ValueError('Test ID mismatch.')
        np.testing.assert_allclose(found[BASE], prev.pred_hours, rtol=0, atol=1e-8)
        p = found[name]
        check_bounds(p, orders)
        dest = OUT / 'candidate.csv'
        pd.DataFrame({'pkg_no': orders.pkg_no, 'pred_hours': p}).to_csv(dest, index=False)
        info = dict(method=name, rows=len(p), mean_hours=p.mean(),
                    sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        (OUT / 'candidate.json').write_text(json.dumps(info, indent=2))
        pd.DataFrame(log).to_csv(OUT / 'candidate_calibration.csv', index=False)
        print(info)
        return
    scores, groups, calibration = [], [], []
    for start, end in [*WINDOWS, COLD]:
        valid = data.loc[(data.pay_time >= start) & (data.pay_time < end)]
        found, log = predict(data, start, valid)
        calibration.extend(log)
        for method, p in found.items():
            check_bounds(p, valid)
            result = score(valid.lgs_company_nm, valid.y_total_hours, p)
            scores.append(dict(cutoff=start, method=method, rows=len(p), **result))
            groups.append(group_metrics(valid, pd.Series(p, index=valid.index), start, method))
            print(start, method, round(result['A'], 4), flush=True)
    s = pd.DataFrame(scores)
    def parent_stats(frame):
        return frame[frame.method == BASE].set_index('cutoff').sort_index()[METRICS]
    pd.testing.assert_frame_equal(parent_stats(s), parent_stats(pd.read_csv(PREVIOUS / 'scores.csv')),
                                  rtol=0, atol=1e-8)
    table = s.pivot(index='cutoff', columns='method', values='A')
    means = table.drop(COLD[0]).mean()
    eligible = means[table.loc[COLD[0]] <= table.at[COLD[0], BASE]]
    winner = eligible.idxmin() if eligible.min() < means[BASE] else BASE
    s.to_csv(OUT / 'scores.csv', index=False)
    pd.concat(groups).to_csv(OUT / 'companies.csv', index=False)
    pd.DataFrame(calibration).to_csv(OUT / 'calibration.csv', index=False)
    (OUT / 'selection.json').write_text(json.dumps(dict(winner=winner, means=means.to_dict(),
                                                        cold=table.loc[COLD[0]].to_dict()), indent=2))
    print('Selected:', winner)


if __name__ == '__main__':
    main()
