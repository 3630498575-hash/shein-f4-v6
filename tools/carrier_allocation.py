"""Redistribute promise hours within a carrier around the recent-tail candidate."""
import argparse
import hashlib
import itertools
import json

import numpy as np
import pandas as pd

from prospective_policy import (COLD, FEATURES, ROOT, WINDOWS, apply_actions, fit_actions,
                                fold_inputs, forecasts, history, prepare, risk_band)
from recent_tail_experiment import METRICS, OUT as PARENT_DIR, adjust, joint_wait_center, recent_delta
from feature_groups_experiment import predict as province_predict
from remaining_experiment import weighted_remaining_predict
from run_province_release import group_metrics
from score import score


PARENT = 'joint_center_protect'
METHODS = [PARENT, 'balanced', 'balanced_alert', 'price', 'price_half', 'price_reserve']
OUT = ROOT / 'output/carrier_allocation'


def tail_rule(value, capacity, large):
    """Find the cutoff and proportional tie share for 20% of usable hours."""
    priority = -value if large else value
    order = np.argsort(priority)
    target = .2 * capacity.sum()
    index = np.searchsorted(np.cumsum(capacity[order]), target)
    cutoff = priority[order[index]]
    tied = priority == cutoff
    share = (target - capacity[priority < cutoff].sum()) / capacity[tied].sum()
    return cutoff, np.clip(share, 0, 1)


def tail_share(value, rule, large):
    cutoff, tied_share = rule
    priority = -value if large else value
    return (priority < cutoff) + tied_share * (priority == cutoff)


def balanced_delta(p, severe, ref):
    # Columns are stay, shorten and extend. Cap both directions at the same
    # usable hours so that clipping at order entry cannot distort the budget.
    down, up = p[:, 0] - p[:, 1], p[:, 2] - p[:, 0]
    capacity = np.minimum(down, up)
    available = ref & (capacity > 1e-9)
    if not available.any():
        return np.zeros(len(ref))
    gain = np.divide(severe[:, 0] - severe[:, 2], up, out=np.zeros(len(up)), where=up > 1e-9)
    cost = np.divide(severe[:, 1] - severe[:, 0], down, out=np.zeros(len(down)), where=down > 1e-9)
    add = tail_rule(gain[available], capacity[available], True)
    remove = tail_rule(cost[available], capacity[available], False)
    change = capacity * (tail_share(gain, add, True) - tail_share(cost, remove, False))
    np.testing.assert_allclose(change[ref].sum(), 0., atol=1e-6)
    return change


def choose_action(p, late, severe, prices):
    if prices is None:
        return np.zeros(len(p), dtype=int)
    return (p + prices[0] * late + prices[1] * severe).argmin(axis=1)


def price_rules(p, late, severe, carrier_share):
    # For a carrier comprising w of orders, one point of probability above
    # a penalty threshold costs 500/w hours per parcel in the local objective.
    rows = np.arange(len(p))
    scales = np.array([0., .125, .25, .5, .75, 1.]) * 500 / carrier_share
    best = {limit: (np.inf, None) for limit in [.03, .0275]}
    for prices in itertools.chain([None], itertools.product(scales, repeat=2)):
        action = choose_action(p, late, severe, prices)
        shared = carrier_share * p[rows, action].mean() + 500 * max(0, late[rows, action].mean() - .10)
        tail = severe[rows, action].mean()
        for limit, (loss, _) in best.items():
            objective = shared + 500 * max(0, tail - limit)
            if objective < loss:
                best[limit] = objective, prices
    return {limit: prices for limit, (_, prices) in best.items()}


def predict(data, start, orders):
    train, joined, n = fold_inputs(data, start, orders)
    predictions = allocation_predictions(data, start, train, joined, n)
    return {m: p[n:] for m, p in predictions.items()}


def allocation_predictions(data, start, train, joined, n):
    prior, _ = province_predict(train, joined, weighted_remaining_predict(train, joined), 'province')
    center = joint_wait_center(train, joined, prior)
    promises, risk, expected = forecasts(train, joined, center)
    horizons = np.column_stack([promises, promises + 24]) - joined.elapsed.clip(lower=0).to_numpy()[:, None]
    risk = adjust(risk, recent_delta(data, train, joined, pd.Timestamp(start), horizons), horizons, True)
    carrier = joined.lgs_company_nm.to_numpy()
    rules = fit_actions(carrier[:n], promises[:n], risk[:n], expected[:n], 'marginal', 12)
    bands = risk_band(risk, 'marginal')
    incumbent = apply_actions(carrier, risk, promises, rules, 'marginal')
    if promises[:n, 0].mean() < expected[:n].mean():
        raise ValueError('Reference mean-gap sign is not certified for the expanded actions.')
    predictions = {m: incumbent.copy() for m in METHODS}
    for company, rows in joined.groupby('lgs_company_nm', observed=True, sort=False).indices.items():
        ref = rows < n
        if ref.sum() < 500 or company not in rules:
            continue
        action = rules[company][bands[rows]]
        idx = np.column_stack([action, action - 1, action + 1])
        p = promises[rows[:, None], idx]
        late = risk[rows[:, None], idx].astype(float)
        severe = risk[rows[:, None], idx + 5].astype(float)
        change = balanced_delta(p, severe, ref)
        predictions['balanced'][rows] += change
        if severe[ref, 0].mean() >= .025:
            predictions['balanced_alert'][rows] += change
        prices = price_rules(p[ref], late[ref], severe[ref], ref.sum() / n)
        for method, limit in [('price', .03), ('price_reserve', .0275)]:
            selected = choose_action(p, late, severe, prices[limit])
            predictions[method][rows] = p[np.arange(len(rows)), selected]
    predictions['price_half'] = (incumbent + predictions['price']) / 2
    return predictions


def check_bounds(p, orders):
    if not np.isfinite(p).all() or (p < np.maximum(1, orders.elapsed) - 1e-8).any():
        raise ValueError('Invalid predicted hours.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['replay', 'candidate'])
    stage = parser.parse_args().stage
    OUT.mkdir(exist_ok=True)
    data = history()
    if stage == 'candidate':
        winner = json.loads((OUT / 'selection.json').read_text())['winner']
        if winner == PARENT:
            print('Incumbent retained:', PARENT_DIR / 'candidate.csv')
            return
        test = prepare(pd.read_csv(ROOT / 'data/test_202609.csv', usecols=['pkg_no'] + FEATURES))
        results = predict(data, '2026-09-21', test)
        prior = pd.read_csv(PARENT_DIR / 'candidate.csv')
        if len(test) != 911415 or not test.pkg_no.is_unique or test.pkg_no.isna().any() or not test.pkg_no.equals(prior.pkg_no):
            raise ValueError('Test ID mismatch.')
        np.testing.assert_allclose(results[PARENT], prior.pred_hours, rtol=0, atol=1e-8)
        p = results[winner]
        check_bounds(p, test)
        dest = OUT / 'candidate.csv'
        pd.DataFrame({'pkg_no': test.pkg_no, 'pred_hours': p}).to_csv(dest, index=False)
        meta = dict(method=winner, rows=len(p), mean_hours=float(p.mean()),
                    changed=int((np.abs(p-results[PARENT]) > 1e-8).sum()),
                    sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        (OUT / 'candidate.json').write_text(json.dumps(meta, indent=2))
        print(meta)
        return
    records, companies = [], []
    for start, end in [*WINDOWS, COLD]:
        valid = data.loc[(data.pay_time >= start) & (data.pay_time < end)]
        results = predict(data, start, valid)
        for method, p in results.items():
            check_bounds(p, valid)
            metrics = score(valid.lgs_company_nm, valid.y_total_hours, p)
            records.append(dict(cutoff=start, method=method, valid_rows=len(p), **metrics))
            companies.append(group_metrics(valid, pd.Series(p, index=valid.index), start, method))
            print(start, method, round(metrics['A'], 4), flush=True)
    scores = pd.DataFrame(records)
    prior = pd.read_csv(PARENT_DIR / 'scores.csv')
    def parent_metrics(frame):
        return frame[frame.method == PARENT].set_index('cutoff').sort_index()[METRICS]
    pd.testing.assert_frame_equal(parent_metrics(scores), parent_metrics(prior), atol=1e-8, rtol=0)
    a = scores.pivot(index='cutoff', columns='method', values='A')
    means = a.drop(COLD[0]).mean()
    eligible = means[a.loc[COLD[0]] <= a.at[COLD[0], PARENT]]
    winner = eligible.idxmin() if eligible.min() < means[PARENT] else PARENT
    scores.to_csv(OUT / 'scores.csv', index=False)
    pd.concat(companies).to_csv(OUT / 'companies.csv', index=False)
    (OUT / 'selection.json').write_text(json.dumps(dict(winner=winner, means=means.to_dict(),
                                                        cold=a.loc[COLD[0]].to_dict()), indent=2))
    print('Selected:', winner, flush=True)


if __name__ == '__main__':
    main()
