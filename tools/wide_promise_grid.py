"""Search a wider promise grid under normal and slower delivery scenarios."""
import argparse
import hashlib
import itertools
import json

import numpy as np
import pandas as pd

from carrier_allocation import OUT as PRIOR_DIR, allocation_predictions, check_bounds
from deadline_guard import calendar_predictions, predict as deadline_predict
from distribution_promise_experiment import empirical_cdf
from prospective_policy import COLD, FEATURES, ROOT, WINDOWS, fold_inputs, history, prepare
from recent_tail_experiment import METRICS, adjust, recent_delta, wait_risks
from run_province_release import group_metrics
from score import score


OUT = ROOT / 'output/wide_promise_grid'
BASE = 'price_half'
CALENDAR = 'weekday_half'
OFFSETS = np.array([0., -36., -24., -18., -12., -6., 6., 12., 24.])
METHODS = ['wide', 'wide_half', 'wide_stress', 'wide_stress_half', 'wait_stress', 'wait_stress_half']
TRIMS = ['trim_low', 'trim_reserve', 'trim_half', 'trim_stress']
BLENDS = ['blend_full', 'blend_base', 'blend_limited']


def trim_low_risk(promises, risks, late_limit, severe_limit):
    k = len(OFFSETS)
    eligible = ((OFFSETS >= -18) & (OFFSETS < 0)
                & (risks[:, :k] <= late_limit) & (risks[:, k:] <= severe_limit))
    eligible[:, 0] = True
    return np.where(eligible, promises, np.inf).min(axis=1)


def action_for(p, late, severe, prices):
    return (p + prices[0] * late + prices[1] * severe).argmin(axis=1)


def fit(p, normal, slower, share, slow_weight):
    """Fit prices on the reference set by scoring each scenario separately."""
    actions = p.shape[1]
    row = np.arange(len(p))
    blend = (1 - slow_weight) * normal + slow_weight * slower
    late, severe = blend[:, :actions], blend[:, actions:]
    scale = np.array([0., .125, .25, .5, .75, 1.]) * 500 / share
    best = (np.inf, None)
    for prices in itertools.chain([None], itertools.product(scale, repeat=2)):
        choice = np.zeros(len(p), dtype=int) if prices is None else action_for(p, late, severe, prices)
        loss = share * p[row, choice].mean()
        for factor, risks in [(1 - slow_weight, normal), (slow_weight, slower)]:
            if factor:
                loss += 500 * factor * (max(0., risks[row, choice].mean(dtype=float) - .10)
                                        + max(0., risks[row, choice + actions].mean(dtype=float) - .03))
        if loss < best[0]:
            best = (loss, prices)
    return best[1]


def predict(data, start, valid):
    train, joined, n = fold_inputs(data, start, valid)
    base = allocation_predictions(data, start, train, joined, n)[BASE]
    cutoff = pd.Timestamp(start)
    weekday = calendar_predictions(train, joined, cutoff, base)[CALENDAR]
    elapsed = joined.elapsed.to_numpy().clip(min=0)
    promises = np.maximum(base[:, None] + OFFSETS, np.maximum(1, elapsed)[:, None])
    deadlines = np.column_stack([promises, promises + 24]) - elapsed[:, None]
    cdf, expected = empirical_cdf(train, joined, deadlines, include_route=True)
    # Both scenario means are below every feasible reference promise mean.
    if promises[:n].min(axis=1).mean() < (expected[:n] + elapsed[:n]).mean() + 12:
        raise ValueError('Reference mean-gap sign uncertified.')
    normal = 1 - cdf
    adjusted = adjust(normal, recent_delta(data, train, joined, cutoff, deadlines), deadlines, True)
    # The slower scenario adds 12 hours to remaining time. Evaluate it by
    # moving each queried deadline 12 hours earlier in the same distribution.
    slower_deadlines = deadlines - 12
    slow_cdf, _ = empirical_cdf(train, joined, slower_deadlines, include_route=True)
    slower = 1 - slow_cdf
    waiting = wait_risks(train, joined, deadlines, adjusted)
    waiting_slow = wait_risks(train, joined, slower_deadlines, slower)
    results = {m: base.copy() for m in METHODS}
    alternatives = [('wide', adjusted, adjusted, 0.),
                    ('wide_stress', adjusted, np.maximum(adjusted, slower), .25),
                    ('wait_stress', waiting, np.maximum(waiting, waiting_slow), .25)]
    for rows in joined.groupby('lgs_company_nm', observed=True).indices.values():
        ref = rows[rows < n]
        if len(ref) < 500:
            continue
        val = rows[rows >= n]
        for method, usual, late_wave, weight in alternatives:
            prices = fit(promises[ref], usual[ref], late_wave[ref], len(ref) / n, weight)
            if prices is not None:
                blend = usual[val] * (1 - weight) + late_wave[val] * weight
                k = len(OFFSETS)
                action = action_for(promises[val], blend[:, :k], blend[:, k:], prices)
                results[method][val] = promises[val, action]
    for method, *_ in alternatives:
        results[method + '_half'] = (base + results[method]) / 2
    cautious = np.maximum(adjusted, waiting)
    results['trim_low'] = trim_low_risk(promises, cautious, .09, .02)
    results['trim_reserve'] = trim_low_risk(promises, cautious, .08, .015)
    results['trim_half'] = (base + results['trim_low']) / 2
    results['trim_stress'] = trim_low_risk(promises, np.maximum(cautious, waiting_slow), .09, .02)
    results.update({BASE: base, CALENDAR: weekday})
    return {name: p[n:] for name, p in results.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['replay', 'blend', 'candidate'])
    stage = parser.parse_args().stage
    OUT.mkdir(exist_ok=True)
    data = history()
    if stage == 'blend':
        scores, groups = [], []
        for start, end in [*WINDOWS, COLD]:
            valid = data.loc[(data.pay_time >= start) & (data.pay_time < end)]
            found, _ = deadline_predict(data, start, valid, include_blends=True)
            for name in BLENDS:
                p = found[name]
                check_bounds(p, valid)
                result = score(valid.lgs_company_nm, valid.y_total_hours, p)
                scores.append(dict(cutoff=start, method=name, rows=len(p), **result))
                groups.append(group_metrics(valid, pd.Series(p, index=valid.index), start, name))
                print(start, name, round(result['A'], 4), flush=True)
        pd.DataFrame(scores).to_csv(OUT / 'blend_scores.csv', index=False)
        pd.concat(groups).to_csv(OUT / 'blend_companies.csv', index=False)
        return
    if stage == 'candidate':
        selection = json.loads((OUT / 'selection.json').read_text())
        if set(selection['means']) != {BASE, CALENDAR, *METHODS, *TRIMS}:
            raise ValueError('Wide-grid replay results are incomplete.')
        # Include the earlier low-score variant under the same ranking.
        prior = pd.read_csv(ROOT / 'output/deadline_guard/scores.csv')
        prior = prior.loc[prior.method == 'local_half'].set_index('cutoff').A
        selection['means']['local_half'] = float(prior.drop(COLD[0]).mean())
        selection['cold']['local_half'] = float(prior.loc[COLD[0]])
        blends = pd.read_csv(OUT / 'blend_scores.csv').pivot(index='cutoff', columns='method', values='A')
        if (set(blends.index) != {start for start, _ in [*WINDOWS, COLD]}
                or set(blends.columns) != set(BLENDS) or not np.isfinite(blends.to_numpy()).all()):
            raise ValueError('Blend replay results are incomplete.')
        selection['means'].update(blends.drop(COLD[0]).mean().to_dict())
        selection['cold'].update(blends.loc[COLD[0]].to_dict())
        winner = min(selection['means'], key=selection['means'].get)
        selection['winner'] = winner
        test = prepare(pd.read_csv(ROOT / 'data/test_202609.csv', usecols=['pkg_no'] + FEATURES))
        if winner == 'local_half' or winner in BLENDS:
            result, _ = deadline_predict(data, '2026-09-21', test, include_blends=True)
        else:
            result = predict(data, '2026-09-21', test)
        incumbent = pd.read_csv(PRIOR_DIR / 'candidate.csv')
        if (len(test) != 911415 or test.pkg_no.hasnans or not test.pkg_no.is_unique
                or not test.pkg_no.equals(incumbent.pkg_no)):
            raise ValueError('Test ID mismatch')
        np.testing.assert_allclose(result[BASE], incumbent.pred_hours, atol=1e-8, rtol=0)
        p = result[winner]
        check_bounds(p, test)
        dest = OUT / 'candidate.csv'
        pd.DataFrame({'pkg_no': test.pkg_no, 'pred_hours': p}).to_csv(dest, index=False)
        meta = dict(method=winner, rows=len(p), mean_hours=p.mean(),
                    sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        (OUT / 'candidate.json').write_text(json.dumps(meta, indent=2))
        (OUT / 'candidate_selection.json').write_text(json.dumps(selection, indent=2))
        print(meta)
        return
    scores, group_scores = [], []
    for start, end in [*WINDOWS, COLD]:
        valid = data.loc[(data.pay_time >= start) & (data.pay_time < end)]
        for name, p in predict(data, start, valid).items():
            check_bounds(p, valid)
            metrics = score(valid.lgs_company_nm, valid.y_total_hours, p)
            scores.append(dict(cutoff=start, method=name, rows=len(p), **metrics))
            group_scores.append(group_metrics(valid, pd.Series(p, index=valid.index), start, name))
            print(start, name, round(metrics['A'], 4), flush=True)
    s = pd.DataFrame(scores)
    prior = pd.read_csv(ROOT / 'output/deadline_guard/scores.csv')
    for name in [BASE, CALENDAR]:
        def metrics(frame):
            return frame.loc[frame.method == name].set_index('cutoff').sort_index()[METRICS]
        pd.testing.assert_frame_equal(metrics(s), metrics(prior), atol=1e-8, rtol=0)
    panel = s.pivot(index='cutoff', columns='method', values='A')
    means = panel.drop(COLD[0]).mean()
    # The primary ranking follows the current objective of lowering A.
    # Report the cold-start tradeoff separately for comparison.
    winner = means.idxmin()
    s.to_csv(OUT / 'scores.csv', index=False)
    pd.concat(group_scores).to_csv(OUT / 'companies.csv', index=False)
    (OUT / 'selection.json').write_text(json.dumps(dict(winner=winner, means=means.to_dict(),
                                                        cold=panel.loc[COLD[0]].to_dict()), indent=2))
    print('Selected:', winner)


if __name__ == '__main__':
    main()
