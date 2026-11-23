"""Replay small changes toward old-history typical remaining times."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from carrier_allocation import check_bounds
from deadline_guard import predict
from prospective_policy import COLD, FEATURES, ROOT, WINDOWS, fold_inputs, history, prepare
from score import score
from wide_promise_grid import OUT as INCUMBENT


OUT = ROOT / 'output/typical_shrink'
BASE = 'blend_full'
SHRINK = 'median_narrow_tail_05'
CITY_METHOD = 'full12_n200'
COMBINED = 'city_combined'
KEYS = ['lgs_company_nm', 'shpp_chnl', 'province']
MIN_SUPPORT = 100
RATES = [.02, .05, .10]


def typical_remaining(train, orders):
    grouped = train.groupby(KEYS, observed=True, dropna=False).remaining
    summaries = grouped.agg(['size', 'mean', 'median'])
    summaries['width'] = grouped.quantile(.90) - summaries['median']
    summaries['q97'] = grouped.quantile(.97)
    summaries = summaries.loc[summaries['size'] >= MIN_SUPPORT]
    lookup = pd.MultiIndex.from_frame(orders[KEYS])
    stats = summaries.reindex(lookup)
    shares = train.lgs_company_nm.value_counts(normalize=True)
    stats['share'] = orders.lgs_company_nm.map(shares).astype(float).to_numpy()
    return stats


def variants(base, elapsed, stats):
    floor = np.maximum(1, elapsed)
    residual = base - elapsed.clip(min=0)
    yield BASE, base
    gates = {
        'all': np.ones(len(base), dtype=bool),
        'narrow_tail': stats.width.le(72).to_numpy(),
        'high_share': stats.share.ge(.10).to_numpy(),
        'narrow_high_share': (stats.width.le(72) & stats.share.ge(.10)).to_numpy(),
    }
    excess = (residual - stats.q97.to_numpy()).clip(min=0)
    delta = np.where(gates['narrow_tail'], .5 * excess, 0.)
    yield 'q97_narrow_tail_excess_half', np.maximum(floor, base - delta)
    for typical in ['mean', 'median']:
        shift = stats[typical].to_numpy() - residual
        supported = np.isfinite(shift)
        for gate, mask in gates.items():
            for rate in RATES:
                delta = np.where(mask & supported, rate * shift, 0.)
                yield f'{typical}_{gate}_{round(rate * 100):02d}', np.maximum(floor, base + delta)


def base_prediction(data, start, orders, base_cache):
    if base_cache is None:
        found, _ = predict(data, start, orders, include_blends=True)
        return found[BASE]
    return np.load(base_cache / f'base9337_{start}.npy')


def predictions(data, start, orders, base_cache=None):
    base = base_prediction(data, start, orders, base_cache)
    train, _, _ = fold_inputs(data, start, orders)
    return variants(base, orders.elapsed.to_numpy(), typical_remaining(train, orders))


def candidate(data):
    winner = json.loads((OUT / 'selection.json').read_text())['winner']
    if winner == BASE:
        print('No eligible candidate.')
        return
    test = prepare(pd.read_csv(ROOT / 'data/test_202609.csv', usecols=['pkg_no'] + FEATURES))
    frozen = pd.read_csv(INCUMBENT / 'candidate.csv', float_precision='round_trip')
    if len(test) != 911415 or test.pkg_no.hasnans or not test.pkg_no.is_unique:
        raise ValueError('Invalid test IDs.')
    pd.testing.assert_series_equal(test.pkg_no, frozen.pkg_no)
    found = predictions(data, '2026-09-21', test)
    _, base = next(found)
    np.testing.assert_allclose(base, frozen.pred_hours, atol=1e-8, rtol=0)
    p = next(p for name, p in found if name == winner)
    check_bounds(p, test)
    dest = OUT / 'candidate.csv'
    pd.DataFrame({'pkg_no': test.pkg_no, 'pred_hours': p}).to_csv(dest, index=False)
    verified = pd.read_csv(dest, float_precision='round_trip')
    pd.testing.assert_series_equal(verified.pkg_no, frozen.pkg_no)
    np.testing.assert_array_equal(verified.pred_hours, p)
    (OUT / 'candidate.json').write_text(json.dumps(dict(
        method=winner, rows=len(p), frozen_rows_verified=len(frozen),
        changed_rows=int(np.count_nonzero(p != base)), mean_hours=float(p.mean()),
        max_frozen_reproduction_difference=float(np.max(np.abs(base - frozen.pred_hours))),
        sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
    ), indent=2))


def replay(data, base_cache):
    scores = []
    for start, end in [*WINDOWS, COLD]:
        valid = data.loc[(data.pay_time >= start) & (data.pay_time < end)]
        for name, forecast in predictions(data, start, valid, base_cache):
            check_bounds(forecast, valid)
            actual = valid.y_total_hours.to_numpy()
            metrics = score(valid.lgs_company_nm, actual, forecast)
            scores.append(dict(cutoff=start, method=name, rows=len(actual),
                               MAE=np.abs(forecast - actual).mean(), **metrics))
        print('Completed', start, flush=True)
    result = pd.DataFrame(scores)
    frozen = pd.read_csv(INCUMBENT / 'blend_scores.csv')
    frozen = frozen.loc[frozen.method.eq(BASE)].reset_index(drop=True)
    base = result.loc[result.method.eq(BASE), frozen.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(base, frozen, atol=1e-8, rtol=0)
    result.to_csv(OUT / 'scores.csv', index=False)
    metrics = ['A', 'ME', 'MAE', 'pen_ontime', 'pen_severe', 'B']
    regular = result.loc[result.cutoff.ne(COLD[0])]
    cold = result.loc[result.cutoff.eq(COLD[0])].set_index('method')[metrics]
    means = regular.groupby('method')[metrics].mean()
    eligible = means.A.lt(means.at[BASE, 'A']) & means.MAE.lt(means.at[BASE, 'MAE'])
    winner = means.loc[eligible, 'A'].idxmin() if eligible.any() else BASE
    summary = means.add_prefix('mean_').join(cold.add_prefix('cold_')).sort_values('mean_A')
    summary['eligible'] = eligible
    summary.to_csv(OUT / 'summary.csv')
    (OUT / 'selection.json').write_text(json.dumps(dict(
        winner=winner, minimum_support=MIN_SUPPORT, narrow_tail_hours=72, high_share=.10,
        rule='Minimize regular mean A while improving regular mean A and MAE.',
    ), indent=2))
    print('Selected:', winner, flush=True)


def combine_replay(data, base_cache):
    from city_residual import OUT as CITY_OUT, city_shift

    scores = []
    for start, end in [*WINDOWS, COLD]:
        valid = data.loc[(data.pay_time >= start) & (data.pay_time < end)]
        train, _, _ = fold_inputs(data, start, valid)
        base = base_prediction(data, start, valid, base_cache)
        stats = typical_remaining(train, valid)
        shrunk = next(p for name, p in variants(base, valid.elapsed.to_numpy(), stats) if name == SHRINK)
        shift = city_shift(train, valid, CITY_METHOD)
        floor = np.maximum(1, valid.elapsed.to_numpy())
        city = np.maximum(floor, base + shift)
        combined = np.maximum(floor, shrunk + shift)
        for name, forecast in [(BASE, base), (SHRINK, shrunk), (CITY_METHOD, city), (COMBINED, combined)]:
            check_bounds(forecast, valid)
            actual = valid.y_total_hours.to_numpy()
            scores.append(dict(cutoff=start, method=name, rows=len(actual),
                               MAE=np.abs(forecast - actual).mean(), **score(valid.lgs_company_nm, actual, forecast)))
        print('Combined', start, scores[-1], flush=True)
    result = pd.DataFrame(scores)
    old_shrink = pd.read_csv(OUT / 'scores.csv')
    old_city = pd.read_csv(CITY_OUT / 'scores.csv').merge(
        pd.read_csv(CITY_OUT / 'mae.csv'), on=['cutoff', 'method'], validate='one_to_one')
    for name, old in [(BASE, old_shrink), (SHRINK, old_shrink), (CITY_METHOD, old_city)]:
        def select(table):
            return table.loc[table.method.eq(name), result.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(select(result), select(old), atol=1e-8, rtol=0)
    regular = result.loc[result.cutoff.ne(COLD[0])].groupby('method')[['A', 'MAE']].mean()
    independent = regular.loc[[SHRINK, CITY_METHOD]].min()
    wins = bool(regular.loc[COMBINED].lt(independent).all())
    result.to_csv(OUT / 'city_combined_scores.csv', index=False)
    (OUT / 'city_combined_selection.json').write_text(json.dumps(dict(
        improved_both=wins, regular=regular.to_dict(orient='index'),
    ), indent=2))
    print(regular, flush=True)


def combine_candidate(data):
    from city_residual import OUT as CITY_OUT, attach_city, city_shift

    selection = json.loads((OUT / 'city_combined_selection.json').read_text())
    if not selection['improved_both']:
        print('Combined policy did not improve both metrics.')
        return
    own = json.loads((OUT / 'candidate.json').read_text())
    if own['method'] != SHRINK or own['frozen_rows_verified'] != 911415:
        raise ValueError('Combined policy requires the independently verified shrink candidate.')
    test = prepare(pd.read_csv(ROOT / 'data/test_202609.csv', usecols=['pkg_no'] + FEATURES + ['shpp_city_nm']))
    test = attach_city(test, test.shpp_city_nm)
    frozen = pd.read_csv(INCUMBENT / 'candidate.csv', float_precision='round_trip')
    shrunk = pd.read_csv(OUT / 'candidate.csv', float_precision='round_trip')
    city = pd.read_csv(CITY_OUT / 'candidate.csv', float_precision='round_trip')
    for frame in [frozen, shrunk, city]:
        pd.testing.assert_series_equal(test.pkg_no, frame.pkg_no)
    train, _, _ = fold_inputs(data, '2026-09-21', test)
    shift = city_shift(train, test, CITY_METHOD)
    floor = np.maximum(1, test.elapsed.to_numpy())
    np.testing.assert_allclose(np.maximum(floor, frozen.pred_hours.to_numpy() + shift),
                               city.pred_hours, atol=1e-8, rtol=0)
    forecast = np.maximum(floor, shrunk.pred_hours.to_numpy() + shift)
    check_bounds(forecast, test)
    dest = OUT / 'city_combined.csv'
    pd.DataFrame({'pkg_no': test.pkg_no, 'pred_hours': forecast}).to_csv(dest, index=False)
    verified = pd.read_csv(dest, float_precision='round_trip')
    pd.testing.assert_series_equal(verified.pkg_no, frozen.pkg_no)
    np.testing.assert_array_equal(verified.pred_hours, forecast)
    (OUT / 'city_combined.json').write_text(json.dumps(dict(
        method=COMBINED, shrink=SHRINK, city=CITY_METHOD, rows=len(forecast),
        changed_rows=int(np.count_nonzero(forecast != frozen.pred_hours)), mean_hours=float(forecast.mean()),
        sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
    ), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['replay', 'candidate', 'combine-replay', 'combine-candidate'])
    parser.add_argument('--base-cache', type=Path)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.stage == 'combine-replay':
        from city_residual import attach_city

        data = history()
        city = pd.read_parquet(ROOT / 'data/train_slim.parquet', columns=['shpp_city_nm']).shpp_city_nm
        combine_replay(attach_city(data, city), args.base_cache)
    elif args.stage == 'combine-candidate':
        from city_residual import attach_city

        data = history()
        city = pd.read_parquet(ROOT / 'data/train_slim.parquet', columns=['shpp_city_nm']).shpp_city_nm
        combine_candidate(attach_city(data, city))
    elif args.stage == 'candidate':
        candidate(history())
    else:
        replay(history(), args.base_cache)


if __name__ == '__main__':
    main()
