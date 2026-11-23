"""Evaluate shrunken city residual corrections to deadline_guard blend_full."""
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from deadline_guard import COLD, predict as parent_predict
from prospective_policy import FEATURES, ROOT, WINDOWS, fold_inputs, history, prepare
from run_province_release import group_metrics
from score import score

OUT = ROOT / 'output/city_residual'
BASE = 'blend_full'
KEYS = ['lgs_company_nm', 'shpp_chnl', 'shpp_prvn_nm']
CITY = KEYS + ['shpp_city_nm']
VARIANTS = {'half12_n200': (.5, 200, 12), 'full12_n200': (1., 200, 12),
            'half24_n200': (.5, 200, 24), 'half12_n500': (.5, 500, 12),
            'quarter12_n200': (.25, 200, 12)}


def attach_city(frame, city):
    """Attach the source-aligned city column to either history or test rows."""
    if len(frame) != len(city):
        raise ValueError('City field length does not match its source rows.')
    result = frame.copy()
    result['shpp_city_nm'] = pd.Series(city, index=result.index).astype('string').fillna('MISSING')
    return result


def city_shift(train, orders, method):
    """Return the frozen city-hour adjustment for one cutoff's mature train and orders."""
    shrink, minimum, cap = VARIANTS[method]
    tq = train.groupby(KEYS, observed=True).remaining.quantile(.90).rename('state_q')
    cq = train.groupby(CITY, observed=True).remaining.quantile(.90).rename('city_q')
    cn = train.groupby(CITY, observed=True).size().rename('n')
    stats = pd.concat([cq, cn], axis=1).join(tq, on=KEYS)
    raw = ((stats.city_q - stats.state_q) * stats.n / (stats.n + 2000)).clip(-24, 24)
    ix = pd.MultiIndex.from_arrays([orders[k].astype('string').fillna('MISSING') for k in CITY], names=CITY)
    shift = np.clip(raw.reindex(ix).fillna(0).to_numpy(), -cap, cap) * shrink
    shift[stats.n.reindex(ix).fillna(0).to_numpy() < minimum] = 0.
    return shift


def build_candidate(data, method):
    test = prepare(pd.read_csv(ROOT / 'data/test_202609.csv', usecols=['pkg_no'] + FEATURES))
    test_city = pd.read_csv(ROOT / 'data/test_202609.csv', usecols=['shpp_city_nm'], dtype={'shpp_city_nm': 'string'}).shpp_city_nm
    test = attach_city(test, test_city.to_numpy())
    train, _, _ = fold_inputs(data, '2026-09-21', test)
    shift = city_shift(train, test, method)
    old = pd.read_csv(ROOT / 'output/wide_promise_grid/candidate.csv')
    if len(test) != 911415 or test.pkg_no.isna().any() or not test.pkg_no.is_unique or not test.pkg_no.equals(old.pkg_no):
        raise ValueError('Test candidate row IDs do not match the incumbent.')
    pred = np.maximum(np.maximum(1., test.elapsed.to_numpy()), old.pred_hours.to_numpy() + shift)
    dest = OUT / 'candidate.csv'
    pd.DataFrame({'pkg_no': test.pkg_no, 'pred_hours': pred}).to_csv(dest, index=False)
    meta = {'method': method, 'rows': len(pred), 'mean_hours': float(pred.mean()),
            'changed_rows': int(np.count_nonzero(np.abs(shift) > 1e-12)),
            'mean_abs_change': float(np.abs(shift).mean()),
            'max_abs_change': float(np.abs(shift).max()),
            'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
    (OUT / 'candidate.json').write_text(json.dumps(meta, indent=2))
    return meta


def main():
    OUT.mkdir(exist_ok=True)
    data = history()
    # slim parquet carries the city column in its original source row order.
    slim_city = pd.read_parquet(ROOT / 'data/train_slim.parquet', columns=['shpp_city_nm']).shpp_city_nm
    data = attach_city(data, slim_city.to_numpy())
    if len(sys.argv) > 1 and sys.argv[1] == 'candidate':
        result = build_candidate(data, 'full12_n200')
        print('candidate', result, flush=True)
        return
    scores, groups, mae_rows = [], [], []
    for start, end in [*WINDOWS, COLD]:
        valid = data.loc[(data.pay_time >= start) & (data.pay_time < end)]
        found, _ = parent_predict(data, start, valid, include_blends=True)
        base = found[BASE]
        np.save(ROOT / 'tmp' / f'base9337_{start}.npy', base)
        train, _, _ = fold_inputs(data, start, valid)
        # Vectorized group quantiles, with state level pooled within carrier and channel.
        tq = train.groupby(KEYS, observed=True).remaining.quantile(.90).rename('state_q')
        cq = train.groupby(CITY, observed=True).remaining.quantile(.90).rename('city_q')
        cn = train.groupby(CITY, observed=True).size().rename('n')
        stats = pd.concat([cq, cn], axis=1).join(tq, on=KEYS)
        stats['raw_delta'] = ((stats.city_q - stats.state_q) * stats.n / (stats.n + 2000)).clip(-24, 24)
        ix = pd.MultiIndex.from_arrays([valid[k].astype('string').fillna('MISSING') for k in CITY], names=CITY)
        delta = stats.raw_delta.reindex(ix).fillna(0).to_numpy()
        support = stats.n.reindex(ix).fillna(0).to_numpy()
        preds = {BASE: base}
        for name, (shrink, minimum, cap) in VARIANTS.items():
            shift = np.clip(delta, -cap, cap) * shrink
            shift[support < minimum] = 0.
            preds[name] = np.maximum(np.maximum(1., valid.elapsed.to_numpy()), base + shift)
        for name, p in preds.items():
            result = score(valid.lgs_company_nm, valid.y_total_hours, p)
            errors = np.abs(p - valid.y_total_hours.to_numpy())
            mae_rows.append(dict(cutoff=start, method=name, MAE=float(errors.mean())))
            scores.append(dict(cutoff=start, method=name, rows=len(p), **result))
            groups.append(group_metrics(valid, pd.Series(p, index=valid.index), start, name))
            print(start, name, round(result['A'], 4), 'MAE', round(float(errors.mean()), 4), flush=True)

    s = pd.DataFrame(scores)
    wide = pd.read_csv(ROOT / 'output/wide_promise_grid/blend_scores.csv')
    expect = wide[wide.method == BASE].set_index('cutoff').sort_index()[['ME', 'pen_ontime', 'pen_severe', 'B', 'A']]
    actual = s[s.method == BASE].set_index('cutoff').sort_index()[['ME', 'pen_ontime', 'pen_severe', 'B', 'A']]
    pd.testing.assert_frame_equal(actual, expect, rtol=0, atol=1e-8)
    pivot = s.pivot(index='cutoff', columns='method', values='A')
    means = pivot.drop(COLD[0]).mean()
    winner = means.idxmin() if means.min() < means[BASE] else BASE
    s.to_csv(OUT / 'scores.csv', index=False)
    pd.concat(groups).to_csv(OUT / 'companies.csv', index=False)
    pd.DataFrame(mae_rows).to_csv(OUT / 'mae.csv', index=False)
    (OUT / 'selection.json').write_text(json.dumps({'winner': winner, 'means': means.to_dict(),
        'cold': pivot.loc[COLD[0]].to_dict(), 'variants': VARIANTS}, indent=2))
    print('Selected:', winner, 'four-window mean A', float(means[winner]), flush=True)
    if winner != BASE and means[winner] < means[BASE]:
        test = prepare(pd.read_csv(ROOT / 'data/test_202609.csv', usecols=['pkg_no'] + FEATURES))
        test_city = pd.read_csv(ROOT / 'data/test_202609.csv', usecols=['shpp_city_nm'], dtype={'shpp_city_nm': 'string'}).shpp_city_nm
        test = attach_city(test, test_city.to_numpy())
        found, _ = parent_predict(data, '2026-09-21', test, include_blends=True)
        train, _, _ = fold_inputs(data, '2026-09-21', test)
        tq = train.groupby(KEYS, observed=True).remaining.quantile(.90).rename('state_q')
        cq = train.groupby(CITY, observed=True).remaining.quantile(.90).rename('city_q')
        cn = train.groupby(CITY, observed=True).size().rename('n')
        stats = pd.concat([cq, cn], axis=1).join(tq, on=KEYS)
        stats['raw_delta'] = ((stats.city_q-stats.state_q)*stats.n/(stats.n+2000)).clip(-24,24)
        ix = pd.MultiIndex.from_arrays([test[k].astype('string').fillna('MISSING') for k in CITY], names=CITY)
        v = VARIANTS[winner]
        shift = np.clip(stats.raw_delta.reindex(ix).fillna(0).to_numpy(), -v[2], v[2]) * v[0]
        shift[stats.n.reindex(ix).fillna(0).to_numpy() < v[1]] = 0.
        pred = np.maximum(np.maximum(1., test.elapsed.to_numpy()), found[BASE] + shift)
        old = pd.read_csv(ROOT / 'output/wide_promise_grid/candidate.csv')
        if len(test) != 911415 or test.pkg_no.isna().any() or not test.pkg_no.is_unique or not test.pkg_no.equals(old.pkg_no):
            raise ValueError('Test candidate row IDs do not match the incumbent.')
        np.testing.assert_allclose(found[BASE], old.pred_hours.to_numpy(), rtol=0, atol=1e-8)
        dest = OUT / 'candidate.csv'
        pd.DataFrame({'pkg_no': test.pkg_no, 'pred_hours': pred}).to_csv(dest, index=False)
        meta = {'method': winner, 'rows': len(pred), 'mean_hours': float(pred.mean()),
                'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
        (OUT / 'candidate.json').write_text(json.dumps(meta, indent=2))
        print('candidate', meta, flush=True)


if __name__ == '__main__':
    main()
