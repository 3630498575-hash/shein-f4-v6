"""Reproduce and validate the frozen F4 submission."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from carrier_allocation import check_bounds
from city_residual import city_shift
from deadline_guard import predict
from prospective_policy import FEATURES, ROOT, fold_inputs, prepare
from typical_shrink import BASE, CITY_METHOD, SHRINK, typical_remaining, variants


CUTOFF = '2026-09-21'
TRAIN_COLUMNS = FEATURES + ['shpp_city_nm', 'sign_time', 'y_total_hours']


def load_history(raw_csv=False):
    slim = ROOT / 'data/train_slim.parquet'
    if slim.is_file() and not raw_csv:
        frame = pd.read_parquet(slim, columns=TRAIN_COLUMNS)
    else:
        dtype = {key: 'string' for key in FEATURES + ['shpp_city_nm', 'sign_time']}
        frame = pd.read_csv(ROOT / 'data/train_202608_0920.csv', usecols=TRAIN_COLUMNS, dtype=dtype)
    frame = prepare(frame)
    frame['shpp_city_nm'] = frame.shpp_city_nm.astype('string').fillna('MISSING')
    frame['sign_time'] = pd.to_datetime(frame.sign_time, errors='raise')
    frame['remaining'] = frame.y_total_hours - frame.elapsed
    return frame


def submission_hours(train, orders, base):
    elapsed = orders.elapsed.to_numpy()
    floor = np.maximum(1., elapsed)
    stats = typical_remaining(train, orders)
    shrunk = next(p for method, p in variants(base, elapsed, stats) if method == SHRINK)
    city = city_shift(train, orders, CITY_METHOD)
    previous = np.maximum(floor, shrunk + city)
    remaining = base - elapsed.clip(min=0)
    difference = np.nan_to_num(stats['median'].to_numpy() - remaining)
    old = .05 * stats.width.le(72).to_numpy()
    new = .08 * np.clip((120 - stats.width.fillna(999).to_numpy()) / 90, 0, 1)
    return np.maximum(floor, previous + (new * difference - old * difference))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'output/reproduced_submission.csv')
    parser.add_argument('--compare', type=Path, help='Optional frozen CSV for row-wise verification')
    parser.add_argument('--raw-csv', action='store_true', help='Read the original training CSV directly')
    args = parser.parse_args()
    history = load_history(args.raw_csv)
    orders = prepare(pd.read_csv(ROOT / 'data/test_202609.csv',
                                 usecols=['pkg_no', 'shpp_city_nm'] + FEATURES))
    if len(orders) != 911415 or orders.pkg_no.hasnans or not orders.pkg_no.is_unique:
        raise ValueError('Test IDs or row count are invalid.')
    forecasts, _ = predict(history, CUTOFF, orders, include_blends=True)
    train, _, _ = fold_inputs(history, CUTOFF, orders)
    hours = submission_hours(train, orders, forecasts[BASE])
    check_bounds(hours, orders)
    if args.compare is not None:
        frozen = pd.read_csv(args.compare)
        if list(frozen) != ['pkg_no', 'pred_hours'] or not orders.pkg_no.equals(frozen.pkg_no):
            raise ValueError('Frozen IDs differ from the test input.')
        np.testing.assert_allclose(hours, frozen.pred_hours, rtol=0, atol=1e-8)
        print('Maximum reproduction difference:', np.max(np.abs(hours - frozen.pred_hours)))
    args.out.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame({'pkg_no': orders.pkg_no, 'pred_hours': hours}).to_csv(args.out, index=False)
    print(json.dumps({'rows': len(hours), 'mean_hours': float(hours.mean()),
                      'sha256': hashlib.sha256(args.out.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
