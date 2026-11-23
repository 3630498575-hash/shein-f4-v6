"""Compare delivery probability models under one F4 decision policy."""
import argparse
import itertools

import numpy as np
import pandas as pd

from feature_groups_experiment import predict as province_predict
from model_experiment import CAT, NUM, features
from province_wait_experiment import WINDOWS
from remaining_experiment import weighted_remaining_predict
from run_province_release import ROOT, group_metrics
from score import score

OFFSETS = np.array([-24., -12., 0., 12., 24.])
GROUPS = ['lgs_company_nm', 'shpp_chnl', 'province']
SEED = 42


def load_history():
    columns = NUM + CAT + ['pay_time', 'issue_time', 'sign_time', 'y_total_hours']
    history = pd.read_parquet(ROOT / 'data/train_slim.parquet', columns=columns)
    for col in ('pay_time', 'issue_time', 'sign_time'):
        history[col] = pd.to_datetime(history[col], errors='raise')
    history['elapsed'] = (history.issue_time - history.pay_time).dt.total_seconds() / 3600
    history['remaining'] = history.y_total_hours - history.elapsed
    history['province'] = history.shpp_prvn_nm.fillna('MISSING').astype(str)
    return history


def empirical_cdf(train, valid, deadlines, include_route=False):
    """Estimate a CDF and expected remaining time with hierarchical pooling."""
    remaining = train.remaining.to_numpy()
    sorted_all = np.sort(remaining)
    cdf = (np.searchsorted(sorted_all, deadlines, side='right') / len(remaining)).astype('float32')
    means = np.full(len(valid), remaining.mean())
    levels = [(GROUPS[:1], 100, 0), (GROUPS[:2], 100, 0), (GROUPS, 500, 2000)]
    if include_route:
        levels.append((GROUPS + ['route_cat'], 500, 2000))
    for keys, minimum, prior in levels:
        train_groups = train.groupby(keys, sort=False).indices
        for key, idx in valid.groupby(keys, sort=False).indices.items():
            train_idx = train_groups.get(key)
            if train_idx is None or len(train_idx) < minimum:
                continue
            values = np.sort(remaining[train_idx])
            weight = len(values) / (len(values) + prior)
            group_cdf = np.searchsorted(values, deadlines[idx], side='right') / len(values)
            cdf[idx] = weight * group_cdf + (1 - weight) * cdf[idx]
            means[idx] = weight * values.mean() + (1 - weight) * means[idx]
    return cdf, means


def tree_cdf(train, valid, deadlines, max_rows, iterations):
    """Learn P(remaining hours <= candidate deadline) with an augmented classifier."""
    from catboost import CatBoostClassifier

    sample = train.sample(n=min(max_rows, len(train)), random_state=SEED).reset_index(drop=True)
    inputs = deadline_features(sample)
    # Two horizons per parcel, selected without reference to its outcome.
    # Use early and late grids to give the classifier coverage of the tail.
    rng = np.random.default_rng(SEED)
    early = rng.choice(np.arange(24., 265., 12.), size=len(sample))
    late = rng.choice(np.arange(144., 601., 12.), size=len(sample))
    design = pd.concat([inputs.assign(deadline_hours=early), inputs.assign(deadline_hours=late)], ignore_index=True)
    target = np.concatenate([sample.remaining.to_numpy() <= early, sample.remaining.to_numpy() <= late]).astype(np.uint8)
    model = CatBoostClassifier(
        iterations=iterations, depth=6, learning_rate=0.12, thread_count=6,
        random_seed=SEED, allow_writing_files=False, verbose=False,
    )
    model.fit(design, target, cat_features=CAT)
    del design, inputs, sample
    cdf = np.empty_like(deadlines, dtype='float32')
    for start in range(0, len(valid), 50000):
        stop = min(start + 50000, len(valid))
        batch = deadline_features(valid.iloc[start:stop])
        for j in range(deadlines.shape[1]):
            batch['deadline_hours'] = deadlines[start:stop, j]
            cdf[start:stop, j] = model.predict_proba(batch, thread_count=6)[:, 1]
    # Enforce a nondecreasing CDF across the evaluated horizons.
    return np.maximum.accumulate(cdf, axis=1)


def deadline_features(rows):
    # Parquet stores some categories as dictionary arrays. Convert before
    # filling missing categories so the sentinel need not be in the dictionary.
    return features(rows.assign(**{name: rows[name].astype('string') for name in CAT}))


def policy(companies, promises, cdf, expected_total, late_limit=0.10, severe_limit=0.03, max_offset=None):
    """Choose one sampled offset per carrier using forecast probabilities only."""
    unique, carrier = np.unique(companies, return_inverse=True)
    counts = np.bincount(carrier).astype(float)
    forecasts = np.column_stack([np.bincount(carrier, weights=p) for p in promises.T])
    ordinary = 1 - cdf[:, :len(OFFSETS)]
    severe = 1 - cdf[:, len(OFFSETS):]
    penalty = np.column_stack([
        500 * (np.maximum(0, np.bincount(carrier, weights=late) / counts - late_limit)
               + np.maximum(0, np.bincount(carrier, weights=very_late) / counts - severe_limit))
        for late, very_late in zip(ordinary.T, severe.T)
    ])
    # For the eight-carrier dataset, enumerate all 5^8 combinations in
    # manageable batches. The absolute mean gap couples carrier decisions.
    carrier_idx = np.arange(len(unique))
    action_indices = range(len(OFFSETS)) if max_offset is None else np.flatnonzero(np.abs(OFFSETS) <= max_offset)
    action_sets = itertools.product(action_indices, repeat=len(unique))
    best, winner = np.inf, None
    while batch := list(itertools.islice(action_sets, 25000)):
        actions = np.asarray(batch, dtype=np.int8)
        gaps = np.abs(forecasts[carrier_idx, actions].sum(axis=1) / len(carrier) - expected_total)
        objective = gaps + penalty[carrier_idx, actions].sum(axis=1)
        local = int(objective.argmin())
        if objective[local] < best:
            best, winner = float(objective[local]), actions[local].copy()
    return promises[np.arange(len(carrier)), winner[carrier]], dict(zip(unique, OFFSETS[winner])), best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['empirical', 'empirical_route', 'delay_stress', 'tree'], required=True)
    parser.add_argument('--max-train', type=int, default=120000)
    parser.add_argument('--iterations', type=int, default=140)
    args = parser.parse_args()
    history = load_history()
    result_rows, group_rows, decisions = [], [], []
    release = pd.read_csv(ROOT / 'output/v5/scores.csv')
    release = release[release.method == 'v5_province'].set_index('cutoff')
    out = ROOT / 'output'
    for start, end in WINDOWS:
        cutoff = pd.Timestamp(start)
        train = history.loc[
            (history.pay_time < cutoff - pd.Timedelta(days=21))
            & (history.issue_time < cutoff) & (history.sign_time < cutoff)
            & (history.elapsed >= 0) & (history.remaining >= 0)
        ]
        valid = history.loc[(history.pay_time >= cutoff) & (history.pay_time < pd.Timestamp(end))]
        v4 = weighted_remaining_predict(train, valid)
        baseline, _ = province_predict(train, valid, v4, 'province')
        elapsed = valid.elapsed.to_numpy().clip(min=0)
        promises = np.maximum(baseline.to_numpy()[:, None] + OFFSETS, np.maximum(elapsed, 1)[:, None])
        deadlines = promises - elapsed[:, None]
        # Use the clipped promise + 24 hours, including at the elapsed-time floor.
        severe_deadlines = promises + 24 - elapsed[:, None]
        query = np.column_stack((deadlines, severe_deadlines))
        query_order = np.argsort(query, axis=1, kind='stable')
        sorted_query = np.take_along_axis(query, query_order, axis=1)
        empirical, means = empirical_cdf(train, valid, sorted_query, include_route=args.model == 'empirical_route')
        reference_mean = float((means + elapsed).mean())
        models = {args.model: empirical}
        if args.model == 'delay_stress':
            # A stress scenario, not a fitted probability of disruption:
            # one fifth of the forecast distribution arrives a day later.
            delayed, _ = empirical_cdf(train, valid, sorted_query - 24)
            models = {'delay_stress': 0.8 * empirical + 0.2 * delayed}
            reference_mean += 0.2 * 24
        if args.model == 'tree':
            print(f'{start}: training deadline classifier', flush=True)
            tree = tree_cdf(train, valid, sorted_query, args.max_train, args.iterations)
            models = {'tree': tree, 'blend': (tree + empirical) / 2}
        base_score = score(valid.lgs_company_nm, valid.y_total_hours, baseline)
        for name, value in base_score.items():
            np.testing.assert_allclose(value, release.loc[start, name], rtol=0, atol=1e-8)
        predictions = [('v5', baseline, np.nan)]
        for name, sorted_cdf in models.items():
            cdf = np.empty_like(sorted_cdf)
            np.put_along_axis(cdf, query_order, sorted_cdf, axis=1)
            # Keep exact ordinary/severe probabilities even when a deadline
            # has been moved up to the order's elapsed-time floor.
            for suffix, late, severe in [('score', .10, .03), ('reserve', .09, .0275)]:
                method = f'{name}_{suffix}'
                pred, chosen, estimated = policy(
                    valid.lgs_company_nm.to_numpy(), promises, cdf, reference_mean, late, severe,
                )
                predictions.append((method, pd.Series(pred, index=valid.index), estimated))
                decisions.extend(dict(cutoff=start, method=method, company=g, offset=delta) for g, delta in chosen.items())
        for method, pred, estimated in predictions:
            if not np.isfinite(pred).all() or not (pred.to_numpy() >= elapsed - 1e-9).all():
                raise ValueError(f'Invalid promises: {method} at {start}')
            values = score(valid.lgs_company_nm, valid.y_total_hours, pred)
            result_rows.append(dict(cutoff=start, method=method, train_rows=len(train), valid_rows=len(valid), estimated_policy_objective=estimated, **values))
            group_rows.append(group_metrics(valid, pred, start, method))
            print(f'{start} {method:20s} A={values["A"]:.4f} ME={values["ME"]:.4f} severe={values["pen_severe"]:.4f}', flush=True)
        pd.DataFrame(result_rows).to_csv(out / f'distribution_{args.model}_scores.csv', index=False)
        pd.concat(group_rows, ignore_index=True).to_csv(out / f'distribution_{args.model}_companies.csv', index=False)
        pd.DataFrame(decisions).to_csv(out / f'distribution_{args.model}_offsets.csv', index=False)


if __name__ == '__main__':
    main()
