"""Compare fixed waiting-time refinements against the accepted province model."""
import argparse

import numpy as np
import pandas as pd

from feature_groups_experiment import predict
from remaining_experiment import weighted_remaining_predict
from run_province_release import ROOT, group_metrics, load_history
from score import score


WINDOWS = [
    ('2026-08-25', '2026-09-01'),
    ('2026-09-01', '2026-09-08'),
    ('2026-09-08', '2026-09-15'),
    ('2026-09-15', '2026-09-21'),
]


def guarded_wait_adjustment(province, joint, standalone_delta):
    """Shorten only on agreement, and use the smaller proposed reduction."""
    joint_delta = joint - province
    reduction = np.maximum(joint_delta, standalone_delta.clip(upper=0))
    return province + reduction.where(joint_delta < 0, joint_delta)


def refinements(train, valid):
    baseline = weighted_remaining_predict(train, valid)
    province, _ = predict(train, valid, baseline, 'province')
    waiting, _ = predict(train, valid, baseline, 'wait_band')
    joint, joint_covered = predict(train, valid, province, 'province_wait')
    # The additive rule gives the waiting-time signal half its standalone
    # weight. The blend instead moderates both signals when they disagree.
    floor = valid.elapsed.clip(lower=1)
    additive = np.maximum(province + 0.5 * (waiting - baseline), floor)
    blend = (province + waiting) / 2
    guarded = guarded_wait_adjustment(province, joint, waiting - baseline)
    return {
        'v5': (province, None),
        'half_wait_adjustment': (additive, None),
        'province_wait_blend': (blend, None),
        'joint_backoff': (joint, joint_covered),
        'agreement_guard': (guarded, int((guarded != province).sum())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--by-issue', action='store_true', help='Group evaluation by warehouse entry date')
    args = parser.parse_args()
    history, source = load_history()
    if source != 'train_slim.parquet':
        print(f'Training input: {source}', flush=True)
    history['wait_band'] = pd.cut(
        history.elapsed, [-np.inf, 6, 12, 24, 48, np.inf], labels=False,
    ).astype('int8').astype(str)
    # Length prefix keeps the composite key unambiguous for any category text.
    history['province_wait'] = (
        history.province.str.len().astype(str) + ':' + history.province + history.wait_band
    )
    if not args.by_issue:
        release_scores = pd.read_csv(ROOT / 'output/v5/scores.csv')
        release_scores = release_scores[release_scores.method == 'v5_province'].set_index('cutoff')
    rows = []
    groups = []
    evaluation_time = history.issue_time if args.by_issue else history.pay_time
    suffix = '_by_issue' if args.by_issue else ''
    for start, end in WINDOWS:
        cutoff = pd.Timestamp(start)
        train = history.loc[
            (history.pay_time < cutoff - pd.Timedelta(days=21))
            & (history.sign_time < cutoff)
            & (history.issue_time < cutoff)
            & (history.elapsed >= 0)
            & (history.remaining >= 0)
        ]
        valid = history.loc[(evaluation_time >= cutoff) & (evaluation_time < pd.Timestamp(end))]
        for name, (prediction, coverage) in refinements(train, valid).items():
            if not prediction.index.equals(valid.index) or not np.isfinite(prediction).all():
                raise ValueError(f'{name} produced invalid predictions for {start}')
            if not (prediction >= valid.elapsed.clip(lower=1) - 1e-9).all():
                raise ValueError(f'{name} predicted before the decision time for {start}')
            metrics = score(valid.lgs_company_nm, valid.y_total_hours, prediction)
            if name == 'v5' and not args.by_issue:
                for key, value in metrics.items():
                    np.testing.assert_allclose(value, release_scores.loc[start, key], rtol=0, atol=1e-8)
            rows.append(dict(cutoff=start, method=name, coverage=coverage, rows=len(valid), **metrics))
            groups.append(group_metrics(valid, prediction, start, name))
            print(f'{start} {name:24s} A={metrics["A"]:.4f} severe={metrics["pen_severe"]:.4f}', flush=True)
    out = ROOT / 'output'
    pd.DataFrame(rows).to_csv(out / f'province_wait_scores{suffix}.csv', index=False)
    pd.concat(groups, ignore_index=True).to_csv(out / f'province_wait_company_metrics{suffix}.csv', index=False)


if __name__ == '__main__':
    main()
