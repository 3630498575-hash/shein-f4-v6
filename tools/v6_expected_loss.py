"""Fit time promises against expected carrier penalties and replay the locked v6."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from carrier_allocation import check_bounds
from prospective_policy import COLD, FEATURES, ROOT, WINDOWS, fold_inputs, prepare
from run_submission import CUTOFF, load_history, submission_hours
from score import score
from typical_shrink import BASE, base_prediction
from deadline_guard import predict


OUT = ROOT / 'output/v6'
KEYS = ['lgs_company_nm', 'shpp_chnl', 'province']
STEP = 6
BINS = 120
BLENDS = [.10, .25, .50]
EXTRA = ('2026-09-12', '2026-09-21')
UNUSED = ('2026-08-22 12:00', '2026-08-23')
POLICIES = ['symmetric', 'add_only', 'quarter_down', 'share_down']


def durations(past, cutoff):
    signed = past.sign_time.lt(cutoff)
    stop = past.sign_time.where(signed, cutoff)
    duration = (stop - past.issue_time).dt.total_seconds().to_numpy() / 3600
    signed = signed.to_numpy() & (duration >= 0)
    units = np.maximum(duration, 0) / STEP
    # A still-open parcel contributes to fully elapsed intervals only.
    seen = np.where(signed, np.maximum(1, np.ceil(units)), np.floor(units)).astype(int)
    return seen.clip(0, BINS), signed & (units <= BINS)


def hazard(groups, seen, events, parent):
    n = len(parent)
    ix = groups * (BINS + 1) + seen
    counts = np.bincount(ix, minlength=n*(BINS+1)).reshape(n, -1)[:, 1:]
    at_risk = np.cumsum(counts[:, ::-1], axis=1)[:, ::-1]
    ends = np.bincount(ix[events], minlength=n*(BINS+1)).reshape(n, -1)[:, 1:]
    return (ends + 120 * parent) / (at_risk + 120)


def index_for(frame, keys):
    if len(keys) == 1:
        return pd.Index(frame[keys[0]])
    return pd.MultiIndex.from_frame(frame[keys])


def choose_promises(survival, counts, share):
    # One action per supported route. Unchanged terms from other carriers cancel.
    grid = np.arange(1, BINS - 3)
    duration = grid * STEP
    late = survival[:, grid-1]
    severe = survival[:, grid+3]
    valid = (late <= .50) & (late >= .002)
    choice = np.argmin(abs(severe - .03), axis=1)
    valid[np.arange(len(choice)), choice] = True
    weights = counts / counts.sum()
    current = np.arange(len(choice))
    r, s = weights @ late[current, choice], weights @ severe[current, choice]
    for _ in range(5):
        changed = False
        for i in np.argsort(-counts, kind='stable'):
            old, w = choice[i], weights[i]
            rates = r + w * (late[i] - late[i, old])
            tails = s + w * (severe[i] - severe[i, old])
            loss = share*w*duration + 500*(np.maximum(0, rates-.10) + np.maximum(0, tails-.03))
            best = np.argmin(np.where(valid[i], loss, np.inf))
            if loss[best] < loss[old] - 1e-10:
                choice[i], r, s = best, rates[best], tails[best]
                changed = True
        if not changed:
            break
    return duration[choice]


def target_hours(data, start, orders):
    cut = pd.Timestamp(start)
    if orders.issue_time.min() < cut:
        raise ValueError('Some decisions precede the fit cutoff')
    past = data.loc[data.issue_time.lt(cut) & data.issue_time.ge(cut-pd.Timedelta(days=35))
                    & data.elapsed.ge(0), KEYS + ['issue_time','sign_time']].copy()
    seen, events = durations(past, cut)
    # Pool hazard estimates first across all carriers, then carrier, channel, state.
    total_counts = np.bincount(seen, minlength=BINS+1)[1:]
    total_risk = np.cumsum(total_counts[::-1])[::-1]
    global_hazard = np.bincount(seen[events], minlength=BINS+1)[1:] / np.maximum(total_risk, 1)
    previous = None
    for depth in (1, 2, 3):
        keys = KEYS[:depth]
        groups = past.groupby(keys, observed=True, dropna=False, sort=True)
        stats = groups.size()
        group_id = groups.ngroup().to_numpy(dtype=int)
        if previous is None:
            prior = np.broadcast_to(global_hazard, (len(stats), BINS))
        else:
            parent_id = parent_index.get_indexer(stats.index.droplevel(-1))
            assert np.all(parent_id >= 0)
            prior = previous[parent_id]
        previous = hazard(group_id, seen, events, prior)
        parent_index = stats.index
    survival = np.cumprod(1-previous, axis=1)
    ref = past.issue_time.ge(cut-pd.Timedelta(days=7)).to_numpy()
    counts = np.bincount(group_id[ref], minlength=len(stats))
    targets = np.full(len(stats), np.nan)
    carriers = stats.index.get_level_values(0)
    for carrier in carriers.unique():
        rows = np.flatnonzero((carriers == carrier) & (counts > 0))
        if len(rows):
            targets[rows] = choose_promises(survival[rows], counts[rows], counts[rows].sum()/counts.sum())
    ids = stats.index.get_indexer(index_for(orders, KEYS))
    remaining = np.where(ids >= 0, targets[np.maximum(ids,0)], np.nan)
    shares = past.loc[ref,KEYS[0]].value_counts(normalize=True)
    share = orders[KEYS[0]].map(shares).astype(float).fillna(0).to_numpy()
    return remaining + orders.elapsed.to_numpy(), share


def adjusted(base, targets, orders, blend, policy):
    target, share = targets
    delta = np.nan_to_num(target - base, nan=0.).clip(-48, 48)
    factor = {'symmetric':1., 'add_only':0., 'quarter_down':.25,
              'share_down':np.minimum(1.,share/.15)}[policy]
    delta = np.maximum(0, delta) + np.minimum(0,delta)*factor
    return np.maximum(np.maximum(1,orders.elapsed.to_numpy()), base + blend*delta)


def replay(data, windows, variants):
    rows = []
    for start, end in windows:
        orders = data.loc[data.pay_time.ge(start) & data.pay_time.lt(end)]
        train, _, _ = fold_inputs(data, start, orders)
        cache = ROOT/'tmp'
        cache_file = f'base9337_{start}.npy'.replace(':','-')
        cached = ':' not in start and (cache/cache_file).is_file()
        base = base_prediction(data,start,orders,cache if cached else None)
        frozen = submission_hours(train, orders, base)
        target = target_hours(data, start, orders)
        for blend, policy in [(0,'base'), *variants]:
            p = adjusted(frozen, target, orders, blend, policy) if blend else frozen
            check_bounds(p, orders)
            assert p.mean() > orders.y_total_hours.mean()
            metrics = score(orders.lgs_company_nm, orders.y_total_hours, p)
            rows.append(dict(start=start, end=end, rows=len(p), blend=blend, policy=policy,
                             MAE=float(np.abs(p-orders.y_total_hours).mean()), **metrics))
        print(start,[round(r['A'],4) for r in rows if r['start']==start],flush=True)
    return pd.DataFrame(rows)


def candidate(data, blend, policy, rebuild):
    orders = prepare(pd.read_csv(ROOT/'data/test_202609.csv', usecols=['pkg_no','shpp_city_nm']+FEATURES))
    released_path = ROOT/'output/submission_final/submission.csv'
    if len(orders)!=911415 or orders.pkg_no.hasnans or not orders.pkg_no.is_unique:
        raise ValueError('Test IDs failed verification')
    base = None
    if released_path.is_file():
        expected = json.loads((released_path.parent/'acceptance.json').read_text())['sha256']
        if hashlib.sha256(released_path.read_bytes()).hexdigest() != expected:
            raise ValueError('Base CSV checksum mismatch')
        released = pd.read_csv(released_path, float_precision='round_trip')
        pd.testing.assert_series_equal(orders.pkg_no,released.pkg_no)
        base = released.pred_hours.to_numpy()
    if rebuild or base is None:
        found, _ = predict(data, CUTOFF, orders, include_blends=True)
        train, _, _ = fold_inputs(data, CUTOFF, orders)
        fresh = submission_hours(train, orders, found[BASE])
        if base is not None:
            np.testing.assert_array_equal(base, fresh)
        base = fresh
    target = target_hours(data, CUTOFF, orders)
    p = adjusted(base, target, orders, blend, policy)
    check_bounds(p, orders)
    dest = OUT/'submission.csv'
    temp = dest.with_suffix('.tmp')
    pd.DataFrame({'pkg_no':orders.pkg_no,'pred_hours':p}).to_csv(temp,index=False)
    checked = pd.read_csv(temp, float_precision='round_trip')
    pd.testing.assert_series_equal(orders.pkg_no, checked.pkg_no)
    np.testing.assert_array_equal(checked.pred_hours,p)
    temp.replace(dest)
    return dict(rows=len(p), shorter=int((p < base).sum()), longer=int((p > base).sum()),
                mean_hours=p.mean(), base_mean=base.mean(), sha256=hashlib.sha256(dest.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['replay','candidate'])
    parser.add_argument('--rebuild-base',action='store_true')
    parser.add_argument('--raw-csv',action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    data = load_history(args.raw_csv)
    if args.stage == 'replay':
        variants = [(b,p) for b in BLENDS for p in POLICIES]
        dev = replay(data, [*WINDOWS,COLD,EXTRA], variants)
        reference = dev.loc[dev.blend.eq(0)].set_index('start').A
        trials = dev.loc[dev.blend.gt(0)].copy()
        trials['delta'] = trials.A - trials.start.map(reference)
        keys = ['blend','policy']
        regular = trials.loc[trials.start.isin([w[0] for w in WINDOWS])]
        ranking = regular.groupby(keys).delta.mean().to_frame('mean_delta')
        ranking['worst_delta'] = trials.groupby(keys).delta.max()
        # Prefer no regressions on any of the six replays. If none qualify,
        # charge the worst regression in full alongside the mean change.
        ranking['criterion'] = ranking.mean_delta + ranking.worst_delta.clip(lower=0)
        eligible = ranking.loc[ranking.worst_delta.le(0)]
        blend, policy = (eligible if len(eligible) else ranking).criterion.idxmin()
        ranking.to_csv(OUT/'comparison.csv')
        (OUT/'settings.json').write_text(json.dumps(dict(blend=blend, policy=policy, selected_on=list(reference.index)),indent=2))
        rest = replay(data, [UNUSED], [(blend,policy)])
        all_scores = pd.concat([dev, rest], ignore_index=True)
        for filename, dates in [('scores.csv', [*WINDOWS,COLD]), ('extra.csv',[EXTRA])]:
            truth = pd.read_csv(ROOT/'output/tail_airport_check'/filename).query("method == 'width_soft'").set_index('cutoff')
            zero = all_scores.loc[all_scores.blend.eq(0)].set_index('start')
            for start,_ in dates:
                np.testing.assert_allclose(zero.loc[start,'A'],truth.loc[start,'A'],rtol=0,atol=1e-8)
        all_scores.to_csv(OUT/'replay.csv',index=False)
    else:
        config = json.loads((OUT/'settings.json').read_text())
        result = candidate(data,config['blend'],config['policy'],args.rebuild_base)
        (OUT/'acceptance.json').write_text(json.dumps(result,indent=2))
        print(result)


if __name__ == '__main__':
    main()
