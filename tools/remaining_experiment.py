"""已过去时间加历史剩余时效：固定规则对照，不使用未来标签校准。"""
from pathlib import Path
import numpy as np
import pandas as pd
from risk_sweep import predict as total_predict
from score import score
from weighted_risk import predict as weighted_predict

ROOT=Path(__file__).resolve().parents[1]
COLS=['lgs_company_nm','shpp_chnl','stk_cate_tp_nm','shpp_prvn_nm','pay_time','issue_time','sign_time','y_total_hours']

def load():
    slim=ROOT/'data/train_slim.parquet'
    d=pd.read_parquet(slim,columns=COLS) if slim.exists() else pd.read_csv(ROOT/'data/train_202608_0920.csv',usecols=COLS)
    for c in ['pay_time','issue_time','sign_time']:
        d[c]=pd.to_datetime(d[c])
    d['elapsed']=(d.issue_time-d.pay_time).dt.total_seconds()/3600
    d['remaining']=d.y_total_hours-d.elapsed
    return d

def remaining_predict(tr,va,extra=None):
    # 时间记录倒挂的样本不用于估计剩余时间；仅依据训练标签筛除。
    tr=tr[(tr.elapsed>=0)&(tr.remaining>=0)]
    def stats(keys):
        g=tr.groupby(keys,observed=True).remaining
        values=np.maximum(g.quantile(.9),g.quantile(.98)-24).clip(lower=0)
        return values[g.size()>=100]
    coarse=stats(['lgs_company_nm'])
    fallback=max(tr.remaining.quantile(.9),tr.remaining.quantile(.98)-24,0)
    pred=va.lgs_company_nm.map(coarse).astype(float).fillna(fallback)
    keys=['lgs_company_nm','shpp_chnl']
    fine=stats(keys)
    pred=pd.Series(pd.MultiIndex.from_frame(va[keys]).map(fine),index=va.index).fillna(pred)
    if extra:
        keys=keys+[extra]
        finer=stats(keys)
        pred=pd.Series(pd.MultiIndex.from_frame(va[keys]).map(finer),index=va.index).fillna(pred)
    elapsed=(pd.to_datetime(va.issue_time)-pd.to_datetime(va.pay_time)).dt.total_seconds()/3600
    return (elapsed.clip(lower=0)+pred).clip(lower=1)

def weighted_remaining_predict(tr,va):
    valid_tr=tr[(tr.elapsed>=0)&(tr.remaining>=0)].copy()
    valid_tr['y_total_hours']=valid_tr.remaining
    # 在剩余时效模型里将验证侧已过去时间置零，仅估计剩余小时数。
    zero_elapsed=va.copy()
    zero_elapsed['pay_time']=zero_elapsed.issue_time
    remaining=weighted_predict(valid_tr,zero_elapsed)
    elapsed=(pd.to_datetime(va.issue_time)-pd.to_datetime(va.pay_time)).dt.total_seconds()/3600
    return (elapsed.clip(lower=0)+remaining).clip(lower=1)

def main():
    d=load()
    print('History invalid remaining:',int(((d.elapsed<0)|(d.remaining<0)).sum()),flush=True)
    results=[]
    for cutoff,end in [('2026-09-01','2026-09-08'),('2026-09-08','2026-09-15'),('2026-09-15','2026-09-21')]:
        cut=pd.Timestamp(cutoff)
        tr=d[(d.pay_time<cut-pd.Timedelta(days=21))&(d.sign_time<cut)&(d.issue_time<cut)]
        va=d[(d.pay_time>=cut)&(d.pay_time<pd.Timestamp(end))]
        for method,extra in [('total_q98',None),('remaining_channel',None),('remaining_warehouse','stk_cate_tp_nm'),('remaining_province','shpp_prvn_nm'),('weighted_total',None),('weighted_remaining',None)]:
            if method=='total_q98':
                p=total_predict(tr,va,.9,.98)
            elif method=='weighted_total':
                p=weighted_predict(tr,va)
            elif method=='weighted_remaining':
                p=weighted_remaining_predict(tr,va)
            else:
                p=remaining_predict(tr,va,extra)
            r=score(va.lgs_company_nm,va.y_total_hours,p)
            results.append(dict(cutoff=cutoff,method=method,train_rows=len(tr),valid_rows=len(va),**r))
            print(cutoff,method,round(r['A'],4),flush=True)
    pd.DataFrame(results).to_csv(ROOT/'output/remaining_scores.csv',index=False)

if __name__=='__main__':
    main()
