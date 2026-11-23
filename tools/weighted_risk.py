"""按历史服务商占比设置尾部余量，规则事先固定，不使用验证标签拟合。"""
from pathlib import Path
import numpy as np
import pandas as pd
from risk_sweep import load_history, predict as uniform_predict
from score import score

ROOT=Path(__file__).resolve().parents[1]

def company_quantiles(tr):
    share=tr.lgs_company_nm.value_counts(normalize=True)
    # 订单份额小的服务商提高尾部分位数；最大服务商不低于97%。
    # 这是固定启发式，不是评分目标的解析最优解。
    return (1-.02*np.sqrt(share/.125)).clip(lower=.97,upper=.995)

def predict(tr,va):
    if tr.empty:
        raise ValueError('训练集不能为空')
    quantiles=company_quantiles(tr)
    coarse={}
    fine={}
    for company,part in tr.groupby('lgs_company_nm',observed=True):
        b=float(quantiles.loc[company])
        coarse[company]=max(part.y_total_hours.quantile(.9),part.y_total_hours.quantile(b)-24)
        groups=part.groupby('shpp_chnl',observed=True).y_total_hours
        values=np.maximum(groups.quantile(.9),groups.quantile(b)-24)
        for channel,value in values[groups.size()>=100].items():
            fine[(company,channel)]=value
    fine=pd.Series(fine,dtype=float)
    if not fine.empty:
        fine.index=pd.MultiIndex.from_tuples(fine.index)
    fallback=max(tr.y_total_hours.quantile(.9),tr.y_total_hours.quantile(.98)-24)
    coarse_pred=va.lgs_company_nm.map(coarse).astype(float).fillna(fallback)
    if fine.empty:
        p=coarse_pred
    else:
        keys=pd.MultiIndex.from_frame(va[['lgs_company_nm','shpp_chnl']])
        p=pd.Series(keys.map(fine),index=va.index).fillna(coarse_pred)
    elapsed=(pd.to_datetime(va.issue_time)-pd.to_datetime(va.pay_time)).dt.total_seconds()/3600
    return np.maximum(p,elapsed).clip(lower=1)

def main():
    d=load_history()
    rows=[]
    for cutoff,end in [('2026-09-01','2026-09-08'),('2026-09-08','2026-09-15'),('2026-09-15','2026-09-21')]:
        cut=pd.Timestamp(cutoff)
        tr=d[(d.pay_time<cut-pd.Timedelta(days=21))&(d.sign_time<cut)]
        va=d[(d.pay_time>=cut)&(d.pay_time<pd.Timestamp(end))]
        for method in ['uniform_q98','weighted_tail']:
            p=uniform_predict(tr,va,.9,.98) if method=='uniform_q98' else predict(tr,va)
            r=score(va.lgs_company_nm,va.y_total_hours,p)
            rows.append(dict(cutoff=cutoff,method=method,**r))
            print(cutoff,method,round(r['A'],4),flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'output/weighted_risk_scores.csv',index=False)

if __name__=='__main__':
    main()
