"""固定附加分组与收缩，不改已验收预测。"""
from pathlib import Path
import numpy as np
import pandas as pd
from remaining_experiment import weighted_remaining_predict
from weighted_risk import company_quantiles
from score import score

ROOT=Path(__file__).resolve().parents[1]
MIN_GROUP_ROWS=500
SHRINK_PRIOR_ROWS=2000

def load():
    columns=['lgs_company_nm','shpp_chnl','shpp_prvn_nm','weight','pay_time','issue_time','sign_time','y_total_hours']
    path=ROOT/'data/train_slim.parquet'
    d=pd.read_parquet(path,columns=columns) if path.exists() else pd.read_csv(ROOT/'data/train_202608_0920.csv',usecols=columns)
    for c in ['pay_time','issue_time','sign_time']:
        d[c]=pd.to_datetime(d[c])
    d['elapsed']=(d.issue_time-d.pay_time).dt.total_seconds()/3600
    d['remaining']=d.y_total_hours-d.elapsed
    d['province']=d.shpp_prvn_nm.fillna('MISSING').astype(str)
    d['weight_band']=pd.cut(d.weight,[-np.inf,100,300,1000,3000,np.inf]).astype(str)
    d['issue_weekday']=d.issue_time.dt.day_of_week.astype(str)
    d['elapsed_band']=pd.cut(d.elapsed,[-np.inf,6,12,24,48,np.inf]).astype(str)
    return d

def predict(tr,va,base,feature):
    tail=company_quantiles(tr)
    fine={}
    counts={}
    for company,part in tr.groupby('lgs_company_nm'):
        g=part.groupby(['shpp_chnl',feature]).remaining
        n=g.size()
        q=np.maximum(g.quantile(.9),g.quantile(float(tail.loc[company]))-24).clip(lower=1)
        for (channel,value),promise in q[n>=MIN_GROUP_ROWS].items():
            key=(company,channel,value)
            fine[key]=promise
            counts[key]=int(n.loc[(channel,value)])
    if not fine:
        return base.copy(),0
    table=pd.Series(fine)
    table.index=pd.MultiIndex.from_tuples(table.index)
    ns=pd.Series(counts)
    ns.index=pd.MultiIndex.from_tuples(ns.index)
    keys=pd.MultiIndex.from_frame(va[['lgs_company_nm','shpp_chnl',feature]])
    detail=pd.Series(keys.map(table),index=va.index)
    size=pd.Series(keys.map(ns),index=va.index).fillna(0)
    ratio=size/(size+SHRINK_PRIOR_ROWS)
    prior=base-va.elapsed.clip(lower=0)
    remaining=ratio*detail.fillna(prior)+(1-ratio)*prior
    return (va.elapsed.clip(lower=0)+remaining).clip(lower=1),int(detail.notna().sum())

def main():
    d=load()
    rows=[]
    daily=[]
    companies=[]
    for cutoff,end in [('2026-08-25','2026-09-01'),('2026-09-01','2026-09-08'),('2026-09-08','2026-09-15'),('2026-09-15','2026-09-21')]:
        cut=pd.Timestamp(cutoff)
        tr=d[(d.pay_time<cut-pd.Timedelta(days=21))&(d.sign_time<cut)&(d.issue_time<cut)&(d.elapsed>=0)&(d.remaining>=0)]
        va=d[(d.pay_time>=cut)&(d.pay_time<pd.Timestamp(end))]
        baseline=weighted_remaining_predict(tr,va)
        features=['baseline','province'] if cutoff=='2026-08-25' else ['baseline','province','weight_band','issue_weekday','elapsed_band']
        for feature in features:
            pred,covered=(baseline,0) if feature=='baseline' else predict(tr,va,baseline,feature)
            r=score(va.lgs_company_nm,va.y_total_hours,pred)
            rows.append(dict(cutoff=cutoff,feature=feature,covered_rows=covered,valid_rows=len(va),**r))
            print(cutoff,feature,round(r['A'],4),'covered',covered,flush=True)
            if feature in ['baseline','province']:
                detail=pd.DataFrame({'day':va.pay_time.dt.strftime('%Y-%m-%d'),'company':va.lgs_company_nm,'actual':va.y_total_hours,'prediction':pred})
                detail['ontime']=detail.actual<=detail.prediction
                detail['severe']=detail.actual-detail.prediction>24
                stats=detail.groupby(['day','company']).agg(rows=('actual','size'),sum_actual=('actual','sum'),sum_prediction=('prediction','sum'),ontime_count=('ontime','sum'),severe_count=('severe','sum')).reset_index()
                stats['cutoff']=cutoff
                stats['method']='v4_baseline' if feature=='baseline' else 'province_candidate'
                daily.append(stats)
                g=detail.groupby('company').agg(rows=('actual','size'),actual_mean=('actual','mean'),pred_mean=('prediction','mean'),ontime=('ontime','mean'),severe=('severe','mean')).reset_index()
                g['cutoff']=cutoff
                g['feature']=feature
                companies.append(g)
    pd.DataFrame(rows).to_csv(ROOT/'output/feature_groups_scores.csv',index=False)
    pd.concat(daily).to_csv(ROOT/'output/province_daily_stats.csv',index=False)
    pd.concat(companies).to_csv(ROOT/'output/province_company_metrics.csv',index=False)

if __name__=='__main__':
    main()
