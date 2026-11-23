"""CatBoost分位数模型，固定参数比较两个时间窗口。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from score import score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'
NUM=['pkg_goods_cnt','weight']
CAT=['shpp_prvn_nm','shpp_city_nm','stk_cate_tp_nm','shpp_chnl','lgs_company_nm','combine_flag','combine_type','start_airport_cd','end_airport_cd','mode_category','business_tp','operation_type','route_cat','sale_area','send_country_nm']
FEATURES=NUM+CAT+['pay_hour','pay_weekday','issue_hour','issue_weekday','elapsed_hours']

def features(d):
    x=d[NUM+CAT].copy()
    for c in CAT:
        x[c]=x[c].fillna('MISSING').astype(str)
    for c in NUM:
        x[c]=pd.to_numeric(x[c],errors='coerce').fillna(-1)
    pay=pd.to_datetime(d.pay_time)
    issue=pd.to_datetime(d.issue_time)
    x['pay_hour']=pay.dt.hour
    x['pay_weekday']=pay.dt.day_of_week
    x['issue_hour']=issue.dt.hour
    x['issue_weekday']=issue.dt.day_of_week
    x['elapsed_hours']=(issue-pay).dt.total_seconds()/3600
    return x[FEATURES]

def fit(train):
    from catboost import CatBoostRegressor

    m=CatBoostRegressor(loss_function='MultiQuantile:alpha=0.9,0.97',iterations=220,depth=6,learning_rate=.08,random_seed=42,thread_count=4,verbose=False,allow_writing_files=False)
    m.fit(features(train),train.y_total_hours,cat_features=CAT)
    return m

def prediction(m,d):
    q=m.predict(features(d))
    elapsed=(pd.to_datetime(d.issue_time)-pd.to_datetime(d.pay_time)).dt.total_seconds().to_numpy()/3600
    return np.maximum.reduce([q[:,0],q[:,1]-24,elapsed,np.ones(len(d))])

def main():
    OUT.mkdir(exist_ok=True)
    d=pd.read_csv(ROOT/'data/train_sample_300k.csv',parse_dates=['pay_time','sign_time'])
    rows=[]
    for cutoff,end in [('2026-09-08','2026-09-15'),('2026-09-15','2026-09-21')]:
        cut=pd.Timestamp(cutoff)
        tr=d[(d.pay_time<cut-pd.Timedelta(days=21))&(d.sign_time<cut)]
        va=d[(d.pay_time>=cut)&(d.pay_time<pd.Timestamp(end))]
        print('Training',cutoff,len(tr),'validation',len(va),flush=True)
        m=fit(tr)
        r=score(va.lgs_company_nm,va.y_total_hours,prediction(m,va))
        rows.append(dict(cutoff=cutoff,train_rows=len(tr),valid_rows=len(va),**r))
        print(rows[-1],flush=True)
        pd.DataFrame(rows).to_csv(OUT/'model_scores.csv',index=False)
    # 两窗口模型已检查过，另存候选，不覆盖统计候选。
    tr=d[d.pay_time<pd.Timestamp('2026-08-31')]
    m=fit(tr)
    m.save_model(str(OUT/'quantile_model.cbm'))
    importance=pd.DataFrame({'feature':FEATURES,'importance':m.feature_importances_}).sort_values('importance',ascending=False)
    importance.to_csv(OUT/'feature_importance.csv',index=False)
    destination=OUT/'submission_model_candidate.csv'
    for i,test in enumerate(pd.read_csv(ROOT/'data/test_202609.csv',chunksize=50000)):
        pred=prediction(m,test)
        assert np.isfinite(pred).all() and (pred>0).all()
        pd.DataFrame({'pkg_no':test.pkg_no,'pred_hours':pred}).to_csv(destination,index=False,mode='w' if i==0 else 'a',header=i==0)
    print('Model candidate complete',flush=True)

if __name__=='__main__':
    main()
