"""v5目的州收缩方案：独立输出到output/v5，保留已验收v4。"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
import numpy as np
import pandas as pd
import pyarrow
from feature_groups_experiment import predict,MIN_GROUP_ROWS,SHRINK_PRIOR_ROWS
from remaining_experiment import weighted_remaining_predict
from weighted_risk import company_quantiles
from score import score

ROOT=Path(__file__).resolve().parents[1]
COLS=['lgs_company_nm','shpp_chnl','shpp_prvn_nm','pay_time','issue_time','sign_time','y_total_hours']
CONFIG={'version':'province-shrink-v5','age_days':21,'base_channel_min_rows':100,'province_min_rows':MIN_GROUP_ROWS,'province_shrink_prior_rows':SHRINK_PRIOR_ROWS,'province_weight':'n/(n+2000)','tail_rule':'clip(1-0.02*sqrt(company_train_share/0.125),0.97,0.995)','final_reference':'2026-09-21'}

def load_history(raw=False):
    test=ROOT/'data/test_202609.csv'
    if not test.is_file():
        raise FileNotFoundError('缺少data/test_202609.csv')
    slim=ROOT/'data/train_slim.parquet'
    csv=ROOT/'data/train_202608_0920.csv'
    if slim.is_file() and not raw:
        d=pd.read_parquet(slim,columns=COLS)
        source=slim.name
    elif csv.is_file():
        d=pd.read_csv(csv,usecols=COLS)
        source=csv.name
    else:
        raise FileNotFoundError('缺少训练数据：data/train_slim.parquet或train_202608_0920.csv')
    for col in ['pay_time','issue_time','sign_time']:
        d[col]=pd.to_datetime(d[col],errors='raise')
    required=[c for c in COLS if c!='shpp_prvn_nm']
    if d[required].isna().any().any() or not np.isfinite(d.y_total_hours).all():
        raise ValueError('必要训练字段缺失或标签不是有限数值')
    d['elapsed']=(d.issue_time-d.pay_time).dt.total_seconds()/3600
    d['remaining']=d.y_total_hours-d.elapsed
    d['province']=d.shpp_prvn_nm.fillna('MISSING').astype(str)
    return d,source

def group_metrics(va,p,cutoff,method):
    z=pd.DataFrame({'company':va.lgs_company_nm,'actual':va.y_total_hours,'prediction':p})
    z['ontime']=z.actual<=z.prediction
    z['severe']=z.actual-z.prediction>24
    g=z.groupby('company').agg(rows=('actual','size'),actual_mean=('actual','mean'),pred_mean=('prediction','mean'),ontime=('ontime','mean'),severe=('severe','mean')).reset_index()
    g['cutoff']=cutoff
    g['method']=method
    return g

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw-csv',action='store_true')
    args=parser.parse_args()
    started=time.perf_counter()
    out=ROOT/'output/v5'
    out.mkdir(parents=True,exist_ok=True)
    d,source=load_history(args.raw_csv)
    scores=[]
    groups=[]
    for cutoff,end in [('2026-08-25','2026-09-01'),('2026-09-01','2026-09-08'),('2026-09-08','2026-09-15'),('2026-09-15','2026-09-21')]:
        cut=pd.Timestamp(cutoff)
        tr=d[(d.pay_time<cut-pd.Timedelta(days=CONFIG['age_days']))&(d.sign_time<cut)&(d.issue_time<cut)&(d.elapsed>=0)&(d.remaining>=0)]
        va=d[(d.pay_time>=cut)&(d.pay_time<pd.Timestamp(end))]
        if tr.empty or va.empty:
            raise ValueError('时间窗口训练或验证样本为空')
        base=weighted_remaining_predict(tr,va)
        refined,covered=predict(tr,va,base,'province')
        for name,p in [('v4_baseline',base),('v5_province',refined)]:
            result=score(va.lgs_company_nm,va.y_total_hours,p)
            scores.append(dict(cutoff=cutoff,method=name,train_rows=len(tr),valid_rows=len(va),covered_rows=covered if name=='v5_province' else 0,**result))
            groups.append(group_metrics(va,p,cutoff,name))
            print(cutoff,name,round(result['A'],4),flush=True)
    pd.DataFrame(scores).to_csv(out/'scores.csv',index=False)
    pd.concat(groups).to_csv(out/'company_metrics.csv',index=False)
    tr=d[(d.pay_time<pd.Timestamp(CONFIG['final_reference'])-pd.Timedelta(days=CONFIG['age_days']))&(d.elapsed>=0)&(d.remaining>=0)]
    t=pd.read_csv(ROOT/'data/test_202609.csv',usecols=['pkg_no','lgs_company_nm','shpp_chnl','shpp_prvn_nm','pay_time','issue_time'])
    for col in ['pay_time','issue_time']:
        t[col]=pd.to_datetime(t[col],errors='raise')
    if t[['pkg_no','lgs_company_nm','shpp_chnl','pay_time','issue_time']].isna().any().any():
        raise ValueError('必要测试字段缺失')
    t['elapsed']=(t.issue_time-t.pay_time).dt.total_seconds()/3600
    t['province']=t.shpp_prvn_nm.fillna('MISSING').astype(str)
    base=weighted_remaining_predict(tr,t)
    p,covered=predict(tr,t,base,'province')
    if len(t)!=911415 or not t.pkg_no.is_unique or not np.isfinite(p).all() or not p.gt(0).all() or not (p>=t.elapsed-1e-9).all():
        raise ValueError('预测结果未通过行数、编号或数值检查')
    destination=out/'submission.csv'
    pd.DataFrame({'pkg_no':t.pkg_no,'pred_hours':p}).to_csv(destination,index=False)
    rules=company_quantiles(tr).rename('tail_quantile').to_frame()
    rules.index.name='lgs_company_nm'
    rules['train_rows']=tr.lgs_company_nm.value_counts()
    rules['train_share']=rules.train_rows/len(tr)
    rules.to_csv(out/'service_rules.csv')
    summary={'config':CONFIG,'training_source':source,'history_rows':len(d),'final_train_rows':len(tr),'prediction_rows':len(t),'province_covered_rows':covered,'fallback_rows':len(t)-covered,'mean_hours':float(p.mean()),'v4_mean_hours':float(base.mean()),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'runtime_seconds':round(time.perf_counter()-started,2),'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'pyarrow':pyarrow.__version__},'official_test_score':'UNKNOWN','validation_scope':'All windows are retrospective/development; no claim of untouched future test.'}
    (out/'run_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'config.json').write_text(json.dumps(CONFIG,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    main()
