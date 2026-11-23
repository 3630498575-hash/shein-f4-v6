"""固定21天观察间隔，比较尾部风险设置；两个窗口均属开发验证。"""
from pathlib import Path
import numpy as np
import pandas as pd
from score import score

ROOT=Path(__file__).resolve().parents[1]

def load_history():
    cols=['lgs_company_nm','shpp_chnl','pay_time','issue_time','sign_time','y_total_hours']
    slim=ROOT/'data/train_slim.parquet'
    if slim.exists():
        d=pd.read_parquet(slim,columns=cols)
    else:
        d=pd.read_csv(ROOT/'data/train_202608_0920.csv',usecols=cols)
    for c in ['pay_time','sign_time']:
        d[c]=pd.to_datetime(d[c],errors='raise')
    if d[['lgs_company_nm','shpp_chnl','pay_time','sign_time','y_total_hours']].isna().any().any():
        raise ValueError('必要训练字段缺失，请检查输入数据')
    return d

def predict(tr,va,a,b):
    if tr.empty or not (0<a<=b<1):
        raise ValueError('训练集不能为空，且分位数需满足0<a<=b<1')
    group=tr.groupby('lgs_company_nm').y_total_hours
    coarse=np.maximum(group.quantile(a),group.quantile(b)-24)
    finegroup=tr.groupby(['lgs_company_nm','shpp_chnl']).y_total_hours
    fine=np.maximum(finegroup.quantile(a),finegroup.quantile(b)-24)
    fine=fine[finegroup.size()>=100]
    keys=pd.MultiIndex.from_frame(va[['lgs_company_nm','shpp_chnl']])
    fallback=max(tr.y_total_hours.quantile(a),tr.y_total_hours.quantile(b)-24)
    p=pd.Series(keys.map(fine),index=va.index).fillna(va.lgs_company_nm.map(coarse)).fillna(fallback)
    elapsed=(pd.to_datetime(va.issue_time)-pd.to_datetime(va.pay_time)).dt.total_seconds()/3600
    return np.maximum(p,elapsed).clip(lower=1)

def main():
    d=load_history()
    (ROOT/'output').mkdir(exist_ok=True)
    rows=[]
    for cutoff,end in [('2026-09-08','2026-09-15'),('2026-09-15','2026-09-21')]:
        cut=pd.Timestamp(cutoff)
        tr=d[(d.pay_time<cut-pd.Timedelta(days=21))&(d.sign_time<cut)]
        va=d[(d.pay_time>=cut)&(d.pay_time<pd.Timestamp(end))]
        for a in [.9,.92,.94]:
            for b in [.97,.98,.99]:
                p=predict(tr,va,a,b)
                rows.append(dict(cutoff=cutoff,a=a,b=b,**score(va.lgs_company_nm,va.y_total_hours,p)))
        print('Finished',cutoff,flush=True)
    r=pd.DataFrame(rows)
    r.to_csv(ROOT/'output/risk_sweep.csv',index=False)
    chosen=r[r.cutoff=='2026-09-08'].sort_values('A').iloc[0]
    print(r[(r.a==chosen.a)&(r.b==chosen.b)].to_string(index=False),flush=True)
    test=pd.read_csv(ROOT/'data/test_202609.csv',usecols=['pkg_no','lgs_company_nm','shpp_chnl','pay_time','issue_time'])
    tr=d[d.pay_time<pd.Timestamp('2026-08-31')]
    test['pred_hours']=predict(tr,test,float(chosen.a),float(chosen.b))
    assert len(test)==911415 and test.pkg_no.is_unique
    assert np.isfinite(test.pred_hours).all() and test.pred_hours.gt(0).all()
    test[['pkg_no','pred_hours']].to_csv(ROOT/'output/submission_risk_candidate.csv',index=False)
    print('New risk candidate saved:',len(test),flush=True)

if __name__=='__main__':
    main()
