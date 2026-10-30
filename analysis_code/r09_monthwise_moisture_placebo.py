#!/usr/bin/env python3
"""R09 retrospective humidity daily-alignment placebo test. NOT a humidity intervention.
Freeze R04 ZIP SHA; regenerate daily data from source via R08 source script; leave one city out.
Within each city/calendar month, circular-shift latent_potential by >=7 days. Preserves
monthly moisture values, some within-month autocorrelation and city-month climatology.
Full candidate refit per shift; same ridge regularization and no held-out target leakage.
"""
import argparse, pathlib, subprocess, sys, hashlib, json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
CITIES=['Doha','Riyadh','Kuwait City','Muscat']
SRC_SHA='3f9611957b6652dee5f314d30f43888daed4346be80b3b4e639ec85076af149f'
RICH=['T','Tmax','Tmin','CDH20','CDH24','CDH28','Tdev2','weekday','GHI','sin1','cos1','sin2','cos2']
ALPHA=10.0
SEED=8209

def sha(f):
 h=hashlib.sha256()
 with open(f,'rb') as b:
  for c in iter(lambda:b.read(1048576),b''):h.update(c)
 return h.hexdigest()
def fit_predict(train,test,features):
 m=make_pipeline(StandardScaler(),Ridge(alpha=ALPHA))
 m.fit(train[features].to_numpy(float),train['e'].to_numpy(float))
 return np.maximum(m.predict(test[features].to_numpy(float)),0)
def rotate_monthly(df,rng):
 d=df.copy()
 for _,grp in d.groupby(['city','month'],sort=True):
  ix=grp.index.to_numpy(); n=len(ix)
  # months all have >= 28 days; choose shift of >=7 and <= n-7 days.
  shift=int(rng.integers(7,n-6))
  d.loc[ix,'latent_potential']=np.roll(df.loc[ix,'latent_potential'].to_numpy(),shift)
 return d

def run(src,engine,out,nrep=200):
 src=pathlib.Path(src);out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
 assert sha(src)==SRC_SHA,(sha(src),SRC_SHA)
 print('Raw source freeze:',SRC_SHA)
 # independently reconstruct daily data using R08 source-generating function
 sys.path.insert(0,str(pathlib.Path(engine).parent))
 import r08_stronger_baselines as r08
 import zipfile
 with zipfile.ZipFile(src) as z:
  assert z.testzip() is None
  data=pd.concat([r08.daily_features(z,slug) for slug in r08.CITYMAP],ignore_index=True)
 assert len(data)==1460 and set(data.city)==set(CITIES)
 data['month']=pd.to_datetime(data.date).dt.month
 assert not data.isna().any().any()
 orig_dig=sha(pathlib.Path(engine).parent/'results'/'R08_Daily_Derived_From_Frozen_R04.csv') if (pathlib.Path(engine).parent/'results'/'R08_Daily_Derived_From_Frozen_R04.csv').exists() else None
 # baseline: rich temperature/solar model vs true humidity model
 rows=[]; forecasts=[]; rng=np.random.default_rng(SEED)
 for city in CITIES:
  tr=data[data.city!=city];te=data[data.city==city]
  yp=fit_predict(tr,te,RICH)
  yt=fit_predict(tr,te,RICH+['latent_potential'])
  mae=lambda p: float(np.mean(np.abs(te.e.to_numpy()-p)))
  mb=mae(yp);mt=mae(yt)
  rows.append({'city':city,'type':'real','seed_index':-1,'baseline_MAE':mb,'candidate_MAE':mt,'improvement_pct':100*(mb-mt)/mb})
  forecasts.extend([{'city':city,'date':str(date),'y':float(y),'temperature_prediction':float(a),'true_humidity_prediction':float(b)} for date,y,a,b in zip(te.date,te.e,yp,yt)])
 # for each replicate, shuffle whole data independently and fit/out-of-city test
 for i in range(nrep):
  shuffled=rotate_monthly(data,rng)
  for city in CITIES:
   tr=shuffled[shuffled.city!=city];te=shuffled[shuffled.city==city]
   yp=fit_predict(tr,te,RICH+['latent_potential'])
   y=te.e.to_numpy(float)
   mb=next(x['baseline_MAE'] for x in rows if x['city']==city and x['type']=='real')
   m=float(np.mean(np.abs(y-yp)))
   rows.append({'city':city,'type':'monthwise_shift','seed_index':i,'baseline_MAE':mb,'candidate_MAE':m,'improvement_pct':100*(mb-m)/mb})
 result=pd.DataFrame(rows)
 result.to_csv(out/'R09_MonthlyShift_NegativeControl_Full.csv',index=False,float_format='%.12g')
 pd.DataFrame(forecasts).to_csv(out/'R09_Fixed_TrueMoisture_HeldOutForecasts.csv',index=False,float_format='%.12g')
 summ=[]
 for city in CITIES:
  a=result[(result.city==city)&(result.type=='real')].iloc[0]
  b=result[(result.city==city)&(result.type=='monthwise_shift')]
  q=np.quantile(b.improvement_pct,[0.025,.5,.975])
  rank=(int(np.sum(b.candidate_MAE>=a.candidate_MAE))+1)/(nrep+1) # fraction placebos worse than real
  p=(int(np.sum(b.candidate_MAE<=a.candidate_MAE))+1)/(nrep+1) # placebo as-good-or-better
  summ.append({'city':city,'baseline_MAE':a.baseline_MAE,'true_moisture_MAE':a.candidate_MAE,
   'true_improvement_pct':a.improvement_pct,'placebo_median_improvement_pct':q[1],
   'placebo_q025_pct':q[0],'placebo_q975_pct':q[2],
   'true_minus_placebo_median_pctpts':a.improvement_pct-q[1],
   'fraction_placebos_better_or_equal':(b.candidate_MAE<=a.candidate_MAE).mean(),
   'placebo_tail_rank_exploratory':p,'n_monthwise_placebos':nrep,
   'test_days':365,'training_city_count':3})
 outtable=pd.DataFrame(summ)
 outtable.to_csv(out/'R09_Placebo_City_Summary.csv',index=False,float_format='%.12g')
 observed_r08=pd.read_csv(pathlib.Path(engine).parent/'results'/'R08_LeaveCityOut_Model_Metrics.csv')
 for x in summ:
  rr=observed_r08[(observed_r08.target=='e')&(observed_r08.heldout_city==x['city'])&(observed_r08.model=='Rich_temperature_solar_plus_moisture')].iloc[0]
  assert abs(float(rr.MAE_kWh_d)-x['true_moisture_MAE'])<1e-8
 print(outtable.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
 (out/'R09_NegativeControl_Protocol.json').write_text(json.dumps({'title':'Month-preserving circular shift negative control of humidity alignment',
  'source_R04_sha256':SRC_SHA,'model':'R08 rich temp/solar/day/season with vs without latent_potential','alpha':ALPHA,'random_seed':SEED,
  'n_placebos':nrep,'cities':CITIES,'scope':'post-R08 retrospective falsification; months within each city shifted 7+ days; no field data or causal intervention',
  'interpretation':'Site-month climatology and unobserved site-level confounders preserved; daily alignment perturbed. Permutation contrasts are diagnostics, not p values, independent prospective validation, or HVAC energy savings.',
  'r08_daily_sha256':orig_dig,'file_hashes':{p.name:sha(p) for p in out.glob('*.csv')}},indent=2),encoding='utf-8')
 return outtable

if __name__=='__main__':
 x=argparse.ArgumentParser();x.add_argument('--source',required=True);x.add_argument('--r08-engine',required=True);x.add_argument('--output',required=True);x.add_argument('--n',type=int,default=200)
 a=x.parse_args();run(a.source,a.r08_engine,a.output,a.n)
