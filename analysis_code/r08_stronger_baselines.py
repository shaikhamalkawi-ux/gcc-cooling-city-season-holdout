#!/usr/bin/env python3
"""Project 82 R08 independent stronger-comparator audit, not a new E+ simulation.

Inputs: unchanged SHA-locked R04 2023 EnergyPlus archive; all targets simulator-generated.
Four cities only because R04 Abu Dhabi has 8 severe warmup failures.
Per fold leave whole target city out. Time/calendar terms are only nuisance controls.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, zipfile
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

SHA_R04='3f9611957b6652dee5f314d30f43888daed4346be80b3b4e639ec85076af149f'
CITYMAP={'riyadh':'Riyadh','doha':'Doha','muscat':'Muscat','kuwait_city':'Kuwait City'}
COLS={'e':'Cooling_Electricity_kWh','total':'Cooling_Coil_Total_Cooling_Energy_kWh','sensible':'Cooling_Coil_Sensible_Cooling_Energy_kWh','latent':'Cooling_Coil_Latent_Cooling_Energy_kWh'}
TREF=24.0; RHREF=0.5

def sha_file(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()

def saturation(t):
 return 610.94*np.exp((17.625*t)/(t+243.04))

def daily_features(z,slug):
 epw=pd.read_csv(io.BytesIO(z.read('EPW_2023/'+slug+'.epw')),skiprows=8,header=None)
 energy=pd.read_csv(io.BytesIO(z.read(f'R04_{slug}_hourly_meters_coils.csv')))
 assert len(epw)==len(energy)==8760 and epw.shape[1]==35
 assert set(epw.iloc[:,0])=={2023} and set(energy['year'])=={2017}
 for k,j in [('month',1),('day',2),('hour',3)]:
  assert np.array_equal(energy[k],epw.iloc[:,j]),(slug,k)
 assert np.array_equal(energy.time_index,np.arange(1,8761))
 t=epw.iloc[:,6].to_numpy(float); dew=epw.iloc[:,7].to_numpy(float)
 rh=epw.iloc[:,8].to_numpy(float); p=epw.iloc[:,9].to_numpy(float)
 ghi=epw.iloc[:,13].to_numpy(float)
 assert np.isfinite(np.stack((t,dew,rh,p,ghi))).all()
 assert p.min()>50000 and p.max()<110000 and (rh>=0).all() and (rh<=100).all()
 assert ghi.min()>=0
 # Source is not modified; identical analysis-side clipping to R06
 pv=saturation(np.minimum(dew,t))
 w=.621945*pv/(p-pv)
 vin=RHREF*saturation(TREF)
 win=.621945*vin/(p-vin)
 latent=np.maximum(w-win,0)*(2501+1.86*TREF)
 h=1.006*t+w*(2501+1.86*t)
 hin=1.006*TREF+win*(2501+1.86*TREF)
 frame=pd.DataFrame({'date':pd.to_datetime(dict(year=2023,month=epw.iloc[:,1],day=epw.iloc[:,2])),
    'T':t,'RH':rh,'Td':dew,'GHI':ghi,'w':w,'enthalpy_excess':np.maximum(h-hin,0),
    'latent_potential':latent,'Tdev2':np.maximum(t-TREF,0)**2,'dew_overshoot_flag':dew>t,
    'e':energy[COLS['e']].to_numpy(float),'latent_coil':energy[COLS['latent']].to_numpy(float)})
 for threshold in [18,20,22,24,26,28,30,32]:
  frame[f'CDH{threshold}']=np.maximum(t-threshold,0)
 assert np.abs(energy[COLS['total']]-energy[COLS['sensible']]-energy[COLS['latent']]).max()<1.e-4
 agg={'e':'sum','latent_coil':'sum','GHI':'sum','latent_potential':'sum','enthalpy_excess':'sum','Tdev2':'sum','dew_overshoot_flag':'sum','T':'mean','RH':'mean','Td':'mean'}
 for threshold in [18,20,22,24,26,28,30,32]: agg[f'CDH{threshold}']='sum'
 day=frame.groupby('date',as_index=False).agg(agg)
 day['Tmax']=frame.groupby('date')['T'].max().values
 day['Tmin']=frame.groupby('date')['T'].min().values
 day['city']=CITYMAP[slug]
 day['weekday']=(day.date.dt.dayofweek<5).astype(int)
 doy=day.date.dt.dayofyear.to_numpy()
 for j in (1,2):
  day[f'sin{j}']=np.sin(2*np.pi*j*(doy-1)/365)
  day[f'cos{j}']=np.cos(2*np.pi*j*(doy-1)/365)
 day['dewpoint_above_db_count']=day['dew_overshoot_flag'].astype(int)
 day=day.drop(columns=['dew_overshoot_flag'])
 assert len(day)==365 and day['e'].sum()>0
 return day

FAMILIES={
 'R06_simple_temperature':['CDH24','weekday'],
 'R06_simple_plus_moisture':['CDH24','weekday','latent_potential'],
 'R06_simple_plus_solar':['CDH24','weekday','GHI'],
 'R06_simple_plus_solar_moisture':['CDH24','weekday','GHI','latent_potential'],
 'Seasonal_temperature_solar':['CDH24','weekday','GHI','sin1','cos1','sin2','cos2'],
 'Seasonal_temperature_solar_plus_moisture':['CDH24','weekday','GHI','sin1','cos1','sin2','cos2','latent_potential'],
 'Flexible_CDH_temperature_solar':['CDH18','CDH22','CDH24','CDH26','CDH30','weekday','GHI','sin1','cos1','sin2','cos2'],
 'Flexible_CDH_temperature_solar_plus_moisture':['CDH18','CDH22','CDH24','CDH26','CDH30','weekday','GHI','sin1','cos1','sin2','cos2','latent_potential'],
 'Rich_temperature_solar':['T','Tmax','Tmin','CDH20','CDH24','CDH28','Tdev2','weekday','GHI','sin1','cos1','sin2','cos2'],
 'Rich_temperature_solar_plus_moisture':['T','Tmax','Tmin','CDH20','CDH24','CDH28','Tdev2','weekday','GHI','sin1','cos1','sin2','cos2','latent_potential'],
 'Rich_temperature_solar_plus_meanRH':['T','Tmax','Tmin','CDH20','CDH24','CDH28','Tdev2','weekday','GHI','sin1','cos1','sin2','cos2','RH'],
 'Rich_temperature_solar_plus_meanTd':['T','Tmax','Tmin','CDH20','CDH24','CDH28','Tdev2','weekday','GHI','sin1','cos1','sin2','cos2','Td'],
 'Rich_temperature_solar_plus_enthalpy':['T','Tmax','Tmin','CDH20','CDH24','CDH28','Tdev2','weekday','GHI','sin1','cos1','sin2','cos2','enthalpy_excess'],
 'Rich_temperature_solar_more_temperature':['T','Tmax','Tmin','CDH20','CDH24','CDH28','Tdev2','weekday','GHI','sin1','cos1','sin2','cos2','CDH32'],
}
PAIRS=[('R06_simple_temperature','R06_simple_plus_moisture'),
('R06_simple_plus_solar','R06_simple_plus_solar_moisture'),
('Seasonal_temperature_solar','Seasonal_temperature_solar_plus_moisture'),
('Flexible_CDH_temperature_solar','Flexible_CDH_temperature_solar_plus_moisture'),
('Rich_temperature_solar','Rich_temperature_solar_plus_moisture'),
('Rich_temperature_solar','Rich_temperature_solar_plus_meanRH'),
('Rich_temperature_solar','Rich_temperature_solar_plus_meanTd'),
('Rich_temperature_solar','Rich_temperature_solar_plus_enthalpy'),
('Rich_temperature_solar_more_temperature','Rich_temperature_solar_plus_moisture')]

def compare(data,target,penalty=10.0):
 out=[]; pp=[]; folds=[]
 for city in CITYMAP.values():
  tr=data[data.city!=city].copy(); te=data[data.city==city].copy()
  assert len(tr)==1095 and len(te)==365 and tr.city.nunique()==3
  for name,cols in FAMILIES.items():
   assert len(cols)==len(set(cols))
   pipe=make_pipeline(StandardScaler(),Ridge(alpha=penalty))
   pipe.fit(tr[cols].values,tr[target].values)
   pred=np.maximum(pipe.predict(te[cols].values),0)
   actual=te[target].to_numpy(float)
   ae=np.abs(actual-pred)
   out.append({'target':target,'heldout_city':city,'model':name,'fold_train_cities':3,
    'train_rows':len(tr),'test_rows':len(te),'n_predictors':len(cols),'ridge_alpha':penalty,
    'MAE_kWh_d':float(ae.mean()),'RMSE_kWh_d':float(np.sqrt(np.mean((actual-pred)**2))),
    'annual_absolute_bias_pct':float(abs(100*(pred.sum()-actual.sum())/actual.sum())),
    'annual_signed_bias_pct':float(100*(pred.sum()-actual.sum())/actual.sum()),
    'summer_MAE_kWh_d':float(ae[te.date.dt.month.isin([6,7,8,9]).values].mean()),
    'winter_MAE_kWh_d':float(ae[te.date.dt.month.isin([12,1,2]).values].mean()),
    'sample_n':len(ae),'MAE_method':'arithmetic mean absolute errors on 365 held-out days'})
   pp.extend({'target':target,'heldout_city':city,'model':name,'date':str(d.date()),
    'actual':float(y),'pred':float(p),'abs_error':float(err)}
    for d,y,p,err in zip(te.date,actual,pred,ae))
 return pd.DataFrame(out),pd.DataFrame(pp)

def make_pairs(metrics, preds, rng_seed=8208, bs=1000, block_days=7):
 rows=[]
 for target in ('e','latent_coil'):
  for city in CITYMAP.values():
   for left,right in PAIRS:
    a=metrics[(metrics.target==target)&(metrics.heldout_city==city)&(metrics.model==left)].iloc[0]
    b=metrics[(metrics.target==target)&(metrics.heldout_city==city)&(metrics.model==right)].iloc[0]
    x=preds[(preds.target==target)&(preds.heldout_city==city)&(preds.model==left)].sort_values('date')
    y=preds[(preds.target==target)&(preds.heldout_city==city)&(preds.model==right)].sort_values('date')
    assert len(x)==len(y)==365 and np.array_equal(x.date.values,y.date.values)
    erra=x.abs_error.values;errb=y.abs_error.values
    assert np.isclose(erra.mean(),a.MAE_kWh_d)
    assert np.isclose(errb.mean(),b.MAE_kWh_d)
    # moving-circular block bootstrap of already fitted prediction errors, conditional only.
    rg=np.random.default_rng(rng_seed + sum(ord(c) for c in (target+city+left+right)))
    starts=rg.integers(0,365,size=(bs,math.ceil(365/block_days)))
    offs=np.arange(block_days)
    idx=((starts[:,:,None]+offs[None,None,:])%365).reshape(bs,-1)[:,:365]
    bootstrap_pct=100*(erra[idx].mean(axis=1)-errb[idx].mean(axis=1))/erra[idx].mean(axis=1)
    summer=(pd.to_datetime(x.date).dt.month.isin([6,7,8,9])).values
    winter=(pd.to_datetime(x.date).dt.month.isin([12,1,2])).values
    rows.append({'target':target,'heldout_city':city,'baseline':left,'added_or_comparator':right,
      'baseline_MAE':float(a.MAE_kWh_d),'candidate_MAE':float(b.MAE_kWh_d),
      'MAE_improvement_pct':float(100*(a.MAE_kWh_d-b.MAE_kWh_d)/a.MAE_kWh_d),
      'summer_MAE_improvement_pct':float(100*(erra[summer].mean()-errb[summer].mean())/erra[summer].mean()),
      'winter_MAE_improvement_pct':float(100*(erra[winter].mean()-errb[winter].mean())/erra[winter].mean()),
      'fixed_predictions_weekblock_95pct_L':float(np.quantile(bootstrap_pct,.025)),
      'fixed_predictions_weekblock_95pct_U':float(np.quantile(bootstrap_pct,.975)),
      'bootstrap_blocks_days':block_days,'bootstrap_draws':bs,'bootstrap_seed_base':rng_seed,
      'bootstrap_scope':'fixed predictions; within one year, not cross-city/annual population inference'})
 return pd.DataFrame(rows)

def main():
 arg=argparse.ArgumentParser();arg.add_argument('--source',required=True);arg.add_argument('--output',required=True);args=arg.parse_args()
 path=Path(args.source); out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 assert sha_file(path)==SHA_R04, 'source R04 SHA mismatched: do not run on unadmitted source'
 with zipfile.ZipFile(path) as z:
  assert z.testzip() is None
  status=json.loads(z.read('RUN_STATUS.json'))
  assert status['simulation_gate']=='HOLD'
  assert sum(x['status']=='PASS' for x in status['sites'])==4
  assert next(x for x in status['sites'] if x['city']=='Abu Dhabi')['severe']==8
  d=pd.concat([daily_features(z,slug) for slug in CITYMAP],ignore_index=True).sort_values(['city','date']).reset_index(drop=True)
 assert len(d)==1460 and d.city.nunique()==4
 csv_path=out/'R08_Daily_Derived_From_Frozen_R04.csv';d.to_csv(csv_path,index=False,float_format='%.8f')
 all_metrics=[];all_preds=[]
 for target in ['e','latent_coil']:
  m,p=compare(d,target);all_metrics.append(m);all_preds.append(p)
 metrics=pd.concat(all_metrics,ignore_index=True);preds=pd.concat(all_preds,ignore_index=True)
 metrics.to_csv(out/'R08_LeaveCityOut_Model_Metrics.csv',index=False,float_format='%.8f')
 preds.to_csv(out/'R08_LeaveCityOut_Predictions.csv',index=False,float_format='%.8f')
 pair=make_pairs(metrics,preds)
 pair.to_csv(out/'R08_Paired_StrongerBaseline_Sensitivity.csv',index=False,float_format='%.8f')
 assert len(metrics)==4*len(FAMILIES)*2 and len(preds)==len(metrics)*365
 assert len(pair)==4*len(PAIRS)*2
 # Repeat original R06 results exactly for A and B to enforce comparability.
 ori=pd.read_csv(out.parent/'legacy'/'R06_LOCO_Paired_Model_Comparisons.csv') if (out.parent/'legacy'/'R06_LOCO_Paired_Model_Comparisons.csv').exists() else None
 if ori is not None:
  for city in CITYMAP.values():
   old=ori[(ori.target=='cooling_kWh')&(ori.heldout_city==city)&(ori.baseline=='A_temp_only')&(ori.candidate=='B_temp_plus_moisture')].iloc[0]
   r=pair[(pair.target=='e')&(pair.heldout_city==city)&(pair.baseline=='R06_simple_temperature')].iloc[0]
   assert abs(old.mae_baseline-r.baseline_MAE)<0.0000001,(city,'baseline',old.mae_baseline,r.baseline_MAE)
   assert abs(old.mae_candidate-r.candidate_MAE)<0.0000001,(city,'candidate',old.mae_candidate,r.candidate_MAE)
 report={'project':'Project82 R08','source_R04_SHA256':sha_file(path),'source_status':'four city R04 outputs; Abu Dhabi excluded',
  'scope':'exploratory 2023 4-city simulator-emulation sensitivity','counts':{'daily_rows':len(d),'folds_cities':4,'model_families':len(FAMILIES),'targets':2,'metrics_rows':len(metrics),'predictions_rows':len(preds),'pair_rows':len(pair)},
  'method':'leave exactly one city out; all scalers and coefficients fitted on other three cities; 365 held-out daily outcomes; standardized Ridge alpha 10; nonnegative clipping; seven-day resampling of FIXED forecasts only',
  'limits':['no new EnergyPlus runs','not independent measurement validation','shared Dubai design days and >600h annual occupied cooling setpoint unmet','no site-year repetitions for simulator','noncausal correlated meteorological inputs','model family choice post R06 means exploratory, not independent confirmatory','within-year resampling not across independent cities'],
  'publication_ready':False,'R05_warmup_gate':'HOLD'}
 (out/'R08_Protocol_And_Gate_Status.json').write_text(json.dumps(report,indent=2))
 # diagnostic summary for console
 interesting=pair[(pair.target=='e') & pair.baseline.isin(['R06_simple_temperature','Seasonal_temperature_solar','Flexible_CDH_temperature_solar','Rich_temperature_solar','Rich_temperature_solar_more_temperature']) & pair.added_or_comparator.str.contains('moisture')]
 print(interesting[['heldout_city','baseline','added_or_comparator','MAE_improvement_pct']].to_string(index=False))
 print('DONE',report['counts'])

if __name__=='__main__':main()
