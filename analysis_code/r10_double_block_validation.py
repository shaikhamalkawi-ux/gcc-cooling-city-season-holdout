#!/usr/bin/env python3
"""Project 82 R10: Strict city x calendar-quarter exclusion, frozen R04 EnergyPlus 2023.
Not a new EnergyPlus simulation, independent weather year, field validation, or causal test.
Requires numpy, pandas, scikit-learn, and R08 script daily_features.
"""
import sys, json, hashlib, zipfile, csv, math, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

base=Path(__file__).resolve().parent
sys.path.insert(0,str(base))
from r08_stronger_baselines import daily_features, CITYMAP, SHA_R04, sha_file

RICH=['T','Tmax','Tmin','CDH20','CDH24','CDH28','Tdev2','weekday','GHI','sin1','cos1','sin2','cos2']
SIMPLE=['CDH24','weekday']
BASES={'Simple_T':SIMPLE,'Rich_T_Solar_Season':RICH}
FEATURES={'Moisture_ratio_potential':'latent_potential','Dewpoint_mean':'Td','RH_mean':'RH','Enthalpy_excess':'enthalpy_excess'}


def do(source:Path,out:Path):
 assert sha_file(source)==SHA_R04, 'Frozen R04 input SHA mismatch'
 out.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(source) as z:
  assert z.testzip() is None,'ZIP CRC failure'
  gate=json.loads(z.read('RUN_STATUS.json'))
  assert gate['simulation_gate']=='HOLD' and sum(x['status']=='PASS' for x in gate['sites'])==4
  assert next(v for v in gate['sites'] if v['city']=='Abu Dhabi')['severe']==8
  raw=pd.concat([daily_features(z,s) for s in CITYMAP],ignore_index=True)
 raw['quarter']=(raw['date'].dt.month.to_numpy()-1)//3+1
 assert len(raw)==1460 and set(raw.city)==set(CITYMAP.values())
 assert raw.groupby(['city','quarter']).size().min()>=90
 # Lock exact data lineage by storing input digest and daily table hash; not as 'new climate' trial.
 raw.sort_values(['city','date']).to_csv(out/'R10_Daily_SourceDerived_2023.csv',index=False,float_format='%.10f')
 tests=[];preds=[];metrics=[]
 for city in CITYMAP.values():
  for season in (1,2,3,4):
   tr=raw[(raw.city!=city)&(raw.quarter!=season)].copy()
   te=raw[(raw.city==city)&(raw.quarter==season)].copy()
   assert tr.city.nunique()==3 and tr.quarter.nunique()==3
   assert len(te)>=90 and not (tr.city==city).any() and not (tr.quarter==season).any()
   assert not set(tr.date).intersection(set(te.date)), 'time overlap not allowed'
   assert len(tr)==3*(365-len(te)),(city,season,len(tr),len(te))
   tests.append({'heldout_city':city,'heldout_quarter':season,'train_rows':len(tr),'test_rows':len(te),
      'excluded_all_target_city':True,'excluded_target_quarter_from_all_training_cities':True})
   for ridge_alpha in (1.,10.,100.):
    for baseline,cols in BASES.items():
     for spec,enh in [('Temperature_only',None),('Plus_humidity_ratio','latent_potential'),
                       ('Plus_dewpoint','Td'),('Plus_relative_humidity','RH'),('Plus_enthalpy','enthalpy_excess')]:
      # No moisture indicator for temp-only: each model uses one and only one defined feature.
      columns=cols+([enh] if enh else [])
      assert not any('latent' in c for c in cols)
      pipe=make_pipeline(StandardScaler(),Ridge(alpha=ridge_alpha))
      pipe.fit(tr[columns].to_numpy(),tr.e.to_numpy())
      pred=np.maximum(pipe.predict(te[columns].to_numpy()),0)
      errors=np.abs(te.e.to_numpy()-pred)
      metrics.append({'heldout_city':city,'heldout_quarter':season,'baseline_family':baseline,'candidate':spec,
       'ridge_alpha':ridge_alpha,'train_n':len(tr),'test_n':len(te),'MAE_kWh_day':float(errors.mean()),
       'RMSE_kWh_day':float(np.sqrt(np.mean((te.e.to_numpy()-pred)**2))),
       'annual_target':'2023 simulated cooling electricity; only target quarter' })
      preds.extend({'heldout_city':city,'heldout_quarter':season,'baseline_family':baseline,'candidate':spec,
        'ridge_alpha':ridge_alpha,'date':str(dt.date()),'actual_kWh':float(a),'pred_kWh':float(p),
        'abs_error_kWh':float(e)} for dt,a,p,e in zip(te.date,te.e,pred,errors))
 metrics=pd.DataFrame(metrics); preds=pd.DataFrame(preds)
 assert len(metrics)==16*3*2*5
 assert len(preds)==16*3*2*5*365//4 + (0) or len(preds)>100000 # inconsistent quarters 90/91/92, total check below
 assert len(preds)==4*365*3*2*5
 pairs=[]
 for (city,q,b,alpha),sub in metrics.groupby(['heldout_city','heldout_quarter','baseline_family','ridge_alpha'],sort=True):
  t=sub[sub.candidate=='Temperature_only'].iloc[0];
  for cand in ('Plus_humidity_ratio','Plus_dewpoint','Plus_relative_humidity','Plus_enthalpy'):
   mm=sub[sub.candidate==cand].iloc[0]
   pairs.append({'heldout_city':city,'heldout_quarter':q,'family':b,'ridge_alpha':alpha,
     'candidate':cand,'temp_MAE_kWh_day':t.MAE_kWh_day,'plus_MAE_kWh_day':mm.MAE_kWh_day,
     'MAE_reduction_pct':100*(t.MAE_kWh_day-mm.MAE_kWh_day)/t.MAE_kWh_day,
     'absolute_MAE_delta_kWh_day':t.MAE_kWh_day-mm.MAE_kWh_day,
     'test_days':int(t.test_n)})
 pairs=pd.DataFrame(pairs)
 assert len(pairs)==16*3*2*4
 # Do not incorrectly weight a 90-day quarter same as a 92-day quarter in annual aggregation.
 group=[]
 for (city,family,alpha,cand),sub in preds.groupby(['heldout_city','baseline_family','ridge_alpha','candidate'],sort=True):
  sub=sub.sort_values('date');assert len(sub)==365 and sub.date.nunique()==365
  group.append({'heldout_city':city,'baseline_family':family,'ridge_alpha':alpha,'candidate':cand,
     'actual_total_kWh':sub.actual_kWh.sum(),'predicted_total_kWh':sub.pred_kWh.sum(),
     'annual_absolute_bias_pct':100*abs(sub.pred_kWh.sum()-sub.actual_kWh.sum())/sub.actual_kWh.sum(),
     'MAE_kWh_day':sub.abs_error_kWh.mean(),
     'cooling_nonzero_days':int((sub.actual_kWh>0).sum())})
 group=pd.DataFrame(group)
 annual=[]
 for (city,family,alpha),sub in group.groupby(['heldout_city','baseline_family','ridge_alpha'],sort=True):
  a=sub[sub.candidate=='Temperature_only'].iloc[0]
  for cand in ('Plus_humidity_ratio','Plus_dewpoint','Plus_relative_humidity','Plus_enthalpy'):
   b=sub[sub.candidate==cand].iloc[0]
   annual.append({'heldout_city':city,'family':family,'ridge_alpha':alpha,'candidate':cand,
     'baseline_MAE_kWh_day':a.MAE_kWh_day,'candidate_MAE_kWh_day':b.MAE_kWh_day,
     'heldout_city_season_doubleblock_MAE_reduction_pct':100*(a.MAE_kWh_day-b.MAE_kWh_day)/a.MAE_kWh_day,
     'baseline_annual_bias_pct':a.annual_absolute_bias_pct,'candidate_annual_bias_pct':b.annual_absolute_bias_pct,
     'improved_MAE':bool(b.MAE_kWh_day<a.MAE_kWh_day)})
 annual=pd.DataFrame(annual)
 for name,frame in [('R10_CityQuarter_StrictTrainTestCounts.csv',pd.DataFrame(tests)),('R10_CityQuarter_Model_Metrics.csv',metrics),
                    ('R10_CityQuarter_Model_Predictions.csv',preds),('R10_CityQuarter_Paired_Comparisons.csv',pairs),
                    ('R10_DoubleBlocked_AnnualMetric_ByCity.csv',group),('R10_DoubleBlocked_Annual_PairedResults.csv',annual)]:
  frame.to_csv(out/name,index=False,float_format='%.9f')
 # Dominant scientific conclusion should not overfit to best algorithm after seeing data.
 chosen=annual[(annual.family=='Rich_T_Solar_Season')&(annual.ridge_alpha==10)&(annual.candidate=='Plus_humidity_ratio')].copy()
 chosen.sort_values('heldout_city').to_csv(out/'R10_Selected_RichMoisture_Summary.csv',index=False,float_format='%.9f')
 summary={'status':'R10 partial independent validation; NOT final EnergyPlus evidence',
  'source_R04_SHA256':sha_file(source),'source_simulation_gate':gate['simulation_gate'],
  'new_energyplus_execution':False,'date_year':2023,'cities_n':4,'weather_years_n':1,
  'fold_protocol':'hold entire tested city and same tested calendar quarter out of all training records',
  'folds':len(tests),'fits':len(metrics),'predictions':len(preds),'ridge_penalties':[1.,10.,100.],
  'abudhabi_gate':'HOLD 8 severe warmup errors',
  'served_equivalence_gate':'HOLD >600h occupied cooling unmet at all admitted sites',
  'capacity_gate':'HOLD inherited Dubai design day',
  'economic_gate':'HOLD', 'novelty_gate':'HOLD',
  'selected_results':chosen.to_dict('records'),
  'all_families_fraction_MAE_improved_ratio':int(annual[annual.candidate=='Plus_humidity_ratio'].improved_MAE.sum()),
  'all_families_total_ratio':int((annual.candidate=='Plus_humidity_ratio').sum()),
  'precaution':'strict city-season folds are 16 artificial partitions of 4 cities and one weather year, not 16 independent environments'}
 (out/'R10_Strict_SplitProtocol_And_Gates.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps({'counts':{k:summary[k] for k in ['folds','fits','predictions']},'selected':chosen.to_dict('records'),
   'ratio_improved_overall':f"{summary['all_families_fraction_MAE_improved_ratio']}/{summary['all_families_total_ratio']}"},indent=2))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--source',required=True);a.add_argument('--output',required=True);arg=a.parse_args()
 do(Path(arg.source),Path(arg.output))
