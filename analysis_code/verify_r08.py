#!/usr/bin/env python3
"""Independent R08 claims/no-leakage verifier for frozen R04 and deterministic R08 outputs.
Re-execute r08_stronger_baselines.py separately before running this verifier.
"""
import argparse, csv, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

P=argparse.ArgumentParser();P.add_argument('--results',required=True);P.add_argument('--source',required=True);P.add_argument('--prior',required=True);a=P.parse_args()
out=Path(a.results); archive=Path(a.source); prior=Path(a.prior)
checks=[]
def check(label,passed,detail=''):
 checks.append({'name':label,'pass':bool(passed),'detail':str(detail)})
 if not passed:raise AssertionError(f'{label}: {detail}')

def shaf(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

check('R04 exact source SHA',shaf(archive)=='3f9611957b6652dee5f314d30f43888daed4346be80b3b4e639ec85076af149f')
with zipfile.ZipFile(archive) as z:
 check('R04 source CRC',z.testzip() is None)
 st=json.loads(z.read('RUN_STATUS.json'))
 check('R04 simulation HOLD',st['simulation_gate']=='HOLD')
 check('four site diagnostic-only PASS',sum(x['status']=='PASS' for x in st['sites'])==4)
 check('Abu Dhabi severe=8',next(x for x in st['sites'] if x['city']=='Abu Dhabi')['severe']==8)
for f in ['R08_Daily_Derived_From_Frozen_R04.csv','R08_LeaveCityOut_Model_Metrics.csv','R08_LeaveCityOut_Predictions.csv','R08_Paired_StrongerBaseline_Sensitivity.csv','R08_Protocol_And_Gate_Status.json']:
 check('R08 output exists '+f,(out/f).is_file())
d=pd.read_csv(out/'R08_Daily_Derived_From_Frozen_R04.csv');m=pd.read_csv(out/'R08_LeaveCityOut_Model_Metrics.csv');pred=pd.read_csv(out/'R08_LeaveCityOut_Predictions.csv');paired=pd.read_csv(out/'R08_Paired_StrongerBaseline_Sensitivity.csv')
check('R08 daily 1460',len(d)==1460)
check('R08 exactly four cities',d.city.nunique()==4)
check('R08 365 days/site',d.groupby('city').size().eq(365).all())
check('R08 metrics 112',len(m)==112)
check('R08 predictions 40880',len(pred)==40880)
check('R08 comparisons 72',len(paired)==72)
check('R08 two simulation targets',set(m.target)=={'e','latent_coil'})
check('R08 none Abu Dhabi',not d.city.str.contains('Abu').any())
check('R08 raw climate dew conflict count=76',d.dewpoint_above_db_count.sum()==76)
check('R08 predictions 365 per fold',pred.groupby(['target','model','heldout_city']).size().eq(365).all())
check('R08 training rows 1095',m.train_rows.eq(1095).all())
check('R08 test rows 365',m.test_rows.eq(365).all())
check('R08 exactly 3 training cities',m.fold_train_cities.eq(3).all())
check('R08 predictions finite and nonnegative',np.isfinite(pred.pred).all() and pred.pred.ge(0).all())
check('R08 MAE recalculated from prediction CSV', all(np.isclose(g.abs_error.mean(),m[(m.target==tg)&(m.model==md)&(m.heldout_city==ct)].iloc[0].MAE_kWh_d,atol=5e-7) for (tg,md,ct),g in pred.groupby(['target','model','heldout_city'])))
check('R08 paired improvements identity',np.allclose(paired.MAE_improvement_pct,100*(paired.baseline_MAE-paired.candidate_MAE)/paired.baseline_MAE,atol=1e-6))
check('R08 physical no exact field measurement claim',json.loads((out/'R08_Protocol_And_Gate_Status.json').read_text())['publication_ready'] is False)
# Quantified check of original R06 day results and true LOCO folds
legacy=pd.read_csv(prior/'R06_FourCity_2023_Daily_Climate_and_Simulated_Energy.csv')
check('R06 daily 1460',len(legacy)==1460)
for legacy_key,new_key in [('cooling_kWh','e'),('coil_latent','latent_coil'),('latent_potential','latent_potential'),('CDH24','CDH24'),('GHI_Whm2','GHI')]:
 x=legacy[['city','date',legacy_key]].merge(d[['city','date',new_key]],on=['city','date'],validate='one_to_one')
 check('R06 & R08 input alignment '+legacy_key,len(x)==1460 and np.allclose(x[legacy_key+'_x'] if legacy_key==new_key else x[legacy_key],x[new_key+'_y'] if legacy_key==new_key else x[new_key],atol=0.000001,rtol=0))
base=pd.read_csv(prior/'R06_LOCO_Paired_Model_Comparisons.csv')
for c in ['Riyadh','Doha','Muscat','Kuwait City']:
 old=base[(base.target=='cooling_kWh')&(base.heldout_city==c)&(base.baseline=='A_temp_only')].iloc[0]
 new=paired[(paired.target=='e')&(paired.heldout_city==c)&(paired.baseline=='R06_simple_temperature')].iloc[0]
 check('R06 replay exact '+c,abs(old.mae_baseline-new.baseline_MAE)<1e-7 and abs(old.mae_candidate-new.candidate_MAE)<1e-7)
for target in ['e','latent_coil']:
 for c in ['Riyadh','Doha','Muscat','Kuwait City']:
  tr=d[d.city!=c];te=d[d.city==c]
  feat=['T','Tmax','Tmin','CDH20','CDH24','CDH28','Tdev2','weekday','GHI','sin1','cos1','sin2','cos2']
  pipe=make_pipeline(StandardScaler(),Ridge(alpha=10))
  pipe.fit(tr[feat].values,tr[target].values)
  actual=np.maximum(pipe.predict(te[feat].values),0)
  candidate=pred[(pred.target==target)&(pred.heldout_city==c)&(pred.model=='Rich_temperature_solar')].sort_values('date')
  check('Independent exact fold replay '+target+' '+c, np.allclose(actual,candidate.pred.values,atol=5e-6,rtol=0))
report={'decision':'PASS_INDEPENDENT_VERIFIER_FOR_R08_NUMERICS_ONLY','n_checks':len(checks),'passed':sum(x['pass'] for x in checks),'checks':checks,
 'scope':'source / daily features / predictive models; no new EnergyPlus simulations or independent field validation',
 'explicit_HOLD':['R05 Abu Dhabi solver convergence','cooling-service equivalence','equipment sizing local design days','multi-year five-city EnergyPlus runs','external metered validation','causal humidity effect','NPV','original scientific novelty and final submission']}
(out/'R08_Independent_Test_Report.json').write_text(json.dumps(report,indent=2))
print('PASS',len(checks),'checks. Partial scientific stage only.')
