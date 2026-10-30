#!/usr/bin/env python3
"""Independent verification of R10 double-block experiment & one reader-facing figure."""
import hashlib,json,zipfile,sys,ast
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1];out=root/'results'
source=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/mnt/data/Project82_R04_WaterLoop_Isolation_20261021T191032Z.zip')
expect='3f9611957b6652dee5f314d30f43888daed4346be80b3b4e639ec85076af149f'
actual=hashlib.sha256(source.read_bytes()).hexdigest()
checks=[]
def t(label,v,details=None):
 checks.append({'test':label,'pass':bool(v),'details':details})
 if not v:raise AssertionError(label+' '+str(details))
t('frozen source SHA',actual==expect)
with zipfile.ZipFile(source) as z:
 t('ZIP CRC',z.testzip() is None)
 st=json.loads(z.read('RUN_STATUS.json'))
 t('Abu Dhabi gate HOLD',next(d for d in st['sites'] if d['city']=='Abu Dhabi')['severe']==8)
 t('four technical PASS only',sum(d['status']=='PASS' for d in st['sites'])==4)
 t('no claim of model admission',st['simulation_gate']=='HOLD')
 t('source EPW 5 cities',len([k for k in z.namelist() if k.startswith('EPW_2023/')])==5)

d=pd.read_csv(out/'R10_Daily_SourceDerived_2023.csv');spl=pd.read_csv(out/'R10_CityQuarter_StrictTrainTestCounts.csv')
met=pd.read_csv(out/'R10_CityQuarter_Model_Metrics.csv'); pred=pd.read_csv(out/'R10_CityQuarter_Model_Predictions.csv')
paired=pd.read_csv(out/'R10_CityQuarter_Paired_Comparisons.csv'); ann=pd.read_csv(out/'R10_DoubleBlocked_Annual_PairedResults.csv')
t('1460 complete source-derived days',len(d)==1460)
t('four city tags',len(d.city.unique())==4)
t('one target year',set(pd.to_datetime(d.date).dt.year)=={2023})
t('unique city-date keys',not d.duplicated(['city','date']).any())
t('16 city/quarter folds',len(spl)==16)
t('only 3 source cities in fold',set(spl['train_rows'].astype(int))=={819,822,825})
t('heldout days quarter range',spl.test_rows.between(90,92).all())
t('all temporal/source isolation flags',spl.excluded_all_target_city.all() and spl.excluded_target_quarter_from_all_training_cities.all())
t('480 model fits',len(met)==480)
t('43800 predictions',len(pred)==43800)
t('no missing numeric',np.isfinite(met.MAE_kWh_day).all() and np.isfinite(pred.pred_kWh).all())
t('16x4 paired season curves',len(paired)==384)
t('96 all-season paired comparisons',len(ann)==96)
t('nonnegative prediction',pred.pred_kWh.min()>=0)
t('exact 365 predictions every city/model/penalty',set(pred.groupby(['heldout_city','baseline_family','ridge_alpha','candidate']).size())=={365})
# Recompute per-fold metrics independently from daily rows
for k,g in pred.groupby(['heldout_city','heldout_quarter','baseline_family','ridge_alpha','candidate'],sort=False):
 target=met[(met.heldout_city==k[0])&(met.heldout_quarter==k[1])&(met.baseline_family==k[2])&(met.ridge_alpha==k[3])&(met.candidate==k[4])]
 assert len(target)==1
 assert abs(np.mean(np.abs(g.actual_kWh-g.pred_kWh))-target.MAE_kWh_day.iloc[0])<1e-7
t('480 independent fold MAE recomputations',True)
# Pairwise independent recalculation
for _,p in paired.iterrows():
 p0=met[(met.heldout_city==p.heldout_city)&(met.heldout_quarter==p.heldout_quarter)&(met.baseline_family==p.family)&(met.ridge_alpha==p.ridge_alpha)]
 m0=p0[p0.candidate=='Temperature_only'].iloc[0].MAE_kWh_day
 m1=p0[p0.candidate==p.candidate].iloc[0].MAE_kWh_day
 assert abs(100*(m0-m1)/m0-p.MAE_reduction_pct)<1e-7
t('384 independently reconstructed pairwise percentages',True)
# Yearwise MAEs weighted by number of days, not naive mean quarterwise percentages
for _,p in ann.iterrows():
 q=paired[(paired.heldout_city==p.heldout_city)&(paired.family==p.family)&(paired.ridge_alpha==p.ridge_alpha)&(paired.candidate==p.candidate)]
 assert len(q)==4
 den=np.average(q.temp_MAE_kWh_day,weights=q.test_days)
 num=np.average(q.plus_MAE_kWh_day,weights=q.test_days)
 assert abs(100*(den-num)/den-p.heldout_city_season_doubleblock_MAE_reduction_pct)<1e-7
t('96 calendar-quarter weighted annual paired effects',True)
selected=ann[(ann.family=='Rich_T_Solar_Season')&(ann.ridge_alpha==10)&(ann.candidate=='Plus_humidity_ratio')].copy().sort_values('heldout_city')
t('selected rich-alpha-10 four-city gains',len(selected)==4 and (selected.heldout_city_season_doubleblock_MAE_reduction_pct>0).all())
qs=paired[(paired.family=='Rich_T_Solar_Season')&(paired.ridge_alpha==10)&(paired.candidate=='Plus_humidity_ratio')]
t('15/16 individual quarter positive, one negative',int((qs.MAE_reduction_pct>0).sum())==15 and int((qs.MAE_reduction_pct<0).sum())==1)
t('Riyadh Q1 negative',float(qs[(qs.heldout_city=='Riyadh')&(qs.heldout_quarter==1)].MAE_reduction_pct.iloc[0])<0)
t('22/24 candidate family/penalty annual gains',int(ann[ann.candidate=='Plus_humidity_ratio'].improved_MAE.sum())==22)

# Figure: per season sign/variability explicitly shown, neutral Matplotlib defaults.
fig,ax=plt.subplots(figsize=(11,5.6))
cols=['Doha','Kuwait City','Muscat','Riyadh']; vals=np.array([[float(qs[(qs.heldout_city==c)&(qs.heldout_quarter==q)].MAE_reduction_pct.iloc[0]) for q in [1,2,3,4]] for c in cols]);
xx=np.arange(4);w=.19
for q in range(4):ax.bar(xx+(q-1.5)*w,vals[:,q],width=w,label=f'Quarter {q+1}')
ax.axhline(0,color='#2e2e2e',linewidth=.9)
ax.set_xticks(xx,cols);ax.set_ylabel('Reduction in held-out MAE after humidity (%)')
ax.set_title('2023 EnergyPlus emulator: strict city × calendar-quarter exclusion',fontsize=13,pad=15)
ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.10),frameon=False)
ax.set_ylim(min(-13,float(vals.min()-6)),max(52,float(vals.max()+7)))
ax.grid(axis='y',alpha=.15)
fig.text(.5,.005,'Source: R04 simulator output; fixed rich Ridge α = 10; no independent weather-year or field validation.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.07,1,1))
fig.savefig(root/'figures'/'R10_Figure6_StrictCityQuarter_Changes.png',dpi=190,bbox_inches='tight')
fig.savefig(root/'figures'/'R10_Figure6_StrictCityQuarter_Changes.pdf',bbox_inches='tight')
plt.close(fig)

qa={'project':'Project82 R10','read_only_source_SHA256':actual,'tests':checks,
  'total_tests':len(checks),'passed':sum(x['pass'] for x in checks),
  'selected_city_results':selected[['heldout_city','baseline_MAE_kWh_day','candidate_MAE_kWh_day','heldout_city_season_doubleblock_MAE_reduction_pct']].to_dict('records'),
  'quarterly_positive':int((qs.MAE_reduction_pct>0).sum()),'quarterly_total':16,
  'blocked_annual_positive':int(ann[ann.candidate=='Plus_humidity_ratio'].improved_MAE.sum()),'blocked_annual_total':24,
  'publication_gate':'HOLD; Abu Dhabi no converged E+, equipment sizing and comfort unresolved',
  'scientific_scope':'retrospective 4 city x 4 season double-block reanalysis, 2023 simulator only'}
(out/'R10_Independent_QA.json').write_text(json.dumps(qa,indent=2))
print(json.dumps({k:qa[k] for k in ['total_tests','passed','quarterly_positive','quarterly_total','blocked_annual_positive','blocked_annual_total']},indent=2))
