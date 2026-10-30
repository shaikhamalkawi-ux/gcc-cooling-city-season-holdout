#!/usr/bin/env python3
"""Deterministic presentation from R08 source-gated output tables. Figures carry explicit scope labels."""
from pathlib import Path
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(os.environ.get('PROJECT82_R08_ROOT',str(Path(__file__).resolve().parent.parent)))
RES=ROOT/'results'; FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
pairs=pd.read_csv(RES/'R08_Paired_StrongerBaseline_Sensitivity.csv')
metric=pd.read_csv(RES/'R08_LeaveCityOut_Model_Metrics.csv')
clim=pd.read_csv(ROOT/'R07_support'/'R07_FiveCity_2015_2024_Climate_Proxy_Results.csv')
citysim=pd.read_csv(ROOT/'R07_support'/'R07_FourCity_2023_EnergyPlus_R04_Diagnostic.csv')
order=['Doha','Riyadh','Kuwait City','Muscat']
sel=lambda base:pairs[(pairs.target=='e')&(pairs.baseline==base)&pairs.added_or_comparator.str.contains('moisture')].set_index('heldout_city').loc[order]
ref=sel('R06_simple_temperature');rich=sel('Rich_temperature_solar')
x=np.arange(4);w=0.35
fig,ax=plt.subplots(figsize=(9.0,4.9))
bars1=ax.bar(x-w/2,ref.MAE_improvement_pct,w,label='Simple CDH24 + weekday')
bars2=ax.bar(x+w/2,rich.MAE_improvement_pct,w,label='Rich temperature, solar, season')
for bars in [bars1,bars2]:
 for bar in bars:
  ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.65,f'{bar.get_height():.1f}%',ha='center',va='bottom',fontsize=9)
ax.axhline(0,linewidth=.8)
ax.set_xticks(x,order);ax.set_ylim(0,42);ax.set_ylabel('Reduction in held-out daily MAE (%)')
ax.set_title('Adding moisture to matched predictors: 2023 R04 simulator only')
ax.legend(loc='upper right',fontsize=9);fig.text(.5,.01,'Not measured-building savings; one year and four city profiles',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.03,1,1));fig.savefig(FIG/'R08_Fig1_Paired_LOCO_Improvements.png',dpi=220);fig.savefig(FIG/'R08_Fig1_Paired_LOCO_Improvements.pdf');plt.close(fig)

cityplot=citysim.set_index('city').loc[['Doha','Muscat','Kuwait City','Riyadh']]
fig,ax=plt.subplots(figsize=(9.0,4.6))
bars=ax.bar(cityplot.index,cityplot.cooling_MWh)
for b,sh in zip(bars,cityplot.coil_latent_share_pct):
 ax.text(b.get_x()+b.get_width()/2,b.get_height()+3,f'{b.get_height():.1f} MWh\nlatent {sh:.1f}%',ha='center',fontsize=8)
ax.set_ylim(0,280);ax.set_ylabel('Simulated annual cooling electricity (MWh, 2023)')
ax.set_title('Four admitted R04 technical outputs (unmet cooling service)')
fig.text(.5,.01,'Latent fractions are coil THERMAL loads, not electricity shares; Abu Dhabi excluded',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.05,1,1));fig.savefig(FIG/'R08_Fig2_EPlus_FourCity_Energy.png',dpi=220);fig.savefig(FIG/'R08_Fig2_EPlus_FourCity_Energy.pdf');plt.close(fig)

c=clim.set_index('city').loc[['Abu Dhabi','Riyadh','Doha','Muscat','Kuwait City']];x=np.arange(5);w=.36
fig,ax=plt.subplots(figsize=(9.2,4.7))
ax.bar(x-w/2,c.mean_sensible_only_kWh_m2,w,label='Temperature-only sensible proxy')
ax.bar(x+w/2,c.mean_component_sum_kWh_m2,w,label='Sum of positive sensible + latent proxies')
ax.set_xticks(x,c.index,rotation=12);ax.set_ylim(0,38);ax.set_ylabel('Reference outdoor-air forcing (kWh/m² per year)')
ax.set_title('Five-city 2015–2024 climate proxies — NOT building cooling kWh')
ax.legend(loc='upper right',fontsize=8)
fig.text(.5,.01,'The combined quantity is larger by construction; it cannot validate electricity uplift',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.05,1,1));fig.savefig(FIG/'R08_Fig3_Climate_Reference_Partition.png',dpi=220);fig.savefig(FIG/'R08_Fig3_Climate_Reference_Partition.pdf');plt.close(fig)

samples=[]
for name,label in [('Rich_temperature_solar_plus_moisture','Humidity-ratio excess'),('Rich_temperature_solar_plus_meanRH','Mean RH'),('Rich_temperature_solar_plus_meanTd','Mean dew point'),('Rich_temperature_solar_plus_enthalpy','Moist-air enthalpy')]:
 for city in order:
  j=pairs[(pairs.target=='e')&(pairs.baseline=='Rich_temperature_solar')&(pairs.added_or_comparator==name)&(pairs.heldout_city==city)].iloc[0]
  samples.append([label,city,j.MAE_improvement_pct])
f=pd.DataFrame(samples,columns=['Feature','City','Delta'])
m=f.pivot(index='Feature',columns='City',values='Delta').loc[['Humidity-ratio excess','Mean RH','Mean dew point','Moist-air enthalpy'],order]
fig,ax=plt.subplots(figsize=(9.2,4.8))
im=ax.imshow(m.values,cmap='coolwarm',vmin=-25,vmax=40,aspect='auto')
ax.set_xticks(np.arange(4),m.columns);ax.set_yticks(np.arange(4),m.index)
for i in range(m.shape[0]):
 for j in range(m.shape[1]):ax.text(j,i,f'{m.iloc[i,j]:+.1f}%',ha='center',va='center',fontsize=10)
ax.set_title('Different moisture representations are not equivalent (2023 LOCO)')
fig.colorbar(im,ax=ax,fraction=.035,pad=.02,label='Reduction in MAE vs matched rich-temperature baseline (%)')
fig.tight_layout();fig.savefig(FIG/'R08_Fig4_Alternative_Moisture_Comparison.png',dpi=220);fig.savefig(FIG/'R08_Fig4_Alternative_Moisture_Comparison.pdf');plt.close(fig)
print('Generated',len(list(FIG.glob('*.png'))),'PNG +',len(list(FIG.glob('*.pdf'))),'vector PDFs')
