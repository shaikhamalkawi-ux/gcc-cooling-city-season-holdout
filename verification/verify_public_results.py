#!/usr/bin/env python3
"""Verify released Project82 output-to-result arithmetic without raw NASA/DOE archives.

Standard library only. This check is NOT a source-to-result replay, EnergyPlus validation,
independent weather-year test, causal experiment, or confirmation of HVAC service.
"""
import csv
import gzip
import io
import hashlib
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
CITIES = {'Doha', 'Muscat', 'Kuwait City', 'Riyadh'}


def rows(name):
    path = DATA / name
    if path.exists():
        with path.open(encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))
    if name == 'R10_CityQuarter_Model_Predictions.csv':
        # Lossless gzip packaging of the original 4,694,819-byte archived CSV.
        raw = gzip.decompress((DATA / (name + '.gz')).read_bytes())
        expected = '8d7ddfc200e604e43ab765fe2d2bb6f14d955eaf862a14c45bc478b1e615cf10'
        checked(hashlib.sha256(raw).hexdigest() == expected,
                'restored prediction CSV matches the frozen source SHA-256')
        with io.StringIO(raw.decode('utf-8'), newline='') as f:
            return list(csv.DictReader(f))
    raise FileNotFoundError(path)


def eq(a, b, tol=1e-6, msg=''):
    if abs(float(a) - float(b)) > tol:
        raise AssertionError(f'{msg}: {a} != {b} within {tol}')


def checked(cond, explanation):
    if not cond:
        raise AssertionError(explanation)
    print('PASS:', explanation)


def main():
    data = rows('R10_CityQuarter_Model_Predictions.csv')
    checked(len(data) == 43800, '43,800 frozen predicted model-day records')
    seen_dates = set()
    errors = defaultdict(lambda: [0.0, 0])
    quarter_errors = defaultdict(lambda: [0.0, 0])
    target_per_day = {}
    for r in data:
        city = r['heldout_city']; quarter = int(r['heldout_quarter'])
        alpha = float(r['ridge_alpha']); family = r['baseline_family']; cand = r['candidate']
        d = r['date']; actual = float(r['actual_kWh']); pred = float(r['pred_kWh'])
        error = float(r['abs_error_kWh'])
        checked_year = d.startswith('2023-')
        if city not in CITIES or quarter not in {1,2,3,4} or not checked_year or pred < -1e-9:
            raise AssertionError('Invalid city/quarter/year or negative prediction in released records')
        eq(error, abs(actual-pred), tol=1.2e-8, msg='published daily abs error')
        key = (city, d)
        if key in target_per_day:
            eq(target_per_day[key], actual, tol=1e-8, msg='shared simulated target across models')
        else:
            target_per_day[key] = actual
        grp=(city, family, alpha, cand)
        errors[grp][0] += error; errors[grp][1] += 1
        qgrp=grp+(quarter,)
        quarter_errors[qgrp][0] += error; quarter_errors[qgrp][1] += 1
        seen_dates.add(key)
    checked(len(seen_dates) == 4*365, '1,460 unique city-day simulator targets in one year')
    checked(len(errors) == 4*2*3*5, '120 city/baseline/alpha/candidate prediction series')
    checked(len(quarter_errors) == 4*4*2*3*5, '480 city/quarter/baseline/alpha/candidate folds')
    for key, (err, n) in errors.items():
        checked(n == 365, f'exactly 365 test days for {key}') if False else None
        if n != 365:
            raise AssertionError((key,n))
    checked(True, '365 test days per city-model series')

    def mae(key):
        a,n=errors[key];return a/n

    annual = rows('R10_DoubleBlocked_Annual_PairedResults.csv')
    checked(len(annual)==96, '96 archived annual paired comparisons (all candidates)')
    for r in annual:
        city=r['heldout_city']; fam=r['family']; alpha=float(r['ridge_alpha'])
        b = mae((city,fam,alpha,'Temperature_only'))
        c = mae((city,fam,alpha,r['candidate']))
        eq(b, r['baseline_MAE_kWh_day'],msg='annual baseline MAE')
        eq(c, r['candidate_MAE_kWh_day'],msg='annual candidate MAE')
        if b > 0:
            eq(100*(b-c)/b, r['heldout_city_season_doubleblock_MAE_reduction_pct'],tol=1e-6,msg='annual MAE reduction')
    checked(True, 'All 96 annual metrics are consistent with independently summed daily errors')

    selected = rows('R10_Selected_RichMoisture_Summary.csv')
    checked(len(selected)==4 and {r['heldout_city'] for r in selected}==CITIES, '4 selected reported city results')
    for r in selected:
        city=r['heldout_city'];fam='Rich_T_Solar_Season';alpha=10.0
        b=mae((city,fam,alpha,'Temperature_only'))
        c=mae((city,fam,alpha,'Plus_humidity_ratio'))
        eq(b,r['baseline_MAE_kWh_day'],msg='selected baseline MAE')
        eq(c,r['candidate_MAE_kWh_day'],msg='selected moisture MAE')
        reduction=100*(b-c)/b
        eq(reduction,r['heldout_city_season_doubleblock_MAE_reduction_pct'],tol=1e-6,msg='selected MAE reduction')
        print(f'REPORTED: {city}: {b:.2f} -> {c:.2f} kWh/day; error reduction {reduction:.2f}%')
    checked(True, '4/4 selected city MAE percentages independently recomputed')

    improvements=[]
    for city in sorted(CITIES):
        for q in (1,2,3,4):
            a,an=quarter_errors[(city,'Rich_T_Solar_Season',10.0,'Temperature_only',q)]
            b,bn=quarter_errors[(city,'Rich_T_Solar_Season',10.0,'Plus_humidity_ratio',q)]
            if an!=bn: raise AssertionError('Mismatched paired day counts')
            gain=100*(a-b)/a
            improvements.append((city,q,gain))
    checked(sum(gain>0 for _,_,gain in improvements)==15, '15 of 16 city-quarter cases improve')
    bad=[(city,q,gain) for city,q,gain in improvements if gain<0]
    checked(len(bad)==1 and bad[0][0]=='Riyadh' and bad[0][1]==1, 'the sole negative selected quarter is Riyadh Q1')
    eq(bad[0][2],-5.79,tol=0.02,msg='published rounded Riyadh Q1 -5.79%')
    positive_annual = [r for r in annual if r['candidate']=='Plus_humidity_ratio' and float(r['baseline_MAE_kWh_day'])>float(r['candidate_MAE_kWh_day'])]
    checked(len(positive_annual)==22, '22 of 24 humidity-ratio city/baseline/penalty annual comparisons improve')

    split = rows('R10_CityQuarter_StrictTrainTestCounts.csv')
    checked(len(split)==16,'16 recorded double-block split counts')
    checked(all(819<=int(r['train_rows'])<=825 and 90<=int(r['test_rows'])<=92 for r in split),
        'declared per-fold train/test sizes within 819–825 and 90–92')
    checked(all(r['excluded_all_target_city']=='True' and r['excluded_target_quarter_from_all_training_cities']=='True' for r in split),
        'split ledger flags indicate both exclusions (ledger assertion, not raw-source proof)')

    shift = rows('R09_MonthlyShift_NegativeControl_Full.csv')
    checked(len(shift)==1200,'4 observed R09 results + 4 x 299 shifted-control results')
    shiftby=defaultdict(lambda: {'real': [],'monthwise_shift': []})
    for r in shift:
        shiftby[r['city']][r['type']].append(float(r['improvement_pct']))
    checked(set(shiftby)==CITIES and all(len(s['real'])==1 and len(s['monthwise_shift'])==299 for s in shiftby.values()),
        'each city has one actual and 299 month-preserving shifts')
    checked(all(s['real'][0]>max(s['monthwise_shift']) for s in shiftby.values()),
        'observed daily humidity improves prediction more than all 299 controls in each city')

    clim=rows('R07_FiveCity_2015_2024_Climate_Proxy_Results.csv')
    checked(len(clim)==5,'5-city climate-only reference summary')
    for r in clim:
        eq(float(r['mean_sensible_only_kWh_m2'])+float(r['mean_positive_latent_kWh_m2']),
           float(r['mean_component_sum_kWh_m2']),tol=2e-8,msg='underlying climate component sum')
    checked(True, '5/5 climate proxy component-sum identities match before display rounding')

    print('\nPUBLIC OUTPUT VERIFIER: PASS')
    print('LIMIT: no raw NASA POWER or EnergyPlus archive, independent field evidence or new weather years are verified here.')

if __name__=='__main__':
    main()
