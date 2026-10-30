# Gulf cities: dry-bulb vs humidity-aware cooling-electricity emulation

**Reviewer-facing frozen-result package for an author-review conference manuscript.** This repository is a **single-year, four-city EnergyPlus output-emulation diagnostic**, not a demonstration of real HVAC savings or a validated five-city design tool. The climate-only screen covers five cities for 2015–2024; electricity-emulator comparisons use 2023 simulation outputs for **Doha, Muscat, Kuwait City and Riyadh**. Abu Dhabi is **excluded** from the electricity tests because the archived EnergyPlus run produced eight severe warmup failures. The four retained city cases had **624–776 occupied cooling-setpoint-unmet hours**, and the simulations retain Dubai-derived design-day conditions.

## Quick independent check — no EnergyPlus required

A reader can download this repository and run, using only Python's standard library:

```bash
python3 verification/verify_public_results.py
```

This decompresses and SHA-256-checks the **released 43,800 predicted city/model/day records** against the recorded 2023 simulator targets, recreates the four selected humidity-versus-temperature MAE improvements, counts the 15/16 positive quarters and 22/24 annual comparisons, and checks the month-preserving 299-shift control ordering for all four cities. It uses **only released frozen outputs**; it does *not* regenerate original hourly weather, re-execute EnergyPlus, or independently establish the source archives' integrity.

For full source-to-result replay, the original frozen 2015–2024 NASA POWER and 2023 EnergyPlus source archives must first be obtained through an authorized route and verified against their original recorded SHA-256 hashes. Those source ZIPs are **not redistributed in this repository**. The original analysis programs are retained in [`analysis_code/`](analysis_code/); they require those source archives and the documented Python dependencies. See [`REVIEWER_GUIDE.md`](REVIEWER_GUIDE.md).

## Core 2023 benchmark (selected Rich T / solar / calendar, Ridge alpha = 10)

| Held-out city | Temperature-only MAE (kWh/day) | + Humidity MAE (kWh/day) | MAE reduction |
|---|---:|---:|---:|
| Doha | 119.58 | 80.16 | 32.97% |
| Muscat | 84.17 | 68.23 | 18.94% |
| Kuwait City | 93.02 | 80.47 | 13.49% |
| Riyadh | 81.95 | 70.30 | 14.22% |

Positive values are **prediction-error reductions**, not simulated energy savings and not field evidence. One Riyadh winter quarter worsened by 5.79% in the selected comparison. All reported tests are retrospective and share a single simulated year and one building archetype.

## What is included

- `data/` — frozen daily derived model inputs and prediction/evaluation tables, with the 43,800-row prediction CSV compressed **losslessly** as `data/R10_CityQuarter_Model_Predictions.csv.gz` (not raw weather/building inputs). The verifier restores and hash-checks its exact original CSV content in memory.
- `verification/verify_public_results.py` — independently implemented output arithmetic checks; standard library only.
- `analysis_code/` — original archived analysis scripts; the full source-archive replay path is inaccessible without authorized raw inputs.
- `verification/*Protocol*`, `R10_Independent_QA.json` — original source-gate audit records, kept distinct from the new publicly runnable output check.
- `figures/` — original six vector figure files, unchanged.
- `SHA256SUMS.txt` — digest of the released files (source archives deliberately excluded).

## Boundaries and reproducibility levels

1. **Runnable from this repository:** output-to-reported-result arithmetic, published metric and split-count checks, negative-control comparison.
2. **Requires authorized source ZIPs:** exact source extraction, original analytical feature construction and script replay.
3. **Not established:** empirical metering validation, adequate thermal service, HVAC installed capacity, humidity causality, cost/NPV savings, multiple independent years or buildings.

**Data/software rights:** Public hosting does not grant reuse or redistribution rights. Copyright and third-party permissions for the original DOE/PNNL model, NASA POWER source records, and any derived artifacts must be reviewed by the authors. No broad software/data license is asserted here.

**Paper status:** Author-approved administrative preparations; conference submission or acceptance is not claimed. No funding or conflicts of interest are declared in the manuscript. AI-assisted editorial disclosure is retained.
