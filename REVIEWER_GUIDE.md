# Reproduction guide for reviewers

## Mode 1: Verify the publicly available frozen outputs

Python 3.10+ is sufficient. From this repository root:

```bash
python3 verification/verify_public_results.py
```

The script restores and SHA-256 verifies `R10_CityQuarter_Model_Predictions.csv` from its published lossless `.csv.gz` file, then checks the R10 annual summary against independently recomputed daily absolute errors. It also checks the frozen R09 shift comparisons and 2015–2024 five-city reference proxy table. A PASS is **output-level arithmetic consistency**, not a confirmation that EnergyPlus, NASA POWER provenance, physical settings or model selection were independently replicated.

## Mode 2: Full extraction and regression replay (author-controlled sources)

The original source-to-output programs are in `analysis_code`. The immutable underlying R04 and R02 ZIPs are *not* in this public repository. To perform the full historical replay, obtain the authorized archives separately through the corresponding author or an approved supplementary-upload route. Verify these recorded original digests:

- R04 2023 source: `3f9611957b6652dee5f314d30f43888daed4346be80b3b4e639ec85076af149f`
- R02 2015–2024 NASA POWER climate replay: `16ab786960ab7fb195473b4c2d4f156c26a9bc8c6832020dea178021ca130c63`

The frozen source-level program imports `r08_stronger_baselines.py` from its Python path and requires NumPy, pandas, scikit-learn and SciPy. For example, using the paths supplied in the controlled archival `README_REVIEWER.md`, the original R10 entrypoint can be invoked:

```bash
python analysis_code/r10_double_block_validation.py --source /AUTHORIZED/PATH/Project82_R04_Frozen_2023_Source.zip --output new_r10_results
```

The original authors' archival reports state 28/28 internal artifact checks PASS. That is a **previously recorded status**, not a new full-source independent run performed for this GitHub preparation.

## Scope and conduct

- Four technically readable 2023 simulator cities only: Doha, Muscat, Kuwait City, Riyadh.
- The five-city 2015–2024 climate proxy is distinct from whole-building electric output.
- The physical thermal-service and local sizing gates remain unsatisfied, so raw model outputs must not be used to support field electricity or capacity claims.
- A protected source archive is not a public GitHub release. Do not upload raw EPWs, original prototype models, full private logs, credentials or private Google Drive share links.
