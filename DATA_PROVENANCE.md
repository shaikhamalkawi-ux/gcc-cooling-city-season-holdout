# Public release boundary and evidence provenance

The public data files in `data/` are curated unchanged from the controlled Project 82 R08–R10 analysis package. `R10_Daily_SourceDerived_2023.csv` contains **derived daily predictors and EnergyPlus target aggregates**, not new field measurements. `R10_CityQuarter_Model_Predictions.csv.gz` is a losslessly compressed copy of the original prediction CSV (uncompressed SHA-256 `8d7ddfc200e604e43ab765fe2d2bb6f14d955eaf862a14c45bc478b1e615cf10`) and contains archived simulator emulation predictions, including 16 held-out city/quarter folds and multiple model families; they are **not independent measured city-year experiments**.

Doha, Muscat, Kuwait City and Riyadh are the only technically readable EnergyPlus outputs from the 2023 archival model. Abu Dhabi was excluded after eight severe warmup errors. The four admitted runs nevertheless exhibited substantial unmet cooling-setpoint hours and inherited Dubai sizing design days; their outputs cannot establish service-equivalent energy or equipment savings.

The original sources (`Project82_R04_Frozen_2023_Source.zip` and `Project82_R02_Frozen_Climate_Reference_Replay.zip`) are deliberately absent pending verification of source-redistribution rights and corresponding-author approval. The checksums in the release identify these source objects but are *not substitutes* for including them or for independent execution.

This release is intended to support external arithmetic verification of the manuscript's reported simulator-emulator metrics. Its results are *descriptive and bounded*; the release does not claim new experimental research, a validated physical HVAC control intervention or cross-year extrapolation.
