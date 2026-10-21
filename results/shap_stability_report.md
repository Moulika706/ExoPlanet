# ExoVet - SHAP stability across folds (Phase 2)

Model `xgboost`, 5 grouped folds, seed 42. Every fold is held out in turn, the model is
refitted on the remaining folds and the global SHAP importance is
computed on the held-out rows, so the rankings are independent of
each other.

## Summary

- Mean Jaccard overlap of the top-5 sets: **1.0000** (worst pair 1.0000).
- Mean Kendall rank correlation between folds: **0.9491** (worst pair 0.8909).
- Per-fold macro-F1: mean 0.7152, standard deviation 0.0076.

## Per-fold macro-F1 (uncalibrated, no calibration fold held out)

| fold | n test | macro-F1 | log-loss |
|---|---|---|---|
| 0 | 1914 | 0.7281 | 0.5483 |
| 1 | 1911 | 0.7076 | 0.5738 |
| 2 | 1914 | 0.7095 | 0.5645 |
| 3 | 1914 | 0.7193 | 0.5474 |
| 4 | 1911 | 0.7115 | 0.5675 |

## Top-5 features per fold

| fold | #1 | #2 | #3 | #4 | #5 |
|---|---|---|---|---|---|
| 0 | koi_period | koi_depth | koi_prad | koi_duration | koi_kepmag |
| 1 | koi_period | koi_depth | koi_prad | koi_duration | koi_kepmag |
| 2 | koi_period | koi_depth | koi_prad | koi_duration | koi_kepmag |
| 3 | koi_period | koi_depth | koi_prad | koi_duration | koi_kepmag |
| 4 | koi_depth | koi_period | koi_prad | koi_duration | koi_kepmag |

## Consensus ranking

| rank | feature | mean abs SHAP | mean rank | rank spread | folds in top-k |
|---|---|---|---|---|---|
| 1 | koi_period | 0.4761 | 1.20 | 0.40 | 5 |
| 2 | koi_depth | 0.4595 | 1.80 | 0.40 | 5 |
| 3 | koi_prad | 0.4125 | 3.00 | 0.00 | 5 |
| 4 | koi_duration | 0.2639 | 4.00 | 0.00 | 5 |
| 5 | koi_kepmag | 0.2466 | 5.00 | 0.00 | 5 |
| 6 | koi_impact | 0.2058 | 6.20 | 0.40 | 0 |
| 7 | koi_steff | 0.1945 | 6.80 | 0.40 | 0 |
| 8 | koi_srad | 0.1606 | 8.00 | 0.00 | 0 |
| 9 | koi_insol | 0.1164 | 9.40 | 0.49 | 0 |
| 10 | koi_slogg | 0.1121 | 9.60 | 0.49 | 0 |
| 11 | koi_teq | 0.0771 | 11.00 | 0.00 | 0 |

## Reading the table

- A Jaccard overlap near 1.0 means every fold picked the same
  features, so the explanation the user sees is not an artefact of
  one particular split.
- The rank spread is the standard deviation of a feature's rank
  across the folds; a feature with a spread near zero is a
  permanent fixture of the model's reasoning.
- The per-fold macro-F1 shows that the model itself is stable; an
  unstable ranking on a stable model would point at the
  explanation rather than at the data.
