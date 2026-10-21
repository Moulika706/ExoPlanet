# ExoVet - Phase 2 training results

Seed 42. Grouped split by host star (`kepid`): 5739 train / 1911 calibration / 1914 test.

| model | macro-F1 | PR-AUC | log-loss | Brier | ECE | recall@99% | conformal cov. | mean set |
|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0.5272 | 0.6088 | 0.7797 | 0.4533 | 0.0234 | 0.0548 | 0.9091 | 1.93 |
| random_forest | 0.6981 | 0.7666 | 0.5796 | 0.3360 | 0.0189 | 0.1485 | 0.9013 | 1.44 |
| xgboost | 0.7290 | 0.7794 | 0.5598 | 0.3178 | 0.0239 | 0.1320 | 0.9075 | 1.41 |

Metrics are computed on the untouched test fold after isotonic
calibration on the held-out calibration fold. `recall@99%` is the
macro average of the per-class recall attainable at 99% precision.
`conformal cov.` is the empirical coverage of the split-conformal
prediction sets at alpha = 0.1.

## Per-class recall

| model | CONFIRMED | CANDIDATE | FALSE POSITIVE |
|---|---|---|---|
| logistic_regression | 0.7964 | 0.0909 | 0.8099 |
| random_forest | 0.8436 | 0.4167 | 0.8388 |
| xgboost | 0.8618 | 0.4823 | 0.8440 |

## Conformal prediction

Target coverage 0.90, mode `marginal` by default.

| model | mode | coverage | cov. CONFIRMED | cov. CANDIDATE | cov. FALSE POSITIVE | mean set | singleton |
|---|---|---|---|---|---|---|---|
| xgboost | marginal | 0.9075 | 0.9327 | 0.7652 | 0.9514 | 1.41 | 0.6311 |
| xgboost | mondrian | 0.9039 | 0.9109 | 0.9066 | 0.8988 | 1.49 | 0.5334 |

The marginal mode uses one threshold for all classes and guarantees
marginal coverage. The Mondrian mode uses one threshold per class
and guarantees class-conditional coverage, which is the stronger
statement for the rare CANDIDATE class.

## Reliability

Expected calibration error of the best model: 0.0239 after calibration (was 0.0379 before).
The full reliability table is in `results/reliability.csv`.

## Global SHAP importance

Mean absolute SHAP value over 1914 test rows of the best model.

| rank | feature | mean abs SHAP |
|---|---|---|
| 1 | koi_period | 0.5001 |
| 2 | koi_depth | 0.4506 |
| 3 | koi_prad | 0.4190 |
| 4 | koi_duration | 0.2553 |
| 5 | koi_kepmag | 0.2531 |
| 6 | koi_impact | 0.2417 |
| 7 | koi_steff | 0.1912 |
| 8 | koi_srad | 0.1705 |
| 9 | koi_slogg | 0.1184 |
| 10 | koi_insol | 0.0970 |

How stable this ranking is across folds is measured separately in
`results/shap_stability_report.md`.

## Best model

`xgboost` - calibrated macro-F1 0.7290, artifact `results\model_xgboost.joblib`.

