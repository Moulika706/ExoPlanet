# ExoVet

Explainable, uncertainty-aware exoplanet candidate vetting on NASA Kepler
KOI data. Produces a class (CONFIRMED / CANDIDATE / FALSE POSITIVE),
calibrated probabilities, split-conformal prediction sets (90% target
coverage), and SHAP-based plain-English reports.

## Setup

```bash
pip install -r requirements.txt
pip install -e .   # optional; scripts add the repo root to sys.path themselves
```

## Commands

```bash
make data       # Phase 1: fetch KOI cumulative table (CSV + SHA256 + retrieval date)
make train      # Phase 2: train, calibrate, conformalise, evaluate
make ablate     # Phase 2: leakage ablation (feature sets and split strategies)
make test       # run pytest suite
make ruff       # lint
```

`make cross` (Phase 5) and `make api` (Phase 6) are stubbed until those
phases are implemented.

## Results summary

Seed 42, grouped split by host star (`kepid`): 5,739 train / 1,911
calibration / 1,914 test. Metrics are on the untouched test fold, after
isotonic calibration fitted on the held-out calibration fold.

| model | macro-F1 | PR-AUC | log-loss | Brier | ECE | conformal cov. | mean set | singleton |
|---|---|---|---|---|---|---|---|---|
| logistic regression | 0.5272 | 0.6088 | 0.7797 | 0.4533 | 0.0234 | 0.9091 | 1.93 | 35.8% |
| random forest | 0.6981 | 0.7666 | 0.5796 | 0.3360 | 0.0189 | 0.9013 | 1.44 | 61.3% |
| **xgboost** | **0.7290** | **0.7794** | **0.5598** | **0.3178** | 0.0239 | 0.9075 | 1.41 | 63.1% |

Per-class recall of the xgboost model: CONFIRMED 0.862, CANDIDATE 0.482,
FALSE POSITIVE 0.844. CANDIDATE is the hard class, as expected - a
candidate is a signal that has not yet been followed up.

Conformal coverage is above the 0.90 target in every case; a set with more
than one class triggers the "needs manual review" abstention.

### Leakage ablation

Same test rows throughout; only the training data changes.

| model | set A (clean) + grouped | set B (leaky) + grouped | feature-leak delta |
|---|---|---|---|
| logistic regression | 0.5272 | 0.8352 | +0.3080 |
| random forest | 0.6981 | 0.8848 | +0.1867 |
| xgboost | 0.7290 | 0.8921 | +0.1631 |

The vetting-output columns are worth 0.16 to 0.31 macro-F1, which is the
size of the leak. `koi_pdisposition` alone decides the class for 99.8% of
the rows it calls FALSE POSITIVE - it is Kepler's own answer, not a
measurement.

The sibling leak (a host star's other KOI appearing in training) moves
macro-F1 by at most 0.01 on this table, because only 11.4% of host stars
carry more than one KOI. It is measured rather than assumed, and the
grouped split stays the default because it costs nothing.

Full reports: `results/train_report.md`, `results/ablation_report.md`.

## Layout

```
src/exovet/
  data.py         loading, label cleaning, feature sets, grouped splits
  splits.py       stratified group k-fold, three-way train/calibration/test
  features.py     imputation, scaling and one-hot encoding
  models.py       logistic regression, random forest, xgboost
  calibration.py  isotonic, confidence-isotonic, vector scaling, temperature
  conformal.py    split conformal prediction sets
  eval.py         macro-F1, PR-AUC, recall@99%, log-loss, Brier, ECE
  explain.py      SHAP global and per-prediction
  report.py       plain-English report from the top-k SHAP features
  pipeline.py     model + calibrator + conformal + explanation, end to end
scripts/
  fetch_data.py   archive download with checksum and retrieval record
  train.py        Phase 2 training and evaluation
  ablate_leakage.py  Phase 2 leakage ablation
```

## Limitations

- KOI labels come from the NASA Exoplanet Archive cumulative table; live
  values may differ from DR25 paper numbers (see docs/DATA_NOTES.md).
- Grouped splits use `kepid` (host-star id); multi-planet systems share a
  host, so sibling KOIs never straddle train/test.
- The model is trained on Kepler only. It has not yet been evaluated on
  TESS objects (Phase 5), so the cross-mission claim is untested.
- `koi_pdisposition` and the other vetting columns are excluded from the
  main model on purpose; a model that uses them would score much better
  and would be meaningless.

## Paper support

`scripts/make_figures.py` and `docs/METHODS.md` (Phase 7).
