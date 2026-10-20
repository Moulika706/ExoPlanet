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
make stability  # Phase 2: SHAP top-feature stability across folds
make phase2     # Phase 2: everything above, in one command
make toi        # Phase 5: fetch the TESS Objects of Interest table
make cross      # Phase 5: Kepler -> TESS zero-shot evaluation
make web        # Phase 6: build the front end
make serve      # Phase 6: build the front end and serve it from the API
make figures    # Phase 7: publication figures from the results bundle
make test       # run pytest suite
make ruff       # lint
```

`make api` (Phase 6) starts the API on its own, without the front end.

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

### Conformal modes

Both modes are fitted on the same calibration fold and scored on the same
test fold.

| mode | coverage | cov. CONFIRMED | cov. CANDIDATE | cov. FALSE POSITIVE | mean set | singleton |
|---|---|---|---|---|---|---|
| marginal | 0.9075 | 0.9327 | 0.7652 | 0.9514 | 1.41 | 63.1% |
| mondrian | 0.9039 | 0.9109 | 0.9066 | 0.8988 | 1.49 | 53.3% |

The marginal mode buys a smaller average set, but it under-covers the rare
CANDIDATE class (0.765 against a 0.90 target). The Mondrian mode gives up
some singleton rate and covers every class to at least 0.90, which is the
guarantee the product should claim.

### Leakage ablation

Same test rows throughout for `grouped` and `sibling_leak`; only the
training data changes. The `random` rows use an ordinary stratified split,
so their test fold differs as well.

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

| model | clean + grouped | clean + random split | delta |
|---|---|---|---|
| logistic regression | 0.5272 | 0.4986 | -0.0286 |
| random forest | 0.6981 | 0.6504 | -0.0477 |
| xgboost | 0.7290 | 0.6857 | -0.0433 |

The random split is not better, it is slightly worse. The grouped split is
the stricter protocol and it does not cost accuracy, so there is no
argument for weakening it.

### SHAP stability across folds

The model is refitted on each grouped fold in turn and the global SHAP
importance is computed on the held-out rows, so the rankings are
independent of each other.

| quantity | value |
|---|---|
| mean Jaccard overlap of the top-5 sets | 1.0000 (worst pair 1.0000) |
| mean Kendall rank correlation between folds | 0.9491 (worst pair 0.8909) |
| per-fold macro-F1 | 0.7076 - 0.7281 (std 0.0076) |

Every fold picks the same five features, in the same order except that one
fold swaps `koi_period` and `koi_depth`. The explanation the user sees is
a property of the model, not of one split.

### Cross-mission: Kepler to TESS

The model is trained on Kepler and applied to the TESS Objects of Interest
table without any change. Nine of the eleven core features exist in both
missions with the same definition and unit; `koi_impact` has no TOI
counterpart and `koi_kepmag` would have to be replaced by a TESS
magnitude, a different photometric system, so both are left out.

| setting | macro-F1 | PR-AUC | conformal cov. |
|---|---|---|---|
| Kepler test fold (in domain) | 0.6749 | 0.7462 | 0.8966 |
| TESS, zero shot | 0.2930 | 0.4407 | 0.4740 |
| TESS, recalibrated (label shift) | 0.4144 | 0.4317 | 0.8853 |
| TESS, trained on TESS (in domain) | 0.4341 | 0.5757 | 0.9085 |

**The model does not transfer.** Three things are going on, and the report
separates them:

- The class prior moves from CANDIDATE 21% to 65%, so a model fitted under
  the Kepler prior almost never predicts CANDIDATE on TESS (recall 0.06).
  A label-shift correction recovers most of that and lifts macro-F1 to
  0.4144, close to the in-domain ceiling.
- The ranking collapses on the other two classes. Per-class PR-AUC on
  FALSE POSITIVE goes from 0.92 on Kepler to 0.33 zero-shot, but a model
  *trained on TESS* only reaches 0.51, because Kepler's FALSE POSITIVE
  class is the automatic pipeline's rejection of a large population of
  obviously non-planet signals, while the TESS class is a human-vetted
  rejection of subtle ones. The two classes are not the same class.
- The conformal coverage collapses with it, to 0.474. The split-conformal
  guarantee is an exchangeability statement, and TESS objects are not a
  fresh sample of Kepler objects.

The conclusion is that the two tables answer different questions, and a
product that claims to work on TESS has to be trained on TESS. This is the
kind of result that a copy of someone else's benchmark would have hidden.

## The demo

```bash
make serve          # builds the front end and serves it on port 8000
```

Then open <http://127.0.0.1:8000/app/>. The API is documented at
<http://127.0.0.1:8000/docs>.

The page has the form, the probability bar, the SHAP chart, the review
badge and the narrated report:

- **Form.** The eleven physical features, with units and a plain-English
  description. A blank field is imputed with the training median, so a
  partial form still produces a verdict. Three real objects from the
  archive are one click away.
- **Verdict and review badge.** The class, the calibrated probability and
  a badge that says *Confident* or *Needs manual review*. The badge is not
  a heuristic: it is set when the conformal prediction set holds more than
  one class, which is the model's own statement that it cannot decide.
- **Probability bar.** The three class probabilities side by side.
- **SHAP chart.** The top features as a diverging bar chart, so it is
  visible which evidence pushed the assessment and which counted against
  it. The report below it mentions only those features, and the API
  refuses to answer if the generated text mentions any other.
- **Correction.** A user who disagrees can send the correction back; it
  is logged with the measurements so the model can be checked against what
  the experts said.

### API

| endpoint | what it does |
|---|---|
| `GET /health` | is the service ready? |
| `GET /model` | what model is loaded, and what is it good for? |
| `GET /features` | the features the form should offer |
| `POST /predict` | vet an object |
| `POST /explain` | the same, with the explanation fields repeated |
| `POST /feedback` | log a correction of a previous verdict |
| `GET /feedback` | the correction log and its summary |

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"values": {"koi_period": 9.488, "koi_duration": 2.9575,
                  "koi_depth": 615.8, "koi_prad": 2.26}}'
```

```json
{
  "predicted_class": "CONFIRMED",
  "probability": 0.866,
  "probabilities": {"CONFIRMED": 0.866, "CANDIDATE": 0.099, "FALSE POSITIVE": 0.035},
  "prediction_set": ["CONFIRMED"],
  "needs_review": false,
  "report": "The model assesses this object as CONFIRMED with a calibrated probability of 87%...",
  "top_features": [{"feature": "koi_kepmag", "shap_value": 0.416, "...": "..."}]
}
```

Full reports: `results/train_report.md`, `results/ablation_report.md`,
`results/shap_stability_report.md`, `results/cross_mission_report.md`.

## Figures

`make figures` builds the publication figures from the bundles in
`results/`, so they can be regenerated without refitting anything. Each
is written to `paper/figures/` as a PNG and a PDF. The captions are in
`paper/FIGURE_CAPTIONS.md`.

| figure | what it shows |
|---|---|
| `reliability` | the calibration curve before and after isotonic calibration |
| `conformal_coverage` | per-class coverage of the two conformal modes against the 0.90 target |
| `leakage_ablation` | the feature leak, the random split and the sibling leak, separated |
| `shap_importance` | the global SHAP importance of the headline model |
| `shap_stability` | the feature ranking in every fold, and the overlap between folds |
| `cross_mission` | the Kepler to TESS drop, and where the ranking collapses |

## Layout

```
src/exovet/
  data.py         loading, label cleaning, feature sets, grouped splits
  splits.py       stratified group k-fold, three-way train/calibration/test
  features.py     imputation, scaling and one-hot encoding
  models.py       logistic regression, random forest, xgboost
  calibration.py  isotonic, confidence-isotonic, vector scaling, temperature, prior
  conformal.py    split conformal prediction sets
  eval.py         macro-F1, PR-AUC, recall@99%, log-loss, Brier, ECE
  explain.py      SHAP global and per-prediction
  stability.py    cross-fold stability of the SHAP feature ranking
  crossmission.py  Kepler/TESS feature and label harmonisation
  report.py       plain-English report from the top-k SHAP features
  pipeline.py     model + calibrator + conformal + explanation, end to end
api/
  main.py         FastAPI: /health /model /features /predict /explain /feedback
  schemas.py      request and response models
  store.py        the loaded pipeline and the feedback log
web/              React front end (Vite)
scripts/
  fetch_data.py   archive download with checksum and retrieval record
  fetch_toi.py    TESS TOI table download (Phase 5)
  train.py        Phase 2 training and evaluation
  ablate_leakage.py  Phase 2 leakage ablation
  shap_stability.py  Phase 2 cross-fold SHAP stability
  cross_mission.py   Phase 5 Kepler to TESS evaluation
```

## Limitations

- KOI labels come from the NASA Exoplanet Archive cumulative table; live
  values may differ from DR25 paper numbers (see docs/DATA_NOTES.md).
- Grouped splits use `kepid` (host-star id); multi-planet systems share a
  host, so sibling KOIs never straddle train/test.
- The model is trained on Kepler only. The TESS evaluation (Phase 5)
  shows that a Kepler-trained model does not transfer: macro-F1 falls from
  0.6749 to 0.2930 zero-shot, and even a model trained on TESS only
  reaches 0.4341, because the two tables define their FALSE POSITIVE
  classes differently.
- The conformal coverage guarantee holds within a mission. Across
  missions it is an observation, not a guarantee: the empirical coverage
  on TESS falls to 0.474 zero-shot.
- The TOI table's `tfopwg_disp` is mapped onto the three canonical labels
  (PC/APC to CANDIDATE, CP/KP to CONFIRMED, FP/FA to FALSE POSITIVE).
  The mapping is stated in the report because the classes are close but
  not identical.
- `koi_pdisposition` and the other vetting columns are excluded from the
  main model on purpose; a model that uses them would score much better
  and would be meaningless.
- The conformal coverage guarantee is marginal by default. The Mondrian
  mode gives class-conditional coverage, which is the stronger claim, at
  the cost of a larger average prediction set.
- The SHAP ranking is computed with `TreeExplainer` in `tree_path` mode.
  The stricter `interventional` mode is implemented but is roughly eighty
  times slower and was not needed for the stability check.

## Paper support

`scripts/make_figures.py` (Phase 7) plots the numbers in
`results/reliability.csv` and `results/shap_fold_importance.csv`.
`docs/METHODS.md` describes the method in the form the paper needs.
