# Methods - ExoVet

The method as it is implemented, in the form a paper needs it. Phase 2
covers the data, the model, the calibration, the prediction sets and the
explanations. Section 10 covers the cross-mission transfer (Phase 5).
Phase 7 (figures) extends this document further.

## 1. Data

Source: NASA Exoplanet Archive, Kepler KOI cumulative table, fetched with
`astroquery` (TAP CSV fallback). The raw CSV, its SHA256 checksum and the
retrieval date are kept in `data/raw/`. The pull is 9,564 rows by 108
columns; see `docs/DATA_NOTES.md` for the live schema.

The target is `koi_disposition`, which takes exactly three values in this
pull: CONFIRMED (2,748), CANDIDATE (1,977) and FALSE POSITIVE (4,839).
Rows with any other value are dropped defensively.

Two feature sets are defined:

- **Set A (physical).** `koi_period`, `koi_duration`, `koi_depth`,
  `koi_prad`, `koi_teq`, `koi_impact`, `koi_insol`, `koi_steff`,
  `koi_slogg`, `koi_srad`, `koi_kepmag`. These are measurements of the
  object and its host star, so they exist for any mission.
- **Set B (leaky).** Set A plus `koi_score`, `koi_pdisposition`,
  `koi_fpflag_nt`, `koi_fpflag_ss`, `koi_fpflag_co`, `koi_fpflag_ec`.
  These are the *output* of the Kepler vetting process, which is the
  thing the model is supposed to predict. They are used only in the
  ablation.

`kepid` identifies the host star and is the grouping key for every split.

## 2. Splits

`StratifiedGroupKFold` with five folds, seeded once (`seed: 42` in
`configs/default.yaml`). Stratification keeps the class balance close
across folds; the group constraint keeps every host star entirely inside
one fold, so sibling KOIs of the same star never straddle the split.

The three folds used for training are:

| fold | role | rows |
|---|---|---|
| 2, 3, 4 | train | 5,739 |
| 1 | calibration | 1,911 |
| 0 | test | 1,914 |

The calibration fold is separate from train so that the calibrator never
sees training data, and the test fold is touched by neither the model nor
the calibrator.

## 3. Models

Three models are compared on the same splits:

| model | role |
|---|---|
| logistic regression (standardised inputs) | transparent baseline |
| random forest (400 trees) | bagged trees |
| XGBoost (400 trees, depth 6, lr 0.05) | headline model |

Preprocessing (median imputation, optional standardisation, one-hot
encoding of the single categorical column) sits inside a scikit-learn
`Pipeline`, so it is refit on each training fold and never sees validation
or test rows.

A `LabelEncodingClassifier` wrapper encodes the string labels inside
`fit`, because XGBoost rejects string targets. Its class order is
alphabetical (CANDIDATE, CONFIRMED, FALSE POSITIVE) and is *not* the
canonical order used everywhere else, so the probability columns are
reordered explicitly before any metric is computed. Reading them in the
wrong order is a silent error: it turns a macro-F1 of 0.73 into 0.36.

## 4. Metrics

Accuracy is deliberately absent - a model that labels everything FALSE
POSITIVE scores 50% on this data set and is useless. The reported metrics
are macro-F1, per-class recall, macro one-vs-rest PR-AUC, recall at 99%
precision (the ExoMiner-style operating point), multiclass log loss, the
multiclass Brier score and the expected calibration error over ten
equal-width confidence bins.

The log loss is computed in the canonical class order rather than through
`sklearn.metrics.log_loss`, whose internal `LabelBinarizer` sorts the
labels and would score the wrong column.

## 5. Calibration

Isotonic regression by default, fitted on the held-out calibration fold
and applied to the test fold. The code also provides Platt scaling,
confidence-isotonic, vector scaling and temperature scaling; all of them
preserve the class column order and the arg-max (except where the
calibrator is explicitly allowed to change it).

## 6. Conformal prediction

Split conformal, implemented directly in numpy following Angelopoulos &
Bates (2023). The nonconformity score is the probability deficit of the
true label, `s(x, y) = 1 - p_y(x)`, and the prediction set is

    C(x) = { y : 1 - p_y(x) <= q }

where `q` is the finite-sample corrected `(1 - alpha)` quantile of the
calibration scores, computed with the rank `ceil((n + 1) * (1 - alpha)) / n`
so that the guarantee `P(y in C(x)) >= 1 - alpha` holds for any
exchangeable data. `alpha = 0.10`, so the target coverage is 0.90.

Two modes are implemented:

- **marginal** - one threshold for all classes, guaranteeing marginal
  coverage.
- **mondrian** - one threshold per class, fitted on the calibration points
  of that class, guaranteeing class-conditional coverage.

The model's arg-max is always added to the set, so a set is never empty.
Adding a class can only grow the set, so the guarantee is unaffected.

The mode is not a free choice. On the test fold the marginal mode covers
CANDIDATE at 0.765, well under the 0.90 target, because a single global
threshold is dominated by the two common classes. The mondrian mode covers
every class above 0.90 at the cost of a larger average set (1.49 against
1.41) and a lower singleton rate (53.3% against 63.1%).

## 7. Explanations

SHAP `TreeExplainer` in `tree_path` mode, computed on the transformed
feature matrix because the tree models consume the output of the
preprocessor. The transformed column names are taken from the fitted
transformer, and one-hot dummies are mapped back to their source column.
If SHAP cannot handle a model, a leave-one-out permutation importance is
used as a documented fallback.

The report is a template filled from the top-k (k = 5) SHAP features and
nothing else. A checker re-reads the generated text and raises if any
feature outside the top-k is mentioned, so the "explainable" claim is
enforced by code rather than by good intentions.

### 7.1 Stability across folds

A single SHAP table says what the model used on one split. To show that
the explanation is a property of the model and not of one split, the model
is refitted on each grouped fold in turn and the global SHAP importance is
computed on the fold that was held out. The rankings are then compared:

- the mean Jaccard overlap of the top-5 sets over all pairs of folds,
- the mean Kendall rank correlation between folds,
- the number of folds in which a feature appears in the top-5,
- the standard deviation of a feature's rank across folds.

The measured result is a top-5 overlap of 1.0000 and a mean Kendall tau
of 0.9491 over five folds; only one fold swaps the first two features.

## 8. Leakage ablation

Three comparisons, all on the same untouched test fold unless stated:

1. **Feature leak** - set A against set B on the grouped split. Adding the
   vetting-output columns is worth +0.16 (XGBoost) to +0.31 (logistic
   regression) macro-F1.
2. **Sibling leak** - the grouped split against a split with the same test
   rows but a randomly drawn training pool, so a test host star usually
   has a sibling KOI in training. The difference is at most 0.01 macro-F1,
   because only 11.4% of host stars carry more than one KOI.
3. **Grouped against random** - the grouped split against an ordinary
   stratified split, which is what most of the literature reports. The
   random split is not better here (-0.03 to -0.05 macro-F1), so the
   stricter protocol costs nothing.

The script also reports the crosstab of `koi_pdisposition` against the
target label: Kepler's own FALSE POSITIVE call is a FALSE POSITIVE in the
archive for 99.8% of those rows, which is the clearest statement of how
leaky that column is.

## 9. Reproducibility

One seed (`42`) governs the splits, the models, the calibration and the
SHAP background sample. `requirements.txt` pins every dependency. The
whole of Phase 2 is reproduced by

    make phase2

which writes `results/train_report.md`, `results/ablation_report.md`,
`results/shap_stability_report.md` and the matching JSON bundles.

## 10. Cross-mission transfer

### 10.1 The test set

The TESS Objects of Interest table of the same archive, 8,151 rows, with
the raw CSV, its checksum and the retrieval date in `data/raw/`. The
model is trained on Kepler and applied to these objects with no change:
the same weights, the same calibrator, the same conformal threshold.

### 10.2 Feature harmonisation

Nine of the eleven core Kepler features exist in the TOI table with the
same definition and the same unit:

| Kepler | TOI | unit |
|---|---|---|
| `koi_period` | `pl_orbper` | days |
| `koi_duration` | `pl_trandurh` | hours |
| `koi_depth` | `pl_trandep` | ppm |
| `koi_prad` | `pl_rade` | Earth radii |
| `koi_teq` | `pl_eqt` | K |
| `koi_insol` | `pl_insol` | Earth flux |
| `koi_steff` | `st_teff` | K |
| `koi_slogg` | `st_logg` | log g |
| `koi_srad` | `st_rad` | solar radii |

Two are left out. `koi_impact` has no TOI counterpart at all. `koi_kepmag`
would have to be replaced by `st_tmag`, which is a different photometric
system; substituting it would hand the model a systematically shifted
feature, so the model is trained on nine features instead of eleven. The
model is never told which mission an object came from.

### 10.3 Label harmonisation

The TOI table carries the TFOP working group disposition, which is mapped
onto the three canonical labels:

| TFOPWG | canonical | rows |
|---|---|---|
| `PC` planetary candidate | CANDIDATE | 4,820 |
| `APC` ambiguous planetary candidate | CANDIDATE | 490 |
| `CP` confirmed planet | CONFIRMED | 815 |
| `KP` known planet | CONFIRMED | 606 |
| `FP` false positive | FALSE POSITIVE | 1,315 |
| `FA` false alarm | FALSE POSITIVE | 100 |

The two vetting pipelines are not the same, and the classes are not
interchangeable. The Kepler FALSE POSITIVE class is the automatic
pipeline's rejection of a large population of obviously non-planet
signals, eclipsing binaries among them. The TOI FALSE POSITIVE class is
a rejection by human vetting of objects that looked plausible enough to
be worth vetting. This turns out to be the main reason the transfer
fails.

### 10.4 Recalibration

Two recalibrations are fitted on a held-out half of the TESS objects,
grouped by host star, and scored on the other half.

*Isotonic* refits the per-class isotonic map on the TESS calibration
half. On a population that is 65% CANDIDATE, the per-class maps push the
arg-max onto that class and bury the other two, so this recovers almost
nothing.

*Label shift* (the prior correction) rescales the class probabilities by
the ratio of the target prior to the source prior,

    p_new(y | x)  proportional to  p_old(y | x) * pi_new(y) / pi_old(y)

where `pi_old` is the class prior of the Kepler training fold and
`pi_new` is estimated from the TESS calibration labels. This is the
classical label-shift correction and it uses no information from the
target features beyond their labels, so it is the most conservative
thing that can be done to a model that has moved population.

### 10.5 Results

| setting | macro-F1 | PR-AUC | coverage |
|---|---|---|---|
| Kepler test fold | 0.6749 | 0.7462 | 0.8966 |
| TESS, zero shot | 0.2930 | 0.4407 | 0.4740 |
| TESS, isotonic recalibration | 0.2939 | 0.4439 | 0.9020 |
| TESS, label-shift recalibration | 0.4144 | 0.4317 | 0.8853 |
| TESS, trained on TESS | 0.4341 | 0.5757 | 0.9085 |

Per-class PR-AUC, against the class prior:

| setting | CONFIRMED | CANDIDATE | FALSE POSITIVE |
|---|---|---|---|
| Kepler test fold | 0.826 | 0.491 | 0.922 |
| TESS, zero shot | 0.319 | 0.677 | 0.326 |
| TESS, in domain | 0.431 | 0.789 | 0.507 |

Two separate things go wrong. The class prior moves from CANDIDATE 21%
to 65%, which a label-shift correction repairs and which accounts for
most of the gap between the zero-shot and in-domain macro-F1. The
ranking collapses on the other two classes, and the in-domain TESS
reference shows that this is mostly the task being different rather than
the transfer failing: the same features on the same objects reach only
PR-AUC 0.51 on the TESS FALSE POSITIVE class.

The conformal coverage collapses to 0.474 in the zero-shot case. That is
expected and is not a bug: the split-conformal guarantee is an
exchangeability statement, and TESS objects are not a fresh sample of
Kepler objects. The zero-shot coverage is reported as an observation, not
as a guarantee.

### 10.6 Conclusion

A Kepler-trained model does not transfer to TESS, and the reason is that
the two tables answer different questions. A product that claims to work
on TESS has to be trained on TESS. This is the kind of result that a
copied benchmark would have hidden, and it is the reason the project
measures the cross-mission claim instead of asserting it.

## 11. The service

The API serves the same `VettingPipeline` object that the training script
fits, loaded once at start-up from `results/model_xgboost.joblib`. A
request is a row of feature values, possibly with gaps; the gaps are
imputed with the training median inside the pipeline, exactly as in
training, so the model never sees a value the imputer did not see.

The response carries the class, the calibrated probability, all three
class probabilities, the conformal prediction set, the `needs_review`
flag, the top SHAP features and the plain-English report.

The `needs_review` flag is not a heuristic. It is set when the conformal
prediction set holds more than one class, which is the model's own
statement that the evidence does not single out one class. The front end
shows this as a badge, and the report says that the model abstains.

Every response also carries the warnings that the model is trained on
Kepler and that the cross-mission evaluation does not support TESS. A
service that hides its own limits is not an explainable service.

Corrections are logged to a JSONL file, one record per line, with the
measurements, the predicted class, the corrected class and a timestamp.
The format needs no database, and the log can be appended without
rewriting it.

## References

- Armstrong, Gamper & Damoulas (2021), MNRAS 504(4), 5327.
  https://arxiv.org/abs/2008.10516
- Shallue & Vanderburg (2018). https://arxiv.org/abs/1712.05044
- Thompson et al. (2018). https://arxiv.org/abs/1710.06758
- Valizadegan et al. (2022). https://arxiv.org/abs/2111.10009
- Angelopoulos & Bates (2023). https://arxiv.org/abs/2107.07511
- SHAP and conformal prediction.
  https://proceedings.mlr.press/v266/johansson25a.html
