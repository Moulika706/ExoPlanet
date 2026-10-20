# Methods - ExoVet

The method as it is implemented, in the form a paper needs it. Phase 2
covers the data, the model, the calibration, the prediction sets and the
explanations. Phase 5 (cross-mission) and Phase 7 (figures) extend this
document.

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

## References

- Armstrong, Gamper & Damoulas (2021), MNRAS 504(4), 5327.
  https://arxiv.org/abs/2008.10516
- Shallue & Vanderburg (2018). https://arxiv.org/abs/1712.05044
- Thompson et al. (2018). https://arxiv.org/abs/1710.06758
- Valizadegan et al. (2022). https://arxiv.org/abs/2111.10009
- Angelopoulos & Bates (2023). https://arxiv.org/abs/2107.07511
- SHAP and conformal prediction.
  https://proceedings.mlr.press/v266/johansson25a.html
