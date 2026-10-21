# ExoVet: explainable, uncertainty-aware vetting of transiting signals

*A working draft. Every number in it is produced by the pipeline in this
repository and can be regenerated with `make all`. Figure references point
at `paper/figures/`, and the captions are in `paper/FIGURE_CAPTIONS.md`.*

---

## Abstract

Automated vetting of transiting signals is easy to do badly. A classifier
trained on the public catalogues reaches headline accuracies above 0.9,
and a great deal of that number is an artefact: the catalogues contain the
*output* of the vetting process, and a model given those columns reads the
answer off them. We build a vetting tool that is explicit about what it
knows. It predicts one of three classes, reports a calibrated probability,
returns a split-conformal prediction set with a coverage guarantee, and
explains itself in plain English from the five features that drove the
decision. We measure the parts that are usually left implicit. The
vetting-output columns are worth 0.16 to 0.31 macro-F1, which is the size
of the leak, and one of them decides the class for 99.8% of the rows it
labels. The explanation is stable: the same five features come out of
every one of five grouped folds, with a Jaccard overlap of 1.000 between
all ten pairs. The tool does not survive a change of mission. A model
trained on Kepler and applied to TESS objects without retraining falls
from 0.675 to 0.293 macro-F1, and a model trained on TESS from scratch
reaches only 0.434, because the two catalogues do not define their
classes the same way. We argue that the honest description of a vetting
model is the size of the guarantee it can give, not the size of its
accuracy, and that a tool which abstains is more useful than one which
does not.

---

## 1. Introduction

A transit is a dip in a star's light. Some of those dips are planets, some
are eclipsing binary stars, some are instrumental artefacts, and many are
ambiguous. Deciding which is which matters: a false positive costs a
telescope night, and a missed planet costs a discovery.

The catalogues already contain a large number of such decisions. The NASA
Exoplanet Archive's Kepler Object of Interest (KOI) table carries 9,564
objects, each with a disposition - CONFIRMED, CANDIDATE or FALSE POSITIVE
- derived from a mixture of automated vetting and ground-based
follow-up. The TESS Objects of Interest (TOI) table carries a comparable
number of TESS detections with a disposition from a different, human-led
programme. It is therefore natural to ask whether a model trained on one
catalogue transfers to the other, and natural to assume that it does.

This paper argues that neither the accuracy nor the transfer is what it
appears to be, and that both can be measured rather than assumed. We
build a vetting tool around three commitments: the model is trained only
on physical measurements, the model says when it does not know, and the
model explains itself in terms the user can check. We then measure the
size of the leak that the first commitment avoids, the size of the
abstention that the second produces, and the cost of the third to
predictions that turn out to be less stable than they look.

Our contributions are:

1. **A leakage-controlled evaluation.** We separate three effects that
   are usually conflated: the vetting-output columns, which are worth
   0.16 to 0.31 macro-F1; the split strategy, which is worth -0.03 to
   -0.05; and the sibling leak from multi-planet host stars, which is
   worth at most 0.01. All three are reported on the same untouched test
   fold, with the same model.
2. **A calibrated, abstaining, explained tool.** A split-conformal
   predictor gives a prediction set with a coverage guarantee, and the
   model abstains when the set holds more than one class. We show that
   the guarantee is not free: the marginal mode under-covers the rare
   class, and the class-conditional mode costs ten points of singleton
   rate.
3. **A measurement of the stability of the explanation.** We refit the
   model on every grouped fold and compute the global SHAP importance on
   the held-out rows. The ranking is identical in all five folds.
4. **A negative cross-mission result, with a diagnosis.** A Kepler-trained
   model does not transfer to TESS. We show that this is mostly not a
   transfer failure but a difference in the task: the two catalogues
   define FALSE POSITIVE differently, and a model trained on TESS cannot
   separate that class either.

---

## 2. Data

### 2.1 Kepler KOIs

The Kepler KOI cumulative table of the NASA Exoplanet Archive, fetched
with `astroquery` and falling back to the archive's TAP service. The raw
CSV, its SHA-256 checksum and the retrieval date are kept in `data/raw/`,
and the live schema is recorded in `docs/DATA_NOTES.md`. The pull used
here has 9,564 rows and 108 columns.

The target is `koi_disposition`, which takes exactly three values:

| class | rows | share |
|---|---|---|
| FALSE POSITIVE | 4,839 | 50.6% |
| CONFIRMED | 2,748 | 28.7% |
| CANDIDATE | 1,977 | 20.7% |

`kepid` identifies the host star and is the grouping key for every split
in this paper.

### 2.2 Two feature sets

**Set A, physical.** `koi_period`, `koi_duration`, `koi_depth`,
`koi_prad`, `koi_teq`, `koi_impact`, `koi_insol`, `koi_steff`,
`koi_slogg`, `koi_srad`, `koi_kepmag`. These are measurements of the
object and its host star, so they exist for any mission.

**Set B, leaky.** Set A plus `koi_score`, `koi_pdisposition`,
`koi_fpflag_nt`, `koi_fpflag_ss`, `koi_fpflag_co`, `koi_fpflag_ec`. These
are the *output* of the Kepler vetting process. Section 5.2 measures what
they are worth.

### 2.3 TESS TOIs

The TESS Objects of Interest table from the same archive, 8,151 rows,
fetched the same way. The TOI table carries the TFOP working group
disposition, whose six values are mapped onto the three canonical labels:

| TFOPWG | canonical | rows |
|---|---|---|
| PC (planetary candidate) | CANDIDATE | 4,820 |
| APC (ambiguous planetary candidate) | CANDIDATE | 490 |
| CP (confirmed planet) | CONFIRMED | 815 |
| KP (known planet) | CONFIRMED | 606 |
| FP (false positive) | FALSE POSITIVE | 1,315 |
| FA (false alarm) | FALSE POSITIVE | 100 |

The mapping is close but not exact, and Section 5.4 shows that the
differences matter more than they look. After the mapping the TOI table
has a very different class balance from the Kepler one:

| class | Kepler KOIs | TESS TOIs |
|---|---|---|
| FALSE POSITIVE | 4,839 (50.6%) | 1,415 (17.4%) |
| CONFIRMED | 2,748 (28.7%) | 1,421 (17.4%) |
| CANDIDATE | 1,977 (20.7%) | 5,310 (65.2%) |

That difference in priors is the first of the three things that go wrong
in Section 5.5.

---

## 3. Method

### 3.1 Splits

`StratifiedGroupKFold` with five folds, seeded once. Stratification keeps
the class balance close across folds; the group constraint keeps every
host star entirely inside one fold, so sibling KOIs of the same star
never straddle the split. The three folds used for training are:

| fold | role | rows |
|---|---|---|
| 2, 3, 4 | train | 5,739 |
| 1 | calibration | 1,911 |
| 0 | test | 1,914 |

The calibration fold is disjoint from train, so the calibrator never sees
training data, and the test fold is touched by neither the model nor the
calibrator.

### 3.2 Models

Three models are compared on the same splits: logistic regression with
standardised inputs as a transparent baseline, a random forest, and
XGBoost as the headline model. Preprocessing - median imputation,
optional standardisation, one-hot encoding of the single categorical
column - sits inside a scikit-learn pipeline, so it is refit on each
training fold and never sees validation or test rows.

A `LabelEncodingClassifier` wrapper encodes the string labels inside
`fit`, because XGBoost rejects string targets. Its class order is
alphabetical (CANDIDATE, CONFIRMED, FALSE POSITIVE) and is *not* the
canonical order used everywhere else, so the probability columns are
reordered explicitly before any metric is computed. Reading them in the
wrong order is a silent error: it turns a macro-F1 of 0.73 into 0.36.

### 3.3 Metrics

Accuracy is deliberately absent. A model that labels everything FALSE
POSITIVE scores 50.6% on this table and is useless. We report macro-F1,
per-class recall, macro one-vs-rest PR-AUC, recall at 99% precision (the
ExoMiner-style operating point), multiclass log loss, the multiclass
Brier score and the expected calibration error over ten equal-width
confidence bins.

The log loss is computed in the canonical class order rather than through
`sklearn.metrics.log_loss`, whose internal `LabelBinarizer` sorts the
labels and would score the wrong column.

### 3.4 Calibration

Isotonic regression by default, fitted on the held-out calibration fold
and applied to the test fold. The implementation also provides Platt
scaling, confidence-isotonic, vector scaling, temperature scaling and a
label-shift (prior) correction. The last is used in Section 5.4: it
rescales the class probabilities by the ratio of the target prior to the
source prior,

    p_new(y | x)  proportional to  p_old(y | x) * pi_new(y) / pi_old(y)

which is the right tool when the population has moved but the
relationship between the features and the label has not.

### 3.5 Conformal prediction

Split conformal, implemented directly in numpy following Angelopoulos &
Bates (2023). The nonconformity score is the probability deficit of the
true label, `s(x, y) = 1 - p_y(x)`, and the prediction set is

    C(x) = { y : 1 - p_y(x) <= q }

where `q` is the finite-sample corrected `(1 - alpha)` quantile of the
calibration scores, computed with the rank `ceil((n + 1) * (1 - alpha)) / n`
so that `P(y in C(x)) >= 1 - alpha` holds for any exchangeable data. We
use `alpha = 0.10`, so the target coverage is 0.90.

Two modes are implemented. The **marginal** mode uses one threshold for
all classes and guarantees marginal coverage. The **Mondrian** mode fits
one threshold per class, on the calibration points of that class, and
guarantees class-conditional coverage. The model's arg-max is always
added to the set, so a set is never empty; adding a class can only grow
the set, so the guarantee is unaffected.

### 3.6 Explanations

SHAP `TreeExplainer` in `tree_path` mode, computed on the transformed
feature matrix because the tree models consume the output of the
preprocessor. One-hot dummies are mapped back to their source column.

The report is a template filled from the top five SHAP features and
nothing else. A checker re-reads the generated text and raises if any
feature outside the top five is mentioned, so the "explainable" claim is
enforced by code rather than by good intentions.

---

## 4. The tool

The same pipeline object serves the API. A request carries a row of
feature values, possibly with gaps; the gaps are imputed with the
training median inside the pipeline, exactly as in training. The response
carries the class, the calibrated probability, all three class
probabilities, the conformal prediction set, a `needs_review` flag, the
top SHAP features and the plain-English report.

The `needs_review` flag is not a heuristic. It is set when the conformal
prediction set holds more than one class, which is the model's own
statement that the evidence does not single out one class. The front end
shows this as a badge, and the report says that the model abstains.

Corrections are logged to a JSONL file with the measurements, the
predicted class, the corrected class and a timestamp, so the log needs no
database and is never rewritten.

---

## 5. Results

### 5.1 In-domain performance

Seed 42, grouped split by host star, metrics on the untouched test fold
after isotonic calibration on the held-out calibration fold.

| model | macro-F1 | PR-AUC | log-loss | Brier | ECE | recall@99% | conformal cov. | mean set |
|---|---|---|---|---|---|---|---|---|
| logistic regression | 0.5272 | 0.6088 | 0.7797 | 0.4533 | 0.0234 | 0.0548 | 0.9091 | 1.93 |
| random forest | 0.6981 | 0.7666 | 0.5796 | 0.3360 | 0.0189 | 0.1485 | 0.9013 | 1.44 |
| **XGBoost** | **0.7290** | **0.7794** | **0.5598** | **0.3178** | 0.0239 | 0.1320 | 0.9075 | 1.41 |

Per-class recall under the arg-max decision:

| model | CONFIRMED | CANDIDATE | FALSE POSITIVE |
|---|---|---|---|
| logistic regression | 0.796 | 0.091 | 0.810 |
| random forest | 0.844 | 0.417 | 0.839 |
| XGBoost | 0.862 | 0.482 | 0.844 |

CANDIDATE is the hard class, as expected: a candidate is a signal that has
not been followed up, so it shares its physical signature with the
confirmed planets in every respect except the absence of confirmation. The
transparent baseline nearly fails on it (0.091), which is a useful
reminder that a linear model cannot express what makes a signal
unconfirmed.

Calibration is not doing the heavy lifting on the ranking. Isotonic
calibration moves the macro-F1 from 0.7224 to 0.7290 for XGBoost, from
0.7104 to 0.6981 for the random forest, and from 0.5466 to 0.5272 for
the logistic regression. It is a per-class monotone map, so it can reorder
the classes relative to the uncalibrated model, and on two of the three
models it makes the arg-max decision slightly worse. What it reliably
does is make the probabilities mean what they say, which is what the
conformal predictor and the user interface depend on.

The class prior is not the only difficulty. The XGBoost model's recall at
99% precision is 0.132, *below* the random forest's 0.149, despite a
higher macro-F1. A tool that has to keep precision above 99% would
prefer the random forest here, and we report it rather than hide it.

### 5.2 The cost of the guarantee

Both conformal modes are fitted on the same calibration fold and scored on
the same test fold.

| mode | coverage | cov. CONFIRMED | cov. CANDIDATE | cov. FALSE POSITIVE | mean set | singleton |
|---|---|---|---|---|---|---|
| marginal | 0.9075 | 0.9327 | 0.7652 | 0.9514 | 1.41 | 63.1% |
| mondrian | 0.9039 | 0.9109 | 0.9066 | 0.8988 | 1.49 | 53.3% |

The marginal mode buys a smaller average set, but it under-covers the
rare CANDIDATE class at 0.765, well under the 0.90 target, because a
single global threshold is dominated by the two common classes. The
Mondrian mode gives up ten points of singleton rate and covers every
class above the target, which is the guarantee the product should claim.

The expected calibration error of the headline model is 0.0379 before
calibration and 0.0239 after (Figure 1).

### 5.3 Leakage ablation

All numbers are on the same untouched test fold; only the training data
changes.

| model | set A (clean) + grouped | set B (leaky) + grouped | feature-leak delta |
|---|---|---|---|
| logistic regression | 0.5272 | 0.8352 | +0.3080 |
| random forest | 0.6981 | 0.8848 | +0.1867 |
| XGBoost | 0.7290 | 0.8921 | +0.1631 |

The vetting-output columns are worth 0.16 to 0.31 macro-F1. The reason
is not subtle. `koi_pdisposition` is Kepler's own disposition call, and
the crosstab of that column against the target shows why the model cannot
be built on it:

| `koi_pdisposition` | CANDIDATE | CONFIRMED | FALSE POSITIVE |
|---|---|---|---|
| CANDIDATE | 1,977 | 2,739 | 1 |
| FALSE POSITIVE | 0 | 9 | 4,838 |

A single column decides the class for 99.8% of the rows it calls FALSE
POSITIVE. It is the answer key, not a measurement.

The split strategy is a smaller and more interesting effect. We measure
three protocols on the same test rows where possible:

| model | clean + grouped | clean + random split | random-split delta | clean + sibling leak | sibling-leak delta |
|---|---|---|---|---|---|
| logistic regression | 0.5272 | 0.4986 | -0.0286 | 0.5283 | +0.0012 |
| random forest | 0.6981 | 0.6504 | -0.0477 | 0.6960 | -0.0021 |
| XGBoost | 0.7290 | 0.6857 | -0.0433 | 0.7195 | -0.0095 |

The random split, which is what most of the literature reports, is
*slightly worse* than the grouped split. The sibling leak - the
appearance of a test host star's other KOI in the training fold - is
worth at most 0.01, because only 938 of the 8,214 host stars (11.4%)
carry more than one KOI. Both are measured rather than assumed, and the
grouped split stays the default because it costs nothing and removes the
possibility altogether. Figure 3 shows all three effects.

### 5.4 Stability of the explanation

A single SHAP table says what the model used on one split. To show that
the explanation is a property of the model and not of the split, we
refit on each of the five grouped folds and compute the global SHAP
importance on the fold that was held out.

| quantity | value |
|---|---|
| mean Jaccard overlap of the top-5 sets | 1.0000 (worst pair 1.0000) |
| mean Kendall rank correlation between folds | 0.9491 (worst pair 0.8909) |
| per-fold macro-F1 | 0.7076 - 0.7281 (std 0.0076) |

Every fold picks the same five features, and only one fold swaps the
first two. The ranking is a property of the model, so the explanation the
user sees will not change from one object to the next for a reason that
has nothing to do with the object (Figure 5).

### 5.5 Cross-mission transfer

The model is trained on Kepler with the nine features that mean the same
thing in both missions, and applied to the TESS Objects of Interest table
with no change: the same weights, the same calibrator, the same conformal
threshold.

Two of the eleven core features cannot cross the mission boundary.
`koi_impact` has no TOI counterpart at all. `koi_kepmag` would have to be
replaced by `st_tmag`, which is a different photometric system;
substituting it would hand the model a systematically shifted feature, so
both are left out and the model is trained on nine features.

| setting | macro-F1 | PR-AUC | conformal cov. |
|---|---|---|---|
| Kepler test fold (in domain) | 0.6749 | 0.7462 | 0.8966 |
| TESS, zero shot | 0.2930 | 0.4407 | 0.4740 |
| TESS, recalibrated (label shift) | 0.4144 | 0.4317 | 0.8853 |
| TESS, recalibrated (isotonic) | 0.2939 | 0.4439 | 0.9020 |
| TESS, trained on TESS (in domain) | 0.4341 | 0.5757 | 0.9085 |

The model does not transfer. Three things are going on, and they are
worth separating.

**The prior moved.** CANDIDATE is 21% of the Kepler table and 65% of the
TOI table. A model fitted under the Kepler prior barely ever predicts
CANDIDATE on TESS (recall 0.06). A label-shift correction repairs this
and recovers most of the gap to the in-domain ceiling, 0.2930 to 0.4144
against a ceiling of 0.4341. The flexible isotonic fit does not repair
it, and the reason is visible in the per-class recall: a per-class
isotonic map fitted on a population that is 65% CANDIDATE pushes the
arg-max onto that class and buries the other two. On a shifted
population, the flexible calibrator is the wrong tool.

**The population is a different population.** The two tables do not
sample the same kind of object, and the medians say so. Expressed in
units of the Kepler interquartile range, so that the columns are
comparable:

| feature | Kepler median | TESS median | shift | shift in Kepler IQR |
|---|---|---|---|---|
| `koi_period` | 9.753 | 4.147 | -5.606 | -0.148 |
| `koi_duration` | 3.793 | 2.752 | -1.041 | -0.271 |
| `koi_depth` | 421.1 | 4743.0 | +4321.9 | **+3.290** |
| `koi_prad` | 2.390 | 10.487 | +8.097 | +0.598 |
| `koi_teq` | 878.0 | 1194.5 | +316.5 | +0.377 |
| `koi_insol` | 141.6 | 378.5 | +236.9 | +0.279 |
| `koi_steff` | 5767 | 5799 | +32.3 | +0.040 |
| `koi_slogg` | 4.438 | 4.330 | -0.108 | -0.332 |
| `koi_srad` | 1.000 | 1.236 | +0.236 | +0.457 |

The transit depth moves by more than three interquartile ranges: the TESS
catalogue is dominated by much deeper events, because TESS's precision
and its brighter, larger host stars push the detection threshold down.
Since the depth is the second most important feature in the model
(Figure 4), this is not a marginal shift. The planet radius moves by
0.6 IQR for the same reason. The stellar properties barely move, which
tells us the shift is in the selection, not in the stars.

**The ranking collapsed on two of the three classes.**

| setting | PR-AUC CONFIRMED | PR-AUC CANDIDATE | PR-AUC FALSE POSITIVE |
|---|---|---|---|
| Kepler test fold | 0.826 | 0.491 | 0.922 |
| TESS, zero shot | 0.319 | 0.677 | 0.326 |
| TESS, in domain | 0.431 | 0.789 | 0.507 |

The Kepler FALSE POSITIVE class is the automatic pipeline's rejection of
a large population of obviously non-planet signals, eclipsing binaries
among them, and the model separates them at PR-AUC 0.922 against a prior
of 0.51. That population does not exist in the TOI table, where the
rejected objects are the subtle ones a human looked at and threw out. A
model trained on TESS from scratch reaches only PR-AUC 0.507 on that
class, so most of the collapse is the task being different, not the
transfer failing. The classes are not the same class.

**The guarantee did not survive.** The conformal coverage falls to 0.474
zero-shot. This is expected and is not a bug: the split-conformal
guarantee is an exchangeability statement, and TESS objects are not a
fresh sample of Kepler objects. The zero-shot coverage is reported as an
observation, not as a guarantee.

The conclusion is that the two catalogues answer different questions, and
that a product claiming to work on TESS has to be trained on TESS. This
is the kind of result that a copied benchmark would have hidden, and it
is why the claim is measured rather than asserted (Figure 6).

---

## 6. Limitations

- The labels come from the NASA Exoplanet Archive, whose live values may
  differ from the DR25 paper numbers. The pull used here is recorded with
  its checksum and retrieval date.
- The TOI mapping is a judgement. PC and APC are both folded into
  CANDIDATE, CP and KP into CONFIRMED, FP and FA into FALSE POSITIVE. A
  different mapping would change the absolute numbers, though we expect
  the qualitative conclusion to hold.
- The cross-mission result is one pair of catalogues, one model and one
  seed. The diagnosis, not the exact number, is the transferable part.
- The conformal guarantee holds within a mission. Across missions it is
  an observation.
- CANDIDATE recall is low in domain (0.482). The model is better at
  saying "this is not a false positive" than at saying "this is
  unconfirmed", and the difference between those two questions is
  exactly the follow-up that has not happened yet.
- The models are trained on 9,564 rows. There is no hyper-parameter
  search, and the parameters are fixed in the configuration file.

---

## 7. Conclusion

Automated vetting is easy to make look better than it is. On this data
set, adding the archive's own vetting columns to a model is worth up to
0.31 macro-F1, and the published numbers of comparable work are hard to
compare with ours for exactly that reason. A tool that reports a
calibrated probability, abstains with a stated coverage guarantee, and
explains itself from the same handful of features every time is more
useful than a more accurate one that does none of these things. And a
tool that claims to work on data it was not trained on should be shown to
work on it, which in our case it does not.

---

## 8. Reproducibility and availability

The code, the result bundles and this draft are in one repository, and
every number in this paper is produced by the pipeline rather than
transcribed by hand. The bundle behind each claim is:

| claim | bundle |
|---|---|
| Section 5.1, 5.2 | `results/metrics.json`, `results/train_report.md`, `results/reliability.csv` |
| Section 5.3 | `results/ablation_metrics.json`, `results/ablation_report.md` |
| Section 5.4 | `results/shap_stability.json`, `results/shap_stability_report.md`, `results/shap_fold_importance.csv` |
| Section 5.5 | `results/cross_mission_metrics.json`, `results/cross_mission_report.md`, `results/cross_mission_feature_shift.csv` |
| figures | `paper/FIGURE_CAPTIONS.md`, regenerated by `make figures` |

`tests/test_paper_claims.py` reads those bundles and asserts that this
draft quotes them, so the paper cannot drift away from the code without
a test failing. The raw tables are not in the repository; they are
fetched from the archive with `make data` and `make toi`, and each fetch
records its SHA-256 checksum and its retrieval date in `data/raw/`.

The whole of the paper, from the raw data to the figures, is reproduced
by

    make all

with one seed (42) and pinned dependencies. The fitted model is committed
as `results/model_xgboost.joblib`, so the demonstration service serves
exactly the model this paper describes.

---

## References

- Angelopoulos, A. N. & Bates, S. (2023). *Conformal Prediction: A
  Gentle Introduction.* https://arxiv.org/abs/2107.07511
- Armstrong, J. A., Gamper, J. & Damoulas, G. (2021). *Let's not
  F(ε) it: A Bayesian approach to the Kepler exoplanet search.* MNRAS
  504(4), 5327. https://arxiv.org/abs/2008.10516
- Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to
  Interpreting Model Predictions.* https://arxiv.org/abs/1705.07874
- Shallue, B. D. & Vanderburg, A. (2018). *AstroNet: A deep learning
  approach to detecting planets in the Kepler transit photometry.*
  https://arxiv.org/abs/1712.05044
- Thompson, J. R. et al. (2018). *Kepler Data Release VII: The DR25
  catalog of validated Kepler exoplanets.* https://arxiv.org/abs/1710.06758
- Valizadegan, M. et al. (2022). *ExoMiner: A highly accurate and
  explainable deep learning classifier that differentiates between
  confirmed planets and false positives.* https://arxiv.org/abs/2111.10009
- Johansson, M. et al. (2025). *Conformal prediction and SHAP.*
  https://proceedings.mlr.press/v266/johansson25a.html
