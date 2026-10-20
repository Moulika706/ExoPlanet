# Figure captions

The figures are built by `scripts/make_figures.py` from the bundles in
`results/`, so they can be regenerated without refitting anything:

```bash
make phase2    # training, ablation, SHAP stability
make cross     # the cross-mission evaluation
make figures   # the figures
```

Each figure is written to `paper/figures/` as a PNG (for slides) and a PDF
(for the paper). Seed 42 throughout; the grouped split by host star gives
5,739 train / 1,911 calibration / 1,914 test rows.

---

**Figure 1. Reliability of the headline model.** *(`reliability.png`)*

Calibration curve of the XGBoost model on the untouched test fold, with
and without isotonic calibration, against the perfect-calibration
diagonal. Isotonic calibration on the held-out calibration fold lowers
the expected calibration error from 0.0379 to 0.0239. The right panel
shows the population in each confidence bin: calibration compresses the
confidences, which is why the curve moves as well as the bins. The
calibration fold is disjoint from the training fold, and the test fold
is touched by neither.

---

**Figure 2. The conformal guarantee, and what it costs.** *(`conformal_coverage.png`)*

Empirical coverage per class of the split-conformal prediction sets at
alpha = 0.1, for the marginal and the Mondrian modes, on the same test
fold. The marginal mode uses one threshold for all classes and reaches
0.91 overall, but it under-covers the rare CANDIDATE class at 0.77,
below the 0.90 target, because a single global threshold is dominated by
the two common classes. The Mondrian mode fits one threshold per class
and covers every class above the target, at the price of a larger
average prediction set (1.49 against 1.41) and a lower singleton rate
(53% against 63%).

---

**Figure 3. Leakage ablation.** *(`leakage_ablation.png`)*

Left: macro-F1 on the same test fold for three models under four
settings. Set A uses only physical measurements; set B adds the six
vetting-output columns of the archive, which are outputs of the very
process the model is supposed to predict. The right panel separates the
three effects that are usually conflated: the feature leak (adding set
B) is worth +0.16 to +0.31 macro-F1, the random split is worth -0.03 to
-0.05, and the sibling leak (a host star's other KOI in the training
fold, same test rows) is worth at most 0.01. The grouped split by host
star is the default because it costs nothing.

---

**Figure 4. What the model uses.** *(`shap_importance.png`)*

Mean absolute SHAP value of each core physical feature, over the test
fold of the headline model, computed with `TreeExplainer` in `tree_path`
mode. The orbital period, the transit depth and the planet radius
dominate. The ranking is stable: it is the same in all five grouped
folds, see Figure 5.

---

**Figure 5. Stability of the explanation across folds.** *(`shap_stability.png`)*

Left: the rank of each of the top five features in each of the five
grouped folds, where every fold is held out in turn, the model is
refitted on the remaining folds and the global SHAP importance is
computed on the held-out rows. The rankings are independent of each
other, and only the order of the first two features moves in one fold.
Right: the Jaccard overlap of the top-five sets for all ten pairs of
folds, which is 1.000 everywhere. The explanation shown to a user is a
property of the model, not of one split.

---

**Figure 6. Cross-mission transfer.** *(`cross_mission.png`)*

Left: macro-F1 of the Kepler-trained model on Kepler and on the TESS
Objects of Interest table. Applied to TESS with no change at all, the
model scores 0.293 against 0.675 in domain. A label-shift correction,
which rescales the class probabilities by the ratio of the target and
source priors, recovers most of the difference (0.414) because the
class prior moved from 21% CANDIDATE in Kepler to 65% in TESS. The
right panel shows why the rest cannot be recovered: the one-vs-rest
PR-AUC of the FALSE POSITIVE class falls from 0.92 on Kepler to 0.33
zero-shot, and even a model trained on TESS from scratch reaches only
0.51, because the two tables define that class differently. The Kepler
FALSE POSITIVE class is the automatic pipeline's rejection of a large
population of obviously non-planet signals; the TESS class is a
human-vetted rejection of subtle ones. A Kepler-trained model does not
transfer, and the reason is that the two tables answer different
questions.
