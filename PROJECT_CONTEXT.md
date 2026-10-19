# ExoVet: Explainable, Uncertainty-Aware Exoplanet Candidate Vetting (working title)

Drop this file in the repo root. It is the full context for building the project.

## 1. Problem
Kepler flagged thousands of Kepler Objects of Interest (KOIs): dips in starlight that might be planets. Many are false positives (eclipsing binaries, centroid offsets, instrument artefacts). Vetting is slow, and most public classifiers output a bare label with inflated accuracy.

## 2. Goal
A web tool that takes a KOI's measured properties and returns:
1. A class: CONFIRMED / CANDIDATE / FALSE POSITIVE.
2. A calibrated probability.
3. A conformal prediction set. If the set is not a single class, the tool says "needs manual review" (abstention with a coverage guarantee).
4. A SHAP-based explanation and a plain-English report built only from the top SHAP features.

## 3. Data
- Source: NASA Exoplanet Archive, KOI cumulative table (about 9,564 rows; confirm the live count when you pull it).
- Pull with `astroquery.ipac.nexsci.nasa_exoplanet_archive` (`NasaExoplanetArchive.query_criteria(table="cumulative", select="*")`) or the archive's TAP service. Save the raw CSV and its checksum.
- Target: `koi_disposition`. Check the actual values in the file and drop any rows that are not CONFIRMED / CANDIDATE / FALSE POSITIVE.
- Core physical features: `koi_period`, `koi_duration`, `koi_depth`, `koi_prad`, `koi_teq`, `koi_impact`, `koi_insol`, `koi_steff`, `koi_slogg`, `koi_srad`, `koi_kepmag` (verify names against the live schema).
- **Leakage-risk columns** (outputs of the vetting process): `koi_score` (confidence 0-1), `koi_pdisposition` (Kepler's own CANDIDATE / FALSE POSITIVE call), `koi_fpflag_nt`, `koi_fpflag_ss`, `koi_fpflag_co`, `koi_fpflag_ec` (pass/fail of the false-positive tests). Exclude them from the main model and report an ablation with and without them.
- Cross-mission test: TESS Objects of Interest (TOI) table from the same archive. Use only physical features shared by both missions; never include mission identity as a feature.

## 4. Method
1. Ingest and clean: keep the three labels, drop ID/name columns, impute missing values inside the CV pipeline.
2. Splits: grouped K-fold by host star so sibling KOIs never straddle train and test (host-star ID column is `kepid`; verify).
3. Models: logistic regression (baseline), random forest, XGBoost (main).
4. Metrics: macro-F1, per-class recall, PR-AUC, recall at 99% precision (ExoMiner-style), log-loss, Brier score, expected calibration error. Do not rely on accuracy alone.
5. Leakage ablation: full features vs. leakage columns removed, same splits.
6. Calibration: isotonic or Platt on a held-out calibration fold.
7. Conformal prediction: split conformal, target coverage 90%. Implement directly in numpy; MAPIE is an option (verify in current docs).
8. Explainability: SHAP `TreeExplainer`, global and per-prediction. Check top-feature stability across folds.
9. Report generator: template filled from top-k SHAP values; optional LLM paraphrase under a "use only these numbers" constraint; automated check that no feature outside the top-k is mentioned.
10. Cross-mission: train on Kepler, evaluate zero-shot on TESS TOI; report the drop and recalibrate if needed.
11. Stretch: feedback endpoint that logs user corrections.

## 5. Suggested repo layout
```
exovet/
  data/raw/  data/processed/
  src/exovet/  data.py features.py splits.py models.py calibration.py conformal.py explain.py report.py eval.py
  scripts/     fetch_data.py train.py ablate_leakage.py cross_mission.py
  api/         main.py            # FastAPI: /predict /explain /feedback
  web/                            # React front end
  notebooks/   tests/   paper/
  requirements.txt  README.md
```

## 6. Done means
- One command reproduces all results from raw data (fixed seeds, pinned requirements).
- Results table: with vs. without leakage columns, grouped vs. random split.
- Reliability diagram and empirical conformal coverage on held-out data.
- Zero-shot Kepler to TESS table.
- Demo: form, probability bar, SHAP chart, review badge, narrated report.

## 7. Base paper and key references
- Base: Armstrong, Gamper & Damoulas (2021), MNRAS 504(4), 5327. https://arxiv.org/abs/2008.10516
- AstroNet: Shallue & Vanderburg (2018). https://arxiv.org/abs/1712.05044
- KOI/DR25 labels: Thompson et al. (2018). https://arxiv.org/abs/1710.06758
- ExoMiner: Valizadegan et al. (2022). https://arxiv.org/abs/2111.10009
- Conformal: Angelopoulos & Bates (2023). https://arxiv.org/abs/2107.07511
- SHAP + conformal: https://proceedings.mlr.press/v266/johansson25a.html
- Full list: the literature survey doc.

## 8. Existing code to read (reference only, no licence file visible in these repos)
- https://github.com/root2d2/NASA-Space-Apps-Challenge (Python layout: train.py, exoplanet_model.py, Streamlit app)
- https://github.com/udohjeremiah/planex (Next.js front end) and https://github.com/Fr33b13101/planex (FastAPI backend with a pre-trained Keras model)
- https://huggingface.co/ZapatoProgramming/ExoMAC-KKT (mission-agnostic feature harmonisation)

## 9. Cautions
- Verify every column name and label value against the live data.
- Without a licence, copied code is not clearly reusable. Write your own and cite ideas.
- Very high accuracies in other work may come from leakage columns; do not copy their benchmarks.
