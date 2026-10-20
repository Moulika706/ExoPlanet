"""Tests for the cross-mission transfer (Phase 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exovet.calibration import apply_calibrator, fit_calibrator
from exovet.crossmission import (
    DISPOSITION_TO_LABEL,
    EXCLUDED_FEATURES,
    TOI_COLUMN_FOR_FEATURE,
    clean_toi,
    excluded_features,
    feature_shift_table,
    harmonise_toi,
    shared_features,
)
from exovet.data import LABELS, TARGET

CLASSES = tuple(LABELS)


def _toi_frame(n: int = 12) -> pd.DataFrame:
    """A small TOI table with the real column names."""
    rng = np.random.default_rng(0)
    dispositions = list(DISPOSITION_TO_LABEL)[: n % 6 or 6]
    rows = []
    for index in range(n):
        rows.append(
            {
                "toi": 1000 + index,
                "tid": index // 2,
                "tfopwg_disp": dispositions[index % len(dispositions)],
                "pl_orbper": float(rng.uniform(1, 40)),
                "pl_trandurh": float(rng.uniform(1, 8)),
                "pl_trandep": float(rng.uniform(100, 5000)),
                "pl_rade": float(rng.uniform(1, 15)),
                "pl_eqt": float(rng.uniform(300, 2000)),
                "pl_insol": float(rng.uniform(1, 1000)),
                "st_teff": float(rng.uniform(4000, 7000)),
                "st_logg": float(rng.uniform(4.0, 4.6)),
                "st_rad": float(rng.uniform(0.5, 2.0)),
                "st_tmag": float(rng.uniform(4, 12)),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# feature harmonisation
# ----------------------------------------------------------------------
def test_shared_features_exclude_the_two_that_cannot_cross() -> None:
    shared = shared_features()
    assert "koi_impact" not in shared
    assert "koi_kepmag" not in shared
    assert set(shared).issubset(set(TOI_COLUMN_FOR_FEATURE))
    # Everything excluded has a stated reason.
    assert set(excluded_features()) == set(EXCLUDED_FEATURES)
    for reason in excluded_features().values():
        assert reason


def test_harmonise_renames_to_the_kepler_feature_names() -> None:
    df = harmonise_toi(_toi_frame())
    for feature in shared_features():
        assert feature in df.columns
    assert TARGET in df.columns
    assert "tid" in df.columns


def test_harmonise_maps_the_disposition_onto_the_canonical_labels() -> None:
    df = clean_toi(harmonise_toi(_toi_frame()))
    assert set(df[TARGET]).issubset(set(CLASSES))
    # Every disposition in the mapping survives the cleaning.
    assert len(df) == 12


def test_harmonise_drops_rows_with_an_unknown_disposition() -> None:
    raw = _toi_frame()
    raw.loc[0, "tfopwg_disp"] = "SOMETHING ELSE"
    raw.loc[1, "tfopwg_disp"] = None
    df = clean_toi(harmonise_toi(raw))
    assert len(df) == 10


def test_harmonise_raises_on_an_unknown_feature() -> None:
    with pytest.raises(ValueError):
        harmonise_toi(_toi_frame(), features=["koi_impact"])


def test_harmonise_raises_on_a_missing_toi_column() -> None:
    raw = _toi_frame().drop(columns=["pl_orbper"])
    with pytest.raises(ValueError):
        harmonise_toi(raw)


def test_feature_shift_table_reports_the_median_move() -> None:
    kepler = pd.DataFrame({"koi_period": [1.0, 2.0, 3.0, 4.0]})
    tess = pd.DataFrame({"koi_period": [10.0, 20.0, 30.0, 40.0]})
    table = feature_shift_table(kepler, tess, features=["koi_period"])
    row = table.iloc[0]
    assert row["kepler_median"] == pytest.approx(2.5)
    assert row["tess_median"] == pytest.approx(25.0)
    assert row["median_shift"] == pytest.approx(22.5)
    assert row["shift_in_kepler_iqr"] > 0


# ----------------------------------------------------------------------
# prior correction (label shift)
# ----------------------------------------------------------------------
def test_prior_correction_shifts_mass_towards_the_new_majority() -> None:
    proba = np.array([[0.5, 0.3, 0.2], [0.4, 0.35, 0.25]])
    y = np.array(["CANDIDATE", "CANDIDATE"])
    state = fit_calibrator("prior", proba, y, CLASSES, source_prior=[0.29, 0.21, 0.50])
    out = apply_calibrator(state, proba)
    assert out.shape == proba.shape
    assert np.allclose(out.sum(axis=1), 1.0)
    # CANDIDATE is 100% of the target and 21% of the source, so its
    # probability must grow.
    assert out[0, 1] > proba[0, 1]
    assert out[0, 0] < proba[0, 0]


def test_prior_correction_is_a_no_op_when_the_priors_match() -> None:
    proba = np.array([[0.5, 0.3, 0.2]])
    y = np.array(["CONFIRMED", "CONFIRMED"])
    state = fit_calibrator("prior", proba, y, CLASSES, source_prior=[1.0, 0.0, 0.0])
    out = apply_calibrator(state, proba)
    assert np.allclose(out, proba)


def test_prior_correction_uses_the_source_prior() -> None:
    proba = np.array([[0.5, 0.3, 0.2]])
    y = np.array(["CANDIDATE", "CONFIRMED"])
    # The target prior is 0.5 for both CONFIRMED and CANDIDATE.
    low = fit_calibrator("prior", proba, y, CLASSES, source_prior=[0.10, 0.90, 0.0])
    high = fit_calibrator("prior", proba, y, CLASSES, source_prior=[0.90, 0.10, 0.0])
    # CANDIDATE was common in the source and rare in the target, so it
    # loses; the opposite source prior makes it gain.
    assert apply_calibrator(low, proba)[0, 1] < proba[0, 1]
    assert apply_calibrator(high, proba)[0, 1] > proba[0, 1]


def test_unknown_calibration_method_still_raises() -> None:
    proba = np.array([[0.5, 0.3, 0.2]])
    y = np.array(["CANDIDATE"])
    with pytest.raises(ValueError):
        fit_calibrator("nope", proba, y, CLASSES)


# ----------------------------------------------------------------------
# recalibration on the pipeline
# ----------------------------------------------------------------------
def _synthetic_koi_frame(n_groups: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    from exovet.features import feature_list

    features = feature_list(False)
    shifts = {"CONFIRMED": 3.0, "CANDIDATE": 0.0, "FALSE POSITIVE": -3.0}
    rows = []
    for group in range(n_groups):
        label = CLASSES[group % 3]
        for _member in range(2):
            row = {f: float(rng.normal(shifts[label], 0.4)) for f in features}
            row["kepid"] = group
            row["koi_disposition"] = label
            rows.append(row)
    return pd.DataFrame(rows)


def test_recalibrate_refits_the_calibrator_and_the_conformal() -> None:
    from exovet.pipeline import VettingPipeline

    df = _synthetic_koi_frame()
    pipeline = VettingPipeline(model_name="random_forest", params={"n_estimators": 20}, seed=42)
    pipeline.fit(df, n_splits=3, test_fold=0, calibration_fold=1)

    before = pipeline.calibrator
    before_conformal = pipeline.conformal
    pipeline.recalibrate(df, target_column="koi_disposition", classes=CLASSES, method="isotonic")
    assert pipeline.calibrator is not before
    assert pipeline.conformal is not before_conformal


def test_recalibrate_with_prior_correction_uses_the_training_prior() -> None:
    from exovet.pipeline import VettingPipeline

    df = _synthetic_koi_frame()
    pipeline = VettingPipeline(model_name="random_forest", params={"n_estimators": 20}, seed=42)
    pipeline.fit(df, n_splits=3, test_fold=0, calibration_fold=1)
    assert pipeline.train_prior
    assert set(pipeline.train_prior) == set(CLASSES)
    pipeline.recalibrate(df, target_column="koi_disposition", classes=CLASSES, method="prior")
    assert pipeline.calibrator["method"] == "prior"
    assert len(pipeline.calibrator["ratio"]) == 3


def test_recalibrate_does_not_move_the_model() -> None:
    from exovet.pipeline import VettingPipeline

    df = _synthetic_koi_frame()
    pipeline = VettingPipeline(model_name="random_forest", params={"n_estimators": 20}, seed=42)
    pipeline.fit(df, n_splits=3, test_fold=0, calibration_fold=1)
    row = df.iloc[0]
    values = {f: float(row[f]) for f in pipeline.features}
    before = pipeline.predict_one(values).predicted_class
    pipeline.recalibrate(df, target_column="koi_disposition", classes=CLASSES, method="prior")
    after = pipeline.predict_one(values).predicted_class
    assert before == after


def test_pipeline_survives_a_save_and_load_with_the_training_prior(tmp_path) -> None:
    from exovet.pipeline import VettingPipeline

    df = _synthetic_koi_frame()
    pipeline = VettingPipeline(model_name="random_forest", params={"n_estimators": 20}, seed=42)
    pipeline.fit(df, n_splits=3, test_fold=0, calibration_fold=1)
    reloaded = VettingPipeline.load(pipeline.save(tmp_path / "model.joblib"))
    assert reloaded.train_prior == pipeline.train_prior
