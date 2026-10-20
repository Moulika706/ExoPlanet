"""Tests for the Phase 2 modules: splits, features, models, calibration,
conformal prediction, metrics, explanations and reports."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exovet.calibration import (
    apply_calibrator,
    fit_calibrator,
    fit_isotonic,
    fit_platt,
    fit_temperature,
)
from exovet.conformal import (
    SplitConformal,
    conformal_scores,
    finite_sample_quantile,
)
from exovet.data import LABELS
from exovet.eval import (
    brier_score,
    evaluate,
    expected_calibration_error,
    macro_f1,
    multiclass_log_loss,
    per_class_recall,
    recall_at_precision,
    reliability_table,
)
from exovet.features import build_preprocessor, feature_list
from exovet.models import build_model
from exovet.report import build_report, check_report_features, display_name
from exovet.splits import (
    grouped_kfold,
    random_train_calibration_test,
    train_calibration_test,
)

CLASSES = tuple(LABELS)


# ----------------------------------------------------------------------
# splits
# ----------------------------------------------------------------------
def _toy_frame(n_groups: int = 30, per_group: int = 4) -> pd.DataFrame:
    rows = []
    for group in range(n_groups):
        label = ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"][group % 3]
        for member in range(per_group):
            rows.append(
                {
                    "kepid": group,
                    "koi_disposition": label,
                    "koi_period": 1.0 + group + member,
                }
            )
    return pd.DataFrame(rows)


def test_grouped_kfold_keeps_groups_together() -> None:
    df = _toy_frame()
    for fold in grouped_kfold(df, n_splits=5, seed=42):
        train_groups = set(df.iloc[fold.train_idx]["kepid"])
        test_groups = set(df.iloc[fold.test_idx]["kepid"])
        assert not (train_groups & test_groups)


def test_grouped_kfold_covers_every_row_once() -> None:
    df = _toy_frame()
    seen = []
    for fold in grouped_kfold(df, n_splits=5, seed=42):
        seen.extend(fold.test_idx.tolist())
    assert sorted(seen) == list(range(len(df)))


def test_train_calibration_test_is_disjoint() -> None:
    df = _toy_frame()
    train, cal, test = train_calibration_test(df, n_splits=5, seed=42)
    assert len(set(train) & set(cal)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(cal) & set(test)) == 0
    assert len(train) + len(cal) + len(test) == len(df)


def test_train_calibration_test_keeps_groups_apart() -> None:
    df = _toy_frame()
    train, cal, test = train_calibration_test(df, n_splits=5, seed=42)
    groups = df["kepid"]
    assert not (set(groups.iloc[train]) & set(groups.iloc[test]))
    assert not (set(groups.iloc[train]) & set(groups.iloc[cal]))


def test_random_split_can_leak_groups() -> None:
    """The random baseline is expected to be leaky; that is the point."""
    df = _toy_frame(n_groups=10, per_group=4)
    train, _, test = random_train_calibration_test(df, n_splits=5, seed=1)
    overlap = set(df["kepid"].iloc[train]) & set(df["kepid"].iloc[test])
    assert len(overlap) > 0


def test_train_calibration_test_rejects_same_fold() -> None:
    df = _toy_frame()
    with pytest.raises(ValueError):
        train_calibration_test(df, n_splits=5, test_fold=1, calibration_fold=1)


# ----------------------------------------------------------------------
# features and models
# ----------------------------------------------------------------------
def test_feature_list_sets() -> None:
    clean = feature_list(False)
    leaky = feature_list(True)
    assert "koi_score" not in clean
    assert "koi_score" in leaky
    assert set(clean).issubset(set(leaky))


def test_preprocessor_imputes_and_one_hot_encodes() -> None:
    features = feature_list(True)
    pre = build_preprocessor(features, scale=False)
    full = pd.DataFrame({f: [1.0, 2.0, 3.0] for f in features})
    full["koi_pdisposition"] = pd.Series(["CANDIDATE", "FALSE POSITIVE", None])
    full.loc[0, "koi_period"] = np.nan
    transformed = pre.fit_transform(full)
    assert transformed.shape[0] == 3
    assert not np.isnan(transformed).any()
    # koi_pdisposition is a string column, so it must be one-hot encoded.
    assert transformed.shape[1] > len(features)


def test_preprocessor_scaled_for_logistic_regression() -> None:
    features = feature_list(False)
    scaled = build_preprocessor(features, scale=True)
    plain = build_preprocessor(features, scale=False)
    frame = pd.DataFrame({f: [1.0, 5.0, 9.0] for f in features})
    a = scaled.fit_transform(frame)
    b = plain.fit_transform(frame)
    assert a.mean() == pytest.approx(0.0, abs=1e-9)
    assert not np.allclose(a, b)


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "xgboost"])
def test_build_model_fits(name: str) -> None:
    features = feature_list(False)
    frame = pd.DataFrame({f: np.random.default_rng(0).normal(size=60) for f in features})
    labels = np.random.default_rng(1).choice(CLASSES, size=60)
    pipeline = build_model(name, {"n_estimators": 20}, seed=42, feature_names=features)
    pipeline.fit(frame, labels)
    proba = pipeline.predict_proba(frame)
    assert proba.shape == (60, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


# ----------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------
def _calibration_data(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.choice(CLASSES, size=n, p=[0.3, 0.2, 0.5])
    logits = rng.normal(size=(n, 3)) * 0.5
    for row, label in enumerate(y):
        logits[row, CLASSES.index(label)] += 2.0
    logits -= logits.max(axis=1, keepdims=True)
    proba = np.exp(logits)
    proba /= proba.sum(axis=1, keepdims=True)
    return proba, y


def test_isotonic_preserves_class_column_order() -> None:
    """Each column must be calibrated against its own class, not a shuffle."""
    proba = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.05, 0.05, 0.90],
        ]
    )
    y = np.array(["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"])
    state = fit_isotonic(proba, y, CLASSES)
    out = apply_calibrator(state, proba)
    assert out.argmax(axis=1).tolist() == [0, 1, 2]
    # The regression for column i must have seen column i.
    assert np.allclose(state["regressions"][0].X_thresholds_, [0.05, 0.90])


def test_temperature_preserves_argmax() -> None:
    proba, y = _calibration_data()
    state = fit_temperature(proba, y, CLASSES)
    out = apply_calibrator(state, proba)
    assert np.array_equal(np.argmax(out, axis=1), np.argmax(proba, axis=1))


def test_platt_output_sums_to_one_and_is_finite() -> None:
    proba, y = _calibration_data()
    state = fit_platt(proba, y, CLASSES)
    out = apply_calibrator(state, proba)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.isfinite(out).all()


def test_unknown_calibration_method_raises() -> None:
    proba, y = _calibration_data()
    with pytest.raises(ValueError):
        fit_calibrator("nope", proba, y, CLASSES)


def test_calibration_none_returns_input() -> None:
    proba, _ = _calibration_data()
    assert np.allclose(apply_calibrator(None, proba), proba)


# ----------------------------------------------------------------------
# conformal prediction
# ----------------------------------------------------------------------
def test_conformal_scores_are_probability_deficits() -> None:
    proba = np.array([[0.7, 0.2, 0.1]])
    scores = conformal_scores(proba, np.array(["CONFIRMED"]), CLASSES)
    assert scores[0] == pytest.approx(0.3)


def test_finite_sample_quantile_is_at_least_target() -> None:
    scores = np.linspace(0, 1, 100)
    q = finite_sample_quantile(scores, 0.10)
    assert q >= 0.90


def test_conformal_set_is_singleton_when_confident() -> None:
    """A confident prediction keeps a singleton set; an uncertain one does not."""
    # A calibration set where most rows are uncertain, so the threshold is
    # loose enough to keep several classes.
    proba = np.array([[0.30, 0.35, 0.35]] * 360 + [[0.95, 0.025, 0.025]] * 40)
    y = np.array(["CONFIRMED"] * 360 + ["CANDIDATE"] * 40)
    conformal = SplitConformal(alpha=0.10, mode="marginal", classes=CLASSES).fit(proba, y)
    sets = conformal.predict_sets(np.array([[0.98, 0.01, 0.01], [0.34, 0.33, 0.33]]))
    assert sets[0] == ["CONFIRMED"]
    assert len(sets[1]) > 1


def test_conformal_never_returns_an_empty_set() -> None:
    rng = np.random.default_rng(3)
    n = 500
    y = rng.choice(CLASSES, size=n)
    proba = rng.dirichlet(np.ones(3) * 0.3, size=n)
    conformal = SplitConformal(alpha=0.10, mode="marginal", classes=CLASSES).fit(proba, y)
    sizes = conformal.set_sizes(proba)
    assert sizes.min() >= 1


def test_conformal_threshold_is_infinite_for_tiny_calibration_sets() -> None:
    """With fewer than 1/alpha points the guarantee forces every class in."""
    proba = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]])
    y = np.array(["CONFIRMED", "CANDIDATE"])
    conformal = SplitConformal(alpha=0.10, mode="marginal", classes=CLASSES).fit(proba, y)
    assert conformal.threshold == float("inf")
    assert len(conformal.predict_sets(proba)[0]) == 3


def test_conformal_coverage_holds_on_average() -> None:
    rng = np.random.default_rng(0)
    n = 3000
    y = rng.choice(CLASSES, size=n)
    proba = rng.dirichlet(np.ones(3) * 0.5, size=n)
    conformal = SplitConformal(alpha=0.10, mode="marginal", classes=CLASSES).fit(proba, y)
    coverage = conformal.coverage(proba, y)
    assert coverage >= 0.85


def test_mondrian_mode_fits_per_class_thresholds() -> None:
    proba, y = _calibration_data()
    conformal = SplitConformal(alpha=0.10, mode="mondrian", classes=CLASSES).fit(proba, y)
    assert set(conformal.class_thresholds) == set(CLASSES)
    summary = conformal.summary(proba, y)
    assert summary["mode"] == "mondrian"
    assert summary["mean_set_size"] > 0


def test_conformal_rejects_unknown_mode() -> None:
    proba, y = _calibration_data()
    with pytest.raises(ValueError):
        SplitConformal(mode="nope", classes=CLASSES).fit(proba, y)


def test_conformal_summary_reports_per_class_coverage() -> None:
    proba, y = _calibration_data()
    conformal = SplitConformal(alpha=0.10, mode="marginal", classes=CLASSES).fit(proba, y)
    summary = conformal.summary(proba, y)
    assert set(summary["per_class_coverage"]) == set(CLASSES)
    for value in summary["per_class_coverage"].values():
        assert 0.0 <= value <= 1.0


def test_conformal_per_class_coverage_is_nan_for_absent_classes() -> None:
    proba = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])
    y = np.array(["CONFIRMED", "CONFIRMED"])
    conformal = SplitConformal(alpha=0.10, mode="marginal", classes=CLASSES).fit(proba, y)
    summary = conformal.summary(proba, y)
    assert np.isnan(summary["per_class_coverage"]["CANDIDATE"])


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def test_log_loss_is_order_safe() -> None:
    """A confident correct prediction must have a small log loss."""
    proba = np.array([[0.9, 0.05, 0.05]])
    y = np.array(["CONFIRMED"])
    assert multiclass_log_loss(y, proba, CLASSES) < 0.2
    # The same probability read as a different class is much worse.
    assert multiclass_log_loss(np.array(["CANDIDATE"]), proba, CLASSES) > 2.0


def test_log_loss_matches_manual_computation() -> None:
    proba, y = _calibration_data()
    expected = float(-np.mean(np.log(proba[np.arange(len(y)), [CLASSES.index(v) for v in y]])))
    assert multiclass_log_loss(y, proba, CLASSES) == pytest.approx(expected)


def test_perfect_predictions_score_well() -> None:
    y = np.array(["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"])
    proba = np.eye(3)
    assert macro_f1(y, proba, CLASSES) == pytest.approx(1.0)
    assert brier_score(y, proba, CLASSES) == pytest.approx(0.0)
    assert expected_calibration_error(y, proba, CLASSES) == pytest.approx(0.0)


def test_per_class_recall_uses_argmax() -> None:
    y = np.array(["CONFIRMED", "CONFIRMED", "CANDIDATE"])
    proba = np.array([[0.9, 0.05, 0.05], [0.3, 0.4, 0.3], [0.2, 0.7, 0.1]])
    recalls = per_class_recall(y, proba, CLASSES)
    # Row 1 is predicted CANDIDATE, so only one CONFIRMED is recovered.
    assert recalls["CONFIRMED"] == pytest.approx(0.5)
    assert recalls["CANDIDATE"] == pytest.approx(1.0)


def test_recall_at_precision_never_exceeds_one() -> None:
    proba, y = _calibration_data()
    recalls = recall_at_precision(y, proba, CLASSES, 0.99)
    for value in recalls.values():
        assert 0.0 <= value <= 1.0


def test_reliability_table_bins_sum_to_total() -> None:
    proba, y = _calibration_data()
    table = reliability_table(y, proba, CLASSES, n_bins=10)
    assert sum(row["count"] for row in table) == len(y)


def test_evaluate_returns_full_bundle() -> None:
    proba, y = _calibration_data()
    bundle = evaluate(y, proba, CLASSES)
    for key in (
        "macro_f1",
        "per_class_recall",
        "macro_pr_auc",
        "recall_at_precision",
        "log_loss",
        "brier_score",
        "expected_calibration_error",
    ):
        assert key in bundle
    assert bundle["n_samples"] == len(y)


# ----------------------------------------------------------------------
# reports
# ----------------------------------------------------------------------
def _shap_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"feature": "koi_period", "value": 12.4, "shap_value": 0.31},
            {"feature": "koi_duration", "value": 3.1, "shap_value": -0.12},
            {"feature": "koi_depth", "value": 800.0, "shap_value": 0.08},
        ]
    )


def test_report_mentions_only_top_features() -> None:
    proba = np.array([0.8, 0.15, 0.05])
    report = build_report(_shap_frame(), proba, CLASSES, top_k=2)
    assert report["predicted_class"] == "CONFIRMED"
    text = report["text"]
    assert "orbital period" in text
    assert "transit depth" not in text  # not in the top-2
    violations = check_report_features(
        text, feature_list(False), [f["feature"] for f in report["top_features"]]
    )
    assert violations == []


def test_report_flags_review_for_ambiguous_sets() -> None:
    proba = np.array([0.45, 0.4, 0.15])
    report = build_report(_shap_frame(), proba, CLASSES, prediction_set=["CONFIRMED", "CANDIDATE"])
    assert report["needs_review"] is True
    assert "manual review" in report["text"]


def test_report_does_not_mention_review_for_singleton() -> None:
    proba = np.array([0.9, 0.05, 0.05])
    report = build_report(_shap_frame(), proba, CLASSES, prediction_set=["CONFIRMED"])
    assert report["needs_review"] is False
    assert "manual review" not in report["text"]


def test_checker_catches_a_smuggled_feature() -> None:
    text = "The orbital period matters and so does the transit depth."
    violations = check_report_features(text, feature_list(False), ["koi_period"])
    assert "koi_depth" in violations


def test_display_names_are_distinct() -> None:
    names = [display_name(f) for f in feature_list(False)]
    assert len(names) == len(set(names))


# ----------------------------------------------------------------------
# end to end
# ----------------------------------------------------------------------
def _synthetic_koi_frame(n_groups: int = 60) -> pd.DataFrame:
    """A small frame with the columns the pipeline expects."""
    rng = np.random.default_rng(0)
    features = feature_list(False)
    rows = []
    for group in range(n_groups):
        label = CLASSES[group % 3]
        for _member in range(2):
            row = {f: float(rng.uniform(0.5, 2.0)) for f in features}
            row["kepid"] = group
            row["koi_disposition"] = label
            rows.append(row)
    return pd.DataFrame(rows)


def test_pipeline_fits_predicts_and_round_trips(tmp_path) -> None:
    from exovet.pipeline import VettingPipeline

    df = _synthetic_koi_frame()
    pipeline = VettingPipeline(
        model_name="random_forest",
        params={"n_estimators": 20},
        seed=42,
    )
    pipeline.fit(df, n_splits=3, test_fold=0, calibration_fold=1)
    assert "calibrated" in pipeline.metrics
    assert pipeline.metrics["split_sizes"]["test"] > 0

    row = df.iloc[0]
    result = pipeline.predict_one({f: float(row[f]) for f in pipeline.features})
    assert result.predicted_class in CLASSES
    assert 0.0 <= result.probability <= 1.0
    assert result.prediction_set
    assert isinstance(result.report, str) and result.report
    assert len(result.top_features) <= pipeline.top_k

    path = pipeline.save(tmp_path / "model.joblib")
    reloaded = VettingPipeline.load(path)
    again = reloaded.predict_one({f: float(row[f]) for f in pipeline.features})
    assert again.predicted_class == result.predicted_class
    assert again.probability == pytest.approx(result.probability)


def test_pipeline_report_never_mentions_a_feature_outside_the_top_k() -> None:
    from exovet.pipeline import VettingPipeline

    df = _synthetic_koi_frame()
    pipeline = VettingPipeline(
        model_name="random_forest", params={"n_estimators": 20}, seed=42, top_k=3
    )
    pipeline.fit(df, n_splits=3, test_fold=0, calibration_fold=1)
    for position in range(5):
        row = df.iloc[position]
        result = pipeline.predict_one({f: float(row[f]) for f in pipeline.features})
        allowed = [entry["feature"] for entry in result.top_features]
        violations = check_report_features(result.report, pipeline.features, allowed)
        assert violations == []


def test_pipeline_probabilities_sum_to_one() -> None:
    from exovet.pipeline import VettingPipeline

    df = _synthetic_koi_frame()
    pipeline = VettingPipeline(model_name="random_forest", params={"n_estimators": 20}, seed=42)
    pipeline.fit(df, n_splits=3, test_fold=0, calibration_fold=1)
    result = pipeline.predict_one({f: float(df.iloc[0][f]) for f in pipeline.features})
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_pipeline_metrics_include_the_reliability_table() -> None:
    """The reliability diagram is a figure; the numbers behind it are here."""
    from exovet.pipeline import VettingPipeline

    df = _synthetic_koi_frame()
    pipeline = VettingPipeline(model_name="random_forest", params={"n_estimators": 20}, seed=42)
    pipeline.fit(df, n_splits=3, test_fold=0, calibration_fold=1)
    reliability = pipeline.metrics["reliability"]
    assert set(reliability) == {"uncalibrated", "calibrated"}
    for table in reliability.values():
        assert len(table) == 10
        assert sum(row["count"] for row in table) == pipeline.metrics["split_sizes"]["test"]


def test_pipeline_metrics_include_per_class_conformal_coverage() -> None:
    from exovet.pipeline import VettingPipeline

    df = _synthetic_koi_frame()
    pipeline = VettingPipeline(model_name="random_forest", params={"n_estimators": 20}, seed=42)
    pipeline.fit(df, n_splits=3, test_fold=0, calibration_fold=1)
    coverage = pipeline.metrics["conformal"]["per_class_coverage"]
    assert set(coverage) == set(CLASSES)
