"""Tests for the cross-fold SHAP stability analysis (Phase 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exovet.explain import source_feature_map
from exovet.features import feature_list
from exovet.stability import cross_fold_analysis, ranking_stability, stability_markdown

CLASSES = ("CONFIRMED", "CANDIDATE", "FALSE POSITIVE")


def _importance_frame(rankings: dict[int, list[str]], values: dict[int, list[float]]):
    """Build an importance frame from per-fold rankings and values."""
    rows = []
    for fold, order in rankings.items():
        for rank, feature in enumerate(order, start=1):
            rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "mean_abs_shap": values[fold][rank - 1],
                    "rank": rank,
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# source feature mapping
# ----------------------------------------------------------------------
def test_source_feature_map_folds_one_hot_columns_back() -> None:
    names = ["koi_period", "koi_pdisposition_CANDIDATE", "koi_pdisposition_FALSE POSITIVE"]
    mapping = source_feature_map(names, feature_list(True))
    assert mapping["koi_period"] == "koi_period"
    assert mapping["koi_pdisposition_CANDIDATE"] == "koi_pdisposition"
    assert mapping["koi_pdisposition_FALSE POSITIVE"] == "koi_pdisposition"


def test_source_feature_map_keeps_unknown_columns() -> None:
    mapping = source_feature_map(["koi_something_new"], feature_list(False))
    assert mapping["koi_something_new"] == "koi_something_new"


# ----------------------------------------------------------------------
# ranking stability
# ----------------------------------------------------------------------
def test_ranking_stability_is_one_for_identical_rankings() -> None:
    order = ["koi_period", "koi_depth", "koi_prad", "koi_duration", "koi_kepmag"]
    values = {0: [0.5, 0.4, 0.3, 0.2, 0.1], 1: [0.5, 0.4, 0.3, 0.2, 0.1]}
    stability = ranking_stability(_importance_frame({0: order, 1: order}, values), top_k=3)
    assert stability["top_k_overlap_mean"] == pytest.approx(1.0)
    assert stability["mean_kendall_tau"] == pytest.approx(1.0)
    assert stability["top_k_frequency"]["koi_period"] == 2


def test_ranking_stability_drops_when_rankings_disagree() -> None:
    order_a = ["koi_period", "koi_depth", "koi_prad", "koi_duration", "koi_kepmag"]
    order_b = ["koi_kepmag", "koi_duration", "koi_prad", "koi_depth", "koi_period"]
    values = {0: [0.5, 0.4, 0.3, 0.2, 0.1], 1: [0.5, 0.4, 0.3, 0.2, 0.1]}
    stability = ranking_stability(_importance_frame({0: order_a, 1: order_b}, values), top_k=3)
    # The top-3 sets share only the middle feature.
    assert stability["top_k_overlap_mean"] == pytest.approx(0.2)
    assert stability["mean_kendall_tau"] < 0.0
    assert stability["top_k_frequency"]["koi_prad"] == 2
    assert stability["top_k_frequency"]["koi_period"] == 1
    assert stability["top_k_frequency"]["koi_kepmag"] == 1


def test_ranking_stability_reports_rank_spread() -> None:
    order = ["koi_period", "koi_depth", "koi_prad"]
    values = {0: [0.5, 0.4, 0.3], 1: [0.5, 0.4, 0.3]}
    stability = ranking_stability(_importance_frame({0: order, 1: order}, values), top_k=2)
    assert stability["rank_spread"]["koi_period"] == pytest.approx(0.0)
    assert stability["rank_spread"]["koi_prad"] == pytest.approx(0.0)


def test_ranking_stability_needs_two_folds() -> None:
    order = ["koi_period", "koi_depth"]
    values = {0: [0.5, 0.4]}
    with pytest.raises(ValueError):
        ranking_stability(_importance_frame({0: order}, values), top_k=1)


def test_ranking_stability_rejects_mismatched_feature_sets() -> None:
    frame = _importance_frame(
        {0: ["koi_period", "koi_depth"], 1: ["koi_teq", "koi_srad"]},
        {0: [0.5, 0.4], 1: [0.4, 0.5]},
    )
    with pytest.raises(ValueError):
        ranking_stability(frame, top_k=1)


# ----------------------------------------------------------------------
# end to end on a small synthetic table
# ----------------------------------------------------------------------
def _synthetic_koi_frame(n_groups: int = 40) -> pd.DataFrame:
    """A small frame whose features actually determine the label.

    The shift moves every feature in the direction of the class, so a
    model that reads the columns in the right order scores far above
    chance. That is what makes the per-fold macro-F1 a useful assertion:
    it catches a mis-aligned probability column.
    """
    rng = np.random.default_rng(0)
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


def test_cross_fold_analysis_produces_one_ranking_per_fold() -> None:
    df = _synthetic_koi_frame()
    analysis = cross_fold_analysis(
        df,
        model_name="random_forest",
        params={"n_estimators": 10},
        n_splits=3,
        max_rows=50,
    )
    importance = analysis["importance"]
    assert sorted(importance["fold"].unique()) == [0, 1, 2]
    assert len(analysis["scores"]) == 3
    for row in analysis["scores"]:
        assert 0.0 <= row["macro_f1"] <= 1.0
    # The same feature set is ranked in every fold.
    per_fold = [set(importance[importance["fold"] == f]["feature"]) for f in (0, 1, 2)]
    assert per_fold[0] == per_fold[1] == per_fold[2]


def test_cross_fold_analysis_scores_the_right_column() -> None:
    """The probability columns must be read in the canonical class order.

    The label encoder inside the pipeline orders the classes
    alphabetically. Scoring against the wrong column order scores the
    model against the wrong labels, and the macro-F1 collapses to chance.
    """
    df = _synthetic_koi_frame()
    analysis = cross_fold_analysis(
        df,
        model_name="random_forest",
        params={"n_estimators": 10},
        n_splits=3,
        max_rows=50,
    )
    for row in analysis["scores"]:
        assert row["macro_f1"] > 0.6


def test_cross_fold_analysis_covers_every_row_once() -> None:
    df = _synthetic_koi_frame()
    analysis = cross_fold_analysis(
        df,
        model_name="random_forest",
        params={"n_estimators": 10},
        n_splits=3,
        max_rows=50,
    )
    scores = analysis["scores"]
    assert sum(row["n_test"] for row in scores) == len(df)
    for row in scores:
        assert row["n_test"] > 0
        assert row["n_train"] == len(df) - row["n_test"]


def test_stability_markdown_mentions_the_headline_numbers() -> None:
    df = _synthetic_koi_frame()
    analysis = cross_fold_analysis(
        df,
        model_name="random_forest",
        params={"n_estimators": 10},
        n_splits=3,
        max_rows=50,
    )
    stability = ranking_stability(analysis["importance"], top_k=3)
    text = stability_markdown(analysis, stability, top_k=3)
    assert "Kendall" in text
    assert "Consensus ranking" in text
    assert f"{stability['top_k_overlap_mean']:.4f}" in text
