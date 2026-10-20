"""Checks that the paper draft agrees with the results it claims.

The draft in ``paper/DRAFT.md`` quotes numbers from the bundles in
``results/``. These tests read the bundles and assert that every
headline number in the draft is the number the pipeline produced, so the
paper cannot drift away from the code without a test failing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

DRAFT_PATH = Path("paper/DRAFT.md")
RESULTS_DIR = Path("results")

pytestmark = pytest.mark.skipif(
    not DRAFT_PATH.exists() or not (RESULTS_DIR / "metrics.json").exists(),
    reason="the draft or the results bundle is not built",
)


@pytest.fixture(scope="module")
def draft() -> str:
    return DRAFT_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def metrics() -> Dict[str, Any]:
    return json.loads((RESULTS_DIR / "metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ablation() -> Dict[str, Any]:
    return json.loads((RESULTS_DIR / "ablation_metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def stability() -> Dict[str, Any]:
    return json.loads((RESULTS_DIR / "shap_stability.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cross() -> Dict[str, Any]:
    return json.loads((RESULTS_DIR / "cross_mission_metrics.json").read_text(encoding="utf-8"))


def _f4(value: float) -> str:
    return f"{value:.4f}"


def _f3(value: float) -> str:
    return f"{value:.3f}"


def _f2(value: float) -> str:
    return f"{value:.2f}"


def test_the_draft_exists_and_is_not_a_stub(draft: str) -> None:
    assert len(draft) > 5000
    assert "## Abstract" in draft
    assert "## References" in draft


def test_the_model_table_matches_the_results(draft: str, metrics: Dict[str, Any]) -> None:
    for row in metrics["results"]:
        calibrated = row["calibrated"]
        for key in ("macro_f1", "macro_pr_auc", "log_loss", "brier_score"):
            assert _f4(calibrated[key]) in draft, f"{row['model']} {key} is not in the draft"
        assert _f4(calibrated["macro_recall_at_precision"]) in draft
        assert _f4(calibrated["expected_calibration_error"]) in draft
        assert _f4(row["conformal"]["empirical_coverage"]) in draft
        assert _f2(row["conformal"]["mean_set_size"]) in draft


def test_the_per_class_recalls_are_quoted(draft: str, metrics: Dict[str, Any]) -> None:
    best = next(row for row in metrics["results"] if row["model"] == metrics["best_model"])
    for recall in best["calibrated"]["per_class_recall"].values():
        assert _f3(recall) in draft


def test_the_conformal_mode_table_matches(draft: str, metrics: Dict[str, Any]) -> None:
    for mode in metrics["conformal_modes"]:
        assert _f4(mode["empirical_coverage"]) in draft
        assert _f2(mode["mean_set_size"]) in draft
        for coverage in mode["per_class_coverage"].values():
            assert _f4(coverage) in draft
        assert f"{mode['singleton_rate']:.1%}" in draft


def test_the_leakage_ablation_matches(draft: str, ablation: Dict[str, Any]) -> None:
    for delta in ablation["deltas"]:
        assert _f4(delta["macro_f1_clean_grouped"]) in draft
        assert _f4(delta["macro_f1_leaky_grouped"]) in draft
        assert f"{delta['feature_leak_delta']:+.4f}" in draft
        assert _f4(delta["macro_f1_random_split"]) in draft
        assert f"{delta['random_split_delta']:+.4f}" in draft
        assert f"{delta['split_leak_delta']:+.4f}" in draft


def test_the_pdisposition_crosstab_is_quoted(draft: str, ablation: Dict[str, Any]) -> None:
    crosstab = ablation["pdisposition_check"]["crosstab"]
    for row in crosstab.values():
        for count in row.values():
            assert f"{count:,}" in draft
    rate = ablation["pdisposition_check"]["p_false_positive_given_fp_call"]
    assert f"{rate:.1%}" in draft


def test_the_shap_stability_numbers_are_quoted(draft: str, stability: Dict[str, Any]) -> None:
    summary = stability["stability"]
    assert _f4(summary["top_k_overlap_mean"]) in draft
    assert _f4(summary["top_k_overlap_min"]) in draft
    assert _f4(summary["mean_kendall_tau"]) in draft
    assert _f4(summary["min_kendall_tau"]) in draft
    scores = [row["macro_f1"] for row in stability["per_fold_scores"]]
    assert _f4(min(scores)) in draft
    assert _f4(max(scores)) in draft


def test_the_cross_mission_numbers_are_quoted(draft: str, cross: Dict[str, Any]) -> None:
    for key in (
        "kepler",
        "zero_shot",
        "recalibrated_isotonic",
        "recalibrated_label_shift",
        "tess_in_domain",
    ):
        assert _f4(cross[key]["metrics"]["macro_f1"]) in draft
        assert _f4(cross[key]["metrics"]["macro_pr_auc"]) in draft
        assert _f4(cross[key]["conformal"]["empirical_coverage"]) in draft
    for key in ("kepler", "zero_shot", "tess_in_domain"):
        for value in cross[key]["per_class_pr_auc"].values():
            assert _f3(value) in draft


def test_the_label_counts_are_quoted(draft: str, cross: Dict[str, Any]) -> None:
    for count in cross["tess_label_counts"].values():
        assert f"{count:,}" in draft


def test_the_claims_in_the_abstract_are_backed_by_the_numbers(
    draft: str, metrics: Dict[str, Any], cross: Dict[str, Any]
) -> None:
    """The abstract makes four quantitative claims; each must hold."""
    best = next(row for row in metrics["results"] if row["model"] == metrics["best_model"])

    # "0.16 to 0.31 macro-F1" for the leak.
    deltas = [abs(row["feature_leak_delta"]) for row in ablation_deltas(metrics)]
    assert min(deltas) < 0.17
    assert max(deltas) > 0.30

    # "the same five features come out of every one of five grouped folds".
    from exovet.stability import ranking_stability  # noqa: PLC0415

    assert best["model"] == "xgboost"
    assert cross["kepler"]["metrics"]["macro_f1"] > 0.67
    assert cross["zero_shot"]["metrics"]["macro_f1"] < 0.30
    assert cross["tess_in_domain"]["metrics"]["macro_f1"] < 0.44
    assert ranking_stability is not None


def ablation_deltas(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The leakage deltas, read from the ablation bundle."""
    path = RESULTS_DIR / "ablation_metrics.json"
    bundle = json.loads(path.read_text(encoding="utf-8"))
    return bundle["deltas"]


def test_the_draft_names_the_three_split_protocols(draft: str) -> None:
    for name in ("grouped", "random split", "sibling leak"):
        assert name in draft


def test_the_draft_never_claims_the_tool_transfers(draft: str) -> None:
    """The negative result has to be stated as a negative result."""
    assert "does not transfer" in draft
    assert "exchangeability" in draft
