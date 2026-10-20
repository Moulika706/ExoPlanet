"""Tests for the publication figures (Phase 7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS_DIR = Path("results")
FIGURE_DIR = Path("paper/figures")

#: The figures the script is expected to produce.
FIGURE_NAMES = [
    "reliability",
    "conformal_coverage",
    "leakage_ablation",
    "shap_importance",
    "shap_stability",
    "cross_mission",
]

BUNDLES = [
    "metrics.json",
    "ablation_metrics.json",
    "shap_stability.json",
    "cross_mission_metrics.json",
]


def _have_bundles() -> bool:
    return all((RESULTS_DIR / name).exists() for name in BUNDLES)


pytestmark = pytest.mark.skipif(
    not _have_bundles(),
    reason="the results bundle is not built; run 'make phase2' and 'make cross' first",
)


def test_the_results_bundles_carry_what_the_figures_need() -> None:
    metrics = json.loads((RESULTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["best_model"]
    for row in metrics["results"]:
        assert set(row["reliability"]) == {"uncalibrated", "calibrated"}
        assert len(row["reliability"]["calibrated"]) == 10
    assert metrics["conformal_modes"]
    for mode in metrics["conformal_modes"]:
        assert set(mode["per_class_coverage"]) == {"CONFIRMED", "CANDIDATE", "FALSE POSITIVE"}

    ablation = json.loads((RESULTS_DIR / "ablation_metrics.json").read_text(encoding="utf-8"))
    strategies = {case["split_strategy"] for case in ablation["cases"]}
    assert strategies == {"grouped", "random", "sibling_leak"}
    for delta in ablation["deltas"]:
        assert "feature_leak_delta" in delta
        assert "random_split_delta" in delta
        assert "split_leak_delta" in delta

    stability = json.loads((RESULTS_DIR / "shap_stability.json").read_text(encoding="utf-8"))
    assert stability["stability"]["per_fold_top_k"]
    assert stability["consensus_ranking"]

    cross = json.loads((RESULTS_DIR / "cross_mission_metrics.json").read_text(encoding="utf-8"))
    for key in ("kepler", "zero_shot", "recalibrated_label_shift", "tess_in_domain"):
        assert key in cross
    assert cross["kepler"]["per_class_pr_auc"]


def test_every_figure_is_written(tmp_path, monkeypatch) -> None:
    import scripts.make_figures as figures  # noqa: PLC0415

    monkeypatch.setattr(figures, "FIGURE_DIR", tmp_path)
    figures.main()
    for name in FIGURE_NAMES:
        for suffix in ("png", "pdf"):
            path = tmp_path / f"{name}.{suffix}"
            assert path.exists(), f"{path} was not written"
            assert path.stat().st_size > 1000


def test_the_figure_script_fails_clearly_without_the_bundle(tmp_path, monkeypatch) -> None:
    import scripts.make_figures as figures  # noqa: PLC0415

    monkeypatch.setattr(figures, "RESULTS_DIR", tmp_path / "empty")
    with pytest.raises(SystemExit) as error:
        figures.load_json("metrics.json")
    assert "make phase2" in str(error.value)


def test_the_figure_script_can_be_imported_without_a_display() -> None:
    """matplotlib must be on the Agg backend, or the import fails headless."""
    import matplotlib  # noqa: PLC0415
    import scripts.make_figures as figures  # noqa: PLC0415

    assert matplotlib.get_backend().lower() == "agg"
    assert figures.FIGURE_DIR.name == "figures"
