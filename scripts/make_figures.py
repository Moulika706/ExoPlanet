"""Publication figures for the ExoVet paper (Phase 7).

Every figure is built from the JSON and CSV bundles in ``results/``, so
the figures can be regenerated without refitting a model or refetching
the data. The pipeline that produced the bundles is:

    make phase2   # training, ablation, SHAP stability
    make cross    # the cross-mission evaluation
    make figures  # this script

Each figure is written as a PNG for the slides and a PDF for the paper.

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS_DIR = Path("results")
FIGURE_DIR = Path("paper/figures")

#: One colour per class, used everywhere.
CLASS_COLOR = {
    "CONFIRMED": "#3fb950",
    "CANDIDATE": "#d29922",
    "FALSE POSITIVE": "#f85149",
}
CLASS_LABEL = {
    "CONFIRMED": "Confirmed",
    "CANDIDATE": "Candidate",
    "FALSE POSITIVE": "False positive",
}
MODEL_COLOR = {
    "logistic_regression": "#8b949e",
    "random_forest": "#58a6ff",
    "xgboost": "#a371f7",
}
MODEL_LABEL = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}
INK = "#1f2328"
MUTED = "#6e7781"
GRID = "#d8dee4"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.bbox": "tight",
    }
)


def load_json(name: str) -> Dict[str, Any]:
    """Read a JSON bundle from the results directory."""
    path = RESULTS_DIR / name
    if not path.exists():
        raise SystemExit(
            f"{path} not found; run the pipeline first "
            "(make phase2, make cross) and then make figures"
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(name: str) -> pd.DataFrame:
    """Read a CSV bundle from the results directory."""
    path = RESULTS_DIR / name
    if not path.exists():
        raise SystemExit(f"{path} not found; run the pipeline first (make phase2)")
    return pd.read_csv(path)


def tidy(ax, spines: tuple[str, ...] = ("top", "right")) -> None:
    """Remove the spines that carry no information."""
    for spine in spines:
        ax.spines[spine].set_visible(False)


def save(fig: plt.Figure, name: str) -> None:
    """Write the figure as PNG and PDF."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = FIGURE_DIR / f"{name}.{suffix}"
        fig.savefig(path)
        print(f"wrote {path}")


# ----------------------------------------------------------------------
# 1. reliability diagram
# ----------------------------------------------------------------------
def figure_reliability(metrics: Dict[str, Any]) -> None:
    """Calibration curve of the best model, before and after calibration."""
    best = metrics["best_model"]
    row = next(r for r in metrics["results"] if r["model"] == best)
    table = pd.DataFrame(row["reliability"]["calibrated"])
    raw = pd.DataFrame(row["reliability"]["uncalibrated"])
    accent = MODEL_COLOR.get(best, "#58a6ff")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.35, 1]})

    ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1, label="Perfect calibration")
    for frame, label, marker, color in (
        (raw, "Uncalibrated", "o", "#8b949e"),
        (table, "Isotonic calibration", "s", accent),
    ):
        frame = frame[frame["count"] > 0]
        ax.plot(
            frame["mean_confidence"],
            frame["accuracy"],
            marker=marker,
            markersize=4.5,
            linewidth=1.6,
            color=color,
            label=label,
        )

    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"Reliability, {MODEL_LABEL.get(best, best)}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    tidy(ax)

    # The population in each confidence bin, before and after calibration.
    # The two distributions differ, which is why calibration moves the
    # curve at all.
    bins = np.linspace(0, 1, len(table) + 1)
    centres = bins[:-1] + np.diff(bins) / 2
    width = np.diff(bins) * 0.4
    ax2.bar(centres - width / 2, raw["count"], width=width, color="#c9d1d9", label="Uncalibrated")
    ax2.bar(centres + width / 2, table["count"], width=width, color=accent, label="Calibrated")
    ax2.set_xlabel("Confidence bin")
    ax2.set_ylabel("Number of test objects")
    ax2.set_title("Population per bin")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8, loc="upper left")
    tidy(ax2)

    fig.suptitle(
        "Expected calibration error "
        f"{row['calibrated']['expected_calibration_error']:.4f} after calibration, "
        f"{row['uncalibrated']['expected_calibration_error']:.4f} before",
        fontsize=9,
        y=1.03,
    )
    fig.tight_layout()
    save(fig, "reliability")


# ----------------------------------------------------------------------
# 2. conformal coverage
# ----------------------------------------------------------------------
def figure_conformal(metrics: Dict[str, Any]) -> None:
    """Per-class coverage of the two conformal modes, against the target."""
    modes = metrics["conformal_modes"]
    classes = list(modes[0]["per_class_coverage"].keys())
    target = modes[0]["target_coverage"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    width = 0.35
    positions = np.arange(len(classes))
    for offset, mode in enumerate(modes):
        values = [mode["per_class_coverage"].get(cls, np.nan) for cls in classes]
        ax.bar(
            positions + (offset - 0.5) * width,
            values,
            width=width,
            color="#58a6ff" if mode["mode"] == "marginal" else "#a371f7",
            label=mode["mode"].capitalize(),
        )
    ax.axhline(target, color="#f85149", linestyle="--", linewidth=1.2, label=f"Target {target:.2f}")
    ax.set_xticks(positions)
    ax.set_xticklabels([CLASS_LABEL.get(cls, cls) for cls in classes])
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title("Coverage per class")
    ax.legend(fontsize=8, loc="lower left")
    tidy(ax)

    for offset, mode in enumerate(modes):
        ax2.bar(
            offset,
            mode["mean_set_size"],
            width=0.5,
            color="#58a6ff" if mode["mode"] == "marginal" else "#a371f7",
        )
        ax2.text(
            offset,
            mode["mean_set_size"] + 0.04,
            f"{mode['singleton_rate']:.0%} singleton",
            ha="center",
            fontsize=8,
            color=MUTED,
        )
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([mode["mode"].capitalize() for mode in modes])
    ax2.set_ylabel("Mean prediction set size")
    ax2.set_ylim(0, max(mode["mean_set_size"] for mode in modes) * 1.35)
    ax2.set_title("Cost of the guarantee")
    tidy(ax2)

    fig.tight_layout()
    save(fig, "conformal_coverage")


# ----------------------------------------------------------------------
# 3. leakage ablation
# ----------------------------------------------------------------------
def figure_ablation(ablation: Dict[str, Any]) -> None:
    """How much the vetting-output columns and the split strategy are worth."""
    models = [row["model"] for row in ablation["deltas"]]

    def pick(model: str, leakage: bool, strategy: str) -> float:
        for case in ablation["cases"]:
            if (
                case["model"] == model
                and case["include_leakage"] == leakage
                and case["split_strategy"] == strategy
            ):
                return case["calibrated"]["macro_f1"]
        raise KeyError((model, leakage, strategy))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.2))

    settings = [
        ("A, grouped", lambda m: pick(m, False, "grouped"), "#3fb950"),
        ("A, random split", lambda m: pick(m, False, "random"), "#8b949e"),
        ("A, sibling leak", lambda m: pick(m, False, "sibling_leak"), "#d29922"),
        ("A+B, grouped", lambda m: pick(m, True, "grouped"), "#f85149"),
    ]
    width = 0.2
    positions = np.arange(len(models))
    for index, (label, getter, color) in enumerate(settings):
        values = [getter(model) for model in models]
        bars = ax.bar(
            positions + (index - 1.5) * width,
            values,
            width=width,
            color=color,
            label=label,
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.2f}",
                ha="center",
                fontsize=6.5,
                color=MUTED,
            )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [MODEL_LABEL.get(model, model).replace(" ", "\n") for model in models], fontsize=8
    )
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("Leakage ablation, test fold")
    ax.legend(
        fontsize=7.5,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        columnspacing=1.2,
        handlelength=1.2,
    )
    tidy(ax)

    labels = [
        "Feature leak\n(A+B minus A)",
        "Random split\n(minus grouped)",
        "Sibling leak\n(minus grouped)",
    ]
    keys = ["feature_leak_delta", "random_split_delta", "split_leak_delta"]
    for offset, model in enumerate(models):
        values = [next(r for r in ablation["deltas"] if r["model"] == model)[key] for key in keys]
        ax2.barh(
            np.arange(len(labels)) + (offset - 1) * 0.26,
            values,
            height=0.24,
            color=MODEL_COLOR.get(model, "#8b949e"),
            label=MODEL_LABEL.get(model, model),
        )
    ax2.axvline(0, color=INK, linewidth=1)
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("Change in macro-F1")
    ax2.set_title("Size of each leak")
    ax2.legend(fontsize=7.5, loc="upper right")
    tidy(ax2)

    fig.tight_layout()
    save(fig, "leakage_ablation")


# ----------------------------------------------------------------------
# 4. SHAP importance
# ----------------------------------------------------------------------
def figure_shap_importance() -> None:
    """The global SHAP importance of the best model."""
    importance = load_csv("shap_global_importance.csv").sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.barh(
        importance["feature"],
        importance["mean_abs_shap"],
        color="#58a6ff",
        alpha=0.9,
        height=0.68,
    )
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("What the model uses")
    ax.grid(axis="y", alpha=0)
    tidy(ax)
    fig.tight_layout()
    save(fig, "shap_importance")


# ----------------------------------------------------------------------
# 5. SHAP stability across folds
# ----------------------------------------------------------------------
def figure_shap_stability(stability: Dict[str, Any]) -> None:
    """The feature ranking in every fold, and the overlap of the top-k sets."""
    folds = load_csv("shap_fold_importance.csv")
    n_folds = int(folds["fold"].nunique())
    top_k = int(stability["top_k"])

    consensus = sorted(stability["consensus_ranking"], key=lambda row: row["consensus_rank"])
    features = [row["feature"] for row in consensus][:top_k]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1.5, 1]})

    for feature in features:
        ranks = [
            int(folds[(folds["fold"] == fold) & (folds["feature"] == feature)]["rank"].iloc[0])
            for fold in range(n_folds)
        ]
        ax.plot(
            range(n_folds),
            ranks,
            marker="o",
            markersize=4,
            linewidth=1.3,
            label=feature,
        )
    ax.invert_yaxis()
    ax.set_xticks(range(n_folds))
    ax.set_xlabel("Fold")
    ax.set_ylabel("Rank of the feature")
    ax.set_title(f"Feature ranking in each of the {n_folds} folds")
    ax.set_yticks(range(1, top_k + 1))
    ax.legend(
        fontsize=7.5,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        columnspacing=1.2,
        handlelength=1.2,
    )
    tidy(ax)

    per_fold = stability["stability"]["per_fold_top_k"]
    overlap = []
    keys = sorted(per_fold)
    for left in range(len(keys)):
        for right in range(left + 1, len(keys)):
            a = set(per_fold[keys[left]])
            b = set(per_fold[keys[right]])
            union = a | b
            overlap.append(len(a & b) / len(union) if union else 1.0)
    ax2.bar(range(len(overlap)), overlap, color="#3fb950", alpha=0.85, width=0.7)
    ax2.set_ylim(0, 1.18)
    ax2.set_xlabel("Pair of folds")
    ax2.set_ylabel(f"Jaccard overlap of the top-{top_k}")
    ax2.set_title(
        f"Agreement between folds\nmean {np.mean(overlap):.3f} over {len(overlap)} pairs",
        fontsize=9,
    )
    tidy(ax2)

    fig.tight_layout()
    save(fig, "shap_stability")


# ----------------------------------------------------------------------
# 6. cross-mission transfer
# ----------------------------------------------------------------------
def figure_cross_mission(cross: Dict[str, Any]) -> None:
    """Kepler to TESS: the drop, and where it comes from."""
    settings = [
        ("Kepler\n(in domain)", "kepler", "#3fb950"),
        ("TESS\nzero shot", "zero_shot", "#f85149"),
        ("TESS\nlabel shift", "recalibrated_label_shift", "#d29922"),
        ("TESS\nin domain", "tess_in_domain", "#58a6ff"),
    ]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2))

    values = [cross[key]["metrics"]["macro_f1"] for _, key, _ in settings]
    bars = ax.bar(
        range(len(settings)),
        values,
        color=[color for _, _, color in settings],
        width=0.62,
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.3f}",
            ha="center",
            fontsize=8,
            color=MUTED,
        )
    ax.set_xticks(range(len(settings)))
    ax.set_xticklabels([label for label, _, _ in settings], fontsize=8)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 0.8)
    ax.set_title("The model does not transfer")
    tidy(ax)

    classes = list(cross["kepler"]["per_class_pr_auc"].keys())
    positions = np.arange(len(classes))
    # Only the settings that carry a per-class PR-AUC can go in this
    # panel; the label-shift variant reports macro metrics only.
    series = [
        (label, key, color)
        for label, key, color in settings
        if cross.get(key, {}).get("per_class_pr_auc")
    ]
    width = 0.8 / max(len(series), 1)
    for index, (label, key, color) in enumerate(series):
        pr_auc = cross[key]["per_class_pr_auc"]
        ax2.bar(
            positions + (index - (len(series) - 1) / 2) * width,
            [pr_auc.get(cls, np.nan) for cls in classes],
            width=width,
            color=color,
            label=label.replace("\n", " "),
        )
    ax2.set_xticks(positions)
    ax2.set_xticklabels([CLASS_LABEL.get(cls, cls) for cls in classes], fontsize=8)
    ax2.set_ylabel("One-vs-rest PR-AUC")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Where the ranking collapses")
    ax2.legend(
        fontsize=7,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        columnspacing=1.0,
        handlelength=1.1,
    )
    tidy(ax2)

    fig.tight_layout()
    save(fig, "cross_mission")


def main() -> None:
    print(f"writing figures to {FIGURE_DIR}")
    metrics = load_json("metrics.json")
    figure_reliability(metrics)
    figure_conformal(metrics)
    figure_ablation(load_json("ablation_metrics.json"))
    figure_shap_importance()
    figure_shap_stability(load_json("shap_stability.json"))
    figure_cross_mission(load_json("cross_mission_metrics.json"))
    print("done")


if __name__ == "__main__":
    main()
