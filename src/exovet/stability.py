"""Cross-fold stability of the SHAP feature ranking.

One global SHAP table on one test fold says what the model used on that
fold. It does not say whether the story survives a different split, and an
unstable explanation is worse than no explanation, because a user is shown
a reason that may not hold for the next object.

This module refits the model on each grouped fold in turn, computes the
global SHAP importance on the fold that was held out, and measures how far
the ranking moves between folds. The reported numbers are:

``top_k_overlap``
    Mean Jaccard overlap of the top-k feature sets over every pair of
    folds. 1.0 means all folds picked exactly the same features.

``mean_kendall_tau``
    Mean rank correlation between folds over the whole feature list.

``top_k_frequency``
    How many folds put a feature in their top-k.

``rank_spread``
    Standard deviation of a feature's rank across the folds.

The models fitted here have no calibration fold, so the per-fold scores are
uncalibrated; they are there to show that the *ranking* is stable, not to
replace the calibrated numbers in the training report.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .data import GROUP_COLUMN, LABELS, TARGET
from .eval import evaluate
from .explain import (
    compute_shap_values,
    global_importance,
    source_feature_map,
    transformed_feature_names,
)
from .features import feature_list
from .models import build_model
from .splits import grouped_kfold


def cross_fold_analysis(
    df: pd.DataFrame,
    model_name: str = "xgboost",
    params: Optional[Dict[str, Any]] = None,
    features: Optional[Sequence[str]] = None,
    include_leakage: bool = False,
    n_splits: int = 5,
    seed: int = 42,
    group_column: str = GROUP_COLUMN,
    target_column: str = TARGET,
    max_rows: int = 1500,
    classes: Sequence[str] = LABELS,
) -> Dict[str, Any]:
    """Refit the model on every grouped fold and collect the SHAP ranking.

    Fold *f* is held out, the model is trained on the remaining folds
    (host stars never straddle the split) and the global SHAP importance is
    computed on the held-out rows. The result is one ranking per fold, so
    the rankings can be compared with each other.
    """
    feature_columns = list(features) if features is not None else feature_list(include_leakage)
    folds = grouped_kfold(
        df,
        n_splits=n_splits,
        group_column=group_column,
        target_column=target_column,
        seed=seed,
    )

    importance_frames: List[pd.DataFrame] = []
    scores: List[Dict[str, Any]] = []

    for fold in folds:
        X_train = df.iloc[fold.train_idx][feature_columns]
        y_train = df.iloc[fold.train_idx][target_column].to_numpy()
        X_test = df.iloc[fold.test_idx][feature_columns]
        y_test = df.iloc[fold.test_idx][target_column].to_numpy()

        model = build_model(
            model_name,
            params=params,
            seed=seed,
            feature_names=feature_columns,
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)
        # The label encoder inside the pipeline orders the classes
        # alphabetically (CANDIDATE, CONFIRMED, FALSE POSITIVE), which is
        # not the canonical order used by the metrics. Reading the columns
        # in the wrong order scores the model against the wrong labels.
        encoder_classes = list(model.named_steps["classifier"].encoder_.classes_)
        order = [encoder_classes.index(cls) for cls in classes]
        proba = proba[:, order]
        metrics = evaluate(y_test, proba, classes)
        scores.append(
            {
                "fold": int(fold.index),
                "n_train": int(len(fold.train_idx)),
                "n_test": int(len(fold.test_idx)),
                "macro_f1": metrics["macro_f1"],
                "log_loss": metrics["log_loss"],
                "per_class_recall": metrics["per_class_recall"],
            }
        )

        if max_rows and len(X_test) > max_rows:
            rng = np.random.default_rng(seed + int(fold.index))
            pick = np.sort(rng.choice(len(X_test), size=max_rows, replace=False))
            X_test = X_test.iloc[pick]

        shap_values = compute_shap_values(model, X_test, classes, seed=seed)
        names = transformed_feature_names(model)
        importance = global_importance(shap_values, names)
        mapping = source_feature_map(names, feature_columns)
        importance["feature"] = importance["feature"].map(mapping)
        # One-hot dummies of the same source column are summed together.
        importance = (
            importance.groupby("feature", as_index=False)["mean_abs_shap"]
            .sum()
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        importance["rank"] = np.arange(1, len(importance) + 1)
        importance["fold"] = int(fold.index)
        importance_frames.append(importance[["fold", "feature", "mean_abs_shap", "rank"]])

    return {
        "model": model_name,
        "n_splits": n_splits,
        "seed": seed,
        "features": feature_columns,
        "importance": pd.concat(importance_frames, ignore_index=True),
        "scores": scores,
    }


def ranking_stability(importance: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
    """Summarise how stable the SHAP ranking is across folds.

    *importance* is the frame returned by :func:`cross_fold_analysis`: one
    row per (fold, feature) with the mean absolute SHAP value and the rank
    inside that fold.
    """
    from scipy.stats import kendalltau

    folds = sorted(int(fold) for fold in importance["fold"].unique())
    if len(folds) < 2:
        raise ValueError("at least two folds are needed to measure stability")

    tables: Dict[int, pd.DataFrame] = {}
    for fold in folds:
        sub = importance[importance["fold"] == fold].sort_values("rank")
        tables[fold] = sub.set_index("feature")

    features = list(tables[folds[0]].index)
    for fold in folds:
        if set(tables[fold].index) != set(features):
            raise ValueError(f"fold {fold} ranks a different feature set")

    overlaps: List[float] = []
    taus: List[float] = []
    for position, left in enumerate(folds):
        for right in folds[position + 1 :]:
            top_left = set(tables[left].index[:top_k])
            top_right = set(tables[right].index[:top_k])
            union = top_left | top_right
            overlaps.append(len(top_left & top_right) / len(union) if union else 1.0)
            values_left = tables[left].loc[features, "mean_abs_shap"].to_numpy()
            values_right = tables[right].loc[features, "mean_abs_shap"].to_numpy()
            tau = kendalltau(values_left, values_right).statistic
            taus.append(0.0 if tau is None or np.isnan(tau) else float(tau))

    rank_matrix = pd.DataFrame({fold: tables[fold].loc[features, "rank"] for fold in folds})
    value_matrix = pd.DataFrame(
        {fold: tables[fold].loc[features, "mean_abs_shap"] for fold in folds}
    )

    top_k_frequency = {
        feature: int(sum(1 for fold in folds if feature in set(tables[fold].index[:top_k])))
        for feature in features
    }
    rank_spread = rank_matrix.std(axis=1, ddof=0)
    mean_importance = value_matrix.mean(axis=1)

    consensus = (
        pd.DataFrame(
            {
                "feature": features,
                "mean_abs_shap": [float(mean_importance[f]) for f in features],
                "mean_rank": [float(rank_matrix.loc[f].mean()) for f in features],
                "rank_spread": [float(rank_spread[f]) for f in features],
                "top_k_count": [top_k_frequency[f] for f in features],
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    consensus.insert(0, "consensus_rank", np.arange(1, len(consensus) + 1))

    return {
        "n_folds": len(folds),
        "top_k": top_k,
        "top_k_overlap_mean": float(np.mean(overlaps)),
        "top_k_overlap_min": float(np.min(overlaps)),
        "top_k_overlap_pairs": [
            {
                "folds": [int(left), int(right)],
                "jaccard": float(overlap),
            }
            for (left, right), overlap in zip(
                [
                    (left, right)
                    for position, left in enumerate(folds)
                    for right in folds[position + 1 :]
                ],
                overlaps,
                strict=True,
            )
        ],
        "mean_kendall_tau": float(np.mean(taus)),
        "min_kendall_tau": float(np.min(taus)),
        "top_k_frequency": top_k_frequency,
        "rank_spread": {f: float(rank_spread[f]) for f in features},
        "per_fold_top_k": {
            int(fold): [str(f) for f in tables[fold].index[:top_k]] for fold in folds
        },
        "consensus_ranking": consensus.to_dict("records"),
    }


def stability_markdown(
    analysis: Dict[str, Any],
    stability: Dict[str, Any],
    top_k: int = 5,
) -> str:
    """Render the stability analysis as a Markdown report."""
    scores = analysis["scores"]
    mean_f1 = float(np.mean([row["macro_f1"] for row in scores]))
    spread_f1 = float(np.std([row["macro_f1"] for row in scores]))

    lines = [
        "# ExoVet - SHAP stability across folds (Phase 2)",
        "",
        f"Model `{analysis['model']}`, {stability['n_folds']} grouped folds, "
        f"seed {analysis['seed']}. Every fold is held out in turn, the model is",
        "refitted on the remaining folds and the global SHAP importance is",
        "computed on the held-out rows, so the rankings are independent of",
        "each other.",
        "",
        "## Summary",
        "",
        f"- Mean Jaccard overlap of the top-{top_k} sets: "
        f"**{stability['top_k_overlap_mean']:.4f}** "
        f"(worst pair {stability['top_k_overlap_min']:.4f}).",
        f"- Mean Kendall rank correlation between folds: "
        f"**{stability['mean_kendall_tau']:.4f}** "
        f"(worst pair {stability['min_kendall_tau']:.4f}).",
        f"- Per-fold macro-F1: mean {mean_f1:.4f}, standard deviation {spread_f1:.4f}.",
        "",
        "## Per-fold macro-F1 (uncalibrated, no calibration fold held out)",
        "",
        "| fold | n test | macro-F1 | log-loss |",
        "|---|---|---|---|",
    ]
    for row in scores:
        lines.append(
            "| {fold} | {n} | {f1:.4f} | {ll:.4f} |".format(
                fold=row["fold"],
                n=row["n_test"],
                f1=row["macro_f1"],
                ll=row["log_loss"],
            )
        )

    lines += [
        "",
        f"## Top-{top_k} features per fold",
        "",
        "| fold | " + " | ".join(f"#{i + 1}" for i in range(top_k)) + " |",
        "|" + "---|" * (top_k + 1),
    ]
    for fold, features in stability["per_fold_top_k"].items():
        cells = list(features) + [""] * (top_k - len(features))
        lines.append(f"| {fold} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Consensus ranking",
        "",
        "| rank | feature | mean abs SHAP | mean rank | rank spread | folds in top-k |",
        "|---|---|---|---|---|---|",
    ]
    for row in stability["consensus_ranking"]:
        lines.append(
            "| {rank} | {feature} | {shap:.4f} | {mean_rank:.2f} | {spread:.2f} | {count} |".format(
                rank=int(row["consensus_rank"]),
                feature=row["feature"],
                shap=row["mean_abs_shap"],
                mean_rank=row["mean_rank"],
                spread=row["rank_spread"],
                count=int(row["top_k_count"]),
            )
        )

    lines += [
        "",
        "## Reading the table",
        "",
        "- A Jaccard overlap near 1.0 means every fold picked the same",
        "  features, so the explanation the user sees is not an artefact of",
        "  one particular split.",
        "- The rank spread is the standard deviation of a feature's rank",
        "  across the folds; a feature with a spread near zero is a",
        "  permanent fixture of the model's reasoning.",
        "- The per-fold macro-F1 shows that the model itself is stable; an",
        "  unstable ranking on a stable model would point at the",
        "  explanation rather than at the data.",
        "",
    ]
    return "\n".join(lines)
