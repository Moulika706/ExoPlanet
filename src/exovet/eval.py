"""Evaluation metrics for the ExoVet pipeline.

Accuracy is deliberately absent: a model that labels everything FALSE
POSITIVE scores 50% on this data set and is useless. The metrics here are
the ones the project brief asks for - macro-F1, per-class recall, PR-AUC,
recall at 99% precision, log-loss, Brier score and expected calibration
error.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    recall_score,
)

from .data import LABELS

EPS = 1e-15


def _one_hot(y: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    """One-hot encode *y* over *classes*."""
    positions = {cls: i for i, cls in enumerate(classes)}
    encoded = np.zeros((len(y), len(classes)), dtype=float)
    for row, label in enumerate(y):
        encoded[row, positions[label]] = 1.0
    return encoded


def macro_f1(y_true: np.ndarray, proba: np.ndarray, classes: Sequence[str] = LABELS) -> float:
    """Macro-averaged F1 over the given classes."""
    from sklearn.metrics import f1_score

    predicted = _predict_labels(proba, classes)
    return float(
        f1_score(y_true, predicted, labels=list(classes), average="macro", zero_division=0)
    )


def per_class_recall(
    y_true: np.ndarray, proba: np.ndarray, classes: Sequence[str] = LABELS
) -> Dict[str, float]:
    """Recall of each class under the arg-max decision."""
    predicted = _predict_labels(proba, classes)
    recalls = recall_score(y_true, predicted, labels=list(classes), average=None, zero_division=0)
    return {cls: float(value) for cls, value in zip(classes, recalls, strict=True)}


def _predict_labels(proba: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    """Arg-max label for every row."""
    return np.asarray(classes)[np.argmax(np.asarray(proba), axis=1)]


def macro_pr_auc(y_true: np.ndarray, proba: np.ndarray, classes: Sequence[str] = LABELS) -> float:
    """Macro one-vs-rest average precision (the PR-AUC analogue of ROC-AUC)."""
    proba = np.asarray(proba, dtype=float)
    scores = []
    for position in range(len(classes)):
        target = (np.asarray(y_true) == classes[position]).astype(int)
        if target.sum() == 0:
            continue
        scores.append(average_precision_score(target, proba[:, position]))
    return float(np.mean(scores)) if scores else float("nan")


def recall_at_precision(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: Sequence[str] = LABELS,
    min_precision: float = 0.99,
) -> Dict[str, float]:
    """Recall attainable for each class at a fixed precision floor.

    This is the ExoMiner-style operating point: how much of a class can be
    recovered while keeping at least ``min_precision`` precision. The value
    is the largest recall whose precision is still at or above the floor.
    """
    proba = np.asarray(proba, dtype=float)
    result: Dict[str, float] = {}
    for position, cls in enumerate(classes):
        target = (np.asarray(y_true) == cls).astype(int)
        if target.sum() == 0:
            result[cls] = float("nan")
            continue
        precision, recall, _ = precision_recall_curve(target, proba[:, position])
        # precision/recall are ordered by increasing threshold; walk from the
        # strictest end and keep the last index that clears the floor.
        qualifying = np.where(precision[:-1] >= min_precision)[0]
        result[cls] = float(recall[qualifying].max()) if len(qualifying) else 0.0
    return result


def multiclass_log_loss(
    y_true: np.ndarray, proba: np.ndarray, classes: Sequence[str] = LABELS
) -> float:
    """Multiclass log loss, computed in the canonical class order.

    ``sklearn.metrics.log_loss`` cannot be used directly here: it builds a
    ``LabelBinarizer`` internally, which sorts the labels, so it expects
    the probability columns in *sorted* order. Our matrices use the
    canonical order from the config, and passing them to scikit-learn
    silently scores the wrong column - a bug that inflates the log loss
    of a well-calibrated model to about 1.3.
    """
    proba = np.asarray(proba, dtype=float)
    clipped = np.clip(proba, EPS, 1.0 - EPS)
    positions = {cls: i for i, cls in enumerate(classes)}
    index = np.array([positions[label] for label in y_true], dtype=int)
    picked = clipped[np.arange(len(index)), index]
    return float(-np.mean(np.log(picked)))


def brier_score(y_true: np.ndarray, proba: np.ndarray, classes: Sequence[str] = LABELS) -> float:
    """Multiclass Brier score: mean squared error of the probability vector."""
    proba = np.asarray(proba, dtype=float)
    encoded = _one_hot(y_true, classes)
    return float(np.mean(np.sum((proba - encoded) ** 2, axis=1)))


def reliability_table(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: Sequence[str] = LABELS,
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    """Bin predictions by confidence and report accuracy per bin."""
    proba = np.asarray(proba, dtype=float)
    predicted = _predict_labels(proba, classes)
    confidence = proba.max(axis=1)
    correct = (predicted == np.asarray(y_true)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    table: List[Dict[str, float]] = []
    for index in range(n_bins):
        low, high = edges[index], edges[index + 1]
        if index == 0:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence > low) & (confidence <= high)
        count = int(mask.sum())
        table.append(
            {
                "bin_low": float(low),
                "bin_high": float(high),
                "count": count,
                "accuracy": float(correct[mask].mean()) if count else float("nan"),
                "mean_confidence": float(confidence[mask].mean()) if count else float("nan"),
            }
        )
    return table


def expected_calibration_error(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: Sequence[str] = LABELS,
    n_bins: int = 10,
) -> float:
    """Expected calibration error over equal-width confidence bins."""
    table = reliability_table(y_true, proba, classes, n_bins)
    total = sum(row["count"] for row in table)
    if total == 0:
        return float("nan")
    error = 0.0
    for row in table:
        if row["count"] == 0:
            continue
        weight = row["count"] / total
        error += weight * abs(row["accuracy"] - row["mean_confidence"])
    return float(error)


def evaluate(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: Sequence[str] = LABELS,
    n_bins: int = 10,
    min_precision: float = 0.99,
) -> Dict[str, object]:
    """Compute the full metric bundle used in the results table."""
    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true)
    recalls = recall_at_precision(y_true, proba, classes, min_precision)
    return {
        "n_samples": int(len(y_true)),
        "macro_f1": macro_f1(y_true, proba, classes),
        "per_class_recall": per_class_recall(y_true, proba, classes),
        "macro_pr_auc": macro_pr_auc(y_true, proba, classes),
        "recall_at_precision": recalls,
        "macro_recall_at_precision": float(
            np.nanmean(list(recalls.values())) if recalls else float("nan")
        ),
        "log_loss": multiclass_log_loss(y_true, proba, classes),
        "brier_score": brier_score(y_true, proba, classes),
        "expected_calibration_error": expected_calibration_error(y_true, proba, classes, n_bins),
        "accuracy": float(np.mean(_predict_labels(proba, classes) == y_true)),
    }
