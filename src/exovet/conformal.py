"""Split conformal prediction sets for the ExoVet pipeline.

Implemented directly in numpy, following Angelopoulos & Bates (2023).
The method is *split* conformal: a calibration fold disjoint from the
training fold fixes a threshold, and the resulting prediction set carries
the coverage guarantee

    P(y_test in C(x_test)) >= 1 - alpha.

The nonconformity score is the probability deficit of the true label:

    s(x, y) = 1 - p_y(x)

For the arg-max class this is exactly ``1 - max_k p_k(x)`` (the ``maxp``
score of the config), so a confident prediction yields the singleton set
``{argmax}`` and an uncertain one yields a set of several classes. The
prediction set is

    C(x) = { y : 1 - p_y(x) <= q }

where ``q`` is the finite-sample corrected (1 - alpha) quantile of the
calibration scores.

Two quantile variants are provided:

``marginal``
    One threshold for all classes. Guarantees marginal coverage.

``mondrian``
    One threshold per class, fitted on the calibration points whose true
    label is that class. Guarantees class-conditional coverage, which is
    the stronger statement for a three-class problem with a rare class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from .data import LABELS


def conformal_scores(
    proba: np.ndarray,
    y: np.ndarray,
    classes: Sequence[str] = LABELS,
) -> np.ndarray:
    """Return the nonconformity score ``1 - p_y(x)`` for every row."""
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y)
    positions = {cls: i for i, cls in enumerate(classes)}
    scores = np.empty(len(y), dtype=float)
    for row, label in enumerate(y):
        if label not in positions:
            raise ValueError(f"unknown label {label!r}")
        scores[row] = 1.0 - proba[row, positions[label]]
    return scores


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the (1 - alpha) quantile with the split-conformal correction.

    Uses the rank ``ceil((n + 1) * (1 - alpha)) / n``, so that the
    guarantee holds for any exchangeable data, not only in expectation.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return float("inf")
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        return float("inf")
    return float(np.quantile(scores, level, method="higher"))


@dataclass
class SplitConformal:
    """Split conformal predictor over a fixed label order."""

    alpha: float = 0.10
    mode: str = "marginal"
    classes: tuple[str, ...] = LABELS
    always_include_argmax: bool = True
    threshold: float = field(default=float("inf"), init=False)
    class_thresholds: dict[str, float] = field(default_factory=dict, init=False)
    n_calibration: int = field(default=0, init=False)

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "SplitConformal":
        """Fit the thresholds on the calibration fold."""
        proba = np.asarray(proba, dtype=float)
        y = np.asarray(y)
        if len(y) == 0:
            raise ValueError("calibration set is empty")
        self.n_calibration = len(y)
        if self.mode == "marginal":
            scores = conformal_scores(proba, y, self.classes)
            self.threshold = finite_sample_quantile(scores, self.alpha)
            self.class_thresholds = {}
        elif self.mode == "mondrian":
            self.threshold = float("inf")
            self.class_thresholds = {}
            for cls in self.classes:
                mask = y == cls
                if mask.sum() == 0:
                    self.class_thresholds[cls] = float("inf")
                    continue
                scores = conformal_scores(proba[mask], y[mask], self.classes)
                self.class_thresholds[cls] = finite_sample_quantile(scores, self.alpha)
        else:
            raise ValueError(f"unknown conformal mode {self.mode!r}")
        return self

    def predict_set(self, proba: np.ndarray) -> np.ndarray:
        """Return a boolean mask of shape ``(n_rows, n_classes)``.

        With ``always_include_argmax`` the model's best guess is always in
        the set, so the set is never empty. Adding a class can only grow
        the set, so the coverage guarantee is unaffected - it simply
        becomes slightly conservative.
        """
        proba = np.asarray(proba, dtype=float)
        mask = np.zeros((len(proba), len(self.classes)), dtype=bool)
        if self.mode == "marginal":
            for position in range(len(self.classes)):
                mask[:, position] = (1.0 - proba[:, position]) <= self.threshold
        else:
            for position, cls in enumerate(self.classes):
                mask[:, position] = (1.0 - proba[:, position]) <= self.class_thresholds[cls]
        if self.always_include_argmax and len(proba):
            mask[np.arange(len(proba)), np.argmax(proba, axis=1)] = True
        return mask

    def predict_sets(self, proba: np.ndarray) -> List[List[str]]:
        """Return the prediction set as a list of label lists."""
        mask = self.predict_set(proba)
        return [[cls for cls, keep in zip(self.classes, row, strict=True) if keep] for row in mask]

    def coverage(self, proba: np.ndarray, y: np.ndarray) -> float:
        """Empirical coverage of the true label on the given data."""
        mask = self.predict_set(proba)
        y = np.asarray(y)
        positions = {cls: i for i, cls in enumerate(self.classes)}
        hits = [
            bool(mask[row, positions[label]]) for row, label in enumerate(y) if label in positions
        ]
        return float(np.mean(hits)) if hits else float("nan")

    def set_sizes(self, proba: np.ndarray) -> np.ndarray:
        """Return the size of each prediction set."""
        return self.predict_set(proba).sum(axis=1)

    def summary(self, proba: np.ndarray, y: np.ndarray) -> dict:
        """Coverage, set size and singleton rate on the given data."""
        sizes = self.set_sizes(proba)
        mask = self.predict_set(proba)
        y = np.asarray(y)
        positions = {cls: i for i, cls in enumerate(self.classes)}
        hits = [
            bool(mask[row, positions[label]]) for row, label in enumerate(y) if label in positions
        ]
        per_class_coverage: dict[str, float] = {}
        for cls in self.classes:
            mask_rows = y == cls
            if not mask_rows.any():
                per_class_coverage[cls] = float("nan")
                continue
            position = positions[cls]
            per_class_coverage[cls] = float(
                np.mean([bool(mask[row, position]) for row in np.flatnonzero(mask_rows)])
            )
        return {
            "alpha": self.alpha,
            "mode": self.mode,
            "n_calibration": self.n_calibration,
            "empirical_coverage": float(np.mean(hits)) if hits else float("nan"),
            "per_class_coverage": per_class_coverage,
            "target_coverage": 1.0 - self.alpha,
            "mean_set_size": float(np.mean(sizes)),
            "median_set_size": float(np.median(sizes)),
            "singleton_rate": float(np.mean(sizes == 1)),
            "empty_set_rate": float(np.mean(sizes == 0)),
            "always_include_argmax": self.always_include_argmax,
            "threshold": self.threshold,
            "class_thresholds": dict(self.class_thresholds),
        }
