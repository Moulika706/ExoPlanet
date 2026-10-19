"""Probability calibration for the ExoVet pipeline.

All calibrators are fitted on a held-out calibration fold that is disjoint
from the training fold, and all of them take and return probability
matrices whose columns follow one fixed, canonical class order (the
``label_order`` of the config). Getting that order right matters: a
per-class calibrator applied to the wrong column silently destroys the
model while still producing plausible-looking probabilities.

Four methods are available:

``isotonic``
    One :class:`~sklearn.isotonic.IsotonicRegression` per class, fitted on
    (probability of class c, indicator that the label is c). This is the
    classic per-class approach used by scikit-learn. It is flexible but
    the per-class curves are fitted independently, so it can reorder the
    classes relative to the uncalibrated model.

``isotonic_confidence``
    One isotonic regression from the model's confidence (its maximum
    probability) to the observed accuracy, applied to the whole probability
    vector. Because the same positive factor rescales every class, the
    arg-max is preserved exactly. Robust and cheap.

``platt``
    Multinomial vector scaling: a logistic regression on the log-odds of
    the class probabilities. Smooth, three effective degrees of freedom,
    good on small calibration sets.

``temperature``
    A single temperature on the log-probabilities, ``softmax(log p / T)``.
    The least flexible of the four, and the only one that provably cannot
    change the arg-max or the prediction order.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _clip(proba: np.ndarray) -> np.ndarray:
    """Clip probabilities away from 0 and 1, then renormalise rows to sum 1."""
    proba = np.clip(np.asarray(proba, dtype=float), EPS, 1.0 - EPS)
    return proba / proba.sum(axis=1, keepdims=True)


def _encode(y: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    """Encode labels as integers following the canonical class order."""
    positions = {cls: i for i, cls in enumerate(classes)}
    return np.array([positions[label] for label in y], dtype=int)


def _log_odds(proba: np.ndarray) -> np.ndarray:
    """Log-odds transform of a probability matrix."""
    clipped = np.clip(proba, EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


# ----------------------------------------------------------------------
# per-class isotonic
# ----------------------------------------------------------------------
def fit_isotonic(proba: np.ndarray, y: np.ndarray, classes: Sequence[str]) -> Dict[str, Any]:
    """Fit one isotonic regression per class, in canonical class order."""
    proba = np.asarray(proba, dtype=float)
    codes = _encode(y, classes)
    regressions = []
    for position in range(len(classes)):
        target = (codes == position).astype(float)
        reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        reg.fit(proba[:, position], target)
        regressions.append(reg)
    return {"method": "isotonic", "classes": tuple(classes), "regressions": regressions}


def apply_isotonic(state: Dict[str, Any], proba: np.ndarray) -> np.ndarray:
    """Apply a fitted per-class isotonic calibrator."""
    proba = np.asarray(proba, dtype=float)
    out = np.zeros_like(proba)
    for position, reg in enumerate(state["regressions"]):
        out[:, position] = reg.predict(proba[:, position])
    return _clip(out)


# ----------------------------------------------------------------------
# confidence isotonic (arg-max preserving)
# ----------------------------------------------------------------------
def fit_isotonic_confidence(
    proba: np.ndarray, y: np.ndarray, classes: Sequence[str]
) -> Dict[str, Any]:
    """Fit an isotonic regression from confidence to observed accuracy."""
    proba = np.asarray(proba, dtype=float)
    codes = _encode(y, classes)
    confidence = proba.max(axis=1)
    predicted = np.argmax(proba, axis=1)
    correct = (predicted == codes).astype(float)
    reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    reg.fit(confidence, correct)
    return {"method": "isotonic_confidence", "classes": tuple(classes), "regression": reg}


def apply_isotonic_confidence(state: Dict[str, Any], proba: np.ndarray) -> np.ndarray:
    """Apply the confidence isotonic calibrator (rescales, keeps the arg-max)."""
    proba = np.asarray(proba, dtype=float)
    confidence = proba.max(axis=1, keepdims=True)
    scaled_confidence = state["regression"].predict(confidence.ravel()).reshape(-1, 1)
    factor = np.divide(
        scaled_confidence, confidence, out=np.zeros_like(confidence), where=confidence > 0
    )
    return _clip(proba * factor)


# ----------------------------------------------------------------------
# vector scaling (Platt)
# ----------------------------------------------------------------------
def fit_platt(proba: np.ndarray, y: np.ndarray, classes: Sequence[str]) -> Dict[str, Any]:
    """Fit multinomial vector scaling on the log-odds of the probabilities."""
    proba = np.asarray(proba, dtype=float)
    codes = _encode(y, classes)
    design = _log_odds(proba)
    model = LogisticRegression(fit_intercept=False, max_iter=2000)
    model.fit(design, codes)
    return {"method": "platt", "classes": tuple(classes), "model": model}


def apply_platt(state: Dict[str, Any], proba: np.ndarray) -> np.ndarray:
    """Apply a fitted vector-scaling calibrator."""
    proba = np.asarray(proba, dtype=float)
    design = _log_odds(proba)
    raw = state["model"].predict_proba(design)
    # The model was fitted on canonical integer codes, so its own
    # ``classes_`` is already in canonical order.
    order = [list(state["model"].classes_).index(i) for i in range(len(state["classes"]))]
    return _clip(raw[:, order])


# ----------------------------------------------------------------------
# temperature scaling
# ----------------------------------------------------------------------
def fit_temperature(proba: np.ndarray, y: np.ndarray, classes: Sequence[str]) -> Dict[str, Any]:
    """Fit a single temperature by minimising the log loss."""
    from scipy.optimize import minimize_scalar

    proba = np.asarray(proba, dtype=float)
    codes = _encode(y, classes)
    log_proba = np.log(np.clip(proba, EPS, 1.0))

    def objective(log_t: float) -> float:
        scaled = log_proba / np.exp(log_t)
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        normalised = exp / exp.sum(axis=1, keepdims=True)
        picked = normalised[np.arange(len(codes)), codes]
        return float(-np.mean(np.log(np.clip(picked, EPS, 1.0))))

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    return {
        "method": "temperature",
        "classes": tuple(classes),
        "temperature": float(np.exp(result.x)),
        "log_loss_before": objective(-3.0),
        "log_loss_after": float(result.fun),
    }


def apply_temperature(state: Dict[str, Any], proba: np.ndarray) -> np.ndarray:
    """Apply the fitted temperature."""
    proba = np.asarray(proba, dtype=float)
    scaled = np.log(np.clip(proba, EPS, 1.0)) / state["temperature"]
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------
# factory
# ----------------------------------------------------------------------
METHODS = ("isotonic", "isotonic_confidence", "platt", "temperature")


def fit_calibrator(
    method: str,
    proba: np.ndarray,
    y: np.ndarray,
    classes: Sequence[str],
) -> Dict[str, Any]:
    """Fit the calibrator named *method* on ``(proba, y)``.

    *proba* must have its columns in the canonical *classes* order.
    """
    if method == "isotonic":
        return fit_isotonic(proba, y, classes)
    if method == "isotonic_confidence":
        return fit_isotonic_confidence(proba, y, classes)
    if method == "platt":
        return fit_platt(proba, y, classes)
    if method == "temperature":
        return fit_temperature(proba, y, classes)
    raise ValueError(f"unknown calibration method {method!r}; expected one of {METHODS}")


def apply_calibrator(state: Optional[Dict[str, Any]], proba: np.ndarray) -> np.ndarray:
    """Apply a calibrator; a missing calibrator returns the input unchanged."""
    if state is None:
        return np.asarray(proba, dtype=float)
    method = state["method"]
    if method == "isotonic":
        return apply_isotonic(state, proba)
    if method == "isotonic_confidence":
        return apply_isotonic_confidence(state, proba)
    if method == "platt":
        return apply_platt(state, proba)
    if method == "temperature":
        return apply_temperature(state, proba)
    raise ValueError(f"unknown calibration method {method!r}")
