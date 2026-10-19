"""SHAP explanations for the ExoVet pipeline.

SHAP is computed on the *transformed* feature matrix, because the tree
models consume the output of the preprocessor, not the raw columns. The
transformed feature names are taken from the fitted transformer so that
one-hot encodings stay attached to the right source column.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .data import LABELS


def transformed_feature_names(pipeline) -> List[str]:
    """Return the names of the columns produced by a fitted pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:  # noqa: BLE001 - fall back to positional names
        names = [f"f{i}" for i in range(preprocessor.transform_shape_[1])]
    return [str(name) for name in names]


def _normalize_values(values, n_samples: int, n_features: int, n_classes: int) -> np.ndarray:
    """Normalise SHAP output into shape ``(n_samples, n_features, n_classes)``."""
    if isinstance(values, list):
        values = np.stack([np.asarray(v) for v in values], axis=-1)
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        if n_classes == 2:
            stacked = np.stack([array, -array], axis=-1)
        else:
            stacked = array[:, :, None]
    elif array.ndim == 3:
        stacked = array
    else:
        raise ValueError(f"unexpected SHAP value shape {array.shape}")
    if stacked.shape[:2] != (n_samples, n_features):
        raise ValueError(f"SHAP values {stacked.shape} do not match data {(n_samples, n_features)}")
    return stacked


def compute_shap_values(
    pipeline,
    X: pd.DataFrame,
    classes: Sequence[str] = LABELS,
    background: Optional[np.ndarray] = None,
    background_size: int = 100,
    seed: int = 42,
    perturbation: str = "tree_path",
) -> np.ndarray:
    """Return SHAP values with shape ``(n_rows, n_features, n_classes)``.

    Uses ``TreeExplainer`` for the tree models and ``LinearExplainer`` for
    logistic regression. *background* is the transformed reference sample
    the explanations are computed against; when it is not given, a sample
    of *X* is used (drawn with replacement, so that a single-row frame
    still yields a usable background).

    *perturbation* is ``tree_path`` (the classic Lundberg formulation,
    and about eighty times faster) or ``interventional`` (attributes
    against the background distribution, which is the stricter estimate
    but needs a tree traversal per background row).

    If SHAP cannot handle a model, permutation importance is used as a
    documented fallback.
    """
    import shap

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    # The classifier may be wrapped for label encoding; SHAP needs the
    # tree model itself.
    base_classifier = getattr(classifier, "base_estimator", classifier)
    transformed = preprocessor.transform(X)
    names = transformed_feature_names(pipeline)
    n_samples, n_features = transformed.shape[0], len(names)
    n_classes = len(classes)

    if background is None:
        rng = np.random.default_rng(seed)
        if transformed.shape[0] >= background_size:
            pick = rng.choice(transformed.shape[0], size=background_size, replace=False)
        else:
            pick = rng.choice(transformed.shape[0], size=background_size, replace=True)
        background = transformed[pick]

    is_tree = (
        hasattr(base_classifier, "estimators_") or type(base_classifier).__name__ == "XGBClassifier"
    )
    try:
        if is_tree:
            try:
                if perturbation == "interventional":
                    explainer = shap.TreeExplainer(base_classifier, background)
                else:
                    explainer = shap.TreeExplainer(base_classifier)
                raw = explainer.shap_values(transformed)
            except Exception:  # noqa: BLE001 - tree_path fallback
                explainer = shap.TreeExplainer(base_classifier)
                raw = explainer.shap_values(transformed)
        else:
            explainer = shap.LinearExplainer(base_classifier, background)
            raw = explainer.shap_values(transformed)
    except Exception:  # noqa: BLE001 - documented fallback
        return _permutation_shap(pipeline, X, names, n_classes, seed)

    return _normalize_values(raw, n_samples, n_features, n_classes)


def _permutation_shap(
    pipeline,
    X: pd.DataFrame,
    names: List[str],
    n_classes: int,
    seed: int,
) -> np.ndarray:
    """Leave-one-out permutation fallback, shaped like SHAP values.

    Each feature's contribution is the change in every class probability
    when that column is shuffled. A positive value means the feature
    pushes the prediction towards that class. This is an approximation of
    SHAP, used only when SHAP itself cannot handle the model.
    """
    rng = np.random.default_rng(seed)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    work = X.copy()
    baseline = classifier.predict_proba(preprocessor.transform(work))
    values = np.zeros((len(X), len(names), n_classes), dtype=float)
    for position, name in enumerate(names):
        if name not in work.columns:
            continue
        original = work[name].to_numpy(copy=True)
        work[name] = original[rng.permutation(len(work))]
        perturbed = classifier.predict_proba(preprocessor.transform(work))
        work[name] = original
        values[:, position, :] = baseline - perturbed
    return values


def global_importance(
    shap_values: np.ndarray,
    names: Sequence[str],
    classes: Sequence[str] = LABELS,
) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, averaged over rows and classes."""
    values = np.asarray(shap_values, dtype=float)
    mean_abs = np.abs(values).mean(axis=(0, 2))
    order = np.argsort(mean_abs)[::-1]
    return pd.DataFrame(
        {
            "feature": [names[i] for i in order],
            "mean_abs_shap": [float(mean_abs[i]) for i in order],
        }
    )


def local_explanation(
    shap_values: np.ndarray,
    row: pd.Series,
    names: Sequence[str],
    class_index: int,
    top_k: int = 5,
) -> pd.DataFrame:
    """SHAP values of a single prediction, ranked by absolute contribution."""
    values = np.asarray(shap_values, dtype=float)
    row_values = values[0, :, class_index]
    order = np.argsort(np.abs(row_values))[::-1][:top_k]
    records = []
    for position in order:
        name = names[position]
        records.append(
            {
                "feature": name,
                "value": row.get(name, np.nan),
                "shap_value": float(row_values[position]),
            }
        )
    return pd.DataFrame(records)


def local_explanation_frame(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    names: Sequence[str],
    class_index: int,
    row_position: int = 0,
    top_k: int = 5,
) -> pd.DataFrame:
    """SHAP ranking for row *row_position* of *X*."""
    values = np.asarray(shap_values, dtype=float)
    row_values = values[row_position, :, class_index]
    order = np.argsort(np.abs(row_values))[::-1][:top_k]
    records = []
    for position in order:
        name = names[position]
        raw = X.iloc[row_position].get(name, np.nan)
        records.append(
            {
                "feature": name,
                "value": float(raw) if raw is not None else float("nan"),
                "shap_value": float(row_values[position]),
            }
        )
    return pd.DataFrame(records)
