"""Model factory for the ExoVet pipeline.

Three models are compared on the same grouped splits:

* logistic regression - transparent baseline,
* random forest - bagged trees,
* XGBoost - the main model (boosted trees, which SHAP's TreeExplainer
  supports exactly).
"""

from __future__ import annotations

from typing import Any, Dict

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .features import build_preprocessor, requires_scaling

MODEL_NAMES = ("logistic_regression", "random_forest", "xgboost")


class LabelEncodingClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a classifier so that it accepts string labels.

    XGBoost rejects string labels, and a ``LabelEncoder`` cannot be a
    step of a scikit-learn ``Pipeline`` because intermediate steps are
    handed the feature matrix, not the target. This wrapper encodes the
    target inside ``fit`` and decodes it again in ``predict``, so every
    model in the zoo takes the same string labels.

    ``classes_`` holds the labels in sorted order, which is the order the
    probability columns follow.
    """

    def __init__(self, base_estimator: Any) -> None:
        self.base_estimator = base_estimator

    def fit(self, X, y):  # noqa: N803 - scikit-learn naming
        self.encoder_ = LabelEncoder()
        encoded = self.encoder_.fit_transform(y)
        self.base_estimator.fit(X, encoded)
        self.classes_ = self.encoder_.classes_
        self.n_features_in_ = getattr(self.base_estimator, "n_features_in_", None)
        return self

    def predict_proba(self, X):  # noqa: N803 - scikit-learn naming
        return self.base_estimator.predict_proba(X)

    def predict(self, X):  # noqa: N803 - scikit-learn naming
        return self.encoder_.inverse_transform(self.base_estimator.predict(X))

    def decision_function(self, X):  # noqa: N803 - scikit-learn naming
        return self.base_estimator.decision_function(X)


def build_model(
    name: str,
    params: Dict[str, Any] | None = None,
    seed: int = 42,
    feature_names: list[str] | None = None,
) -> Pipeline:
    """Build a fitted-ready pipeline named *name*.

    The returned pipeline is ``preprocessor -> labelencoder -> classifier``
    so that a single ``fit`` call handles imputation, encoding and training
    in the right order, and the preprocessor is refit per fold.

    The label encoder is part of the pipeline because XGBoost rejects
    string labels, and because the class order it produces (sorted) is
    what the probability columns follow.
    """
    params = dict(params or {})
    features = list(feature_names) if feature_names is not None else []
    preprocessor = build_preprocessor(
        features,
        scale=requires_scaling(name),
    )

    if name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=params.get("max_iter", 2000),
            random_state=seed,
        )
    elif name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 400),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 2),
            random_state=seed,
            n_jobs=-1,
        )
    elif name == "xgboost":
        from xgboost import XGBClassifier

        classifier = XGBClassifier(
            n_estimators=params.get("n_estimators", 400),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.9),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_lambda=params.get("reg_lambda", 1.0),
            random_state=seed,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
    else:
        raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")

    from sklearn.preprocessing import LabelEncoder  # noqa: F401 - used by the wrapper

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LabelEncodingClassifier(classifier)),
        ]
    )


def is_tree_model(name: str) -> bool:
    """Whether the model is tree based (and so SHAP's TreeExplainer applies)."""
    return name in ("random_forest", "xgboost")
