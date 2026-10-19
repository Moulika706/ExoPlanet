"""Feature preprocessing for the ExoVet pipeline.

Every imputation and scaling happens inside a fitted scikit-learn pipeline,
so it is refit on each training fold and never sees validation or test rows.
"""

from __future__ import annotations

from typing import List, Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CORE_FEATURES, LEAKAGE_FEATURES

#: Columns in the leakage set that are categorical rather than numeric.
CATEGORICAL_FEATURES: List[str] = ["koi_pdisposition"]


def numeric_columns(features: Sequence[str]) -> List[str]:
    """Return the numeric subset of *features*."""
    return [f for f in features if f not in CATEGORICAL_FEATURES]


def categorical_columns(features: Sequence[str]) -> List[str]:
    """Return the categorical subset of *features*."""
    return [f for f in features if f in CATEGORICAL_FEATURES]


def build_preprocessor(
    features: Sequence[str],
    scale: bool = False,
) -> ColumnTransformer:
    """Build the preprocessing transformer for *features*.

    Numeric columns get median imputation (and optional standardisation,
    which distance-based models need and trees do not). Categorical columns
    are one-hot encoded; ``koi_pdisposition`` is a string column in the
    archive and would otherwise break the numeric pipeline.
    """
    num = numeric_columns(features)
    cat = categorical_columns(features)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    transformers = [("num", numeric_pipe, num)]
    if cat:
        categorical_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", categorical_pipe, cat))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def requires_scaling(model_name: str) -> bool:
    """Whether the model benefits from standardised inputs."""
    return model_name == "logistic_regression"


def feature_list(include_leakage: bool = False) -> List[str]:
    """Return the raw column list for set A (default) or set B."""
    features = list(CORE_FEATURES)
    if include_leakage:
        features.extend(LEAKAGE_FEATURES)
    return features
