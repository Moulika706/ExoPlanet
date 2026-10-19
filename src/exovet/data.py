"""Loading and cleaning for the ExoVet KOI vetting pipeline.

The raw source is the NASA Exoplanet Archive KOI cumulative table. Cleaning
keeps the three vetted dispositions (CONFIRMED, CANDIDATE, FALSE POSITIVE)
and exposes the core physical feature set (A) plus the leakage-risk
vetting outputs (B) documented in configs/default.yaml.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import yaml

TARGET = "koi_disposition"
GROUP_COLUMN = "kepid"
LABELS = ("CONFIRMED", "CANDIDATE", "FALSE POSITIVE")

CORE_FEATURES: List[str] = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_impact",
    "koi_insol",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_kepmag",
]

LEAKAGE_FEATURES: List[str] = [
    "koi_score",
    "koi_pdisposition",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
]


def load_config(path: str = "configs/default.yaml") -> dict:
    """Load YAML configuration from *path*."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw KOI cumulative CSV from *path*."""
    return pd.read_csv(path)


def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose disposition is one of the three vetted classes.

    Rows with any other disposition or a missing disposition are dropped.
    The index is reset.
    """
    mask = df[TARGET].isin(LABELS)
    out = df.loc[mask].copy()
    return out.reset_index(drop=True)


def feature_set(include_leakage: bool = False) -> List[str]:
    """Return feature columns for set A (default) or set B (with leakage)."""
    features = list(CORE_FEATURES)
    if include_leakage:
        features.extend(LEAKAGE_FEATURES)
    return features


def missing_value_rates(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    """Fraction of missing values per column, sorted descending."""
    rates = df[columns].isna().mean()
    return rates.sort_values(ascending=False)


def split_by_group(
    df: pd.DataFrame,
    group_column: str = GROUP_COLUMN,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into train/test so no *group_column* value appears in both.

    Uses scikit-learn GroupShuffleSplit, seeded by *seed*, so sibling KOIs
    (same host star) never straddle train and test.
    """
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_column]))
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    return train, test
