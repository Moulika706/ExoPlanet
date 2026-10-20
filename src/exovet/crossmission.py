"""Cross-mission transfer: train on Kepler, apply to TESS.

The product claims to vet *any* transiting signal, not only Kepler's.
That claim is only worth something if the model survives a change of
mission, so this module does the transfer the brief asks for: train on
Kepler, evaluate zero-shot on TESS, report the drop, and recalibrate if
the drop is a calibration problem rather than a ranking problem.

Two things make the transfer non-trivial, and both are handled here.

**Feature harmonisation.** The TOI table does not have the same columns as
the KOI table. Nine of the eleven core Kepler features exist in the TOI
table with the same definition and the same unit, and two do not: the
impact parameter is absent, and the Kepler magnitude would have to be
replaced by the TESS magnitude, which is a different photometric system.
Substituting the TESS magnitude for the Kepler magnitude would hand the
model a systematically shifted feature, so both are left out and the model
is trained on the nine that mean the same thing in both missions.

**Label semantics.** The TOI table carries the TFOP working group
disposition, whose six values are mapped onto the three canonical labels.
The mapping is stated in the report, because the two vetting pipelines are
not the same and the classes are not perfectly interchangeable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .data import LABELS, TARGET

#: The TOI column that carries each Kepler feature, for the features that
#: mean the same thing in both missions. Same definition, same unit.
TOI_COLUMN_FOR_FEATURE: Dict[str, str] = {
    "koi_period": "pl_orbper",
    "koi_duration": "pl_trandurh",
    "koi_depth": "pl_trandep",
    "koi_prad": "pl_rade",
    "koi_teq": "pl_eqt",
    "koi_insol": "pl_insol",
    "koi_steff": "st_teff",
    "koi_slogg": "st_logg",
    "koi_srad": "st_rad",
}

#: The core Kepler features that cannot be carried across missions, and why.
EXCLUDED_FEATURES: Dict[str, str] = {
    "koi_impact": "the TOI table has no impact parameter",
    "koi_kepmag": (
        "the TOI table reports TESS magnitudes, a different photometric "
        "system from the Kepler magnitude"
    ),
}

#: TFOP working group disposition to the canonical label.
DISPOSITION_TO_LABEL: Dict[str, str] = {
    "PC": "CANDIDATE",
    "APC": "CANDIDATE",
    "CP": "CONFIRMED",
    "KP": "CONFIRMED",
    "FP": "FALSE POSITIVE",
    "FA": "FALSE POSITIVE",
}

#: The TOI column that holds the TFOP working group disposition.
DISPOSITION_COLUMN = "tfopwg_disp"

#: The TOI column that identifies the host star (TESS Input Catalog id).
TOI_GROUP_COLUMN = "tid"


def shared_features() -> List[str]:
    """The core Kepler features that survive the trip to TESS."""
    return list(TOI_COLUMN_FOR_FEATURE)


def excluded_features() -> Dict[str, str]:
    """The core Kepler features that do not survive the trip, and why."""
    return dict(EXCLUDED_FEATURES)


def harmonise_toi(
    df: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    disposition_column: str = DISPOSITION_COLUMN,
    group_column: str = TOI_GROUP_COLUMN,
) -> pd.DataFrame:
    """Turn a raw TOI table into a frame the Kepler pipeline can read.

    The TOI columns are renamed to their Kepler names, the disposition is
    mapped to the canonical labels, and the result carries the same target
    column name as the KOI table, so the two can be compared directly.
    """
    feature_names = list(features) if features is not None else shared_features()
    missing = [c for c in feature_names if c not in TOI_COLUMN_FOR_FEATURE]
    if missing:
        raise ValueError(f"no TOI column known for {missing}")

    # Strict on purpose: a feature that cannot be harmonised must not be
    # dropped silently, because that would quietly change the feature set
    # the model was trained on.
    missing_columns = [
        TOI_COLUMN_FOR_FEATURE[feature]
        for feature in feature_names
        if TOI_COLUMN_FOR_FEATURE[feature] not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"the TOI table has no column {missing_columns}")

    out = pd.DataFrame(index=df.index)
    for feature in feature_names:
        out[feature] = pd.to_numeric(df[TOI_COLUMN_FOR_FEATURE[feature]], errors="coerce")

    if disposition_column in df.columns:
        out[TARGET] = df[disposition_column].map(DISPOSITION_TO_LABEL)
    else:
        out[TARGET] = np.nan

    if group_column in df.columns:
        out[group_column] = df[group_column]
    else:
        out[group_column] = np.arange(len(df))

    out["toi"] = df["toi"] if "toi" in df.columns else np.arange(len(df))
    return out.reset_index(drop=True)


def clean_toi(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the TOI rows whose disposition maps to a canonical label."""
    out = df[df[TARGET].isin(LABELS)].copy()
    return out.reset_index(drop=True)


def feature_shift_table(
    kepler: pd.DataFrame,
    tess: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compare the Kepler and TESS distributions of every shared feature.

    A median that moves by a large fraction of its own spread is a
    warning sign for the transfer, and it is worth seeing before the
    metrics rather than after.
    """
    feature_names = list(features) if features is not None else shared_features()
    rows = []
    for feature in feature_names:
        k = pd.to_numeric(kepler[feature], errors="coerce").dropna()
        t = pd.to_numeric(tess[feature], errors="coerce").dropna()
        if len(k) == 0 or len(t) == 0:
            continue
        k_median = float(k.median())
        t_median = float(t.median())
        iqr = float(k.quantile(0.75) - k.quantile(0.25))
        rows.append(
            {
                "feature": feature,
                "kepler_median": k_median,
                "tess_median": t_median,
                "median_shift": t_median - k_median,
                "shift_in_kepler_iqr": (t_median - k_median) / iqr if iqr else float("nan"),
                "kepler_missing": float(kepler[feature].isna().mean()),
                "tess_missing": float(tess[feature].isna().mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_transfer(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: Sequence[str] = LABELS,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Metrics for a set of probabilities on the target mission."""
    from .eval import evaluate, reliability_table

    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true)
    metrics = evaluate(y_true, proba, classes, n_bins=n_bins)
    metrics["reliability"] = reliability_table(y_true, proba, classes, n_bins)
    return metrics


def transfer_table(rows: List[Dict[str, Any]], classes: Sequence[str] = LABELS) -> str:
    """Render the cross-mission results as a Markdown table."""
    header = (
        "| setting | set | macro-F1 | PR-AUC | log-loss | ECE | "
        + " | ".join(f"recall {cls}" for cls in classes)
        + " | conformal cov. | mean set |"
    )
    lines = [header, "|" + "---|" * (5 + len(classes) + 2)]
    for row in rows:
        metrics = row["metrics"]
        recalls = metrics["per_class_recall"]
        cells = " | ".join(f"{recalls.get(cls, float('nan')):.4f}" for cls in classes)
        conformal = row.get("conformal", {})
        lines.append(
            "| {setting} | {name} | {f1:.4f} | {pr:.4f} | {ll:.4f} | {ece:.4f} | "
            "{cells} | {cov} | {size} |".format(
                setting=row["setting"],
                name=row["name"],
                f1=metrics["macro_f1"],
                pr=metrics["macro_pr_auc"],
                ll=metrics["log_loss"],
                ece=metrics["expected_calibration_error"],
                cells=cells,
                cov=(f"{conformal['empirical_coverage']:.4f}" if conformal else "-"),
                size=(f"{conformal['mean_set_size']:.2f}" if conformal else "-"),
            )
        )
    return "\n".join(lines)
