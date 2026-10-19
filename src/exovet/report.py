"""Plain-English report generation from SHAP explanations.

The report is built from the top-k SHAP features and nothing else. A
checker (:func:`check_report_features`) re-reads the generated text and
fails if any feature outside the top-k is mentioned, so the "explainable"
claim is enforced by code rather than by good intentions.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .data import LABELS

#: Human-readable name and unit for each core physical feature.
FEATURE_DISPLAY: Dict[str, tuple[str, str]] = {
    "koi_period": ("orbital period", "days"),
    "koi_duration": ("transit duration", "hours"),
    "koi_depth": ("transit depth", "ppm"),
    "koi_prad": ("planet radius", "Earth radii"),
    "koi_teq": ("equilibrium temperature", "K"),
    "koi_impact": ("impact parameter", ""),
    "koi_insol": ("insolation", "Earth flux"),
    "koi_steff": ("stellar effective temperature", "K"),
    "koi_slogg": ("stellar surface gravity", "log g"),
    "koi_srad": ("stellar radius", "solar radii"),
    "koi_kepmag": ("Kepler magnitude", "mag"),
    "koi_score": ("Kepler vetting score", ""),
    "koi_pdisposition": ("Kepler pipeline disposition", ""),
    "koi_fpflag_nt": ("neutral transit flag", ""),
    "koi_fpflag_ss": ("stellar secondary flag", ""),
    "koi_fpflag_co": ("centroid offset flag", ""),
    "koi_fpflag_ec": ("eclipsing binary flag", ""),
}

#: Phrases used to describe a feature that supports or opposes the class.
PUSHES_TOWARD = "pushes the assessment towards"
PUSHES_AGAINST = "counts against"


def display_name(feature: str) -> str:
    """Return the human-readable name of *feature*."""
    if feature in FEATURE_DISPLAY:
        return FEATURE_DISPLAY[feature][0]
    return feature.replace("_", " ").replace("koi ", "")


def feature_unit(feature: str) -> str:
    """Return the unit of *feature* (empty string when unitless)."""
    if feature in FEATURE_DISPLAY:
        return FEATURE_DISPLAY[feature][1]
    return ""


def format_value(feature: str, value: float) -> str:
    """Format a raw feature value with a sensible precision and unit."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "an imputed value"
    if feature == "koi_period":
        return f"{value:.3f} days"
    if feature == "koi_duration":
        return f"{value:.3f} hours"
    if feature == "koi_depth":
        return f"{value:.0f} ppm"
    if feature == "koi_prad":
        return f"{value:.2f} Earth radii"
    if feature == "koi_teq":
        return f"{value:.0f} K"
    if feature == "koi_impact":
        return f"{value:.3f}"
    if feature == "koi_insol":
        return f"{value:.1f} Earth flux"
    if feature == "koi_steff":
        return f"{value:.0f} K"
    if feature == "koi_slogg":
        return f"{value:.3f}"
    if feature == "koi_srad":
        return f"{value:.3f} solar radii"
    if feature == "koi_kepmag":
        return f"{value:.2f} mag"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{value:.3f}"


def build_report(
    shap_frame: pd.DataFrame,
    proba: np.ndarray,
    classes: Sequence[str] = LABELS,
    prediction_set: Sequence[str] | None = None,
    top_k: int = 5,
) -> Dict[str, object]:
    """Build the plain-English report from a ranked SHAP frame.

    *shap_frame* must have the columns ``feature``, ``value`` and
    ``shap_value``, already sorted by absolute SHAP value. Only these rows
    are allowed to influence the text.
    """
    proba = np.asarray(proba, dtype=float)
    best = int(np.argmax(proba))
    predicted_class = classes[best]
    probability = float(proba[best])

    rows = shap_frame.head(top_k).to_dict("records")
    contributions = []
    for row in rows:
        shap_value = float(row["shap_value"])
        direction = (
            PUSHES_TOWARD
            if shap_value > 0
            else (PUSHES_AGAINST if shap_value < 0 else "is neutral for")
        )
        contributions.append(
            {
                "feature": row["feature"],
                "display_name": display_name(row["feature"]),
                "value": row.get("value"),
                "formatted_value": format_value(row["feature"], row.get("value")),
                "shap_value": shap_value,
                "direction": direction,
                "magnitude": abs(shap_value),
            }
        )

    text = _render_text(predicted_class, probability, contributions, prediction_set)
    return {
        "predicted_class": predicted_class,
        "probability": probability,
        "prediction_set": list(prediction_set) if prediction_set is not None else None,
        "needs_review": prediction_set is not None and len(set(prediction_set)) > 1,
        "top_features": contributions,
        "text": text,
    }


def _render_text(
    predicted_class: str,
    probability: float,
    contributions: List[Dict[str, object]],
    prediction_set: Sequence[str] | None,
) -> str:
    """Render the report sentence by sentence."""
    parts: List[str] = []
    parts.append(
        f"The model assesses this object as {predicted_class} "
        f"with a calibrated probability of {probability:.0%}."
    )

    if not contributions:
        parts.append("No single feature dominates the decision.")
        return " ".join(parts)

    strongest = contributions[0]
    parts.append(
        f"The strongest evidence is the {strongest['display_name']} "
        f"({strongest['formatted_value']}), which {strongest['direction']} "
        f"{predicted_class}."
    )

    if len(contributions) > 1:
        follow_ups = []
        for contribution in contributions[1:]:
            follow_ups.append(
                f"the {contribution['display_name']} "
                f"({contribution['formatted_value']}), which {contribution['direction']} "
                f"{predicted_class}"
            )
        if len(follow_ups) == 1:
            joined = follow_ups[0]
        else:
            joined = "; ".join(follow_ups[:-1]) + "; and " + follow_ups[-1]
        parts.append("This is followed by " + joined + ".")

    if prediction_set is not None and len(set(prediction_set)) > 1:
        others = [c for c in prediction_set if c != predicted_class]
        parts.append(
            "The evidence is not decisive on its own, so the model abstains and "
            f"flags this object for manual review against {', '.join(others)}."
        )

    return " ".join(parts)


def check_report_features(
    text: str,
    all_features: Sequence[str],
    allowed_features: Sequence[str],
) -> List[str]:
    """Return the features mentioned in *text* that are not allowed.

    Both the human-readable name and the raw column name are searched, so
    a report cannot smuggle in a feature by renaming it.
    """
    allowed = set(allowed_features)
    lowered = text.lower()
    violations: List[str] = []
    for feature in all_features:
        if feature in allowed:
            continue
        candidates = [display_name(feature), feature]
        for candidate in candidates:
            if not candidate:
                continue
            pattern = re.escape(candidate.lower())
            if re.search(rf"(?<![\w]){pattern}(?![\w])", lowered):
                violations.append(feature)
                break
    return violations
