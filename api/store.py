"""The model store and the feedback log for the ExoVet API.

The fitted pipeline is loaded once at start-up and kept in memory. The
feedback endpoint appends to a JSONL file, one record per line, so the
log can be read and appended without a database and without rewriting
the file.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from exovet.pipeline import VettingPipeline

DEFAULT_ARTIFACT = Path("results/model_xgboost.joblib")
DEFAULT_FEEDBACK_LOG = Path("data/feedback/feedback.jsonl")

#: Reminders that travel with every answer, so the user is never misled
#: about what the model is good for.
WARNINGS: List[str] = [
    "The model is trained on Kepler KOIs. The cross-mission evaluation "
    "shows it does not transfer to TESS objects.",
    "A prediction set with more than one class means the model abstains: "
    "the object needs manual review.",
]


class ModelStore:
    """Holds the fitted pipeline and hands out predictions."""

    def __init__(self, artifact: Path | str = DEFAULT_ARTIFACT) -> None:
        self.artifact = Path(artifact)
        self.pipeline: Optional[VettingPipeline] = None
        self.load_error: Optional[str] = None

    def load(self) -> "ModelStore":
        """Load the artifact, recording the reason if it fails."""
        if not self.artifact.exists():
            self.load_error = (
                f"{self.artifact} not found; run 'python scripts/train.py' (make train) first"
            )
            return self
        try:
            self.pipeline = VettingPipeline.load(self.artifact)
        except Exception as exc:  # noqa: BLE001 - a bad artifact must not kill the service
            self.load_error = f"could not load {self.artifact}: {exc}"
            return self
        self.load_error = None
        return self

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    def require(self) -> VettingPipeline:
        """Return the pipeline, or explain why there is none."""
        if self.pipeline is None:
            raise RuntimeError(self.load_error or "the model is not loaded")
        return self.pipeline

    def predict(self, values: Dict[str, float]) -> Dict[str, Any]:
        """Vet one object and return the result as plain data."""
        pipeline = self.require()
        result = pipeline.predict_one(values)
        return {
            "predicted_class": result.predicted_class,
            "probability": result.probability,
            "probabilities": result.probabilities,
            "prediction_set": result.prediction_set,
            "needs_review": result.needs_review,
            "top_features": result.top_features,
            "report": result.report,
            "model": pipeline.model_name,
            "conformal_alpha": pipeline.conformal_alpha,
            "target_coverage": 1.0 - pipeline.conformal_alpha,
            "warnings": list(WARNINGS),
        }


class FeedbackLog:
    """An append-only JSONL log of user corrections."""

    def __init__(self, path: Path | str = DEFAULT_FEEDBACK_LOG) -> None:
        self.path = Path(path)

    def append(
        self,
        values: Dict[str, float],
        predicted_class: str,
        corrected_class: str,
        comment: Optional[str] = None,
        object_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write one correction and return the stored record."""
        entry = {
            "id": uuid.uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "object_id": object_id,
            "values": values,
            "predicted_class": predicted_class,
            "corrected_class": corrected_class,
            "comment": comment,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        return entry

    def read(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Read the most recent corrections, newest last."""
        if not self.path.exists():
            return []
        entries: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries[-limit:]

    def summary(self) -> Dict[str, Any]:
        """How many corrections there are, and how often the model was wrong."""
        entries = self.read(limit=100000)
        disagreements = [e for e in entries if e.get("predicted_class") != e.get("corrected_class")]
        by_class: Dict[str, int] = {}
        for entry in disagreements:
            key = f"{entry.get('predicted_class')}->{entry.get('corrected_class')}"
            by_class[key] = by_class.get(key, 0) + 1
        return {
            "n_entries": len(entries),
            "n_disagreements": len(disagreements),
            "disagreement_rate": (len(disagreements) / len(entries)) if entries else 0.0,
            "by_transition": by_class,
        }


def feedback_path_from_env() -> Path:
    """The feedback log location, overridable by an environment variable."""
    return Path(os.environ.get("EXOVET_FEEDBACK_LOG", DEFAULT_FEEDBACK_LOG))
