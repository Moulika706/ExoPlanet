"""Tests for the ExoVet API (Phase 6)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from api.schemas import FEATURE_ORDER
from api.store import FeedbackLog, ModelStore

from exovet.data import LABELS
from exovet.features import feature_list

CLASSES = tuple(LABELS)
ARTIFACT = Path("results/model_xgboost.joblib")


def _synthetic_koi_frame(n_groups: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    features = feature_list(False)
    shifts = {"CONFIRMED": 3.0, "CANDIDATE": 0.0, "FALSE POSITIVE": -3.0}
    rows = []
    for group in range(n_groups):
        label = CLASSES[group % 3]
        for _member in range(2):
            row = {f: float(rng.normal(shifts[label], 0.4)) for f in features}
            row["kepid"] = group
            row["koi_disposition"] = label
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> ModelStore:
    """A store holding a small model fitted on synthetic data."""
    from exovet.pipeline import VettingPipeline

    df = _synthetic_koi_frame()
    pipeline = VettingPipeline(model_name="random_forest", params={"n_estimators": 20}, seed=42)
    pipeline.fit(df, n_splits=3, test_fold=0, calibration_fold=1)
    path = tmp_path_factory.mktemp("artifacts") / "model.joblib"
    pipeline.save(path)
    return ModelStore(path).load()


@pytest.fixture(scope="module")
def client(store, monkeypatch_module=None):
    """A test client whose store holds the synthetic model."""
    import api.main as main
    from fastapi.testclient import TestClient

    main.store.pipeline = store.pipeline
    main.store.load_error = None
    with TestClient(main.app) as test_client:
        yield test_client


# ----------------------------------------------------------------------
# the model store
# ----------------------------------------------------------------------
def test_store_reports_a_missing_artifact(tmp_path) -> None:
    missing = ModelStore(tmp_path / "nope.joblib").load()
    assert not missing.ready
    assert "not found" in missing.load_error
    with pytest.raises(RuntimeError):
        missing.require()


def test_store_predicts_a_class_and_a_set(store) -> None:
    result = store.predict({"koi_period": 10.0, "koi_prad": 2.0})
    assert result["predicted_class"] in CLASSES
    assert 0.0 <= result["probability"] <= 1.0
    assert result["prediction_set"]
    assert set(result["prediction_set"]) <= set(CLASSES)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert result["top_features"]
    assert result["report"]


def test_store_always_carries_the_warnings(store) -> None:
    result = store.predict({"koi_period": 10.0})
    assert any("Kepler" in warning for warning in result["warnings"])


# ----------------------------------------------------------------------
# the feedback log
# ----------------------------------------------------------------------
def test_feedback_log_appends_and_reads(tmp_path) -> None:
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    assert log.read() == []
    log.append({"koi_period": 1.0}, "CONFIRMED", "CANDIDATE", "looked wrong")
    log.append({"koi_period": 2.0}, "CANDIDATE", "CANDIDATE")
    entries = log.read()
    assert len(entries) == 2
    assert entries[0]["predicted_class"] == "CONFIRMED"
    assert entries[0]["comment"] == "looked wrong"
    assert entries[0]["id"] != entries[1]["id"]


def test_feedback_log_summarises_the_disagreements(tmp_path) -> None:
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    log.append({"koi_period": 1.0}, "CONFIRMED", "CANDIDATE")
    log.append({"koi_period": 2.0}, "CANDIDATE", "CANDIDATE")
    summary = log.summary()
    assert summary["n_entries"] == 2
    assert summary["n_disagreements"] == 1
    assert summary["disagreement_rate"] == pytest.approx(0.5)
    assert summary["by_transition"] == {"CONFIRMED->CANDIDATE": 1}


def test_feedback_log_is_valid_jsonl(tmp_path) -> None:
    path = tmp_path / "feedback.jsonl"
    log = FeedbackLog(path)
    log.append({"koi_period": 1.0}, "CONFIRMED", "CANDIDATE")
    for line in path.read_text(encoding="utf-8").splitlines():
        assert json.loads(line)["predicted_class"] == "CONFIRMED"


# ----------------------------------------------------------------------
# the endpoints
# ----------------------------------------------------------------------
def test_health_reports_the_service_state(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_endpoint_describes_the_loaded_model(client) -> None:
    response = client.get("/model")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] in ("random_forest", "xgboost", "logistic_regression")
    assert body["classes"] == list(CLASSES)
    assert 0.0 < body["target_coverage"] < 1.0
    assert body["warnings"]


def test_features_endpoint_lists_the_form_fields(client) -> None:
    response = client.get("/features")
    assert response.status_code == 200
    features = response.json()
    assert [f["name"] for f in features] == FEATURE_ORDER
    for feature in features:
        assert feature["display_name"]
        assert feature["description"]


def test_predict_returns_a_full_verdict(client) -> None:
    response = client.post(
        "/predict",
        json={
            "values": {
                "koi_period": 10.0,
                "koi_duration": 3.0,
                "koi_depth": 800.0,
                "koi_prad": 2.4,
                "koi_teq": 900.0,
            }
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in CLASSES
    assert body["prediction_set"]
    assert body["needs_review"] == (len(body["prediction_set"]) > 1)
    assert body["report"]
    assert body["top_features"]
    for feature in body["top_features"]:
        assert {"feature", "display_name", "shap_value", "formatted_value"} <= set(feature)


def test_predict_accepts_a_partial_form(client) -> None:
    response = client.post("/predict", json={"values": {"koi_period": 10.0}})
    assert response.status_code == 200
    assert response.json()["predicted_class"] in CLASSES


def test_predict_rejects_an_unknown_feature(client) -> None:
    response = client.post("/predict", json={"values": {"koi_bogus": 1.0}})
    assert response.status_code == 422


def test_explain_returns_the_same_verdict(client) -> None:
    payload = {"values": {"koi_period": 10.0, "koi_prad": 2.0}}
    predict = client.post("/predict", json=payload).json()
    explain = client.post("/explain", json=payload).json()
    assert explain["predicted_class"] == predict["predicted_class"]
    assert explain["report"] == predict["report"]
    assert explain["top_features"] == predict["top_features"]


def test_feedback_endpoint_records_a_correction(client, tmp_path, monkeypatch) -> None:
    import api.main as main

    log = FeedbackLog(tmp_path / "feedback.jsonl")
    monkeypatch.setattr(main, "feedback", log)
    response = client.post(
        "/feedback",
        json={
            "values": {"koi_period": 10.0},
            "predicted_class": "CONFIRMED",
            "corrected_class": "CANDIDATE",
            "comment": "eclipse",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["recorded"] is True
    assert body["corrected_class"] == "CANDIDATE"
    assert log.read()[0]["comment"] == "eclipse"


def test_feedback_endpoint_reads_the_log_back(client, tmp_path, monkeypatch) -> None:
    import api.main as main

    log = FeedbackLog(tmp_path / "feedback.jsonl")
    monkeypatch.setattr(main, "feedback", log)
    log.append({"koi_period": 1.0}, "CONFIRMED", "CANDIDATE")
    body = client.get("/feedback").json()
    assert body["summary"]["n_entries"] == 1
    assert len(body["entries"]) == 1


def test_openapi_schema_is_valid(client) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "/predict" in schema["paths"]
    assert "/feedback" in schema["paths"]


# ----------------------------------------------------------------------
# the real artifact, when it has been built
# ----------------------------------------------------------------------
@pytest.mark.skipif(not ARTIFACT.exists(), reason="the trained artifact is not built")
def test_the_real_artifact_answers() -> None:
    import api.main as main
    from fastapi.testclient import TestClient

    with TestClient(main.app) as test_client:
        body = test_client.post(
            "/predict",
            json={
                "values": {
                    "koi_period": 9.488,
                    "koi_duration": 2.9575,
                    "koi_depth": 615.8,
                    "koi_prad": 2.26,
                    "koi_teq": 793.0,
                    "koi_impact": 0.146,
                    "koi_insol": 93.59,
                    "koi_steff": 5455.0,
                    "koi_slogg": 4.467,
                    "koi_srad": 0.927,
                    "koi_kepmag": 15.347,
                }
            },
        ).json()
    # A confirmed planet from the archive comes back as a confirmed planet.
    assert body["predicted_class"] == "CONFIRMED"
    assert body["probability"] > 0.5
    assert body["needs_review"] is False
