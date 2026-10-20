"""The ExoVet API: vet a transiting signal and explain the verdict.

Endpoints:

``GET  /health``    is the service ready?
``GET  /model``     what model is loaded, and what is it good for?
``GET  /features``  the features the form should offer.
``POST /predict``   vet an object.
``POST /explain``   the same, with the explanation fields repeated.
``POST /feedback``  log a correction of a previous verdict.
``GET  /feedback``  the correction log and its summary.

The split conformal guarantee is the reason the response carries a
prediction set and a ``needs_review`` flag. A set with more than one
class means the model abstains, and the object goes to a human.

Run it with:

    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api.schemas import (  # noqa: E402
    FEATURE_DESCRIPTIONS,
    FEATURE_ORDER,
    ExplainResponse,
    FeatureInfo,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
)
from api.store import WARNINGS, FeedbackLog, ModelStore, feedback_path_from_env  # noqa: E402
from exovet.report import display_name, feature_unit  # noqa: E402

VERSION = "0.2.0"

store = ModelStore()
feedback = FeedbackLog(feedback_path_from_env())


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the fitted pipeline once, at start-up."""
    store.load()
    yield


app = FastAPI(
    title="ExoVet",
    version=VERSION,
    description=(
        "Explainable, uncertainty-aware vetting of transiting signals. "
        "Returns a class, a calibrated probability, a conformal prediction "
        "set and a plain-English report built only from the top SHAP features."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _model_info() -> ModelInfo:
    pipeline = store.require()
    metrics = pipeline.metrics.get("calibrated", {})
    return ModelInfo(
        model=pipeline.model_name,
        classes=list(pipeline.classes),
        features=list(pipeline.features or []),
        conformal_alpha=pipeline.conformal_alpha,
        target_coverage=1.0 - pipeline.conformal_alpha,
        trained_on="NASA Exoplanet Archive, Kepler KOI cumulative table",
        metrics={
            "macro_f1": metrics.get("macro_f1"),
            "macro_pr_auc": metrics.get("macro_pr_auc"),
            "expected_calibration_error": metrics.get("expected_calibration_error"),
            "log_loss": metrics.get("log_loss"),
            "brier_score": metrics.get("brier_score"),
        },
        warnings=list(WARNINGS),
    )


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    """Whether the service can answer."""
    return HealthResponse(
        status="ok" if store.ready else "degraded",
        model_loaded=store.ready,
        version=VERSION,
    )


@app.get("/model", response_model=ModelInfo, tags=["service"])
def model_info() -> ModelInfo:
    """Metadata about the loaded model."""
    if not store.ready:
        raise HTTPException(status_code=503, detail=store.load_error)
    return _model_info()


@app.get("/features", response_model=List[FeatureInfo], tags=["service"])
def features() -> List[FeatureInfo]:
    """The features the form should offer, with units and descriptions."""
    return [
        FeatureInfo(
            name=name,
            display_name=display_name(name),
            unit=feature_unit(name),
            description=FEATURE_DESCRIPTIONS.get(name, ""),
        )
        for name in FEATURE_ORDER
    ]


@app.post("/predict", response_model=PredictResponse, tags=["vetting"])
def predict(request: PredictRequest) -> PredictResponse:
    """Vet one object and explain the verdict."""
    if not store.ready:
        raise HTTPException(status_code=503, detail=store.load_error)
    result = store.predict(request.values.to_row())
    result["object_id"] = request.object_id
    return PredictResponse(**result)


@app.post("/explain", response_model=ExplainResponse, tags=["vetting"])
def explain(request: PredictRequest) -> ExplainResponse:
    """Vet one object, focusing on the explanation."""
    if not store.ready:
        raise HTTPException(status_code=503, detail=store.load_error)
    result = store.predict(request.values.to_row())
    result["object_id"] = request.object_id
    return ExplainResponse(**result)


@app.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
def log_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Record a correction of a previous verdict."""
    entry = feedback.append(
        values=request.values.to_row(),
        predicted_class=request.predicted_class,
        corrected_class=request.corrected_class,
        comment=request.comment,
        object_id=request.object_id,
    )
    return FeedbackResponse(
        recorded=True,
        entry_id=entry["id"],
        predicted_class=entry["predicted_class"],
        corrected_class=entry["corrected_class"],
    )


@app.get("/feedback", tags=["feedback"])
def read_feedback(limit: int = 50) -> dict:
    """The correction log and a summary of it."""
    return {"entries": feedback.read(limit=limit), "summary": feedback.summary()}


# ----------------------------------------------------------------------
# the built front end, when it has been built
# ----------------------------------------------------------------------
WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"

if WEB_DIST.exists():
    app.mount("/app", StaticFiles(directory=str(WEB_DIST), html=True), name="app")

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        """Send the browser to the single-page app."""
        return RedirectResponse(url="/app/")
