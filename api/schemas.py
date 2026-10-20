"""Request and response schemas for the ExoVet API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from exovet.data import CORE_FEATURES


class FeatureValues(BaseModel):
    """The physical measurements of one object.

    Every field is optional. A missing value is imputed with the training
    median inside the pipeline, so a user can submit a partial form.
    """

    model_config = ConfigDict(extra="forbid")

    koi_period: Optional[float] = Field(None, description="Orbital period, days")
    koi_duration: Optional[float] = Field(None, description="Transit duration, hours")
    koi_depth: Optional[float] = Field(None, description="Transit depth, ppm")
    koi_prad: Optional[float] = Field(None, description="Planet radius, Earth radii")
    koi_teq: Optional[float] = Field(None, description="Equilibrium temperature, K")
    koi_impact: Optional[float] = Field(None, description="Impact parameter")
    koi_insol: Optional[float] = Field(None, description="Insolation, Earth flux")
    koi_steff: Optional[float] = Field(None, description="Stellar effective temperature, K")
    koi_slogg: Optional[float] = Field(None, description="Stellar surface gravity, log g")
    koi_srad: Optional[float] = Field(None, description="Stellar radius, solar radii")
    koi_kepmag: Optional[float] = Field(None, description="Kepler magnitude, mag")

    def to_row(self) -> Dict[str, float]:
        """The submitted values as a feature dictionary, dropping the gaps."""
        values = self.model_dump()
        return {key: value for key, value in values.items() if value is not None}


class PredictRequest(BaseModel):
    """A vetting request."""

    model_config = ConfigDict(extra="forbid")

    values: FeatureValues = Field(..., description="Measured properties of the object")
    object_id: Optional[str] = Field(
        None, description="Free-form identifier, for example a TOI or KOI number"
    )


class FeatureContribution(BaseModel):
    """One SHAP contribution."""

    feature: str
    display_name: str
    value: Optional[float] = None
    formatted_value: str
    shap_value: float
    direction: str
    magnitude: float


class PredictResponse(BaseModel):
    """The vetting result for one object."""

    predicted_class: str
    probability: float
    probabilities: Dict[str, float]
    prediction_set: List[str]
    needs_review: bool
    top_features: List[FeatureContribution]
    report: str
    object_id: Optional[str] = None
    model: str
    conformal_alpha: float
    target_coverage: float
    warnings: List[str] = Field(default_factory=list)


class ExplainResponse(PredictResponse):
    """The same result, with the explanation fields repeated for convenience."""


class FeedbackRequest(BaseModel):
    """A user correction of a previous vetting result."""

    model_config = ConfigDict(extra="forbid")

    values: FeatureValues
    predicted_class: str
    corrected_class: str
    comment: Optional[str] = None
    object_id: Optional[str] = None


class FeedbackResponse(BaseModel):
    """The record of a logged correction."""

    recorded: bool
    entry_id: str
    predicted_class: str
    corrected_class: str


class FeatureInfo(BaseModel):
    """What the API knows about one feature."""

    name: str
    display_name: str
    unit: str
    description: str


class ModelInfo(BaseModel):
    """Metadata about the loaded model."""

    model: str
    classes: List[str]
    features: List[str]
    conformal_alpha: float
    target_coverage: float
    trained_on: str
    metrics: Dict[str, object]
    warnings: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Whether the service is ready to answer."""

    status: str
    model_loaded: bool
    version: str


#: Short human-readable descriptions of the features, for the form.
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "koi_period": "Time the planet takes to complete one orbit of its star.",
    "koi_duration": "How long the transit lasts, from first to fourth contact.",
    "koi_depth": "The size of the flux dip, in parts per million.",
    "koi_prad": "The radius of the planet, in Earth radii.",
    "koi_teq": "The temperature the planet would have if it were in equilibrium.",
    "koi_impact": "How close the planet passes to the centre of the star.",
    "koi_insol": "The stellar energy the planet receives, relative to Earth.",
    "koi_steff": "The effective temperature of the host star.",
    "koi_slogg": "The surface gravity of the host star.",
    "koi_srad": "The radius of the host star, in solar radii.",
    "koi_kepmag": "The brightness of the host star in the Kepler band.",
}

#: The order the form should present the features in.
FEATURE_ORDER = list(CORE_FEATURES)
