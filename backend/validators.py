"""
validators.py — Pydantic v2 input/output schema validation.

Validates incoming prediction request bodies and enforces
consistent response shapes.

Usage:
    from backend.validators import WeatherPredictionRequest, ValidationError

    try:
        req = WeatherPredictionRequest(**request.get_json())
    except ValidationError as e:
        return jsonify({"error": "Validation failed", "details": e.errors()}), 400

    location = req.location   # already sanitized
    days     = req.days       # guaranteed 1-14
"""

import re
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic import — graceful fallback so the rest of the app still works
# if pydantic is not yet installed.
# ---------------------------------------------------------------------------
try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    from pydantic import ValidationError          # re-export for callers
    PYDANTIC_AVAILABLE = True
    logger.debug("✅ Pydantic available")
except ImportError:
    PYDANTIC_AVAILABLE = False
    logger.warning(
        "⚠️  Pydantic not installed — WeatherPredictionRequest will use "
        "a lightweight fallback. Run: pip install pydantic>=2.0.0"
    )

    # Lightweight stand-in so import never hard-fails
    class ValidationError(Exception):  # type: ignore[no-redef]
        def errors(self):
            return [{"msg": str(self)}]

    class _FallbackModel:  # type: ignore[too-few-public-methods]
        """Minimal dict-wrapper used when Pydantic is unavailable."""
        def __init__(self, **data):
            for k, v in data.items():
                setattr(self, k, v)

    BaseModel = _FallbackModel  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Allowed values
# ---------------------------------------------------------------------------
ALLOWED_METHODS = {
    "ensemble",
    "rag_llm",
    "langgraph",
    "statistical",
    "multiquery_rag",
    "langchain_rag",
    "hybrid",
}

ALLOWED_SEASONS = {"auto", "Winter", "Spring", "Summer", "Autumn"}

# Location: letters, digits, spaces, comma, hyphen, dot, parentheses, slash
_LOCATION_SAFE = re.compile(r"[^a-zA-Z0-9\s,\-\.\(\)/]")


# ---------------------------------------------------------------------------
if PYDANTIC_AVAILABLE:

    class WeatherPredictionRequest(BaseModel):
        """Validated and sanitized incoming prediction request."""

        location: str = Field(
            ...,
            min_length=2,
            max_length=100,
            description="City or location name",
        )
        days: int = Field(
            default=3,
            ge=1,
            le=14,
            description="Forecast horizon in days (1–14)",
        )
        method: Optional[str] = Field(
            default="ensemble",
            description=f"Prediction method: one of {sorted(ALLOWED_METHODS)}",
        )
        season: Optional[str] = Field(
            default="auto",
            description=f"Season override: one of {sorted(ALLOWED_SEASONS)}",
        )
        enable_multiquery: bool = Field(
            default=True,
            description="Allow multi-query RAG expansion",
        )

        @field_validator("location")
        @classmethod
        def sanitize_location(cls, v: str) -> str:
            clean = _LOCATION_SAFE.sub("", v).strip()
            if not clean:
                raise ValueError(
                    "Location contains no valid characters after sanitization"
                )
            return clean

        @field_validator("method")
        @classmethod
        def validate_method(cls, v: Optional[str]) -> Optional[str]:
            if v is not None and v not in ALLOWED_METHODS:
                raise ValueError(
                    f"'{v}' is not a valid method. "
                    f"Choose from: {sorted(ALLOWED_METHODS)}"
                )
            return v

        @field_validator("season")
        @classmethod
        def validate_season(cls, v: Optional[str]) -> Optional[str]:
            if v is not None and v not in ALLOWED_SEASONS:
                raise ValueError(
                    f"'{v}' is not a valid season. "
                    f"Choose from: {sorted(ALLOWED_SEASONS)}"
                )
            return v

    # -----------------------------------------------------------------------
    class PredictionResponseValidator(BaseModel):
        """
        Validates the shape of a prediction response dict.

        Used by the eval pipeline and monitoring to assert response contracts.
        """

        success: bool
        method:  str  = Field(min_length=1)
        confidence_level: str

        # Optional — may be absent on statistical fallback
        prediction:  Optional[str]  = None
        model_used:  Optional[str]  = None
        trace_id:    Optional[str]  = None
        quality_score: Optional[float] = None
        warnings:    Optional[List[str]] = None

        @field_validator("confidence_level")
        @classmethod
        def valid_confidence(cls, v: str) -> str:
            allowed = {"Very Low", "Low", "Medium", "High", "Very High"}
            if v not in allowed:
                raise ValueError(
                    f"confidence_level must be one of {sorted(allowed)}, got '{v}'"
                )
            return v

        @model_validator(mode="after")
        def prediction_required_on_success(self) -> "PredictionResponseValidator":
            if self.success and (
                not self.prediction or len(self.prediction) < 50
            ):
                raise ValueError(
                    "Response has success=True but prediction is missing or "
                    "too short (< 50 chars)"
                )
            return self

else:
    # Fallback stubs when Pydantic is not installed
    class WeatherPredictionRequest:  # type: ignore[no-redef]
        def __init__(self, **data):
            self.location          = str(data.get("location", "")).strip()
            self.days              = int(data.get("days", 3))
            self.method            = data.get("method", "ensemble")
            self.season            = data.get("season", "auto")
            self.enable_multiquery = bool(data.get("enable_multiquery", True))

            if not self.location:
                raise ValidationError("location is required")
            if not (1 <= self.days <= 14):
                raise ValidationError("days must be between 1 and 14")

    class PredictionResponseValidator:  # type: ignore[no-redef]
        def __init__(self, **data):
            for k, v in data.items():
                setattr(self, k, v)
