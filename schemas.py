"""
schemas.py
==========
Pydantic v2 data contracts shared across the multi-agent pipeline.

These models enforce the input/output guardrails required by the system:
  * `PredictionRequest`  - validated inbound payload.
  * `SanitizedText`      - Agent 1 output (PHI-free).
  * `FormalClinicalText` - Agent 2 output (formalised prose + concepts).
  * `HistoricalMatch`    - a single BigQuery RAG neighbour.
  * `ICDPrediction`      - the strict final output schema (Agent 3).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

from config import settings

# Compile the ICD-10 validation pattern once at import time.
_ICD10_PATTERN = re.compile(settings.icd10_regex)


class PredictionRequest(BaseModel):
    """Inbound request payload. Length limits act as an input guardrail."""

    specimen_source: str = Field(
        ...,
        min_length=1,
        description="Anatomic source of the specimen (e.g. 'Left breast, core biopsy').",
    )
    diagnosis: str = Field(
        ...,
        min_length=1,
        description="Raw pathology diagnosis text / clinical shorthand.",
    )

    @field_validator("specimen_source", "diagnosis")
    @classmethod
    def _enforce_length_limit(cls, value: str) -> str:
        """Reject oversized fields to prevent prompt-injection / DoS via size."""
        if len(value) > settings.max_input_chars:
            raise ValueError(
                f"Field exceeds maximum of {settings.max_input_chars} characters."
            )
        return value.strip()


class SanitizedText(BaseModel):
    """Output of Agent 1 - guaranteed PHI-scrubbed text."""

    specimen_source: str
    diagnosis: str
    phi_found: bool = Field(
        default=False,
        description="True if any PHI was detected and redacted.",
    )
    redaction_method: str = Field(
        default="dlp",
        description="Which redactor produced this output: 'dlp' or 'regex'.",
    )


class FormalClinicalText(BaseModel):
    """Output of Agent 2 - formalised clinical prose and extracted concepts."""

    formal_text: str = Field(
        ..., description="Textbook-style formal medical prose."
    )
    clinical_concepts: list[str] = Field(
        default_factory=list,
        description="Key extracted clinical concepts / entities.",
    )
    embedding: list[float] | None = Field(
        default=None,
        description="PubMedBERT sentence embedding reused by Agent 3 for retrieval.",
    )


class HistoricalMatch(BaseModel):
    """A single nearest-neighbour record returned by BigQuery RAG."""

    icd_code: str
    diagnosis_text: str
    specimen_source: str | None = None
    # For vector search this is cosine distance (lower == closer). For keyword
    # search this is an inverted relevance score. Normalised downstream.
    distance: float = Field(default=0.0)
    frequency: int = Field(default=1, ge=0)


class ICDPrediction(BaseModel):
    """
    Strict final output schema (Agent 3).

    The `predicted_icd_code` field is validated against the ICD-10-CM format
    regex as an output guardrail - malformed codes are rejected before the
    result can leave the system.
    """

    predicted_icd_code: str = Field(
        ..., description="Predicted ICD-10-CM code, e.g. 'C50.911'."
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in [0, 1]."
    )
    justification: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="One-sentence explanation grounded in the historical match.",
    )

    @field_validator("predicted_icd_code")
    @classmethod
    def _validate_icd10_format(cls, value: str) -> str:
        """Enforce a valid ICD-10-CM code format (output guardrail)."""
        code = value.strip().upper()
        if not _ICD10_PATTERN.match(code):
            raise ValueError(f"'{value}' is not a valid ICD-10-CM code format.")
        return code


class PipelineResult(BaseModel):
    """Top-level API response wrapping the prediction plus trace metadata."""

    success: bool
    prediction: ICDPrediction | None = None
    error: str | None = None
    phi_found: bool = False
    stages_completed: list[str] = Field(default_factory=list)
