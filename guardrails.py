"""
guardrails.py
=============
Enterprise input/output guardrails for the ICD-10 pipeline.

Input guardrails
----------------
  * `validate_input_payload`  - parse & validate raw JSON into a
    `PredictionRequest` (enforces JSON structure + per-field size limits).

Output guardrails
-----------------
  * `validate_prediction_output` - re-validate a model dict against the strict
    `ICDPrediction` schema (ICD-10 regex + confidence bounds).
  * `contains_blocked_content`   - lightweight safety net for toxic/harmful
    text (complements Vertex AI safety filters).

These functions raise `GuardrailError` on violation so the orchestrator can
fail closed with a clean error instead of leaking a partial/invalid result.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from config import settings
from logging_utils import get_logger
from schemas import ICDPrediction, PredictionRequest

logger = get_logger(__name__)


class GuardrailError(Exception):
    """Raised when an input or output guardrail is violated."""


# A conservative blocklist as a defence-in-depth complement to Vertex AI's
# built-in safety filters. Real deployments should rely primarily on the
# Vertex AI safety settings configured in Agent 2.
_BLOCKED_PATTERNS = [
    re.compile(r"\b(kill|harm|suicide)\s+(yourself|myself)\b", re.IGNORECASE),
]


# --------------------------------------------------------------------------- #
# INPUT GUARDRAILS
# --------------------------------------------------------------------------- #
def validate_input_payload(raw_payload: str | bytes | dict[str, Any]) -> PredictionRequest:
    """
    Validate and coerce a raw inbound payload into a `PredictionRequest`.

    Accepts a JSON string/bytes or an already-parsed dict.

    Raises
    ------
    GuardrailError
        If the payload is not valid JSON, is not an object, is missing
        required fields, or violates the per-field size limit.
    """
    # 1. Ensure we have a dict (parse JSON if needed).
    if isinstance(raw_payload, (str, bytes)):
        try:
            data = json.loads(raw_payload)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("input_guardrail.invalid_json", error=str(exc))
            raise GuardrailError("Payload is not valid JSON.") from exc
    elif isinstance(raw_payload, dict):
        data = raw_payload
    else:
        raise GuardrailError("Payload must be a JSON object, string or bytes.")

    if not isinstance(data, dict):
        raise GuardrailError("Payload must decode to a JSON object.")

    # 2. Validate against the Pydantic schema (structure + size limits).
    try:
        request = PredictionRequest(**data)
    except ValidationError as exc:
        logger.warning("input_guardrail.schema_violation", errors=exc.errors())
        raise GuardrailError(f"Input validation failed: {exc.errors()}") from exc

    logger.info(
        "input_guardrail.passed",
        specimen_len=len(request.specimen_source),
        diagnosis_len=len(request.diagnosis),
    )
    return request


def contains_blocked_content(text: str) -> bool:
    """Return True if `text` matches any blocked/harmful pattern."""
    return any(pattern.search(text) for pattern in _BLOCKED_PATTERNS)


# --------------------------------------------------------------------------- #
# OUTPUT GUARDRAILS
# --------------------------------------------------------------------------- #
def validate_prediction_output(candidate: dict[str, Any] | ICDPrediction) -> ICDPrediction:
    """
    Enforce the strict output schema on a candidate prediction.

    Validates ICD-10 code format, confidence bounds and justification length,
    and screens the justification for harmful content.

    Raises
    ------
    GuardrailError
        If the candidate fails schema validation or contains blocked content.
    """
    try:
        prediction = (
            candidate
            if isinstance(candidate, ICDPrediction)
            else ICDPrediction(**candidate)
        )
    except ValidationError as exc:
        logger.warning("output_guardrail.schema_violation", errors=exc.errors())
        raise GuardrailError(f"Output validation failed: {exc.errors()}") from exc

    # Confidence floor (configurable business rule).
    if prediction.confidence_score < settings.min_confidence_threshold:
        raise GuardrailError(
            f"Confidence {prediction.confidence_score:.3f} below minimum "
            f"threshold {settings.min_confidence_threshold:.3f}."
        )

    # Safety screen on any free-text field leaving the system.
    if contains_blocked_content(prediction.justification):
        logger.warning("output_guardrail.blocked_content")
        raise GuardrailError("Output contained blocked/harmful content.")

    logger.info(
        "output_guardrail.passed",
        icd_code=prediction.predicted_icd_code,
        confidence=prediction.confidence_score,
    )
    return prediction
