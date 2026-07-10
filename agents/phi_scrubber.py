"""
agents/phi_scrubber.py
======================
Agent 1 - The Privacy & Sanitization Agent (PHI Scrubber).

Responsibility
--------------
Takes the raw `specimen_source` and `diagnosis` text and removes any Protected
Health Information (PHI) *before* it can reach any downstream LLM.

Primary implementation
-----------------------
Uses the Google Cloud Data Loss Prevention (DLP) API to inspect and redact a
configurable set of InfoTypes (names, dates, MRNs, SSNs, addresses, ...).

Fail-safe design
----------------
Redaction must never fail-open. If the DLP API is unreachable and the regex
fallback is enabled, a strict local regex redactor runs instead. If neither is
available, the agent raises so the orchestrator fails closed.
"""

from __future__ import annotations

import asyncio
import re

from google.api_core.exceptions import GoogleAPIError
from google.cloud import dlp_v2
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings
from logging_utils import get_logger
from schemas import PredictionRequest, SanitizedText

logger = get_logger(__name__)

# Regex fallback patterns (defence in depth - deliberately aggressive).
_REGEX_REDACTORS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),                       # SSN
    (re.compile(r"\b\d{10}\b"), "[NPI]"),                                  # 10-digit IDs
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),              # email
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), "[DATE]"),          # dates
    (re.compile(r"\bMRN[:#]?\s*\w+\b", re.IGNORECASE), "[MRN]"),           # MRN
    (re.compile(r"\bDOB[:#]?\s*[\w/-]+\b", re.IGNORECASE), "[DOB]"),       # DOB
]


class PHIScrubberAgent:
    """Agent 1: strips PHI from raw clinical text using GCP DLP."""

    def __init__(self) -> None:
        self._parent = f"projects/{settings.gcp_project_id}/locations/{settings.dlp_location}"
        # Lazily-initialised async DLP client (created on first use).
        self._client: dlp_v2.DlpServiceAsyncClient | None = None

    async def _get_client(self) -> dlp_v2.DlpServiceAsyncClient:
        if self._client is None:
            self._client = dlp_v2.DlpServiceAsyncClient()
        return self._client

    def _build_deidentify_config(self) -> dict:
        """Build DLP inspect + de-identify (redact) configuration."""
        info_types = [{"name": name} for name in settings.dlp_info_types]
        return {
            "inspect_config": {
                "info_types": info_types,
                "min_likelihood": settings.dlp_min_likelihood,
            },
            "deidentify_config": {
                "info_type_transformations": {
                    "transformations": [
                        {
                            "primitive_transformation": {
                                # Replace each finding with its InfoType name,
                                # e.g. "[PERSON_NAME]".
                                "replace_with_info_type_config": {}
                            }
                        }
                    ]
                }
            },
        }

    @retry(
        retry=retry_if_exception_type(GoogleAPIError),
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _dlp_redact(self, text: str) -> tuple[str, bool]:
        """
        Redact PHI from a single string via the DLP API.

        Returns
        -------
        (redacted_text, phi_found)
        """
        if not text:
            return text, False

        client = await self._get_client()
        config = self._build_deidentify_config()

        response = await client.deidentify_content(
            request={
                "parent": self._parent,
                "inspect_config": config["inspect_config"],
                "deidentify_config": config["deidentify_config"],
                "item": {"value": text},
            },
            timeout=settings.request_timeout_seconds,
        )
        redacted = response.item.value
        phi_found = redacted != text
        return redacted, phi_found

    def _regex_redact(self, text: str) -> tuple[str, bool]:
        """Strict local fallback redactor used only when DLP is unavailable."""
        original = text
        for pattern, replacement in _REGEX_REDACTORS:
            text = pattern.sub(replacement, text)
        return text, text != original

    async def run(self, request: PredictionRequest) -> SanitizedText:
        """
        Sanitize the request's specimen source and diagnosis.

        Raises
        ------
        RuntimeError
            If DLP fails and no fallback is permitted (fail closed).
        """
        logger.info("phi_scrubber.start")

        try:
            # Redact both fields concurrently for lower latency.
            (spec_clean, spec_phi), (diag_clean, diag_phi) = await asyncio.gather(
                self._dlp_redact(request.specimen_source),
                self._dlp_redact(request.diagnosis),
            )
            method = "dlp"
            phi_found = spec_phi or diag_phi

        except GoogleAPIError as exc:
            logger.error("phi_scrubber.dlp_failed", error=str(exc))
            if not settings.dlp_enable_regex_fallback:
                # Never fail-open on PHI - propagate so the pipeline stops.
                raise RuntimeError("DLP redaction failed and fallback disabled.") from exc

            logger.warning("phi_scrubber.using_regex_fallback")
            spec_clean, spec_phi = self._regex_redact(request.specimen_source)
            diag_clean, diag_phi = self._regex_redact(request.diagnosis)
            method = "regex"
            phi_found = spec_phi or diag_phi

        logger.info("phi_scrubber.done", phi_found=phi_found, method=method)
        return SanitizedText(
            specimen_source=spec_clean,
            diagnosis=diag_clean,
            phi_found=phi_found,
            redaction_method=method,
        )
