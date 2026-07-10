"""
agents/clinical_reasoning.py
============================
Agent 2 - The Clinical Reasoning Agent (PubMedBERT translator).

Responsibility
--------------
Takes the PHI-free text from Agent 1 and uses a locally-hosted **PubMedBERT**
biomedical encoder to:
  1. Produce a formal, normalised clinical representation of the input.
  2. Extract the core clinical concepts / entities (ranked by semantic
     relevance to the whole report).
  3. Compute a 768-dim sentence embedding that Agent 3 reuses for BigQuery
     vector retrieval (single embedding pass, no duplicate model calls).

Why PubMedBERT
--------------
PubMedBERT is an encoder (not a generative decoder), so it cannot "write" prose
the way a chat LLM does. It is state-of-the-art for biomedical *understanding*:
we leverage it for embeddings and concept ranking, and assemble the formal
representation deterministically. All model inference is blocking and therefore
executed in a worker thread via `asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio

from config import settings
from logging_utils import get_logger
from pubmedbert import PubMedBERTModel, get_pubmedbert_model
from schemas import FormalClinicalText, SanitizedText

logger = get_logger(__name__)


class ClinicalReasoningAgent:
    """Agent 2: formalises clinical text using a local PubMedBERT encoder."""

    def __init__(self, model: PubMedBERTModel | None = None) -> None:
        # Reuse the process-wide PubMedBERT singleton (loaded once).
        self._model = model or get_pubmedbert_model()

    @staticmethod
    def _build_formal_text(sanitized: SanitizedText, concepts: list[str]) -> str:
        """
        Assemble a formal, textbook-style clinical representation.

        PubMedBERT cannot generate free text, so we produce a deterministic,
        normalised description grounded strictly in the (PHI-free) input plus
        the model-ranked concepts - never inventing unsupported findings.
        """
        concept_str = ", ".join(concepts) if concepts else "none extracted"
        return (
            f"Specimen source: {sanitized.specimen_source.strip()}. "
            f"Diagnostic findings: {sanitized.diagnosis.strip()}. "
            f"Salient clinical concepts: {concept_str}."
        )

    async def run(self, sanitized: SanitizedText) -> FormalClinicalText:
        """
        Formalise the sanitized clinical text and compute its embedding.

        Raises
        ------
        RuntimeError
            If PubMedBERT inference fails.
        """
        logger.info("clinical_reasoning.start")

        # Concatenate the two fields for a single joint representation.
        combined = f"{sanitized.specimen_source}. {sanitized.diagnosis}"

        try:
            # Run the blocking model inference off the event loop.
            embedding, concepts = await asyncio.to_thread(
                self._model.analyze, combined, settings.max_concepts
            )
        except Exception as exc:  # noqa: BLE001 - surface as a clean failure
            logger.error("clinical_reasoning.pubmedbert_failed", error=str(exc))
            raise RuntimeError(f"PubMedBERT inference failed: {exc}") from exc

        formal_text = self._build_formal_text(sanitized, concepts)
        result = FormalClinicalText(
            formal_text=formal_text,
            clinical_concepts=concepts,
            embedding=embedding,
        )
        logger.info(
            "clinical_reasoning.done",
            concept_count=len(result.clinical_concepts),
            embedding_dim=len(embedding),
        )
        return result
