"""
pubmedbert.py
=============
Local Hugging Face **PubMedBERT** model wrapper.

PubMedBERT (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`) is
a biomedical **encoder** (BERT architecture). It is not a generative decoder,
so it is used here for what it excels at:

  * Producing 768-dimensional contextual **embeddings** of clinical text
    (used by Agent 3 for BigQuery vector retrieval).
  * **Concept extraction** by ranking candidate terms via cosine similarity of
    each term's contextual embedding against the whole-sentence embedding.

The model is loaded once as a process-wide singleton. All inference is
CPU/GPU-bound and blocking, so callers should invoke `analyze()` /`embed()`
inside `asyncio.to_thread(...)`.
"""

from __future__ import annotations

import re
import threading

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from config import settings
from logging_utils import get_logger

logger = get_logger(__name__)

# Minimal English/medical stopword set for concept-candidate filtering.
_STOPWORDS = {
    "the", "and", "with", "for", "of", "in", "on", "at", "to", "from", "by",
    "a", "an", "is", "are", "was", "were", "be", "been", "no", "not", "left",
    "right", "grade", "pt", "patient", "showing", "shows", "seen", "noted",
    "consistent", "compatible", "identified", "present", "negative", "positive",
}

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]{2,}")


class PubMedBERTModel:
    """Thread-safe singleton wrapper around a PubMedBERT encoder."""

    _instance: "PubMedBERTModel | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._device = self._resolve_device(settings.pubmedbert_device)
        logger.info(
            "pubmedbert.loading",
            model=settings.pubmedbert_model_name,
            device=self._device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(settings.pubmedbert_model_name)
        self._model = AutoModel.from_pretrained(settings.pubmedbert_model_name)
        self._model.to(self._device)
        self._model.eval()
        logger.info("pubmedbert.loaded")

    # ------------------------------------------------------------------ #
    # Singleton accessor
    # ------------------------------------------------------------------ #
    @classmethod
    def instance(cls) -> "PubMedBERTModel":
        """Return the process-wide singleton, constructing it on first use."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @staticmethod
    def _resolve_device(configured: str) -> str:
        if configured != "auto":
            return configured
        return "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------ #
    # Inference helpers
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _forward(self, text: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Run a single forward pass.

        Returns
        -------
        (sentence_embedding, token_embeddings, tokens)
            * sentence_embedding: mean-pooled, L2-normalised (D,).
            * token_embeddings:   per-token hidden states (T, D).
            * tokens:             decoded token strings aligned to rows above.
        """
        encoded = self._tokenizer(
            text or "",
            return_tensors="pt",
            truncation=True,
            max_length=settings.pubmedbert_max_length,
        ).to(self._device)

        outputs = self._model(**encoded)
        hidden = outputs.last_hidden_state[0]          # (T, D)
        mask = encoded["attention_mask"][0].unsqueeze(-1).float()  # (T, 1)

        # Mean pooling over real (non-padding) tokens.
        summed = (hidden * mask).sum(dim=0)
        counts = mask.sum(dim=0).clamp(min=1e-9)
        sentence = (summed / counts)
        sentence = torch.nn.functional.normalize(sentence, p=2, dim=0)

        tokens = self._tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        return (
            sentence.cpu().numpy(),
            hidden.cpu().numpy(),
            tokens,
        )

    def embed(self, text: str) -> list[float]:
        """Return the L2-normalised sentence embedding as a plain list."""
        sentence, _, _ = self._forward(text)
        return sentence.astype(float).tolist()

    def analyze(self, text: str, max_concepts: int | None = None) -> tuple[list[float], list[str]]:
        """
        Return `(embedding, concepts)` for the given text.

        Concepts are content words ranked by cosine similarity between each
        word's mean subword embedding and the whole-sentence embedding.
        """
        max_concepts = max_concepts or settings.max_concepts
        sentence, token_embeddings, tokens = self._forward(text)

        # Reconstruct whole words from WordPiece tokens and average their vectors.
        word_vectors: dict[str, list[np.ndarray]] = {}
        current_word = ""
        current_vecs: list[np.ndarray] = []

        def _flush() -> None:
            if current_word:
                word = current_word.lower()
                if _WORD_RE.fullmatch(word) and word not in _STOPWORDS:
                    word_vectors.setdefault(word, [])
                    word_vectors[word].extend(current_vecs)

        for tok, vec in zip(tokens, token_embeddings):
            if tok in ("[CLS]", "[SEP]", "[PAD]"):
                continue
            if tok.startswith("##"):
                current_word += tok[2:]
                current_vecs.append(vec)
            else:
                _flush()
                current_word = tok
                current_vecs = [vec]
        _flush()

        # Rank words by cosine similarity to the sentence embedding.
        scored: list[tuple[str, float]] = []
        for word, vecs in word_vectors.items():
            wv = np.mean(vecs, axis=0)
            norm = np.linalg.norm(wv)
            if norm == 0:
                continue
            sim = float(np.dot(sentence, wv / norm))
            scored.append((word, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        concepts = [w for w, _ in scored[:max_concepts]]
        return sentence.astype(float).tolist(), concepts


def get_pubmedbert_model() -> PubMedBERTModel:
    """Convenience accessor for the PubMedBERT singleton."""
    return PubMedBERTModel.instance()
