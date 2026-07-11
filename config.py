"""
config.py
=========
Centralised, environment-driven configuration for the ICD-10 Multi-Agent
Orchestration System.

All secrets / project-specific values are read from environment variables
(loaded from a local `.env` during development) so that nothing sensitive is
hard-coded. In production on GCP these are injected via Cloud Run / GKE
environment variables or Secret Manager.

Usage
-----
    from config import settings
    print(settings.gcp_project_id)
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings validated at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Google Cloud Platform - core project settings
    # ------------------------------------------------------------------ #
    gcp_project_id: str = Field(
        default="qwiklabs-gcp-03-d9bd89368565",
        description="GCP project ID hosting Vertex AI, DLP and BigQuery.",
    )
    gcp_location: str = Field(
        default="us-central1",
        description="Default GCP region for Vertex AI and BigQuery jobs.",
    )

    # ------------------------------------------------------------------ #
    # Agent 1 - Data Loss Prevention (PHI scrubbing)
    # ------------------------------------------------------------------ #
    # The DLP API location. "global" works for most inspect templates.
    dlp_location: str = Field(default="global")
    # InfoTypes to redact. Kept broad for clinical/pathology safety.
    dlp_info_types: list[str] = Field(
        default_factory=lambda: [
            "PERSON_NAME",
            "DATE_OF_BIRTH",
            "DATE",
            "AGE",
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "US_SOCIAL_SECURITY_NUMBER",
            "MEDICAL_RECORD_NUMBER",
            "US_HEALTHCARE_NPI",
            "STREET_ADDRESS",
            "LOCATION",
            "CREDIT_CARD_NUMBER",
        ]
    )
    # Minimum DLP finding likelihood to redact (POSSIBLE | LIKELY | VERY_LIKELY).
    dlp_min_likelihood: str = Field(default="POSSIBLE")
    # When True and DLP fails, fall back to the regex-based scrubber instead
    # of hard-failing (fail-open on redaction is NEVER allowed for PHI, so the
    # fallback is a *stricter* local redactor, not a bypass).
    dlp_enable_regex_fallback: bool = Field(default=True)

    # ------------------------------------------------------------------ #
    # Agent 2 - Clinical Reasoning (local PubMedBERT encoder)
    # ------------------------------------------------------------------ #
    # PubMedBERT is a biomedical BERT encoder used for embeddings + concept
    # extraction. Microsoft also publishes it under the "BiomedBERT" name.
    pubmedbert_model_name: str = Field(
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        description="Hugging Face model id for the local PubMedBERT encoder.",
    )
    # Inference device: 'auto' -> cuda if available else cpu, or force 'cpu'/'cuda'.
    pubmedbert_device: str = Field(default="cpu")
    # Max sequence length fed to the tokenizer.
    pubmedbert_max_length: int = Field(default=512, ge=16, le=512)
    # Embedding dimensionality (PubMedBERT base = 768). Must match BigQuery table.
    embedding_dim: int = Field(default=768)
    # Maximum number of clinical concepts to extract from the text.
    max_concepts: int = Field(default=12, ge=1, le=50)

    # ------------------------------------------------------------------ #
    # Agent 3 - BigQuery RAG & Validation
    # ------------------------------------------------------------------ #
    bq_dataset: str = Field(
        default="pathology",
        description="BigQuery dataset containing historical pathology data.",
    )
    bq_history_table: str = Field(
        default="historical_transactions",
        description="Table with ~1M historical pathology transactions.",
    )
    # Number of nearest neighbours to retrieve from BigQuery vector search.
    rag_top_k: int = Field(default=10, ge=1, le=100)
    # If vector search is unavailable, use keyword/BM25-style search instead.
    rag_use_vector_search: bool = Field(default=True)

    # ------------------------------------------------------------------ #
    # Orchestration - Google ADK (Agent Development Kit)
    # ------------------------------------------------------------------ #
    adk_app_name: str = Field(
        default="icd_predictor",
        description="ADK app name used for session/runner scoping.",
    )
    adk_user_id: str = Field(
        default="system",
        description="ADK user id used when running the pipeline.",
    )

    # ------------------------------------------------------------------ #
    # Guardrails - input / output limits
    # ------------------------------------------------------------------ #
    max_input_chars: int = Field(
        default=8000,
        description="Reject any single field longer than this many characters.",
    )
    # ICD-10-CM validation regex:
    #   1 alpha + 2 alphanumeric + optional '.' + up to 4 alphanumeric chars.
    icd10_regex: str = Field(
        default=r"^[A-TV-Z][0-9][0-9AB](?:\.[0-9A-TV-Z]{1,4})?$",
    )
    min_confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    # ------------------------------------------------------------------ #
    # Operational
    # ------------------------------------------------------------------ #
    log_level: str = Field(default="INFO")
    request_timeout_seconds: float = Field(default=60.0, gt=0)
    # Total retry attempts for transient GCP errors.
    max_retries: int = Field(default=3, ge=1, le=10)

    @property
    def bq_history_fqn(self) -> str:
        """Fully-qualified `project.dataset.table` name for the history table."""
        return f"{self.gcp_project_id}.{self.bq_dataset}.{self.bq_history_table}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


# Module-level convenience singleton.
settings: Settings = get_settings()
