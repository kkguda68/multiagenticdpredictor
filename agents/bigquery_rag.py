"""
agents/bigquery_rag.py
======================
Agent 3 - The BigQuery RAG & Validation Agent.

Responsibility
--------------
Takes the formal clinical text from Agent 2 and searches a BigQuery table of
~1,000,000 historical pathology transactions to find the closest matches, then
formulates the final prediction:
    * predicted_icd_code
    * confidence_score
    * justification

Retrieval strategies
--------------------
1. Vector search (preferred): embeds the query with a Vertex AI text-embedding
   model registered as a BigQuery ML remote model, then runs `VECTOR_SEARCH`
   against a pre-computed embedding column / vector index.
2. Keyword search (fallback): a `LIKE`/token overlap query when vector search
   is disabled or unavailable.

Confidence & justification
--------------------------
Confidence is derived from (a) how close the nearest neighbours are and (b) how
frequently the winning ICD code appears among the top-k neighbours. The
justification is a single grounded sentence referencing the best match.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

from google.api_core.exceptions import GoogleAPIError
from google.cloud import bigquery
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings
from logging_utils import get_logger
from pubmedbert import get_pubmedbert_model
from schemas import FormalClinicalText, HistoricalMatch, ICDPrediction

logger = get_logger(__name__)


class BigQueryRAGAgent:
    """Agent 3: retrieval-augmented ICD-10 prediction over BigQuery history."""

    def __init__(self) -> None:
        self._client = bigquery.Client(
            project=settings.gcp_project_id,
            location=settings.gcp_location,
        )

    # ------------------------------------------------------------------ #
    # Query builders
    # ------------------------------------------------------------------ #
    def _vector_search_sql(self) -> str:
        """
        Build a parameterised VECTOR_SEARCH query.

        The caller supplies a pre-computed PubMedBERT query embedding (passed as
        an ARRAY<FLOAT64> parameter), which is compared against the pre-computed
        `embedding` column of the history table.
        """
        return f"""
        SELECT
            base.icd_code           AS icd_code,
            base.diagnosis          AS diagnosis_text,
            base.specimen_source    AS specimen_source,
            distance                AS distance,
            1                       AS frequency
        FROM VECTOR_SEARCH(
            TABLE `{settings.bq_history_fqn}`,
            'embedding',
            (SELECT @query_embedding AS embedding),
            top_k => @top_k,
            distance_type => 'COSINE'
        )
        ORDER BY distance ASC
        """

    def _keyword_search_sql(self) -> str:
        """Fallback keyword/token-overlap search using SEARCH / LIKE."""
        return f"""
        SELECT
            icd_code                                        AS icd_code,
            diagnosis                                       AS diagnosis_text,
            specimen_source                                 AS specimen_source,
            -- crude inverse-relevance so lower == better, matching vector search
            (1.0 / (1.0 + ts_rank))                         AS distance,
            COUNT(*) OVER (PARTITION BY icd_code)           AS frequency
        FROM (
            SELECT
                icd_code,
                diagnosis,
                specimen_source,
                -- token overlap score between query and stored diagnosis
                (
                    SELECT COUNT(*)
                    FROM UNNEST(SPLIT(LOWER(@query_text), ' ')) AS q
                    WHERE STRPOS(LOWER(diagnosis), q) > 0
                ) AS ts_rank
            FROM `{settings.bq_history_fqn}`
            WHERE SEARCH(diagnosis, @query_text)
        )
        WHERE ts_rank > 0
        ORDER BY ts_rank DESC
        LIMIT @top_k
        """

    # ------------------------------------------------------------------ #
    # BigQuery execution
    # ------------------------------------------------------------------ #
    @retry(
        retry=retry_if_exception_type(GoogleAPIError),
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _run_query(
        self,
        sql: str,
        query_text: str,
        query_embedding: list[float] | None = None,
    ) -> list[HistoricalMatch]:
        """Execute a parameterised query off the event loop and map rows."""
        params: list = [
            bigquery.ScalarQueryParameter("query_text", "STRING", query_text),
            bigquery.ScalarQueryParameter("top_k", "INT64", settings.rag_top_k),
        ]
        if query_embedding is not None:
            params.append(
                bigquery.ArrayQueryParameter(
                    "query_embedding", "FLOAT64", query_embedding
                )
            )
        job_config = bigquery.QueryJobConfig(query_parameters=params)

        def _execute() -> list[HistoricalMatch]:
            # `query(...).result()` is blocking; run it in a worker thread.
            job = self._client.query(sql, job_config=job_config)
            rows = job.result(timeout=settings.request_timeout_seconds)
            return [
                HistoricalMatch(
                    icd_code=str(row["icd_code"]),
                    diagnosis_text=str(row["diagnosis_text"]),
                    specimen_source=(
                        str(row["specimen_source"])
                        if row["specimen_source"] is not None
                        else None
                    ),
                    distance=float(row["distance"]),
                    frequency=int(row["frequency"]),
                )
                for row in rows
            ]

        return await asyncio.to_thread(_execute)

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    @staticmethod
    def _aggregate_prediction(matches: list[HistoricalMatch]) -> ICDPrediction:
        """
        Aggregate top-k neighbours into a single prediction.

        Strategy: weight each neighbour by proximity (1 - distance) so the
        closest, most frequent code wins. Confidence = winning code's share of
        the total proximity weight, blended with the best raw similarity.
        """
        if not matches:
            raise ValueError("No historical matches found for the query.")

        weight_by_code: dict[str, float] = defaultdict(float)
        best_match_by_code: dict[str, HistoricalMatch] = {}

        for m in matches:
            # For COSINE distance, similarity ~= 1 - distance (clamped).
            similarity = max(0.0, min(1.0, 1.0 - m.distance))
            weight_by_code[m.icd_code] += similarity
            # Track the single closest match per code (smallest distance).
            if (
                m.icd_code not in best_match_by_code
                or m.distance < best_match_by_code[m.icd_code].distance
            ):
                best_match_by_code[m.icd_code] = m

        total_weight = sum(weight_by_code.values()) or 1.0
        winning_code = max(weight_by_code, key=weight_by_code.get)
        best_match = best_match_by_code[winning_code]

        # Confidence blends "vote share" with the best raw similarity.
        vote_share = weight_by_code[winning_code] / total_weight
        best_similarity = max(0.0, min(1.0, 1.0 - best_match.distance))
        confidence = round(0.5 * vote_share + 0.5 * best_similarity, 4)
        confidence = max(0.0, min(1.0, confidence))

        occurrences = sum(1 for m in matches if m.icd_code == winning_code)
        justification = (
            f"Selected {winning_code} because it best matches the historical "
            f"record '{best_match.diagnosis_text[:120]}' and appeared in "
            f"{occurrences} of the top {len(matches)} nearest cases."
        )

        return ICDPrediction(
            predicted_icd_code=winning_code,
            confidence_score=confidence,
            justification=justification,
        )

    # ------------------------------------------------------------------ #
    # Public entrypoint
    # ------------------------------------------------------------------ #
    async def run(self, formal: FormalClinicalText) -> ICDPrediction:
        """
        Predict the ICD-10 code for the formalised clinical text.

        Raises
        ------
        GoogleAPIError
            On unrecoverable BigQuery failure.
        ValueError
            If no historical matches are returned.
        """
        logger.info("bq_rag.start", use_vector=settings.rag_use_vector_search)

        # Combine formal text + concepts into a single keyword search string.
        query_text = formal.formal_text
        if formal.clinical_concepts:
            query_text += " | " + ", ".join(formal.clinical_concepts)

        # Reuse the PubMedBERT embedding computed by Agent 2. Fall back to
        # computing it here if it was not provided (keeps the agent standalone).
        query_embedding = formal.embedding
        if settings.rag_use_vector_search and not query_embedding:
            logger.info("bq_rag.embedding_missing_recompute")
            query_embedding = await asyncio.to_thread(
                get_pubmedbert_model().embed, query_text
            )

        use_vector = settings.rag_use_vector_search and bool(query_embedding)
        sql = self._vector_search_sql() if use_vector else self._keyword_search_sql()

        try:
            matches = await self._run_query(
                sql, query_text, query_embedding if use_vector else None
            )
        except GoogleAPIError as exc:
            logger.error("bq_rag.query_failed", error=str(exc))
            # If vector search failed, try keyword search once as a fallback.
            if use_vector:
                logger.warning("bq_rag.falling_back_to_keyword")
                matches = await self._run_query(self._keyword_search_sql(), query_text)
            else:
                raise

        logger.info("bq_rag.matches_retrieved", count=len(matches))
        prediction = self._aggregate_prediction(matches)
        logger.info(
            "bq_rag.done",
            icd_code=prediction.predicted_icd_code,
            confidence=prediction.confidence_score,
        )
        return prediction
