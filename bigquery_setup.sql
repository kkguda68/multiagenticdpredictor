-- =========================================================================
-- bigquery_setup.sql
-- ICD-10 Multi-Agent Predictor - Agent 3 data layer
--
-- Structures & indexes the ~1,000,000 historical pathology transactions so
-- that Agent 3 can perform fast semantic (vector) or keyword retrieval.
--
-- Embeddings are produced by the application's local **PubMedBERT** encoder
-- (768-dim, COSINE). An ingestion/backfill job writes them into the
-- `embedding` column; BigQuery does NOT generate them (no remote model).
--
-- Replace `qwiklabs-gcp-03-d9bd89368565` with your GCP project ID (see config.py).
-- Run in order. Requires BigQuery only (Vertex AI not needed for embeddings).
-- =========================================================================

-- -------------------------------------------------------------------------
-- 0. Dataset
-- -------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `qwiklabs-gcp-03-d9bd89368565.pathology`
OPTIONS (location = 'us-central1');

-- -------------------------------------------------------------------------
-- 1. Historical transactions table (~1M rows)
--    `embedding` holds the pre-computed PubMedBERT vector (768 dims).
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `qwiklabs-gcp-03-d9bd89368565.pathology.historical_transactions`
(
  transaction_id    STRING    NOT NULL,   -- unique record id
  specimen_source   STRING,               -- anatomic source of specimen
  diagnosis         STRING    NOT NULL,    -- historical (de-identified) diagnosis text
  icd_code          STRING    NOT NULL,    -- ground-truth ICD-10-CM code
  reported_date     DATE,                  -- when the case was signed out
  embedding         ARRAY<FLOAT64>,        -- 768-dim PubMedBERT vector (nullable until backfilled)
  content_hash      STRING                 -- dedupe / idempotency key
)
CLUSTER BY icd_code;                        -- accelerates code-level aggregation

-- -------------------------------------------------------------------------
-- 2. Backfill embeddings (done OUTSIDE BigQuery with PubMedBERT).
--    Run a batch job that:
--      a) reads rows WHERE embedding IS NULL,
--      b) computes PubMedBERT embeddings via pubmedbert.PubMedBERTModel.embed,
--      c) loads them back (streaming insert / MERGE / load job).
--    Example MERGE from a staging table `pathology.embedding_staging`
--    (transaction_id STRING, embedding ARRAY<FLOAT64>):
--
--    MERGE `qwiklabs-gcp-03-d9bd89368565.pathology.historical_transactions` T
--    USING `qwiklabs-gcp-03-d9bd89368565.pathology.embedding_staging` S
--    ON T.transaction_id = S.transaction_id
--    WHEN MATCHED THEN UPDATE SET T.embedding = S.embedding;
-- -------------------------------------------------------------------------

-- -------------------------------------------------------------------------
-- 3. Vector index for fast approximate nearest-neighbour VECTOR_SEARCH.
--    IVF works well at the 1M-row scale. COSINE matches the query in
--    agents/bigquery_rag.py.
-- -------------------------------------------------------------------------
-- NOTE: Commented out because BigQuery requires at least one non-null embedding
-- to infer the array dimension (768) before creating the index. 
-- Run this manually ONLY AFTER you have backfilled the embeddings in Step 2.
-- CREATE VECTOR INDEX IF NOT EXISTS historical_embedding_idx
-- ON `qwiklabs-gcp-03-d9bd89368565.pathology.historical_transactions`(embedding)
-- OPTIONS (
--   index_type    = 'IVF',
--   distance_type = 'COSINE',
--   ivf_options   = '{"num_lists": 1000}'
-- );

-- -------------------------------------------------------------------------
-- 4. Full-text SEARCH index for the keyword fallback path.
-- -------------------------------------------------------------------------
CREATE SEARCH INDEX IF NOT EXISTS historical_diagnosis_search_idx
ON `qwiklabs-gcp-03-d9bd89368565.pathology.historical_transactions`(diagnosis);

-- -------------------------------------------------------------------------
-- 5. Example query mirroring Agent 3's vector search (for validation).
--    The application passes the PubMedBERT query embedding as the
--    @query_embedding ARRAY<FLOAT64> parameter (768 values).
-- -------------------------------------------------------------------------
-- DECLARE query_embedding ARRAY<FLOAT64> DEFAULT [/* 768 PubMedBERT floats */];
-- SELECT base.icd_code, base.diagnosis, distance
-- FROM VECTOR_SEARCH(
--   TABLE `qwiklabs-gcp-03-d9bd89368565.pathology.historical_transactions`,
--   'embedding',
--   (SELECT query_embedding AS embedding),
--   top_k => 10,
--   distance_type => 'COSINE'
-- )
-- ORDER BY distance ASC;
