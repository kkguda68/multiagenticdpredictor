# ICD-10 Multi-Agent Orchestration System

Production-ready GCP pipeline that predicts diagnostic **ICD-10-CM** codes from
anatomic pathology reports (`Specimen Source` + `Diagnosis`) using a 3-agent
architecture orchestrated with the **Google Agent Development Kit (ADK)**.

## Architecture

```
Raw JSON  ->  [Input Guardrail]
              -> Agent 1: PHI Scrubber        (Google Cloud DLP)
              -> Agent 2: Clinical Reasoning   (local PubMedBERT encoder)
              -> Agent 3: BigQuery RAG          (PubMedBERT vector search over 1M rows)
              -> [Output Guardrail]  ->  ICDPrediction (JSON)
```

Orchestration uses an ADK `SequentialAgent` (three `BaseAgent` nodes) executed
by an `InMemoryRunner`; agents communicate via shared session state.

| Agent | File | Responsibility |
|-------|------|----------------|
| 1. PHI Scrubber | [agents/phi_scrubber.py](agents/phi_scrubber.py) | Redacts PHI via DLP API (regex fallback, fail-closed) |
| 2. Clinical Reasoning | [agents/clinical_reasoning.py](agents/clinical_reasoning.py) | PubMedBERT embeddings + concept extraction ([pubmedbert.py](pubmedbert.py)) |
| 3. BigQuery RAG | [agents/bigquery_rag.py](agents/bigquery_rag.py) | Retrieves nearest historical cases via PubMedBERT vector search, formulates prediction |

Supporting modules: [config.py](config.py), [schemas.py](schemas.py),
[guardrails.py](guardrails.py), [orchestrator.py](orchestrator.py),
[main.py](main.py).

> **Note on PubMedBERT:** PubMedBERT is a biomedical *encoder* (not a generative
> LLM). Agent 2 uses it for 768-dim embeddings and semantic concept ranking; the
> formal clinical representation is assembled deterministically from the
> (PHI-free) input, and the embedding is reused by Agent 3 for retrieval.

## Guardrails

- **Input:** valid JSON, per-field character limit, DLP runs before any LLM.
- **Output:** strict Pydantic schema, ICD-10 regex `^[A-TV-Z][0-9][0-9AB](?:\.[0-9A-TV-Z]{1,4})?$`, confidence bounds, and a content blocklist screening any free text before it leaves the system.

## Setup

```powershell
# 1. Create & activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
Copy-Item .env.example .env   # then edit values
gcloud auth application-default login

# 4. Provision BigQuery (edit project id first)
bq query --use_legacy_sql=false < bigquery_setup.sql
```

## Run

```powershell
# CLI demo
python main.py

# HTTP service (Cloud Run compatible)
uvicorn main:app --host 0.0.0.0 --port 8080
# POST /predict  {"specimen_source": "...", "diagnosis": "..."}
```

## Example response

```json
{
  "success": true,
  "prediction": {
    "predicted_icd_code": "C50.911",
    "confidence_score": 0.87,
    "justification": "Selected C50.911 because it best matches the historical record '...' and appeared in 8 of the top 10 nearest cases."
  },
  "phi_found": true,
  "stages_completed": ["input_guardrail", "phi_scrubber", "clinical_reasoning", "bigquery_rag", "output_guardrail"]
}
```

## CI/CD — Build & Deploy to Google Cloud

The app is containerized ([Dockerfile](Dockerfile)) and shipped to **Cloud Run**
via **Cloud Build** ([cloudbuild.yaml](cloudbuild.yaml)). Cloud Run provides an
external public HTTPS URL for integration.

### One-time setup

```powershell
# Enable required APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com `
    artifactregistry.googleapis.com aiplatform.googleapis.com `
    dlp.googleapis.com bigquery.googleapis.com

# Artifact Registry repo for the container image
gcloud artifacts repositories create icd-predictor `
    --repository-format=docker --location=us-central1

# Runtime service account with least-privilege roles
gcloud iam service-accounts create icd-predictor-runtime
$PROJECT = gcloud config get-value project
foreach ($role in @('roles/dlp.user','roles/aiplatform.user','roles/bigquery.dataViewer','roles/bigquery.jobUser')) {
  gcloud projects add-iam-policy-binding $PROJECT `
    --member="serviceAccount:icd-predictor-runtime@$PROJECT.iam.gserviceaccount.com" `
    --role=$role
}
```

### Deploy manually

```powershell
gcloud builds submit --config cloudbuild.yaml --substitutions=_REGION=us-central1
```

The final build step prints the **public service URL**, e.g.
`https://icd-predictor-xxxxxxxxxx-uc.a.run.app`.

### Automate on every push

Create a trigger so each commit builds & deploys automatically:

```powershell
gcloud builds triggers create github `
    --repo-name=icdpredictor_multiagent --repo-owner=<your-org> `
    --branch-pattern=^main$ --build-config=cloudbuild.yaml
```

### Integrate

```powershell
$URL = "https://icd-predictor-xxxxxxxxxx-uc.a.run.app"
curl "$URL/healthz"
curl -X POST "$URL/predict" -H "Content-Type: application/json" `
     -d '{"specimen_source":"Left breast core biopsy","diagnosis":"Invasive ductal carcinoma, grade 2"}'
```

> The pipeline deploys with `--allow-unauthenticated` for a public integration
> URL. To require callers to authenticate, remove that flag and grant
> `roles/run.invoker` to specific identities (or front it with API Gateway/IAP).
