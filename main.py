"""
main.py
=======
Entrypoint for the ICD-10 Multi-Agent Orchestration System.

Provides two ways to run:

1. CLI / demo mode (default)::

       python main.py

   Runs a sample pathology report through the full pipeline and prints the
   validated JSON prediction.

2. HTTP service mode (FastAPI)::

       uvicorn main:app --host 0.0.0.0 --port 8080

   Exposes POST /predict accepting {"specimen_source": ..., "diagnosis": ...}
   and returning a `PipelineResult`. Suitable for Cloud Run deployment.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from logging_utils import get_logger
from orchestrator import ICDPredictionOrchestrator
from schemas import PipelineResult

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Shared orchestrator instance (agents + graph built once).
# --------------------------------------------------------------------------- #
_orchestrator: ICDPredictionOrchestrator | None = None


def get_orchestrator() -> ICDPredictionOrchestrator:
    """Lazily construct and cache the orchestrator (and its GCP clients)."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ICDPredictionOrchestrator()
    return _orchestrator


# --------------------------------------------------------------------------- #
# FastAPI application
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="ICD-10 Multi-Agent Predictor",
    version="1.0.0",
    description="Predicts ICD-10 codes from anatomic pathology reports.",
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request) -> JSONResponse:
    """
    Predict an ICD-10 code from a raw JSON pathology payload.

    The raw body is passed to the input guardrail unmodified so that malformed
    JSON and oversized payloads are rejected consistently.
    """
    raw_body = await request.body()
    orchestrator = get_orchestrator()
    result: PipelineResult = await orchestrator.predict(raw_body)
    status_code = 200 if result.success else 400
    return JSONResponse(status_code=status_code, content=result.model_dump())


# --------------------------------------------------------------------------- #
# CLI / demo runner
# --------------------------------------------------------------------------- #
async def _run_demo() -> None:
    """Run a representative sample through the pipeline and print the result."""
    sample_payload = {
        "specimen_source": "Left breast, 2 o'clock, ultrasound-guided core biopsy",
        "diagnosis": (
            "Invasive ductal carcinoma, Nottingham grade 2. "
            "Pt Jane Doe, MRN 123456789, DOB 01/02/1970. ER+/PR+, HER2 negative."
        ),
    }

    logger.info("demo.start")
    orchestrator = get_orchestrator()
    result = await orchestrator.predict(json.dumps(sample_payload))

    print("\n=== ICD-10 PREDICTION RESULT ===")
    print(json.dumps(result.model_dump(), indent=2))


def main() -> None:
    """Synchronous entrypoint wrapper for `python main.py`."""
    try:
        asyncio.run(_run_demo())
    except KeyboardInterrupt:
        logger.info("demo.interrupted")


if __name__ == "__main__":
    main()
