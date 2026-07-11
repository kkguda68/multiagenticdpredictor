"""
orchestrator.py
===============
Google **ADK** (Agent Development Kit) orchestration of the 3-agent ICD-10
prediction pipeline.

Flow
----
    [input guardrail]  (applied in `predict`, seeds session state)
          |
          v
    SequentialAgent
      ├─ Agent 1: PHI Scrubber        (writes `sanitized`)
      ├─ Agent 2: Clinical Reasoning   (writes `formal` + embedding)
      └─ Agent 3: BigQuery RAG          (writes `prediction`)
          |
          v
    [output guardrail]  --> ICDPrediction

Each agent is an ADK `BaseAgent` (`_run_async_impl`) that reads/writes the
shared `ctx.session.state`. If any agent sets `state["error"]`, downstream
agents short-circuit so the pipeline fails closed. The graph is executed by an
ADK `InMemoryRunner`.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.genai import types

from agents import BigQueryRAGAgent, ClinicalReasoningAgent, PHIScrubberAgent
from config import settings
from guardrails import GuardrailError, validate_input_payload, validate_prediction_output
from logging_utils import get_logger
from schemas import PipelineResult

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# ADK agent nodes - thin wrappers over the business-logic agents.
# Pydantic `arbitrary_types_allowed` lets us hold the logic objects as fields.
# --------------------------------------------------------------------------- #
class PHIScrubberNode(BaseAgent):
    """ADK node for Agent 1 (PHI scrubbing). Fails closed on any error."""

    model_config = {"arbitrary_types_allowed": True}
    logic: PHIScrubberAgent

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        if state.get("error"):
            yield Event(author=self.name)
            return
        try:
            sanitized = await self.logic.run(state["request"])
            state["sanitized"] = sanitized
            state["phi_found"] = sanitized.phi_found
            state.setdefault("stages", []).append("phi_scrubber")
        except Exception as exc:  # noqa: BLE001 - never fail-open on PHI
            logger.error("orchestrator.phi_scrubber_error", error=str(exc))
            state["error"] = f"phi_scrubber: {exc}"
        yield Event(author=self.name)


class ClinicalReasoningNode(BaseAgent):
    """ADK node for Agent 2 (PubMedBERT clinical reasoning)."""

    model_config = {"arbitrary_types_allowed": True}
    logic: ClinicalReasoningAgent

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        if state.get("error"):
            yield Event(author=self.name)
            return
        try:
            formal = await self.logic.run(state["sanitized"])
            state["formal"] = formal
            state.setdefault("stages", []).append("clinical_reasoning")
        except Exception as exc:  # noqa: BLE001
            logger.error("orchestrator.clinical_reasoning_error", error=str(exc))
            state["error"] = f"clinical_reasoning: {exc}"
        yield Event(author=self.name)


# class BigQueryRAGNode(BaseAgent):
#     """ADK node for Agent 3 (BigQuery RAG) + output guardrail."""
# 
#     model_config = {"arbitrary_types_allowed": True}
#     logic: BigQueryRAGAgent
# 
#     async def _run_async_impl(
#         self, ctx: InvocationContext
#     ) -> AsyncGenerator[Event, None]:
#         state = ctx.session.state
#         if state.get("error"):
#             yield Event(author=self.name)
#             return
#         try:
#             raw_prediction = await self.logic.run(state["formal"])
#             # Output guardrail: re-validate against the strict schema.
#             prediction = validate_prediction_output(raw_prediction)
#             state["prediction"] = prediction
#             state.setdefault("stages", []).extend(["bigquery_rag", "output_guardrail"])
#         except (GuardrailError, ValueError) as exc:
#             logger.error("orchestrator.bigquery_rag_error", error=str(exc))
#             state["error"] = f"bigquery_rag: {exc}"
#         except Exception as exc:  # noqa: BLE001
#             logger.error("orchestrator.bigquery_rag_unexpected", error=str(exc))
#             state["error"] = f"bigquery_rag: {exc}"
#         yield Event(author=self.name)

class GeminiPredictionNode(BaseAgent):
    """ADK node that replaces BigQuery RAG with a direct Gemini prediction call."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        if state.get("error"):
            yield Event(author=self.name)
            return
        try:
            formal = state["formal"]
            client = genai.Client(
                vertexai=True,
                project=settings.gcp_project_id,
                location=settings.gcp_location
            )
            
            prompt = (
                "You are an expert medical coder. Predict the most specific ICD-10-CM code "
                "for the following pathology report.\n\n"
                f"Formal Text: {formal.formal_text}\n"
                f"Extracted Concepts: {', '.join(formal.clinical_concepts)}\n"
            )
            
            # Force Gemini to return JSON that matches our Pydantic schema
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ICDPrediction,
                temperature=0.0
            )
            
            response = await client.aio.models.generate_content(
                model='gemini-1.5-pro',
                contents=prompt,
                config=config
            )
            
            raw_prediction = json.loads(response.text)
            # Output guardrail: re-validate against the strict schema rules
            prediction = validate_prediction_output(raw_prediction)
            state["prediction"] = prediction
            state.setdefault("stages", []).extend(["gemini_prediction", "output_guardrail"])
            
        except (GuardrailError, ValueError) as exc:
            logger.error("orchestrator.gemini_prediction_error", error=str(exc))
            state["error"] = f"gemini_prediction: {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.error("orchestrator.gemini_prediction_unexpected", error=str(exc))
            state["error"] = f"gemini_prediction: {exc}"
        yield Event(author=self.name)


class ICDPredictionOrchestrator:
    """Builds and runs the ADK `SequentialAgent` pipeline via `InMemoryRunner`."""

    def __init__(self) -> None:
        # Instantiate business-logic agents once and reuse across requests.
        # (Agent 2 loads the PubMedBERT singleton on first use.)
        phi_node = PHIScrubberNode(name="phi_scrubber", logic=PHIScrubberAgent())
        reasoning_node = ClinicalReasoningNode(
            name="clinical_reasoning", logic=ClinicalReasoningAgent()
        )
        # rag_node = BigQueryRAGNode(name="bigquery_rag", logic=BigQueryRAGAgent())
        gemini_node = GeminiPredictionNode(name="gemini_prediction")

        self._root_agent = SequentialAgent(
            name="icd_pipeline",
            # agents=[phi_node, reasoning_node, rag_node],
            agents=[phi_node, reasoning_node, gemini_node],
        )
        self._runner = InMemoryRunner(
            agent=self._root_agent, app_name=settings.adk_app_name
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def predict(self, raw_payload: str | bytes | dict) -> PipelineResult:
        """
        Run the full pipeline for a single request.

        Always returns a `PipelineResult` (never raises) so callers get a
        clean, schema-valid response even on failure.
        """
        logger.info("orchestrator.predict.start")

        # --- Input guardrail (before any agent runs) -------------------- #
        try:
            request = validate_input_payload(raw_payload)
        except GuardrailError as exc:
            return PipelineResult(success=False, error=f"input_guardrail: {exc}")

        # Seed the ADK session with the validated request.
        session = await self._runner.session_service.create_session(
            app_name=settings.adk_app_name,
            user_id=settings.adk_user_id,
            state={"request": request, "stages": ["input_guardrail"], "error": None},
        )

        trigger = types.Content(role="user", parts=[types.Part(text="run pipeline")])

        try:
            async for _event in self._runner.run_async(
                user_id=settings.adk_user_id,
                session_id=session.id,
                new_message=trigger,
            ):
                # Events are consumed to drive execution; state holds results.
                pass
        except Exception as exc:  # noqa: BLE001 - final safety net
            logger.error("orchestrator.pipeline_crashed", error=str(exc))
            return PipelineResult(success=False, error=f"pipeline_crashed: {exc}")

        # --- Collect final state ---------------------------------------- #
        final = await self._runner.session_service.get_session(
            app_name=settings.adk_app_name,
            user_id=settings.adk_user_id,
            session_id=session.id,
        )
        state = final.state if final else {}
        stages = state.get("stages", [])
        phi_found = bool(state.get("phi_found"))

        if state.get("error"):
            return PipelineResult(
                success=False,
                error=state["error"],
                phi_found=phi_found,
                stages_completed=stages,
            )

        return PipelineResult(
            success=True,
            prediction=state.get("prediction"),
            phi_found=phi_found,
            stages_completed=stages,
        )
