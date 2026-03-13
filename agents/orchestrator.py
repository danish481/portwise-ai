"""
PortWise AI - Multi-Agent Orchestrator
LangGraph-based agent coordination system
"""
import json
import logging
import threading
import asyncio
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from core.config import config, AGENT_PROMPTS, MARITIME_CONTEXT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AgentType(Enum):
    DOCUMENT_PROCESSOR = "document_processor"
    PREDICTIVE_ANALYST = "predictive_analyst"
    COMPLIANCE_GUARDIAN = "compliance_guardian"
    VOICE_ORCHESTRATOR = "voice_orchestrator"


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# Valid intents the router will recognise — anything else falls back to "general"
_VALID_INTENTS = frozenset({"document", "prediction", "compliance", "general"})


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class MaritimeState(TypedDict):
    """Shared state passed between LangGraph nodes."""
    messages: List[Dict[str, str]]
    current_task: Optional[str]
    document_data: Optional[Dict[str, Any]]
    analysis_results: Optional[Dict[str, Any]]
    compliance_report: Optional[Dict[str, Any]]
    predictions: Optional[Dict[str, Any]]
    voice_command: Optional[str]
    final_response: Optional[str]
    errors: List[str]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class MaritimeAgentOrchestrator:
    """
    Multi-agent orchestrator for maritime operations.
    Uses LangGraph for deterministic workflow management.
    """

    def __init__(self):
        if not config.llm_ready:
            logger.warning(
                "MaritimeAgentOrchestrator initialised without a valid GEMINI_API_KEY. "
                "All LLM calls will raise AuthenticationError until the key is configured."
            )
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            temperature=config.GEMINI_TEMPERATURE,
            max_tokens=config.GEMINI_MAX_TOKENS,
            google_api_key=config.GEMINI_API_KEY,
        )
        self.workflow = self._build_workflow()
        self.task_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Workflow construction
    # ------------------------------------------------------------------

    def _build_workflow(self) -> Any:
        """Build and compile the LangGraph workflow."""
        workflow = StateGraph(MaritimeState)

        workflow.add_node("parse_intent", self._parse_intent)
        workflow.add_node("document_processor", self._document_processor_agent)
        workflow.add_node("predictive_analyst", self._predictive_analyst_agent)
        workflow.add_node("compliance_guardian", self._compliance_guardian_agent)
        workflow.add_node("synthesize_response", self._synthesize_response)

        workflow.add_conditional_edges(
            "parse_intent",
            self._route_to_agent,
            {
                "document": "document_processor",
                "prediction": "predictive_analyst",
                "compliance": "compliance_guardian",
                'general': 'synthesize_response',
                'end': END,
            },
        )

        workflow.add_edge("document_processor", "synthesize_response")
        workflow.add_edge("predictive_analyst", "synthesize_response")
        workflow.add_edge("compliance_guardian", "synthesize_response")
        workflow.add_edge("synthesize_response", END)

        workflow.set_entry_point("parse_intent")
        return workflow.compile()

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _parse_intent(self, state: MaritimeState) -> MaritimeState:
        """Classify user intent and store it in state."""
        messages = state.get("messages", [])
        if not messages:
            state["errors"] = state.get("errors", []) + ["No messages provided"]
            state["current_task"] = "general"
            return state

        last_message = messages[-1].get("content", "")

        intent_prompt = (
            'Classify the following maritime operations query into EXACTLY one of these '
            'four categories (respond with the category name only, nothing else):\n\n'
            '- "document"   : processing, extracting, or analysing shipping documents '
            '(BOL, manifests, DG declarations)\n'
            '- "prediction" : forecasting delays, port congestion, or future events\n'
            '- "compliance" : regulations, safety checks, DG compliance, or validation\n'
            '- "general"    : all other queries\n\n'
            f'Query: "{last_message}"'
        )

        try:
            response = self.llm.invoke([HumanMessage(content=intent_prompt)])
            raw = response.content.strip().lower()
            # Whitelist — discard anything the LLM hallucinates
            intent = raw if raw in _VALID_INTENTS else "general"
            state["current_task"] = intent
        except Exception as exc:
            logger.exception("Intent parsing failed: %s", exc)
            state["errors"] = state.get("errors", []) + [f"Intent parsing error: {exc}"]
            state["current_task"] = "general"

        return state

    def _route_to_agent(self, state: MaritimeState) -> str:
        """Map intent string to graph edge label."""
        routing = {
            "document": "document",
            "prediction": "prediction",
            "compliance": "compliance",
            "general": "general",
        }
        return routing.get(state.get("current_task", "general"), "end")

    def _document_processor_agent(self, state: MaritimeState) -> MaritimeState:
        """Process maritime documents and extract structured data via LLM."""
        messages = state.get("messages", [])
        last_message = messages[-1].get("content", "") if messages else ""

        system_msg = SystemMessage(
            content=AGENT_PROMPTS["document_processor"] + "\n\n" + MARITIME_CONTEXT
        )
        human_msg = HumanMessage(
            content=(
                "Extract all structured data from the following document/query and respond "
                f"with valid JSON only (no markdown fences):\n\n{last_message}"
            )
        )

        try:
            response = self.llm.invoke([system_msg, human_msg])
            raw = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                result = {"raw_response": response.content, "extracted_data": {}, "confidence": 0.7}
            state["document_data"] = result
        except Exception as exc:
            logger.exception("Document processor agent failed: %s", exc)
            state["errors"] = state.get("errors", []) + [f"Document processing error: {exc}"]
            state["document_data"] = {"error": str(exc)}

        return state

    def _predictive_analyst_agent(self, state: MaritimeState) -> MaritimeState:
        """Analyse data and make predictions about maritime operations."""
        messages = state.get("messages", [])
        last_message = messages[-1].get("content", "") if messages else ""

        system_msg = SystemMessage(
            content=AGENT_PROMPTS["predictive_analyst"] + "\n\n" + MARITIME_CONTEXT
        )
        human_msg = HumanMessage(
            content=(
                "Analyse the following query and provide a structured prediction. "
                "Clearly label all estimates as model-based forecasts, not guarantees.\n\n"
                f"Query: {last_message}"
            )
        )

        try:
            response = self.llm.invoke([system_msg, human_msg])
            predictions = {
                "_data_source": "LLM forecast — not a live data feed",
                "query": last_message,
                "analysis": response.content,
                "timestamp": datetime.now().isoformat(),
            }
            state["predictions"] = predictions
        except Exception as exc:
            logger.exception("Predictive analyst agent failed: %s", exc)
            state["errors"] = state.get("errors", []) + [f"Prediction error: {exc}"]
            state["predictions"] = {"error": str(exc)}

        return state

    def _compliance_guardian_agent(self, state: MaritimeState) -> MaritimeState:
        """Check compliance with maritime regulations."""
        messages = state.get("messages", [])
        last_message = messages[-1].get("content", "") if messages else ""

        system_msg = SystemMessage(
            content=AGENT_PROMPTS["compliance_guardian"] + "\n\n" + MARITIME_CONTEXT
        )
        human_msg = HumanMessage(
            content=(
                "Perform a compliance check for the following query. "
                "Cite specific regulation codes where possible.\n\n"
                f"Query: {last_message}"
            )
        )

        try:
            response = self.llm.invoke([system_msg, human_msg])
            compliance_report = {
                "_data_source": "LLM regulatory analysis — verify against official IMDG/SOLAS publications",
                "query": last_message,
                "analysis": response.content,
                "timestamp": datetime.now().isoformat(),
            }
            state["compliance_report"] = compliance_report
        except Exception as exc:
            logger.exception("Compliance guardian agent failed: %s", exc)
            state["errors"] = state.get("errors", []) + [f"Compliance check error: {exc}"]
            state["compliance_report"] = {"error": str(exc)}

        return state

    def _synthesize_response(self, state: MaritimeState) -> MaritimeState:
        """Synthesise a final professional response from all agent outputs."""
        document_data = state.get("document_data") or {}
        predictions = state.get("predictions") or {}
        compliance_report = state.get("compliance_report") or {}
        errors = state.get("errors") or []

        # Truncate large blobs to avoid exceeding context limits
        def _safe_dump(obj: dict, max_chars: int = 2000) -> str:
            s = json.dumps(obj, indent=2)
            return s[:max_chars] + "\n... (truncated)" if len(s) > max_chars else s

        synthesis_prompt = (
            "Synthesise a comprehensive, professional response for a maritime operations "
            "specialist based on the following agent outputs.\n\n"
            f"DOCUMENT ANALYSIS:\n{_safe_dump(document_data) if document_data else 'Not performed'}\n\n"
            f"PREDICTIVE ANALYSIS:\n{_safe_dump(predictions) if predictions else 'Not performed'}\n\n"
            f"COMPLIANCE REPORT:\n{_safe_dump(compliance_report) if compliance_report else 'Not performed'}\n\n"
            f"ERRORS:\n{json.dumps(errors) if errors else 'None'}\n\n"
            "Your response must:\n"
            "1. Summarise key findings concisely\n"
            "2. Highlight critical or urgent information first\n"
            "3. Provide specific, actionable recommendations\n"
            "4. Use professional maritime terminology\n"
            "5. Be clearly formatted with sections"
        )

        try:
            system_msg = SystemMessage(
                content="You are a Maritime Operations Assistant. Provide clear, professional responses."
            )
            response = self.llm.invoke([system_msg, HumanMessage(content=synthesis_prompt)])
            state["final_response"] = response.content
        except Exception as exc:
            logger.exception("Response synthesis failed: %s", exc)
            state["final_response"] = (
                "Analysis completed but response synthesis encountered an error. "
                f"Details: {exc}"
            )

        return state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_query(
        self, query: str, context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the multi-agent system.

        Args:
            query:   Natural language query from the user.
            context: Optional dict with extra context (document content, history, etc.).

        Returns:
            Dict with intent, agent outputs, synthesised response, and any errors.
        """
        initial_state: MaritimeState = {
            "messages": [{"role": "user", "content": query}],
            "current_task": None,
            "document_data": None,
            "analysis_results": None,
            "compliance_report": None,
            "predictions": None,
            "voice_command": None,
            "final_response": None,
            "errors": [],
        }

        if context:
            initial_state["messages"].append(
                {"role": "system", "content": f"Context: {json.dumps(context)}"}
            )

        try:
            result = self.workflow.invoke(initial_state)
            return {
                "success": True,
                "query": query,
                "intent": result.get("current_task"),
                "document_analysis": result.get("document_data"),
                "predictions": result.get("predictions"),
                "compliance_report": result.get("compliance_report"),
                "response": result.get("final_response"),
                "errors": result.get("errors", []),
            }
        except Exception as exc:
            logger.exception("Workflow execution failed: %s", exc)
            return {
                "success": False,
                "query": query,
                "error": str(exc),
                "response": "An error occurred while processing your request. Please try again.",
            }

    async def process_voice_command(self, transcription: str) -> Dict[str, Any]:
        """
        Process a voice command (async — safe for use inside FastAPI / async frameworks).
        """
        enhanced_query = f"[VOICE COMMAND] {transcription}"
        return await self.process_query(enhanced_query)

    def process_query_sync(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Synchronous wrapper around process_query for use in Streamlit
        (which runs in a regular synchronous context).
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an existing event loop (e.g. Jupyter); use a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.process_query(query, context))
                    return future.result()
            else:
                return loop.run_until_complete(self.process_query(query, context))
        except RuntimeError:
            return asyncio.run(self.process_query(query, context))


# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_orchestrator: Optional[MaritimeAgentOrchestrator] = None
_orchestrator_lock = threading.Lock()


def get_orchestrator() -> MaritimeAgentOrchestrator:
    """Return the shared orchestrator instance (thread-safe, lazy-initialised)."""
    global _orchestrator
    if _orchestrator is None:
        with _orchestrator_lock:
            if _orchestrator is None:  # double-checked locking
                _orchestrator = MaritimeAgentOrchestrator()
    return _orchestrator
