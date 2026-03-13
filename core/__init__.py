"""
Core business logic module.

Imports are kept explicit so that optional heavy dependencies
(speech_recognition, pydub) do not break import of other modules
when they are not installed.
"""
from .config import config, Config, MARITIME_CONTEXT, AGENT_PROMPTS
from .document_processor import MaritimeDocumentProcessor, get_document_processor
from .predictive_engine import PredictiveEngine, get_predictive_engine

# VoiceInterface depends on speech_recognition which is optional; import lazily
def get_voice_interface():
    from .voice_interface import get_voice_interface as _get
    return _get()

__all__ = [
    "config", "Config", "MARITIME_CONTEXT", "AGENT_PROMPTS",
    "MaritimeDocumentProcessor", "get_document_processor",
    "get_voice_interface",
    "PredictiveEngine", "get_predictive_engine",
]
