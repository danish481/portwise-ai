"""
PortWise AI - Voice Interface Module
Speech-to-text and voice command processing.

DEPENDENCIES
------------
Online STT:  SpeechRecognition + Google Speech API (requires internet)
Offline STT: openai-whisper (CPU-only, no API key needed)
TTS:         Placeholder — wire up Google Cloud TTS or Amazon Polly for production.
             See generate_voice_response() for the recommended integration point.
"""
import io
import os
import re
import logging
import tempfile
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

import speech_recognition as sr
from pydub import AudioSegment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class VoiceCommand:
    """Represents a processed voice command."""
    transcription: str
    confidence: float
    command_type: str
    entities: Dict[str, Any]
    timestamp: datetime
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

COMMAND_PATTERNS: Dict[str, List[str]] = {
    "container_query":  ["container", "containers", "box", "boxes", "teu"],
    "vessel_query":     ["vessel", "ship", "boat", "imo", "voyage"],
    "port_query":       ["port", "harbor", "terminal", "berth", "dock"],
    "compliance_check": ["compliance", "dg", "dangerous goods", "imdg", "safety", "check"],
    "prediction":       ["predict", "forecast", "delay", "when", "arrival", "eta", "congestion"],
    "document":         ["document", "bol", "bill of lading", "manifest", "extract", "upload"],
    "status":           ["status", "where", "location", "position", "tracking"],
}

_COMMON_PORTS = [
    "mumbai", "chennai", "visakhapatnam", "kolkata", "singapore",
    "rotterdam", "shanghai", "dubai", "los angeles", "hamburg",
    "felixstowe", "antwerp", "busan", "tokyo", "sydney", "port botany",
]

# Static keyword dates — matched separately from regex
_STATIC_DATE_KEYWORDS = {"tomorrow", "today", "next week", "next month", "yesterday"}

# Regex-only date patterns
_DATE_REGEXES = [
    r"\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+\d{4})?",
    r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}",
]


# ---------------------------------------------------------------------------
# Voice Interface
# ---------------------------------------------------------------------------

class VoiceInterface:
    """
    Voice interface for maritime operations.

    STT priority:
      1. Google Cloud STT (online, requires internet)
      2. OpenAI Whisper  (offline fallback, CPU)
    """

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.command_history: List[Dict[str, Any]] = []
        self._max_history = 200
        self._whisper_model = None  # lazy-loaded only when needed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_audio_file(
        self, audio_file_path: str, language: str = "en-IN"
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file and classify the maritime command it contains.

        Args:
            audio_file_path: Path to the audio file (wav, mp3, webm, m4a, etc.).
            language:        BCP-47 language tag for Google STT (default: en-IN).

        Returns:
            Dict with transcription, confidence, command_type, entities, and metadata.
        """
        start = datetime.now()

        wav_path = audio_file_path
        _tmp_wav = None
        if not audio_file_path.lower().endswith(".wav"):
            wav_path, _tmp_wav = self._convert_to_wav(audio_file_path)

        try:
            result = self._transcribe_wav(wav_path, language)
        finally:
            if _tmp_wav and os.path.exists(_tmp_wav):
                os.unlink(_tmp_wav)

        if result["success"]:
            command = self._classify(result["transcription"])
            result["command_type"] = command.command_type
            result["entities"] = command.entities
        else:
            result["command_type"] = "unknown"
            result["entities"] = {}

        result["processing_time_ms"] = (datetime.now() - start).total_seconds() * 1000
        result["timestamp"] = datetime.now().isoformat()

        self._append_history(result)
        return result

    def process_audio_bytes(
        self, audio_bytes: bytes, fmt: str = "wav", language: str = "en-IN"
    ) -> Dict[str, Any]:
        """
        Transcribe audio supplied as raw bytes (e.g. from a web upload).

        Args:
            audio_bytes: Raw audio bytes.
            fmt:         Format string ('wav', 'mp3', 'webm', 'ogg', etc.).
            language:    BCP-47 language tag.

        Returns:
            Same structure as process_audio_file().
        """
        suffix = f".{fmt}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            return self.process_audio_file(tmp_path, language)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def generate_voice_response(self, text: str) -> Dict[str, Any]:
        """
        Generate a text-to-speech response payload.

        Production integration points:
          - Google Cloud TTS:  https://cloud.google.com/text-to-speech
          - Amazon Polly:      https://aws.amazon.com/polly/
          - Azure Cognitive:   https://azure.microsoft.com/en-us/services/cognitive-services/text-to-speech/

        Returns a dict describing the TTS request rather than raw audio bytes so that
        the caller can choose the TTS provider. Replace this method body with an actual
        TTS SDK call once a provider is selected.
        """
        return {
            "text": text,
            "status": "tts_not_configured",
            "instruction": (
                "Wire up a TTS provider in generate_voice_response(). "
                "Recommended: Google Cloud TTS with voice 'en-IN-Standard-C'."
            ),
            "recommended_voice": "en-IN-Standard-C",
            "char_count": len(text),
        }

    def get_command_suggestions(self, partial_command: str) -> List[str]:
        """
        Return up to five command suggestions that match the partial input.
        Matches against both prefix and substring.
        """
        partial_lower = partial_command.lower()
        templates = [
            "Show me container status for",
            "Check compliance for DG shipment",
            "Predict delays for port",
            "Extract data from bill of lading",
            "Where is vessel",
            "What is the ETA for",
            "Show port congestion at",
            "Validate container number",
            "Generate manifest for voyage",
            "Check dangerous goods classification for",
        ]
        seen: set = set()
        results: List[str] = []
        for t in templates:
            tl = t.lower()
            if (tl.startswith(partial_lower) or partial_lower in tl) and tl not in seen:
                results.append(t)
                seen.add(tl)
            if len(results) >= 5:
                break
        return results

    def get_command_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.command_history[-limit:]

    # ------------------------------------------------------------------
    # Internal transcription
    # ------------------------------------------------------------------

    def _transcribe_wav(self, wav_path: str, language: str) -> Dict[str, Any]:
        """Try Google STT first; fall back to Whisper if offline or unavailable."""
        # ── Google STT ──────────────────────────────────────────────────────
        try:
            with sr.AudioFile(wav_path) as source:
                audio = self.recognizer.record(source)
            all_results = self.recognizer.recognize_google(audio, language=language, show_all=True)

            if all_results and isinstance(all_results, dict):
                alternatives = all_results.get("alternative", [])
                if alternatives:
                    best = alternatives[0]
                    text = best.get("transcript", "")
                    # confidence is not always present
                    confidence = float(best.get("confidence", 0.80))
                    return {"success": True, "transcription": text, "confidence": round(confidence, 2), "engine": "google"}

            # show_all=True returned empty
            text = self.recognizer.recognize_google(audio, language=language, show_all=False)
            return {"success": True, "transcription": text, "confidence": 0.80, "engine": "google"}

        except sr.UnknownValueError:
            logger.info("Google STT: audio not understood — trying Whisper fallback")
        except sr.RequestError as exc:
            logger.warning("Google STT unavailable (%s) — trying Whisper fallback", exc)
        except Exception as exc:
            logger.exception("Google STT unexpected error: %s", exc)

        # ── Whisper offline fallback ─────────────────────────────────────────
        return self._transcribe_whisper(wav_path)

    def _transcribe_whisper(self, wav_path: str) -> Dict[str, Any]:
        """Offline transcription using OpenAI Whisper (runs on CPU)."""
        try:
            import whisper  # lazy import — only required if Google fails
            if self._whisper_model is None:
                logger.info("Loading Whisper 'base' model (first-time, may take a moment)…")
                self._whisper_model = whisper.load_model("base")
            result = self._whisper_model.transcribe(wav_path)
            text = result.get("text", "").strip()
            return {
                "success": bool(text),
                "transcription": text,
                "confidence": 0.75,  # Whisper doesn't expose per-utterance confidence
                "engine": "whisper",
            }
        except ImportError:
            logger.error("whisper package not installed — install with: pip install openai-whisper")
        except Exception as exc:
            logger.exception("Whisper transcription failed: %s", exc)

        return {
            "success": False,
            "error": "All STT engines failed. Check internet connectivity or install openai-whisper.",
            "transcription": None,
            "confidence": 0.0,
            "engine": "none",
        }

    def _convert_to_wav(self, audio_file_path: str) -> Tuple[str, str]:
        """Convert audio to WAV. Returns (wav_path, tmp_path_to_delete)."""
        try:
            audio = AudioSegment.from_file(audio_file_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                audio.export(tmp.name, format="wav")
                return tmp.name, tmp.name
        except Exception as exc:
            logger.exception("Audio conversion failed for %s: %s", audio_file_path, exc)
            return audio_file_path, ""  # Return original; STT may still handle it

    # ------------------------------------------------------------------
    # Command classification
    # ------------------------------------------------------------------

    def _classify(self, transcription: str) -> VoiceCommand:
        """Classify a transcription into a command type and extract entities."""
        tl = transcription.lower()

        command_type = "general"
        max_matches = 0
        for ctype, keywords in COMMAND_PATTERNS.items():
            matches = sum(1 for kw in keywords if kw in tl)
            if matches > max_matches:
                max_matches = matches
                command_type = ctype

        entities = self._extract_entities(transcription)
        return VoiceCommand(
            transcription=transcription,
            confidence=0.0,  # filled in by caller
            command_type=command_type,
            entities=entities,
            timestamp=datetime.now(),
            processing_time_ms=0.0,
        )

    def _extract_entities(self, transcription: str) -> Dict[str, Any]:
        """
        Extract structured entities from a transcription string.
        Regex patterns and static keywords are kept strictly separate
        to avoid passing plain strings to re.findall().
        """
        entities: Dict[str, Any] = {
            "container_numbers": [],
            "vessel_names": [],
            "ports": [],
            "dates": [],
            "numbers": [],
        }

        upper = transcription.upper()
        lower = transcription.lower()

        # Container numbers (ISO 6346 format)
        entities["container_numbers"] = re.findall(r"\b([A-Z]{3}[UJZ]\d{7})\b", upper)

        # Port names
        for port in _COMMON_PORTS:
            if port in lower:
                entities["ports"].append(port.title())

        # Dates — regex patterns only
        for pattern in _DATE_REGEXES:
            matches = re.findall(pattern, lower)
            entities["dates"].extend(matches)

        # Static date keywords — plain string check, not regex
        for kw in _STATIC_DATE_KEYWORDS:
            if kw in lower and kw not in entities["dates"]:
                entities["dates"].append(kw)

        # Numbers
        entities["numbers"] = re.findall(r"\b\d+\b", transcription)

        return entities

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _append_history(self, record: Dict[str, Any]) -> None:
        self.command_history.append(record)
        if len(self.command_history) > self._max_history:
            self.command_history = self.command_history[-self._max_history:]


# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_voice_interface: Optional[VoiceInterface] = None
_voice_lock = threading.Lock()


def get_voice_interface() -> VoiceInterface:
    """Return the shared voice interface instance (thread-safe, lazy-initialised)."""
    global _voice_interface
    if _voice_interface is None:
        with _voice_lock:
            if _voice_interface is None:
                _voice_interface = VoiceInterface()
    return _voice_interface
