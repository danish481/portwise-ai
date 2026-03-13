"""
PortWise AI - Core Configuration
Maritime Operations Intelligence Platform
"""
import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Application configuration"""

    # API Keys
    GEMINI_API_KEY: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    OPENWEATHER_API_KEY: str = field(default_factory=lambda: os.getenv("OPENWEATHER_API_KEY", ""))

    # Application Settings
    APP_NAME: str = "PortWise AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")

    # LLM Configuration
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_TEMPERATURE: float = 0.3
    GEMINI_MAX_TOKENS: int = 4096

    # Vector Database
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "maritime_docs"
    EMBEDDING_MODEL: str = "models/embedding-001"

    # Document Processing
    MAX_FILE_SIZE_MB: int = 50
    SUPPORTED_DOC_TYPES: List[str] = field(default_factory=list)

    # Voice Settings
    VOICE_LANGUAGE: str = "en-IN"
    VOICE_RECOGNITION_TIMEOUT: int = 10

    # Maritime Data
    DEFAULT_PORTS: List[str] = field(default_factory=list)
    AIS_API_ENDPOINT: str = "https://ais.spire.com"

    # Agent Configuration
    MAX_AGENT_ITERATIONS: int = 10
    AGENT_TIMEOUT_SECONDS: int = 60

    # Prediction engine
    RANDOM_SEED: int = 42  # Keep predictions reproducible in demo mode

    def __post_init__(self):
        self.SUPPORTED_DOC_TYPES = [
            "application/pdf",
            "image/png",
            "image/jpeg",
            "text/plain",
        ]
        self.DEFAULT_PORTS = [
            "Mumbai", "Chennai", "Visakhapatnam", "Kolkata",
            "Singapore", "Rotterdam", "Shanghai", "Dubai",
        ]

    def validate(self) -> None:
        """
        Validate required configuration.
        Warns (does not raise) if the API key is missing so the project can be
        explored without a live key. Set GEMINI_API_KEY in .env to enable LLM features.
        """
        _PLACEHOLDER = "your_gemini_api_key_here"
        if not self.GEMINI_API_KEY or self.GEMINI_API_KEY == _PLACEHOLDER:
            print(
                "\n⚠️  PortWise AI — GEMINI_API_KEY not configured.\n"
                "   LLM-powered features (document AI, compliance, predictions) will be\n"
                "   unavailable until you add a valid key to your .env file.\n"
                "   Get a free key at: https://ai.google.dev/\n"
            )

        _KNOWN_MODELS = {"gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro", "gemini-1.5-flash-latest", "gemini-2.5-flash"}
        if self.GEMINI_MODEL not in _KNOWN_MODELS:
            print(
                f"⚠️  GEMINI_MODEL='{self.GEMINI_MODEL}' is not in the known model list "
                f"{_KNOWN_MODELS}. This may cause API errors."
            )

    @property
    def llm_ready(self) -> bool:
        """Returns True only if the Gemini API key looks usable."""
        _PLACEHOLDER = "your_gemini_api_key_here"
        return bool(self.GEMINI_API_KEY) and self.GEMINI_API_KEY != _PLACEHOLDER


# ---------------------------------------------------------------------------
# Global config singleton
# ---------------------------------------------------------------------------
config = Config()
config.validate()


# ---------------------------------------------------------------------------
# Maritime domain knowledge injected into every agent system prompt
# ---------------------------------------------------------------------------
MARITIME_CONTEXT = """
You are a Maritime Operations Intelligence Specialist with deep expertise in:

1. SHIPPING DOCUMENTS:
   - Bill of Lading (BOL): Contract of carriage, receipt of goods, document of title
   - Container Manifest: List of all containers on vessel with details
   - Dangerous Goods Declaration: IMDG code compliance, proper shipping names
   - Port Call Reports: Arrival/departure times, cargo operations, bunkering
   - Vessel Inspection Reports: Safety, maintenance, compliance checks

2. MARITIME REGULATIONS:
   - IMDG Code (International Maritime Dangerous Goods)
   - SOLAS (Safety of Life at Sea)
   - MARPOL (Marine Pollution)
   - ISPS Code (International Ship and Port Security)

3. SHIPPING OPERATIONS:
   - Container types: 20'DC, 40'DC, 40'HC, Reefer, Tank, Open Top, Flat Rack
   - Incoterms: FOB, CIF, DDP, EXW, etc.
   - Port operations: Berthing, stevedoring, customs, documentation
   - Vessel types: Container ships, bulk carriers, tankers, Ro-Ro

4. COMMON PAIN POINTS:
   - Document processing delays
   - DG compliance violations
   - Port congestion and delays
   - Communication gaps between stakeholders
   - Manual data entry errors
"""

AGENT_PROMPTS = {
    "document_processor": """You are a Document Intelligence Agent specialising in maritime shipping documents.

Your tasks:
1. Extract ALL relevant information from the provided document text
2. Identify document type (BOL, Manifest, DG Declaration, etc.)
3. Extract structured data: container numbers, ports, dates, cargo details
4. Flag missing or suspicious information
5. Validate against known patterns and regulations

Output format: Structured JSON with confidence scores""",

    "predictive_analyst": """You are a Predictive Analytics Agent for maritime operations.

Your tasks:
1. Analyse the query and any provided context to identify the prediction requested
2. Forecast port congestion, vessel delays, or route optimisation as required
3. Recommend optimal arrival/departure windows
4. Identify risk factors for specific routes
5. Provide confidence intervals for your predictions

Consider: Weather, port traffic, vessel history, seasonal patterns, geopolitical factors.
Always clearly state when a prediction is based on limited data.""",

    "compliance_guardian": """You are a Compliance Guardian Agent specialising in maritime regulations.

Your tasks:
1. Validate Dangerous Goods declarations against IMDG code
2. Check documentation completeness for customs
3. Verify safety certifications and inspections
4. Flag potential regulatory violations with specific regulation references
5. Suggest corrective actions with clear priority levels

Regulations: IMDG, SOLAS, MARPOL, ISPS, Customs requirements""",

    "voice_orchestrator": """You are a Voice Command Orchestrator for maritime operations.

Your tasks:
1. Interpret natural language voice commands accurately
2. Route requests to the appropriate specialised agent
3. Synthesise responses suitable for voice output (concise, clear)
4. Handle multi-step operational workflows
5. Maintain conversation context across turns

Common commands: "Show containers", "Check compliance", "Predict delays", "Find documents" """,
}
