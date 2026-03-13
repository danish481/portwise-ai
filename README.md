# 🚢 PortWise AI: Maritime Operations Intelligence Platform

PortWise AI is a voice-first, stateful multi-agent system designed to automate maritime logistics, validate shipping compliance, and forecast port congestion. Built on **LangGraph**, **Scikit-Learn**, and **Gemini 1.5 Flash**, the system replaces fragile NLP pipelines with deterministic, graph-based routing and mathematical fallbacks.

## 🏗️ System Architecture & Core Concepts

The application is built on a strict 4-tier architecture (UI ➔ Agent Orchestration ➔ Core Business Logic ➔ External Services). To prevent race conditions in multi-threaded environments, all core engines utilize the **Double-Checked Locking Singleton** pattern.

### 1. LangGraph Multi-Agent Orchestrator (`agents/orchestrator.py`)

Instead of a linear monolithic LLM chain, PortWise uses a Directed Acyclic Graph (DAG) for agentic reasoning.

- **Concept - Shared State (`MaritimeState`):** Uses a strict Python `TypedDict` to pass context between nodes. This ensures type safety at compile-time and prevents silent key errors during graph execution.  
- **Logic - Deterministic Routing:** The `parse_intent` node classifies queries into a strict whitelist (`_VALID_INTENTS`). The graph's conditional edges route the query to specialist agents, preventing the LLM from hallucinating invalid execution paths.  
- **Logic - Prompt Composition:** Each specialist agent receives a system prompt that concatenates their specific role with a globally shared `MARITIME_CONTEXT` (injecting domain knowledge like IMDG codes and Incoterms without fine-tuning).

> 🚀 **Enhancement Note (Next Level):** Introduce a **Checkpointer** (e.g., PostgreSQL or Redis) to the LangGraph compiler. This would allow for "Human-in-the-Loop" (HITL) workflows, where the graph pauses for a human manager to approve a compliance warning before resuming the AI workflow.

---

### 2. Document Intelligence Engine (`core/document_processor.py`)

A hybrid NLP and rules-based extraction pipeline for Bills of Lading (BOL), Manifests, and Dangerous Goods Declarations.

- **Concept - Composition over Inheritance:** The domain model is built using nested Python `@dataclass` structures (`BillOfLadingData` contains `ExtractedContainer`, `ExtractedVessel`, etc.), allowing recursive JSON serialization.  
- **Logic - ISO 6346 Mathematical Validation:** To prevent LLM hallucinations, container numbers extracted via Regex are passed through a deterministic, from-scratch implementation of the ISO 6346 check-digit algorithm (a weighted sum modulo 11).  
- **Logic - Multi-Modal OCR:** Uses `pdfplumber` for spatial PDF parsing and `pytesseract` for image OCR, applying `ImageEnhance.Contrast` and `ImageFilter.SHARPEN` pre-processing to handle degraded port scans.

> 🚀 **Enhancement Note (Next Level):** Rip out the Tesseract/Regex pipeline and replace it with a **Vision-Language Model (VLM)** (like Gemini 1.5 Pro Vision). Pass the raw document image directly to the model with a strict JSON schema output parser for zero-shot spatial layout understanding.

---

### 3. Predictive Analytics Pipeline (`core/predictive_engine.py`)

An embedded machine learning engine for operational forecasting.

- **Concept - Data Leakage Prevention:** Uses `sklearn.pipeline.Pipeline` to chain `StandardScaler` transformations with the estimators, ensuring training statistics do not leak into validation steps.  
- **Logic - Ensemble Methods:**  
  - Uses **Random Forest Regressor** to predict continuous target variables (vessel delay hours). A max depth of 8 is enforced to prevent overfitting on the baseline data.  
  - Uses **Gradient Boosting Classifier** to predict categorical severity (Low, Moderate, High congestion), capitalizing on sequential error correction.  
- **Logic - Seeded Simulation:** Uses isolated, heavily seeded Numpy Random Number Generators (`np.random.RandomState`) to generate synthetic baseline training data, ensuring demo reproducibility.

> 🚀 **Enhancement Note (Next Level):** Deprecate the synthetic data generation and connect the pipeline to live **AIS (Automatic Identification System)** telemetry APIs (e.g., Spire or MarineTraffic) and live weather APIs. Store historical telemetry in a Feature Store (like Feast) for continuous model retraining.

---

### 4. Voice Interface & UI (`core/voice_interface.py` & `ui/app.py`)

A reactive, voice-enabled command center built on Streamlit.

- **Concept - Dual-Engine STT:** Implements a fallback resilience pattern. It attempts transcription via Google Web Speech API (optimized for regional accents) and fails over to keyword heuristic matching if network/API limits are hit.  
- **Logic - Async/Sync Bridging:** Because LangGraph operates asynchronously but Streamlit is synchronous, the app uses a custom event-loop detection pattern (`asyncio.get_event_loop().is_running()`) to safely spawn ThreadPoolExecutors without crashing the UI thread.

> 🚀 **Enhancement Note (Next Level):** Decouple the backend. Move the LangGraph orchestrator and ML pipeline into a dedicated **FastAPI** backend microservice. Connect the Streamlit frontend via WebSockets to allow real-time, token-by-token streaming of the agent's thought process.

---

## 💻 Local Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/portwise-ai.git
cd portwise-ai

# 2. Create a Conda Environment
conda create --name portwise python=3.11 -y
conda activate portwise

# 3. Install Dependencies & System Packages
python -m pip install -r requirements.txt
conda install -c conda-forge ffmpeg -y

# 4. Environment Variables (.env file)
GEMINI_API_KEY="your_google_ai_studio_key_here"
GEMINI_MODEL="gemini-2.5-flash"

# 5. Run the Application
streamlit run ui/app.py