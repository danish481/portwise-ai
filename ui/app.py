"""
PortWise AI - Streamlit Web Application
Maritime Operations Intelligence Platform

Run with:
    streamlit run ui/app.py
"""
import sys
import os
import json
import logging
from datetime import datetime
from typing import Optional

# Make sure project root is on the path when launched as `streamlit run ui/app.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PortWise AI",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports (avoid crashing the UI on missing optional deps)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_document_processor():
    from core.document_processor import get_document_processor
    return get_document_processor()

@st.cache_resource(show_spinner=False)
def _load_predictive_engine():
    from core.predictive_engine import get_predictive_engine
    return get_predictive_engine()

@st.cache_resource(show_spinner=False)
def _load_voice_interface():
    from core.voice_interface import get_voice_interface
    return get_voice_interface()

@st.cache_resource(show_spinner=False)
def _load_orchestrator():
    from agents.orchestrator import get_orchestrator
    return get_orchestrator()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
from core.config import config

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Global ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e3a5f 100%); }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebarNav"] a { color: #93c5fd !important; }

/* ── Metric cards ─────────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #1e3a5f, #2563eb);
    border-radius: 12px; padding: 20px; color: white;
    text-align: center; margin: 4px;
    box-shadow: 0 4px 12px rgba(37,99,235,0.3);
}
.metric-number { font-size: 2.4rem; font-weight: 800; margin: 0; }
.metric-label  { font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }

/* ── Status badges ────────────────────────────────────────────────────── */
.badge-green  { background:#16a34a; color:#fff; padding:3px 10px; border-radius:999px; font-size:0.8rem; font-weight:600; }
.badge-amber  { background:#d97706; color:#fff; padding:3px 10px; border-radius:999px; font-size:0.8rem; font-weight:600; }
.badge-red    { background:#dc2626; color:#fff; padding:3px 10px; border-radius:999px; font-size:0.8rem; font-weight:600; }
.badge-blue   { background:#2563eb; color:#fff; padding:3px 10px; border-radius:999px; font-size:0.8rem; font-weight:600; }

/* ── Demo warning ─────────────────────────────────────────────────────── */
.demo-notice {
    background: #fef3c7; border-left: 4px solid #d97706;
    border-radius: 6px; padding: 10px 14px; font-size: 0.85rem; color: #78350f;
    margin-bottom: 8px;
}

/* ── Section headers ──────────────────────────────────────────────────── */
.section-header {
    background: linear-gradient(90deg, #1e3a5f, #2563eb);
    color: white; padding: 10px 18px; border-radius: 8px;
    font-weight: 700; font-size: 1.05rem; margin-bottom: 12px;
}

/* ── Feature tag ──────────────────────────────────────────────────────── */
.feature-tag {
    background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe;
    padding:3px 10px; border-radius:999px; font-size:0.78rem;
    display:inline-block; margin:2px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🚢 PortWise AI")
    st.markdown("*Maritime Operations Intelligence*")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🎤 Voice Command", "📄 Document Intelligence",
         "🔮 Predictive Analytics", "✅ Compliance Guardian", "🤖 AI Assistant"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**System Status**")
    if config.llm_ready:
        st.success("🟢 LLM: Connected")
    else:
        st.warning("🟡 LLM: Key not set")
        st.caption("Add GEMINI_API_KEY to .env to enable AI features.")

    st.info("🔵 Analytics: Ready (demo models)")
    st.info("🔵 Document Processor: Ready")

    st.markdown("---")
    st.caption(f"PortWise AI v{config.APP_VERSION}")
    st.caption(f"Model: {config.GEMINI_MODEL}")


# ---------------------------------------------------------------------------
# Helper widgets
# ---------------------------------------------------------------------------
def _demo_notice(text: str = "Predictions use synthetic demo models — not live port data."):
    st.markdown(f'<div class="demo-notice">⚠️ {text}</div>', unsafe_allow_html=True)

def _section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def _badge(label: str, color: str = "blue") -> str:
    return f'<span class="badge-{color}">{label}</span>'

def _metric_card(number: str, label: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-number">{number}</div>
        <div class="metric-label">{label}</div>
    </div>"""


# ===========================================================================
# PAGES
# ===========================================================================

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
if page == "🏠 Dashboard":
    st.title("🚢 PortWise AI")
    st.subheader("Maritime Operations Intelligence Platform")

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(_metric_card("250+", "Vessels Monitored"), unsafe_allow_html=True)
    with col2: st.markdown(_metric_card("95%", "Compliance Accuracy"), unsafe_allow_html=True)
    with col3: st.markdown(_metric_card("3 min", "Avg Doc Processing"), unsafe_allow_html=True)
    with col4: st.markdown(_metric_card("$50K", "Annual Savings / Vessel"), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🎯 Platform Capabilities")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        with st.expander("🎤 Voice Command Centre", expanded=True):
            st.markdown("""
Natural language queries for maritime operations.
- Speech-to-text with Indian English optimisation
- Context-aware maritime responses
- Container tracking by voice
- Multi-turn conversations
""")
            for tag in ["STT", "Whisper fallback", "Multi-language"]:
                st.markdown(f'<span class="feature-tag">{tag}</span>', unsafe_allow_html=True)

        with st.expander("📄 Document Intelligence", expanded=True):
            st.markdown("""
AI-powered extraction from maritime documents.
- Bill of Lading (BOL) parsing
- Dangerous Goods Declaration analysis
- Container Manifest processing
- ISO 6346 container validation
""")
            for tag in ["PDF", "OCR", "DG Docs", "BOL", "Manifests"]:
                st.markdown(f'<span class="feature-tag">{tag}</span>', unsafe_allow_html=True)

    with r1c2:
        with st.expander("🔮 Predictive Analytics", expanded=True):
            st.markdown("""
ML-powered forecasting for operations planning.
- Port congestion (7–14 day forecasts)
- Vessel delay prediction
- Route optimisation & cost estimates
- Fleet performance analytics
""")
            _demo_notice("Analytics use synthetic models trained on baseline data.")
            for tag in ["RandomForest", "GradientBoosting", "Seeded RNG"]:
                st.markdown(f'<span class="feature-tag">{tag}</span>', unsafe_allow_html=True)

        with st.expander("✅ Compliance Guardian", expanded=True):
            st.markdown("""
Automated regulatory compliance checking.
- IMDG Code validation
- SOLAS / MARPOL / ISPS checks
- Documentation completeness review
- Actionable corrective recommendations
""")
            for tag in ["IMDG", "SOLAS", "MARPOL", "ISPS"]:
                st.markdown(f'<span class="feature-tag">{tag}</span>', unsafe_allow_html=True)

    st.markdown("---")
    _section("📊 Quick Port Congestion Snapshot")
    _demo_notice()

    engine = _load_predictive_engine()
    ports_to_show = ["Mumbai", "Singapore", "Rotterdam", "Shanghai", "Dubai"]
    cols = st.columns(len(ports_to_show))
    for col, port in zip(cols, ports_to_show):
        with col:
            forecast = engine.predict_port_congestion(port, days_ahead=1)
            color_map = {"low": "🟢", "moderate": "🟡", "high": "🟠", "severe": "🔴"}
            icon = color_map.get(forecast.congestion_level, "⚪")
            st.metric(
                port,
                f"{forecast.average_waiting_hours:.1f} h wait",
                delta=f"{icon} {forecast.congestion_level.capitalize()}",
            )


# ---------------------------------------------------------------------------
# Voice Command
# ---------------------------------------------------------------------------
elif page == "🎤 Voice Command":
    st.title("🎤 Voice Command Centre")
    st.markdown("Query maritime operations using natural language — type or upload audio.")
    st.markdown("---")

    tab_text, tab_audio = st.tabs(["💬 Text Input", "🎙️ Audio Upload"])

    with tab_text:
        _section("Type Your Command")
        st.markdown("""
**Example commands:**
- *"Show me container status for vessel Pacific Star"*
- *"Predict delays for Mumbai port next week"*
- *"Check compliance for DG shipment UN2348"*
- *"What is the ETA for container MSCU7894562?"*
""")
        user_input = st.text_area("Enter your maritime query:", height=100,
                                   placeholder="e.g. Predict delays for Singapore port")

        if st.button("🚀 Process Command", type="primary"):
            if user_input.strip():
                with st.spinner("Processing through AI agents…"):
                    if config.llm_ready:
                        try:
                            orch = _load_orchestrator()
                            result = orch.process_query_sync(user_input)

                            st.success("✅ Query processed successfully")
                            col_l, col_r = st.columns([1, 2])
                            with col_l:
                                st.metric("Intent Detected", result.get("intent", "general").title())
                                if result.get("errors"):
                                    st.warning(f"⚠️ {len(result['errors'])} warning(s)")
                            with col_r:
                                if result.get("response"):
                                    st.markdown("**AI Response:**")
                                    st.info(result["response"])
                        except Exception as exc:
                            st.error(f"Agent error: {exc}")
                    else:
                        # Graceful demo mode without a key
                        voice = _load_voice_interface()
                        cmd = voice._classify(user_input)
                        st.info(
                            f"**Demo mode** (add GEMINI_API_KEY for full AI responses)\n\n"
                            f"**Detected intent:** `{cmd.command_type}`\n\n"
                            f"**Entities found:** {json.dumps(cmd.entities, indent=2)}"
                        )
            else:
                st.warning("Please enter a query.")

    with tab_audio:
        _section("Upload Audio File")
        st.markdown("Supported formats: WAV, MP3, M4A, WebM, OGG")
        audio_file = st.file_uploader("Upload audio command", type=["wav", "mp3", "m4a", "webm", "ogg"])

        lang = st.selectbox("Language", ["en-IN", "en-US", "en-GB"], index=0)

        if audio_file and st.button("🎙️ Transcribe & Process", type="primary"):
            with st.spinner("Transcribing audio…"):
                voice = _load_voice_interface()
                audio_bytes = audio_file.read()
                fmt = audio_file.name.rsplit(".", 1)[-1].lower()
                result = voice.process_audio_bytes(audio_bytes, fmt=fmt, language=lang)

            if result["success"]:
                st.success("✅ Transcription successful")
                col_l, col_r = st.columns(2)
                with col_l:
                    st.metric("Engine", result.get("engine", "unknown").capitalize())
                    st.metric("Confidence", f"{result.get('confidence', 0):.0%}")
                    st.metric("Command Type", result.get("command_type", "—").replace("_", " ").title())
                with col_r:
                    st.markdown("**Transcription:**")
                    st.info(result.get("transcription", ""))
                    if result.get("entities"):
                        with st.expander("🔍 Extracted Entities"):
                            st.json(result["entities"])
            else:
                st.error(f"Transcription failed: {result.get('error', 'Unknown error')}")
                st.caption("Tip: Ensure ffmpeg is installed for non-WAV formats.")


# ---------------------------------------------------------------------------
# Document Intelligence
# ---------------------------------------------------------------------------
elif page == "📄 Document Intelligence":
    st.title("📄 Document Intelligence Engine")
    st.markdown("Upload maritime documents for AI-powered data extraction.")
    st.markdown("---")

    tab_upload, tab_validate = st.tabs(["📤 Upload & Extract", "🔢 Container Validator"])

    with tab_upload:
        _section("Upload Document")
        uploaded = st.file_uploader(
            "Upload a maritime document",
            type=["pdf", "png", "jpg", "jpeg", "txt"],
            help="Supports PDF (text + scanned), images (OCR), and plain text.",
        )
        doc_type_hint = st.selectbox(
            "Document type hint (optional — auto-detected if blank)",
            ["Auto-detect", "bol", "manifest", "dg_declaration"],
        )

        if uploaded and st.button("⚡ Extract Data", type="primary"):
            import tempfile, os
            suffix = "." + uploaded.name.rsplit(".", 1)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            with st.spinner(f"Processing '{uploaded.name}'…"):
                processor = _load_document_processor()
                hint = None if doc_type_hint == "Auto-detect" else doc_type_hint
                try:
                    result = processor.process_document(tmp_path, hint)
                finally:
                    os.unlink(tmp_path)

            data = result.get("extracted_data", {})
            conf = data.get("extract_confidence", 0.0)

            # Header row
            c1, c2, c3 = st.columns(3)
            c1.metric("Document Type", result.get("document_type", "Unknown").upper())
            c2.metric("Confidence", f"{conf:.0%}")
            c3.metric("Document ID", result.get("document_id", "—"))

            st.markdown("---")

            doc_t = result.get("document_type", "unknown")

            if doc_t == "bol":
                col_l, col_r = st.columns(2)
                with col_l:
                    _section("📋 Document Details")
                    st.markdown(f"**BOL Number:** `{data.get('bol_number', 'Not found')}`")
                    st.markdown(f"**Booking Number:** `{data.get('booking_number', 'Not found')}`")
                    st.markdown(f"**Incoterm:** `{data.get('incoterm', 'Not found')}`")
                    st.markdown(f"**Freight Terms:** `{data.get('freight_terms', 'Not found')}`")
                    st.markdown(f"**Issue Date:** `{data.get('issue_date', 'Not found')}`")

                    _section("🚢 Vessel")
                    v = data.get("vessel", {})
                    st.markdown(f"**Name:** {v.get('vessel_name', '—')}")
                    st.markdown(f"**Voyage:** {v.get('voyage_number', '—')}")
                    st.markdown(f"**IMO:** {v.get('imo_number', '—')}")

                with col_r:
                    _section("🌍 Ports")
                    p = data.get("ports", {})
                    st.markdown(f"**Port of Loading:** {p.get('port_of_loading', '—')}")
                    st.markdown(f"**Port of Discharge:** {p.get('port_of_discharge', '—')}")
                    st.markdown(f"**Place of Receipt:** {p.get('place_of_receipt', '—')}")
                    st.markdown(f"**Place of Delivery:** {p.get('place_of_delivery', '—')}")

                    _section("👥 Parties")
                    parties = data.get("parties", {})
                    st.markdown(f"**Shipper:** {parties.get('shipper_name', '—')}")
                    st.markdown(f"**Consignee:** {parties.get('consignee_name', '—')}")
                    st.markdown(f"**Notify Party:** {parties.get('notify_party', '—')}")
                    st.markdown(f"**Carrier:** {parties.get('carrier_name', '—')}")

                _section(f"📦 Containers ({len(data.get('containers', []))} found)")
                for ctr in data.get("containers", []):
                    with st.expander(f"🔷 {ctr['container_number']} — {ctr['container_type']}"):
                        cc1, cc2 = st.columns(2)
                        cc1.markdown(f"**Seal:** `{ctr.get('seal_number', '—')}`")
                        cc1.markdown(f"**Weight:** {format(ctr.get('weight_kg'), ',.0f') + ' KGS' if ctr.get('weight_kg') else '—'}")
                        cc2.markdown(f"**DG:** {'🔴 Yes' if ctr.get('is_dangerous_goods') else '🟢 No'}")
                        if ctr.get("is_dangerous_goods"):
                            cc2.markdown(f"**UN:** {ctr.get('un_number', '—')}")
                            cc2.markdown(f"**IMDG Class:** {ctr.get('imdg_class', '—')}")

            elif doc_t == "dg_declaration":
                _section("☢️ Dangerous Goods Summary")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Total DG Items", data.get("total_dg_items", 0))
                    st.markdown("**UN Numbers:**")
                    for un in data.get("un_numbers", []):
                        st.markdown(f"- `UN{un}`")
                with c2:
                    st.markdown("**IMDG Classes:**")
                    for cls in data.get("imdg_classes", []):
                        st.markdown(f"- Class `{cls}`")
                    st.markdown("**Packing Groups:**")
                    for pg in data.get("packing_groups", []):
                        st.markdown(f"- Group `{pg}`")

                if data.get("proper_shipping_names"):
                    _section("📋 Proper Shipping Names")
                    for name in data["proper_shipping_names"]:
                        st.markdown(f"- {name}")
            else:
                with st.expander("📋 Raw Extracted Data", expanded=True):
                    st.json(data)

            # AI enhancement (if key configured)
            if config.llm_ready:
                st.markdown("---")
                if st.button("🤖 Enhance with AI Analysis"):
                    with st.spinner("Sending to AI agent…"):
                        orch = _load_orchestrator()
                        query = f"Analyse this extracted document data and provide insights: {json.dumps(data)}"
                        ai_result = orch.process_query_sync(query)
                    if ai_result.get("response"):
                        st.info(ai_result["response"])

    with tab_validate:
        _section("🔢 ISO 6346 Container Number Validator")
        st.markdown("Validate container numbers using the official ISO 6346 check-digit algorithm.")

        container_input = st.text_input(
            "Container Number",
            max_chars=11,
            placeholder="e.g. MSCU7894562",
        ).strip().upper()

        if st.button("✅ Validate", type="primary"):
            if container_input:
                processor = _load_document_processor()
                is_valid, msg = processor.validate_container_number(container_input)
                if is_valid:
                    st.success(f"✅ **{container_input}** — {msg}")
                    owner = container_input[:3]
                    category_map = {"U": "Freight Container", "J": "Detachable Freight Container Equipment", "Z": "Trailer / Chassis"}
                    st.info(
                        f"**Owner Code:** `{owner}`  \n"
                        f"**Category:** {category_map.get(container_input[3], 'Unknown')}  \n"
                        f"**Serial:** `{container_input[4:10]}`  \n"
                        f"**Check Digit:** `{container_input[10]}`"
                    )
                else:
                    st.error(f"❌ **{container_input}** — {msg}")
            else:
                st.warning("Enter a container number to validate.")


# ---------------------------------------------------------------------------
# Predictive Analytics
# ---------------------------------------------------------------------------
elif page == "🔮 Predictive Analytics":
    st.title("🔮 Predictive Analytics")
    st.markdown("ML-powered forecasting for ports, vessels, and routes.")
    _demo_notice("All predictions use synthetic models trained on baseline constants. For production, connect real AIS and port telemetry data.")
    st.markdown("---")

    tab_port, tab_vessel, tab_route, tab_fleet = st.tabs(
        ["🏗️ Port Congestion", "⚓ Vessel Delay", "🗺️ Route Optimiser", "📊 Fleet Performance"]
    )

    engine = _load_predictive_engine()

    with tab_port:
        _section("Port Congestion Forecast")
        c1, c2, c3 = st.columns(3)
        with c1:
            port = st.selectbox("Select Port", list(engine.PORT_BASELINE.keys()))
        with c2:
            days_ahead = st.slider("Days Ahead", 1, 14, 7)
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            run = st.button("📊 Generate Forecast", type="primary")

        if run:
            with st.spinner("Running congestion model…"):
                forecast = engine.predict_port_congestion(port, days_ahead=days_ahead)

            color_map = {"low": "green", "moderate": "amber", "high": "amber", "severe": "red"}
            badge_color = color_map.get(forecast.congestion_level, "blue")

            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Waiting Time", f"{forecast.average_waiting_hours:.1f} h")
            col2.metric("Berth Availability", f"{forecast.berth_availability_pct:.0f}%")
            col3.metric("Confidence", f"{forecast.confidence_score:.0%}")

            st.markdown(
                f"**Congestion Level:** {_badge(forecast.congestion_level.upper(), badge_color)} "
                f"&nbsp; **Forecast Date:** {forecast.forecast_date[:10]}",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Recommendation:** {forecast.recommendation}")

            with st.expander("📋 Contributing Factors"):
                for f in forecast.contributing_factors:
                    st.markdown(f"- {f}")

            # All-ports comparison
            st.markdown("---")
            _section("All-Ports Comparison")
            import pandas as pd
            rows = []
            for p in engine.PORT_BASELINE.keys():
                fc = engine.predict_port_congestion(p, days_ahead=days_ahead)
                rows.append({
                    "Port": p,
                    "Wait (h)": fc.average_waiting_hours,
                    "Berths (%)": fc.berth_availability_pct,
                    "Level": fc.congestion_level.capitalize(),
                    "Confidence": f"{fc.confidence_score:.0%}",
                })
            df = pd.DataFrame(rows).sort_values("Wait (h)", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tab_vessel:
        _section("Vessel Delay Prediction")
        c1, c2 = st.columns(2)
        with c1:
            vessel_name = st.text_input("Vessel Name", "MV Pacific Star")
            voyage_num = st.text_input("Voyage Number", "PS247N")
        with c2:
            dest_port = st.selectbox("Destination Port", [""] + list(engine.PORT_BASELINE.keys()), index=0)
            current_eta = st.text_input("Current ETA (ISO format, optional)", placeholder="2025-06-15T08:00:00")

        if st.button("🔮 Predict Delay", type="primary"):
            if vessel_name and voyage_num:
                with st.spinner("Running delay model…"):
                    pred = engine.predict_vessel_delay(
                        vessel_name, voyage_num,
                        destination_port=dest_port or None,
                        current_eta=current_eta or None,
                    )

                risk_colors = {"low": "green", "medium": "amber", "high": "amber", "critical": "red"}
                rc = risk_colors.get(pred.risk_level.value, "blue")

                col1, col2, col3 = st.columns(3)
                col1.metric("Predicted Delay", f"{pred.predicted_delay_hours:.1f} h")
                col2.metric("Delay Probability", f"{pred.delay_probability:.0%}")
                col3.metric("Confidence", f"{pred.confidence_score:.0%}")

                st.markdown(
                    f"**Risk Level:** {_badge(pred.risk_level.value.upper(), rc)}",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Revised ETA:** `{pred.estimated_arrival[:16]}`")

                with st.expander("🔍 Primary Causes"):
                    for cause in pred.primary_causes:
                        st.markdown(f"- {cause}")

                with st.expander("💡 Recommended Actions"):
                    for action in pred.alternative_actions:
                        st.markdown(f"- {action}")
            else:
                st.warning("Enter vessel name and voyage number.")

    with tab_route:
        _section("Route Optimiser")
        ports = list(engine.PORT_BASELINE.keys())
        c1, c2, c3 = st.columns(3)
        with c1:
            origin = st.selectbox("Origin Port", ports, index=0)
            container_num = st.text_input("Container Number", "MSCU7894562")
        with c2:
            destination = st.selectbox("Destination Port", ports, index=1)
            ctype = st.selectbox("Container Type", ["40'DC", "20'DC", "40'HC", "40'RF", "20'TK"])
        with c3:
            st.markdown("<br><br>", unsafe_allow_html=True)
            predict_route = st.button("🗺️ Optimise Route", type="primary")

        if predict_route:
            if origin == destination:
                st.warning("Origin and destination must be different.")
            else:
                with st.spinner("Calculating optimal route…"):
                    pred = engine.predict_container_route(container_num, origin, destination, ctype)

                col1, col2, col3 = st.columns(3)
                col1.metric("Transit Time", f"{pred.predicted_transit_days:.1f} days")
                col2.metric("Cost Estimate", f"${pred.cost_estimate_usd:,.0f} USD")
                col3.metric("Confidence", f"{pred.confidence_score:.0%}")

                _section("🗺️ Optimal Route")
                route_display = " → ".join(pred.optimal_route)
                st.markdown(f"**{route_display}**")

                with st.expander("⚠️ Risk Factors"):
                    for risk in pred.risk_factors:
                        st.markdown(f"- {risk}")

    with tab_fleet:
        _section("Fleet Performance Analytics")
        fleet_input = st.text_area(
            "Enter vessel names (one per line):",
            "MV Pacific Star\nMV Atlantic Eagle\nMV Indian Ocean\nMV Arabian Sea\nMV Bay of Bengal",
        )
        period = st.slider("Analysis Period (days)", 7, 90, 30)

        if st.button("📊 Analyse Fleet", type="primary"):
            vessels = [v.strip() for v in fleet_input.splitlines() if v.strip()]
            if vessels:
                with st.spinner("Analysing fleet performance…"):
                    result = engine.analyse_fleet_performance(vessels, period)

                summary = result["fleet_summary"]
                col1, col2, col3 = st.columns(3)
                col1.metric("Fleet OTP", f"{summary['fleet_on_time_performance']:.0%}")
                col2.metric("Avg Delay", f"{summary['fleet_average_delay_hours']:.1f} h")
                col3.metric("Vessels", summary["total_vessels"])

                col_top, col_att = st.columns(2)
                with col_top:
                    _section("🏆 Top Performers")
                    for v in result["top_performers"]:
                        st.markdown(f"**{v['vessel_name']}** — OTP: {v['on_time_rate']:.0%}, "
                                    f"Reliability: {v['reliability_score']:.0%}")

                with col_att:
                    if result["needs_attention"]:
                        _section("⚠️ Needs Attention")
                        for v in result["needs_attention"]:
                            st.markdown(f"**{v['vessel_name']}** — Reliability: {v['reliability_score']:.0%}")
                    else:
                        st.success("All vessels meeting reliability targets.")

                with st.expander("💡 Fleet Recommendations"):
                    for rec in result["recommendations"]:
                        st.markdown(f"- {rec}")
            else:
                st.warning("Enter at least one vessel name.")


# ---------------------------------------------------------------------------
# Compliance Guardian
# ---------------------------------------------------------------------------
elif page == "✅ Compliance Guardian":
    st.title("✅ Compliance Guardian")
    st.markdown("Automated maritime regulatory compliance checking.")
    st.markdown("---")

    tab_dg, tab_doc = st.tabs(["☢️ DG Compliance", "📋 Documentation Check"])

    with tab_dg:
        _section("Dangerous Goods Compliance Check")

        c1, c2 = st.columns(2)
        with c1:
            un_number = st.text_input("UN Number", "2348", max_chars=4)
            proper_name = st.text_input("Proper Shipping Name", "BUTYRALDEHYDE")
        with c2:
            imdg_class = st.selectbox("IMDG Class",
                ["1 - Explosives", "2 - Gases", "3 - Flammable Liquids",
                 "4 - Flammable Solids", "5 - Oxidizers", "6 - Toxic",
                 "7 - Radioactive", "8 - Corrosives", "9 - Miscellaneous"])
            packing_group = st.selectbox("Packing Group", ["I", "II", "III", "N/A"])

        query = (
            f"Check IMDG compliance for: UN{un_number} {proper_name}, "
            f"{imdg_class}, Packing Group {packing_group}"
        )

        if st.button("✅ Run Compliance Check", type="primary"):
            if config.llm_ready:
                with st.spinner("Running compliance analysis via LLM…"):
                    orch = _load_orchestrator()
                    result = orch.process_query_sync(query)

                st.success("✅ Compliance analysis complete")
                if result.get("response"):
                    st.markdown("**AI Compliance Report:**")
                    st.info(result["response"])
                if result.get("compliance_report"):
                    with st.expander("📋 Raw Report"):
                        st.json(result["compliance_report"])
            else:
                # Rule-based demo mode
                st.warning("⚠️ Running in demo mode — add GEMINI_API_KEY for full AI analysis.")
                st.markdown(f"""
**Demo Compliance Check — UN{un_number} {proper_name}**

| Check | Status |
|-------|--------|
| UN Number format | ✅ Valid 4-digit number |
| IMDG Class | ✅ {imdg_class} |
| Packing Group | ✅ {packing_group} |
| Documentation required | 📋 DG Declaration, MSDS, Packing Certificate |

**⚠️ Always verify against the current IMDG Code publication.**
""")

    with tab_doc:
        _section("Documentation Completeness Check")
        st.markdown("Check that all required documents are present for a shipment.")

        doc_checklist = {
            "Bill of Lading": st.checkbox("Bill of Lading (BOL)", value=True),
            "Packing List": st.checkbox("Packing List", value=True),
            "Commercial Invoice": st.checkbox("Commercial Invoice", value=False),
            "Certificate of Origin": st.checkbox("Certificate of Origin", value=False),
            "DG Declaration": st.checkbox("Dangerous Goods Declaration", value=False),
            "MSDS": st.checkbox("Material Safety Data Sheet (MSDS)", value=False),
            "Phytosanitary Certificate": st.checkbox("Phytosanitary Certificate", value=False),
            "Insurance Certificate": st.checkbox("Insurance Certificate", value=False),
        }

        if st.button("📋 Check Documentation", type="primary"):
            provided = [k for k, v in doc_checklist.items() if v]
            missing = [k for k, v in doc_checklist.items() if not v]

            col1, col2 = st.columns(2)
            with col1:
                st.success(f"✅ **{len(provided)} documents provided:**")
                for doc in provided:
                    st.markdown(f"- ✅ {doc}")
            with col2:
                if missing:
                    st.warning(f"⚠️ **{len(missing)} documents missing:**")
                    for doc in missing:
                        st.markdown(f"- ❌ {doc}")
                else:
                    st.success("✅ All documents accounted for!")

            completeness = len(provided) / max(len(doc_checklist), 1)
            st.progress(completeness, text=f"Documentation completeness: {completeness:.0%}")


# ---------------------------------------------------------------------------
# AI Assistant
# ---------------------------------------------------------------------------
elif page == "🤖 AI Assistant":
    st.title("🤖 AI Maritime Assistant")
    st.markdown("Conversational AI powered by Google Gemini — ask anything about maritime operations.")
    st.markdown("---")

    if not config.llm_ready:
        st.warning(
            "🔑 **GEMINI_API_KEY not configured.**\n\n"
            "Add your key to `.env` and restart the app to enable the AI Assistant.\n\n"
            "Get a free API key at: https://ai.google.dev/"
        )
        st.stop()

    # Chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask a maritime question…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                orch = _load_orchestrator()
                result = orch.process_query_sync(prompt)
            response = result.get("response", "I encountered an error. Please try again.")
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

        if result.get("errors"):
            with st.expander("⚠️ Agent warnings"):
                for e in result["errors"]:
                    st.caption(e)

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
