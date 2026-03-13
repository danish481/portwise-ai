"""
PortWise AI - Maritime Document Processor
Handles Bills of Lading, Manifests, DG Declarations, and more.
"""
import re
import json
import logging
import threading
import hashlib
import base64
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import pdfplumber
from PIL import Image, ImageFilter, ImageEnhance
import pytesseract

# LangChain / Gemini Vision Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from core.config import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ExtractedContainer:
    container_number: str
    container_type: str
    seal_number: Optional[str] = None
    weight_kg: Optional[float] = None
    cargo_description: Optional[str] = None
    is_dangerous_goods: bool = False
    imdg_class: Optional[str] = None
    un_number: Optional[str] = None


@dataclass
class ExtractedVessel:
    vessel_name: Optional[str] = None
    voyage_number: Optional[str] = None
    imo_number: Optional[str] = None
    flag: Optional[str] = None


@dataclass
class ExtractedPorts:
    port_of_loading: Optional[str] = None
    port_of_discharge: Optional[str] = None
    place_of_receipt: Optional[str] = None
    place_of_delivery: Optional[str] = None
    transshipment_ports: List[str] = None

    def __post_init__(self):
        if self.transshipment_ports is None:
            self.transshipment_ports = []


@dataclass
class ExtractedParties:
    shipper_name: Optional[str] = None
    shipper_address: Optional[str] = None
    consignee_name: Optional[str] = None
    consignee_address: Optional[str] = None
    notify_party: Optional[str] = None
    carrier_name: Optional[str] = None


@dataclass
class BillOfLadingData:
    document_type: str = "bol"
    bol_number: Optional[str] = None
    booking_number: Optional[str] = None
    vessel: ExtractedVessel = None
    ports: ExtractedPorts = None
    parties: ExtractedParties = None
    containers: List[ExtractedContainer] = None
    incoterm: Optional[str] = None
    freight_terms: Optional[str] = None
    issue_date: Optional[str] = None
    onboard_date: Optional[str] = None
    extract_confidence: float = 0.0
    raw_text: Optional[str] = None

    def __post_init__(self):
        if self.vessel is None:
            self.vessel = ExtractedVessel()
        if self.ports is None:
            self.ports = ExtractedPorts()
        if self.parties is None:
            self.parties = ExtractedParties()
        if self.containers is None:
            self.containers = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# BIC-registered owner-code prefixes
# ---------------------------------------------------------------------------
_KNOWN_OWNER_CODES = frozenset({
    "MSCU", "MAEU", "CMAU", "HLCU", "EGLV", "OOLU", "CSNU", "YMLU",
    "APHU", "COSU", "EVGU", "PCIU", "TGHU", "MRKU", "FSCU", "TCNU",
    "SEAU", "TTNU", "TRHU", "GLDU", "CCLU", "NYKU", "WILU", "KKFU",
    "MATU", "GESU", "KNLU", "INBU", "HDMU", "ZIMU", "REGU", "SGCU",
})

PORT_MAPPING: Dict[str, str] = {
    "INBOM": "Mumbai",
    "INMAA": "Chennai",
    "INVTZ": "Visakhapatnam",
    "INCCU": "Kolkata",
    "SGSIN": "Singapore",
    "NLRTM": "Rotterdam",
    "CNSHA": "Shanghai",
    "AEDXB": "Dubai",
    "USLAX": "Los Angeles",
    "DEHAM": "Hamburg",
    "GBFXT": "Felixstowe",
    "BEANR": "Antwerp",
    "KRPUS": "Busan",
    "JPTYO": "Tokyo",
    "AUPOR": "Port Botany",
}

_POL_KEYWORDS = frozenset({"port of loading", "pol", "load port", "loading port"})
_POD_KEYWORDS = frozenset({"port of discharge", "pod", "discharge port", "destination port"})
_RECEIPT_KEYWORDS = frozenset({"place of receipt", "por", "receipt"})
_DELIVERY_KEYWORDS = frozenset({"place of delivery", "final destination", "delivery place"})


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class MaritimeDocumentProcessor:
    """
    Advanced document processor for maritime shipping documents.
    Supports native LLM Vision API for images, falling back to legacy OCR/Regex for PDFs.
    """

    PATTERNS = {
        "container_number": r"\b([A-Z]{3}[UJZ]\d{7})\b",
        "bol_number": r"(?:B/L\s*(?:No\.?|Number)?|BOL\s*(?:No\.?|Number)?|Bill\s+of\s+Lading\s*(?:No\.?)?)[\s#:]+([A-Z0-9][\w\-]{2,20})",
        "vessel_name": r"Vessel\s+Name[\s:]+([A-Za-z][^\n\r,]{2,50})",
        "voyage_number": r"Voyage(?:\s+(?:No\.?|Number))?[\s#:]+([A-Z0-9\-]+)",
        "imo_number": r"IMO[\s#:]+(\d{7})",
        "port_code": r"\b([A-Z]{2}[A-Z0-9]{3})\b",
        "seal_number": r"Seal(?:\s+(?:No\.?|Number))?[\s#:]+([A-Z0-9\-]+)",
        "un_number": r"\bUN[\s\-]?(\d{4})\b",
        "imdg_class": r"(?:Class|IMDG\s+Class)[\s:]+(\d+\.?\d*[A-Z]?)",
        "weight": r"(\d[\d,\.]*)\s*(?:kgs?|kilos?|KGS?)",
        "date": r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{1,2}\s+\w{3,9}\s+\d{4})\b",
        "incoterm": r"\b(FOB|CIF|CFR|CIP|CPT|DAP|DDP|DPU|EXW|FAS|FCA)\b",
        "booking_number": r"(?:Booking\s+(?:No\.?|Number|Ref\.?))[\s#:]+([A-Z0-9][\w\-]{3,20})",
    }

    def __init__(self):
        self.processed_documents: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process_document(
        self, file_path: str, doc_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a maritime document via Gemini Vision or Legacy extraction."""
        
        lower = file_path.lower()
        is_image = lower.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp"))

        # 1. ATTEMPT GEMINI VISION FIRST (For images)
        if is_image and config.llm_ready:
            logger.info("Image detected. Routing directly to Gemini Vision API...")
            vision_data = self._extract_with_gemini_vision(file_path)
            
            if vision_data and vision_data.get("document_type") != "unknown":
                result = {
                    "document_id": hashlib.md5(file_path.encode()).hexdigest()[:12],
                    "processed_at": datetime.now().isoformat(),
                    "file_path": file_path,
                    "document_type": vision_data.get("document_type", "bol"),
                    "extracted_data": vision_data,
                    "raw_text_preview": "Extracted natively via Gemini 1.5 Flash Vision API.",
                }
                self.processed_documents.append(result)
                return result
            else:
                logger.warning("Vision extraction failed or returned unknown. Falling back to legacy OCR.")

        # 2. FALLBACK TO LEGACY REGEX PIPELINE
        raw_text = self._read_file(file_path)
        if not doc_type:
            doc_type = self._detect_document_type(raw_text)

        extractor_map = {
            "bol": self._extract_bol_data,
            "manifest": self._extract_manifest_data,
            "dg_declaration": self._extract_dg_data,
        }
        extract_fn = extractor_map.get(doc_type, self._extract_generic_data)
        extracted_data = extract_fn(raw_text)

        result = {
            "document_id": hashlib.md5(file_path.encode()).hexdigest()[:12],
            "processed_at": datetime.now().isoformat(),
            "file_path": file_path,
            "document_type": doc_type,
            "extracted_data": extracted_data,
            "raw_text_preview": raw_text[:1000] if raw_text else None,
        }
        self.processed_documents.append(result)
        return result

    # ------------------------------------------------------------------
    # GEMINI VISION EXTRACTOR
    # ------------------------------------------------------------------

    def _extract_with_gemini_vision(self, file_path: str) -> Dict[str, Any]:
        """Bypasses Tesseract and uses Gemini Vision for native data mapping."""
        try:
            llm = ChatGoogleGenerativeAI(
                model=config.GEMINI_MODEL, 
                temperature=0.0 # Deterministic structured output
            )

            with open(file_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")

            prompt = """
            You are an expert maritime document extraction system.
            Analyze this image (likely a Bill of Lading, Manifest, or DG Declaration) and extract the data into a strict JSON format.
            Ignore background watermarks like 'ORIGINAL' or 'COPY'.
            Extract the data exactly as it appears. Ensure numbers like weight are parsed as floats.

            Return ONLY a valid JSON object matching this exact schema (use null if not found):
            {
              "document_type": "bol", 
              "bol_number": "",
              "booking_number": "",
              "issue_date": "",
              "freight_terms": "",
              "vessel": {"vessel_name": "", "voyage_number": "", "imo_number": ""},
              "ports": {"port_of_loading": "", "port_of_discharge": "", "place_of_receipt": "", "place_of_delivery": ""},
              "parties": {"shipper_name": "", "consignee_name": "", "notify_party": "", "carrier_name": ""},
              "containers": [
                 {"container_number": "", "container_type": "20'DC", "weight_kg": 0.0, "seal_number": "", "is_dangerous_goods": false}
              ],
              "extract_confidence": 0.98
            }
            """

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]
            )

            response = llm.invoke([message])
            clean_text = response.content.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)

            # Ensure extract_confidence exists for the UI
            if "extract_confidence" not in data:
                data["extract_confidence"] = 0.95

            return data

        except Exception as e:
            logger.error(f"Gemini Vision extraction failed: {e}")
            return {}

    # ------------------------------------------------------------------
    # File reading (Legacy)
    # ------------------------------------------------------------------

    def _read_file(self, file_path: str) -> str:
        lower = file_path.lower()
        if lower.endswith(".pdf"):
            return self._extract_from_pdf(file_path)
        if lower.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
            return self._extract_from_image(file_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                return fh.read()
        except OSError as exc:
            logger.error("Cannot read file %s: %s", file_path, exc)
            return ""

    def _extract_from_pdf(self, file_path: str) -> str:
        try:
            with pdfplumber.open(file_path) as pdf:
                pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
                return "\n".join(pages)
        except Exception as exc:
            logger.exception("PDF extraction error for %s: %s", file_path, exc)
            return f"[PDF extraction error: {exc}]"

    def _extract_from_image(self, file_path: str) -> str:
        try:
            img = Image.open(file_path).convert("L")
            img = ImageEnhance.Contrast(img).enhance(2.0)
            img = img.filter(ImageFilter.SHARPEN)
            return pytesseract.image_to_string(img, config="--psm 6 --oem 3")
        except Exception as exc:
            logger.exception("OCR extraction error for %s: %s", file_path, exc)
            return f"[Image OCR error: {exc}]"

    # ------------------------------------------------------------------
    # Document type detection (Legacy)
    # ------------------------------------------------------------------

    def _detect_document_type(self, text: str) -> str:
        t = text.lower()
        scores = {"bol": 0, "manifest": 0, "dg_declaration": 0, "port_call_report": 0}

        for kw in ("bill of lading", "b/l number", "bol number", "shipper", "consignee"):
            if kw in t: scores["bol"] += 2
        if "vessel" in t and "container" in t: scores["bol"] += 1

        for kw in ("cargo manifest", "container manifest", "manifest number"):
            if kw in t: scores["manifest"] += 3
        if "manifest" in t: scores["manifest"] += 1

        for kw in ("dangerous goods", "dg declaration", "imdg", "un number", "proper shipping name", "packing group"):
            if kw in t: scores["dg_declaration"] += 2

        for kw in ("port call", "arrival time", "departure time", "bunkering", "berth"):
            if kw in t: scores["port_call_report"] += 2

        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "unknown"

    # ------------------------------------------------------------------
    # Extractors (Legacy)
    # ------------------------------------------------------------------

    def _extract_bol_data(self, text: str) -> Dict[str, Any]:
        bol = BillOfLadingData(raw_text=text)

        for key, attr in [("bol_number", "bol_number"), ("booking_number", "booking_number")]:
            m = re.search(self.PATTERNS[key], text, re.IGNORECASE)
            if m: setattr(bol, attr, m.group(1).strip())

        for key, attr in [("vessel_name", "vessel_name"), ("voyage_number", "voyage_number"), ("imo_number", "imo_number")]:
            m = re.search(self.PATTERNS[key], text, re.IGNORECASE)
            if m: setattr(bol.vessel, attr, m.group(1).strip())

        m = re.search(self.PATTERNS["incoterm"], text)
        if m: bol.incoterm = m.group(1)

        dates = re.findall(self.PATTERNS["date"], text)
        if dates: bol.issue_date = dates[0]
        if len(dates) > 1: bol.onboard_date = dates[1]

        bol.containers = self._extract_containers(text)
        bol.ports = self._extract_ports(text)
        bol.parties = self._extract_parties(text)

        m = re.search(r"(Prepaid|Collect|As\s+Arranged)", text, re.IGNORECASE)
        if m: bol.freight_terms = m.group(1)

        bol.extract_confidence = self._calculate_bol_confidence(bol)
        return bol.to_dict()

    def _extract_manifest_data(self, text: str) -> Dict[str, Any]:
        containers = self._extract_containers(text)
        m_vessel = re.search(self.PATTERNS["vessel_name"], text, re.IGNORECASE)
        m_voyage = re.search(self.PATTERNS["voyage_number"], text, re.IGNORECASE)
        ports = self._extract_ports(text)

        return {
            "document_type": "Container Manifest",
            "vessel_name": m_vessel.group(1).strip() if m_vessel else None,
            "voyage_number": m_voyage.group(1).strip() if m_voyage else None,
            "total_containers": len(containers),
            "containers": [asdict(c) for c in containers],
            "ports": asdict(ports),
            "extract_confidence": round(0.6 + min(len(containers) * 0.05, 0.35), 2),
        }

    def _extract_dg_data(self, text: str) -> Dict[str, Any]:
        un_numbers = list(set(re.findall(self.PATTERNS["un_number"], text)))
        imdg_classes = list(set(re.findall(self.PATTERNS["imdg_class"], text, re.IGNORECASE)))
        containers = self._extract_containers(text)

        shipping_names: List[str] = []
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if re.search(r"proper\s+shipping\s+name|psn", line, re.IGNORECASE):
                for j in range(i + 1, min(i + 5, len(lines))):
                    candidate = lines[j].strip()
                    if candidate and len(candidate) > 3 and not re.match(r"^[\d\s\-:]+$", candidate):
                        shipping_names.append(candidate)
                        break

        packing_groups = list(set(re.findall(r"Packing\s+Group[\s:]+([I]{1,3})", text, re.IGNORECASE)))

        return {
            "document_type": "Dangerous Goods Declaration",
            "un_numbers": un_numbers,
            "imdg_classes": imdg_classes,
            "packing_groups": packing_groups,
            "containers": [asdict(c) for c in containers],
            "proper_shipping_names": shipping_names[:10],
            "total_dg_items": len(un_numbers),
            "extract_confidence": round(0.5 + min(len(un_numbers) * 0.1, 0.45), 2),
        }

    def _extract_generic_data(self, text: str) -> Dict[str, Any]:
        containers = self._extract_containers(text)
        return {
            "document_type": "Unknown",
            "containers": [asdict(c) for c in containers],
            "extracted_dates": list(set(re.findall(self.PATTERNS["date"], text)))[:10],
            "text_length": len(text),
            "extract_confidence": 0.4,
        }

    # ------------------------------------------------------------------
    # Sub-extractors (Legacy)
    # ------------------------------------------------------------------

    def _extract_containers(self, text: str) -> List[ExtractedContainer]:
        all_matches = list(set(re.findall(self.PATTERNS["container_number"], text)))
        known = [c for c in all_matches if c[:4] in _KNOWN_OWNER_CODES]
        candidates = known if known else all_matches

        containers: List[ExtractedContainer] = []
        for num in candidates:
            ctype = self._detect_container_type(text, num)
            context = self._get_context(text, num, window=300)
            is_dg = bool(re.search(r"\bUN\s*\d{4}|dangerous\s+goods|IMDG", context, re.IGNORECASE))
            un_match = re.search(self.PATTERNS["un_number"], context)
            imdg_match = re.search(self.PATTERNS["imdg_class"], context, re.IGNORECASE)
            seal_match = re.search(rf"{re.escape(num)}.*?Seal[\s#:]+([A-Z0-9\-]+)", text, re.DOTALL | re.IGNORECASE)
            weight_match = re.search(self.PATTERNS["weight"], context)

            containers.append(ExtractedContainer(
                container_number=num,
                container_type=ctype,
                seal_number=seal_match.group(1).strip() if seal_match else None,
                weight_kg=float(weight_match.group(1).replace(",", "")) if weight_match else None,
                is_dangerous_goods=is_dg,
                un_number=un_match.group(1) if un_match else None,
                imdg_class=imdg_match.group(1) if imdg_match else None,
            ))
        return containers

    def _detect_container_type(self, text: str, container_num: str) -> str:
        context = self._get_context(text, container_num, window=200).lower()
        if any(t in context for t in ("reefer", "refrigerated", "20'rf", "40'rf", "rf")): return "40'RF"
        if any(t in context for t in ("tank", "isotank", "20'tk")): return "20'TK"
        if any(t in context for t in ("open top", "40'ot", "ot")): return "40'OT"
        if any(t in context for t in ("flat rack", "40'fr", "fr")): return "40'FR"
        if "40" in context and "hc" in context: return "40'HC"
        if "40" in context: return "40'DC"
        if "20" in context: return "20'DC"
        return "20'DC"

    def _get_context(self, text: str, token: str, window: int = 200) -> str:
        pos = text.find(token)
        if pos < 0: return ""
        return text[max(0, pos - window): pos + len(token) + window]

    def _extract_ports(self, text: str) -> ExtractedPorts:
        ports = ExtractedPorts()
        lines = text.split("\n")

        def _find_after_label(keywords: frozenset) -> Optional[str]:
            for i, line in enumerate(lines):
                if any(kw in line.lower() for kw in keywords):
                    if ":" in line:
                        val = line.split(":", 1)[1].strip()
                        if val: return val
                    if i + 1 < len(lines):
                        nxt = lines[i + 1].strip()
                        if nxt: return nxt
            return None

        ports.port_of_loading = _find_after_label(_POL_KEYWORDS)
        ports.port_of_discharge = _find_after_label(_POD_KEYWORDS)
        ports.place_of_receipt = _find_after_label(_RECEIPT_KEYWORDS)
        ports.place_of_delivery = _find_after_label(_DELIVERY_KEYWORDS)

        codes = re.findall(self.PATTERNS["port_code"], text)
        for code in codes:
            if code in PORT_MAPPING:
                name = PORT_MAPPING[code]
                if not ports.port_of_loading: ports.port_of_loading = name
                elif not ports.port_of_discharge and name != ports.port_of_loading: ports.port_of_discharge = name
                elif name not in (ports.port_of_loading, ports.port_of_discharge):
                    if name not in ports.transshipment_ports: ports.transshipment_ports.append(name)
        return ports

    def _extract_parties(self, text: str) -> ExtractedParties:
        parties = ExtractedParties()
        lines = text.split("\n")
        _label_map = {"shipper": "shipper", "consignee": "consignee", "notify": "notify_party", "carrier": "carrier"}

        i = 0
        while i < len(lines):
            ll = lines[i].lower().strip()
            for keyword, field_name in _label_map.items():
                if keyword in ll:
                    value = lines[i].split(":", 1)[1].strip() if ":" in lines[i] else ""
                    addr_lines = [value] if value else []
                    j = i + 1
                    while j < len(lines):
                        nxt = lines[j].strip()
                        if not nxt or any(kw in nxt.lower() for kw in _label_map): break
                        addr_lines.append(nxt)
                        j += 1

                    full = " ".join(addr_lines).strip()
                    if full:
                        if field_name == "shipper":
                            parties.shipper_name = addr_lines[0] if addr_lines else full
                            if len(addr_lines) > 1: parties.shipper_address = " ".join(addr_lines[1:])
                        elif field_name == "consignee":
                            parties.consignee_name = addr_lines[0] if addr_lines else full
                            if len(addr_lines) > 1: parties.consignee_address = " ".join(addr_lines[1:])
                        elif field_name == "notify_party": parties.notify_party = full
                        elif field_name == "carrier": parties.carrier_name = full
            i += 1
        return parties

    def _calculate_bol_confidence(self, bol: BillOfLadingData) -> float:
        checks = [
            (bol.bol_number is not None, 0.20),
            (bol.vessel.vessel_name is not None, 0.15),
            (bool(bol.containers), 0.25),
            (bol.ports.port_of_loading is not None, 0.15),
            (bol.ports.port_of_discharge is not None, 0.15),
            (bol.parties.shipper_name is not None, 0.05),
            (bol.parties.consignee_name is not None, 0.05),
        ]
        return round(sum(w for ok, w in checks if ok), 2)

    # ------------------------------------------------------------------
    # Container number validation (ISO 6346)
    # ------------------------------------------------------------------

    def validate_container_number(self, container_num: str) -> Tuple[bool, str]:
        cn = container_num.strip().upper()
        if len(cn) != 11: return False, f"Container number must be 11 characters (got {len(cn)})"
        
        owner_code, category, serial, check_char = cn[:3], cn[3], cn[4:10], cn[10]
        if not owner_code.isalpha(): return False, "Owner code (first 3 chars) must be letters"
        if category not in ("U", "J", "Z"): return False, f"Category identifier must be U, J, or Z (got '{category}')"
        if not serial.isdigit(): return False, "Serial number (chars 5–10) must be 6 digits"
        if not check_char.isdigit(): return False, "Check digit (char 11) must be a digit"

        def _letter_value(ch: str) -> int:
            base = ord(ch) - ord("A") + 10
            return base + base // 10

        total = sum((_letter_value(ch) if ch.isalpha() else int(ch)) * (2 ** pos) for pos, ch in enumerate(cn[:10]))
        remainder = total % 11
        expected = 0 if remainder == 10 else remainder

        if str(expected) == check_char: return True, "Valid container number"
        return False, f"Invalid check digit — expected {expected}, got {check_char}"

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_processed_count(self) -> int:
        return len(self.processed_documents)


# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_processor: Optional[MaritimeDocumentProcessor] = None
_processor_lock = threading.Lock()


def get_document_processor() -> MaritimeDocumentProcessor:
    """Return the shared document processor instance."""
    global _processor
    if _processor is None:
        with _processor_lock:
            if _processor is None:
                _processor = MaritimeDocumentProcessor()
    return _processor