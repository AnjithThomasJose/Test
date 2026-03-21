# -- coding: utf-8 --

"""

JobsifyAI - Hardened Resume Text Extractor (Merged Version)

Combines:
- New modular structure with quality triage and PII extraction
- Existing robust features: OCR, Canva detection, repair, encryption handling

Robust routing for PDF / DOCX / DOC

Text quality triage + normalization

Alternate extractors (PyPDF2, pdfminer column-aware)

Deterministic PII (email/phone) + name heuristic to avoid LLM misses

Minimal logging with quality metrics

Public API (unchanged):

    download_resume_text(file_url: str, file_type: str = "resume") -> str

"""

from __future__ import annotations

import io, os, re, tempfile, subprocess, platform, logging, asyncio, json
from typing import Optional, Tuple, Dict, Any
from urllib.parse import unquote
import requests
from concurrent.futures import ThreadPoolExecutor

# LangSmith tracing
try:
    from langsmith.run_helpers import traceable
    _HAS_LANGSMITH = True
except Exception:
    _HAS_LANGSMITH = False
    # Create a no-op decorator if langsmith is not available
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Try to import httpx for async HTTP requests
try:
    import httpx
    _HAS_HTTPX = True
except Exception:
    _HAS_HTTPX = False

# Try to import http_client for shared async client
try:
    from core.http_client import get_http_client
    _HAS_HTTP_CLIENT = True
except Exception:
    _HAS_HTTP_CLIENT = False

# ---- Optional deps (best-effort imports) -------------------------------------

try:
    import magic  # python-magic for mime via magic bytes
except Exception:
    magic = None  # fallback to extension/headers

# Initialize logger early for use in import error handling
log = logging.getLogger('resume_utils')

# PyMuPDF removed - using PyPDF2 and pdfminer instead

# pdfminer for column-aware fallback
try:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer, LTTextBoxHorizontal, LTTextLineHorizontal
    _HAS_PDFMINER = True
except Exception:
    _HAS_PDFMINER = False

# pymupdf4llm removed - not using PyMuPDF
_HAS_PYMUPDF4LLM = False

# DOCX paths
try:
    import mammoth
    _HAS_MAMMOTH = True
except Exception:
    _HAS_MAMMOTH = False

try:
    import docx  # python-docx
    _HAS_PYDOCX = True
except Exception:
    _HAS_PYDOCX = False

# PyPDF2 fallback
try:
    from PyPDF2 import PdfReader
    _HAS_PYPDF2 = True
except Exception:
    _HAS_PYPDF2 = False

# DOC fallback
try:
    import docx2txt
    import olefile
    _HAS_DOC_FALLBACK = True
except Exception:
    _HAS_DOC_FALLBACK = False

# Gemini for text extraction (using new google-genai SDK)
try:
    from google import genai
    _HAS_GEMINI = True
    log.info("✅ Gemini extraction: google.genai is available and enabled")
except Exception as e:
    _HAS_GEMINI = False
    genai = None
    log.warning(f"❌ Gemini extraction: google.genai not available ({e}), will use fallback methods")

# ------------------------------------------------------------------------------

# Configure logger if no handlers
if not log.handlers:
    logging.basicConfig(level=logging.INFO)

# Thread pool executor for subprocess operations (non-blocking)
_subprocess_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="resume_subprocess")

# ==========================
# Configuration
# ==========================

TEXT_MIN_CHARS = int(os.getenv("PDF_TEXT_MIN_CHARS", "800"))
ALPHA_RATIO_MIN = float(os.getenv("PDF_ALPHA_RATIO_MIN", "0.35"))

# ==========================
# Normalization & Quality
# ==========================

def _normalize_text(s: str) -> str:
    """Normalize text: clean encoding, ligatures, whitespace, de-hyphenate."""
    if not s:
        return ""
    s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
    rep = {
        "\u00a0": " ", "\u200b": "", "\ufeff": "",
        "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2026": "...", "ﬁ": "fi", "ﬂ": "fl",
    }
    for a, b in rep.items():
        s = s.replace(a, b)
    # de-hyphenate at EOL
    s = re.sub(r"(\w)-\s*\n(\w)", r"\1\2", s)
    # remove page labels
    s = re.sub(r"\sPage\s+\d+\s(of\s+\d+)?\s*", "\n", s, flags=re.I)
    # collapse whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _quality_diag(text: str) -> dict:
    """Diagnose text quality: length, word count, alpha ratio, uniqueness."""
    t = (text or "").strip()
    alpha = sum(ch.isalpha() for ch in t)
    words = len(re.findall(r"\b\w+\b", t))
    unique_ratio = len(set(t)) / max(1, len(t))
    alpha_ratio = alpha / max(1, len(t))
    weak = (len(t) < TEXT_MIN_CHARS) or (alpha_ratio < ALPHA_RATIO_MIN)
    return {
        "len": len(t),
        "words": words,
        "alpha_ratio": round(alpha_ratio, 3),
        "unique_ratio": round(unique_ratio, 3),
        "weak": weak
    }

def is_text_weak(s: str) -> bool:
    """Check if extracted text is too weak (short or low alpha ratio)."""
    t = (s or "").strip()
    if len(t) < TEXT_MIN_CHARS:
        return True
    alpha = sum(ch.isalpha() for ch in t)
    return (alpha / max(1, len(t))) < ALPHA_RATIO_MIN

# ==========================
# PII Baseline (deterministic)
# ==========================

# Enhanced email pattern - handles emails split across lines
EMAIL_RE = re.compile(r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}', re.I)

# Enhanced phone patterns - handles various formats including Indian 10-digit numbers
PHONE_PATTERNS = [
    r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # Standard formats: +1-234-567-8900, (234) 567-8900
    r'\b\d{10}\b',  # 10-digit numbers (Indian format: 8294442121)
    r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # US format: 234-567-8900
    r'\+?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}',  # International formats
    r'\(\d{3}\)\s?\d{3}[-.]?\d{4}',  # (123) 456-7890
]

def extract_pii_baseline(text: str) -> dict:
    """
    Deterministic extraction so emails/phones never come back empty.
    Enhanced to handle various phone formats including Indian 10-digit numbers.
    Filters out postal codes, years, and other non-phone numbers.
    """
    # Extract emails
    emails = list(dict.fromkeys(EMAIL_RE.findall(text)))
    
    # Extract phones using multiple patterns
    phones = []
    for pattern in PHONE_PATTERNS:
        matches = re.findall(pattern, text)
        phones.extend(matches)
    
    # Clean and deduplicate phones, filtering out false positives
    cleaned_phones = []
    seen_cleaned = set()
    
    for phone in phones:
        # Remove common separators for comparison
        cleaned = re.sub(r'[-.\s()]', '', phone)
        
        # Filter out obvious non-phone numbers:
        # - Postal codes (usually 5-6 digits, often near "Address", "City", etc.)
        # - Years (4 digits: 1990-2024)
        # - Very short numbers (< 7 digits)
        # - Very long numbers without country code (> 15 digits)
        
        if len(cleaned) < 7:
            continue  # Too short to be a phone number
        
        if len(cleaned) == 4 and cleaned.isdigit():
            year = int(cleaned)
            if 1900 <= year <= 2030:  # Likely a year
                continue
        
        if len(cleaned) > 15:
            continue  # Too long
        
        # Check if it's near postal code indicators
        phone_lower = phone.lower()
        context_words = ['address', 'zip', 'postal', 'pincode', 'pin code', 'city', 'state']
        if any(word in text.lower() for word in context_words):
            # Check if this number appears near postal code context
            phone_pos = text.lower().find(phone_lower)
            if phone_pos != -1:
                context = text[max(0, phone_pos-50):phone_pos+len(phone)+50].lower()
                if any(word in context for word in ['address', 'zip', 'postal', 'pincode', 'pin code']):
                    # Likely a postal code, skip if it's 5-6 digits
                    if 5 <= len(cleaned) <= 6:
                        continue
        
        # Deduplicate
        if cleaned not in seen_cleaned:
            seen_cleaned.add(cleaned)
            cleaned_phones.append(phone)
    
    # Format phones nicely
    final_phones = []
    for phone in cleaned_phones:
        cleaned = re.sub(r'[-.\s()]', '', phone)
        if len(cleaned) == 10 and cleaned.isdigit():
            # Format Indian phone number: XXX-XXX-XXXX
            final_phones.append(f"{cleaned[:3]}-{cleaned[3:6]}-{cleaned[6:]}")
        elif len(cleaned) >= 7:  # Valid phone number length
            final_phones.append(phone)
    
    return {"emails": emails, "phones": list(dict.fromkeys(final_phones))}

def guess_name(text: str) -> str:
    """
    Heuristic: scan first ~20 lines for a realistic human name.
    Enhanced to handle names that appear on the same line as labels.
    """
    head = "\n".join(text.splitlines()[:20])
    candidates = []
    
    for ln in head.splitlines():
        # Skip lines that are clearly not names (contain common labels)
        if any(label in ln.lower() for label in ['phone', 'email', 'address', 'linkedin', 'github', 'website', 'e-mail']):
            continue
        
        # Extract words that look like name components
        w = [w for w in re.findall(r"[A-Za-z][A-Za-z\-']+", ln)]
        
        # Filter out common non-name words
        skip_words = {'talent', 'acquisition', 'specialist', 'address', 'bengaluru', 'linkedin', 
                     'phone', 'email', 'e-mail', 'github', 'website', 'location', 'city', 'state'}
        w = [word for word in w if word.lower() not in skip_words]
        
        if 2 <= len(w) <= 4:  # Typical name: 2-4 words
            cap = sum(1 for x in w if x[0].isupper())
            # Require at least 2 capitalized words (first and last name)
            if cap >= 2 and not ln.isupper() and len(" ".join(w)) <= 40:
                # Additional check: names usually don't contain numbers
                if not any(char.isdigit() for char in " ".join(w)):
                    candidates.append(" ".join(w))
    
    # Return the first candidate (usually the most prominent name at the top)
    return candidates[0].strip() if candidates else ""

# ==========================
# Text Clamping (Performance)
# ==========================

def _maybe_clamp_text(text: str, file_type: str) -> str:
    """Clamp large extracted text early to reduce downstream processing cost.
    
    For resumes, use a much higher limit to ensure all sections are captured.
    """
    try:
        from core.config import get_agent_config, AgentConfig
        if (file_type or "").lower() == "jd":
            max_chars = getattr(get_agent_config("job_description_parser"), "max_prompt_chars", 30000)
        else:
            # For resumes, use much higher limit (100k) to capture all sections
            # The parser will handle trimming if needed, but we want full extraction here
            max_chars = getattr(get_agent_config("groq_resume_parser"), "max_prompt_chars", 100000)
            # If config has a lower value, still use at least 50k for resumes
            if max_chars < 50000:
                max_chars = 100000
        if not isinstance(max_chars, int) or max_chars <= 0:
            max_chars = 100000 if (file_type or "").lower() == "resume" else 30000
    except Exception:
        max_chars = 100000 if (file_type or "").lower() == "resume" else 30000
    
    if not text or len(text) <= max_chars:
        return text or ""
    
    # Only truncate if text is extremely long (over limit)
    # Log warning so we know if content is being cut
    log.warning(f"⚠️ Text length ({len(text)} chars) exceeds limit ({max_chars} chars) for {file_type}")
    
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    last_newline = truncated.rfind('\n')
    cutoff = max(last_period, last_newline)
    if cutoff > int(max_chars * 0.8):
        return truncated[:cutoff + 1]
    return truncated

# ==========================
# PDF Repair & Utilities
# ==========================

def _repair_pdf_bytes_sync(pdf_bytes: bytes) -> bytes:
    """Internal sync function for PDF repair - runs subprocess."""
    try:
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "in.pdf")
            dst = os.path.join(td, "out.pdf")
            with open(src, "wb") as f:
                f.write(pdf_bytes)
            subprocess.run(
                ["mutool", "clean", "-gg", src, dst],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30
            )
            with open(dst, "rb") as f:
                return f.read()
    except Exception:
        return pdf_bytes

def repair_pdf_bytes(pdf_bytes: bytes) -> bytes:
    """Try mutool clean to repair broken PDFs (xref/XFA/permission issues).
    
    This function runs subprocess operations in a thread pool to avoid blocking.
    If called from async context, returns a coroutine. Otherwise, runs synchronously.
    """
    try:
        # Check if we're in an async context
        loop = asyncio.get_running_loop()
        # We're in async context, return coroutine
        return loop.run_in_executor(_subprocess_executor, _repair_pdf_bytes_sync, pdf_bytes)
    except RuntimeError:
        # No event loop running, execute directly
        return _repair_pdf_bytes_sync(pdf_bytes)

def _pdf_page_count(pdf_bytes: bytes) -> int:
    """Get PDF page count quickly using PyPDF2."""
    try:
        if _HAS_PYPDF2:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            return len(reader.pages)
    except Exception:
        pass
    return 0

def _truncate_pdf_file(src_path: str, page_limit: int) -> str:
    """Write a temporary PDF containing only the first `page_limit` pages."""
    # PyPDF2 doesn't support PDF writing, so truncation is not available without PyMuPDF
    # Return original path
    if page_limit <= 0:
        return src_path
    log.debug("⚠️ PDF truncation not available (PyMuPDF removed)")
    return src_path

# ==========================
# OCR Hook (pluggable)
# ==========================

def _ocr_pdf_to_markdown_sync(pdf_bytes: bytes) -> str:
    """Internal sync function for OCR - runs subprocess."""
    # OCR requires ocrmypdf and pymupdf4llm - simplified to return empty if not available
    # PyMuPDF removed, so OCR is not available
    log.warning("⚠️ OCR requested but pymupdf4llm not available (PyMuPDF removed)")
    return ""

def _ocr_pdf_to_markdown(pdf_bytes: bytes) -> str:
    """OCR PDF - currently disabled (PyMuPDF removed).
    
    This function runs subprocess operations in a thread pool to avoid blocking.
    If called from async context, returns a coroutine. Otherwise, runs synchronously.
    """
    try:
        # Check if we're in an async context
        loop = asyncio.get_running_loop()
        # We're in async context, return coroutine
        return loop.run_in_executor(_subprocess_executor, _ocr_pdf_to_markdown_sync, pdf_bytes)
    except RuntimeError:
        # No event loop running, execute directly
        return _ocr_pdf_to_markdown_sync(pdf_bytes)

# ==========================
# PDF Extraction Pipeline
# ==========================

def _pdf_extract_with_pypdf2(pdf_bytes: bytes) -> str:
    """Extract text using PyPDF2 - can be better for certain PDF types."""
    if not _HAS_PYPDF2:
        return ""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages_text.append(page_text)
            except Exception as e:
                log.debug(f"PyPDF2 page extraction failed: {e}")
                continue
        
        if pages_text:
            text = "\n\n".join(pages_text).strip()
            log.info(f"✅ PyPDF2 extracted {len(text)} characters from {len(pages_text)} pages")
            return _normalize_text(text)
        return ""
    except Exception as e:
        log.debug(f"PyPDF2 extraction failed: {e}")
        return ""

def _pdf_extract_column_aware(pdf_bytes: bytes) -> str:
    """pdfminer-based two-column heuristic."""
    if not _HAS_PDFMINER:
        return ""
    pages_text: list[str] = []
    try:
        for page_layout in extract_pages(io.BytesIO(pdf_bytes)):
            boxes = [e for e in page_layout if isinstance(e, (LTTextContainer, LTTextBoxHorizontal))]
            if not boxes:
                lines = []
                for e in page_layout:
                    if isinstance(e, LTTextLineHorizontal):
                        lines.append(e.get_text())
                pages_text.append("".join(lines))
                continue

            centers = []
            left_elems, right_elems = [], []
            for b in boxes:
                x0, y0, x1, y1 = b.bbox
                centers.append((x0 + x1) / 2.0)

            if not centers:
                pages_text.append("\n".join(b.get_text() for b in boxes))
                continue

            median_x = sorted(centers)[len(centers) // 2]
            for b in boxes:
                x0, y0, x1, y1 = b.bbox
                cx = (x0 + x1) / 2.0
                (left_elems if cx <= median_x else right_elems).append(b)

            def sort_join(elems):
                es = sorted(elems, key=lambda e: (-e.bbox[1], e.bbox[0]))
                return "".join(e.get_text() for e in es)

            stitched = sort_join(left_elems).rstrip() + "\n\n" + sort_join(right_elems).lstrip()
            pages_text.append(stitched)

        return _normalize_text("\n\n".join(pages_text))
    except Exception as e:
        log.debug(f"pdfminer column-aware failed: {e}")
        return ""

def _pdf_should_ocr(pdf_bytes: bytes, meta: dict) -> bool:
    """Determine if PDF needs OCR based on content analysis."""
    # Simple heuristic: use OCR if text extraction yielded very little text
    # or if image ratio is high
    try:
        text_length = meta.get("words_total", 0)
        pages = meta.get("pages", 1)
        words_per_page = text_length / max(pages, 1)
        
        # Canva-style PDFs: high image ratio or very few words
        is_canva_style = meta.get("is_canva") or meta.get("image_to_text_ratio", 0) > 0.5
        if is_canva_style:
            return (text_length == 0 or words_per_page < 50 or meta.get("image_to_text_ratio", 0) > 0.6)
        
        # General case: very few words per page
        return words_per_page < 20
    except Exception:
        return False

def _pdf_extract_best(pdf_bytes: bytes, ocr_func=_ocr_pdf_to_markdown) -> Tuple[str, dict]:
    """Extract PDF text using PyPDF2 and pdfminer (no PyMuPDF)."""
    diag = {"used": [], "words_total": 0, "images_total": 0, "pages": 0, 
            "encrypted": False, "repaired": False, "is_canva": False, 
            "image_to_text_ratio": 0.0}
    text = ""

    # Try PyPDF2 first
    if _HAS_PYPDF2:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            diag["pages"] = len(reader.pages)
            
            # Check for encryption
            if reader.is_encrypted:
                diag["encrypted"] = True
                try:
                    reader.decrypt("")  # Try empty password
                except Exception:
                    return "", {**diag, "error": "PDF is encrypted and cannot be opened"}
            
            # Extract text from all pages
            page_texts = []
            for page in reader.pages:
                try:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        page_texts.append(page_text)
                        # Count words
                        words = page_text.split()
                        diag["words_total"] += len(words)
                except Exception as e:
                    log.debug(f"PyPDF2 page extraction failed: {e}")
                    continue
            
            if page_texts:
                text = "\n\n".join(page_texts).strip()
                diag["used"].append("pypdf2:primary")
        except Exception as e:
            log.debug(f"PyPDF2 extraction failed: {e}")

    # If PyPDF2 failed or not available, try pdfminer column-aware
    if not text or not text.strip():
        if _HAS_PDFMINER:
            alt = _pdf_extract_column_aware(pdf_bytes)
            if alt and len(alt) > len(text):
                text = alt
                diag["used"].append("pdfminer:column-aware")
                # Update page count if we got text
                if not diag["pages"]:
                    try:
                        from PyPDF2 import PdfReader
                        reader = PdfReader(io.BytesIO(pdf_bytes))
                        diag["pages"] = len(reader.pages)
                    except Exception:
                        pass

    # If still weak or empty, try OCR (if available)
    q = _quality_diag(text) if text else {"weak": True}
    if (not text) or q.get("weak", False):
        ocr_text = ""
        if callable(ocr_func):
            try:
                ocr_result = ocr_func(pdf_bytes)
                # Handle async result if needed
                if asyncio.iscoroutine(ocr_result):
                    ocr_text = _ocr_pdf_to_markdown_sync(pdf_bytes) or ""
                else:
                    ocr_text = (ocr_result or "").strip()
            except Exception as e:
                log.debug(f"OCR extraction failed: {e}")
                ocr_text = ""
        
        if ocr_text:
            if text:
                # Merge OCR with existing text
                text_lower = text.strip().lower()
                ocr_lower = ocr_text.strip().lower()
                if ocr_lower not in text_lower:
                    text = f"{text.strip()}\n\n{ocr_text.strip()}"
                    diag["used"].append("ocr:merged")
                    log.info(f"✅ Merged extracted text ({len(text)} chars) + OCR ({len(ocr_text)} chars)")
                else:
                    diag["used"].append("ocr:verified")
            else:
                # Use OCR as primary if no text extracted
                text = ocr_text.strip()
                diag["used"].append("ocr:primary")
                log.info(f"✅ Using OCR only ({len(ocr_text)} chars)")

    text = _normalize_text(text)
    diag.update(_quality_diag(text))
    return text, diag

# ==========================
# DOCX Extraction
# ==========================

def _extract_docx_textboxes_raw(docx_bytes: bytes) -> list:
    """
    Extract text from DOCX text boxes using raw XML parsing.
    This catches text in shapes/drawings that mammoth might miss.
    """
    parts = []
    try:
        import zipfile
        from xml.etree import ElementTree as ET
        
        with zipfile.ZipFile(io.BytesIO(docx_bytes)) as z:
            # Parse document.xml for text boxes
            if 'word/document.xml' in z.namelist():
                xml_content = z.read('word/document.xml')
                root = ET.fromstring(xml_content)
                
                # Define namespaces used in DOCX
                ns = {
                    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
                    'wps': 'http://schemas.microsoft.com/office/word/2010/wordprocessingShape',
                    'wpg': 'http://schemas.microsoft.com/office/word/2010/wordprocessingGroup',
                    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
                    'mc': 'http://schemas.openxmlformats.org/markup-compatibility/2006',
                }
                
                # Find all text elements within drawing shapes
                for t_elem in root.iter():
                    if t_elem.tag.endswith('}t') and t_elem.text and t_elem.text.strip():
                        # Check if this is inside a drawing/shape element
                        tag_local = t_elem.tag.split('}')[-1]
                        if tag_local == 't':
                            parts.append(t_elem.text.strip())
            
            # Also check header files
            for fname in z.namelist():
                if fname.startswith('word/header') and fname.endswith('.xml'):
                    xml_content = z.read(fname)
                    root = ET.fromstring(xml_content)
                    for t_elem in root.iter():
                        if t_elem.tag.endswith('}t') and t_elem.text and t_elem.text.strip():
                            parts.append(t_elem.text.strip())
                            
    except Exception as e:
        log.debug(f"Raw textbox extraction failed: {e}")
    
    return parts


def _docx_with_mammoth_to_text(docx_bytes: bytes) -> str:
    """Extract DOCX using Mammoth (preserves formatting better) with textbox fallback."""
    if not _HAS_MAMMOTH:
        return ""
    try:
        html = mammoth.convert_to_html(io.BytesIO(docx_bytes)).value
        # Lightweight HTML -> text
        html = re.sub(r"(?is)</p\s*>", "\n\n", html)
        html = re.sub(r"(?is)<li[^>]*>", "\n• ", html)
        text = re.sub(r"(?is)<[^>]+>", "", html)
        mammoth_text = _normalize_text(text)
        
        # Also extract text boxes that mammoth might miss
        textbox_parts = _extract_docx_textboxes_raw(docx_bytes)
        if textbox_parts:
            # Prepend textbox content (often contains name/contact info at top)
            textbox_text = " ".join(textbox_parts)
            # Only add if it contains new content
            if textbox_text and textbox_text not in mammoth_text:
                # Check if any significant part is missing
                for part in textbox_parts:
                    if part and len(part) > 3 and part not in mammoth_text:
                        mammoth_text = part + "\n" + mammoth_text
        
        return mammoth_text
    except Exception as e:
        log.debug(f"Mammoth failed: {e}")
        return ""

def _extract_text_from_docx_element(element) -> list:
    """Extract text from a docx element (header, footer, or body)."""
    parts = []
    if hasattr(element, 'paragraphs'):
        parts.extend(p.text for p in element.paragraphs if p.text and p.text.strip())
    if hasattr(element, 'tables'):
        for table in element.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
    return parts


def _extract_text_from_docx_shapes(document) -> list:
    """
    Extract text from text boxes and shapes in DOCX.
    Text boxes are stored in the document's XML as drawing elements.
    """
    parts = []
    try:
        from docx.oxml.ns import qn
        
        # Navigate the XML to find text in drawing elements (text boxes)
        for element in document.element.iter():
            # Look for text in drawing/textbox elements
            if element.tag.endswith('}t'):  # Text elements
                parent_tags = []
                parent = element.getparent()
                while parent is not None:
                    parent_tags.append(parent.tag)
                    parent = parent.getparent()
                
                # Check if this text is inside a drawing/shape (text box)
                is_in_textbox = any(
                    'drawing' in tag or 'txbxContent' in tag or 'wsp' in tag
                    for tag in parent_tags
                )
                
                if is_in_textbox and element.text and element.text.strip():
                    parts.append(element.text.strip())
    except Exception as e:
        log.debug(f"Text box extraction failed: {e}")
    
    return parts


def _docx_with_python_docx(docx_bytes: bytes) -> str:
    """Extract DOCX using python-docx with enhanced header/footer/textbox support."""
    if not _HAS_PYDOCX:
        return ""
    try:
        buf = io.BytesIO(docx_bytes)
        document = docx.Document(buf)
        parts = []
        
        # Extract from headers first (names often appear in headers)
        for section in document.sections:
            if section.header:
                header_parts = _extract_text_from_docx_element(section.header)
                if header_parts:
                    parts.extend(header_parts)
            # Also check for different header types
            if hasattr(section, 'first_page_header') and section.first_page_header:
                header_parts = _extract_text_from_docx_element(section.first_page_header)
                if header_parts:
                    parts.extend(header_parts)
        
        # Extract text from text boxes/shapes
        textbox_parts = _extract_text_from_docx_shapes(document)
        if textbox_parts:
            parts.extend(textbox_parts)
        
        # Extract from main body paragraphs
        parts.extend(p.text for p in document.paragraphs if p.text and p.text.strip())
        
        # Extract from tables
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        
        # Extract from footers (may contain contact info)
        for section in document.sections:
            if section.footer:
                footer_parts = _extract_text_from_docx_element(section.footer)
                if footer_parts:
                    parts.extend(footer_parts)
        
        # Deduplicate while preserving order
        seen = set()
        unique_parts = []
        for part in parts:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)
        
        return _normalize_text("\n".join(unique_parts))
    except Exception as e:
        log.debug(f"python-docx failed: {e}")
        return ""

def _docx_to_pdf_with_libreoffice_sync(docx_bytes: bytes) -> Optional[bytes]:
    """Internal sync function for DOCX to PDF conversion."""
    if not (_have_sync("libreoffice") or platform.system() == "Windows"):
        return None
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.docx")
        out = os.path.join(td, "in.pdf")
        with open(src, "wb") as f:
            f.write(docx_bytes)
        soffice = (r"C:\Program Files\LibreOffice\program\soffice.exe"
                   if platform.system() == "Windows" else "libreoffice")
        try:
            subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", td, src],
                           check=True, capture_output=True, timeout=60)
            if os.path.exists(out):
                return open(out, "rb").read()
        except Exception as e:
            log.debug(f"LibreOffice docx->pdf failed: {e}")
    return None

def _docx_to_pdf_sync(docx_bytes: bytes) -> Optional[bytes]:
    """
    Convert DOCX to PDF using LibreOffice only.
    Returns PDF bytes or None if conversion fails.
    """
    pdf_bytes = _docx_to_pdf_with_libreoffice_sync(docx_bytes)
    if not pdf_bytes:
        log.error("❌ LibreOffice DOCX conversion failed")
    return pdf_bytes

def _docx_to_pdf_with_libreoffice(docx_bytes: bytes) -> Optional[bytes]:
    """Convert DOCX to PDF using LibreOffice - runs in thread pool if in async context."""
    try:
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(_subprocess_executor, _docx_to_pdf_with_libreoffice_sync, docx_bytes)
    except RuntimeError:
        return _docx_to_pdf_with_libreoffice_sync(docx_bytes)

# ==========================
# Public Helper Functions (for backward compatibility)
# ==========================

# PyMuPDF DOCX extraction functions removed - using Mammoth/python-docx instead

@traceable(
    name="gemini_text_extraction",
    run_type="llm",
    metadata={"extraction_method": "gemini", "output_format": "markdown"}
)
def _extract_text_with_gemini(file_bytes: bytes, file_type: str = "pdf") -> str:
    """
    Extract text from PDF or DOCX files using Gemini and convert to markdown.
    
    Args:
        file_bytes: Binary content of the file
        file_type: "pdf" or "docx"
    
    Returns:
        Extracted text in markdown format
    """
    if not _HAS_GEMINI:
        log.warning("❌ Gemini extraction: google.genai not available, using fallback methods")
        return ""
    
    log.debug(f"🤖 Gemini extraction: Attempting to extract text from {file_type.upper()} using Google Gemini API")
    
    try:
        # Import settings here to avoid circular imports
        from settings import settings
        
        # Initialize Gemini Client (new google-genai SDK API)
        client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        
        # Create a temporary file
        suffix = ".pdf" if file_type.lower() == "pdf" else ".docx"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        try:
            # Upload file to Gemini (new API - use 'file' parameter, not 'path')
            uploaded_file = client.files.upload(file=tmp_path)
            
            # OPTIMIZATION: Wait for file processing with aggressive polling for faster response
            # ✅ PERFORMANCE: Use asyncio.sleep() if in async context, otherwise time.sleep()
            import time
            max_wait_time = 15  # Reduced from 20 to 15 seconds
            start_wait = time.time()
            wait_interval = 0.2  # Start with 0.2s (faster polling)
            max_wait_interval = 1.0  # Reduced from 2.0s to 1.0s max
            
            # Check if we're in an async context (function may be called from async code)
            try:
                loop = asyncio.get_running_loop()
                use_async_sleep = True
            except RuntimeError:
                use_async_sleep = False
            
            while uploaded_file.state.name == "PROCESSING":
                elapsed = time.time() - start_wait
                if elapsed > max_wait_time:
                    log.error(f"File processing timeout after {max_wait_time}s")
                    try:
                        client.files.delete(name=uploaded_file.name)
                    except:
                        pass
                    return ""
                
                # ✅ PERFORMANCE: Use asyncio.sleep() in async context to avoid blocking thread pool
                sleep_duration = min(wait_interval, max_wait_interval)
                if use_async_sleep:
                    # This is a sync function, but if called from async, we can't use asyncio.sleep here
                    # Keep time.sleep but note it's in thread pool (acceptable)
                    time.sleep(sleep_duration)
                else:
                    time.sleep(sleep_duration)
                wait_interval = min(wait_interval * 1.3, max_wait_interval)  # Less aggressive backoff (1.3x vs 1.5x)
                
                uploaded_file = client.files.get(name=uploaded_file.name)
            
            if uploaded_file.state.name == "FAILED":
                log.error(f"File upload failed: {uploaded_file.state}")
                return ""
            
            # OPTIMIZATION: Shorter, more direct prompt for faster processing
            prompt = "Extract all text to markdown. Preserve structure and formatting."
            
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=[prompt, uploaded_file]
            )
            
            # Get the extracted text
            extracted_text = response.text if hasattr(response, 'text') and response.text else ""
            
            # Clean up uploaded file
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception as e:
                log.debug(f"Failed to delete uploaded file: {e}")
            
            result = extracted_text.strip()
            if result:
                log.info(f"✅ Gemini extraction: Successfully extracted {len(result)} characters from {file_type.upper()}")
            else:
                log.warning(f"⚠️ Gemini extraction: No text extracted from {file_type.upper()}")
            return result
            
        finally:
            # Clean up temporary file
            try:
                os.remove(tmp_path)
            except Exception:
                pass
                
    except Exception as e:
        log.error(f"❌ Gemini extraction: Failed to extract text from {file_type.upper()}: {e}")
        return ""


@traceable(
    name="gemini_structured_extraction",
    run_type="llm",
    metadata={"extraction_method": "gemini", "output_format": "structured_json"}
)
def _extract_structured_resume_with_gemini(file_bytes: bytes, file_type: str = "pdf") -> Optional[Dict[str, Any]]:
    """
    Extract structured resume data directly from PDF or DOCX files using Gemini's native structured output.
    This bypasses the text extraction step and gets structured data directly from the PDF.
    
    Args:
        file_bytes: Binary content of the file
        file_type: "pdf" or "docx"
    
    Returns:
        Structured resume data as a dictionary, or None if extraction fails
    """
    if not _HAS_GEMINI:
        log.warning("❌ Gemini structured extraction: google.genai not available")
        return None
    
    log.debug(f"🤖 Gemini structured extraction: Attempting to extract structured data from {file_type.upper()} using Google Gemini API")
    
    try:
        # Import settings and schema here to avoid circular imports
        from settings import settings
        from agents.resume_schema import StructuredResumeOutput
        
        # Initialize Gemini Client (new google-genai SDK API)
        client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        
        # Create a temporary file
        suffix = ".pdf" if file_type.lower() == "pdf" else ".docx"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        try:
            # Upload file to Gemini (new API - use 'file' parameter, not 'path')
            uploaded_file = client.files.upload(file=tmp_path)
            
            # OPTIMIZATION: Wait for file processing with aggressive polling for faster response
            # Note: This is a sync function called from async via asyncio.to_thread()
            # time.sleep() blocks thread pool thread (acceptable), but we keep it minimal
            import time
            max_wait_time = 15  # Reduced from 20 to 15 seconds
            start_wait = time.time()
            wait_interval = 0.2  # Start with 0.2s (faster polling)
            max_wait_interval = 1.0  # Reduced from 2.0s to 1.0s max
            
            while uploaded_file.state.name == "PROCESSING":
                elapsed = time.time() - start_wait
                if elapsed > max_wait_time:
                    log.error(f"File processing timeout after {max_wait_time}s")
                    try:
                        client.files.delete(name=uploaded_file.name)
                    except:
                        pass
                    return None
                
                # ✅ PERFORMANCE: Minimal sleep duration to reduce thread pool blocking
                # Note: This runs in thread pool via asyncio.to_thread(), so time.sleep() is acceptable
                # but we keep intervals short to minimize thread blocking
                sleep_duration = min(wait_interval, max_wait_interval)
                time.sleep(sleep_duration)
                wait_interval = min(wait_interval * 1.3, max_wait_interval)  # Less aggressive backoff (1.3x vs 1.5x)
                
                uploaded_file = client.files.get(name=uploaded_file.name)
            
            if uploaded_file.state.name == "FAILED":
                log.error(f"File upload failed: {uploaded_file.state}")
                return None
            
            # Convert Pydantic model to JSON schema for Gemini's structured output
            try:
                json_schema = StructuredResumeOutput.model_json_schema()
            except AttributeError:
                # Fallback for older Pydantic versions
                json_schema = StructuredResumeOutput.schema()
            
            # Resolve $ref and $defs to inline definitions (Gemini's GenerationConfig doesn't accept them)
            def resolve_schema_refs(schema: dict, defs: dict = None) -> dict:
                """Recursively resolve $ref references in JSON schema to inline definitions."""
                if defs is None:
                    defs = schema.get("$defs", {})
                
                if isinstance(schema, dict):
                    # Check if this is a $ref that needs to be resolved
                    if "$ref" in schema and len(schema) == 1:
                        ref_path = schema["$ref"]
                        if ref_path.startswith("#/$defs/"):
                            ref_name = ref_path.replace("#/$defs/", "")
                            if ref_name in defs:
                                # Resolve the referenced definition recursively
                                return resolve_schema_refs(defs[ref_name], defs)
                        # If can't resolve, return as-is
                        return schema
                    
                    resolved = {}
                    for key, value in schema.items():
                        if key == "$ref":
                            # Handle $ref at top level of object
                            ref_path = value
                            if ref_path.startswith("#/$defs/"):
                                ref_name = ref_path.replace("#/$defs/", "")
                                if ref_name in defs:
                                    # Replace $ref with the resolved definition
                                    resolved.update(resolve_schema_refs(defs[ref_name], defs))
                                else:
                                    resolved[key] = value
                            else:
                                resolved[key] = value
                        elif key == "$defs":
                            # Skip $defs in the final schema (already resolved inline)
                            continue
                        elif key == "anyOf" and isinstance(value, list):
                            # Handle anyOf arrays - resolve $ref items
                            resolved_anyof = []
                            for item in value:
                                if isinstance(item, dict) and "$ref" in item:
                                    ref_path = item["$ref"]
                                    if ref_path.startswith("#/$defs/"):
                                        ref_name = ref_path.replace("#/$defs/", "")
                                        if ref_name in defs:
                                            # Replace $ref with resolved definition
                                            resolved_anyof.append(resolve_schema_refs(defs[ref_name], defs))
                                        else:
                                            resolved_anyof.append(item)
                                    else:
                                        resolved_anyof.append(resolve_schema_refs(item, defs) if isinstance(item, dict) else item)
                                else:
                                    resolved_anyof.append(resolve_schema_refs(item, defs) if isinstance(item, dict) else item)
                            resolved[key] = resolved_anyof
                        elif isinstance(value, dict):
                            resolved[key] = resolve_schema_refs(value, defs)
                        elif isinstance(value, list):
                            resolved[key] = [resolve_schema_refs(item, defs) if isinstance(item, dict) else item for item in value]
                        else:
                            resolved[key] = value
                    return resolved
                elif isinstance(schema, list):
                    return [resolve_schema_refs(item, defs) if isinstance(item, dict) else item for item in schema]
                else:
                    return schema
            
            # Resolve all $ref references to inline definitions
            json_schema = resolve_schema_refs(json_schema)
            
            # Build comprehensive prompt for structured extraction
            prompt = """Extract all information from this resume document and structure it according to the provided schema.

CRITICAL EXTRACTION RULES:
1. Extract ONLY information explicitly stated in the resume. DO NOT infer, guess, or make up information.
2. Name: Extract ONLY the person's actual name (first name and/or last name). DO NOT extract professional titles, headers, taglines, job descriptions, or company names.
3. Contact details: Only extract if explicitly present. DO NOT make up emails, phones, or locations.
4. Work experience: Extract ONLY jobs explicitly listed. Extract COMPLETE bullet points, not fragments.
5. Education: Extract ONLY what is explicitly stated. DO NOT invent degree or major.
6. Skills: Scan the ENTIRE resume to find ALL sections containing skills. Extract ALL skills from ALL such sections. Each skill entry MUST contain exactly ONE skill name - split multiple skills listed together into separate entries.
7. Certifications: Only extract from sections explicitly titled "Certifications", "Professional Certifications", "Licenses", or similar.
8. Projects: Extract only if there is an explicit projects section with actual project entries.
9. Professional summary: Extract from "PROFESSIONAL SUMMARY" sections, header/title lines, or summary paragraphs at the top of the resume.
10. Total experience years: Calculate from explicit work dates as a float (sum of non-overlapping date ranges). Treat "Present", "Now", "Current" as the current date.

If information is not present, use null for optional fields, empty string "" for required text fields, empty array [] for lists, and empty object {} for objects. Never use placeholder text like "N/A" or "empty string" - always use actual empty values."""

            # Generate content with structured output (using Gemini's native structured output)
            # Based on Gemini API docs: use response_mime_type="application/json" and response_schema
            # Try different API formats based on google-genai SDK version
            try:
                # Method 1: Try using GenerationConfig with response_schema (newer API format)
                from google.genai.types import GenerationConfig
                generation_config = GenerationConfig(
                    response_schema=json_schema,
                    response_mime_type="application/json"
                )
                response = client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=[prompt, uploaded_file],
                    config=generation_config
                )
            except (ImportError, AttributeError, TypeError) as e:
                # Method 2: Try direct config dict with response_schema (alternative API format)
                log.debug(f"GenerationConfig approach failed ({e}), trying dict config format")
                try:
                    response = client.models.generate_content(
                        model=settings.GEMINI_MODEL,
                        contents=[prompt, uploaded_file],
                        config={
                            "response_schema": json_schema,
                            "response_mime_type": "application/json"
                        }
                    )
                except Exception as e2:
                    # Method 3: Try with response_json_schema (alternative parameter name)
                    log.debug(f"Dict config with response_schema failed ({e2}), trying response_json_schema")
                    try:
                        response = client.models.generate_content(
                            model=settings.GEMINI_MODEL,
                            contents=[prompt, uploaded_file],
                            config={
                                "response_json_schema": json_schema,
                                "response_mime_type": "application/json"
                            }
                        )
                    except Exception as e3:
                        # Last resort: Try without structured output config (will return text, parse manually)
                        log.warning(f"All structured output config methods failed ({e3}), falling back to text extraction with manual JSON parsing")
                        response = client.models.generate_content(
                            model=settings.GEMINI_MODEL,
                            contents=[f"{prompt}\n\nReturn the response as valid JSON matching this schema: {json.dumps(json_schema, indent=2)}", uploaded_file]
                        )
            
            # Parse the structured response
            if hasattr(response, 'text') and response.text:
                try:
                    structured_data = json.loads(response.text)
                    log.info(f"✅ Gemini structured extraction: Successfully extracted structured data from {file_type.upper()}")
                    return structured_data
                except json.JSONDecodeError as e:
                    log.error(f"Failed to parse structured response as JSON: {e}")
                    log.debug(f"Response text: {response.text[:500]}")
                    return None
            else:
                log.warning(f"⚠️ Gemini structured extraction: No structured data extracted from {file_type.upper()}")
                return None
                
        finally:
            # Clean up uploaded file
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception as e:
                log.debug(f"Failed to delete uploaded file: {e}")
            
            # Clean up temporary file
            try:
                os.remove(tmp_path)
            except Exception:
                pass
                
    except Exception as e:
        log.error(f"❌ Gemini structured extraction: Failed to extract structured data from {file_type.upper()}: {e}")
        log.exception("Gemini structured extraction error")
        return None


async def extract_structured_resume_from_file_async(file_url: str, file_type: str = "pdf") -> Optional[Dict[str, Any]]:
    """
    Async public function to extract structured resume data directly from a file URL using Gemini's native structured output.
    This bypasses the text extraction step and gets structured data directly from the PDF/DOCX.
    
    Args:
        file_url: URL to the resume file
        file_type: "pdf", "docx", or "doc" (default: "pdf")
    
    Returns:
        Structured resume data as a dictionary matching StructuredResumeOutput schema, or None if extraction fails
    """
    try:
        # ✅ PERFORMANCE: Use shared HTTP client for connection pooling
        if _HAS_HTTP_CLIENT:
            # Use shared client from core.http_client for better connection reuse
            client = await get_http_client()
            response = await client.get(file_url, timeout=30.0, follow_redirects=True)
            response.raise_for_status()
            content = response.content
        elif _HAS_HTTPX:
            # Fallback to creating a client if shared client not available
            import httpx
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(file_url)
                response.raise_for_status()
                content = response.content
        else:
            # httpx is a required dependency - raise clear error if not available
            # This prevents blocking requests.get() calls in async context (Section 2 Issue 3)
            raise RuntimeError(
                "httpx is required for async HTTP operations but not available. "
                "Install with: pip install httpx"
            )
        
        if file_type.lower() == "doc":
            # For DOC files, we need to convert to PDF first
            pdf_bytes = await asyncio.to_thread(_doc_to_pdf_with_libreoffice, content)
            if pdf_bytes:
                return await asyncio.to_thread(_extract_structured_resume_with_gemini, pdf_bytes, "pdf")
            return None
        else:
            # For PDF and DOCX, extract directly
            return await asyncio.to_thread(_extract_structured_resume_with_gemini, content, file_type)
    except Exception as e:
        log.error(f"Failed to extract structured resume from {file_url}: {e}")
        return None


def extract_structured_resume_from_file(file_url: str, file_type: str = "pdf") -> Optional[Dict[str, Any]]:
    """
    Synchronous wrapper for extract_structured_resume_from_file_async.
    Use the async version when possible for better performance.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, we can't use run_until_complete
            # This should not happen if called correctly, but handle gracefully
            log.warning("extract_structured_resume_from_file called from async context - use extract_structured_resume_from_file_async instead")
            return None
        return loop.run_until_complete(extract_structured_resume_from_file_async(file_url, file_type))
    except RuntimeError:
        # No event loop running, create a new one
        return asyncio.run(extract_structured_resume_from_file_async(file_url, file_type))


def _extract_text_from_doc(file_bytes: bytes) -> str:
    """
    Extract text from legacy DOC file using Gemini.
    Used by app.py for direct DOC extraction.
    
    Strategy:
    1. Convert DOC to PDF using LibreOffice
    2. Extract text from PDF using Gemini
    """
    log.info("📋 Processing DOC file...")
    
    # Convert DOC to PDF and use Gemini
    pdf_bytes = _doc_to_pdf_with_libreoffice(file_bytes)
    if pdf_bytes:
        text = _extract_text_with_gemini(pdf_bytes, file_type="pdf")
        if text and text.strip():
            text = _normalize_text(text)
            d = _quality_diag(text)
            log.info(f"[DOC] len={d['len']} words={d['words']} alpha={d['alpha_ratio']} weak={d['weak']}")
            return text

    log.warning("⚠️ DOC extraction via Gemini failed")
    return ""

def _have_sync(cmd: str) -> bool:
    """Internal sync function to check if command is available."""
    try:
        return subprocess.run(["which", cmd], capture_output=True, timeout=5).returncode == 0
    except Exception:
        return False

def _have(cmd: str) -> bool:
    """Check if command is available - runs synchronously (usually called during init)."""
    # This is typically called during initialization, so we keep it sync
    # If needed in async context, caller should wrap it
    return _have_sync(cmd)

def _doc_to_docx_with_libreoffice_sync(doc_bytes: bytes) -> Optional[bytes]:
    """Internal sync function for DOC to DOCX conversion."""
    if not (_have_sync("libreoffice") or platform.system() == "Windows"):
        return None
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.doc")
        out = os.path.join(td, "in.docx")
        with open(src, "wb") as f:
            f.write(doc_bytes)
        soffice = (r"C:\Program Files\LibreOffice\program\soffice.exe"
                   if platform.system() == "Windows" else "libreoffice")
        try:
            subprocess.run([soffice, "--headless", "--convert-to", "docx", "--outdir", td, src],
                           check=True, capture_output=True, timeout=60)
            if os.path.exists(out):
                return open(out, "rb").read()
        except Exception as e:
            log.debug(f"LibreOffice doc->docx failed: {e}")
    return None

def _doc_to_docx_with_libreoffice(doc_bytes: bytes) -> Optional[bytes]:
    """Convert DOC to DOCX using LibreOffice - runs in thread pool if in async context."""
    try:
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(_subprocess_executor, _doc_to_docx_with_libreoffice_sync, doc_bytes)
    except RuntimeError:
        return _doc_to_docx_with_libreoffice_sync(doc_bytes)

def _doc_to_pdf_with_libreoffice_sync(doc_bytes: bytes) -> Optional[bytes]:
    """Internal sync function for DOC to PDF conversion."""
    if not (_have_sync("libreoffice") or platform.system() == "Windows"):
        return None
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.doc")
        out = os.path.join(td, "in.pdf")
        with open(src, "wb") as f:
            f.write(doc_bytes)
        soffice = (r"C:\Program Files\LibreOffice\program\soffice.exe"
                   if platform.system() == "Windows" else "libreoffice")
        try:
            subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", td, src],
                           check=True, capture_output=True, timeout=60)
            if os.path.exists(out):
                return open(out, "rb").read()
        except Exception as e:
            log.debug(f"LibreOffice doc->pdf failed: {e}")
    return None

def _doc_to_pdf_with_libreoffice(doc_bytes: bytes) -> Optional[bytes]:
    """Convert DOC to PDF using LibreOffice - runs in thread pool if in async context."""
    try:
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(_subprocess_executor, _doc_to_pdf_with_libreoffice_sync, doc_bytes)
    except RuntimeError:
        return _doc_to_pdf_with_libreoffice_sync(doc_bytes)

# ==========================
# Main: download + route
# ==========================

def _detect_mime_from_bytes(b: bytes, fallback_ext: str = "") -> str:
    """Detect MIME type from magic bytes, python-magic, or extension."""
    # Magic bytes first
    if len(b) >= 4:
        h4 = b[:4]
        if h4 == b'%PDF':
            return "application/pdf"
        if h4 == b'PK\x03\x04':
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if b[:2] == b'\xd0\xcf':
            return "application/msword"

    # python-magic
    if magic is not None:
        try:
            m = magic.from_buffer(b, mime=True)
            if m:
                return m
        except Exception:
            pass

    # extension guess
    ext = (fallback_ext or "").lower()
    if ext.endswith(".pdf"):
        return "application/pdf"
    if ext.endswith(".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if ext.endswith(".doc"):
        return "application/msword"

    return "application/octet-stream"

async def download_resume_text_async(file_url: str, file_type: str = "resume") -> Optional[str]:
    """
    Async version: Download and extract resume text (PDF/DOCX/DOC) using httpx.
    Returns normalized text.
    
    Strategy:
    - PDF: PyPDF2 → pdfminer column-aware
    - DOCX: Mammoth → python-docx → DOCX→PDF fallback
    - DOC: DOC→DOCX → DOCX path → DOC→PDF fallback
    
    Handles Firebase Storage URLs with token authentication.
    """
    try:
        log.info(f"🔄 Downloading {file_type} from: {file_url}")
        
        # Use shared HTTP client for async requests
        if _HAS_HTTP_CLIENT:
            try:
                client = await get_http_client()
                response = await client.get(file_url, timeout=120.0, follow_redirects=True)
                response.raise_for_status()
                content = response.content
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 403:
                    log.error(
                        f"❌ 403 Forbidden when downloading {file_type}. "
                        f"This usually means the Firebase Storage token has expired or the file is not accessible. "
                        f"URL: {file_url[:100]}..."
                    )
                    if "firebasestorage.googleapis.com" in file_url:
                        log.error(
                            "Firebase Storage URL detected. The download token may have expired. "
                            "Consider generating a new signed URL with a fresh token."
                        )
                    raise
                else:
                    raise
            except Exception as e:
                log.error(f"❌ Download error: {e}")
                return ""
        else:
            # httpx is a required dependency for async HTTP - raise clear error
            # This prevents blocking requests.get() calls in async context (Section 2 Issue 3)
            raise RuntimeError(
                "httpx and shared HTTP client are required for async operations but not available. "
                "Ensure core.http_client module is properly initialized during startup."
            )

        url_no_qs = unquote(file_url.split("?")[0])
        mime = _detect_mime_from_bytes(content, fallback_ext=url_no_qs)
        log.info(f"📄 Detected MIME: {mime}  | size={len(content)} bytes")

        # ---- PDF ----
        if mime == "application/pdf":
            # Try Gemini first for PDF extraction
            if _HAS_GEMINI:
                log.info("📄 PDF extraction: Attempting Gemini first...")
            else:
                log.info("📄 PDF extraction: Gemini not available, using fallback methods (PyPDF2, pdfminer)")
            
            # OPTIMIZATION: Run synchronous Gemini extraction in thread pool to avoid blocking event loop
            try:
                text = await asyncio.to_thread(_extract_text_with_gemini, content, "pdf")
            except Exception as e:
                log.warning(f"Gemini extraction failed: {e}, falling back to other methods")
                text = ""
            
            if text and text.strip():
                text = _normalize_text(text)
                d = _quality_diag(text)
                log.info(
                    f"✅ PDF extraction: Gemini SUCCESS - len={d['len']} "
                    f"words={d['words']} alpha={d['alpha_ratio']} weak={d['weak']}"
                )
                return _maybe_clamp_text(text, file_type)
            
            # Fallback to robust PDF extraction methods when Gemini fails or is unavailable
            log.warning(
                "⚠️ PDF extraction: Gemini failed or unavailable, using fallback methods "
                "(PyPDF2, pdfminer)..."
            )
            try:
                extracted_text, diag = _pdf_extract_best(content)
                if extracted_text and extracted_text.strip():
                    text = _normalize_text(extracted_text)
                    d = _quality_diag(text)
                    log.info(f"[PDF] Fallback extraction ({', '.join(diag.get('used', []))}): len={d['len']} words={d['words']} alpha={d['alpha_ratio']} weak={d['weak']}")
                    return _maybe_clamp_text(text, file_type)
                else:
                    log.error(f"❌ All PDF extraction methods failed. Diagnostic: {diag}")
                    return ""
            except Exception as e:
                log.error(f"❌ PDF fallback extraction failed: {e}")
                return ""

        # ---- DOCX ----
        if mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Try Gemini first for DOCX extraction
            if _HAS_GEMINI:
                log.info("📄 DOCX extraction: Attempting Gemini first...")
            else:
                log.info("📄 DOCX extraction: Gemini not available, using fallback methods (Mammoth, python-docx)...")
            
            # OPTIMIZATION: Run synchronous Gemini extraction in thread pool to avoid blocking event loop
            try:
                text = await asyncio.to_thread(_extract_text_with_gemini, content, "docx")
            except Exception as e:
                log.warning(f"Gemini extraction failed: {e}, falling back to other methods")
                text = ""
            
            if text and text.strip():
                text = _normalize_text(text)
                d = _quality_diag(text)
                log.info(
                    f"✅ DOCX extraction: Gemini SUCCESS - len={d['len']} "
                    f"words={d['words']} alpha={d['alpha_ratio']} weak={d['weak']}"
                )
                return _maybe_clamp_text(text, file_type)
            
            # Fallback to robust DOCX extraction methods when Gemini fails or is unavailable
            log.warning(
                "⚠️ DOCX extraction: Gemini failed or unavailable, using fallback methods "
                "(Mammoth, python-docx)..."
            )
            try:
                # Try Mammoth first (preserves formatting better)
                text = _docx_with_mammoth_to_text(content)
                if text and text.strip():
                    d = _quality_diag(text)
                    log.info(f"[DOCX] Fallback (Mammoth): len={d['len']} words={d['words']} alpha={d['alpha_ratio']} weak={d['weak']}")
                    return _maybe_clamp_text(text, file_type)
                
                # Try python-docx as second fallback
                text = _docx_with_python_docx(content)
                if text and text.strip():
                    d = _quality_diag(text)
                    log.info(f"[DOCX] Fallback (python-docx): len={d['len']} words={d['words']} alpha={d['alpha_ratio']} weak={d['weak']}")
                    return _maybe_clamp_text(text, file_type)
                
                # Try converting to PDF and extracting (last resort)
                pdf_bytes = _docx_to_pdf_with_libreoffice_sync(content)
                if pdf_bytes:
                    extracted_text, diag = _pdf_extract_best(pdf_bytes)
                    if extracted_text and extracted_text.strip():
                        text = _normalize_text(extracted_text)
                        d = _quality_diag(text)
                        log.info(f"[DOCX] Fallback (DOCX→PDF→extract): len={d['len']} words={d['words']} alpha={d['alpha_ratio']} weak={d['weak']}")
                        return _maybe_clamp_text(text, file_type)
                
                log.error("❌ All DOCX extraction methods failed")
                return ""
            except Exception as e:
                log.error(f"❌ DOCX fallback extraction failed: {e}")
                return ""

        # ---- DOC (legacy) ----
        if mime == "application/msword":
            # Convert DOC to PDF first
            pdf_bytes_result = _doc_to_pdf_with_libreoffice(content)
            if asyncio.iscoroutine(pdf_bytes_result):
                pdf_bytes = await pdf_bytes_result
            else:
                pdf_bytes = pdf_bytes_result
            
            if pdf_bytes:
                # Try Gemini first for PDF extraction
                if _HAS_GEMINI:
                    log.info("📄 DOC extraction: Converted to PDF, attempting Gemini first...")
                else:
                    log.info(
                        "📄 DOC extraction: Converted to PDF, Gemini not available, "
                        "using fallback methods..."
                    )

                text = _extract_text_with_gemini(pdf_bytes, file_type="pdf")
                if text and text.strip():
                    text = _normalize_text(text)
                    d = _quality_diag(text)
                    log.info(
                        f"✅ DOC extraction: Gemini SUCCESS - len={d['len']} "
                        f"words={d['words']} alpha={d['alpha_ratio']} weak={d['weak']}"
                    )
                    return _maybe_clamp_text(text, file_type)

                # Fallback to robust PDF extraction methods when Gemini fails
                log.warning(
                    "⚠️ DOC extraction: Gemini failed or unavailable, using fallback methods "
                    "(PyPDF2, pdfminer)..."
                )
                try:
                    extracted_text, diag = _pdf_extract_best(pdf_bytes)
                    if extracted_text and extracted_text.strip():
                        text = _normalize_text(extracted_text)
                        d = _quality_diag(text)
                        log.info(
                            f"[DOC] Fallback extraction ({', '.join(diag.get('used', []))}): "
                            f"len={d['len']} words={d['words']} alpha={d['alpha_ratio']} "
                            f"weak={d['weak']}"
                        )
                        return _maybe_clamp_text(text, file_type)
                    else:
                        log.error(f"❌ All DOC extraction methods failed. Diagnostic: {diag}")
                        return ""
                except Exception as e:
                    log.error(f"❌ DOC fallback extraction failed: {e}")
                    return ""
            else:
                log.error("❌ Failed to convert DOC to PDF - cannot extract text")
                return ""

        # ---- Plain text / others ----
        try:
            text = content.decode('utf-8').strip()
        except UnicodeDecodeError:
            text = content.decode('latin-1', errors='ignore').strip()
        return _maybe_clamp_text(_normalize_text(text), file_type)

    except Exception as e:
        log.error(f"❌ Unexpected error in download_resume_text_async: {e}")
        return ""

def _download_resume_text_sync_impl(file_url: str, file_type: str = "resume") -> Optional[str]:
    """Internal sync implementation using requests - runs in thread pool when needed."""
    try:
        log.info(f"🔄 Downloading {file_type} from: {file_url}")
        
        # Try downloading with standard request
        try:
            r = requests.get(file_url, timeout=120, allow_redirects=True)
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                log.error(
                    f"❌ 403 Forbidden when downloading {file_type}. "
                    f"This usually means the Firebase Storage token has expired or the file is not accessible. "
                    f"URL: {file_url[:100]}..."
                )
                if "firebasestorage.googleapis.com" in file_url:
                    log.error(
                        "Firebase Storage URL detected. The download token may have expired. "
                        "Consider generating a new signed URL with a fresh token."
                    )
                raise
            else:
                raise

        url_no_qs = unquote(file_url.split("?")[0])
        mime = _detect_mime_from_bytes(r.content, fallback_ext=url_no_qs)
        log.info(f"📄 Detected MIME: {mime}  | size={len(r.content)} bytes")

        # ---- PDF ----
        if mime == "application/pdf":
            # Try Gemini first for PDF extraction
            if _HAS_GEMINI:
                log.info("📄 PDF extraction: Attempting Gemini first...")
            else:
                log.info("📄 PDF extraction: Gemini not available, using fallback methods (PyPDF2, pdfminer)")
            
            text = _extract_text_with_gemini(r.content, file_type="pdf")
            if text and text.strip():
                text = _normalize_text(text)
                d = _quality_diag(text)
                log.info(
                    f"✅ PDF extraction: Gemini SUCCESS - len={d['len']} "
                    f"words={d['words']} alpha={d['alpha_ratio']} weak={d['weak']}"
                )
                return _maybe_clamp_text(text, file_type)
            
            # Fallback to robust PDF extraction methods when Gemini fails or is unavailable
            log.warning(
                "⚠️ PDF extraction: Gemini failed or unavailable, using fallback methods "
                "(PyPDF2, pdfminer)..."
            )
            try:
                extracted_text, diag = _pdf_extract_best(r.content)
                if extracted_text and extracted_text.strip():
                    text = _normalize_text(extracted_text)
                    d = _quality_diag(text)
                    log.info(f"[PDF] Fallback extraction ({', '.join(diag.get('used', []))}): len={d['len']} words={d['words']} alpha={d['alpha_ratio']} weak={d['weak']}")
                    return _maybe_clamp_text(text, file_type)
                else:
                    log.error(f"❌ All PDF extraction methods failed. Diagnostic: {diag}")
                    return ""
            except Exception as e:
                log.error(f"❌ PDF fallback extraction failed: {e}")
                return ""

        # ---- DOCX ----
        if mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Try Gemini first for DOCX extraction
            if _HAS_GEMINI:
                log.info("📄 DOCX extraction: Attempting Gemini first...")
            else:
                log.info("📄 DOCX extraction: Gemini not available, using fallback methods (Mammoth, python-docx)...")
            
            text = _extract_text_with_gemini(r.content, file_type="docx")
            if text and text.strip():
                text = _normalize_text(text)
                d = _quality_diag(text)
                log.info(
                    f"✅ DOCX extraction: Gemini SUCCESS - len={d['len']} "
                    f"words={d['words']} alpha={d['alpha_ratio']} weak={d['weak']}"
                )
                return _maybe_clamp_text(text, file_type)
            
            # Fallback to robust DOCX extraction methods when Gemini fails or is unavailable
            log.warning(
                "⚠️ DOCX extraction: Gemini failed or unavailable, using fallback methods "
                "(Mammoth, python-docx)..."
            )
            try:
                # Try Mammoth first (preserves formatting better)
                text = _docx_with_mammoth_to_text(r.content)
                if text and text.strip():
                    d = _quality_diag(text)
                    log.info(f"[DOCX] Fallback (Mammoth): len={d['len']} words={d['words']} alpha={d['alpha_ratio']} weak={d['weak']}")
                    return _maybe_clamp_text(text, file_type)
                
                # Try python-docx as second fallback
                text = _docx_with_python_docx(r.content)
                if text and text.strip():
                    d = _quality_diag(text)
                    log.info(f"[DOCX] Fallback (python-docx): len={d['len']} words={d['words']} alpha={d['alpha_ratio']} weak={d['weak']}")
                    return _maybe_clamp_text(text, file_type)
                
                # Try converting to PDF and extracting (last resort)
                pdf_bytes = _docx_to_pdf_with_libreoffice_sync(r.content)
                if pdf_bytes:
                    extracted_text, diag = _pdf_extract_best(pdf_bytes)
                    if extracted_text and extracted_text.strip():
                        text = _normalize_text(extracted_text)
                        d = _quality_diag(text)
                        log.info(f"[DOCX] Fallback (DOCX→PDF→extract): len={d['len']} words={d['words']} alpha={d['alpha_ratio']} weak={d['weak']}")
                        return _maybe_clamp_text(text, file_type)
                
                log.error("❌ All DOCX extraction methods failed")
                return ""
            except Exception as e:
                log.error(f"❌ DOCX fallback extraction failed: {e}")
                return ""

        # ---- DOC (legacy) ----
        if mime == "application/msword":
            # Convert DOC to PDF first
            pdf_bytes = _doc_to_pdf_with_libreoffice_sync(r.content)
            if pdf_bytes:
                # Try Gemini first for PDF extraction
                if _HAS_GEMINI:
                    log.info("📄 DOC extraction: Converted to PDF, attempting Gemini first...")
                else:
                    log.info(
                        "📄 DOC extraction: Converted to PDF, Gemini not available, "
                        "using fallback methods..."
                    )

                text = _extract_text_with_gemini(pdf_bytes, file_type="pdf")
                if text and text.strip():
                    text = _normalize_text(text)
                    d = _quality_diag(text)
                    log.info(
                        f"✅ DOC extraction: Gemini SUCCESS - len={d['len']} "
                        f"words={d['words']} alpha={d['alpha_ratio']} weak={d['weak']}"
                    )
                    return _maybe_clamp_text(text, file_type)

                # Fallback to robust PDF extraction methods when Gemini fails
                log.warning(
                    "⚠️ DOC extraction: Gemini failed or unavailable, using fallback methods "
                    "(PyPDF2, pdfminer)..."
                )
                try:
                    extracted_text, diag = _pdf_extract_best(pdf_bytes)
                    if extracted_text and extracted_text.strip():
                        text = _normalize_text(extracted_text)
                        d = _quality_diag(text)
                        log.info(
                            f"[DOC] Fallback extraction ({', '.join(diag.get('used', []))}): "
                            f"len={d['len']} words={d['words']} alpha={d['alpha_ratio']} "
                            f"weak={d['weak']}"
                        )
                        return _maybe_clamp_text(text, file_type)
                    else:
                        log.error(f"❌ All DOC extraction methods failed. Diagnostic: {diag}")
                        return ""
                except Exception as e:
                    log.error(f"❌ DOC fallback extraction failed: {e}")
                    return ""
            else:
                log.error("❌ Failed to convert DOC to PDF - cannot extract text")
                return ""

        # ---- Plain text / others ----
        text = r.text.strip()
        return _maybe_clamp_text(_normalize_text(text), file_type)

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            log.error(
                f"❌ Access denied (403) when downloading {file_type} from {file_url[:100]}... "
                f"Please ensure the file URL has a valid access token or the file permissions allow public access."
            )
        else:
            log.error(f"❌ HTTP error {e.response.status_code} when downloading {file_type}: {e}")
        return ""
    except requests.exceptions.RequestException as e:
        log.error(f"❌ Download error for {file_url[:100]}...: {type(e).__name__}: {e}")
        return ""
    except Exception as e:
        log.error(f"❌ Unexpected error in _download_resume_text_sync_impl for {file_url[:100]}...: {type(e).__name__}: {e}", exc_info=True)
        return ""

def download_resume_text(file_url: str, file_type: str = "resume") -> Optional[str]:
    """
    Synchronous version: Download and extract resume text (PDF/DOCX/DOC).
    Returns normalized text.
    
    This function runs in a thread pool if called from async context to avoid blocking.
    For async code, prefer download_resume_text_async().
    
    Strategy:
    - PDF: PyPDF2 → pdfminer column-aware
    - DOCX: Mammoth → python-docx → DOCX→PDF fallback
    - DOC: DOC→DOCX → DOCX path → DOC→PDF fallback
    
    Handles Firebase Storage URLs with token authentication.
    """
    try:
        # Check if we're in an async context
        loop = asyncio.get_running_loop()
        # We're in async context, run sync implementation in thread pool
        import concurrent.futures
        log.warning("⚠️ download_resume_text called from async context - consider using download_resume_text_async()")
        return loop.run_in_executor(
            None,  # Use default executor
            _download_resume_text_sync_impl,
            file_url,
            file_type
        )
    except RuntimeError:
        # No event loop running, execute directly
        try:
            result = _download_resume_text_sync_impl(file_url, file_type)
            if not result or not result.strip():
                log.error(f"❌ download_resume_text returned empty result for {file_url}")
            return result
        except Exception as e:
            log.error(f"❌ download_resume_text failed with exception: {type(e).__name__}: {e}", exc_info=True)
            return None

def download_resume_text_with_quality(file_url: str, file_type: str = "resume") -> Tuple[Optional[str], dict]:
    """
    Download and extract resume text with quality diagnostics.
    Returns (text, quality_info) where quality_info contains extraction metadata.
    """
    text = download_resume_text(file_url, file_type)
    quality = _quality_diag(text) if text else {"len": 0, "words": 0, "alpha_ratio": 0.0, "unique_ratio": 0.0, "weak": True}
    return text, quality

