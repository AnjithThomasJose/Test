from pathlib import Path


def extract_from_pdf(fp: Path) -> str:
    """Extract text from PDF using Gemini."""
    try:
        from utils.resume_utils import _extract_text_with_gemini
        with open(fp, "rb") as f:
            data = f.read()
        text = _extract_text_with_gemini(data, file_type="pdf")
        if text and text.strip():
            return text.strip()
    except Exception:
        pass

    return ""


def extract_markdown_from_pdf(fp: Path) -> str:
    """Extract high-quality Markdown from a PDF using Gemini."""
    try:
        from utils.resume_utils import _extract_text_with_gemini
        with open(fp, "rb") as f:
            data = f.read()
        md = _extract_text_with_gemini(data, file_type="pdf")
        if md and len(md.strip()) > 0:
            return md
    except Exception:
        pass

    return ""

def extract_from_docx(fp: Path) -> str:
    """Extract text from DOCX using Gemini."""
    try:
        from utils.resume_utils import _extract_text_with_gemini
        with open(fp, "rb") as f:
            data = f.read()
        md = _extract_text_with_gemini(data, file_type="docx")
        if md and md.strip():
            return md.strip()
    except Exception:
        pass

    return ""


def extract_from_doc(fp: Path) -> str:
    """Extract text from legacy .doc files using Gemini."""
    try:
        from utils.resume_utils import _extract_text_from_doc as _extract_doc_bytes
        with open(fp, "rb") as f:
            return _extract_doc_bytes(f.read())
    except Exception:
        return ""


def extract_text(fp: Path) -> str:
    ext = fp.suffix.lower()
    if ext == ".pdf":
        return extract_from_pdf(fp)
    if ext == ".docx":
        return extract_from_docx(fp)
    if ext == ".doc":
        return extract_from_doc(fp)
    raise ValueError("Unsupported file type")


