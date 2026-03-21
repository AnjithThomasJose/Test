import os
from typing import Dict, Any, Optional


def _safe_get(d: Optional[Dict[str, Any]], key: str, default=None):
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


class RetrievalBackend:
    CHROMA = "chroma"
    STUB = "stub"


def get_backend() -> str:
    return os.getenv("RETRIEVAL_BACKEND", RetrievalBackend.STUB).lower()


def get_job_context(job_id: Optional[str]) -> Dict[str, Any]:
    backend = get_backend()
    if backend == RetrievalBackend.CHROMA:
        try:
            return _chroma_get("jobs", job_id)
        except Exception:
            pass
    return _stub_job(job_id)


def get_company_context(company_id: Optional[str]) -> Dict[str, Any]:
    backend = get_backend()
    if backend == RetrievalBackend.CHROMA:
        try:
            return _chroma_get("companies", company_id)
        except Exception:
            pass
    return _stub_company(company_id)


def summarize_for_prompt(job_ctx: Dict[str, Any], company_ctx: Dict[str, Any]) -> Dict[str, str]:
    title = _safe_get(job_ctx, "title", "Software Developer")
    level = _safe_get(job_ctx, "level", "")
    responsibilities = _safe_get(job_ctx, "responsibilities", [])
    requirements = _safe_get(job_ctx, "requirements", [])

    company_name = _safe_get(company_ctx, "name", "Tech Company")
    industry = _safe_get(company_ctx, "industry", "")
    products = _safe_get(company_ctx, "products", [])

    job_summary = ", ".join([p for p in [title, level] if p])
    resp_summary = "; ".join(responsibilities[:3]) if responsibilities else ""
    req_summary = "; ".join(requirements[:3]) if requirements else ""

    company_summary = ", ".join([p for p in [company_name, industry] if p])
    product_summary = ", ".join(products[:3]) if products else ""

    return {
        "job_summary": job_summary,
        "job_responsibilities": resp_summary,
        "job_requirements": req_summary,
        "company_summary": company_summary,
        "company_products": product_summary,
    }


# -----------------
# Chroma backend
# -----------------
def _chroma_client():
    import chromadb  # type: ignore
    return chromadb.Client()


def _chroma_get(collection: str, key: Optional[str]) -> Dict[str, Any]:
    if not key:
        return {}
    client = _chroma_client()
    coll = client.get_or_create_collection(name=collection)
    # Use key as id; fetch stored metadata
    result = coll.get(ids=[str(key)], include=["metadatas"])
    metas = result.get("metadatas") or []
    return metas[0] if metas else {}


# -----------------
# Stub fallback
# -----------------
def _stub_job(job_id: Optional[str]) -> Dict[str, Any]:
    return {
        "id": job_id or "job_stub",
        "title": "Software Developer",
        "level": "Mid",
        "responsibilities": [
            "Build and maintain features",
            "Collaborate with cross-functional teams",
            "Write tests and documentation",
        ],
        "requirements": [
            "3+ years experience",
            "Proficiency in Python/JS",
            "Familiarity with cloud and CI/CD",
        ],
    }


def _stub_company(company_id: Optional[str]) -> Dict[str, Any]:
    return {
        "id": company_id or "company_stub",
        "name": "Tech Company",
        "industry": "Software",
        "size": "500-1000",
        "products": ["Core Platform", "Analytics Suite", "Mobile App"],
        "values": ["Customer obsession", "Ownership", "Bias for action"],
    }


