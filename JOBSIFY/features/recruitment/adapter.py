"""
Recruitment Adapter

Bridges the recruitment domain with the core search service.
Converts RecruitmentQuery → StandardSearchRequest
Converts StandardSearchResponse → RecruitmentResult
"""

import asyncio
import json
import logging
import re
import time
from typing import List, Optional, Dict, Any, Tuple

from core_search_service.models import (
    StandardSearchRequest, 
    StandardSearchResponse, 
    SearchFilter,
    SearchFilterOperator
)
from core_search_service.search_gateway import SearchGateway
from models.llm_invoker import invoke_llm
from core.model_registry import TaskType
from .models import RecruitmentQuery, RecruitmentResult, CandidateMatch
from .query_parser import QueryParser
from .normalizer import Normalizer
from settings import settings

log = logging.getLogger(__name__)

# LLM re-rank: fetch more candidates from vector search, then score/filter with LLM (tuned for speed)
RECRUITMENT_LLM_FETCH_MULTIPLIER = 2
RECRUITMENT_LLM_MIN_POOL = 25
RECRUITMENT_LLM_MIN_SCORE = 0.5
RECRUITMENT_LLM_MAX_CANDIDATES = 15
RECRUITMENT_LLM_TIMEOUT_SECONDS = 60
# Blend vector + LLM score for final ranking (better when LLM is strict)
RECRUITMENT_VECTOR_SCORE_WEIGHT = 0.35
RECRUITMENT_LLM_SCORE_WEIGHT = 0.65
# Skip LLM re-rank when vector results are already high-confidence
RECRUITMENT_RERANK_SKIP_MAX_CANDIDATES = 3
RECRUITMENT_RERANK_SKIP_MIN_SCORE = 0.70

# Seniority post-filter: normalized seniority -> allowed seniority_level values
RECRUITMENT_SENIORITY_ALLOWED = {
    "senior": ["senior", "lead"],
    "lead": ["lead"],
    "mid": ["mid"],
    "junior": ["junior"],
}


def _seniority_for_post_filter(normalized: Dict[str, Any], query: RecruitmentQuery) -> Optional[str]:
    """Return normalized seniority key for post-filter (senior, lead, mid, junior) or None."""
    s = (normalized.get("seniority") or "").strip().lower()
    if s and s in RECRUITMENT_SENIORITY_ALLOWED:
        return s
    raw = (query.parsed_experience or "").strip().lower()
    if not raw:
        return None
    if raw in ("senior", "sr", "lead", "principal", "staff"):
        return "senior" if raw in ("senior", "sr", "staff") else "lead"
    if raw in ("junior", "jr", "entry", "entry-level"):
        return "junior"
    if raw in ("mid", "mid-level", "intermediate"):
        return "mid"
    return None


def _display_candidate_id(match: CandidateMatch) -> str:
    """Resolve display candidate id (uid) from match; strip chunk suffix if present."""
    cid = match.candidate_id or ""
    if "_chunk_" in cid:
        return cid.split("_chunk_")[0]
    return cid


def _expand_search_text_for_role(parsed_role: Optional[str]) -> List[str]:
    """Return extra terms for vector recall when role is set (query expansion)."""
    if not parsed_role or not isinstance(parsed_role, str):
        return []
    role_lower = parsed_role.lower().strip()
    # AI/ML roles: add common terms for better recall
    if "ai" in role_lower or "ml" in role_lower or "machine learning" in role_lower:
        return ["machine learning", "ML", "AI", "deep learning", "PyTorch", "TensorFlow"]
    if "python" in role_lower:
        return ["Python"]
    if "frontend" in role_lower or "react" in role_lower:
        return ["React", "JavaScript", "frontend"]
    if "backend" in role_lower:
        return ["backend", "API", "database"]
    if "data" in role_lower:
        return ["data", "SQL", "analytics"]
    return []


def _derive_seniority_from_years(years: float) -> str:
    """Derive seniority level from total experience years (same logic as smart_tagger)."""
    if years < 2:
        return "junior"
    if years < 5:
        return "mid"
    if years < 10:
        return "senior"
    return "lead"


def _derive_education_level_from_resume(full_resume: Dict[str, Any]) -> str:
    """Derive highest education level from resume education list (same logic as smart_tagger)."""
    edu_list = full_resume.get("education") or full_resume.get("Education") or []
    if not isinstance(edu_list, list) or not edu_list:
        return "bachelors"
    education_level = "bachelors"
    for e in edu_list:
        if not isinstance(e, dict):
            continue
        degree = (e.get("degree") or e.get("Degree") or "").lower()
        if "phd" in degree or "doctor" in degree:
            return "phd"
        if "master" in degree or "mba" in degree or "ms" in degree or "ma" in degree:
            education_level = "masters"
        elif "bachelor" in degree or "bs" in degree or "ba" in degree or "b.tech" in degree:
            education_level = "bachelors"
        elif "high school" in degree or "diploma" in degree:
            education_level = "high_school"
    return education_level


async def analyze_recruitment_matches_with_llm(
    query: RecruitmentQuery,
    matches: List[CandidateMatch],
) -> List[Dict[str, Any]]:
    """
    Score and filter candidates with LLM (similar to ranker).
    Returns list of { candidate_id (display), match_score, rationale } for candidates
    with score >= RECRUITMENT_LLM_MIN_SCORE, sorted by match_score descending.
    """
    if not matches:
        return []

    # Build candidate summaries for prompt
    candidates_text = ""
    for i, m in enumerate(matches[:RECRUITMENT_LLM_MAX_CANDIDATES], start=1):
        display_id = _display_candidate_id(m)
        skills_str = ", ".join((m.candidate_skills or [])[:30])
        if not skills_str:
            skills_str = "Not listed"
        candidates_text += f"""
--- CANDIDATE {i} ---
ID: {display_id}
Name: {m.candidate_name or "N/A"}
Skills: {skills_str}
Experience: {m.total_experience_years} years
Seniority level: {m.seniority_level or "N/A"}
Location: {m.current_location or "N/A"}
Education level: {m.education_level or "N/A"}
"""

    # Build query context
    role = query.parsed_role or query.raw_query
    skills_part = f", skills: {', '.join(query.parsed_skills)}" if query.parsed_skills else ""
    exp_part = f", experience: {query.parsed_experience}" if query.parsed_experience else ""
    edu_part = f", education: {query.parsed_education}" if query.parsed_education else ""
    loc_part = f", location: {query.parsed_location}" if query.parsed_location else ""
    search_context = f"Role: {role}{skills_part}{exp_part}{edu_part}{loc_part}".strip()

    location_instruction = ""
    if query.parsed_location:
        location_instruction = f"""
LOCATION (required): The search specifies location "{query.parsed_location}". Candidates whose current_location does NOT match this (e.g. different city/state) must get match_score < 0.5 and be excluded from results. Only include candidates whose location matches or clearly includes the requested location.
"""

    seniority_instruction = ""
    if query.parsed_experience:
        seniority_instruction = f"""
SENIORITY (required): The search specifies experience: "{query.parsed_experience}". Candidates whose seniority_level does NOT match must get match_score < 0.5 and be excluded. Matching: Senior/Sr -> candidate must be senior or lead; Junior/Jr -> junior; Mid -> mid; Lead/Principal -> lead. Only include candidates whose seniority_level matches the requested experience level.
"""

    prompt = f"""You are an expert recruitment matching AI. Score how well each candidate fits this search.

ROLE INTERPRETATION (strict):
- "AI Developer" / "ML Engineer" / "AI/ML" = candidate MUST have AI/ML/relevant skills (e.g. ML, deep learning, PyTorch, TensorFlow, LLM, NLP, RAG, computer vision). Generic software/frontend/backend-only = weak (score <0.5).
- "Python Developer" = Python + software development; "Frontend Developer" = frontend tech (React, Vue, etc.). Match the QUALIFIER, not just "developer".
- If the search has explicit skills listed, treat those as required; candidate should have several of them for 0.5+.
{location_instruction}
{seniority_instruction}
SEARCH QUERY: "{query.raw_query}"
SEARCH CONTEXT: {search_context}

CANDIDATES:
{candidates_text}

EXAMPLES of scoring:
- Strong fit (0.75-0.95): Search "AI Developer", candidate has PyTorch, TensorFlow, LLM, RAG → high score; skills_matched: ["PyTorch", "TensorFlow", "LLM", "RAG"].
- Weak fit (0.2-0.4): Search "AI Developer", candidate has only React, JavaScript, no ML/AI → low score; omit from results or score <0.5.

For EACH candidate, return:
1. **skills_matched**: List of skills/terms from the search context (role + any listed skills) that this candidate actually has. Use exact or close names from the candidate's skills. Empty list if none.
2. **match_score**: Relevance 0.0-1.0. Only give 0.5+ if the candidate genuinely fits the role/context. Give <0.5 for weak or irrelevant fits.
3. **rationale**: 1-2 sentences explaining fit or why weak.

Scoring: 0.7-1.0 = strong fit, 0.5-0.7 = moderate fit, <0.5 = weak (exclude from results).
Return ONLY a JSON array (no markdown):
[
  {{ "candidate_id": "exact ID from above", "skills_matched": ["skill1", "skill2"], "match_score": 0.75, "rationale": "..." }},
  ...
]
Include ONLY candidates with match_score >= 0.5. Omit candidates below 0.5.
"""

    try:
        raw = await asyncio.wait_for(
            invoke_llm(
                prompt=prompt,
                task_type=TaskType.CLASSIFICATION,
                preferred_model=getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash"),
                agent_name="recruitment_llm_rerank",
                response_mime_type="application/json",
            ),
            timeout=RECRUITMENT_LLM_TIMEOUT_SECONDS,
        )
        content = (raw or "").strip()
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```\s*", "", content)
        json_match = re.search(r"\[[\s\S]*\]", content)
        if not json_match:
            log.warning("Recruitment LLM: no JSON array in response")
            return []
        json_str = re.sub(r",(\s*[}\]])", r"\1", json_match.group(0))
        results = json.loads(json_str)
        out = []
        for r in results:
            cid = r.get("candidate_id")
            score = r.get("match_score")
            rationale = r.get("rationale") or ""
            skills_matched = r.get("skills_matched")
            if not isinstance(skills_matched, list):
                skills_matched = []
            if cid is None:
                continue
            if not isinstance(score, (int, float)):
                score = 0.0
            score = max(0.0, min(1.0, float(score)))
            if score >= RECRUITMENT_LLM_MIN_SCORE:
                out.append({
                    "candidate_id": str(cid),
                    "match_score": score,
                    "rationale": rationale,
                    "skills_matched": [str(s) for s in skills_matched if s],
                })
        out.sort(key=lambda x: x["match_score"], reverse=True)
        log.info(f"Recruitment LLM: {len(out)} candidates passed threshold (>= {RECRUITMENT_LLM_MIN_SCORE})")
        return out
    except asyncio.TimeoutError:
        log.warning("Recruitment LLM: timeout")
        return []
    except Exception as e:
        log.warning(f"Recruitment LLM failed: {e}", exc_info=True)
        return []


class RecruitmentAdapter:
    """
    Adapter between recruitment domain and core search service.
    
    Flow:
        1. Parse recruiter query → RecruitmentQuery
        2. Normalize entities → Standardized values
        3. Convert to StandardSearchRequest
        4. Execute search via SearchGateway
        5. Convert StandardSearchResponse → RecruitmentResult
    """
    
    def __init__(
        self,
        collection_name: str = "resume",  # Changed default from "candidates_v1" to "resume"
        model: str = settings.GEMINI_MODEL
    ):
        """
        Initialize Recruitment Adapter.
        
        Args:
            collection_name: ChromaDB collection to search
            model: Gemini model for query parsing
        """
        self.collection_name = collection_name
        self.model = model
        
        # Initialize components
        self.query_parser = QueryParser(model=model)
        self.normalizer = Normalizer()
        self.search_gateway = SearchGateway()
        
        log.info(f"RecruitmentAdapter initialized for collection: {collection_name}")
    
    async def search_candidates(
        self,
        raw_query: str,
        top_k: Optional[int] = None
    ) -> RecruitmentResult:
        """
        Complete candidate search flow.
        
        Args:
            raw_query: Natural language query from recruiter
            top_k: Number of candidates to return (defaults to RECRUITMENT_DEFAULT_TOP_K from settings)
            
        Returns:
            RecruitmentResult with matched candidates
            
        Example:
            >>> adapter = RecruitmentAdapter()
            >>> result = await adapter.search_candidates("Senior Python Dev in Dubai")
            >>> print(f"Found {result.total_found} candidates")
            >>> print(f"Top match: {result.matches[0].candidate_id}")
        """
        # Use configurable default if not provided
        if top_k is None:
            top_k = settings.RECRUITMENT_DEFAULT_TOP_K
        start_time = time.time()
        
        log.info(f"Starting candidate search: {raw_query}")
        
        try:
            # Step 1: Parse query (also returns normalized values → skips separate Groq calls)
            recruitment_query = await self.query_parser.parse_query(raw_query)
            log.info(f"Query parsed: role={recruitment_query.parsed_role}, location={recruitment_query.parsed_location}")
            
            # Override top_k with parsed limit when the query specifies a number (e.g. "top 3", "give me 10")
            if recruitment_query.parsed_limit is not None:
                top_k = min(recruitment_query.parsed_limit, settings.RECRUITMENT_MAX_TOP_K)
                log.info(f"Using parsed_limit from query: top_k={top_k}")
            
            # Step 2: Normalize entities (instant when parser already provided normalized values)
            normalized = await self._normalize_query(recruitment_query)
            log.info(f"Query normalized: {normalized}")
            
            # Step 3: Convert to StandardSearchRequest (fetch larger pool for LLM re-rank)
            fetch_top_k = max(top_k * RECRUITMENT_LLM_FETCH_MULTIPLIER, RECRUITMENT_LLM_MIN_POOL)
            search_request = self._to_search_request(recruitment_query, normalized, fetch_top_k)
            log.info(f"Search request created: {search_request.search_text}, filters: {len(search_request.filters)}")
            
            # Step 4: Pre-compute embedding once so fallback attempts don't re-embed
            precomputed_embedding = await self.search_gateway.embed_text(search_request.search_text)
            if precomputed_embedding:
                search_request.query_embeddings = precomputed_embedding
            
            # Step 5: Execute search with fallback logic
            search_response = await self._execute_search_with_fallback(search_request)
            log.info(f"Search complete: {len(search_response.results)} results")
            
            # Step 6: Convert to RecruitmentResult
            recruitment_result = self._to_recruitment_result(
                recruitment_query,
                search_response,
                normalized
            )
            
            # Step 7: Deduplicate by candidate (keep best vector score per candidate)
            seen: Dict[str, CandidateMatch] = {}
            for m in recruitment_result.matches:
                display_id = _display_candidate_id(m)
                if display_id not in seen or m.match_score > seen[display_id].match_score:
                    seen[display_id] = m
            deduplicated = list(seen.values())
            log.info(f"Deduplicated: {len(recruitment_result.matches)} -> {len(deduplicated)} candidates")
            
            # Step 8: LLM re-rank — skip when vector results are already high-confidence
            skip_rerank = (
                len(deduplicated) <= RECRUITMENT_RERANK_SKIP_MAX_CANDIDATES
                and len(deduplicated) > 0
                and all(m.match_score >= RECRUITMENT_RERANK_SKIP_MIN_SCORE for m in deduplicated)
            )
            if skip_rerank:
                log.info(f"Skipping LLM re-rank: {len(deduplicated)} candidates all above {RECRUITMENT_RERANK_SKIP_MIN_SCORE}")
                llm_results = None
            else:
                llm_results = await analyze_recruitment_matches_with_llm(recruitment_query, deduplicated)
            if llm_results:
                llm_by_id = {r["candidate_id"]: r for r in llm_results}
                final_matches = []
                for m in deduplicated:
                    display_id = _display_candidate_id(m)
                    if display_id not in llm_by_id:
                        continue
                    r = llm_by_id[display_id]
                    vector_score = m.match_score
                    llm_score = r["match_score"]
                    m.match_score = (
                        RECRUITMENT_VECTOR_SCORE_WEIGHT * vector_score
                        + RECRUITMENT_LLM_SCORE_WEIGHT * llm_score
                    )
                    m.rationale = r.get("rationale", "")
                    m.skills_matched = r.get("skills_matched") or []
                    final_matches.append(m)
                final_matches.sort(key=lambda x: x.match_score, reverse=True)
                final_matches = final_matches[:top_k]
                log.info(f"LLM re-rank: {len(final_matches)} candidates after filter (threshold >= {RECRUITMENT_LLM_MIN_SCORE})")
            else:
                final_matches = deduplicated
                final_matches.sort(key=lambda x: x.match_score, reverse=True)
                final_matches = final_matches[:top_k]
                log.info(f"LLM re-rank skipped or empty; using top {len(final_matches)} by vector score")
            
            # Step 9: Location post-filter (when parsed_location is set, drop non-matching candidates)
            # Use normalized location first (e.g. "Dubai") so "DXB" matches candidates with "Dubai, UAE"
            location_required = (normalized.get("location") or recruitment_query.parsed_location or "").strip()
            if location_required:
                location_term = location_required.lower()
                before_loc = len(final_matches)
                final_matches = [
                    m for m in final_matches
                    if (m.current_location or "").strip() and location_term in (m.current_location or "").lower()
                ]
                if before_loc != len(final_matches):
                    log.info(f"Location filter '{location_required}': {before_loc} -> {len(final_matches)} candidates")

            # Step 10: Seniority post-filter (when parsed_experience/seniority is set, drop non-matching candidates)
            seniority_required = _seniority_for_post_filter(normalized, recruitment_query)
            if seniority_required and seniority_required in RECRUITMENT_SENIORITY_ALLOWED:
                allowed_levels = set(RECRUITMENT_SENIORITY_ALLOWED[seniority_required])
                before_sen = len(final_matches)
                final_matches = [
                    m for m in final_matches
                    if (m.seniority_level or "").strip().lower() in allowed_levels
                ]
                if before_sen != len(final_matches):
                    log.info(f"Seniority filter '{seniority_required}': {before_sen} -> {len(final_matches)} candidates")
            
            recruitment_result.matches = final_matches
            recruitment_result.total_found = len(final_matches)
            recruitment_result.search_strategy = "vector_similarity_with_filters_llm_rerank" if llm_results else "vector_similarity_with_filters"
            
            # Calculate metrics
            recruitment_result.processing_time_ms = int((time.time() - start_time) * 1000)
            recruitment_result.calculate_metrics()
            
            log.info(f"Search complete: {recruitment_result.total_found} candidates, {recruitment_result.processing_time_ms}ms")
            return recruitment_result
            
        except Exception as e:
            log.error(f"Search failed: {e}", exc_info=True)
            # Return empty result with error
            return RecruitmentResult(
                query=RecruitmentQuery(raw_query=raw_query),
                matches=[],
                total_found=0,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def _normalize_query(self, query: RecruitmentQuery) -> Dict[str, Any]:
        """Build normalized dict from parser-provided values, falling back to parallel Groq calls."""
        normalized: Dict[str, Any] = {}

        # Collect Groq tasks only for fields the parser didn't already normalize
        tasks: list = []
        task_keys: list = []

        if query.normalized_location:
            normalized["location"] = query.normalized_location
        elif query.parsed_location:
            tasks.append(self.normalizer.normalize_location(query.parsed_location))
            task_keys.append("location")

        if query.normalized_seniority:
            normalized["seniority"] = query.normalized_seniority
        elif query.parsed_experience:
            tasks.append(self.normalizer.normalize_seniority(query.parsed_experience))
            task_keys.append("seniority")

        if query.normalized_education:
            normalized["education"] = query.normalized_education
        elif query.parsed_education:
            tasks.append(self.normalizer.normalize_education(query.parsed_education))
            task_keys.append("education")

        # Run any remaining Groq calls in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for key, result in zip(task_keys, results):
                if not isinstance(result, Exception) and result is not None:
                    normalized[key] = result

        return normalized
    
    async def _execute_search_with_fallback(
        self,
        search_request: StandardSearchRequest
    ) -> StandardSearchResponse:
        """
        Execute search with fallback logic if no results found.
        
        Strategy:
        1. Try with all filters (strict matching)
        2. If no results, try without location filter (location often has variations)
        3. If still no results, try without all filters (pure vector search)
        """
        # Try 1: With all filters
        search_response = await self.search_gateway.search(search_request)
        
        if len(search_response.results) > 0:
            log.info(f"Found {len(search_response.results)} results with all filters")
            return search_response
        
        # Reuse pre-computed embedding across fallback attempts
        _emb = search_request.query_embeddings

        # Try 2: Without location filter (if we have location filter)
        location_filters = [f for f in search_request.filters if f.field == "current_city"]
        
        if location_filters:
            log.info("No results with location filter, trying without location filter...")
            relaxed_filters = [f for f in search_request.filters if f.field != "current_city"]
            relaxed_request = StandardSearchRequest(
                target_collection=search_request.target_collection,
                search_text=search_request.search_text,
                filters=relaxed_filters,
                top_k=search_request.top_k,
                include_documents=search_request.include_documents,
                min_score=search_request.min_score,
                query_embeddings=_emb
            )
            search_response = await self.search_gateway.search(relaxed_request)
            
            if len(search_response.results) > 0:
                log.info(f"Found {len(search_response.results)} results without location filter")
                return search_response
        
        # Try 3: Without all filters (pure vector search)
        if search_request.filters:
            log.info("No results with filters, trying pure vector search...")
            no_filter_request = StandardSearchRequest(
                target_collection=search_request.target_collection,
                search_text=search_request.search_text,
                filters=[],
                top_k=search_request.top_k,
                include_documents=search_request.include_documents,
                min_score=search_request.min_score,
                query_embeddings=_emb
            )
            search_response = await self.search_gateway.search(no_filter_request)
            
            if len(search_response.results) > 0:
                log.info(f"Found {len(search_response.results)} results with fallback search")
                return search_response
        
        # No results found even without filters
        log.warning("No results found even without filters - collection may be empty or query doesn't match any candidates")
        return search_response
    
    def _to_search_request(
        self,
        query: RecruitmentQuery,
        normalized: Dict[str, Any],
        top_k: int
    ) -> StandardSearchRequest:
        """Convert RecruitmentQuery to StandardSearchRequest"""
        
        # Build search text from role and skills
        # If searching by name, prioritize name in search text
        search_parts = []
        
        # If name is provided, use it as primary search text
        if query.parsed_name:
            search_parts.append(query.parsed_name)
        
        # If university is provided, include for vector recall (post-filter by university later)
        if query.parsed_university:
            search_parts.append(query.parsed_university)
        
        # If location is provided, include for vector recall (post-filter by location later; no metadata filter - avoids strict match and fallback)
        if query.parsed_location:
            search_parts.append(query.parsed_location)
        
        if query.parsed_experience:
            search_parts.append(query.parsed_experience)
        
        if query.parsed_role:
            search_parts.append(query.parsed_role)
            # Query expansion: add role-related terms for better recall
            for term in _expand_search_text_for_role(query.parsed_role):
                if term and term not in search_parts:
                    search_parts.append(term)
        
        if query.parsed_skills:
            search_parts.extend(query.parsed_skills)
        
        search_text = " ".join(search_parts) if search_parts else query.raw_query
        
        # Build filters
        # Note: We do NOT add a location metadata filter here. ChromaDB EQUALS on current_city
        # often returns 0 (e.g. "Kerala" vs "Kerala, India"), triggering fallback. Location is
        # included in search_text for recall and enforced by post-filter (Step 9) via substring match.
        filters = []
        
        if normalized.get("seniority"):
            # Only filter by seniority if we have this metadata
            # Old resumes might not have seniority_level, so this is optional
            filters.append(SearchFilter(
                field="seniority_level",
                operator=SearchFilterOperator.EQUALS,
                value=normalized["seniority"]
            ))
        
        if normalized.get("education"):
            # Only filter by education if we have this metadata
            filters.append(SearchFilter(
                field="education_level",
                operator=SearchFilterOperator.EQUALS,
                value=normalized["education"]
            ))
        
        # NOTE: We don't use a name metadata filter here. ChromaDB doesn't support
        # substring matching in metadata (CONTAINS becomes exact equality). Partial
        # name matching (e.g. "Ashwin" matching "Ashwin Nair") is done via
        # post-filtering in _to_recruitment_result.
        
        return StandardSearchRequest(
            target_collection=self.collection_name,
            search_text=search_text,
            filters=filters,
            top_k=top_k,
            include_documents=True  # Include documents to extract candidate name
        )
    
    def _to_recruitment_result(
        self,
        query: RecruitmentQuery,
        response: StandardSearchResponse,
        normalized: Dict[str, Any]
    ) -> RecruitmentResult:
        """Convert StandardSearchResponse to RecruitmentResult"""
        import chroma as _chroma

        matches = []
        name_search_term = (query.parsed_name or "").strip().lower() or None
        university_search_term = (query.parsed_university or "").strip().lower() or None

        # Per-request cache: fetch each candidate's full resume at most once
        _resume_cache: Dict[str, Any] = {}

        def _cached_full_resume(cid: str) -> Optional[Dict]:
            if not cid:
                return None
            if cid in _resume_cache:
                return _resume_cache[cid]
            try:
                fr = _chroma.get_resume(cid)
                _resume_cache[cid] = fr
                return fr
            except Exception:
                _resume_cache[cid] = None
                return None

        def _extract_skills_from_dict(data: Dict) -> List[str]:
            out: List[str] = []
            skills_data = data.get("Skills") or data.get("skills", [])
            if isinstance(skills_data, list):
                for skill in skills_data:
                    if isinstance(skill, dict):
                        name = skill.get("SkillName") or skill.get("Name") or skill.get("name", "")
                        if name:
                            out.append(str(name).strip())
                    elif isinstance(skill, str):
                        out.append(skill.strip())
            return out

        def _extract_name_from_dict(data: Dict) -> str:
            nv = data.get("Name") or data.get("name")
            if isinstance(nv, list) and nv:
                return nv[0]
            if isinstance(nv, str):
                return nv
            if nv:
                return str(nv)
            pi = data.get("personalInformation", {})
            if isinstance(pi, dict):
                nv2 = pi.get("Name") or pi.get("name")
                if isinstance(nv2, list) and nv2:
                    return nv2[0]
                if isinstance(nv2, str):
                    return nv2
            return ""

        for result in response.results:
            metadata = result.metadata or {}

            candidate_id = metadata.get("candidate_id") or metadata.get("uid")
            if not candidate_id and "_chunk_" in result.id:
                candidate_id = result.id.split("_chunk_")[0]

            candidate_universities: List[str] = []
            if metadata.get("university"):
                candidate_universities.append(str(metadata["university"]).strip())

            # --- Skills ---
            skills_str = metadata.get("skills", "")
            candidate_skills = [s.strip() for s in skills_str.split(",") if s.strip()] if skills_str else []
            if not candidate_skills and result.document:
                try:
                    doc_data = json.loads(result.document) if isinstance(result.document, str) else result.document
                    if isinstance(doc_data, dict):
                        candidate_skills = _extract_skills_from_dict(doc_data)
                except (json.JSONDecodeError, AttributeError, TypeError):
                    pass
            if not candidate_skills:
                fr = _cached_full_resume(candidate_id)
                if fr and isinstance(fr, dict):
                    candidate_skills = _extract_skills_from_dict(fr)

            seen_skills: set = set()
            unique_skills: List[str] = []
            for skill in candidate_skills:
                sl = skill.lower().strip()
                if sl and sl not in seen_skills:
                    seen_skills.add(sl)
                    unique_skills.append(skill)
            candidate_skills = unique_skills

            # --- Seniority / education from metadata ---
            seniority_level = metadata.get("seniority_level", "") or ""
            education_level = metadata.get("education_level", "") or ""

            # --- Total experience ---
            total_exp = metadata.get("total_years_exp")
            if total_exp is None:
                total_exp_str = metadata.get("total_experience_years", "0")
                if isinstance(total_exp_str, (int, float)):
                    total_exp = float(total_exp_str)
                elif isinstance(total_exp_str, str):
                    years_match = re.search(r'(\d+\.?\d*)\s*years?', total_exp_str.lower())
                    if years_match:
                        total_exp = float(years_match.group(1))
                        months_match = re.search(r'(\d+)\s*months?', total_exp_str.lower())
                        if months_match:
                            total_exp += float(months_match.group(1)) / 12.0
                    else:
                        total_exp = 0.0
                else:
                    total_exp = 0.0
            else:
                total_exp = float(total_exp)

            current_location = metadata.get("current_city") or metadata.get("location", "")

            # --- Name ---
            candidate_name = metadata.get("name") or metadata.get("candidate_name") or metadata.get("Name") or ""
            if isinstance(candidate_name, list) and candidate_name:
                candidate_name = candidate_name[0]
            elif not isinstance(candidate_name, str):
                candidate_name = str(candidate_name) if candidate_name else ""

            if not candidate_name and result.document:
                try:
                    doc_data = json.loads(result.document) if isinstance(result.document, str) else result.document
                    if isinstance(doc_data, dict):
                        candidate_name = _extract_name_from_dict(doc_data)
                        if not current_location:
                            cd = doc_data.get("ContactDetails", {})
                            if isinstance(cd, dict):
                                current_location = cd.get("Location", "")
                except (json.JSONDecodeError, AttributeError, TypeError):
                    pass

            # Fetch full resume once (cached) for any remaining missing fields
            if not candidate_name or not seniority_level or not education_level:
                fr = _cached_full_resume(candidate_id)
                if fr and isinstance(fr, dict):
                    if not candidate_name:
                        candidate_name = _extract_name_from_dict(fr)
                    if not current_location:
                        cd = fr.get("ContactDetails", {})
                        if isinstance(cd, dict):
                            current_location = cd.get("Location", "") or ""
                        if not current_location:
                            current_location = fr.get("Location", "")
                    edu_list = fr.get("education") or fr.get("Education") or []
                    for e in edu_list:
                        if isinstance(e, dict):
                            u = e.get("university") or e.get("institution") or ""
                            if u and str(u).strip() and str(u).strip() not in candidate_universities:
                                candidate_universities.append(str(u).strip())
                    if not seniority_level:
                        exp_val = fr.get("total_experience_years", 0)
                        if isinstance(exp_val, (int, float)):
                            seniority_level = _derive_seniority_from_years(float(exp_val))
                        elif isinstance(exp_val, str) and exp_val:
                            ym = re.search(r"(\d+\.?\d*)\s*years?", exp_val.lower())
                            seniority_level = _derive_seniority_from_years(float(ym.group(1)) if ym else 0.0)
                        else:
                            seniority_level = _derive_seniority_from_years(total_exp)
                    if not education_level:
                        education_level = _derive_education_level_from_resume(fr)

            if not seniority_level:
                seniority_level = _derive_seniority_from_years(total_exp)

            if not candidate_name or not candidate_name.strip():
                candidate_name = "N/A"

            log.info(f"Final candidate_name for {result.id}: '{candidate_name}' (type: {type(candidate_name)})")

            if name_search_term and candidate_name and candidate_name != "N/A":
                if name_search_term not in str(candidate_name).lower():
                    continue

            if university_search_term:
                if not candidate_universities and candidate_id:
                    fr = _cached_full_resume(candidate_id)
                    if fr and isinstance(fr, dict):
                        edu_list = fr.get("education") or fr.get("Education") or []
                        for e in edu_list:
                            if isinstance(e, dict):
                                u = e.get("university") or e.get("institution") or ""
                                if u and str(u).strip() and str(u).strip() not in candidate_universities:
                                    candidate_universities.append(str(u).strip())
                universities_text = " ".join(str(u) for u in candidate_universities).lower()
                if university_search_term not in universities_text:
                    continue

            # Create CandidateMatch - ensure candidate_name is always a string
            match = CandidateMatch(
                candidate_id=result.id,
                candidate_name=str(candidate_name) if candidate_name else "N/A",
                candidate_skills=candidate_skills,  # All candidate skills
                match_score=result.score,
                total_experience_years=round(float(total_exp), 1),
                current_location=current_location,
                seniority_level=seniority_level or "",
                education_level=education_level or "",
                distance_score=result.distance,
                metadata=metadata
            )
            
            # Verify the name was set correctly
            if not match.candidate_name or match.candidate_name == "":
                log.warning(f"⚠️ CandidateMatch created with empty name for {result.id}, setting to 'N/A'")
                match.candidate_name = "N/A"
            
            matches.append(match)
        
        return RecruitmentResult(
            query=query,
            matches=matches,
            total_found=len(matches),
            search_strategy="vector_similarity_with_filters",
            filters_applied=normalized
        )
