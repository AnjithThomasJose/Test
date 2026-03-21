import hashlib
import json
import logging
import asyncio
import re
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pydantic import BaseModel

from chroma import query_job_descriptions, get_job_description, job_descriptions_collection, match_job_description, fetch_structured_resume, get_resume_doc, get_similar_jobs_for_job, get_top_matched_jobs_for_candidate, get_candidate_job_ranking, upsert_candidate_job_ranking
from core.model_registry import TaskType
from models.llm_invoker import invoke_llm, estimate_tokens, record_direct_llm_usage, invoke_structured_llm
from agents.ranker import compute_skill_match_score, compute_tier_from_llm_results, normalize_skill_match_from_required, sanitize_resume_for_llm
from agents.matching_utils import validate_skills_against_candidate

# ✅ Skill extraction utility (moved from skill_matcher to avoid semantic matching dependency)
from agents.skill_utils import extract_primary_skills
from agents.matching_prompts import (
    ORDERED_ANALYSIS_STEPS_UNIFIED,
    SCORING_GUIDELINES_0_100,
    CRITICAL_RULES,
    normalize_match_score_0_100_to_0_1,
)

from core.utils import _extract_json_from_response
from core.gemini_embedding_cache import cached_embed_texts
from settings import settings as _settings
from agents.prompt_generator import _calculate_resume_hash

log = logging.getLogger(__name__)


def _get_resume_hash_for_cache(structured_resume: Optional[Dict[str, Any]]) -> Optional[str]:
    """Get resume hash for cache validation. Uses _resume_hash if present, else computes."""
    if not structured_resume or not isinstance(structured_resume, dict):
        return None
    return structured_resume.get("_resume_hash") or _calculate_resume_hash(structured_resume)

# Helper function to run CPU-intensive operations in thread pool
async def run_cpu_intensive(func, *args, **kwargs):
    """
    Run CPU-intensive operations in thread pool to avoid blocking event loop.
    
    Uses the default executor (asyncio.to_thread) which is shared across all requests.
    This ensures fair resource allocation between concurrent requests (e.g., job_matcher
    and JD analysis can both make progress simultaneously).
    """
    # Use asyncio.to_thread which uses Python's default thread pool executor
    # This is shared across all async tasks, ensuring fair access
    return await asyncio.to_thread(func, *args, **kwargs)

# Configuration for LLM rationale generation
class JobMatcherConfig:
    """Configuration constants for the job_matcher agent."""
    LLM_RATIONALE_MIN_MATCH = 0.30  # Minimum match % to use LLM rationale
    LLM_RATIONALE_MAX_MATCH = 0.85  # Maximum match % to use LLM rationale
    RATIONALE_TIMEOUT = 45  # Timeout for LLM rationale generation
    CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5  # Failures before opening
    CIRCUIT_BREAKER_TIMEOUT = 60  # Seconds before half-open
    MAX_JOBS_TO_PROCESS = 200  # Maximum number of jobs to retrieve from vector search (optimized: use vector search instead of fetching all)
    EARLY_FILTER_THRESHOLD = 0.10  # Minimum skill match to process (10%) - lowered to allow fuzzy matching to work properly
    RATIONALE_TOP_K = 10  # Generate rationale only for top 10 matches (reduced from 15 for performance)
    VECTOR_SIMILARITY_THRESHOLD = 0.30  # Minimum vector similarity to process job (filter in DB, not in Python)
    MIN_SKILL_MATCH_PERCENTAGE = 50  # Minimum skill match % to consider a job a good match (contextual quality bar)
    LLM_OPTIMISTIC_THRESHOLD = 0.15  # Only scale down when LLM score exceeds skill match by this margin (avoids double penalty)

# Circuit breaker for LLM calls (shared pattern with ranker)
class LLMCircuitBreaker:
    """Circuit breaker pattern for LLM calls to prevent cascading failures."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self._lock = asyncio.Lock()
        self.total_calls = 0
        self.blocked_calls = 0
    
    async def can_proceed(self) -> bool:
        """Check if LLM calls can proceed."""
        async with self._lock:
            self.total_calls += 1
            if self.state == "closed":
                return True
            elif self.state == "open":
                # Check if timeout has passed
                if self.last_failure_time and \
                   time.time() - self.last_failure_time > self.timeout:
                    self.state = "half_open"
                    log.info("🔄 Circuit breaker: Moving to half-open state")
                    return True
                self.blocked_calls += 1
                return False
            else:  # half_open
                return True
    
    async def record_success(self):
        """Record successful LLM call."""
        async with self._lock:
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                log.info("✅ Circuit breaker: Closed (recovered)")
            elif self.state == "closed":
                # Reset failure count on success
                self.failure_count = 0
    
    async def record_failure(self):
        """Record failed LLM call."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                log.warning(
                    f"🚫 Circuit breaker OPEN: {self.failure_count} failures "
                    f"(threshold: {self.failure_threshold})"
                )
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "total_calls": self.total_calls,
            "blocked_calls": self.blocked_calls
        }

# Global circuit breaker instance
_llm_circuit_breaker = LLMCircuitBreaker(
    failure_threshold=JobMatcherConfig.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    timeout=JobMatcherConfig.CIRCUIT_BREAKER_TIMEOUT
)

# Semaphore to limit concurrent LLM calls (prevents API rate limiting)
# Lazy initialization to avoid event loop issues
_llm_semaphore: Optional[asyncio.Semaphore] = None

def _get_llm_semaphore() -> asyncio.Semaphore:
    """Get or create LLM semaphore (lazy initialization)."""
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(5)  # Max 5 concurrent LLM calls
    return _llm_semaphore

def extract_company_from_description(full_description: str) -> str:
    """Extract company name from full job description text"""
    if not full_description:
        return None
        
    # Look for "Company: [Name]" pattern - stop at next field
    company_match = re.search(r'Company:\s*([^L]+?)(?:\s+Location:|$)', full_description, re.IGNORECASE)
    if company_match:
        return company_match.group(1).strip()
    
    # Look for other common patterns
    patterns = [
        r'at\s+([A-Z][A-Za-z\s&]+(?:Bank|Corp|Inc|Ltd|LLC|Company|Technologies|Solutions))',
        r'([A-Z][A-Za-z\s&]+(?:Bank|Corp|Inc|Ltd|LLC|Company|Technologies|Solutions))\s+is\s+seeking',
        r'([A-Z][A-Za-z\s&]+(?:Bank|Corp|Inc|Ltd|LLC|Company|Technologies|Solutions))\s+Location:',
        r'([A-Z][A-Za-z\s&]+(?:Bank|Corp|Inc|Ltd|LLC|Company|Technologies|Solutions))\s+Job Type:'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, full_description, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


def extract_job_title_from_description(full_description: str) -> str:
    """Extract full job title from description"""
    if not full_description:
        return None
        
    # Look for "Job Title: [Full Title]" pattern - stop at next field
    title_match = re.search(r'Job Title:\s*(.+?)(?:\s+Company:|$)', full_description, re.IGNORECASE)
    if title_match:
        return title_match.group(1).strip()
    
    # Look for other patterns
    patterns = [
        r'Position:\s*([^L]+?)(?:\s+Location:|$)',
        r'Role:\s*([^L]+?)(?:\s+Location:|$)',
        r'Hiring for\s*([^L]+?)(?:\s+Location:|$)',
        r'Looking for\s*([^L]+?)(?:\s+Location:|$)',
        r'Seeking\s*([^L]+?)(?:\s+Location:|$)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, full_description, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


@dataclass
class JobMatchResult:
    job_id: str
    job_title: str
    company_name: str
    location: str
    required_experience: str
    matched_experience: str
    match_score: float
    matched_skills: List[str]
    matched_education: List[str]
    rationale: str  # ✅ NEW: AI-generated explanation


def flatten_resume_fields(resume: Dict[str, Any]) -> str:
    """
    Convert structured resume fields into a single searchable text.
    Optimized for better semantic matching with job descriptions.
    """
    parts = []
    
    # Add name for context
    if resume.get("Name"):
        parts.append(f"Candidate: {resume['Name']}")
    
    # Add skills with emphasis (most important for matching)
    skills = resume.get("Skills", [])
    if skills:
        skill_text = "Skills: "
        if isinstance(skills, list):
            skill_list = []
            for s in skills:
                if isinstance(s, dict):
                    skill_list.extend(str(v) for v in s.values() if v)
                elif s:
                    skill_list.append(str(s))
            skill_text += ", ".join(skill_list)
        else:
            skill_text += str(skills)
        parts.append(skill_text)
    
    # Add work experience with details
    work_exp = resume.get("WorkExperience", [])
    if work_exp:
        exp_text = "Experience: "
        exp_parts = []
        for job in work_exp:
            if isinstance(job, dict):
                title = job.get("title", job.get("position", ""))
                company = job.get("company", "")
                description = job.get("description", "")
                if title:
                    exp_parts.append(f"{title} at {company}" if company else title)
                if description:
                    exp_parts.append(description)
            elif job:
                exp_parts.append(str(job))
        exp_text += " | ".join(exp_parts)
        parts.append(exp_text)
    
    # Add education
    education = resume.get("Education", [])
    if education:
        edu_text = "Education: "
        edu_parts = []
        for edu in education:
            if isinstance(edu, dict):
                degree = edu.get("degree", "")
                major = edu.get("major", "")
                university = edu.get("university", edu.get("institution", ""))
                if degree or major:
                    edu_parts.append(f"{degree} {major}".strip())
                if university:
                    edu_parts.append(university)
            elif edu:
                edu_parts.append(str(edu))
        edu_text += " | ".join(edu_parts)
        parts.append(edu_text)
    
    # Add certifications if available
    certs = resume.get("certifications", [])
    if certs:
        cert_text = "Certifications: "
        cert_list = []
        for cert in certs:
            if isinstance(cert, dict):
                cert_list.extend(str(v) for v in cert.values() if v)
            elif cert:
                cert_list.append(str(cert))
        cert_text += ", ".join(cert_list)
        parts.append(cert_text)
    
    return " ".join(parts)


async def extract_matched_skills(resume: Dict[str, Any], jd_data: Dict[str, Any]) -> List[str]:
    """
    Enhanced skill matching using the same logic as ranker.
    Extracts and matches skills between resume and job description.
    Returns only matched skills (for backward compatibility).
    """
    matched_skills, _ = await extract_matched_and_unmatched_skills(resume, jd_data)
    return matched_skills


async def extract_matched_and_unmatched_skills(resume: Dict[str, Any], jd_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    ⚠️ DEPRECATED: This function is not used by V2 agents (job_matcher_agent).
    V2 agents use LLM-based skill matching instead.
    
    Legacy function for skill matching. This would fail if called as skill_matcher was removed.
    Kept for backward compatibility but should not be used.
    """
    log.warning("⚠️ extract_matched_and_unmatched_skills is deprecated and will fail - skill_matcher was removed")
    raise NotImplementedError(
        "extract_matched_and_unmatched_skills is deprecated. "
        "V2 agents use LLM-based matching. Use job_matcher_agent() instead."
    )


def get_all_job_descriptions_from_chroma() -> List[Dict[str, Any]]:
    """
    Retrieve all job descriptions from ChromaDB.
    Returns a list of job description dictionaries.
    """
    try:
        log.info("🔍 Retrieving all job descriptions from ChromaDB...")
        
        # DEBUG: Log database configuration to verify correct DB
        try:
            from settings import settings
            log.debug(f"ChromaDB Configuration:")
            log.debug(f"   Database: {settings.CHROMA_DATABASE}")
            log.debug(f"   Tenant: {settings.CHROMA_TENANT}")
            log.debug(f"   Collection name: job_descriptions")
        except ImportError:
            log.warning("⚠️ Could not import settings for debugging")
        
        # DEBUG: Verify collection exists and get count
        collection_count = 0
        try:
            collection_count = job_descriptions_collection.count()
            log.debug(f"Collection count: {collection_count} documents")
        except Exception as e:
            log.warning(f"⚠️ Could not get collection count: {e}")
        
        # Try multiple methods to retrieve documents
        results = None
        
        # Method 1: Try get() without parameters (should get all)
        try:
            results = job_descriptions_collection.get()
            if results and results.get('ids'):
                log.debug(f"Method 1 (get()): Found {len(results.get('ids', []))} documents")
            else:
                log.warning(f"Method 1 (get()): Returned empty results")
        except Exception as e:
            log.warning(f"Method 1 (get()) failed: {e}")
        
        # Method 2: If count > 0 but get() returned empty, try with explicit limit
        if (not results or not results.get('ids')) and collection_count > 0:
            try:
                log.debug(f"Trying Method 2: get() with limit={collection_count}")
                results = job_descriptions_collection.get(limit=collection_count)
                if results and results.get('ids'):
                    log.debug(f"Method 2 (get(limit={collection_count})): Found {len(results.get('ids', []))} documents")
                else:
                    log.warning(f"Method 2: Still returned empty results")
            except Exception as e:
                log.warning(f"Method 2 failed: {e}")
        
        # Method 3: Try peek() to see if collection has any data
        if not results or not results.get('ids'):
            try:
                log.debug(f"Trying Method 3: peek() to check collection")
                peek_results = job_descriptions_collection.peek(limit=100)
                if peek_results and peek_results.get('ids'):
                    log.debug(f"Method 3 (peek): Found {len(peek_results.get('ids', []))} sample documents")
                    results = peek_results
                else:
                    log.warning(f"Method 3 (peek): No documents found")
            except Exception as e:
                log.warning(f"Method 3 (peek) failed: {e}")
        
        # Method 4: Try querying with a generic query to find any jobs
        if not results or not results.get('ids'):
            try:
                log.debug(f"Trying Method 4: query() to search for jobs")
                query_results = job_descriptions_collection.query(
                    query_texts=["job description"],
                    n_results=100
                )
                if query_results and query_results.get('ids'):
                    # Extract IDs from query results
                    ids = query_results['ids']
                    if isinstance(ids[0], list):
                        ids = ids[0]
                    
                    if ids:
                        # Get the actual documents using the IDs from query
                        results = job_descriptions_collection.get(ids=ids)
                        log.debug(f"Method 4 (query): Found {len(ids)} documents via query")
                else:
                    log.warning(f"Method 4 (query): No documents found")
            except Exception as e:
                log.warning(f"Method 4 (query) failed: {e}")
        
        if not results or not results.get('ids'):
            log.warning("⚠️ No job descriptions found in ChromaDB after trying all methods")
            log.warning(f"   Collection count reported: {collection_count}")
            log.warning(f"   Results keys: {list(results.keys()) if results else 'None'}")
            return []
        
        log.info(f"✅ Retrieved {len(results['ids'])} job description IDs")
        if results['ids']:
            log.debug(f"Sample IDs: {results['ids'][:5]}")
        
        job_descriptions = []
        for i, doc_id in enumerate(results['ids']):
            try:
                # Parse the JSON document
                jd_data = json.loads(results['documents'][i])
                
                # IMPORTANT: Use the existing job_id from ChromaDB (document ID)
                # Also check metadata if available
                metadata = results.get('metadatas', [])
                if metadata and i < len(metadata):
                    # Prefer job_id from metadata, fallback to document ID
                    jd_data['job_id'] = metadata[i].get('job_id', doc_id)
                else:
                    jd_data['job_id'] = doc_id
                
                # Also set jd_id and id for backward compatibility
                jd_data['jd_id'] = jd_data['job_id']
                jd_data['id'] = jd_data['job_id']
                
                job_descriptions.append(jd_data)
                
                # DEBUG: Log first job details
                if i == 0:
                    log.debug(f"First job sample:")
                    log.debug(f"   ID: {jd_data.get('job_id', doc_id)}")
                    log.debug(f"   Title: {jd_data.get('jobTitle', 'N/A')}")
                    log.debug(f"   Company: {jd_data.get('company', 'N/A')}")
                    log.debug(f"   Skills: {jd_data.get('requiredSkills', [])[:3]}")
                    
            except json.JSONDecodeError as e:
                log.warning(f"⚠️ Failed to parse job description {doc_id}: {e}")
                log.warning(f"   Document preview: {results['documents'][i][:200]}...")
                continue
        
        log.info(f"✅ Successfully parsed {len(job_descriptions)} job descriptions")
        return job_descriptions
        
    except Exception as e:
        log.error(f"❌ Failed to retrieve job descriptions from ChromaDB: {e}")
        import traceback
        log.error(f"❌ Traceback: {traceback.format_exc()}")
        return []


def _normalize_llm_domain(domain: str) -> str:
    """
    Normalize LLM-returned domain (uppercase with underscores) to lowercase format.
    
    Args:
        domain: Domain string from LLM (e.g., "SALES_BUSINESS", "SOFTWARE_ENGINEERING")
        
    Returns:
        Normalized lowercase domain (e.g., "sales", "software_engineering")
    """
    # Mapping from LLM uppercase format to lowercase format
    domain_mapping = {
        "SOFTWARE_ENGINEERING": "software_engineering",
        "DATA_ANALYTICS": "data_analytics",
        "DATA_SCIENCE": "data_science",
        "AI_ML": "ai_ml",
        "TECHNICAL_INFRASTRUCTURE": "technical_infrastructure",
        "TRADITIONAL_ENGINEERING": "mechanical",  # Map to primary traditional eng
        "BUSINESS_STRATEGIC": "management",
        "SALES_BUSINESS": "sales",
        "MARKETING": "marketing",
        "HR": "hr",
        "FINANCE": "finance",
        "LEGAL": "legal",
        "MANAGEMENT": "management",
        "HEALTHCARE": "healthcare",
        "UI_UX_DESIGN": "ui_ux_design",
        "CREATIVE_DESIGN": "creative_design",
        "MANUFACTURING": "manufacturing",
        "SUPPLY_CHAIN": "supply_chain",
        "OPERATIONS": "operations",
        "CUSTOMER_SUPPORT": "customer_support",
        "CUSTOMER_SUCCESS": "customer_success",
        "ADMIN": "admin",
        "EDUCATION": "education",
        "SCIENCE": "science",
        "CONSTRUCTION": "construction",
        "CONSULTING": "consulting",
        "HOSPITALITY": "hospitality",
        "RETAIL": "retail",
        "TRADES": "trades",
        "REAL_ESTATE": "real_estate",
        "FITNESS": "fitness",
        "BEAUTY": "beauty",
        "AVIATION": "aviation",
        "TRANSPORTATION": "transportation",
        "MILITARY": "military",
        "NONPROFIT": "nonprofit",
        "PUBLIC_SECTOR": "public_sector",
        "ARTS": "arts",
        "SPORTS": "sports",
        "ENVIRONMENTAL": "environmental",
        "SECURITY": "security",
        "CLEANING_JANITORIAL": "cleaning_janitorial",
        "PERSONAL_CARE": "personal_care",
        "CHILDCARE": "childcare",
        "OTHER": "other",
    }
    
    domain_upper = domain.upper()
    return domain_mapping.get(domain_upper, domain.lower())


# Cache for domain classification by resume hash (reduces embedding API calls)
_DOMAIN_CACHE_MAX = 2000
_resume_domain_cache: Dict[str, Tuple[str, float]] = {}


def _evict_cache_if_needed(cache: Dict[str, Tuple[str, float]], max_size: int = _DOMAIN_CACHE_MAX) -> None:
    """Evict oldest half of cache when over max size."""
    if len(cache) >= max_size:
        keys_to_remove = list(cache.keys())[: max_size // 2]
        for k in keys_to_remove:
            cache.pop(k, None)
        log.debug(f"Domain cache evicted {len(keys_to_remove)} entries (size was {max_size})")


async def _classify_domain_with_embeddings(resume: Dict[str, Any]) -> Tuple[str, float]:
    """
    Semantic embedding-based domain classification using Google Gemini embeddings.
    
    Creates embeddings for domain descriptions and resume, then finds best match
    using cosine similarity. Results are cached by resume content hash to reduce API calls.
    
    Returns:
        Tuple of (domain, confidence) where domain is lowercase and confidence is 0.0-1.0
    """
    try:
        from google import genai
        from settings import settings
        import numpy as np
        
        # Build resume summary text (same as below, for cache key)
        resume_parts = []
        skills = resume.get("Skills", []) or resume.get("skills", [])
        if isinstance(skills, list):
            for skill in skills[:30]:
                if isinstance(skill, dict):
                    skill_name = skill.get("SkillName") or skill.get("name", "")
                    if skill_name:
                        resume_parts.append(str(skill_name))
                elif isinstance(skill, str):
                    resume_parts.append(skill)
        experience = resume.get("WorkExperience", []) or resume.get("experience", [])
        if isinstance(experience, list):
            for exp in experience[:10]:
                if isinstance(exp, dict):
                    title = exp.get("JobTitle") or exp.get("job_title") or exp.get("title", "")
                    if title:
                        resume_parts.append(str(title))
                    resp = exp.get("Responsibilities") or exp.get("responsibilities", [])
                    if isinstance(resp, list):
                        resume_parts.extend([str(r) for r in resp[:3]])
                    elif isinstance(resp, str):
                        resume_parts.append(resp)
        summary = resume.get("Summary") or resume.get("summary", "")
        if summary:
            resume_parts.append(str(summary))
        education = resume.get("Education", []) or resume.get("education", [])
        if isinstance(education, list):
            for edu in education[:3]:
                if isinstance(edu, dict):
                    degree = edu.get("Degree") or edu.get("degree") or edu.get("major", "")
                    if degree:
                        resume_parts.append(str(degree))
        resume_text = " ".join(resume_parts)[:3000]
        if not resume_text.strip():
            log.warning("No meaningful text extracted from resume for domain classification")
            return "OTHER", 0.0

        # Check cache before embedding
        cache_key = hashlib.sha256(resume_text.encode("utf-8")).hexdigest()
        if cache_key in _resume_domain_cache:
            cached = _resume_domain_cache[cache_key]
            log.debug(f"Resume domain cache HIT: {cached[0]} (confidence: {cached[1]:.3f})")
            return cached[0], cached[1]

        # Get API key
        api_key = settings.GOOGLE_API_KEY
        if not api_key:
            log.warning("Google API key not found, falling back to OTHER domain")
            return "OTHER", 0.0
        
        client = genai.Client(api_key=api_key)
        
        # Define domain descriptions (semantic, not keywords) - resume_text already built above for cache key
        domain_descriptions = {
            "software_engineering": "Software development, programming, web development, mobile apps, frontend, backend, full-stack development, coding, computer science, software engineering",
            "data_analytics": "Business intelligence, data analysis, reporting, dashboards, data visualization, analytics tools like Tableau and Power BI, SQL queries and reporting",
            "data_science": "Data science, machine learning model development, statistical analysis, data mining, big data processing with Python, R, Spark, Hadoop",
            "ai_ml": "Artificial intelligence, machine learning engineering, deep learning, neural networks, LLMs, AI model development with TensorFlow, PyTorch, transformers",
            "technical_infrastructure": "DevOps, cloud engineering, system administration, SRE, platform engineering, infrastructure automation, CI/CD pipelines, containerization",
            "traditional_engineering": "Mechanical engineering, electrical engineering, civil engineering, chemical engineering, CAD design, circuit design, manufacturing processes",
            "business_strategic": "Product management, business analysis, strategy consulting, project management, business planning, roadmap development",
            "sales_business": "Sales, account management, business development, client relations, revenue generation, deal closing, sales strategy",
            "marketing": "Digital marketing, SEO, SEM, content marketing, brand management, social media marketing, advertising campaigns",
            "hr": "Human resources, talent acquisition, recruitment, employee relations, HR management, hiring, onboarding",
            "finance": "Finance, accounting, auditing, financial analysis, tax preparation, investment banking, financial planning",
            "legal": "Legal services, attorney, lawyer, compliance, corporate law, litigation, legal counsel",
            "management": "Executive leadership, general management, operations management, strategic management, C-suite roles",
            "healthcare": "Medical services, healthcare, nursing, clinical care, patient care, hospital administration, medical practice",
            "ui_ux_design": "User interface design, user experience design, UX research, prototyping, design systems, Figma, Sketch",
            "creative_design": "Graphic design, video editing, animation, photography, art direction, creative services",
            "manufacturing": "Production, assembly line, machining, quality control, lean manufacturing, factory operations",
            "supply_chain": "Supply chain management, logistics, procurement, warehouse operations, distribution, inventory management",
            "operations": "Business operations, operational excellence, process improvement, workflow optimization, operations management",
            "customer_support": "Customer service, help desk, technical support, customer care, support ticketing systems",
            "customer_success": "Customer success management, client retention, product adoption, account health, renewals",
            "admin": "Administrative assistant, office management, executive assistant, receptionist, clerical work",
            "education": "Teaching, educational services, training, academic instruction, curriculum development",
            "science": "Research scientist, laboratory work, R&D, biology, chemistry, biotechnology, pharmaceutical research",
            "construction": "Construction management, site supervision, civil engineering projects, facilities management, safety",
            "consulting": "Management consulting, strategy consulting, business advisory, transformation consulting",
            "hospitality": "Hotel management, restaurant service, culinary arts, food service, event planning, tourism, bartending, housekeeping",
            "retail": "Retail sales, store management, merchandising, e-commerce, customer service in retail",
            "trades": "Skilled trades, plumbing, electrical work, carpentry, welding, HVAC technician",
            "real_estate": "Real estate agent, broker, property management, leasing, real estate sales",
            "fitness": "Personal training, fitness coaching, nutrition, wellness coaching, exercise physiology",
            "beauty": "Cosmetology, esthetics, hairstyling, makeup artistry, skincare, salon services",
            "aviation": "Pilot, flight attendant, air traffic control, aircraft maintenance, aviation operations",
            "transportation": "Truck driving, delivery driver, logistics driving, fleet management, transportation",
            "military": "Military service, armed forces, defense, veterans, security clearance",
            "nonprofit": "Nonprofit organization, NGO, fundraising, grant writing, volunteer coordination",
            "public_sector": "Government, public administration, policy development, regulatory compliance",
            "arts": "Performing arts, theater, dance, music performance, fine arts, museum curation",
            "sports": "Athletic coaching, sports training, sports management, sports medicine",
            "environmental": "Environmental science, conservation, ecology, sustainability, wildlife management",
            "security": "Security guard, security officer, loss prevention, surveillance, access control, patrol services",
            "cleaning_janitorial": "Janitorial services, cleaning, custodial work, housekeeping, sanitation, facility cleaning",
            "personal_care": "Personal care aide, caregiver, home health aide, elderly care, disability support, assisted living",
            "childcare": "Childcare, daycare, nanny, babysitting, early childhood education, preschool teaching"
        }
        
        # Prepare texts for embedding: resume + all domain descriptions (resume_text from cache key above)
        all_texts = [resume_text] + list(domain_descriptions.values())
        
        # Batch embed all texts (cached across restarts via gemini_embedding_cache)
        # Issue 4.1: Wrap synchronous Gemini call in asyncio.to_thread to avoid blocking event loop
        log.debug(f"Embedding resume + {len(domain_descriptions)} domain descriptions...")
        import asyncio
        
        def _sync_embed():
            return cached_embed_texts(client, "models/gemini-embedding-001", all_texts)
        
        # Run in thread pool to avoid blocking the event loop
        try:
            loop = asyncio.get_running_loop()
            embeddings = await asyncio.to_thread(_sync_embed)
        except RuntimeError:
            # No running event loop, call directly (sync context)
            embeddings = _sync_embed()
        
        if not embeddings:
            log.error("Unexpected embedding response format")
            return "OTHER", 0.0
        
        if len(embeddings) != len(all_texts):
            log.error(f"Embedding count mismatch: expected {len(all_texts)}, got {len(embeddings)}")
            return "OTHER", 0.0
        
        # Convert to numpy arrays
        resume_embedding = np.array(embeddings[0])
        domain_embeddings = np.array(embeddings[1:])
        
        # Normalize
        resume_norm = np.linalg.norm(resume_embedding)
        if resume_norm > 0:
            resume_embedding = resume_embedding / resume_norm
        
        domain_norms = np.linalg.norm(domain_embeddings, axis=1, keepdims=True)
        domain_embeddings = np.where(domain_norms > 0, domain_embeddings / domain_norms, domain_embeddings)
        
        # Calculate cosine similarities
        similarities = np.dot(domain_embeddings, resume_embedding)
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = float(similarities[best_idx])
        domain_names = list(domain_descriptions.keys())
        best_domain = domain_names[best_idx]
        
        log.info(f"✅ Semantic domain classification: {best_domain} (confidence: {best_score:.3f})")
        log.debug(f"   Top 3 domains: {[(domain_names[i], float(similarities[i])) for i in np.argsort(similarities)[-3:][::-1]]}")
        
        # Store in cache for future requests with same resume
        _evict_cache_if_needed(_resume_domain_cache)
        _resume_domain_cache[cache_key] = (best_domain, best_score)
        
        return best_domain, best_score
        
    except ImportError:
        log.error("Google GenAI package not installed")
        return "OTHER", 0.0
    except Exception as e:
        log.error(f"❌ Semantic domain classification failed: {e}")
        import traceback
        log.error(f"Traceback: {traceback.format_exc()}")
        return "OTHER", 0.0


def _get_domain_context_deterministic(resume: Dict[str, Any]) -> Tuple[str, float]:
    """
    DEPRECATED: Deterministic keyword-based domain detection for resumes.
    Kept for reference only. Use classify_candidate_domain() instead.
    
    Returns:
        Tuple of (domain, confidence) where domain is lowercase and confidence is 0.0-1.0
    """
    # Extract skills and experience from resume
    skills = resume.get("Skills", [])
    experience = resume.get("WorkExperience", []) or resume.get("experience", [])
    
    # Normalize skills to strings
    skills_text = []
    for skill in skills:
        if isinstance(skill, dict):
            skill_name = skill.get("SkillName") or skill.get("Name") or skill.get("name", "")
            if skill_name:
                skills_text.append(str(skill_name).lower())
        elif isinstance(skill, str):
            skills_text.append(skill.lower())
    
    # Extract job titles from experience
    titles = []
    for exp in experience:
        if isinstance(exp, dict):
            title = exp.get("JobTitle") or exp.get("jobTitle") or exp.get("title") or exp.get("Title", "")
            if title:
                titles.append(str(title).lower())
    
    all_text = " ".join(skills_text + titles).lower()
    
    # Import domain keywords from ranker's JobDescription class
    # We'll create a helper to get the domain keywords
    # For now, let's use a simplified version that matches ranker's structure
    # In production, you might want to share this mapping between files
    
    # Domain keywords mapping - Comprehensive coverage matching ranker.py
    domain_keywords = {
        "software_engineering": [
            "software engineer", "developer", "programmer", "coding", "computer science",
            "full stack", "backend", "frontend", "sdlc", "agile", "scrum",
            "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "php", "ruby", "swift", "kotlin",
            "react", "angular", "vue", "node.js", "django", "spring boot", ".net", "html", "css",
            "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform", "ci/cd", "devops",
            "ios", "android", "react native", "flutter",
            "sql", "nosql", "postgresql", "mongodb", "redis"
        ],
        "data_analytics": [
            "data analyst", "business intelligence", "bi", "reporting", "dashboards",
            "tableau", "power bi", "qlik", "looker", "excel", "sql", "etl",
            "data visualization", "metrics", "kpi", "analytics", "insights", "reporting analyst"
        ],
        "data_science": [
            "data scientist", "statistics", "statistical analysis", "hypothesis testing",
            "pandas", "numpy", "scipy", "r", "jupyter", "data analysis", "exploratory data analysis",
            "eda", "regression", "classification", "clustering", "data mining", "big data", "spark", "hadoop"
        ],
        "ai_ml": [
            "machine learning", "ml", "artificial intelligence", "ai", "deep learning",
            "neural networks", "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost",
            "nlp", "natural language processing", "computer vision", "cv", "opencv",
            "llm", "large language models", "gpt", "transformer", "bert", "llama",
            "reinforcement learning", "rl", "mlops", "model deployment", "model training",
            "feature engineering", "model evaluation", "ai engineer", "ml engineer",
            "prompt engineering", "langchain", "hugging face", "stable diffusion", "midjourney",
            "generative ai", "genai", "chatgpt", "claude", "anthropic"
        ],
        "product": [
            "product manager", "product owner", "pm", "product strategy", "roadmap", 
            "user stories", "backlog", "jira", "confluence", "prd", "go-to-market", "kpi",
            "a/b testing", "product lifecycle", "stakeholder management"
        ],
        "mechanical": [
            "mechanical engineer", "mechanical design", "mechatronics", "robotics", 
            "cad", "solidworks", "catia", "autocad", "creo", "ansys", "pro-e",
            "thermodynamics", "fluid mechanics", "hydraulics", "pneumatics", "hvac",
            "fea", "finite element analysis", "cfd", "gd&t", "geometric dimensioning",
            "manufacturing engineering", "prototyping", "3d printing", "automotive", "aerospace"
        ],
        "electrical": [
            "electrical engineer", "electronics", "circuit design", "pcb", "schematic",
            "altium", "orcad", "eagle", "fpga", "verilog", "vhdl", "microcontrollers",
            "embedded systems", "plc", "scada", "power systems", "high voltage",
            "control systems", "instrumentation", "rf", "signal processing", "iot"
        ],
        "civil": [
            "civil engineer", "structural engineer", "geotechnical", "environmental engineering",
            "construction management", "infrastructure", "autocad civil 3d", "revit", "bim", 
            "staad", "etabs", "surveying", "land development", "hydrology", 
            "urban planning", "concrete", "steel structures", "bridge design", "transportation"
        ],
        "chemical": [
            "chemical engineer", "process engineer", "process simulation", "aspen hysys",
            "aspen plus", "matlab", "polymers", "material science", "metallurgy",
            "petrochemical", "refinery", "distillation", "reaction engineering",
            "mass transfer", "heat transfer", "chromatography", "quality control"
        ],
        "sales": [
            "sales", "business development", "account executive", "sdr", "bdr",
            "account management", "enterprise sales", "saas sales", "revenue", "quota",
            "prospecting", "lead generation", "closing", "negotiation", "crm", "salesforce",
            "cold calling", "pipeline management", "forecasting"
        ],
        "marketing": [
            "marketing", "digital marketing", "seo", "sem", "content marketing",
            "social media", "brand management", "email marketing", "growth hacking",
            "google ads", "analytics", "copywriting", "public relations", "pr",
            "campaign management", "market research"
        ],
        "finance": [
            "finance", "accounting", "auditing", "financial analyst", "cpa", "tax",
            "bookkeeping", "payroll", "fp&a", "forecasting", "budgeting", "compliance",
            "gaap", "ifrs", "investment banking", "private equity", "excel"
        ],
        "hr": [
            "human resources", "hr", "recruiter", "talent acquisition", "hrbp",
            "employee relations", "benefits", "compensation", "onboarding", 
            "performance management", "learning and development", "hris", "workday"
        ],
        "legal": [
            "legal", "attorney", "lawyer", "paralegal", "general counsel", "litigation",
            "corporate law", "contracts", "compliance", "regulatory", "intellectual property",
            "gdpr", "privacy", "employment law", "mergers and acquisitions"
        ],
        "management": [
            "ceo", "cto", "cfo", "coo", "vice president", "vp", "director", "head of",
            "general manager", "operations manager", "strategy", "leadership", 
            "p&l", "budget management", "change management", "organizational development"
        ],
        "healthcare": [
            "healthcare", "medical", "nursing", "rn", "doctor", "physician", 
            "clinical", "patient care", "hospital", "pharmacy", "public health",
            "medical records", "emr", "ehr", "health administration"
        ],
        "ui_ux_design": [
            "ui/ux", "user interface", "user experience", "ux design", "ui design",
            "interaction design", "ux research", "user research", "usability testing",
            "wireframing", "prototyping", "figma", "sketch", "adobe xd", "invision",
            "design systems", "information architecture", "user flows", "personas"
        ],
        "creative_design": [
            "graphic design", "visual design", "web design", "video editing", "animation",
            "motion graphics", "photography", "videography", "art director", 
            "creative director", "adobe creative suite", "photoshop", "illustrator",
            "premiere pro", "after effects", "indesign", "branding", "print design"
        ],
        "manufacturing": [
            "manufacturing", "production", "plant manager", "assembly", "machining",
            "lean manufacturing", "six sigma", "kaizen", "quality assurance", "qa",
            "production planning", "manufacturing engineer", "production supervisor",
            "cnc", "fabrication", "welding", "machining", "assembly line"
        ],
        "supply_chain": [
            "supply chain", "logistics", "procurement", "warehouse",
            "inventory", "distribution", "sourcing", "vendor management",
            "purchasing", "supply chain management", "logistics coordinator"
        ],
        "operations": [
            "operations", "operations manager", "business operations", "operational excellence",
            "process improvement", "operations analyst", "business process", "workflow"
        ],
        "customer_support": [
            "customer service", "support", "help desk", "technical support", 
            "troubleshooting", "ticketing", "zendesk", "intercom", "freshdesk",
            "customer support", "support engineer", "support specialist", "it support",
            "call center", "contact center", "live chat", "email support"
        ],
        "customer_success": [
            "customer success", "csm", "customer success manager", "onboarding", 
            "implementation", "retention", "churn reduction", "client success", 
            "account health", "qbr", "quarterly business review", "adoption", "renewals",
            "account management", "customer advocacy", "expansion", "upsell"
        ],
        "admin": [
            "administrative", "admin", "office manager", "executive assistant", "ea", 
            "receptionist", "data entry", "clerical", "secretary", "scheduling", 
            "travel arrangements", "office support", "calendar management"
        ],
        "education": [
            "education", "teacher", "teaching", "tutor", "curriculum", "instructional design",
            "training", "l&d", "learning and development", "faculty", "academic", 
            "professor", "k-12", "higher education"
        ],
        "science": [
            "science", "research", "biology", "chemistry", "laboratory", "lab", 
            "scientist", "physics", "biotech", "r&d", "clinical trials", "genomics",
            "microbiology", "pharmaceutical", "research and development"
        ],
        "construction": [
            "construction", "civil engineer", "site manager", "project manager", 
            "estimator", "surveyor", "architecture", "real estate", "property manager",
            "facilities", "blueprint", "safety", "osha"
        ],
        "consulting": [
            "consultant", "consulting", "management consulting", "strategy", "advisor",
            "due diligence", "mergers and acquisitions", "m&a", "transformation",
            "business process", "frameworks", "risk management", "advisory",
            "client delivery", "engagement manager", "principal"
        ],
        "hospitality": [
            "hospitality", "hotel", "restaurant", "culinary", "chef", "food service",
            "event planning", "catering", "banquet", "front desk", "concierge",
            "hospitality management", "tourism", "travel",
            # Food & Beverage Service
            "tea maker", "coffee maker", "beverage service", "beverage preparation",
            "pantry", "pantry assistant", "pantry helper", "kitchen helper",
            "kitchen assistant", "food preparation", "food service assistant",
            "cafeteria", "canteen", "dining service", "wait staff", "server",
            "waiter", "waitress", "barista", "bartender", "food handler", 
            "hygiene", "cleanliness", "host", "hostess", "busser", "busboy",
            "dishwasher", "line cook", "prep cook", "sous chef", "pastry chef",
            "food runner", "room service", "banquet server", "catering staff",
            # Hotel Services
            "housekeeping", "housekeeper", "laundry", "laundry attendant",
            "maintenance", "hotel maintenance", "bellhop", "valet", "parking attendant",
            "guest services", "reservations", "front office", "night auditor"
        ],
        
        "security": [
            "security guard", "security officer", "security personnel", "security staff",
            "loss prevention", "surveillance", "access control", "patrol", "patrolling",
            "security management", "security supervisor", "security coordinator",
            "armed guard", "unarmed guard", "event security", "retail security",
            "corporate security", "facility security", "gate guard", "watchman",
            "security system", "cctv", "monitoring", "security operations"
        ],
        
        "cleaning_janitorial": [
            "janitor", "janitorial", "cleaner", "cleaning", "custodial", "custodian",
            "housekeeping", "sanitation", "sanitation worker", "cleaning staff",
            "office cleaner", "building cleaner", "floor care", "window cleaning",
            "carpet cleaning", "deep cleaning", "maintenance cleaning", "industrial cleaning",
            "commercial cleaning", "residential cleaning", "maid", "house cleaner"
        ],
        
        "personal_care": [
            "personal care", "personal care aide", "caregiver", "home care", "home health aide",
            "elderly care", "senior care", "elder care", "disability support", "disability care",
            "companion care", "respite care", "assisted living", "nursing aide", "cna",
            "certified nursing assistant", "patient care", "personal support worker"
        ],
        
        "childcare": [
            "childcare", "child care", "daycare", "day care", "nanny", "babysitter",
            "babysitting", "early childhood", "preschool teacher", "daycare teacher",
            "childcare provider", "childcare worker", "after school care", "summer camp",
            "childcare assistant", "toddler care", "infant care", "child development"
        ],
        "retail": [
            "retail", "store manager", "merchandising", "inventory", "point of sale",
            "pos", "customer service", "sales associate", "visual merchandising",
            "retail management", "e-commerce", "omnichannel"
        ],
        "trades": [
            "plumber", "electrician", "carpenter", "welder", "mason", "roofer",
            "hvac technician", "journeyman", "apprentice", "master tradesman",
            "construction trades", "skilled trades", "trade certification"
        ],
        "real_estate": [
            "real estate", "real estate agent", "broker", "property management",
            "commercial real estate", "residential real estate", "leasing",
            "property development", "real estate investment", "mls"
        ],
        "fitness": [
            "fitness", "personal trainer", "nutritionist", "yoga instructor",
            "strength and conditioning", "wellness", "health coach",
            "exercise physiology", "group fitness", "fitness training"
        ],
        "beauty": [
            "cosmetology", "esthetics", "barber", "hairstylist", "makeup artist",
            "nail technician", "spa", "salon", "beauty therapy", "skincare"
        ],
        "aviation": [
            "pilot", "flight attendant", "air traffic controller", "aviation",
            "aircraft", "faa", "commercial pilot", "private pilot", "airline",
            "aviation maintenance", "airport operations"
        ],
        "transportation": [
            "truck driver", "delivery driver", "logistics driver", "cdl",
            "transportation", "fleet management", "dispatch", "route planning",
            "shipping", "freight", "warehouse driver"
        ],
        "military": [
            "military", "veteran", "armed forces", "navy", "army", "air force",
            "marines", "coast guard", "military service", "veteran affairs",
            "defense", "security clearance"
        ],
        "nonprofit": [
            "nonprofit", "ngo", "fundraising", "grant writing", "donor relations",
            "volunteer coordination", "nonprofit management", "advocacy",
            "community outreach", "social work", "philanthropy"
        ],
        "public_sector": [
            "government", "public administration", "policy", "regulatory compliance",
            "public service", "federal", "state", "local government", "civil service"
        ],
        "arts": [
            "performing arts", "theater", "dance", "music", "acting", "directing",
            "fine arts", "sculpture", "painting", "art history", "art curator",
            "museum", "gallery", "arts administration"
        ],
        "sports": [
            "coaching", "athletic training", "sports", "sports management",
            "strength and conditioning", "sports medicine", "athletic director",
            "player development", "sports analytics"
        ],
        "environmental": [
            "environmental", "conservation", "ecology", "sustainability",
            "wildlife", "forestry", "park ranger", "environmental science",
            "renewable energy", "climate", "environmental policy"
        ],
        "technical_infrastructure": [
            "devops", "cloud engineer", "system administrator", "sre", "platform engineer",
            "infrastructure", "site reliability", "cloud infrastructure", "kubernetes", "docker",
            "terraform", "ansible", "ci/cd", "infrastructure as code", "iac"
        ],
    }
    
    detected_domains = []
    domain_scores = {}
    
    for domain, keywords in domain_keywords.items():
        score = 0
        # Check titles
        for title in titles:
            if any(keyword in title for keyword in keywords):
                detected_domains.append(domain)
                score += 2
        
        # Check skills
        matching_keywords = [kw for kw in keywords if kw in all_text]
        if len(matching_keywords) >= 2:
            detected_domains.append(domain)
            score += len(matching_keywords)
        
        if score > 0:
            domain_scores[domain] = score
    
    if detected_domains:
        from collections import Counter
        domain_counts = Counter(detected_domains)
        primary_domain, count = domain_counts.most_common(1)[0]
        
        total_detections = len(detected_domains)
        confidence = count / total_detections if total_detections > 0 else 0.0
        
        if primary_domain in domain_scores:
            score_boost = min(domain_scores[primary_domain] / 10.0, 0.3)
            confidence = min(confidence + score_boost, 1.0)
        
        # Map to uppercase domain names used by LLM
        domain_mapping = {
            "software_engineering": "SOFTWARE_ENGINEERING",
            "data_analytics": "DATA_ANALYTICS",
            "data_science": "DATA_SCIENCE",
            "ai_ml": "AI_ML",
            "product": "BUSINESS_STRATEGIC",
            "mechanical": "TRADITIONAL_ENGINEERING",
            "electrical": "TRADITIONAL_ENGINEERING",
            "civil": "TRADITIONAL_ENGINEERING",
            "chemical": "TRADITIONAL_ENGINEERING",
            "sales": "SALES_BUSINESS",
            "marketing": "MARKETING",
            "finance": "FINANCE",
            "hr": "HR",
            "legal": "LEGAL",
            "management": "MANAGEMENT",
            "healthcare": "HEALTHCARE",
            "ui_ux_design": "UI_UX_DESIGN",
            "creative_design": "CREATIVE_DESIGN",
            "manufacturing": "MANUFACTURING",
            "supply_chain": "SUPPLY_CHAIN",
            "operations": "OPERATIONS",
            "customer_support": "CUSTOMER_SUPPORT",
            "customer_success": "CUSTOMER_SUCCESS",
            "admin": "ADMIN",
            "education": "EDUCATION",
            "science": "SCIENCE",
            "construction": "CONSTRUCTION",
            "consulting": "CONSULTING",
            "hospitality": "HOSPITALITY",
            "retail": "RETAIL",
            "trades": "TRADES",
            "real_estate": "REAL_ESTATE",
            "fitness": "FITNESS",
            "beauty": "BEAUTY",
            "aviation": "AVIATION",
            "transportation": "TRANSPORTATION",
            "military": "MILITARY",
            "nonprofit": "NONPROFIT",
            "public_sector": "PUBLIC_SECTOR",
            "arts": "ARTS",
            "sports": "SPORTS",
            "environmental": "ENVIRONMENTAL",
            "technical_infrastructure": "TECHNICAL_INFRASTRUCTURE",
        }
        
        mapped_domain = domain_mapping.get(primary_domain, primary_domain.upper())
        return mapped_domain, confidence
    else:
        return "OTHER", 0.0


# Short role context per domain for LLM contextual matching (aligns with embedding domain_descriptions)
DOMAIN_ROLE_CONTEXT = {
    "software_engineering": "Software development, programming, web/mobile development, coding, computer science",
    "data_analytics": "Business intelligence, data analysis, reporting, dashboards, data visualization, analytics",
    "data_science": "Data science, machine learning, statistical analysis, data mining, Python/R/Spark",
    "ai_ml": "AI, machine learning engineering, deep learning, LLMs, TensorFlow/PyTorch",
    "technical_infrastructure": "DevOps, cloud, SRE, platform engineering, CI/CD, containerization",
    "traditional_engineering": "Mechanical, electrical, civil, chemical engineering, CAD, manufacturing",
    "business_strategic": "Product management, business analysis, strategy consulting, project management",
    "sales_business": "Sales, account management, business development, client relations, revenue",
    "marketing": "Digital marketing, SEO, content marketing, brand management, social media",
    "hr": "Human resources, talent acquisition, recruitment, employee relations, hiring",
    "finance": "Finance, accounting, auditing, financial analysis, tax, investment",
    "legal": "Legal services, attorney, compliance, corporate law, litigation",
    "management": "Executive leadership, general/operations/strategic management, C-suite",
    "healthcare": "Medical services, healthcare, nursing, clinical care, hospital administration",
    "ui_ux_design": "UI/UX design, UX research, prototyping, design systems, Figma/Sketch",
    "creative_design": "Graphic design, video editing, animation, photography, art direction",
    "manufacturing": "Production, assembly, machining, quality control, lean manufacturing",
    "supply_chain": "Supply chain, logistics, procurement, warehouse, distribution",
    "operations": "Business operations, process improvement, workflow optimization",
    "customer_support": "Customer service, help desk, technical support, customer care",
    "customer_success": "Customer success, client retention, product adoption, renewals",
    "admin": "Administrative assistant, office management, executive assistant, clerical",
    "education": "Teaching, educational services, training, academic instruction, curriculum development",
    "science": "Research scientist, laboratory, R&D, biology, chemistry, biotechnology",
    "construction": "Construction management, site supervision, civil projects, facilities, safety",
    "consulting": "Management consulting, strategy consulting, business advisory",
    "hospitality": "Hotel management, restaurant, culinary, food service, event planning, tourism",
    "retail": "Retail sales, store management, merchandising, e-commerce",
    "trades": "Skilled trades, plumbing, electrical, carpentry, welding, HVAC",
    "real_estate": "Real estate agent, broker, property management, leasing",
    "fitness": "Personal training, fitness coaching, nutrition, wellness",
    "beauty": "Cosmetology, esthetics, hairstyling, makeup, skincare, salon",
    "aviation": "Pilot, flight attendant, air traffic control, aircraft maintenance",
    "transportation": "Truck driving, delivery, logistics driving, fleet management",
    "military": "Military service, armed forces, defense, veterans",
    "nonprofit": "Nonprofit, NGO, fundraising, grant writing, volunteer coordination",
    "public_sector": "Government, public administration, policy, regulatory compliance",
    "arts": "Performing arts, theater, dance, music, fine arts, museum",
    "sports": "Athletic coaching, sports training, sports management, sports medicine",
    "environmental": "Environmental science, conservation, ecology, sustainability",
    "security": "Security guard, loss prevention, surveillance, access control",
    "cleaning_janitorial": "Janitorial, cleaning, custodial, housekeeping, sanitation",
    "personal_care": "Personal care aide, caregiver, home health aide, elderly care",
    "childcare": "Childcare, daycare, nanny, early childhood education, preschool teaching",
}


def _get_domain_role_context(domain: str) -> str:
    """Return a short role/context description for the given domain for LLM contextual matching."""
    if not domain or domain == "OTHER":
        return "General professional background"
    return DOMAIN_ROLE_CONTEXT.get(domain, domain.replace("_", " ").title())


def classify_jd_domains(jd_data: dict) -> List[str]:
    """Classify a job description into professional domains using keyword matching.
    
    Lightweight, deterministic classifier that can be used when the JD hasn't been
    parsed through the Groq JD parser (which produces jobDomains via LLM).
    Reuses the same domain_keywords map used for resume classification.
    
    Args:
        jd_data: Job description dict (expects jobTitle, requiredSkills, fullJobDescription, etc.)
        
    Returns:
        List of up to 3 domain strings (lowercase), e.g. ["ai_ml", "data_science", "software_engineering"].
        Returns ["other"] if no domain is detected.
    """
    parts = []
    title = jd_data.get("jobTitle") or jd_data.get("job_title") or jd_data.get("title") or ""
    if title:
        parts.append(title.lower())
    
    skills = jd_data.get("requiredSkills") or jd_data.get("required_skills") or []
    if isinstance(skills, list):
        parts.extend(str(s).lower() for s in skills if s)
    elif isinstance(skills, str):
        parts.append(skills.lower())
    
    desc = jd_data.get("fullJobDescription") or jd_data.get("full_job_description") or jd_data.get("description") or ""
    if desc:
        parts.append(desc[:2000].lower())
    
    all_text = " ".join(parts)
    if not all_text.strip():
        return ["other"]
    
    # Reuse the domain_keywords map defined in _get_domain_context_deterministic
    domain_keywords = {
        "software_engineering": [
            "software engineer", "developer", "programmer", "full stack", "backend", "frontend",
            "python", "java", "javascript", "typescript", "c++", "react", "angular", "vue",
            "node.js", "django", "spring boot", ".net", "docker", "kubernetes", "devops",
        ],
        "data_analytics": [
            "data analyst", "business intelligence", "bi", "tableau", "power bi", "qlik",
            "data visualization", "metrics", "kpi", "analytics", "reporting analyst",
        ],
        "data_science": [
            "data scientist", "statistics", "pandas", "numpy", "scipy", "jupyter",
            "regression", "classification", "clustering", "big data", "spark", "hadoop",
        ],
        "ai_ml": [
            "machine learning", "ml", "artificial intelligence", "ai", "deep learning",
            "tensorflow", "pytorch", "nlp", "computer vision", "llm", "large language models",
            "generative ai", "genai", "langchain", "hugging face", "prompt engineering",
        ],
        "product": [
            "product manager", "product owner", "product strategy", "roadmap", "prd",
        ],
        "mechanical": [
            "mechanical engineer", "solidworks", "catia", "autocad", "thermodynamics",
            "fea", "cad", "manufacturing engineering",
        ],
        "electrical": [
            "electrical engineer", "electronics", "pcb", "fpga", "embedded systems", "plc",
        ],
        "civil": [
            "civil engineer", "structural engineer", "construction management", "bim", "revit",
        ],
        "sales": [
            "sales", "business development", "account executive", "saas sales", "crm",
        ],
        "marketing": [
            "marketing", "digital marketing", "seo", "sem", "content marketing", "social media",
        ],
        "finance": [
            "finance", "accounting", "auditing", "financial analyst", "cpa", "tax", "fp&a",
        ],
        "hr": [
            "human resources", "hr", "recruiter", "talent acquisition", "hrbp",
        ],
        "legal": [
            "legal", "attorney", "lawyer", "compliance", "litigation",
        ],
        "healthcare": [
            "healthcare", "medical", "nursing", "clinical", "patient care", "pharmacy",
        ],
        "ui_ux_design": [
            "ui/ux", "ux design", "ui design", "figma", "sketch", "wireframing", "prototyping",
        ],
        "education": [
            "education", "teacher", "teaching", "tutor", "curriculum", "instructional design",
        ],
        "hospitality": [
            "hospitality", "hotel", "restaurant", "culinary", "chef", "food service",
        ],
        "customer_support": [
            "customer service", "help desk", "technical support", "support engineer",
        ],
        "retail": [
            "retail", "store manager", "merchandising", "e-commerce",
        ],
        "construction": [
            "construction", "site manager", "estimator", "surveyor", "osha",
        ],
        "technical_infrastructure": [
            "devops", "cloud engineer", "sre", "platform engineer", "terraform", "ansible",
        ],
    }
    
    domain_scores: dict = {}
    for domain, keywords in domain_keywords.items():
        score = sum(1 for kw in keywords if kw in all_text)
        if title and any(kw in title for kw in keywords):
            score += 3
        if score > 0:
            domain_scores[domain] = score
    
    if not domain_scores:
        return ["other"]
    
    sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in sorted_domains[:3]]


def _job_matches_domain(jd_data: dict, candidate_domain: str) -> bool:
    """Return True if job's job_domains (from metadata or doc) contains candidate_domain."""
    if not candidate_domain or candidate_domain.lower() == "other":
        return True
    domain = candidate_domain.lower().strip()
    raw = (jd_data.get("_job_domains") or jd_data.get("job_domains") or "").lower()
    return domain in raw


def _build_job_domain_where_clause(candidate_domain: str) -> Optional[dict]:
    """Build a ChromaDB where clause for job_descriptions collection: filter by job_domain_primary metadata.
    ChromaDB metadata 'where' only supports $eq/$ne/$in/$nin/$gt/$gte/$lt/$lte — NOT $contains.
    We store the primary domain as a single string in 'job_domain_primary' and filter with $eq.
    Returns None if no domain or domain is 'other' (no pre-filter).
    """
    if not candidate_domain or candidate_domain.lower().strip() == "other":
        return None
    domain = candidate_domain.lower().strip()
    return {"job_domain_primary": {"$eq": domain}}


def _get_candidate_domain_from_resume(resume: Dict[str, Any]) -> Optional[str]:
    """Return primary candidate domain from stored resume data (e.g. from parser), or None if not available."""
    if not resume or not isinstance(resume, dict):
        return None
    domains = resume.get("candidate_domains") or resume.get("candidateDomains") or []
    if not domains or not isinstance(domains, list):
        return None
    primary = domains[0]
    return (primary.strip() or None) if primary else None


def _enhance_resume_query_with_domain(resume_text: str, candidate_domain: str) -> str:
    """Prepend domain context to resume text for vector query (mirrors ranker's _enhance_query_with_domain_context_async).
    Makes the embedding more domain-focused to improve retrieval precision.
    """
    if not candidate_domain or candidate_domain.lower() == "other" or not resume_text:
        return resume_text
    domain = candidate_domain.lower().strip()
    role_context = DOMAIN_ROLE_CONTEXT.get(domain, "")
    if not role_context:
        return resume_text
    return f"DOMAIN: {domain.upper()}\nDomain context: {role_context}\n\n{resume_text}"


def _build_enhanced_candidate_query(
    resume_text: str,
    candidate_domain: str,
    candidate_skills: List[str],
) -> str:
    """Build a single enhanced candidate query for vector search (ranker-style).
    One query string: domain context + resume text + Key Skills Emphasis (top skills repeated).
    Used for single-query retrieval against job_descriptions, mirroring ranker's enhanced JD query.
    """
    base = _enhance_resume_query_with_domain(resume_text, candidate_domain)
    if not base:
        base = resume_text or ""
    if candidate_skills:
        skills_list = list(candidate_skills) if not isinstance(candidate_skills, list) else candidate_skills
        top_skills = skills_list[:5] if len(skills_list) > 5 else skills_list
        skills_emphasis = " ".join(str(s).strip() for s in top_skills if s).strip()
        if skills_emphasis:
            base = f"{base}\n\nKey Skills Emphasis: {' '.join(top_skills * 2)}"
    return base


def _build_skill_overlap_query(candidate_skills, candidate_domain: str) -> str:
    """Build a query string for semantic retrieval by skill overlap.
    Used so Chroma returns jobs whose required skills overlap with the candidate's skills.
    candidate_skills can be list or set (extract_primary_skills returns set).
    """
    if not candidate_skills:
        return ""
    skills_list = list(candidate_skills) if not isinstance(candidate_skills, list) else candidate_skills
    # Cap to avoid huge query; emphasis on skills as job requirements
    skills_list = skills_list[:50] if len(skills_list) > 50 else skills_list
    skills_text = ", ".join(str(s).strip() for s in skills_list if s)
    if not skills_text:
        return ""
    base = f"Job requiring skills: {skills_text}"
    if candidate_domain and candidate_domain.lower() != "other":
        role_context = DOMAIN_ROLE_CONTEXT.get(candidate_domain.lower().strip(), "")
        if role_context:
            base = f"DOMAIN: {candidate_domain.upper()}\n{base}\n\n{role_context}"
    return base


def _parse_chroma_results_to_jobs(
    query_results: Optional[Dict],
    semantic_threshold: float = 0.30,
) -> List[Dict[str, Any]]:
    """Parse Chroma query result into list of job dicts with job_id, _vector_similarity, _job_domains."""
    jobs = []
    if not query_results or not query_results.get("ids") or not query_results["ids"][0]:
        return jobs
    job_ids = query_results["ids"][0]
    distances = query_results.get("distances") and query_results["distances"][0] or []
    documents = query_results.get("documents") and query_results["documents"][0] or []
    metadatas = query_results.get("metadatas") or []
    if metadatas and isinstance(metadatas[0], list):
        metadatas = metadatas[0]
    else:
        metadatas = []
    for idx, job_id in enumerate(job_ids):
        try:
            distance = distances[idx] if idx < len(distances) else 1.0
            vector_similarity = max(0.0, min(1.0, 1.0 / (1.0 + distance)))
            if vector_similarity < semantic_threshold:
                continue
            doc = documents[idx] if idx < len(documents) else None
            if not doc:
                continue
            if isinstance(doc, str):
                jd_data = json.loads(doc)
            elif isinstance(doc, dict):
                jd_data = doc
            else:
                continue
            jd_data["job_id"] = str(job_id)
            jd_data["jd_id"] = str(job_id)
            jd_data["id"] = str(job_id)
            jd_data["_vector_similarity"] = vector_similarity
            meta = metadatas[idx] if idx < len(metadatas) else {}
            jd_data["_job_domains"] = meta.get("job_domains", "") if isinstance(meta, dict) else ""
            if not jd_data.get("jobTitle") and not jd_data.get("title"):
                if isinstance(meta, dict):
                    title_from_meta = meta.get("job_title") or meta.get("jobTitle") or meta.get("title")
                    if title_from_meta:
                        jd_data["jobTitle"] = title_from_meta
                if not jd_data.get("jobTitle"):
                    title_from_doc = (
                        jd_data.get("job_title")
                        or jd_data.get("position")
                        or jd_data.get("role")
                        or jd_data.get("designation")
                    )
                    if title_from_doc:
                        jd_data["jobTitle"] = title_from_doc
            jobs.append(jd_data)
        except Exception as e:
            log.warning(f"⚠️ Error processing job {job_id}: {e}")
    return jobs


def _merge_job_results_by_best_similarity(
    jobs_a: List[Dict[str, Any]],
    jobs_b: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge two job lists by job_id, keeping the entry with higher _vector_similarity. Order by similarity desc."""
    by_id = {}
    for j in jobs_a + jobs_b:
        jid = j.get("job_id") or j.get("id") or j.get("jd_id")
        if not jid:
            continue
        sim = j.get("_vector_similarity") or 0.0
        if jid not in by_id or (by_id[jid].get("_vector_similarity") or 0.0) < sim:
            by_id[jid] = j
    return sorted(by_id.values(), key=lambda x: (x.get("_vector_similarity") or 0.0), reverse=True)


async def classify_candidate_domain(resume: Dict[str, Any]) -> str:
    """
    Semantic embedding-based domain classification using Google Gemini.
    Replaces deterministic keyword matching with pure semantic similarity.
    
    This helps prevent cross-domain matching (e.g., DevOps engineer → truck driver).
    """
    try:
        # Validate resume has content
        if not resume or not isinstance(resume, dict):
            log.warning("⚠️ Empty or invalid resume provided for domain classification")
            return "OTHER"
        
        # Check if resume has meaningful content
        has_content = any([
            resume.get("Skills"),
            resume.get("WorkExperience"),
            resume.get("Education"),
            resume.get("Summary"),
            resume.get("Name")
        ])
        
        if not has_content:
            log.warning("⚠️ Resume has no meaningful content for domain classification")
            return "OTHER"
        
        # ✅ NEW: Use semantic embedding-based classification
        log.info("🔍 Using semantic embedding-based domain classification...")
        domain, confidence = await _classify_domain_with_embeddings(resume)
        
        CONFIDENCE_THRESHOLD = 0.5
        
        if confidence >= CONFIDENCE_THRESHOLD:
            log.info(f"✅ Semantic domain classification: '{domain}' (confidence: {confidence:.2f})")
            return domain
        
        # STEP 2: Fallback to LLM for low confidence cases
        log.info(f"⚠️ Low confidence ({confidence:.2f}) for domain '{domain}', using LLM fallback")
        
        # Prepare anonymized resume data
        anonymized_resume = anonymize_candidate_info(resume)
        resume_text = json.dumps(anonymized_resume, indent=2)
        
        # Check if anonymized resume is empty
        if len(resume_text.strip()) < 10:  # Very short resume text
            log.warning("⚠️ Anonymized resume is too short for domain classification")
            return domain  # Return deterministic result
        
        log.info(f"📝 Resume text length: {len(resume_text)} characters")
        log.info(f"📝 Resume skills: {resume.get('Skills', [])}")
        
        prompt = f"""Analyze this resume and classify the candidate's primary professional domain.

RESUME:
{resume_text}

Return ONLY a JSON object with this structure:
{{
    "primary_domain": "SOFTWARE_ENGINEERING|DATA_ANALYTICS|DATA_SCIENCE|AI_ML|TECHNICAL_INFRASTRUCTURE|TRADITIONAL_ENGINEERING|BUSINESS_STRATEGIC|SALES_BUSINESS|MARKETING|HR|FINANCE|LEGAL|MANAGEMENT|HEALTHCARE|UI_UX_DESIGN|CREATIVE_DESIGN|MANUFACTURING|SUPPLY_CHAIN|OPERATIONS|CUSTOMER_SUPPORT|CUSTOMER_SUCCESS|ADMIN|EDUCATION|SCIENCE|CONSTRUCTION|CONSULTING|HOSPITALITY|RETAIL|TRADES|REAL_ESTATE|FITNESS|BEAUTY|AVIATION|TRANSPORTATION|MILITARY|NONPROFIT|PUBLIC_SECTOR|ARTS|SPORTS|ENVIRONMENTAL|SECURITY|CLEANING_JANITORIAL|PERSONAL_CARE|CHILDCARE|OTHER",
    "confidence": 0.95,
    "reasoning": "Brief explanation of classification"
}}

Domain Definitions:
- SOFTWARE_ENGINEERING: Software Development, Programming, Frontend/Backend Development, Mobile Development, Full-Stack Development
- DATA_ANALYTICS: Business Intelligence, Reporting, Dashboards, Data Visualization, Analytics (Tableau, Power BI, SQL)
- DATA_SCIENCE: Data Science, Statistical Analysis, Hypothesis Testing, Data Mining, Big Data (Python, R, Spark, Hadoop)
- AI_ML: Machine Learning, Artificial Intelligence, Deep Learning, Neural Networks, LLMs, AI Engineering (TensorFlow, PyTorch, GPT, LangChain)
- TECHNICAL_INFRASTRUCTURE: DevOps, Cloud Engineering, System Administration, SRE, Platform Engineering, Infrastructure
- TRADITIONAL_ENGINEERING: Mechanical, Electrical, Civil, Chemical Engineering (CAD, SolidWorks, PCB design, etc.)
- BUSINESS_STRATEGIC: Product Management, Business Analysis, Strategy, Project Management
- SALES_BUSINESS: Sales, Account Management, Business Development, Client Relations
- MARKETING: Digital Marketing, SEO, SEM, Content Marketing, Brand Management, Advertising
- HR: Human Resources, Talent Acquisition, Recruitment, Employee Relations
- FINANCE: Finance, Accounting, Auditing, Financial Analysis, Tax, Investment Banking
- LEGAL: Legal, Attorney, Lawyer, Compliance, Corporate Law, Litigation
- MANAGEMENT: Executive Leadership, General Management, Operations Management, Strategy
- HEALTHCARE: Medical, Healthcare, Nursing, Clinical, Patient Care, Hospital Administration
- UI_UX_DESIGN: User Interface/Experience Design, UX Research, Prototyping, Design Systems (Figma, Sketch)
- CREATIVE_DESIGN: Graphic Design, Video Editing, Animation, Photography, Art Direction
- MANUFACTURING: Production, Assembly, Machining, Quality Control, Lean Manufacturing
- SUPPLY_CHAIN: Supply Chain, Logistics, Procurement, Warehouse, Distribution, Sourcing
- OPERATIONS: Business Operations, Operational Excellence, Process Improvement, Workflow
- CUSTOMER_SUPPORT: Customer Service, Help Desk, Technical Support, Ticketing (Zendesk, Intercom)
- CUSTOMER_SUCCESS: Customer Success Management, Retention, Adoption, Renewals, Account Health
- ADMIN: Administrative, Office Management, Executive Assistant, Receptionist, Clerical
- EDUCATION: Teaching, Educational Services, Training, Academic, Instructional Design
- SCIENCE: Research, Laboratory, R&D, Biology, Chemistry, Biotech, Pharmaceutical
- CONSTRUCTION: Construction, Site Management, Civil Engineering, Facilities, Safety
- CONSULTING: Management Consulting, Strategy, Advisory, Business Transformation
- HOSPITALITY: Hotel, Restaurant, Culinary, Food Service, Event Planning, Tourism, Tea Maker, Coffee Maker, Beverage Service, Pantry, Kitchen Helper, Server, Waiter, Waitress, Barista, Bartender, Host, Hostess, Busser, Dishwasher, Line Cook, Prep Cook, Sous Chef, Pastry Chef, Food Runner, Room Service, Housekeeping, Laundry, Bellhop, Valet, Parking Attendant, Guest Services, Reservations, Front Office, Night Auditor
- RETAIL: Retail, Store Management, Merchandising, E-commerce, Visual Merchandising
- TRADES: Plumbing, Electrical Work, Carpentry, Welding, HVAC, Skilled Trades
- REAL_ESTATE: Real Estate Agent, Broker, Property Management, Leasing, MLS
- FITNESS: Personal Training, Nutrition, Wellness, Health Coaching, Exercise Physiology
- BEAUTY: Cosmetology, Esthetics, Hairstyling, Makeup, Skincare, Salon
- AVIATION: Pilot, Flight Attendant, Air Traffic Control, Aircraft, Aviation Maintenance
- TRANSPORTATION: Truck Driver, Delivery Driver, Logistics Driver, Fleet Management, CDL
- MILITARY: Military, Veteran, Armed Forces, Defense, Security Clearance
- NONPROFIT: Nonprofit, NGO, Fundraising, Grant Writing, Volunteer Coordination
- PUBLIC_SECTOR: Government, Public Administration, Policy, Regulatory Compliance
- ARTS: Performing Arts, Theater, Dance, Music, Fine Arts, Museum, Gallery
- SPORTS: Coaching, Athletic Training, Sports Management, Sports Medicine
- ENVIRONMENTAL: Environmental, Conservation, Ecology, Sustainability, Wildlife, Forestry
- SECURITY: Security Guard, Security Officer, Loss Prevention, Surveillance, Access Control, Patrol, Armed Guard, Unarmed Guard, Event Security, Retail Security, Corporate Security, Facility Security, Gate Guard, Watchman, CCTV Monitoring, Security Operations
- CLEANING_JANITORIAL: Janitor, Janitorial, Cleaner, Cleaning, Custodial, Custodian, Housekeeping, Sanitation, Sanitation Worker, Cleaning Staff, Office Cleaner, Building Cleaner, Floor Care, Window Cleaning, Carpet Cleaning, Deep Cleaning, Maintenance Cleaning, Industrial Cleaning, Commercial Cleaning, Residential Cleaning, Maid, House Cleaner
- PERSONAL_CARE: Personal Care, Personal Care Aide, Caregiver, Home Care, Home Health Aide, Elderly Care, Senior Care, Elder Care, Disability Support, Disability Care, Companion Care, Respite Care, Assisted Living, Nursing Aide, CNA, Certified Nursing Assistant, Patient Care, Personal Support Worker
- CHILDCARE: Childcare, Child Care, Daycare, Day Care, Nanny, Babysitter, Babysitting, Early Childhood, Preschool Teacher, Daycare Teacher, Childcare Provider, Childcare Worker, After School Care, Summer Camp, Childcare Assistant, Toddler Care, Infant Care, Child Development
- OTHER: Any other domain not listed above

Return ONLY the JSON object."""

        log.info("🤖 Classifying candidate domain using LLM...")
        log.info(f"📝 Prompt length: {len(prompt)} characters")
        
        response = await invoke_llm(
            prompt=prompt,
            task_type="classification",
            agent_name="domain_classifier",
            max_retries=2
        )
        
        log.info(f"🤖 LLM response received, length: {len(response)} characters")
        log.info(f"🤖 Raw response: {response[:200]}...")
        
        # Extract JSON from response (handles markdown code fences)
        result = _extract_json_from_response(response)
        if not result:
            log.warning("⚠️ Could not extract JSON from response, using default domain")
            return "OTHER"
        
        domain = result.get("primary_domain", "OTHER")
        confidence = result.get("confidence", 0.0)
        reasoning = result.get("reasoning", "")
        
        # ✅ FIX: Normalize domain to lowercase format for consistent comparison
        domain_normalized = _normalize_llm_domain(domain)
        
        log.info(f"✅ Candidate classified as: {domain} → {domain_normalized} (confidence: {confidence:.2f}) - {reasoning}")
        return domain_normalized
        
    except Exception as e:
        log.error(f"❌ Failed to parse LLM response: {e}")
        log.error(f"❌ Raw response: {response[:500]}...")
        return "OTHER"
    except Exception as e:
        log.error(f"❌ Domain classification failed: {e}")
        import traceback
        log.error(f"❌ Traceback: {traceback.format_exc()}")
        return "OTHER"


async def llm_filter_and_rank_jobs(resume: Dict[str, Any], job_descriptions: List[Dict]) -> List[Dict]:
    """
    Use LLM to intelligently filter and rank job matches.
    This prevents inappropriate cross-domain matching.
    """
    try:
        # Prepare anonymized resume data
        anonymized_resume = anonymize_candidate_info(resume)
        resume_text = json.dumps(anonymized_resume, indent=2)
        
        # Create job summaries for LLM analysis - full JD data (no truncation)
        job_summaries = []
        for jd in job_descriptions[:50]:  # Limit to 50 jobs to prevent token limits
            job_summaries.append({
                "id": jd.get("jd_id", ""),
                "title": jd.get("jobTitle", ""),
                "company": jd.get("company", ""),
                "location": jd.get("location", ""),
                "required_skills": jd.get("requiredSkills", []) or [],  # Full list - no truncation
                "description": (jd.get("description", "") or "")[:800]  # Sufficient for filter; batch analysis gets full JD
            })
        
        log.info(f"📊 Created {len(job_summaries)} job summaries for LLM analysis")
        log.info(f"🔍 Sample job summaries:")
        for i, summary in enumerate(job_summaries[:3]):
            log.info(f"   {i+1}. {summary['title']} at {summary['company']} (Skills: {summary['required_skills'][:3]})")
        
        prompt = f"""You are an expert job matching assistant. Analyze the candidate's background and match them with appropriate jobs.

CANDIDATE RESUME:
{resume_text}

AVAILABLE JOBS:
{json.dumps(job_summaries, indent=2)}

TASK: Return ONLY a JSON array of job matches ranked by suitability. Each match should include:
- job_id: The job identifier
- match_score: Score from 0-100 (use 75+ for good matches, 85+ for excellent matches)
- match_reason: Brief explanation of why this is a good match (IMPORTANT: Use second person - "you", "your", "you have", etc. - when addressing the candidate)
- domain_match: Whether the job domain aligns with candidate's background (MUST be true for inclusion)

CRITICAL RULES:
1. NEVER match technical candidates (engineers, developers, DevOps) with non-technical roles (drivers, retail, manual labor, etc.)
2. Consider domain alignment (DevOps → Infrastructure/Cloud roles, not transportation)
3. Focus on skill relevance and career progression
4. Reject jobs that are clearly outside the candidate's domain
5. Prioritize jobs where the candidate's skills directly apply
6. Consider seniority level matching
7. Only assign scores of 75+ if the candidate genuinely matches well
8. Only mark domain_match as true if the job is clearly in the candidate's professional domain

SCORING GUIDELINES:
- 85-100: Excellent match - candidate has most required skills and strong domain alignment
- 75-84: Good match - candidate has key skills and domain alignment
- 60-74: Moderate match - some skills but significant gaps (DO NOT include)
- Below 60: Poor match (DO NOT include)

Return format:
[
  {{
    "job_id": "job_123",
    "match_score": 85,
    "match_reason": "Your strong DevOps skills align well with this cloud infrastructure role",
    "domain_match": true
  }}
]

IMPORTANT: In match_reason, always use second person (you, your, you have) when addressing the candidate. Never use third person (the candidate, they, their).

Return ONLY the JSON array, no other text."""

        log.info("🤖 Using LLM for intelligent job matching...")
        log.info(f"📝 Prompt length: {len(prompt)} characters")
        
        response = await invoke_llm(
            prompt=prompt,
            task_type="classification",
            agent_name="job_matcher",
            max_retries=2
        )
        
        log.info(f"🤖 LLM response received, length: {len(response)} characters")
        
        # Parse LLM response (handles markdown code fences)
        try:
            # First try to extract JSON array directly (for job matches)
            import re
            # Try to find JSON array in markdown fences
            array_match = re.search(r'```json\s*(\[.*?\])\s*```', response, re.DOTALL | re.IGNORECASE)
            if array_match:
                matches = json.loads(array_match.group(1))
                log.info(f"✅ Successfully parsed LLM response from markdown: {len(matches)} matches")
            else:
                # Try to find JSON array without fences
                array_match = re.search(r'(\[.*?\])', response, re.DOTALL)
                if array_match:
                    matches = json.loads(array_match.group(1))
                    log.info(f"✅ Successfully parsed LLM response: {len(matches)} matches")
                else:
                    # Fallback to _extract_json_from_response for dict responses
                    extracted = _extract_json_from_response(response)
                    if isinstance(extracted, list):
                        matches = extracted
                    elif isinstance(extracted, dict):
                        matches = extracted.get("matches", [])
                    else:
                        matches = []
                    log.info(f"✅ Successfully parsed LLM response (fallback): {len(matches)} matches")
        except json.JSONDecodeError as e:
            log.error(f"❌ Failed to parse LLM response as JSON: {e}")
            log.error(f"❌ Raw response: {response[:500]}...")
            return []
        except Exception as e:
            log.error(f"❌ Failed to parse LLM response: {e}")
            log.error(f"❌ Raw response: {response[:500]}...")
            return []
        
        # ✅ STRICTER: Filter out low-quality matches - REQUIRE domain_match=True
        # This prevents cross-domain matches (e.g., tech candidates → chef jobs)
        quality_matches = []
        for m in matches:
            score = m.get("match_score", 0)
            domain_match = m.get("domain_match", False)
            log.info(f"   🔍 Evaluating match: job_id={m.get('job_id', 'unknown')}, score={score}, domain_match={domain_match}")
            
            # ✅ CHANGED: Require domain_match=True AND score >= 60
            # Removed: "or score >= 80" exception that allowed domain mismatches
            if score >= 60 and domain_match == True:
                    quality_matches.append(m)
                    log.info(f"   ✅ Match passed filter: {m.get('job_id', 'unknown')}")
                
            else:
                if not domain_match:
                    log.info(f"   ❌ Match filtered: domain_match=False (domain mismatch - prevents cross-domain matches)")
                else:
                    log.info(f"   ❌ Match filtered: score {score} < 60")
        
        log.info(f"📊 LLM matches before filtering: {len(matches)}")
        log.info(f"📊 LLM matches after filtering: {len(quality_matches)}")
        
        sorted_matches = sorted(quality_matches, key=lambda x: x["match_score"], reverse=True)
        top_matches = sorted_matches[:5]
        
        # Final quality check: ensure best match meets minimum quality threshold
        if top_matches and top_matches[0]["match_score"] >= 60:  # Lowered from 75
            log.info(f"✅ LLM found {len(top_matches)} high-quality job matches (best score: {top_matches[0]['match_score']})")
            return top_matches
        else:
            best_score = top_matches[0]['match_score'] if top_matches else 0
            log.info(f"⚠️ LLM matches do not meet quality threshold (best score: {best_score})")
            return []
        
    except Exception as e:
        log.error(f"❌ LLM job matching failed: {e}")
        return []


async def enhanced_llm_job_matcher(resume: Dict[str, Any], job_descriptions: List[Dict]) -> List[Dict]:
    """
    Multi-stage LLM matching with domain validation.
    This is the main function that orchestrates the LLM-based matching process.
    """
    try:
        log.info("🚀 Starting enhanced LLM job matching...")
        log.info(f"📊 Processing {len(job_descriptions)} job descriptions")
        
        # Stage 1: Domain Classification (use stored domains from resume when available)
        log.info("🔍 Stage 1: Domain Classification")
        candidate_domain = _get_candidate_domain_from_resume(resume)
        if not candidate_domain:
            candidate_domain = await classify_candidate_domain(resume)
        log.info(f"✅ Candidate classified as: {candidate_domain}")
        
        # Stage 2: LLM-based filtering and ranking
        log.info("🔍 Stage 2: LLM-based filtering and ranking")
        llm_matches = await llm_filter_and_rank_jobs(resume, job_descriptions)
        log.info(f"✅ LLM filtering found {len(llm_matches)} matches")
        
        if not llm_matches:
            log.warning("⚠️ LLM filtering returned no matches")
            return []
        
        # Stage 3: Process matches into expected format
        log.info("🔍 Stage 3: Processing matches")
        processed_matches = []
        for match in llm_matches:
            try:
                job_id = match.get("job_id", "")
                log.info(f"🔍 Processing match: job_id={job_id}")
                
                jd_data = await run_cpu_intensive(get_job_description, job_id)
                if not jd_data:
                    log.warning(f"⚠️ Job description not found for job_id={job_id}, trying to use match data directly")
                    jd_data = next((jd for jd in job_descriptions if jd.get("jd_id") == job_id or jd.get("id") == job_id), None)
                    
                if jd_data:
                    # Perform detailed skill matching for additional context
                    matched_skills, unmatched_skills = await extract_matched_and_unmatched_skills(resume, jd_data)
                    matched_exp = match_experience(resume, jd_data.get("experience", ""))
                    matched_edu = match_education(resume, jd_data)
                    
                    reason = match.get("match_reason", "")
                    processed_match = {
                        "job_id": match["job_id"],
                        "job_title": jd_data.get("jobTitle", match.get("job_title", "Unknown")),
                        "company_name": jd_data.get("company", match.get("company_name", "Unknown")),
                        "location": jd_data.get("location", match.get("location", "Remote")),
                        "match_score": match["match_score"] / 100.0,  # Convert to 0-1 scale
                        "match_reason": reason,
                        "matched_skills": matched_skills,
                        "unmatched_skills": unmatched_skills,  # Add unmatched skills (skills gap)
                        "matched_education": matched_edu,
                        "matched_experience": matched_exp,
                        "rationale": reason,
                        "positive_rationale": match.get("positive_rationale", ""),
                        "negative_rationale": match.get("negative_rationale", ""),
                        "how_to_improve": match.get("how_to_improve", ""),
                        "candidate_domain": candidate_domain
                    }
                    processed_matches.append(processed_match)
                    log.info(f"✅ Processed match: {processed_match['job_title']} (score: {processed_match['match_score']:.2f})")
                    log.info(f"   Matched skills: {len(matched_skills)}, Unmatched skills: {len(unmatched_skills)}")
                else:
                    log.error(f"❌ Could not retrieve job data for job_id={job_id}, skipping match")
                    
            except Exception as e:
                log.warning(f"⚠️ Failed to process match {match.get('job_id', 'unknown')}: {e}")
                import traceback
                log.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        log.info(f"✅ Enhanced LLM matching completed with {len(processed_matches)} matches")
        return processed_matches
        
    except Exception as e:
        log.error(f"❌ Enhanced LLM job matching failed: {e}")
        import traceback
        log.error(f"❌ Traceback: {traceback.format_exc()}")
        return []


def calculate_experience_years(work_experience: List[Dict[str, Any]]) -> float:
    """
    Calculate total years of experience from work history.
    Uses centralized shared utility for consistency.
    """
    from utils.experience_years import calculate_total_experience_years
    return calculate_total_experience_years(work_experience)


def match_experience(resume: Dict[str, Any], required_exp: str) -> str:
    """
    Compare resume experience with JD experience requirement.
    Returns formatted string showing match status.
    """
    work_exp = resume.get("WorkExperience", [])
    if not work_exp:
        return "0 years (no experience listed)"
    
    total_exp = calculate_experience_years(work_exp)
    
    # Parse required experience
    required_exp_lower = required_exp.lower()
    req_years = 0.0
    
    # Try to extract number from requirement
    import re
    numbers = re.findall(r'\d+\.?\d*', required_exp_lower)
    if numbers:
        req_years = float(numbers[0])
    
    # Format response
    if req_years > 0:
        if total_exp >= req_years:
            return f"{total_exp} years (meets requirement of {req_years} years)"
        else:
            return f"{total_exp} years (below requirement of {req_years} years)"
    else:
        return f"{total_exp} years (experience detected)"


def match_education(resume: Dict[str, Any], jd_data: Dict[str, Any]) -> List[str]:
    """
    Enhanced education matching with fuzzy matching and degree variations.
    Finds matching education details between resume and job description.
    """
    from difflib import SequenceMatcher
    
    matches = []
    education = resume.get("Education", [])
    
    # OPTIMIZATION: Removed debug logging (was causing performance issues)
    
    # Get JD education requirements from multiple sources
    jd_education = jd_data.get("education", jd_data.get("requiredEducation", ""))
    jd_description = jd_data.get("description", "")
    jd_title = jd_data.get("jobTitle", "")
    
    # Also check alternative field names
    if not jd_education:
        jd_education = jd_data.get("education_requirements", "")
    if not jd_description:
        jd_description = jd_data.get("job_description", "")
    
    jd_text = f"{jd_education} {jd_description} {jd_title}".lower()
    
    # If no structured education found, try to extract from full job description
    if not jd_education:
        full_jd_text = jd_data.get("fullJobDescription", "")
        if full_jd_text:
            # Look for education patterns in the full text
            education_patterns = [
                r'(?:education|degree|qualification|requirement)\s*:?\s*([^.\n]+)',
                r'(?:bachelor|master|phd|diploma|certificate)\s*[^.\n]*',
                r'(?:university|college|institute)\s*[^.\n]*'
            ]
            for pattern in education_patterns:
                pattern_matches = re.findall(pattern, full_jd_text, re.IGNORECASE)
                for match in pattern_matches:
                    if isinstance(match, tuple):
                        match = match[0] if match[0] else match[1] if len(match) > 1 else ""
                    if match and len(match.strip()) > 5:
                        jd_text += f" {match.strip().lower()}"
    
    for edu in education:
        if not isinstance(edu, dict):
            continue
            
        degree = edu.get("degree", "")
        major = edu.get("major", edu.get("field", ""))
        university = edu.get("university", edu.get("institution", edu.get("school", "")))
        
        matched_parts = []
        
        # Enhanced degree matching
        if degree:
            degree_lower = degree.lower()
            # Exact match
            if degree_lower in jd_text:
                matched_parts.append(degree)
            else:
                # Fuzzy match for common degree variations
                degree_variations = {
                    "bachelor": ["bachelor's", "bachelor", "bsc", "ba", "bs", "b.tech", "btech"],
                    "master": ["master's", "master", "msc", "ma", "ms", "m.tech", "mtech", "mba"],
                    "phd": ["phd", "ph.d", "doctorate", "doctoral", "d.phil"],
                    "diploma": ["diploma", "certificate", "certification"],
                    "associate": ["associate", "aas", "aa"]
                }
                
                for base_degree, variations in degree_variations.items():
                    if any(var in degree_lower for var in variations):
                        if any(var in jd_text for var in variations):
                            matched_parts.append(degree)
                            break
        
        # Enhanced major matching
        if major:
            major_lower = major.lower()
            if major_lower in jd_text:
                matched_parts.append(major)
            else:
                # Fuzzy match for major
                major_words = major_lower.split()
                for word in major_words:
                    if len(word) > 3 and word in jd_text:
                        matched_parts.append(major)
                        break
        
        # Institution matching (optional, less strict)
        if university and len(matched_parts) > 0:
            university_lower = university.lower()
            if university_lower in jd_text:
                matched_parts.append(f"from {university}")
        
        if matched_parts:
            match_str = " ".join(matched_parts)
            matches.append(match_str)
    
    return matches


def anonymize_candidate_info(structured_resume: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove all personally identifiable information from resume.
    Returns anonymized version safe for LLM processing.
    Includes skills, education, experience, certifications, projects, extras, professional_summary.
    Handles both lowercase and capitalized keys.
    """
    anonymized = {}
    
    # Extract only non-PII fields (handle both case variations)
    skills = structured_resume.get("skills") or structured_resume.get("Skills") or structured_resume.get("SKILLS", [])
    if skills:
        anonymized["skills"] = skills
    
    education = structured_resume.get("education") or structured_resume.get("Education") or structured_resume.get("EDUCATION", [])
    if education:
        edu_list = []
        for edu in education:
            if isinstance(edu, dict):
                edu_anonymized = {
                    "degree": edu.get("degree") or edu.get("Degree", ""),
                    "major": edu.get("major") or edu.get("Major", ""),
                }
                edu_list.append(edu_anonymized)
        anonymized["education"] = edu_list
    
    # Handle work experience (both case variations)
    experience = structured_resume.get("experience") or structured_resume.get("Experience") or structured_resume.get("WorkExperience") or structured_resume.get("WORK_EXPERIENCE", [])
    if experience:
        exp_list = []
        for exp in experience:
            if isinstance(exp, dict):
                title = exp.get("job_title") or exp.get("title") or exp.get("JobTitle", "")
                exp_anonymized = {
                    "title": title,
                    "years": exp.get("dates", ""),
                    "domain": exp.get("domain", ""),
                }
                exp_list.append(exp_anonymized)
        anonymized["work_experience"] = exp_list
    
    # Certifications: support certification_name (groq) and name
    certs = structured_resume.get("certifications") or structured_resume.get("Certifications", [])
    if certs:
        cert_list = []
        for cert in certs:
            if isinstance(cert, dict):
                name = cert.get("certification_name") or cert.get("name") or cert.get("certificationName", "")
                if name:
                    cert_list.append(name)
            elif cert:
                cert_list.append(str(cert))
        if cert_list:
            anonymized["certifications"] = cert_list
    
    # Projects (for LLM to consider in match)
    projects = structured_resume.get("projects") or structured_resume.get("Projects", [])
    if projects:
        anonymized["projects"] = projects
    
    # Extras (awards, etc.)
    extras = structured_resume.get("extras") or structured_resume.get("Extras", [])
    if extras:
        anonymized["extras"] = extras
    
    # Professional summary
    summary = structured_resume.get("professional_summary") or structured_resume.get("Summary") or structured_resume.get("professionalSummary", "")
    if summary and isinstance(summary, str) and summary.strip():
        anonymized["professional_summary"] = summary.strip()[:2000]
    
    # Total experience (for context)
    total_exp = structured_resume.get("total_experience_years") or structured_resume.get("totalExperienceYears", "")
    if total_exp:
        anonymized["total_experience_years"] = total_exp
    
    return anonymized


async def generate_match_rationale(
    job_title: str,
    company_name: str,
    location: str,
    required_experience: str,
    matched_skills: List[str],
    matched_education: List[str],
    matched_experience: str,
    match_score: float,
    jd_data: Dict[str, Any],
    recruiter_questions: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Generate AI-powered rationale explaining why this job matches the candidate.
    Uses only anonymized, non-PII data.
    ✅ ENHANCED: Now includes recruiter questions and candidate answers from payload
    """
    try:
        log.info(f"🤖 Starting rationale generation for: {job_title}")
        
        # ✅ NEW: Format recruiter questions and answers
        recruiter_qa = ""
        if recruiter_questions:
            if isinstance(recruiter_questions, list):
                qa_pairs = []
                for qa in recruiter_questions:
                    if isinstance(qa, dict):
                        q = qa.get("question") or qa.get("Question") or ""
                        a = qa.get("answer") or qa.get("Answer") or ""
                        if q and a:
                            qa_pairs.append(f"Q: {q}\nA: {a}")
                if qa_pairs:
                    recruiter_qa = "\n\n**Recruiter Questions & Candidate Answers:**\n" + "\n\n".join(qa_pairs[:5])  # Limit to top 5
                    log.debug(f"✅ Including {len(qa_pairs)} recruiter Q&A pairs in rationale")
            elif isinstance(recruiter_questions, str):
                recruiter_qa = f"\n\n**Recruiter Questions & Answers:**\n{recruiter_questions[:500]}"
        
        # Prepare anonymized context for LLM
        prompt = f"""You are a professional job matching assistant. Generate a brief, professional rationale (2-3 sentences) explaining why this job is a good match. IMPORTANT: Address the candidate directly using second person ("you", "your", "you have", etc.).

**Job Information:**
- Title: {job_title}
- Company: {company_name}
- Location: {location}
- Required Experience: {required_experience}

**Match Analysis:**
- Match Score: {match_score:.0%}
- Candidate's Experience: {matched_experience}
- Matched Skills: {', '.join(matched_skills) if matched_skills else 'None explicitly matched'}
- Matched Education: {', '.join(matched_education) if matched_education else 'None explicitly matched'}

**Job Requirements Summary:**
- Required Skills: {', '.join(jd_data.get('requiredSkills', []) or [])}
- Job Type: {jd_data.get('jobType', 'Not specified')}
{recruiter_qa}

Generate a concise rationale that:
1. Highlights the key strengths of this match (use "your skills", "your experience", etc.)
2. Mentions specific skills or qualifications that align (use "you have", "your background in", etc.)
3. ✅ NEW: References relevant recruiter questions and candidate answers if provided (e.g., "Based on your answer about X, you demonstrate Y")
4. Provides actionable insight for the candidate (use "you", "your", etc.)

CRITICAL: Always use second person pronouns (you, your, you have, your experience, etc.). Never use third person (the candidate, they, their, etc.).

Keep it professional, positive, and under 100 words. Do NOT include any personal information, names, or identifying details about the candidate."""

        log.info(f"🤖 Calling LLM for rationale generation...")
        
        # Call LLM to generate rationale with shorter timeout for faster response
        rationale = await invoke_llm(
            prompt=prompt,
            task_type="text_generation",
            agent_name="job_matcher",
            max_retries=1
        )
        
        log.info(f"🤖 LLM returned response, length: {len(rationale)}")
        
        # Clean up the rationale
        rationale = rationale.strip()
        
        # Remove any markdown formatting
        rationale = rationale.replace("**", "").replace("*", "")
        
        # Remove any code blocks
        if "```" in rationale:
            # Extract content between code blocks
            parts = rationale.split("```")
            if len(parts) >= 3:
                rationale = parts[1].strip()
        
        # Ensure it's not too long
        if len(rationale) > 500:
            rationale = rationale[:497] + "..."
        
        # Validate we got something meaningful
        if not rationale or len(rationale) < 20:
            raise ValueError(f"Rationale too short or empty: '{rationale}'")
        
        log.info(f"✅ Successfully generated rationale for {job_title}: {rationale[:80]}...")
        return rationale
        
    except asyncio.TimeoutError:
        log.error(f"⏰ Timeout generating rationale for {job_title}")
        return f"This {job_title} position matches your profile with a {match_score:.0%} compatibility score. Your experience with {', '.join(matched_skills[:3]) if matched_skills else 'relevant technologies'} aligns well with the role requirements at {company_name}."
        
    except Exception as e:
        log.error(f"❌ Failed to generate rationale for {job_title}: {str(e)}")
        import traceback
        log.error(f"Traceback: {traceback.format_exc()}")
        
        # Detailed fallback based on available data
        fallback = f"This {job_title} position at {company_name} shows strong alignment with your profile ({match_score:.0%} match). "
        
        if matched_skills:
            if len(matched_skills) <= 3:
                fallback += f"Your skills in {', '.join(matched_skills)} directly match the job requirements. "
            else:
                fallback += f"You have {len(matched_skills)} matching skills including {', '.join(matched_skills[:3])}. "
        
        if matched_education:
            fallback += f"Your educational background in {matched_education[0]} is relevant for this role. "
        
        fallback += f"With {matched_experience}, you meet the experience criteria for this position."
        
        return fallback


# ⚠️ REMOVED: skill_matcher.py has been completely removed
# V2 agents (ranker_agent, job_matcher_agent, compare_candidate_with_job) use LLM-based matching
# Legacy functions that referenced skill_matcher are deprecated and will fail if called


async def _calculate_vector_similarity_for_job(
    structured_resume: Dict[str, Any],
    job_description: Dict[str, Any]
) -> float:
    """
    Calculate vector similarity between candidate resume and job description.
    Uses the same method as the ranker to ensure consistency.
    
    Args:
        structured_resume: Candidate's structured resume
        job_description: Job description dictionary
        
    Returns:
        Vector similarity score (0-1), or 0.0 if calculation fails
    """
    try:
        # Build job description text query (same format as ranker uses)
        job_title = job_description.get("jobTitle", "") or job_description.get("title", "")
        required_skills = job_description.get("requiredSkills", []) or job_description.get("required_skills", [])
        
        # Create query text from job description (same as ranker)
        jd_text = f"{job_title} {' '.join(required_skills)}"
        
        if not jd_text.strip():
            log.warning("Empty job description text, cannot calculate vector similarity")
            return 0.0
        
        # Build resume text for querying (need candidate ID from resume)
        # For job_matcher, we need to query resumes using job description
        # But we don't have candidate_id here, so we'll use a different approach
        # We can query job descriptions using resume text, but that's backwards
        # Actually, we should query resumes collection using JD text and find the candidate
        # But we don't have candidate_id in structured_resume...
        
        # Alternative: Use the resume text to query job descriptions and find this specific job
        # But that's also backwards...
        
        # Best approach: Query resumes using JD text (like ranker does)
        # But we need the candidate's resume ID. Since we don't have it, we'll skip vector similarity
        # and use 0.0 as fallback. The job_matcher is used for candidate-side matching where
        # we're matching a resume to multiple jobs, so we can't easily get vector similarity
        # without the candidate's resume ID in ChromaDB.
        
        # For now, return 0.0 - this is acceptable since job_matcher is primarily skill-based
        # and the vector similarity is more important in the ranker (corporate side)
        log.debug("Vector similarity calculation skipped in job_matcher (resume ID not available)")
        return 0.0
        
    except Exception as e:
        log.error(f"Error calculating vector similarity: {e}", exc_info=True)
        return 0.0


# ⚠️ REMOVED: skill_matcher.py has been completely removed
# V2 agents use LLM-based matching instead


def _quick_filter_job(
    jd: Dict[str, Any],
    candidate_skills_normalized: set,
    candidate_skills_lower: set,
    candidate_domain: Optional[str] = None  # ✅ NEW: Add candidate domain for domain filtering
) -> bool:
    """
    PRIORITY 2 & 5: Early filtering with pre-computed normalized sets.
    Quick checks before expensive skill matching.
    Returns True if job should be processed, False if it can be skipped.
    
    ✅ ENHANCED: Now includes domain filtering to prevent cross-domain matches
    (e.g., tech candidates → chef jobs, business candidates → trades).
    
    This filters out jobs that are clearly not a match to avoid expensive operations.
    Uses pre-computed normalized sets to avoid repeated normalization.
    """
    # Filter 1: Must have job_id
    job_id = jd.get("job_id") or jd.get("jd_id") or jd.get("id")
    if not job_id:
        return False
    
    # Filter 2: Must have required skills
    jd_skills = jd.get("requiredSkills", []) or jd.get("required_skills", [])
    if not jd_skills or not isinstance(jd_skills, list) or len(jd_skills) == 0:
        return False
    
    # Filter 3: Quick skill overlap check - must have at least some skill match
    # This filters out jobs that are clearly not a fit before expensive processing
    # NOTE: This uses simple exact matching for speed. The actual fuzzy matching
    # happens later in _process_single_job, so we can be lenient here.
    jd_skills_normalized = {str(s).lower().strip() for s in jd_skills}
    jd_skills_set = set(jd_skills_normalized)
    
    # Check for any skill overlap (normalized or lower case)
    # This is a quick check - fuzzy matching will happen later
    has_overlap = (
        bool(jd_skills_set & candidate_skills_normalized) or
        bool(jd_skills_set & candidate_skills_lower)
    )
    
    if not has_overlap:
        # If no exact overlap, check for substring matches (quick fuzzy check)
        # This helps catch cases like "recruitment lifecycle" vs "end-to-end recruitment lifecycle"
        for jd_skill in jd_skills_set:
            for candidate_skill in candidate_skills_normalized | candidate_skills_lower:
                # Check if one skill contains the other (for multi-word skills)
                if len(jd_skill) > 5 and len(candidate_skill) > 5:
                    if jd_skill in candidate_skill or candidate_skill in jd_skill:
                        has_overlap = True
                        break
                # Check for word overlap (at least 2 words in common for multi-word skills)
                jd_words = set(jd_skill.split())
                candidate_words = set(candidate_skill.split())
                if len(jd_words) >= 2 and len(candidate_words) >= 2:
                    common_words = jd_words & candidate_words
                    if len(common_words) >= 2:  # At least 2 words in common
                        has_overlap = True
                        break
            if has_overlap:
                break
    
    if not has_overlap:
        return False
    
    # ✅ NEW: Filter 4: Domain alignment check - prevent cross-domain matches
    # This prevents tech candidates from matching chef jobs, business candidates from matching trades, etc.
    if candidate_domain and candidate_domain.upper() != "OTHER":
        try:
            from agents.ranker import JobDescription
            jd_obj = JobDescription.from_dict(jd)
            job_domain, _ = jd_obj._get_domain_context()
            
            # ✅ FIX: Normalize domains to lowercase for consistent comparison
            # classify_candidate_domain returns uppercase (e.g., "SALES_BUSINESS")
            # JobDescription._get_domain_context returns lowercase (e.g., "sales")
            def normalize_domain(domain: str) -> str:
                """Normalize domain to lowercase format for comparison."""
                if not domain:
                    return "other"
                
                domain_upper = domain.upper()
                # Map uppercase domains from classify_candidate_domain to lowercase equivalents
                domain_mapping = {
                    "SOFTWARE_ENGINEERING": "software_engineering",
                    "DATA_ANALYTICS": "data_analytics",
                    "DATA_SCIENCE": "data_science",
                    "AI_ML": "ai_ml",
                    "BUSINESS_STRATEGIC": "product",  # Maps to product domain
                    "TRADITIONAL_ENGINEERING": "mechanical",  # Maps to mechanical as representative
                    "SALES_BUSINESS": "sales",
                    "MARKETING": "marketing",
                    "FINANCE": "finance",
                    "HR": "hr",
                    "LEGAL": "legal",
                    "MANAGEMENT": "management",
                    "HEALTHCARE": "healthcare",
                    "UI_UX_DESIGN": "ui_ux_design",
                    "CREATIVE_DESIGN": "creative_design",
                    "MANUFACTURING": "manufacturing",
                    "SUPPLY_CHAIN": "supply_chain",
                    "OPERATIONS": "operations",
                    "CUSTOMER_SUPPORT": "customer_support",
                    "CUSTOMER_SUCCESS": "customer_success",
                    "ADMIN": "admin",
                    "EDUCATION": "education",
                    "SCIENCE": "science",
                    "CONSTRUCTION": "construction",
                    "CONSULTING": "consulting",
                    "HOSPITALITY": "hospitality",
                    "RETAIL": "retail",
                    "TRADES": "trades",
                    "REAL_ESTATE": "real_estate",
                    "FITNESS": "fitness",
                    "BEAUTY": "beauty",
                    "AVIATION": "aviation",
                    "TRANSPORTATION": "transportation",
                    "MILITARY": "military",
                    "NONPROFIT": "nonprofit",
                    "PUBLIC_SECTOR": "public_sector",
                    "ARTS": "arts",
                    "SPORTS": "sports",
                    "ENVIRONMENTAL": "environmental",
                    "TECHNICAL_INFRASTRUCTURE": "technical_infrastructure",
                    "SECURITY": "security",
                    "CLEANING_JANITORIAL": "cleaning_janitorial",
                    "PERSONAL_CARE": "personal_care",
                    "CHILDCARE": "childcare",
                    "OTHER": "other",
                }
                # Check if it's an uppercase domain from classify_candidate_domain
                normalized = domain_mapping.get(domain_upper)
                if normalized:
                    return normalized
                # If already lowercase or unknown format, return lowercase version
                return domain.lower() if domain else "other"
            
            candidate_domain_normalized = normalize_domain(candidate_domain)
            job_domain_normalized = normalize_domain(job_domain)
            
            # Define incompatible domain pairs (bidirectional) - using lowercase
            incompatible_domains = {
                # Tech domains incompatible with hospitality/culinary/trades
                ("software_engineering", "hospitality"), ("software_engineering", "trades"),
                ("software_engineering", "real_estate"), ("software_engineering", "fitness"),
                ("data_analytics", "hospitality"), ("data_analytics", "trades"),
                ("data_analytics", "real_estate"), ("data_analytics", "fitness"),
                ("data_science", "hospitality"), ("data_science", "trades"),
                ("ai_ml", "hospitality"), ("ai_ml", "trades"),
                ("ai_ml", "real_estate"), ("ai_ml", "fitness"),
                ("technical_infrastructure", "hospitality"), ("technical_infrastructure", "trades"),
                # Hospitality/culinary incompatible with tech/business
                ("hospitality", "software_engineering"), ("hospitality", "data_analytics"),
                ("hospitality", "data_science"), ("hospitality", "ai_ml"),
                ("hospitality", "mechanical"), ("hospitality", "electrical"),
                ("hospitality", "finance"), ("hospitality", "sales"),
                # Business incompatible with trades/culinary/hospitality
                ("finance", "trades"), ("finance", "hospitality"), ("finance", "real_estate"),
                ("sales", "trades"), ("sales", "hospitality"), ("sales", "real_estate"),
                ("marketing", "trades"), ("marketing", "hospitality"),
                # Trades incompatible with tech/business
                ("trades", "software_engineering"), ("trades", "data_analytics"),
                ("trades", "ai_ml"), ("trades", "finance"), ("trades", "sales"),
                # Real estate incompatible with tech
                ("real_estate", "software_engineering"), ("real_estate", "data_analytics"),
                ("real_estate", "ai_ml"), ("real_estate", "mechanical"),
                # Fitness incompatible with tech/business
                ("fitness", "software_engineering"), ("fitness", "data_analytics"),
                ("fitness", "finance"), ("fitness", "sales"),
            }
            
            # Check if domains are incompatible (using normalized domains)
            if (candidate_domain_normalized, job_domain_normalized) in incompatible_domains or \
               (job_domain_normalized, candidate_domain_normalized) in incompatible_domains:
                log.debug(
                    f"⚠️ Domain mismatch filter: candidate={candidate_domain} ({candidate_domain_normalized}), "
                    f"job={job_domain} ({job_domain_normalized}), "
                    f"job_id={jd.get('job_id', 'unknown')}, job_title={jd.get('jobTitle', 'unknown')}"
                )
                return False
            
            # Log domain alignment for debugging
            if candidate_domain_normalized == job_domain_normalized:
                log.debug(f"✅ Domain match: candidate={candidate_domain_normalized}, job={job_domain_normalized}")
            else:
                log.debug(f"✅ Domain compatible: candidate={candidate_domain_normalized}, job={job_domain_normalized}")
        except Exception as e:
            log.debug(f"⚠️ Could not check domain for job {jd.get('job_id', 'unknown')}: {e}")
            # On error, allow through (fail open) - don't block on domain check failures
    
    # Additional check: At least EARLY_FILTER_THRESHOLD skill match
    # This ensures we only process jobs with meaningful skill overlap
    # Count unique matches (union of both intersections to avoid double counting)
    matched_skills_set = (jd_skills_set & candidate_skills_normalized) | (jd_skills_set & candidate_skills_lower)
    matched_skills_count = len(matched_skills_set)
    match_ratio = matched_skills_count / len(jd_skills) if jd_skills else 0.0
    
    # RELAXED FILTER: If we detected overlap via substring/word-match (but no exact strings),
    # bypass the threshold check and let the full fuzzy matcher decide.
    # Only enforce threshold when we have zero exact matches AND zero fuzzy overlap.
    # This prevents dropping jobs like "Doppler" vs "Doppler Studies" that Chroma retrieved.
    if match_ratio < JobMatcherConfig.EARLY_FILTER_THRESHOLD and matched_skills_count == 0:
        # has_overlap was True (from substring/word check), so don't filter out
        # ⚠️ NOTE: V2 agents use LLM-based matching instead of perform_skill_matching (removed)
        pass
    
    return True


def generate_fallback_rationale_for_job(
    matched_skills: List[str],
    unmatched_skills: List[str],
    required_skills_count: int,
    match_score: float,
    job_title: str = "",
    company_name: str = ""
) -> str:
    """
    Generate template rationale for job matching (second person - addressing candidate).
    
    Args:
        matched_skills: List of matched skills
        unmatched_skills: List of unmatched skills
        required_skills_count: Total required skills
        match_score: Final match score
        job_title: Job title (optional)
        company_name: Company name (optional)
        
    Returns:
        Human-readable rationale in second person
    """
    matched_count = len(matched_skills)
    unmatched_count = len(unmatched_skills)
    
    # Calculate ACTUAL skill match percentage
    skill_percentage = int((matched_count / max(1, required_skills_count)) * 100)
    
    # Determine match quality based on SKILL PERCENTAGE
    if skill_percentage >= 70:
        quality = "strong"
        fit = "excellent fit"
    elif skill_percentage >= 50:
        quality = "moderate"
        fit = "reasonable fit"
    elif skill_percentage >= 30:
        quality = "fair"
        fit = "partial fit"
    else:
        quality = "weak"
        fit = "partial fit"
    
    # Build rationale in second person
    parts = []
    
    # Opening with skill-based percentage
    if job_title:
        parts.append(
            f"This {job_title} position shows a {quality} match for your profile "
            f"({skill_percentage}% skill overlap - you have {matched_count} out of {required_skills_count} required skills)."
        )
    else:
        parts.append(
            f"This position shows a {quality} match for your profile "
            f"({skill_percentage}% skill overlap - you have {matched_count} out of {required_skills_count} required skills)."
        )
    
    # Strengths
    if matched_skills:
        if matched_count <= 3:
            skills_str = ", ".join(matched_skills)
            parts.append(f"Your skills in {skills_str} directly align with the role requirements.")
        else:
            top_skills = ", ".join(matched_skills[:3])
            remaining = matched_count - 3
            parts.append(
                f"Your skills including {top_skills} "
                f"and {remaining} others match the job requirements."
            )
    
    # Weaknesses
    if unmatched_skills and skill_percentage < 100:
        if unmatched_count <= 3:
            gaps_str = ", ".join(unmatched_skills)
            parts.append(
                f"However, you may need to develop experience in {gaps_str} "
                f"to fully meet all requirements."
            )
        else:
            top_gaps = ", ".join(unmatched_skills[:3])
            remaining_gaps = unmatched_count - 3
            parts.append(
                f"However, you may need to develop experience in {top_gaps} "
                f"and {remaining_gaps} other areas to fully meet all requirements."
            )
    
    # Overall assessment
    if skill_percentage >= 70:
        parts.append(
            f"Overall, this represents an {fit} with strong "
            f"alignment to the role requirements."
        )
    elif skill_percentage >= 50:
        parts.append(
            f"Overall, this represents a {fit}, though some skill "
            f"gaps would need to be addressed."
        )
    else:
        parts.append(
            f"Overall, this represents a {fit} that may require "
            f"additional skill development."
        )
    
    return " ".join(parts)


async def generate_llm_rationale_for_job(
    jd_text: str,
    sanitized_resume: Dict[str, Any],
    matched_skills: List[str],
    unmatched_skills: List[str],
    match_score: float = 0.0,
    required_skills_count: int = 0,
    job_title: str = "",
    company_name: str = "",
    recruiter_questions: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Generate rationale using LLM for job matching (second person - addressing candidate).
    
    OPTIMIZED: Uses circuit breaker to prevent cascading failures.
    ✅ ENHANCED: Now includes recruiter questions and candidate answers from payload
    
    Returns:
        Rationale text in second person (never empty - uses fallback if LLM fails)
    """
    # Check circuit breaker before proceeding
    can_proceed = await _llm_circuit_breaker.can_proceed()
    if not can_proceed:
        log.warning("🚫 Circuit breaker OPEN: Skipping LLM call, using fallback")
        return generate_fallback_rationale_for_job(
            matched_skills,
            unmatched_skills,
            required_skills_count or len(matched_skills) + len(unmatched_skills),
            match_score,
            job_title,
            company_name
        )
    
    # ✅ NEW: Format recruiter questions and answers
    recruiter_qa_section = ""
    if recruiter_questions:
        if isinstance(recruiter_questions, list):
            qa_pairs = []
            for qa in recruiter_questions:
                if isinstance(qa, dict):
                    q = qa.get("question") or qa.get("Question") or ""
                    a = qa.get("answer") or qa.get("Answer") or ""
                    if q and a:
                        qa_pairs.append(f"Q: {q}\nA: {a}")
            if qa_pairs:
                recruiter_qa_section = "\n\n**Recruiter Questions & Candidate Answers:**\n" + "\n\n".join(qa_pairs[:5])  # Limit to top 5
                log.debug(f"✅ Including {len(qa_pairs)} recruiter Q&A pairs in rationale")
        elif isinstance(recruiter_questions, str):
            recruiter_qa_section = f"\n\n**Recruiter Questions & Answers:**\n{recruiter_questions[:500]}"
    
    resume_text = json.dumps(sanitized_resume, indent=2, ensure_ascii=False)
    
    # ✅ FIXED: Send full resume and JD - no truncation for better analysis
    # Note: Modern LLMs can handle large contexts, so we send complete data
    
    prompt = f"""You are an expert job-matching assistant. Generate a brief, professional rationale (2-3 sentences) explaining why this job is a good match for the candidate. IMPORTANT: Address the candidate directly using second person ("you", "your", "you have", etc.).

Job Description:
{jd_text}

Candidate Resume Summary:
{resume_text}

Skills Analysis (already computed):
- Matched Skills ({len(matched_skills)}): {', '.join(matched_skills[:5]) if matched_skills else 'None'}
- Missing Skills ({len(unmatched_skills)}): {', '.join(unmatched_skills[:5]) if unmatched_skills else 'None'}
- Match Score: {match_score:.0%}
{recruiter_qa_section}

Provide a concise 2-3 sentence rationale explaining:
1. Why this job is a good match for the candidate (use "you", "your", "you have")
2. Key strengths that align with the position (use "your skills", "your experience")
3. ✅ NEW: Reference relevant recruiter questions/answers if provided (e.g., "Your answer about X shows Y")
4. Any critical gaps or areas for improvement (use "you may need", "you could benefit")

CRITICAL: Always use second person pronouns (you, your, you have, your experience, etc.). Never use third person (the candidate, they, their, etc.).

Return ONLY a JSON object:
{{
    "rationale": "Your 2-3 sentence professional explanation here in second person"
}}

Example: {{"rationale": "This {job_title or 'position'} at {company_name or 'the company'} aligns well with your technical background, particularly your experience with {', '.join(matched_skills[:3]) if matched_skills else 'relevant technologies'}. Your skills in these areas directly match the role requirements. However, you may want to develop experience in {', '.join(unmatched_skills[:2]) if unmatched_skills else 'additional areas'} to fully meet all requirements."}}

Be specific, professional, and always use second person. Do NOT include text outside the JSON.
"""
    
    # Strategy 1: Try structured output with timeout (centralized invoke_structured_llm)
    try:
        class _Rationale(BaseModel):
            rationale: str

        log.debug("Attempting structured LLM rationale generation for job")
        start_time = time.time()
        structured, raw_response = await asyncio.wait_for(
            invoke_structured_llm(
                prompt,
                _Rationale,
                task_type=TaskType.TEXT_GENERATION,
                preferred_model=_settings.GEMINI_MODEL,
                agent_name="job_matcher",
                max_output_tokens=1500,
                temperature=0.2,
                timeout=float(JobMatcherConfig.RATIONALE_TIMEOUT),
                include_raw=True,
                raise_on_fallback=False,
            ),
            timeout=JobMatcherConfig.RATIONALE_TIMEOUT,
        )
        rationale = structured.model_dump().get("rationale", "").strip() if structured else ""
        if raw_response is not None:
            record_direct_llm_usage(
                response=raw_response,
                prompt=prompt,
                content=rationale,
                agent_name="job_matcher",
                start_time=start_time,
                model_name=_settings.GEMINI_MODEL,
            )

        if rationale and len(rationale) > 20:
            log.debug("✅ Structured LLM rationale generated successfully")
            await _llm_circuit_breaker.record_success()
            return rationale
        else:
            log.warning("⚠️ Structured LLM returned empty/short rationale")
            raise ValueError("Empty structured response")
            
    except asyncio.TimeoutError:
        log.warning("⏱️ Structured LLM rationale timed out")
        await _llm_circuit_breaker.record_failure()
    except Exception as e:
        log.warning(f"⚠️ Structured LLM rationale failed: {str(e)[:100]}")
        await _llm_circuit_breaker.record_failure()
    
    # Strategy 2: Try raw LLM with JSON parsing
    try:
        log.debug("Attempting raw LLM rationale generation for job")
        raw_response = await asyncio.wait_for(
            invoke_llm(prompt, agent_name="job_matcher", max_output_tokens=1500),
            timeout=JobMatcherConfig.RATIONALE_TIMEOUT
        )
        
        # Try to extract JSON
        if raw_response:
            try:
                response_data = json.loads(raw_response)
                rationale = response_data.get("rationale", "").strip()
                
                if rationale and len(rationale) > 20:
                    log.debug("✅ Raw LLM rationale extracted successfully")
                    await _llm_circuit_breaker.record_success()
                    return rationale
            except:
                # Try regex extraction
                match = re.search(r'"rationale"\s*:\s*"([^"]+)"', raw_response, re.DOTALL)
                if match:
                    rationale = match.group(1).strip()
                    if len(rationale) > 20:
                        log.debug("✅ Raw LLM rationale extracted via regex")
                        await _llm_circuit_breaker.record_success()
                        return rationale
        
        log.warning("⚠️ Raw LLM response parsing failed")
    except asyncio.TimeoutError:
        log.warning("⏱️ Raw LLM rationale timed out")
        await _llm_circuit_breaker.record_failure()
    except Exception as e:
        log.warning(f"⚠️ Raw LLM rationale failed: {str(e)[:100]}")
        await _llm_circuit_breaker.record_failure()
    
    # Strategy 3: ALWAYS return fallback rationale
    log.info("📝 Generating fallback rationale (LLM unavailable)")
    return generate_fallback_rationale_for_job(
        matched_skills,
        unmatched_skills,
        required_skills_count or len(matched_skills) + len(unmatched_skills),
        match_score,
        job_title,
        company_name
    )


async def _process_single_job(
    jd: Dict[str, Any],
    candidate_skills: set,
    structured_resume: Optional[Dict[str, Any]] = None,
    threshold: float = 0.20,
    idx: int = 0,
    total: int = 0,
    candidate_skills_normalized: Optional[set] = None,
    candidate_skills_lower: Optional[set] = None,
    vector_similarities: Optional[Dict[str, float]] = None,
    recruiter_questions: Optional[List[Dict[str, str]]] = None,
    candidate_domain: Optional[str] = None  # ✅ NEW: Add candidate domain for domain filtering
) -> Dict[str, Any] | None:
    """
    Process a single job description for matching (async helper for parallel processing).
    
    Returns:
        Match result dict for ALL jobs (regardless of threshold)
        Rationale will only be generated for top 15 matches later
    """
    try:
        # PRIORITY 2 & 5: Early filtering with pre-computed normalized sets
        if candidate_skills_normalized is None:
            candidate_skills_normalized = candidate_skills
        if candidate_skills_lower is None:
            # Run normalization in thread pool (CPU-intensive)
            candidate_skills_lower = await run_cpu_intensive(
                lambda: {str(s).lower().strip() for s in candidate_skills}
            )
        
        if not _quick_filter_job(jd, candidate_skills_normalized, candidate_skills_lower, candidate_domain):
            return None
        
        # Get job details
        job_id = jd.get("job_id") or jd.get("jd_id") or jd.get("id")
        
        if not job_id:
            return None
        
        job_title = (jd.get("jobTitle") or jd.get("title") or 
                    jd.get("job_title") or "Unknown Position")
        company_name = (jd.get("company") or jd.get("companyName") or 
                      jd.get("company_name") or "Unknown Company")
        location = jd.get("location") or "Remote"
        
        # Extract required skills
        jd_skills = jd.get("requiredSkills", []) or jd.get("required_skills", [])
        if not jd_skills or not isinstance(jd_skills, list):
            return None
        
        required_skills_list = jd_skills
        if not required_skills_list:
            return None
        
        # ⚠️ DEPRECATED: _process_single_job is not used by V2 job_matcher_agent
        # V2 uses LLM-based matching instead. This function would fail as skill_matcher was removed.
        log.warning("⚠️ _process_single_job is deprecated - skill_matcher was removed")
        raise NotImplementedError(
            "_process_single_job is deprecated. "
            "V2 job_matcher_agent uses LLM-based matching. This function should not be called."
        )
        
        # OPTIMIZATION: Don't match education yet - do it only if job passes threshold
        # This avoids expensive education matching for jobs that will be filtered out
        matched_education = []  # Will be populated later if job passes threshold
        
        # Calculate experience match score
        experience_match_score = 1.0  # Default to full score if no experience requirement
        candidate_years = 0.0
        required_years = 0.0
        max_years = None  # For ranges like "3-5 years"
        
        try:
            from utils.experience_years import calculate_total_experience_years
            from agents.ranker import calculate_experience_match_score
            import re
            
            # Extract candidate experience from structured_resume (run in thread pool if CPU-intensive)
            if structured_resume:
                # First, try to use pre-calculated total_experience_years if available
                total_exp_str = structured_resume.get("total_experience_years", "")
                if total_exp_str:
                    # Parse strings like "8 years 4 months" or "8.3 years"
                    try:
                        # Try to extract number from string like "8 years 4 months"
                        years_match = re.search(r'(\d+(?:\.\d+)?)', str(total_exp_str))
                        if years_match:
                            candidate_years = float(years_match.group(1))
                            # If there are months, add them as fraction of year
                            months_match = re.search(r'(\d+)\s*months?', str(total_exp_str).lower())
                            if months_match:
                                months = float(months_match.group(1))
                                candidate_years += months / 12.0
                            log.debug(f"✅ Used pre-calculated total_experience_years: {candidate_years:.1f} years")
                        else:
                            # Fallback to calculating from work experience
                            raise ValueError("Could not parse total_experience_years")
                    except (ValueError, AttributeError) as e:
                        log.debug(f"Could not parse total_experience_years '{total_exp_str}', calculating from work experience: {e}")
                        # Fall through to calculate from work experience
                        total_exp_str = None
                
                # If pre-calculated value not available, calculate from work experience
                if not total_exp_str or candidate_years == 0.0:
                    # Check multiple field names (same pattern as anonymize_candidate_info)
                    work_exp = (
                        structured_resume.get("WorkExperience") or
                        structured_resume.get("experience") or
                        structured_resume.get("Experience") or
                        structured_resume.get("WORK_EXPERIENCE") or
                        []
                    )
                if work_exp:
                    candidate_years = await run_cpu_intensive(
                        calculate_total_experience_years,
                        work_exp
                    )
                    log.debug(f"✅ Calculated experience from work_experience: {candidate_years:.1f} years")
            
            # Extract required experience from JD (with range support) - run regex in thread pool
            required_exp = jd.get("experience", "")
            if required_exp:
                def parse_experience(exp_str):
                    """Helper function to parse experience (CPU-intensive regex)."""
                    # Pattern 1: Range "X-Y years"
                    range_pattern = re.findall(r'(\d+)\s*-\s*(\d+)\s*years?\s*(?:of\s*)?(?:experience|exp)', str(exp_str).lower())
                    if range_pattern:
                        return float(range_pattern[0][0]), float(range_pattern[0][1])
                    else:
                        # Pattern 2: Single value "X years" or "X+ years"
                        exp_patterns = re.findall(r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)', str(exp_str).lower())
                        if exp_patterns:
                            return float(exp_patterns[0]), None
                        else:
                            # Pattern 3: "minimum X years"
                            min_exp_patterns = re.findall(r'minimum\s*(?:of\s*)?(\d+)\+?\s*years?', str(exp_str).lower())
                            if min_exp_patterns:
                                return float(min_exp_patterns[0]), None
                    return 0.0, None
                
                required_years, max_years = await run_cpu_intensive(
                    parse_experience,
                    required_exp
                )
            
            # Calculate experience match score (with range support)
            if required_years > 0 or candidate_years > 0:
                experience_match_score = calculate_experience_match_score(candidate_years, required_years, max_years)
                log.debug(
                    f"📊 Experience match for job {job_id}: "
                    f"candidate={candidate_years:.1f} years, required={required_years:.1f} years, "
                    f"score={experience_match_score:.3f}"
                )
        except Exception as e:
            log.debug(f"⚠️ Could not calculate experience match for job {job_id}: {e}")
            experience_match_score = 1.0  # Default to full score on error
        
        # Calculate match score
        # Get vector similarity from pre-computed mapping (if available)
        vector_similarity = 0.0
        if vector_similarities is not None:
            # Try to get vector similarity for this job
            job_id_str = str(job_id)
            if job_id_str in vector_similarities:
                vector_similarity = vector_similarities[job_id_str]
            else:
                # Try alternative ID formats
                for alt_id in [jd.get("jd_id"), jd.get("id")]:
                    if alt_id and str(alt_id) in vector_similarities:
                        vector_similarity = vector_similarities[str(alt_id)]
                        break
        
        matched_count = len(matched_skills)
        required_count = len(required_skills_list)
        
        skill_match_percentage_decimal, match_score = compute_skill_match_score(
            matched_count=matched_count,
            required_count=required_count,
            similarity=vector_similarity,
            experience_match=experience_match_score
        )
        
        # Calculate match percentage for smart LLM decision
        match_percentage = matched_count / len(required_skills_list) if required_skills_list else 0.0
        
        # Smart LLM Rationale Decision
        # OPTIMIZATION: Only use LLM for borderline cases (30-85% match by default)
        # - Very high match (>85%): Use template (obvious good match)
        # - Very low match (<30%): Use template (obvious poor match)
        # - Medium match (30-85%): Use LLM (nuanced analysis needed)
        use_llm_rationale = (
            JobMatcherConfig.LLM_RATIONALE_MIN_MATCH <= match_percentage <= JobMatcherConfig.LLM_RATIONALE_MAX_MATCH and
            matched_count > 0  # Has some matches (not completely mismatched)
        )
        
        # Generate rationale (deferred - will be done in parallel batch later)
        # Store rationale generation info for batch processing
        rationale = None  # Will be generated in batch
        needs_llm_rationale = use_llm_rationale
        rationale_context = {
            "job_id": job_id,
            "job_title": job_title,
            "company_name": company_name,
            "location": location,
            "matched_skills": matched_skills,
            "unmatched_skills": unmatched_skills,
            "required_skills_list": required_skills_list,
            "match_score": match_score,
            "match_percentage": match_percentage,
            "structured_resume": structured_resume,
            "recruiter_questions": recruiter_questions  # ✅ NEW: Include Q&A in context
        } if use_llm_rationale else None
        
        # Format experience match status (same as ranker)
        experience_match_status = ""
        if required_years > 0:
            if max_years and max_years > required_years:
                # Range specified (e.g., "3-5 years")
                if required_years <= candidate_years <= max_years:
                    experience_match_status = f"{candidate_years:.1f} years (within ideal range of {required_years:.1f}-{max_years:.1f} years)"
                elif candidate_years < required_years:
                    experience_match_status = f"{candidate_years:.1f} years (below minimum requirement of {required_years:.1f} years, range: {required_years:.1f}-{max_years:.1f})"
                else:
                    experience_match_status = f"{candidate_years:.1f} years (above preferred range of {required_years:.1f}-{max_years:.1f} years)"
            else:
                # Single value requirement
                if candidate_years >= required_years:
                    if candidate_years > required_years * 2:
                        experience_match_status = f"{candidate_years:.1f} years (exceeds requirement of {required_years:.1f} years - over-qualified)"
                    else:
                        experience_match_status = f"{candidate_years:.1f} years (meets requirement of {required_years:.1f} years)"
                else:
                    experience_match_status = f"{candidate_years:.1f} years (below requirement of {required_years:.1f} years)"
        elif candidate_years > 0:
            experience_match_status = f"{candidate_years:.1f} years (no requirement specified)"
        else:
            experience_match_status = "0 years (no experience listed)"
        
        # Return ALL jobs with match scores (even if below threshold)
            # OPTIMIZATION: Only match education for jobs that pass threshold
        # This avoids expensive education matching for very low matches
        if match_score >= threshold and not matched_education and structured_resume:
                try:
                    # Run education matching in thread pool (CPU-intensive)
                    matched_education = await run_cpu_intensive(
                        match_education,
                        structured_resume,
                        jd
                    )
                except Exception as e:
                    log.debug(f"⚠️ Could not match education for job {job_id}: {e}")
                    matched_education = []
            
        # Return result for ALL jobs (regardless of threshold)
        # Rationale will only be generated for top 15 matches later
        result = {
                "job_id": str(job_id),
                "job_title": job_title,
                "company_name": company_name,
                "location": location,
                "match_score": round(match_score, 2),
                "matched_skills": matched_skills,
                "unmatched_skills": unmatched_skills,
                "matched_education": matched_education,
                "matched_experience": "",  # Deprecated - use experience_match_status instead
            "rationale": None,  # Will be generated in batch (only for top 15)
            "needs_llm_rationale": needs_llm_rationale if match_score >= threshold else False,  # Only top matches get rationale
                "matching_method": "skill_based_matching",
                # Experience information (same format as ranker)
                "candidate_experience_years": round(candidate_years, 1),
                "required_experience_years": round(required_years, 1) if required_years > 0 else None,
                "max_experience_years": round(max_years, 1) if max_years else None,  # For ranges
                "experience_match_status": experience_match_status,
                "experience_match_score": round(experience_match_score, 3)
            }
            
        # Only store context if LLM rationale is needed (and score >= threshold)
        if needs_llm_rationale and rationale_context and match_score >= threshold:
                result["rationale_context"] = rationale_context
            
            
        return result
        
    except Exception as e:
        log.debug(f"Error processing job {jd.get('job_id', 'unknown')}: {e}")
        return None


async def analyze_job_batch_with_llm(
    candidate_skills: List[str],
    candidate_experience: str,
    candidate_education: str,
    candidate_location: str,
    jobs_batch: List[Dict[str, Any]],
    batch_idx: int = 0,
    candidate_certifications: Optional[List[str]] = None,
    candidate_projects: Optional[List[Dict[str, Any]]] = None,
    professional_summary: Optional[str] = None,
    candidate_domain: Optional[str] = None,
    candidate_role_context: Optional[str] = None,
    structured_resume: Optional[Dict[str, Any]] = None,
    recruiter_questions: Optional[List[Dict[str, str]]] = None,
    interview_feedback: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Analyze a single job (or small batch) using LLM with the FULL structured resume.
    
    Returns list of dicts with:
    - job_id
    - skills_matched: List[str]
    - match_score: float (0.0-1.0)
    - rationale: str
    
    Args:
        candidate_skills: List of candidate's skills
        candidate_experience: Candidate's experience
        candidate_education: Candidate's education
        candidate_location: Candidate's location
        jobs_batch: List of job dicts (ideally 1 job for best quality)
        batch_idx: Batch index for logging
        candidate_certifications: Optional list of certification names for LLM to consider
        candidate_projects: Optional list of project dicts for LLM to consider
        professional_summary: Optional summary text for LLM to consider
        structured_resume: Full structured resume dict (sanitized) for deep analysis
        
    Returns:
        List of analyzed jobs with skills_matched, match_score, rationale
    """
    candidate_certifications = candidate_certifications or []
    candidate_projects = candidate_projects or []
    cand_skills_set = set(candidate_skills) if candidate_skills else set()

    # Pre-validate skills for each job BEFORE LLM (skill dropping first, then rationale)
    pre_validated_by_job: Dict[str, List[str]] = {}
    for job in jobs_batch:
        required_skills = job.get("requiredSkills", [])
        if isinstance(required_skills, str):
            required_skills = [s.strip() for s in required_skills.split(",")]
        job_id = str(job.get("job_id") or job.get("jd_id") or job.get("id", ""))
        validated = validate_skills_against_candidate(required_skills, cand_skills_set, required_skills)
        pre_validated_by_job[job_id] = validated

    # Format jobs for LLM
    jobs_text = ""
    for i, job in enumerate(jobs_batch, start=1):
        job_title = job.get("jobTitle") or job.get("title", "N/A")
        required_skills = job.get("requiredSkills", [])
        if isinstance(required_skills, str):
            required_skills = [s.strip() for s in required_skills.split(",")]
        job_id = str(job.get("job_id") or job.get("jd_id") or job.get("id", f"job_{i}"))
        pre_validated = pre_validated_by_job.get(job_id, [])
        
        preferred_skills = job.get("preferredSkills") or []
        work_mode = job.get("workMode") or ""
        must_have_skills = job.get("_must_have_skills") or []
        
        company = job.get("company", "N/A")
        experience = job.get("experience") or job.get("experienceRequired", "N/A")
        job_location = job.get("location") or job.get("jobLocation") or "N/A"
        job_education = job.get("educationRequired") or job.get("education") or job.get("requiredEducation") or "Not specified"
        
        jd_description = job.get("fullJobDescription") or job.get("description") or ""
        jd_responsibilities = job.get("responsibilities") or job.get("keyResponsibilities") or []
        if isinstance(jd_responsibilities, list):
            jd_responsibilities_text = "\n".join([f"- {r}" for r in jd_responsibilities])
        else:
            jd_responsibilities_text = str(jd_responsibilities) if jd_responsibilities else ""
        
        if must_have_skills:
            must_have_text = f"Must-Have Skills (PRIORITY - {len(must_have_skills)}): {', '.join(must_have_skills)}"
            other_skills = [s for s in required_skills if s not in must_have_skills]
            if other_skills:
                other_skills_text = f"\nAdditional Required Skills (from JD parsing - {len(other_skills)}): {', '.join(other_skills)}"
            else:
                other_skills_text = ""
            skills_section = must_have_text + other_skills_text
        else:
            skills_section = f"Required Skills ({len(required_skills)}): {', '.join(required_skills) if required_skills else 'Not listed'}"
        
        responsibilities_section = ""
        if jd_responsibilities_text:
            responsibilities_section = f"\nKey Responsibilities:\n{jd_responsibilities_text}\n"
        
        verified_skills_line = f"VERIFIED skills_matched (use ONLY these): {', '.join(pre_validated) if pre_validated else 'None'}"
        jobs_text += f"""
--- JOB {i} ---
ID: {job_id}
Title: {job_title}
{verified_skills_line}
Company: {company}
Location: {job_location}
Work Mode: {work_mode}
{skills_section}
Preferred Skills ({len(preferred_skills)}): {', '.join(preferred_skills) if preferred_skills else 'None'}
Experience Required: {experience}
Education Required: {job_education}
Full Job Description: {jd_description}{responsibilities_section}
"""
    
    # Build candidate section: prefer full resume JSON when available (matches compare flow quality)
    if structured_resume:
        resume_json = json.dumps(structured_resume, indent=2, ensure_ascii=False)
        candidate_section = f"""CANDIDATE FULL RESUME (JSON):
{resume_json}
"""
    else:
        certs_line = f"Certifications: {', '.join(candidate_certifications)}" if candidate_certifications else "Certifications: None listed"
        projects_summary = ""
        if candidate_projects:
            names = []
            for p in candidate_projects[:10]:
                if isinstance(p, dict):
                    names.append(p.get("project_name") or p.get("name") or p.get("title") or str(p)[:50])
                else:
                    names.append(str(p)[:50])
            projects_summary = f"Projects: {', '.join(names)}"
        else:
            projects_summary = "Projects: None listed"
        summary_line = f"Professional Summary: {professional_summary[:500]}" if professional_summary else ""
        candidate_section = f"""CANDIDATE:
Skills ({len(candidate_skills)}): {', '.join(candidate_skills)}
Experience: {candidate_experience} years
Education: {candidate_education}
Location: {candidate_location or 'N/A'}
{certs_line}
{projects_summary}
{summary_line}
"""
    
    candidate_context_block = ""
    if candidate_domain and candidate_role_context:
        candidate_context_block = f"""
CANDIDATE CONTEXT (use for contextual scoring):
- Candidate's professional domain: {candidate_domain}
- Candidate's role/career context: {candidate_role_context}
- You MUST be aware of both the candidate's domain and each job's domain/role. Only recommend jobs that ALIGN with the candidate's career context.
- Do NOT recommend jobs in unrelated domains (e.g. do not match education/teaching candidates to data analyst or software engineering roles; do not match data/tech candidates to teaching-only roles unless the job explicitly fits).
- When in doubt, score cross-domain or misaligned roles lower (e.g. 0.2-0.4) and state in rationale that the role does not align with the candidate's professional domain.

"""

    # Static system instruction for Gemini context caching (~1,000 tokens cached per session)
    # Resume and jobs stay in prompt to avoid hallucination (reverted commit put resume in system_instruction)
    system_instruction = f"""You are an expert job matching AI. Perform deep CONTEXTUAL analysis for candidates against jobs.

CRITICAL - Skill validation happens BEFORE your analysis: Each job has "VERIFIED skills_matched" (deterministic match from resume). You MUST use ONLY those skills for skills_matched. Do NOT add any skills not in the verified list. Generate match_score and rationale based on the verified skills.

For EACH job, analyze the candidate's COMPLETE resume and return (same matching logic as ranker and job_matcher):
1. **skills_matched**: Skills from that job's "Required Skills" list that the candidate has (use ONLY the VERIFIED list)
   - ⚠️ PRIORITY: Must match skills from "Must-Have Skills" section first (if present)
   - ⚠️ Then consider "Additional Required Skills" (from JD parsing)
   - ⚠️ DO NOT include education degrees/majors in skills_matched
   - ⚠️ ONLY include technical/professional skills (e.g., "Python", "React", "AWS", "Agile")
   - Education will be checked separately below
2. **preferred_skills_matched**: Skills from the "Preferred Skills" list that the candidate has (bonus points)
3. **match_score**: Overall match score from 0-100 (integer). Use ORDERED CONTEXTUAL ANALYSIS:
{ORDERED_ANALYSIS_STEPS_UNIFIED}
{SCORING_GUIDELINES_0_100}
4. **domain_relevant_experience_years**: Estimated years of experience RELEVANT to this job's domain/role (e.g. TypeScript, AI/ML, RAG). Infer from work history and role titles; if unclear use total experience. Return a number (e.g. 4.5) or null.
5. **positive_rationale**: 2-3 sentences on why you got this score (what aligned—must-have/required skills, preferred skills, experience, education, certifications, projects). Be specific. Use second person: "you", "your", "you have".
6. **negative_rationale**: 2-3 sentences on where the rest went (gaps—missing/weak skills, experience shortfalls, education mismatch, concerns). Be specific. Use second person: "you", "your", "you could", "you may need".
7. **rationale**: Optional one-line overall summary in second person; can leave empty if positive_rationale + negative_rationale cover it.
8. **how_to_improve**: 2-4 sentences telling the candidate what to do to improve. Be specific and actionable. Use second person: "You can", "Your profile", "Consider adding", "Take assessments". Gently nudge to use relevant platform features (update profile/resume, take recommended assessments, check course recommendations, practice mock interviews). Do NOT mention career coach.

VOICE: Write positive_rationale, negative_rationale, rationale, and how_to_improve in SECOND PERSON only. Address the candidate directly ("you", "your", "you have", "your experience"). Never use third person ("the candidate", "they", "their").

{CRITICAL_RULES}
- positive_rationale, negative_rationale, and how_to_improve must be complete sentences with proper punctuation, all in second person
- ⚠️ CERTIFICATIONS: If the candidate has certifications, do NOT say they have "no certifications" in negative_rationale or how_to_improve. Reference their actual certifications when relevant to the role.
"""

    # Prompt = dynamic content only (candidate, jobs) - enables Gemini to cache system_instruction
    prompt = f"""Analyze this candidate against {len(jobs_batch)} job(s).
{candidate_context_block}
{candidate_section}

JOB(S):
{jobs_text}
"""
    extra_sections = ""
    if recruiter_questions:
        qa_pairs = []
        for qa in recruiter_questions[:5]:
            q = qa.get("question") or qa.get("Question", "")
            a = qa.get("answer") or qa.get("Answer", "")
            if q and a:
                qa_pairs.append(f"Q: {q}\nA: {a}")
        if qa_pairs:
            extra_sections += "\n\nCANDIDATE Q&A RESPONSES:\n" + "\n\n".join(qa_pairs)
            extra_sections += "\n\n⚠️ IMPORTANT: Consider these Q&A responses when analyzing fit. If answers contradict the resume or show lack of required skills, factor this into match_score and rationale.\n"
    if interview_feedback:
        if isinstance(interview_feedback, dict):
            feedback_text = json.dumps(interview_feedback, indent=2, ensure_ascii=False)
        elif isinstance(interview_feedback, list):
            feedback_text = "\n".join(f"- {item}" for item in interview_feedback)
        else:
            feedback_text = str(interview_feedback)
        extra_sections += f"""
POST-INTERVIEW FEEDBACK (recruiter/interviewer notes):
{feedback_text}

⚠️ CRITICAL: This candidate has been interviewed. Use the post-interview feedback to:
1. ADJUST match_score — interview performance should influence the final score.
2. INCORPORATE interview observations into positive_rationale and negative_rationale.
3. For EACH job in the array, add "hire_recommendation": "strong_hire"|"hire"|"lean_hire"|"no_hire" and "hire_rationale": "1-2 sentences summarizing the recommendation."
"""
    json_extra = ',\n    "hire_recommendation": "strong_hire|hire|lean_hire|no_hire",\n    "hire_rationale": "1-2 sentences summarizing recommendation"' if interview_feedback else ""

    # CONTEXT CACHING: Put resume + static prompts in system_instruction for Gemini caching.
    # Same content sent N times → cached after first call → 15-25% cost savings, lower latency.
    system_instruction = f"""You are an expert job matching AI. Perform deep CONTEXTUAL analysis for candidates against jobs.
{candidate_context_block}
{candidate_section}
{extra_sections}
For EACH job, analyze the candidate's COMPLETE resume and return (same matching logic as ranker and job_matcher):
1. **skills_matched**: Skills from that job's "Required Skills" list that the candidate has
   - ⚠️ PRIORITY: Must match skills from "Must-Have Skills" section first (if present)
   - ⚠️ Then consider "Additional Required Skills" (from JD parsing)
   - ⚠️ DO NOT include education degrees/majors in skills_matched
   - ⚠️ ONLY include technical/professional skills (e.g., "Python", "React", "AWS", "Agile")
   - Education will be checked separately below
2. **preferred_skills_matched**: Skills from the "Preferred Skills" list that the candidate has (bonus points)
3. **match_score**: Overall match score from 0-100 (integer). Use ORDERED CONTEXTUAL ANALYSIS:
{ORDERED_ANALYSIS_STEPS_UNIFIED}
{SCORING_GUIDELINES_0_100}
4. **domain_relevant_experience_years**: Estimated years of experience RELEVANT to this job's domain/role (e.g. TypeScript, AI/ML, RAG). Infer from work history and role titles; if unclear use total experience. Return a number (e.g. 4.5) or null.
5. **positive_rationale**: 2-3 sentences on why you got this score (what aligned—must-have/required skills, preferred skills, experience, education, certifications, projects). Be specific. Use second person: "you", "your", "you have".
6. **negative_rationale**: 2-3 sentences on where the rest went (gaps—missing/weak skills, experience shortfalls, education mismatch, concerns). Be specific. Use second person: "you", "your", "you could", "you may need".
7. **rationale**: Optional one-line overall summary in second person; can leave empty if positive_rationale + negative_rationale cover it.
8. **how_to_improve**: 2-4 sentences telling the candidate what to do to improve. Be specific and actionable. Use second person: "You can", "Your profile", "Consider adding", "Take assessments". Gently nudge to use relevant platform features (update profile/resume, take recommended assessments, check course recommendations, practice mock interviews). Do NOT mention career coach.

VOICE: Write positive_rationale, negative_rationale, rationale, and how_to_improve in SECOND PERSON only. Address the candidate directly ("you", "your", "you have", "your experience"). Never use third person ("the candidate", "they", "their").

{CRITICAL_RULES}
- positive_rationale, negative_rationale, and how_to_improve must be complete sentences with proper punctuation, all in second person
- ⚠️ CERTIFICATIONS: If the candidate has certifications, do NOT say they have "no certifications" in negative_rationale or how_to_improve. Reference their actual certifications when relevant to the role.
"""

    # Dynamic prompt: only jobs (changes per batch) — enables Gemini context caching
    prompt = f"""JOB(S) TO ANALYZE ({len(jobs_batch)} job(s)):

{jobs_text}

Return ONLY a JSON array (no markdown, no extra text):
[
  {{
    "job_id": "exact_id_from_above",
    "skills_matched": ["skill1", "skill2", ...],
    "preferred_skills_matched": ["skill3", ...],
    "match_score": 85,
    "domain_relevant_experience_years": 4.5,
    "positive_rationale": "Your experience with X and Y aligns well with this role; your skills in Z are a strong match.",
    "negative_rationale": "You may want to strengthen A; your experience in B is limited for this role.",
    "rationale": "Optional one-line summary in second person or leave empty.",
    "how_to_improve": "You can add X to your profile and take assessments in Y; check course recommendations to build Z. Do not mention career coach."{json_extra}
  }},
  ...
]
"""
    
    try:
        # Pre-call token estimation (system_instruction cached; prompt is dynamic per batch)
        estimated_system_tokens = estimate_tokens(system_instruction)
        estimated_prompt_tokens = estimate_tokens(prompt)
        log.info(
            f"📝 Batch {batch_idx + 1}: system_instruction ~{estimated_system_tokens} tokens (cached), "
            f"prompt ~{estimated_prompt_tokens} tokens"
        )
        total_estimated = estimated_system_tokens + estimated_prompt_tokens
        if total_estimated > 90_000:
            log.warning(f"⚠️ Batch {batch_idx + 1}: Large prompt (~{total_estimated} tokens) may approach context limits")
        
        # Route through invoke_llm for token tracking, quota, and observability
        # system_instruction enables Gemini context caching (resume + static prompts cached)
        # Retry once on JSON parse failure (truncated responses from LLM)
        max_parse_attempts = 2
        results = None
        for attempt in range(max_parse_attempts):
            try:
                content = await asyncio.wait_for(
                    invoke_llm(
                        prompt=prompt,
                        task_type="classification",
                        agent_name="job_matcher",
                        max_output_tokens=6000,
                        response_mime_type="application/json",
                        system_instruction=system_instruction,
                    ),
                    timeout=60.0  # 60 second timeout per batch
                )
            except asyncio.TimeoutError:
                log.error(f"❌ Batch {batch_idx + 1}: LLM call timed out (attempt {attempt + 1}/{max_parse_attempts})")
                if attempt < max_parse_attempts - 1:
                    continue
                return []
            
            content = content.strip() if content else ""

            # Extract JSON from response (handle markdown and other formatting)
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)

            # Try to find JSON array
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                json_str = json_match.group(0)
                # Clean up common JSON issues
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                try:
                    results = json.loads(json_str)
                    break
                except json.JSONDecodeError as e:
                    if attempt < max_parse_attempts - 1:
                        log.warning(f"⚠️ Batch {batch_idx + 1}: JSON parsing failed (attempt {attempt + 1}), retrying: {e}")
                    else:
                        log.error(f"❌ Batch {batch_idx + 1}: JSON parsing failed after {max_parse_attempts} attempts: {e}")
                        log.debug(f"Attempted to parse: {json_str[:500]}...")
                        return []
            else:
                if attempt < max_parse_attempts - 1:
                    log.warning(f"⚠️ Batch {batch_idx + 1}: No JSON found in LLM response (attempt {attempt + 1}), retrying")
                else:
                    log.error(f"❌ Batch {batch_idx + 1}: No JSON found in LLM response after {max_parse_attempts} attempts")
                    return []

        if results is None:
            return []

        # Validate and enrich results
        enriched_results = []
        for result in results:
            # Find original job data
            job_id = result.get("job_id")
            original_job = next(
                (j for j in jobs_batch if (j.get("job_id") or j.get("jd_id") or j.get("id")) == job_id),
                None
            )
            if original_job:
                result["job_data"] = original_job
                result["vector_similarity"] = original_job.get("_vector_similarity") or original_job.get("vector_similarity", 0.0)
                
                # Override skills_matched with pre-validated list (skill dropping done before LLM)
                job_id_str = str(job_id) if job_id else ""
                result["skills_matched"] = pre_validated_by_job.get(job_id_str, result.get("skills_matched") or [])
                if not isinstance(result["skills_matched"], list):
                    result["skills_matched"] = []
                
                # Ensure preferred_skills_matched is a list
                if not isinstance(result.get("preferred_skills_matched"), list):
                    result["preferred_skills_matched"] = []
                
                # Ensure match_score is valid; convert 0-100 to 0-1 (unified with ranker and job_matcher)
                if not isinstance(result.get("match_score"), (int, float)):
                    result["match_score"] = 0.0
                else:
                    result["match_score"] = normalize_match_score_0_100_to_0_1(float(result["match_score"]))
                
                # Normalize rationale fields (trim, cap length)
                def _norm_r(s: str, max_len: int = 600) -> str:
                    if not s or not isinstance(s, str):
                        return ""
                    return s.strip()[:max_len]
                result["positive_rationale"] = _norm_r(result.get("positive_rationale"), 600)
                result["negative_rationale"] = _norm_r(result.get("negative_rationale"), 600)
                result["how_to_improve"] = _norm_r(result.get("how_to_improve"), 500)
                if not result.get("rationale"):
                    combined = f"{result.get('positive_rationale', '')} {result.get('negative_rationale', '')}".strip()
                    result["rationale"] = combined[:800] if combined else "Analysis unavailable."
                
                dr_years = result.get("domain_relevant_experience_years")
                if dr_years is not None:
                    try:
                        result["domain_relevant_experience_years"] = float(dr_years) if float(dr_years) >= 0 else None
                    except (ValueError, TypeError):
                        result["domain_relevant_experience_years"] = None
                
                enriched_results.append(result)
        
        log.info(f"✅ Batch {batch_idx + 1}: Analyzed {len(enriched_results)}/{len(jobs_batch)} jobs")
        return enriched_results

    except Exception as e:
        log.error(f"❌ Batch {batch_idx + 1}: LLM analysis failed: {e}")
        return []


# =============================================================================
# JOB MATCHER PREPROCESSOR
# =============================================================================
# Runs before job_matcher. Resolves job_ids, fetches full job docs, checks cache, builds optional_reviewer_qns.
# Agent input is always: { "job_id": ["j_1", "j_2", ...], "optional_reviewer_qns": [[...], [], ...] }


def _align_optional_reviewer_qns(
    job_ids: List[str],
    recruiter_questions: Any,
) -> List[List[Dict[str, str]]]:
    """
    Build optional_reviewer_qns aligned with job_ids: one list per job.
    If recruiter_questions is list-of-lists with same length as job_ids, use as-is.
    Else if flat list of Q&A dicts, replicate for each job.
    """
    if not job_ids:
        return []
    qns = recruiter_questions or []
    if isinstance(qns, list) and len(qns) > 0:
        first = qns[0]
        if isinstance(first, list):
            return qns if len(qns) == len(job_ids) else ([qns[0] if qns else []] * len(job_ids))
        if isinstance(first, dict) and "question" in first:
            return [qns] * len(job_ids)
    return [[] for _ in job_ids]


async def job_matcher_preprocessor(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocessor before job_matcher. Unified flow for 1 or N jobs.
    - Compare + resume: MERGED — payload job_id first, then Chroma-retrieved jobs (deduped).
    - Compare without resume: legacy path (single job_id or second-pass multi-job).
    - Career flow (no job_id): Chroma retrieval by domain → fill job_ids.
    - Fetch full job docs for each ID.
    - For each (uid, job_id[i]), check Chroma for stored match details.
    - Build optional_reviewer_qns aligned with job_ids.
    """
    body = state.get("body") or {}
    uid = state.get("uid") or body.get("uid")
    session_id = state.get("session_id") or body.get("session_id")
    job_id = state.get("job_id") or body.get("job_id")
    endpoint_name = state.get("endpoint_name") or body.get("endpoint_name")
    is_compare_flow = (endpoint_name == "compare_candidate_job") or (body.get("request_type") == "candidate_job_match" or state.get("request_type") == "candidate_job_match")

    log.info(
        "📋 Preprocessor: uid=%s session_id=%s job_id=%s is_compare=%s has_structured_resume=%s",
        uid, session_id or "(none)", job_id or "(none)", is_compare_flow,
        bool(state.get("structured_resume") or body.get("structured_resume"))
    )
    if is_compare_flow:
        log.info(f"📋 Preprocessor: compare flow detected | job_id={job_id} | state.job_id={state.get('job_id')} | body.job_id={body.get('job_id')}")
    structured_resume = state.get("structured_resume") or body.get("structured_resume")

    # Career flow (no job_id): hydrate structured_resume from Chroma when missing or has no skills
    if not job_id and not is_compare_flow and uid:
        _needs_resume = not structured_resume or not isinstance(structured_resume, dict)
        if not _needs_resume:
            try:
                _needs_resume = not extract_primary_skills(structured_resume)
            except Exception:
                _needs_resume = True
        if _needs_resume:
            full_resume = None
            try:
                doc = await run_cpu_intensive(get_resume_doc, uid)
                if doc and isinstance(doc, dict) and doc.get("structured_resume"):
                    cand = doc["structured_resume"]
                    if isinstance(cand, dict) and extract_primary_skills(cand):
                        full_resume = cand
                        log.info("📋 Preprocessor: Hydrated structured_resume from get_resume_doc")
            except Exception as e:
                log.debug(f"Preprocessor get_resume_doc: {e}")
            if not full_resume:
                try:
                    cand = await run_cpu_intensive(fetch_structured_resume, uid, None, session_id)
                    if cand and isinstance(cand, dict) and extract_primary_skills(cand):
                        full_resume = cand
                        log.info("📋 Preprocessor: Hydrated structured_resume from fetch_structured_resume (session=%s)", session_id or "find_by_uid")
                except Exception as e:
                    log.debug(f"Preprocessor fetch_structured_resume: {e}")
            if full_resume:
                structured_resume = full_resume
                state = {**state, "structured_resume": full_resume}
            elif not structured_resume:
                log.warning(
                    "⚠️ Preprocessor: Career flow but no structured_resume (uid=%s, session=%s) - skipping retrieval; job_matcher will return no_matches",
                    uid, session_id
                )

    recruiter_questions = (
        state.get("recruiter_questions")
        or body.get("recruiter_questions")
        or body.get("recruiterQuestions")
        or body.get("questions")
    )
    interview_feedback = (
        state.get("interview_feedback")
        or body.get("interview_feedback")
        or body.get("interviewFeedback")
    )
    use_cache = not is_compare_flow and not (recruiter_questions or interview_feedback)

    # ── Determine whether this is a merged compare+resume run ────────────────
    has_resume_in_body = bool(body.get("resume_url") or (body.get("resume_text") or "").strip())
    is_merged_compare = is_compare_flow and has_resume_in_body and bool(job_id) and bool(structured_resume)

    # Legacy second-pass detection (compare without resume in body)
    _compare_flow_multi_job_run = state.get("_compare_flow_multi_job_run", False)
    is_second_preprocessor_run = _compare_flow_multi_job_run and is_compare_flow

    # ── Normalize job_id → job_ids ───────────────────────────────────────────
    if is_second_preprocessor_run:
        log.info(
            "📋 Preprocessor: Compare flow legacy 2nd run — Chroma retrieval for all jobs. "
            "Ignoring job_id=%s.", job_id,
        )
        job_ids: list = []
        job_id = None
    elif job_id:
        job_ids = [job_id] if isinstance(job_id, str) else (job_id if isinstance(job_id, list) else [])
    else:
        job_ids = []

    if is_compare_flow and not is_second_preprocessor_run and not is_merged_compare and not job_ids:
        log.warning("⚠️ Preprocessor: compare flow but no job_id provided - returning empty job_ids")
        return {
            "job_ids": [], "job_id": [], "job_docs": [], "optional_reviewer_qns": [],
            "cached_results": {}, "recruiter_questions": recruiter_questions,
            "interview_feedback": interview_feedback,
        }

    # ── Chroma retrieval helper (reused for career, merged-compare, and legacy 2nd run) ──
    async def _chroma_retrieve(resume: dict) -> List[Dict[str, Any]]:
        SEMANTIC_THRESHOLD = 0.40
        fetch_limit = 80
        SUPPLEMENT_JOBS = 10
        candidate_domain = _get_candidate_domain_from_resume(resume)
        if not candidate_domain:
            candidate_domain = await classify_candidate_domain(resume)
        domain_where = _build_job_domain_where_clause(candidate_domain)
        resume_text = await run_cpu_intensive(flatten_resume_fields, resume)
        candidate_skills = await run_cpu_intensive(extract_primary_skills, resume)
        if not resume_text and candidate_skills:
            resume_text = " ".join(str(s) for s in list(candidate_skills)[:20])
        if not resume_text:
            resume_text = "professional experience"
        query_text = _build_enhanced_candidate_query(resume_text, candidate_domain, candidate_skills or [])

        def run_query(where_clause=None):
            return query_job_descriptions(query_text, top_k=fetch_limit, where_clause=where_clause)

        if domain_where:
            domain_results = await run_cpu_intensive(run_query, domain_where)
            if domain_results is None:
                log.warning("📋 Preprocessor: Chroma query returned None (domain filter)")
            domain_jobs = await run_cpu_intensive(_parse_chroma_results_to_jobs, domain_results, SEMANTIC_THRESHOLD)
            full_results = await run_cpu_intensive(run_query, None)
            all_jobs = await run_cpu_intensive(_parse_chroma_results_to_jobs, full_results, SEMANTIC_THRESHOLD)
            domain_job_ids = {j.get("id") or j.get("_id") or j.get("job_id") for j in domain_jobs}
            supplement = [
                j for j in all_jobs
                if (j.get("id") or j.get("_id") or j.get("job_id")) not in domain_job_ids
            ]
            supplement.sort(key=lambda j: j.get("_vector_similarity") or j.get("vector_similarity", 0.0), reverse=True)
            supplement = supplement[:SUPPLEMENT_JOBS]
            retrieved = domain_jobs + supplement
            log.info(
                f"📋 Preprocessor: {len(domain_jobs)} domain + {len(supplement)} supplement "
                f"= {len(retrieved)} jobs (domain: {candidate_domain})"
            )
        else:
            query_results = await run_cpu_intensive(run_query, None)
            if query_results is None:
                log.warning("📋 Preprocessor: Chroma query returned None (no domain filter)")
            retrieved = await run_cpu_intensive(_parse_chroma_results_to_jobs, query_results, SEMANTIC_THRESHOLD)
            log.info(f"📋 Preprocessor: {len(retrieved)} jobs from Chroma (no domain filter)")
        return retrieved

    # ── Build job_docs ────────────────────────────────────────────────────────
    job_docs: List[Dict[str, Any]] = []
    primary_job_id_str: str = ""

    if is_merged_compare:
        # ✅ MERGED COMPARE+RESUME: payload job first, then Chroma-retrieved (deduped)
        primary_job_id_str = job_ids[0] if job_ids else ""
        log.info(f"📋 Preprocessor: Merged compare+resume — primary job_id={primary_job_id_str}, fetching primary + Chroma")
        primary_doc = await run_cpu_intensive(get_job_description, primary_job_id_str)
        if primary_doc:
            primary_doc.setdefault("job_id", primary_job_id_str)
            primary_doc.setdefault("jd_id", primary_job_id_str)
            primary_doc.setdefault("id", primary_job_id_str)
            job_docs.append(primary_doc)
        else:
            log.warning(f"⚠️ Preprocessor: primary job doc not found for {primary_job_id_str}")
            job_docs.append({"job_id": primary_job_id_str, "jd_id": primary_job_id_str, "id": primary_job_id_str})
        try:
            retrieved = await _chroma_retrieve(structured_resume)
            primary_id_set = {primary_job_id_str}
            for rj in retrieved:
                rid = str(rj.get("job_id") or rj.get("jd_id") or rj.get("id", ""))
                if rid and rid not in primary_id_set:
                    job_docs.append(rj)
                    primary_id_set.add(rid)
        except Exception as e:
            log.warning("⚠️ Preprocessor: Chroma retrieval failed during merged compare: %s", e, exc_info=True)
        job_ids = [
            str(j.get("job_id") or j.get("jd_id") or j.get("id", ""))
            for j in job_docs if j.get("job_id") or j.get("jd_id") or j.get("id")
        ]
        log.info(f"📋 Preprocessor: Merged compare — {len(job_docs)} total jobs (primary first)")

    elif is_compare_flow and job_id and not is_second_preprocessor_run:
        log.info(f"📋 Preprocessor: compare flow (no resume merge) - using provided job_id (single job)")

    elif (not job_ids and structured_resume and not is_compare_flow) or (is_second_preprocessor_run and structured_resume):
        label = "Compare flow legacy 2nd run" if is_second_preprocessor_run else "Career flow"
        log.info(f"📋 Preprocessor: {label} — running Chroma retrieval")
        try:
            job_docs = await _chroma_retrieve(structured_resume)
            if job_docs:
                job_ids = [
                    str(j.get("job_id") or j.get("jd_id") or j.get("id", ""))
                    for j in job_docs if j.get("job_id") or j.get("jd_id") or j.get("id")
                ]
            else:
                log.info("📋 Preprocessor: Chroma retrieval returned 0 jobs")
        except Exception as e:
            log.warning("⚠️ Preprocessor retrieval failed: %s", e, exc_info=True)

    # Fetch full docs for IDs that don't already have docs
    if job_ids and not job_docs:
        for jid in job_ids:
            doc = await run_cpu_intensive(get_job_description, jid)
            if doc:
                doc.setdefault("job_id", jid)
                doc.setdefault("jd_id", jid)
                doc.setdefault("id", jid)
                job_docs.append(doc)
            else:
                log.warning(f"⚠️ Preprocessor: no job doc for {jid}")
                job_docs.append({"job_id": jid, "jd_id": jid, "id": jid})

    # Cache lookup
    cached_results: Dict[str, Optional[Dict[str, Any]]] = {}
    resume_hash = _get_resume_hash_for_cache(structured_resume) if structured_resume else None
    if uid and use_cache:
        for jid in job_ids:
            cached = await run_cpu_intensive(get_candidate_job_ranking, uid, jid, resume_hash=resume_hash)
            cached_results[jid] = cached
        hit_count = sum(1 for v in cached_results.values() if v is not None)
        if hit_count:
            log.info(f"📋 Preprocessor: {hit_count}/{len(job_ids)} cached match results found")

    optional_reviewer_qns = _align_optional_reviewer_qns(job_ids, recruiter_questions)

    out: Dict[str, Any] = {
        "job_ids": job_ids,
        "job_id": job_ids,
        "job_docs": job_docs,
        "optional_reviewer_qns": optional_reviewer_qns,
        "cached_results": cached_results,
        "recruiter_questions": recruiter_questions,
        "interview_feedback": interview_feedback,
    }
    if primary_job_id_str:
        out["_compare_primary_job_id"] = primary_job_id_str
    return out


# Gemini embedding for resume-job similarity (compare flow)
_QA_EMBEDDING_MODEL = "models/gemini-embedding-001"


async def _calculate_semantic_similarity_resume_job(
    structured_resume: Dict[str, Any],
    job_description: Dict[str, Any],
) -> float:
    """Calculate semantic similarity between resume and job using Gemini embeddings."""
    try:
        from google import genai
        import numpy as np

        api_key = _settings.GOOGLE_API_KEY
        if not api_key:
            log.warning("Google API key not found, using fallback similarity")
            return 0.5

        client = genai.Client(api_key=api_key)
        resume_parts = []
        skills = structured_resume.get("skills", [])
        if isinstance(skills, list):
            skill_names = []
            for skill in skills[:30]:
                if isinstance(skill, str):
                    skill_names.append(skill)
                elif isinstance(skill, dict):
                    name = skill.get("SkillName") or skill.get("skillName") or skill.get("name", "")
                    if name:
                        skill_names.append(name)
            if skill_names:
                resume_parts.append(f"Skills: {', '.join(skill_names)}")
        experience = structured_resume.get("experience", [])
        if isinstance(experience, list):
            for exp in experience[:3]:
                if isinstance(exp, dict):
                    title = exp.get("job_title") or exp.get("title", "")
                    company = exp.get("company", "")
                    if title:
                        resume_parts.append(f"{title} at {company}")
        education = structured_resume.get("education", [])
        if isinstance(education, list) and education:
            edu = education[0]
            if isinstance(edu, dict) and edu.get("degree"):
                resume_parts.append(f"Education: {edu.get('degree', '')}")
        resume_text = " | ".join(resume_parts) if resume_parts else "No resume data"

        jd_parts = []
        if job_description.get("jobTitle") or job_description.get("title"):
            jd_parts.append(f"Job: {job_description.get('jobTitle') or job_description.get('title')}")
        rs = job_description.get("requiredSkills") or job_description.get("required_skills", [])
        if isinstance(rs, list) and rs:
            jd_parts.append(f"Required Skills: {', '.join(rs)}")
        elif isinstance(rs, str):
            jd_parts.append(f"Required Skills: {rs}")
        desc = job_description.get("description") or job_description.get("fullJobDescription", "")
        if desc:
            jd_parts.append(desc[:500])
        jd_text = " | ".join(jd_parts) if jd_parts else "No job description"

        # Batch embed resume + job (cached across restarts via gemini_embedding_cache)
        def _embed_batch():
            return cached_embed_texts(client, _QA_EMBEDDING_MODEL, [resume_text, jd_text])

        embeddings = await run_cpu_intensive(_embed_batch)
        if len(embeddings) != 2:
            return 0.5

        r = np.array(embeddings[0])
        j = np.array(embeddings[1])
        rn = np.linalg.norm(r)
        jn = np.linalg.norm(j)
        if rn > 0:
            r = r / rn
        if jn > 0:
            j = j / jn
        sim = float(np.dot(r, j))
        return max(0.0, min(1.0, sim))
    except Exception as e:
        log.error(f"Error calculating semantic similarity: {e}")
        return 0.5


async def compare_candidate_with_job(
    candidate_id: str,
    job_id: str,
    job_description: Optional[Dict[str, Any]] = None,
    recruiter_questions: Optional[List[Dict[str, str]]] = None,
    interview_feedback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Compare one candidate with one job.
    Uses semantic similarity + LLM analysis. Returns the same format as the current compare flow.
    """
    log.info("=" * 70)
    mode = "POST-INTERVIEW" if interview_feedback else "STANDARD"
    log.info(f"🚀 JOB_MATCHER COMPARE [{mode}]: {candidate_id} → {job_id}")
    log.info("=" * 70)

    try:
        structured_resume = await run_cpu_intensive(fetch_structured_resume, candidate_id)
        if not structured_resume:
            return {"error": "Structured resume not found for candidate", "candidate_id": candidate_id, "similar_jobs": []}

        if not job_description:
            job_description = await run_cpu_intensive(get_job_description, job_id)
        if not job_description:
            return {"error": "Job description not found", "job_id": job_id, "similar_jobs": []}

        job_id_val = job_description.get("job_id") or job_description.get("jd_id") or job_description.get("id") or job_id
        semantic_similarity = await _calculate_semantic_similarity_resume_job(structured_resume, job_description)
        log.info(f"📊 Semantic similarity: {semantic_similarity:.2%}")

        SEMANTIC_THRESHOLD = 0.30
        required_skills = job_description.get("required_skills") or job_description.get("requiredSkills", [])
        if isinstance(required_skills, str):
            required_skills = [s.strip() for s in required_skills.split(",")]

        async def _get_similar_jobs():
            """Prefer stored candidate-job matches; fall back to JD-based similar jobs. Returns (jobs, source)."""
            stored = await run_cpu_intensive(
                get_top_matched_jobs_for_candidate, candidate_id, job_id_val, 10
            )
            if stored:
                return (stored, "cache")
            try:
                semantic = await run_cpu_intensive(
                    get_similar_jobs_for_job, job_description, job_id_val, 10
                )
                return (semantic, "semantic_search")
            except Exception:
                return ([], "semantic_search")

        async def _low_match():
            similar, source = await _get_similar_jobs()
            return {
                "candidate_id": candidate_id,
                "job_id": job_id_val,
                "match_score": round(semantic_similarity * 0.5, 4),
                "skill_match_percentage": 0.0,
                "skill_match_count": 0,
                "skills_matched": [],
                "skills_unmatched": required_skills,
                "total_required_skills": len(required_skills),
                "vector_similarity": semantic_similarity,
                "rationale": f"Low semantic similarity ({semantic_similarity:.1%}) between your profile and this job.",
                "positive_rationale": "",
                "negative_rationale": f"Profile and job have low alignment ({semantic_similarity:.1%}).",
                "processing_method": "job_matcher_compare",
                "below_threshold": True,
                "similar_jobs": similar,
                "similar_jobs_source": source,
                "tier": 5,
                "tier_name": "No Match",
                "tier_action": "Review manually or skip",
                "tier_criteria": {},
                "hire_recommendation": "no_hire" if interview_feedback else None,
                "hire_rationale": "Profile and job have low alignment." if interview_feedback else None,
                "interview_feedback_used": bool(interview_feedback),
            }

        if semantic_similarity < SEMANTIC_THRESHOLD:
            log.warning(f"⚠️ Semantic similarity below threshold ({SEMANTIC_THRESHOLD:.0%})")
            return await _low_match()

        candidate_skills = await run_cpu_intensive(extract_primary_skills, structured_resume)
        if not candidate_skills:
            candidate_skills = []
        candidate_experience = structured_resume.get("total_experience_years") or structured_resume.get("total_experience") or "N/A"
        edu_list = structured_resume.get("education", []) or structured_resume.get("Education", [])
        if edu_list and isinstance(edu_list, list):
            e0 = edu_list[0]
            if isinstance(e0, dict):
                candidate_education = f"{e0.get('degree','')} {e0.get('major','')}".strip() or "N/A"
            else:
                candidate_education = str(e0) if e0 else "N/A"
        else:
            candidate_education = structured_resume.get("highest_education", "N/A")
        candidate_location = structured_resume.get("location") or structured_resume.get("Location", "N/A")
        candidate_certifications = []
        for c in (structured_resume.get("certifications") or [])[:10]:
            if isinstance(c, dict):
                candidate_certifications.append(c.get("certification_name") or c.get("name", "") or str(c))
            else:
                candidate_certifications.append(str(c))
        candidate_projects = structured_resume.get("projects") or structured_resume.get("Projects", [])
        professional_summary = structured_resume.get("professional_summary") or structured_resume.get("Summary", "") or ""
        candidate_domain = _get_candidate_domain_from_resume(structured_resume)
        if not candidate_domain:
            candidate_domain = await classify_candidate_domain(structured_resume)
        candidate_role_context = _get_domain_role_context(candidate_domain)
        sanitized_resume = sanitize_resume_for_llm(structured_resume)

        job_with_sim = dict(job_description)
        job_with_sim["_vector_similarity"] = semantic_similarity
        job_with_sim["job_id"] = job_id_val
        job_with_sim["jd_id"] = job_id_val
        job_with_sim["id"] = job_id_val

        batch_results = await analyze_job_batch_with_llm(
            candidate_skills=candidate_skills,
            candidate_experience=str(candidate_experience),
            candidate_education=str(candidate_education),
            candidate_location=str(candidate_location),
            jobs_batch=[job_with_sim],
            batch_idx=0,
            candidate_certifications=candidate_certifications,
            candidate_projects=candidate_projects,
            professional_summary=professional_summary,
            candidate_domain=candidate_domain,
            candidate_role_context=candidate_role_context,
            structured_resume=sanitized_resume,
            recruiter_questions=recruiter_questions,
            interview_feedback=interview_feedback,
        )

        if not batch_results:
            log.warning("⚠️ LLM analysis failed, using fallback")
            similar, source = await _get_similar_jobs()
            return {
                "candidate_id": candidate_id,
                "job_id": job_id_val,
                "match_score": round(semantic_similarity * 0.8, 4),
                "skill_match_percentage": 0.0,
                "skill_match_count": 0,
                "skills_matched": [],
                "skills_unmatched": required_skills,
                "total_required_skills": len(required_skills),
                "vector_similarity": semantic_similarity,
                "rationale": "Analysis unavailable due to processing error.",
                "positive_rationale": "",
                "negative_rationale": "Analysis unavailable.",
                "processing_method": "job_matcher_compare",
                "llm_failed": True,
                "similar_jobs": similar,
                "similar_jobs_source": source,
                "tier": None,
                "tier_name": "Unknown",
                "tier_action": "",
                "tier_criteria": {},
                "hire_recommendation": None,
                "hire_rationale": None,
                "interview_feedback_used": bool(interview_feedback),
            }

        r = batch_results[0]
        required_skills_list = job_description.get("requiredSkills") or job_description.get("required_skills", [])
        if isinstance(required_skills_list, str):
            required_skills_list = [s.strip() for s in required_skills_list.split(",")]
        # Skills already pre-validated in analyze_job_batch_with_llm (skill dropping before rationale)
        skills_matched = r.get("skills_matched", [])
        skills_unmatched = [s for s in required_skills_list if s not in skills_matched]
        skill_match_pct = round(len(skills_matched) / len(required_skills_list) * 100, 2) if required_skills_list else 0.0
        rationale = r.get("rationale") or f"{r.get('positive_rationale','')} {r.get('negative_rationale','')}".strip()[:800]
        
        match_score = r.get("match_score", 0.0)
        if match_score > 1.0:
            match_score = match_score / 100.0
        match_score = max(0.0, min(1.0, match_score))

        try:
            must_have = job_description.get("_must_have_skills") or []
            tier_input = {
                "skills_matched": skills_matched,
                "match_score": match_score,
                "domain_relevant_experience_years": r.get("domain_relevant_experience_years"),
                "current_location": candidate_location,
                "education_summary": candidate_education,
                "candidate": {"structuredResume": structured_resume},
            }
            tier_result = compute_tier_from_llm_results(
                candidate=tier_input,
                jd_skills=required_skills_list,
                jd_must_have_skills=must_have,
                jd_experience=job_description.get("experience") or job_description.get("experienceRequired", ""),
                jd_location=job_description.get("location") or job_description.get("jobLocation", ""),
                jd_education=job_description.get("educationRequired") or job_description.get("education", ""),
                work_mode=job_description.get("workMode", ""),
            )
            tier = tier_result["tier"]
            tier_name = tier_result["tier_name"]
            tier_action = tier_result["tier_action"]
            tier_criteria = tier_result["criteria"]
            if tier_result.get("location_bonus", 0) > 0:
                match_score = round(min(1.0, match_score + tier_result["location_bonus"]), 4)
        except Exception as te:
            log.warning(f"⚠️ Tier classification failed: {te}")
            tier, tier_name, tier_action, tier_criteria = None, "Unknown", "", {}

        similar_jobs, similar_jobs_source = await _get_similar_jobs()

        result = {
            "candidate_id": candidate_id,
            "job_id": job_id_val,
            "match_score": match_score,
            "skill_match_percentage": round(skill_match_pct, 1),
            "skill_match_count": len(skills_matched),
            "skills_matched": skills_matched,
            "skills_unmatched": skills_unmatched,
            "total_required_skills": len(required_skills_list),
            "vector_similarity": semantic_similarity,
            "rationale": rationale,
            "positive_rationale": r.get("positive_rationale", ""),
            "negative_rationale": r.get("negative_rationale", ""),
            "processing_method": "job_matcher_compare",
            "similar_jobs": similar_jobs,
            "similar_jobs_source": similar_jobs_source,
            "tier": tier,
            "tier_name": tier_name,
            "tier_action": tier_action,
            "tier_criteria": tier_criteria,
            "domain_relevant_experience_years": r.get("domain_relevant_experience_years"),
            "hire_recommendation": r.get("hire_recommendation"),
            "hire_rationale": r.get("hire_rationale"),
            "interview_feedback_used": bool(interview_feedback),
        }
        log.info(f"✅ Compare complete: score={result['match_score']:.2%}, skills={result['skill_match_count']}/{result['total_required_skills']}")
        return result
    except Exception as e:
        log.error(f"❌ compare_candidate_with_job error: {e}", exc_info=True)
        return {
            "error": "Unexpected error during candidate-job matching",
            "candidate_id": candidate_id,
            "job_id": job_id,
            "details": str(e),
            "similar_jobs": [],
        }


async def job_matcher_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main job matcher: Semantic filtering (30%) + Batch LLM analysis.
    
    NO SKILL_MATCHER DEPENDENCY - Pure semantic + LLM approach.
    
    Flow:
    1. Get candidate resume and skills
    2. Query ChromaDB for jobs with semantic score ≥ 30%
    3. Batch process jobs through LLM (5 per batch)
    4. LLM returns: skills_matched, match_score, rationale, positive_rationale, negative_rationale, how_to_improve for each job
    5. Sort by match_score and return top jobs
    
    Args:
        state: State dictionary containing structured_resume
        
    Returns:
        Updated state with matched_jobs (ranked by LLM match_score)
    """
    log.info("="*70)
    log.info("🚀 REDESIGNED JOB_MATCHER V2: Semantic Filter + Batch LLM")
    log.info("="*70)
    _matcher_start = time.time()

    uid = state.get("uid") or (state.get("body") or {}).get("uid")
    structured_resume = state.get("structured_resume") or (state.get("body") or {}).get("structured_resume")
    job_ids = state.get("job_ids") or []
    job_docs = state.get("job_docs") or []
    optional_reviewer_qns = state.get("optional_reviewer_qns") or []
    cached_results = state.get("cached_results") or {}
    interview_feedback = state.get("interview_feedback") or (state.get("body") or {}).get("interview_feedback")
    _compare_primary_job_id = state.get("_compare_primary_job_id") or ""
    chroma_jobs = None  # Set by career flow when job_docs from preprocessor; legacy path does retrieval

    if not uid:
        log.error("❌ Missing UID in state")
        return {
            "error": "Missing UID in state",
            "status": "error",
            "top_matches": [],
            "matched_jobs": [],
            "total_matches_found": 0,
            "job_matcher_status": "error"
        }

    # ============================================================
    # UNIFIED PATH: Preprocessor provided job_ids, job_docs (retrieval done in preprocessor)
    # - Compare flow (1 job): use compare_candidate_with_job (single candidate-job match)
    # - Compare flow (N jobs, 2nd run): use batch LLM like normal job matcher
    # - Career flow (N jobs): use batch LLM (no retrieval here)
    # ==============================================    # ============================================================
    # ============================================================
    if job_ids and job_docs:
        log.info("="*70)
        log.info(f"🚀 JOB_MATCHER: {len(job_ids)} job(s) from preprocessor (no retrieval)")
        log.info("="*70)
        is_compare = (state.get("endpoint_name") or (state.get("body") or {}).get("endpoint_name")) == "compare_candidate_job"

        # Compare flow, single job only: use compare_candidate_with_job (deep single-JD analysis)
        # Multi-job (compare 2nd run or career): use batch LLM path below, same as normal job matcher
        if len(job_ids) == 1 and is_compare:
            matched_jobs = []
            for i, jid in enumerate(job_ids):
                doc = job_docs[i] if i < len(job_docs) else None
                qns = optional_reviewer_qns[i] if i < len(optional_reviewer_qns) else []
                cached = cached_results.get(jid) if cached_results else None

                if cached is not None and not interview_feedback:
                    matched_jobs.append(cached)
                    log.info(f"✅ Job {jid}: using cached result (score={cached.get('match_score', 0):.2%})")
                else:
                    result = await compare_candidate_with_job(
                        uid, jid,
                        job_description=doc,
                        recruiter_questions=qns if qns else None,
                        interview_feedback=interview_feedback,
                    )
                    if result.get("error"):
                        log.warning(f"⚠️ Job {jid}: {result.get('error')}")
                        continue
                    if uid and jid and not result.get("error") and not is_compare:
                        resume_hash = _get_resume_hash_for_cache(structured_resume)
                        await run_cpu_intensive(upsert_candidate_job_ranking, uid, jid, result, resume_hash=resume_hash)
                    matched_jobs.append(result)

            matched_jobs.sort(key=lambda j: j.get("match_score", 0.0), reverse=True)
            processing_time = time.time() - _matcher_start
            single_result = matched_jobs[0] if len(matched_jobs) == 1 else None
            out = {
                "matched_jobs": matched_jobs,
                "top_matches": matched_jobs,
                "total_matches_found": len(matched_jobs),
                "total_jobs_matched": len(matched_jobs),
                "job_matcher_status": "completed",
                "processing_time_seconds": round(processing_time, 2),
            }
            if single_result:
                out["candidate_job_match_result"] = single_result
                out["match_score"] = single_result.get("match_score", 0.0)
                out["skill_match_percentage"] = single_result.get("skill_match_percentage", 0.0)
                out["skill_match_count"] = single_result.get("skill_match_count", 0)
                out["skills_matched"] = single_result.get("skills_matched", [])
                out["skills_unmatched"] = single_result.get("skills_unmatched", [])
                out["hire_recommendation"] = single_result.get("hire_recommendation")
                out["hire_rationale"] = single_result.get("hire_rationale")
            log.info(f"✅ Job matcher (compare): {len(matched_jobs)} jobs in {processing_time:.2f}s")
            return out

        # Career flow: multiple jobs — use batch LLM (job_docs from preprocessor, no retrieval)
        chroma_jobs = job_docs
        # Fall through to try block (skip retrieval, use chroma_jobs)

    # ============================================================
    # Compare flow with empty job_ids: preprocessor returned empty (missing job_id)
    # ============================================================
    body_agent = state.get("body") or {}
    is_compare = (state.get("endpoint_name") or body_agent.get("endpoint_name")) == "compare_candidate_job"
    if is_compare and not job_ids:
        log.error("❌ Compare flow: no job_id provided")
        return {
            "error": "Compare flow requires job_id",
            "status": "error",
            "top_matches": [],
            "matched_jobs": [],
            "total_matches_found": 0,
            "job_matcher_status": "error",
            "candidate_job_match_result": {},
        }

    # Career flow with empty job_docs: preprocessor retrieval returned nothing — no retry
    if not job_ids and not is_compare:
        log.info("📋 Career flow: no jobs from preprocessor (retrieval returned empty)")
        return {
            "matched_jobs": [],
            "top_matches": [],
            "total_matches_found": 0,
            "job_matcher_status": "no_matches",
            "message": "No jobs found matching your profile",
        }

    # =====================================================    # =====================================================    if not structured_resume:
        log.error("❌ Missing structured_resume in state")
        return {
            "error": "Missing structured_resume in state",
            "status": "error",
            "top_matches": [],
            "matched_jobs": [],
            "total_matches_found": 0,
            "job_matcher_status": "error"
        }

    try:
        # ============================================================
        # STEP 1: Extract candidate details
        # ============================================================
        candidate_skills = await run_cpu_intensive(extract_primary_skills, structured_resume)
        
        if not candidate_skills:
            log.warning("⚠️ No skills found in candidate resume")
            return {
                "error": "No skills found in resume",
                "status": "error",
                "top_matches": [],
                "matched_jobs": [],
                "total_matches_found": 0,
                "job_matcher_status": "error"
            }
        
        # Extract experience and education (actual format uses "total_experience_years")
        candidate_experience = structured_resume.get("total_experience_years") or structured_resume.get("total_experience") or structured_resume.get("TotalExperience", "N/A")
        
        # Get education from education array (full details)
        edu_list = structured_resume.get("education", []) or structured_resume.get("Education", [])
        if edu_list and isinstance(edu_list, list) and len(edu_list) > 0:
            edu_item = edu_list[0]
            if isinstance(edu_item, dict):
                degree = edu_item.get("degree") or edu_item.get("Degree", "")
                major = edu_item.get("major") or edu_item.get("Major") or edu_item.get("field", "")
                candidate_education = f"{degree} {major}".strip() if degree or major else "N/A"
            else:
                candidate_education = str(edu_item) if edu_item else "N/A"
        else:
            candidate_education = structured_resume.get("highest_education") or structured_resume.get("HighestEducation", "N/A")

        # Extract candidate location (best-effort)
        candidate_location = (
            structured_resume.get("location")
            or structured_resume.get("Location")
            or structured_resume.get("current_location")
            or structured_resume.get("CurrentLocation")
        )
        if not candidate_location:
            personal_info = structured_resume.get("personalInformation") or structured_resume.get("personal_info") or {}
            if isinstance(personal_info, dict):
                candidate_location = (
                    personal_info.get("location")
                    or personal_info.get("city")
                    or personal_info.get("state")
                    or personal_info.get("country")
                )
        candidate_location = candidate_location or "N/A"
        
        # Extract certifications for LLM (support certification_name and name)
        candidate_certifications = []
        certs = structured_resume.get("certifications") or structured_resume.get("Certifications", [])
        for c in (certs if isinstance(certs, list) else []):
            if isinstance(c, dict):
                name = c.get("certification_name") or c.get("name") or c.get("certificationName", "")
                if name:
                    candidate_certifications.append(name)
            elif c:
                candidate_certifications.append(str(c))
        
        # Extract projects for LLM
        candidate_projects = structured_resume.get("projects") or structured_resume.get("Projects", [])
        if not isinstance(candidate_projects, list):
            candidate_projects = []
        
        # Professional summary for LLM
        professional_summary = structured_resume.get("professional_summary") or structured_resume.get("Summary") or structured_resume.get("professionalSummary", "") or ""
        if not isinstance(professional_summary, str):
            professional_summary = str(professional_summary or "")
        
        log.info(
            f"👤 Candidate: {len(candidate_skills)} skills | "
            f"Experience: {candidate_experience} | Education: {candidate_education} | "
            f"Certifications: {len(candidate_certifications)} | Projects: {len(candidate_projects)}"
        )
        
        # ============================================================
        # STEP 1b: Candidate domain for contextual matching (use stored when available)
        # ============================================================
        candidate_domain = _get_candidate_domain_from_resume(structured_resume)
        if not candidate_domain:
            candidate_domain = await classify_candidate_domain(structured_resume)
        candidate_role_context = _get_domain_role_context(candidate_domain)
        log.info(f"🎯 Candidate domain: {candidate_domain} | role context: {candidate_role_context[:80]}{'...' if len(candidate_role_context) > 80 else ''}")
        
        # ============================================================
        # STEP 2: Semantic filtering — SKIP when chroma_jobs from preprocessor
        # ============================================================
        SEMANTIC_THRESHOLD = 0.40
        if chroma_jobs is None:
            resume_text = await run_cpu_intensive(flatten_resume_fields, structured_resume)
            if not resume_text or not resume_text.strip():
                resume_text = " ".join(str(s) for s in candidate_skills)
            fetch_limit = 80
            SUPPLEMENT_JOBS = 10
            query_text = _build_enhanced_candidate_query(resume_text, candidate_domain, list(candidate_skills))
            domain_where = _build_job_domain_where_clause(candidate_domain)

            def run_query(where_clause=None):
                return query_job_descriptions(query_text, top_k=fetch_limit, where_clause=where_clause)

            log.info(f"🔎 Querying ChromaDB: enhanced candidate query (limit={fetch_limit})")
            if domain_where:
                log.info(f"🎯 Domain-filtered retrieval for candidate domain: {candidate_domain}")
                domain_results = await asyncio.to_thread(run_query, domain_where)
                domain_jobs = await run_cpu_intensive(
                    _parse_chroma_results_to_jobs, domain_results, SEMANTIC_THRESHOLD
                )
                full_results = await asyncio.to_thread(run_query, None)
                all_jobs = await run_cpu_intensive(
                    _parse_chroma_results_to_jobs, full_results, SEMANTIC_THRESHOLD
                )
                domain_job_ids = {j.get("id") or j.get("_id") or j.get("job_id") for j in domain_jobs}
                supplement = [
                    j for j in all_jobs
                    if (j.get("id") or j.get("_id") or j.get("job_id")) not in domain_job_ids
                ]
                supplement.sort(
                    key=lambda j: j.get("_vector_similarity") or j.get("vector_similarity", 0.0),
                    reverse=True,
                )
                supplement = supplement[:SUPPLEMENT_JOBS]
                chroma_jobs = domain_jobs + supplement
                log.info(
                    f"✅ Semantic filtering: {len(domain_jobs)} domain + {len(supplement)} supplement "
                    f"= {len(chroma_jobs)} total [domain: {candidate_domain}]"
                )
            else:
                log.info("ℹ️ No domain filter, using full search")
                query_results = await asyncio.to_thread(run_query, None)
                chroma_jobs = await run_cpu_intensive(
                    _parse_chroma_results_to_jobs, query_results, SEMANTIC_THRESHOLD
                )
                log.info(f"✅ Semantic filtering: {len(chroma_jobs)} jobs")
        else:
            log.info(f"📋 Using {len(chroma_jobs)} jobs from preprocessor (no retrieval)")
        
        if not chroma_jobs:
            state["matched_jobs"] = []
            state["top_matches"] = []  # ✅ FIX: Add for backward compatibility
            state["total_matches_found"] = 0  # ✅ FIX: Add count
            state["job_matcher_status"] = "no_matches"  # ✅ FIX: Add status
            state["message"] = f"No jobs found with ≥{SEMANTIC_THRESHOLD*100}% semantic similarity"
            return state
        
        # ============================================================
        # STEP 3: Select top N jobs for LLM analysis (cap for best matches only)
        # ============================================================
        # Sort by semantic similarity (highest first)
        chroma_jobs.sort(key=lambda j: j.get("_vector_similarity") or j.get("vector_similarity", 0.0), reverse=True)
        
        # Dynamic cap: send a reasonable fraction of matches, bounded by min/max
        MIN_JOBS_FOR_LLM = 10
        MAX_JOBS_FOR_LLM = 35
        LLM_FRACTION = 0.30  # send up to 30% of retrieved jobs when we have many matches
        
        # Check if called from chatbot with limit (only process top 3-4 jobs)
        chatbot_top_k = state.get("chatbot_top_k")
        if chatbot_top_k and isinstance(chatbot_top_k, int) and chatbot_top_k > 0:
            log.info(f"🤖 Chatbot mode: Limiting to top {chatbot_top_k} jobs for LLM analysis")
            jobs_for_llm = chroma_jobs[:chatbot_top_k]
            jobs_no_llm = chroma_jobs[chatbot_top_k:]
        else:
            n = len(chroma_jobs)
            if n <= MAX_JOBS_FOR_LLM:
                top_n = n  # send all when within cap
            else:
                # scale with match count: send ~FRACTION of jobs, clamped to [MIN, MAX]
                top_n = max(MIN_JOBS_FOR_LLM, min(MAX_JOBS_FOR_LLM, int(n * LLM_FRACTION)))
            jobs_for_llm = chroma_jobs[:top_n]
            jobs_no_llm = chroma_jobs[top_n:]
        
        # Pin primary payload job in jobs_for_llm for merged compare (even if low similarity)
        if _compare_primary_job_id:
            primary_in_llm = any(
                (j.get("job_id") or j.get("jd_id") or j.get("id", "")) == _compare_primary_job_id
                for j in jobs_for_llm
            )
            if not primary_in_llm:
                for idx, j in enumerate(jobs_no_llm):
                    if (j.get("job_id") or j.get("jd_id") or j.get("id", "")) == _compare_primary_job_id:
                        jobs_for_llm.insert(0, jobs_no_llm.pop(idx))
                        log.info(f"📌 Pinned primary job {_compare_primary_job_id} into jobs_for_llm (was outside top-N)")
                        break
                else:
                    for j in chroma_jobs:
                        if (j.get("job_id") or j.get("jd_id") or j.get("id", "")) == _compare_primary_job_id:
                            jobs_for_llm.insert(0, j)
                            log.info(f"📌 Pinned primary job {_compare_primary_job_id} from chroma_jobs")
                            break

        log.info(
            f"🎯 Selecting top {len(jobs_for_llm)} jobs (of {len(chroma_jobs)} candidates) for LLM analysis"
        )
        
        def get_vector_sim(job):
            return job.get("_vector_similarity") or job.get("vector_similarity", 0.0)
        
        if jobs_for_llm:
            log.info(
                f"   - For LLM: {len(jobs_for_llm)} jobs (semantic: "
                f"{get_vector_sim(jobs_for_llm[0]):.3f} - "
                f"{get_vector_sim(jobs_for_llm[-1]):.3f})"
            )
        if jobs_no_llm:
            log.info(
                f"   - No LLM: {len(jobs_no_llm)} jobs (semantic: "
                f"{get_vector_sim(jobs_no_llm[0]):.3f} - "
                f"{get_vector_sim(jobs_no_llm[-1]):.3f})"
            )
        
        # ============================================================
        # STEP 4: LLM analysis — one job per call for best quality
        # ============================================================
        # Sanitize resume once, reuse for every LLM call
        sanitized_resume = sanitize_resume_for_llm(structured_resume) if structured_resume else None
        
        batch_size = 1  # One job per LLM call for nuanced, ChatGPT-quality scoring
        batches = [
            jobs_for_llm[i:i + batch_size]
            for i in range(0, len(jobs_for_llm), batch_size)
        ]
        
        log.info(
            f"🤖 Processing top {len(jobs_for_llm)} jobs individually "
            f"({len(batches)} LLM calls, full resume context)"
        )
        
        analyzed_jobs = []
        partial_timeout = False

        # Pipeline deadline: return early with partial results when approaching timeout
        pipeline_started = state.get("_pipeline_started_at") or 0
        pipeline_timeout = state.get("_pipeline_timeout_seconds") or 300
        pipeline_deadline = pipeline_started + pipeline_timeout
        PARTIAL_BUFFER_SECONDS = 45  # Stop 45s before deadline to allow callback delivery

        # Process jobs in parallel (5 concurrent LLM calls to maintain throughput)
        parallel_batch_size = 5
        for batch_group_idx in range(0, len(batches), parallel_batch_size):
            # Check if approaching pipeline deadline - return partial results
            if pipeline_started and time.time() + PARTIAL_BUFFER_SECONDS > pipeline_deadline:
                partial_timeout = True
                log.warning(
                    f"⏱️ Approaching pipeline timeout ({int(pipeline_deadline - time.time())}s left) - "
                    f"returning partial results ({len(analyzed_jobs)}/{len(jobs_for_llm)} jobs analyzed)"
                )
                break

            batch_group = batches[batch_group_idx:batch_group_idx + parallel_batch_size]
            
            log.info(
                f"📦 Processing group {batch_group_idx // parallel_batch_size + 1}/"
                f"{(len(batches) + parallel_batch_size - 1) // parallel_batch_size} "
                f"({len(batch_group)} jobs in parallel)"
            )
            
            tasks = [
                analyze_job_batch_with_llm(
                    candidate_skills=list(candidate_skills),
                    candidate_experience=str(candidate_experience),
                    candidate_education=str(candidate_education),
                    candidate_location=str(candidate_location),
                    jobs_batch=batch,
                    batch_idx=batch_group_idx + i,
                    candidate_certifications=candidate_certifications,
                    candidate_projects=candidate_projects,
                    professional_summary=professional_summary or None,
                    candidate_domain=candidate_domain,
                    candidate_role_context=candidate_role_context,
                    structured_resume=sanitized_resume,
                )
                for i, batch in enumerate(batch_group)
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    log.error(f"Job analysis failed: {result}")
                    continue
                if result:
                    analyzed_jobs.extend(result)
        
        log.info(f"✅ LLM analysis complete: {len(analyzed_jobs)}/{len(jobs_for_llm)} jobs analyzed")
        
        if not analyzed_jobs:
            state["matched_jobs"] = []
            state["top_matches"] = []  # ✅ FIX: Add for backward compatibility
            state["total_matches_found"] = 0  # ✅ FIX: Add count
            state["job_matcher_status"] = "llm_failed"  # ✅ FIX: Add status
            state["message"] = "LLM analysis failed for all jobs"
            return state
        
        # ============================================================
        # STEP 5: Filter and sort by match_score
        # ============================================================
        # ✅ EXPLICIT: Only LLM-analyzed jobs are included in final results
        log.info(
            f"📊 Processing {len(analyzed_jobs)} LLM-analyzed jobs "
            f"(bottom {len(jobs_no_llm)} jobs excluded - no LLM analysis)"
        )
        
        # ✅ VALIDATION: Ensure all jobs have LLM analysis fields (rationale or positive_rationale)
        llm_validated_jobs = []
        for job in analyzed_jobs:
            has_rationale = job.get("rationale") or job.get("positive_rationale")
            if "match_score" not in job or not has_rationale:
                log.warning(f"⚠️ Skipping job {job.get('job_id', 'unknown')} - missing LLM analysis fields")
                continue
            llm_validated_jobs.append(job)
        
        if len(llm_validated_jobs) < len(analyzed_jobs):
            log.warning(f"⚠️ Filtered out {len(analyzed_jobs) - len(llm_validated_jobs)} jobs missing LLM fields")
        
        analyzed_jobs = llm_validated_jobs
        analyzed_jobs.sort(
            key=lambda j: j.get("match_score", 0.0),
            reverse=True
        )
        
        # Format output - ONLY LLM-analyzed jobs (Option B: scale score when skill match low, do not filter out)
        matched_jobs = []
        adjusted_count = 0
        for rank, job in enumerate(analyzed_jobs, start=1):
            # ✅ DOUBLE-CHECK: Verify this job was LLM-analyzed
            has_rationale = job.get("rationale") or job.get("positive_rationale")
            if "match_score" not in job or not has_rationale:
                log.error(f"❌ CRITICAL: Job {job.get('job_id')} missing LLM fields - skipping")
                continue
            job_data = job.get("job_data", {})
            llm_skills_matched = job.get("skills_matched", [])
            
            # Required skills from job (camelCase from ChromaDB)
            required_skills = job_data.get("requiredSkills", [])
            if isinstance(required_skills, str):
                required_skills = [s.strip() for s in required_skills.split(",")]
            
            # Skills already pre-validated in analyze_job_batch_with_llm (skill dropping before rationale)
            skills_matched = job.get("skills_matched", [])
            skills_unmatched = [s for s in required_skills if s not in skills_matched]
            skill_match_percentage = round(len(skills_matched) / len(required_skills) * 100, 2) if required_skills else 0.0
            rationale = job.get("rationale", "")
            
            # Option B: when skill match is low and LLM is "optimistic", scale down match_score
            llm_match_score = job.get("match_score", 0.0)
            skill_implied_score = skill_match_percentage / 100.0
            llm_optimistic = llm_match_score > skill_implied_score + JobMatcherConfig.LLM_OPTIMISTIC_THRESHOLD
            if (
                skill_match_percentage < JobMatcherConfig.MIN_SKILL_MATCH_PERCENTAGE
                and llm_optimistic
            ):
                effective_match_score = llm_match_score * (
                    skill_match_percentage / JobMatcherConfig.MIN_SKILL_MATCH_PERCENTAGE
                )
                adjusted_count += 1
            else:
                effective_match_score = llm_match_score
                        
            # Resolve job title from job_data or job (Chroma may use jobTitle, title, job_title, position, role)
            job_title_out = (
                job_data.get("jobTitle")
                or job_data.get("title")
                or job_data.get("job_title")
                or job_data.get("position")
                or job_data.get("role")
                or job_data.get("designation")
                or job.get("job_title")
                or "N/A"
            )
            # Compute tier (unified with ranker)
            try:
                jd_experience = job_data.get("experience") or job_data.get("experienceRequired", "")
                jd_location = job_data.get("location") or job_data.get("jobLocation", "")
                jd_education = job_data.get("educationRequired") or job_data.get("education", "")
                jd_work_mode = job_data.get("workMode") or ""
                jd_must_have = job_data.get("_must_have_skills") or []
                
                tier_input = {
                    "skills_matched": skills_matched,
                    "match_score": effective_match_score,
                    "domain_relevant_experience_years": job.get("domain_relevant_experience_years"),
                    "current_location": candidate_location,
                    "education_summary": candidate_education,
                    "candidate": {"structuredResume": structured_resume},
                }
                tier_result = compute_tier_from_llm_results(
                    candidate=tier_input,
                    jd_skills=required_skills,
                    jd_must_have_skills=jd_must_have,
                    jd_experience=jd_experience,
                    jd_location=jd_location,
                    jd_education=jd_education,
                    work_mode=jd_work_mode,
                )
                tier_num = tier_result["tier"]
                tier_name = tier_result["tier_name"]
                tier_action = tier_result["tier_action"]
                tier_criteria = tier_result["criteria"]
                location_bonus = tier_result.get("location_bonus", 0)
                if location_bonus > 0:
                    effective_match_score = round(min(1.0, effective_match_score + location_bonus), 4)
            except Exception:
                tier_num = None
                tier_name = "Unknown"
                tier_action = ""
                tier_criteria = {}
            
            matched_jobs.append({
                "rank": rank,
                "job_id": job["job_id"],
                "job_title": job_title_out,
                "company": job_data.get("company", "N/A"),
                "match_score": round(effective_match_score, 4),
                "skill_match_percentage": round(skill_match_percentage, 1),
                "skill_match_count": len(skills_matched),
                "skills_matched": skills_matched,
                "skills_unmatched": skills_unmatched,
                "rationale": rationale,
                "positive_rationale": job.get("positive_rationale", ""),
                "negative_rationale": job.get("negative_rationale", ""),
                "how_to_improve": job.get("how_to_improve", ""),
                "vector_similarity": job.get("vector_similarity") or job.get("_vector_similarity", 0.0),
                "tier": tier_num,
                "tier_name": tier_name,
                "tier_action": tier_action,
                "tier_criteria": tier_criteria,
            })
        
        # Re-sort by effective match score (Option B may have lowered some) and reassign ranks
        matched_jobs.sort(key=lambda j: j.get("match_score", 0.0), reverse=True)
        for idx, j in enumerate(matched_jobs, start=1):
            j["rank"] = idx
        
        if adjusted_count > 0:
            log.info(
                f"📉 Adjusted score for {adjusted_count} jobs with skill match below {JobMatcherConfig.MIN_SKILL_MATCH_PERCENTAGE}% "
                f"(fit reflected in lower score; {len(matched_jobs)} jobs ranked)"
            )
        
        if not matched_jobs:
            state["matched_jobs"] = []
            state["top_matches"] = []
            state["total_matches_found"] = 0
            state["job_matcher_status"] = "no_matches"
            state["message"] = "No matching jobs found"
            return state
        
        log.info("="*70)
        log.info("🎯 FINAL JOB MATCHES (LLM Match Score)")
        log.info("="*70)
        for job in matched_jobs[:10]:  # Show top 10
            tier_label = f"T{job.get('tier', '?')}({job.get('tier_name', '?')})"
            log.info(
                f"Rank {job['rank']}: {job['job_title']} at {job['company']} | "
                f"Match: {job['match_score']:.3f} | "
                f"Skills: {job['skill_match_count']} matched | "
                f"Vector: {job.get('vector_similarity', 0.0):.3f} | "
                f"{tier_label}"
            )
        log.info("="*70)
        
        # Calculate metrics
        processing_time = time.time() - _matcher_start
        confidence_score = (
            sum(j["match_score"] for j in matched_jobs) / len(matched_jobs)
            if matched_jobs else 0.0
        )
        
        # ✅ FINAL VALIDATION: Ensure all returned jobs were LLM-analyzed
        for job in matched_jobs:
            if (job.get("rationale") is None and job.get("positive_rationale") is None) or job.get("match_score") is None:
                log.error(f"❌ CRITICAL: Job {job.get('job_id')} in results but missing LLM analysis!")
                raise ValueError("Non-LLM analyzed job found in final results")
        
        log.info(f"✅ VALIDATED: All {len(matched_jobs)} returned jobs were LLM-analyzed")
        
        # If called from chatbot, limit results to chatbot_top_k (3-4 matches)
        chatbot_top_k = state.get("chatbot_top_k")
        if chatbot_top_k is not None and isinstance(chatbot_top_k, int):
            # Limit to chatbot_top_k (typically 3-4) when called from chatbot
            original_count = len(matched_jobs)
            matched_jobs = matched_jobs[:chatbot_top_k]
            log.info(f"🎯 Chatbot mode: Limited results to top {chatbot_top_k} matches (from {original_count} total)")
        
        state["matched_jobs"] = matched_jobs
        state["top_matches"] = matched_jobs  # ✅ FIX: Add for backward compatibility with app.py
        state["confidence_score"] = round(confidence_score, 4)
        state["processing_time_seconds"] = round(processing_time, 3)
        state["total_jobs_matched"] = len(matched_jobs)
        state["total_matches_found"] = len(matched_jobs)  # ✅ FIX: Add for backward compatibility
        state["semantic_filter_threshold"] = SEMANTIC_THRESHOLD
        state["processing_method"] = "semantic_filter_batch_llm_v2"
        state["batches_processed"] = len(batches)
        state["job_matcher_status"] = "partial" if partial_timeout else "success"

        # ── Merged compare: build candidate_job_match_result from the primary-JD batch row ──
        if _compare_primary_job_id and matched_jobs:
            primary_row = next(
                (j for j in matched_jobs if j.get("job_id") == _compare_primary_job_id), None
            )
            if primary_row:
                similar_jobs: list = []
                similar_source = "batch_peers"
                try:
                    stored = await run_cpu_intensive(
                        get_top_matched_jobs_for_candidate, uid, _compare_primary_job_id, 10
                    )
                    if stored:
                        similar_jobs = stored
                        similar_source = "cache"
                    else:
                        primary_jd = next(
                            (d for d in job_docs if str(d.get("job_id") or d.get("jd_id") or d.get("id", "")) == _compare_primary_job_id),
                            None,
                        )
                        if primary_jd:
                            semantic = await run_cpu_intensive(
                                get_similar_jobs_for_job, primary_jd, _compare_primary_job_id, 10
                            )
                            if semantic:
                                similar_jobs = semantic
                                similar_source = "semantic_search"
                except Exception as sim_err:
                    log.warning("⚠️ similar_jobs enrichment failed: %s", sim_err)

                candidate_result = {
                    "candidate_id": uid,
                    "job_id": _compare_primary_job_id,
                    "match_score": primary_row.get("match_score", 0.0),
                    "skill_match_percentage": primary_row.get("skill_match_percentage", 0.0),
                    "skill_match_count": primary_row.get("skill_match_count", 0),
                    "skills_matched": primary_row.get("skills_matched", []),
                    "skills_unmatched": primary_row.get("skills_unmatched", []),
                    "total_required_skills": len(primary_row.get("skills_matched", []))
                        + len(primary_row.get("skills_unmatched", [])),
                    "vector_similarity": primary_row.get("vector_similarity", 0.0),
                    "rationale": primary_row.get("rationale", ""),
                    "positive_rationale": primary_row.get("positive_rationale", ""),
                    "negative_rationale": primary_row.get("negative_rationale", ""),
                    "processing_method": "job_matcher_merged_compare",
                    "similar_jobs": similar_jobs,
                    "similar_jobs_source": similar_source,
                    "tier": primary_row.get("tier"),
                    "tier_name": primary_row.get("tier_name", "Unknown"),
                    "tier_action": primary_row.get("tier_action", ""),
                    "tier_criteria": primary_row.get("tier_criteria", {}),
                    "hire_recommendation": primary_row.get("hire_recommendation"),
                    "hire_rationale": primary_row.get("hire_rationale"),
                    "interview_feedback_used": bool(interview_feedback),
                }
                state["candidate_job_match_result"] = candidate_result
                state["match_score"] = candidate_result["match_score"]
                state["skill_match_percentage"] = candidate_result["skill_match_percentage"]
                state["skill_match_count"] = candidate_result["skill_match_count"]
                state["skills_matched"] = candidate_result["skills_matched"]
                state["skills_unmatched"] = candidate_result["skills_unmatched"]
                log.info(
                    "✅ Merged compare: candidate_job_match_result built from batch for primary job %s "
                    "(score=%.2f%%, similar_jobs=%d [%s])",
                    _compare_primary_job_id, candidate_result["match_score"] * 100,
                    len(similar_jobs), similar_source,
                )
            else:
                log.warning("⚠️ Primary job %s not found in batch results", _compare_primary_job_id)

        elif state.get("candidate_job_match_result"):
            log.info("✅ Preserved candidate_job_match_result from single-JD compare (multi-job run)")
        if partial_timeout:
            state["partial_timeout"] = True
            state["jobs_analyzed"] = len(analyzed_jobs)
            state["jobs_total"] = len(jobs_for_llm)
        
        log.info(
            f"✅ Job Matcher V2 complete: {len(matched_jobs)} jobs matched in {processing_time:.2f}s"
            + (" (partial - approaching timeout)" if partial_timeout else "")
        )
        
        return state
        
    except Exception as e:
        log.error(f"❌ Unexpected error in job_matcher_agent_v2: {e}", exc_info=True)
                        
        return {
            "error": str(e),
            "status": "error",
                "top_matches": [],
            "matched_jobs": [],
                "total_matches_found": 0,
            "job_matcher_status": "error"
        }