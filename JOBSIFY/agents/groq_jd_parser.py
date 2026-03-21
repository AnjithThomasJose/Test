import logging
import asyncio
import json
import re
from typing import Dict, Any, Optional, List

from langchain_groq import ChatGroq
from pydantic import ValidationError
from core.text_cleaner import (
    clean_full_description, 
    clean_jd_fields, 
    clean_experience,
    clean_salary,
    clean_and_normalize_jd_text
)

from agents.groq_jd_prompt import (
    generate_comprehensive_jd_prompt,
)

from settings import settings
from agents.jd_schema import StructuredJDOutput, extract_minimal_jd
from utils.normalize_text import normalize
from chroma import insert_job_description, get_job_description, update_job_description
from core.utils import run_blocking_io
from core.config import get_agent_config
from core.security import (
    validate_tenant_id, redact_pii, filter_injection_attempts, sanitize_text_for_llm,
    PII_PATTERNS, INJECTION_FILTERS
)
from core.memory import BaseAgentMemory, get_agent_memory

log = logging.getLogger(__name__)


def _dedupe_skills_case_insensitive(skills: List) -> List[str]:
    """Remove duplicate skills by case-insensitive comparison; preserve first occurrence and original casing."""
    if not skills:
        return []
    seen = set()
    out = []
    for s in skills:
        if not isinstance(s, str) or not s.strip():
            continue
        key = s.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s.strip())
    return out


# Phrase boundaries that often concatenate multiple skills into one sentence-like string
_SKILL_SENTENCE_SPLIT_PATTERNS = re.compile(
    r"\s+(?:Experience in|Experience with|Knowledge of|Good knowledge of|Basic knowledge of|"
    r"Familiar with|Understanding of|Proficiency in|Working knowledge of|Ability to)\s+",
    re.IGNORECASE
)
_MAX_SKILL_CHARS = 50  # Treat as sentence-like if longer than this


def _split_sentence_like_skills(skills: List) -> List[str]:
    """
    Split any skill item that looks like a concatenated sentence into atomic skills.
    E.g. "Good knowledge of SDLC Experience in test cases Knowledge of Jira" -> ["SDLC", "Test cases", "Jira"].
    """
    if not skills:
        return []
    result = []
    for item in skills:
        if not isinstance(item, str) or not item.strip():
            continue
        s = item.strip()
        # Already short and no sentence-like phrase -> keep as-is
        if len(s) <= _MAX_SKILL_CHARS and not _SKILL_SENTENCE_SPLIT_PATTERNS.search(s):
            result.append(s)
            continue
        # Split on phrase boundaries and clean each part
        parts = _SKILL_SENTENCE_SPLIT_PATTERNS.split(s)
        for part in parts:
            part = part.strip()
            # Also split on " and " / " or " when part is still long (likely list)
            if len(part) > _MAX_SKILL_CHARS and (" and " in part or " or " in part):
                for sub in re.split(r"\s+and\s+|\s+or\s+", part, flags=re.IGNORECASE):
                    sub = sub.strip().strip(".,;")
                    if len(sub) > 1 and len(sub) < 120:
                        result.append(sub)
            elif len(part) > 1 and len(part) < 120:
                part = part.strip(".,;")
                result.append(part)
    return _dedupe_skills_case_insensitive(result)


def clean_for_llm_jd(raw_text: str, max_chars: int = None, fallback_chars: int = 20000) -> str:
    # Use config value if max_chars not provided
    if max_chars is None:
        max_chars = JD_LLM_INPUT_MAX_CHARS
    
    # Step 1: Normalize the text
    s = normalize(raw_text or "")
    
    # Step 2: Remove emojis and decorative Unicode characters
    s = re.sub(r'[\U0001F300-\U0001F9FF]', '', s)  
    s = re.sub(r'[\U0001FA00-\U0001FAFF]', '', s)  
    s = re.sub(r'[💸🚀💰🔥📊✅🌱🔒🛠🎯🎓🗣🔥⚙]', '', s) 
    
    # Remove null bytes and other problematic characters
    s = re.sub(r'[\u0000]', '', s)
    
    # Step 3: Remove null tokens
    null_tokens = {"null", "n/a", "na", "none"}
    cleaned_lines: List[str] = []
    for line in s.splitlines():
        words = [w for w in line.split() if w.lower() not in null_tokens]
        cleaned_lines.append(" ".join(words))
    s = "\n".join(cleaned_lines).strip()
    
    # Step 4: Strip bullet prefixes (reduce bias)
    bullet_chars = ("- ", "• ", "▸ ", "► ", "· ")
    debulleted: List[str] = []
    for line in s.splitlines():
        stripped = line
        for b in bullet_chars:
            if stripped.startswith(b):
                stripped = stripped[len(b):]
                break
        debulleted.append(stripped)
    s = "\n".join(debulleted)
    
    # Step 5: Deduplicate consecutive identical lines
    deduped: List[str] = []
    last_line = None
    for line in s.splitlines():
        if line != last_line:
            deduped.append(line)
        last_line = line
    s = "\n".join(deduped)
    
    # Step 6: Truncate to hard limit with smart boundary detection
    try:
        char_limit = max_chars
    except:
        char_limit = fallback_chars
    
    if len(s) > char_limit:
        truncated = s[:char_limit]
        # Try to truncate at sentence boundary
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        cutoff = max(last_period, last_newline)
        
        # Only use sentence boundary if we're not losing too much content
        if cutoff > char_limit * 0.9:
            s = truncated[:cutoff + 1]
        else:
            s = truncated
        
        log.info(f"📏 JD truncated from {len(raw_text)} to {len(s)} characters")
    
    # Step 7: Apply injection filtering (without PII redaction)
    s = filter_injection_attempts(s)
    
    return s.strip()

# Config with safe defaults (preserve existing behavior)
config = get_agent_config("groq_jd_parser")
GROQ_TIMEOUT_SECONDS = getattr(config, "timeout_seconds", 120)
GROQ_MAX_RETRIES = getattr(config, "llm_retry_attempts", 3)
GROQ_MAX_COMPLETION_TOKENS = getattr(config, "max_completion_tokens", 8000)
JD_LLM_INPUT_MAX_CHARS = getattr(config, "max_prompt_chars", 30000)

def _create_groq_model(temperature: float = 0.0, max_retries: int = None) -> ChatGroq:
    """Create a Groq model instance."""
    if max_retries is None:
        max_retries = GROQ_MAX_RETRIES
    return ChatGroq(
        model=settings.GROQ_MODEL,
        groq_api_key=settings.GROQ_API_KEY,
        temperature=temperature,
        max_retries=max_retries,
        timeout=GROQ_TIMEOUT_SECONDS,
        model_kwargs={
            "top_p": 0.0,
            "max_completion_tokens": GROQ_MAX_COMPLETION_TOKENS,
        },
    )


def _get_structured_groq_model(temperature: float = 0.1, max_retries: int = None):
    """Create a Groq model with structured output."""
    if max_retries is None:
        max_retries = GROQ_MAX_RETRIES
    base_model = _create_groq_model(temperature, max_retries)
    return base_model.with_structured_output(StructuredJDOutput, method="json_mode")

async def validate_jd(jd_text: str, jd_url: str, uid: str, tenant_id: str, callback_url: str, job_id: str, body: dict = None):
    log.info("Starting JD validation pipeline...")
    
    log.debug(f"Original JD length: {len(jd_text)} chars")
    
    # 🔥 NEW: Pre-LLM normalization and clamping (uses JD_LLM_INPUT_MAX_CHARS from config)
    normalized_jd_text = clean_for_llm_jd(jd_text)
    
    log.debug(f"After normalization & truncation: {len(normalized_jd_text)} chars")

    # Parse using Groq with normalized text
    parsed_jd = await _parse_jd_with_groq(normalized_jd_text, original_jd_text=jd_text)

    if not parsed_jd:
        log.error("JD parsing failed. Returning invalid response.")
        return {
            "jd_url": jd_url,
            "uid": uid,
            "tenant_id": tenant_id,
            "callback_url": callback_url,
            "job_id": job_id,
            "is_valid_jd": False,
            "job_description": {},
        }

    # Apply post-processing cleaning
    parsed_jd.fullJobDescription = clean_full_description(parsed_jd.fullJobDescription)
    parsed_jd.experience = clean_experience(parsed_jd.experience)
    parsed_jd.salary = clean_salary(parsed_jd.salary)

    # Extract minimal JD fields
    minimal_jd = extract_minimal_jd(parsed_jd)
    
    # Build clean result
    result = {
        "jd_url": jd_url,
        "uid": uid,
        "tenant_id": tenant_id,
        "callback_url": callback_url,
        "job_id": job_id,
        "is_valid_jd": True,
        "job_description": minimal_jd,
    }

    log.info(f"JD validation complete.")
    log.debug(f"Response keys: {list(result.keys())}")
    log.debug(f"Job description keys: {list(result['job_description'].keys())}")
    log.debug(f"Experience: {result['job_description']['experience']}")
    log.debug(f"Salary: {result['job_description']['salary']}")
    
    return result

async def _parse_jd_with_groq(
    jd_text: str,
    original_jd_text: Optional[str] = None
) -> Optional[StructuredJDOutput]:    
    log.info(f"🤖 Parsing JD with Groq")
    log.debug(f"📏 Input text length: {len(jd_text)} chars")

    # Use fixed timeout from config (MATCHING RESUME PARSER - no length-based adjustment)
    timeout_duration = GROQ_TIMEOUT_SECONDS
    
    # Use single comprehensive prompt
    prompt = generate_comprehensive_jd_prompt(jd_text)

    try:
        # TIER 1: Try structured output first (best reliability)
        try:
            structured_model = _get_structured_groq_model(temperature=0.1, max_retries=1)
            log.debug(f"Attempting Groq structured output mode...")

            result = await asyncio.wait_for(
                structured_model.ainvoke(prompt),
                timeout=timeout_duration
            )

            if isinstance(result, StructuredJDOutput):
                log.debug(f"Structured output received from Groq")
                log.info(f"Extracted: jobTitle={result.jobTitle or 'N/A'}, company={result.company}, location={result.location}")
                return result

        except Exception as structured_error:
            error_str = str(structured_error)
            log.warning(f"Structured output failed: {error_str[:100]}")
            log.info(f"Falling back to manual JSON parsing...")

        # TIER 2: Fallback to manual JSON parsing (MATCHING RESUME PARSER)
        groq_model = _create_groq_model(temperature=0.0, max_retries=1)

        json_prompt = f"""You MUST return ONLY a valid JSON object. NOTHING ELSE.

{prompt}

🚨 Return ONLY JSON starting with {{ and ending with }}."""

        raw_response = await asyncio.wait_for(
            groq_model.ainvoke(json_prompt),
            timeout=timeout_duration
        )

        response_text = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)

        # Extract JSON from response
        json_str = None
        
        # Method 1: Look for ```json code blocks
        if '```json' in response_text.lower():
            parts = re.split(r'```json\s*', response_text, flags=re.IGNORECASE)
            if len(parts) > 1:
                json_part = parts[1]
                end_idx = json_part.find('```')
                json_str = json_part[:end_idx].strip() if end_idx > 0 else json_part.strip()

        # Method 2: Find balanced braces
        if not json_str:
            first_brace = response_text.find('{')
            if first_brace >= 0:
                brace_count = 0
                for i in range(first_brace, len(response_text)):
                    if response_text[i] == '{':
                        brace_count += 1
                    elif response_text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = response_text[first_brace:i+1]
                            break

        if not json_str:
            raise ValueError("No valid JSON found in response")

        # Parse JSON
        parsed_dict = json.loads(json_str)

        # Normalize and create StructuredJDOutput (dedupe skills so "python" and "Python" become one)
        normalized_dict = {
            'jobTitle': (parsed_dict.get('jobTitle') or '').strip(),
            'company': parsed_dict.get('company', ''),
            'location': parsed_dict.get('location', ''),
            'jobType': parsed_dict.get('jobType', 'Full-Time'),
            'experience': parsed_dict.get('experience', ''),
            'educationRequired': parsed_dict.get('educationRequired', ''),
            'salary': parsed_dict.get('salary', ''),
            'requiredSkills': _dedupe_skills_case_insensitive(parsed_dict.get('requiredSkills', [])),
            'preferredSkills': _dedupe_skills_case_insensitive(parsed_dict.get('preferredSkills', [])),
            'jobDomains': parsed_dict.get('jobDomains', []),
            'fullJobDescription': parsed_dict.get('fullJobDescription', jd_text),
            'fileUrl': '',
        }

        result = StructuredJDOutput(**normalized_dict)
        log.info(f"Groq parsing successful")
        log.info(f"Extracted: company={result.company}, location={result.location}")
        return result

    except asyncio.TimeoutError:
        log.warning(f"Groq parsing timed out after {timeout_duration}s")
        return None

    except Exception as e:
        log.error(f"Groq parsing failed: {str(e)[:100]}")
        import traceback
        log.debug(f"Traceback: {traceback.format_exc()}")
        return None

async def groq_jd_parser_agent(
    jd_text: str,
    job_id: str = None,
    uid: str = None,
    company: str = None,
    skip_rerank_on_update: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Main parser agent using Groq - with ChromaDB storage integration.
    Mirrors the resume_assembler pattern for consistency.
    Returns minimal dict or None.
    """
    try:
        log.info("[MODE] Using Groq JD parser")

        if not jd_text or len(jd_text.strip()) < 50:
            log.error("[ERROR] JD text too short or empty")
            return None
        
        log.debug(f"[INFO] Original JD length: {len(jd_text)} chars")
        
        # Apply pre-LLM normalization and clamping
        normalized_jd_text = clean_for_llm_jd(jd_text)
        log.debug(f"[INFO] Normalized from {len(jd_text)} to {len(normalized_jd_text)} chars")

        # Parse with Groq
        parsed_jd = await _parse_jd_with_groq(normalized_jd_text, original_jd_text=jd_text)

        if not parsed_jd:
            log.error("[ERROR] Groq parsing failed")
            return None

        # Apply post-processing cleaning
        parsed_jd.fullJobDescription = clean_full_description(parsed_jd.fullJobDescription)
        parsed_jd.experience = clean_experience(parsed_jd.experience)
        parsed_jd.salary = clean_salary(parsed_jd.salary)

        # Extract minimal result (includes jobTitle when parsed)
        minimal_result = extract_minimal_jd(parsed_jd)
        log.info(f"Successfully parsed: jobTitle={getattr(parsed_jd, 'jobTitle', '') or 'N/A'}, company={parsed_jd.company}, location={parsed_jd.location}")

        # Preserve full extracted JD text; LLM may truncate fullJobDescription due to token limit
        minimal_result["fullJobDescription"] = jd_text
        minimal_result["company"] = company or minimal_result.get("company", "Unknown")
        # Split any sentence-like skill strings into atomic skills (e.g. "Experience in X Knowledge of Y" -> ["X", "Y"])
        minimal_result["requiredSkills"] = _split_sentence_like_skills(minimal_result.get("requiredSkills", []))
        minimal_result["preferredSkills"] = _split_sentence_like_skills(minimal_result.get("preferredSkills", []))
        log.info(f"[INFO] Set company in result: {minimal_result['company']}")
        
        # --- STORAGE INTEGRATION (mirrors resume_assembler pattern) ---
        if job_id:
            try:
                log.debug(f"🔍 JD_PARSER: Starting database save for job_id {job_id}")
                
                # Prepare complete JD data for storage (includes job title when parsed)
                jd_data = {
                    "jobTitle": minimal_result.get("jobTitle", "") or "",
                    "company": minimal_result.get("company", "Unknown"),
                    "location": minimal_result.get("location", ""),
                    "jobType": minimal_result.get("jobType", "Full-Time"),
                    "workMode": minimal_result.get("workMode", ""),
                    "experience": minimal_result.get("experience", ""),
                    "educationRequired": minimal_result.get("educationRequired", ""),
                    "salary": minimal_result.get("salary", ""),
                    "requiredSkills": minimal_result.get("requiredSkills", []),
                    "preferredSkills": minimal_result.get("preferredSkills", []),
                    "jobDomains": minimal_result.get("jobDomains", []),
                    "fullJobDescription": minimal_result.get("fullJobDescription", ""),
                    "fileUrl": minimal_result.get("fileUrl", "")
                }
                
                # Prepare metadata
                metadata = {
                    "uploaded_by_uid": uid or "unknown",
                    "job_id": job_id
                }
                
                log.debug(f"🔍 JD_PARSER: Calling insert_job_description for job_id={job_id}")
                
                # ✅ Check if JD already exists (to detect updates)
                existing_jd = await run_blocking_io(get_job_description, job_id)
                is_update = existing_jd is not None
                
                if is_update:
                    log.info(f"🔄 Detected existing JD for job_id {job_id} - this is an UPDATE")
                else:
                    log.info(f"🆕 New JD for job_id {job_id} - this is a CREATE")
                
                # Store in ChromaDB (mirrors insert_resume call)
                await run_blocking_io(insert_job_description, job_id, jd_data, metadata)
                
                log.debug(f"🔍 JD_PARSER: insert_job_description completed for job_id {job_id}")
                log.info(f"Successfully saved/updated job description for job_id {job_id} in the database.")
                
                # Verify storage (mirrors resume_assembler verification)
                log.debug(f"🔍 JD_PARSER: Verifying storage for job_id={job_id}")
                verify_jd = await run_blocking_io(get_job_description, job_id)
                
                if verify_jd:
                    has_company = bool(verify_jd.get('company'))
                    has_skills = bool(verify_jd.get('requiredSkills'))
                    log.info(f"✅ Verified: JD document exists for job_id {job_id}, has_company={has_company}, has_skills={has_skills}")
                    log.info(f"✅ JD stored successfully: job_id={job_id} (has_company={has_company}, has_skills={has_skills})")
                    
                    # ✅ NEW: If this is an UPDATE, trigger re-ranking with updated JD
                    # Skip when called from corporate flow (job_description_parser -> prescreening_questions -> ranker)
                    # since the graph runs ranker after prescreening_questions
                    if is_update and not skip_rerank_on_update:
                        try:
                            log.info(f"🔄 JD update detected for job_id {job_id}, triggering re-rank with updated JD")
                            
                            # Trigger re-rank with the FRESHLY PARSED JD data (not from ChromaDB)
                            from agents.ranker import ranker_agent
                            
                            # Prepare state with UPDATED JD data from parser
                            state = {
                                "job_description": jd_data,  # ✅ Use freshly parsed JD, not from ChromaDB
                                "job_id": job_id,
                                "tenant_id": uid or "default_tenant",
                                "uid": job_id,
                                "rerank_trigger": "jd_parser_update"  # Mark as triggered by JD parser update
                            }
                            
                            # Trigger async re-rank (non-blocking)
                            async def trigger_rerank_with_updated_jd():
                                try:
                                    log.info(f"🔄 Re-ranking with updated JD for job_id {job_id}")
                                    result = await ranker_agent(state)
                                    ranked_count = len(result.get("ranked_candidates", []))
                                    log.info(f"✅ Re-rank completed for updated JD {job_id}: {ranked_count} candidates ranked")
                                except Exception as e:
                                    log.error(f"❌ Re-rank failed for updated JD {job_id}: {e}")
                            
                            # Schedule async task (non-blocking)
                            try:
                                loop = asyncio.get_event_loop()
                                if loop.is_running():
                                    asyncio.create_task(trigger_rerank_with_updated_jd())
                                    log.info(f"📋 Re-rank task scheduled for updated JD {job_id}")
                                else:
                                    asyncio.run(trigger_rerank_with_updated_jd())
                            except RuntimeError:
                                log.warning(f"⚠️ Could not trigger async re-rank for JD {job_id} (no event loop)")
                        except Exception as rerank_err:
                            log.warning(f"⚠️ Failed to trigger re-rank for updated JD {job_id}: {rerank_err}")
                            # Don't fail JD parsing if re-rank trigger fails
                else:
                    log.error(f"❌ FAILED VERIFICATION: JD document NOT found for job_id {job_id}")
                    log.warning(f"❌ WARNING: JD verification failed for job_id={job_id}")
                    
            except Exception as e_store:
                log.error(f"❌ Could not store JD for job_id {job_id}: {e_store}")
                import traceback
                error_tb = traceback.format_exc()
                log.error(f"❌ Full traceback: {error_tb}")
                log.error(f"❌ ERROR storing JD: {e_store}")
                log.debug(f"❌ ERROR traceback: {error_tb}")
                # Continue even if storage fails - don't break the parsing flow
        else:
            log.info("ℹ️ JD_PARSER: No job_id provided, skipping ChromaDB storage")
        
        return minimal_result

    except Exception as e:
        log.error(f"Parser agent error: {type(e).__name__}: {str(e)}")
        import traceback
        log.debug(f"Traceback: {traceback.format_exc()}")
        return None
