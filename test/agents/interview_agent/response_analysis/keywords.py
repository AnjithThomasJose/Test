"""
LLM-based keyword and skill extraction.

Extracts keywords and skills from response text.
"""

import logging
from typing import List, Optional, Any

from ..llm_utils import invoke_llm

log = logging.getLogger(__name__)


async def extract_keywords_llm(
    text: str,
    client: Optional[Any] = None
) -> List[str]:
    """
    Extract keywords from response using LLM.
    
    Args:
        text: Response text to extract keywords from
        client: Optional custom LLM client
        
    Returns:
        List of lowercase, normalized keywords
    """
    if not text or not text.strip():
        return []
    
    prompt = f"""Extract keywords from this interview response.

RESPONSE TEXT:
"{text.strip()}"

Return STRICT JSON array of keywords (lowercase, no duplicates):
["keyword1", "keyword2", "keyword3"]

Focus on:
- Technical terms
- Tools/technologies
- Concepts
- Domain-specific terms"""
    
    try:
        result = await invoke_llm(
            prompt=prompt,
            model=None,
            enforce_json=True,
            client=client
        )
        
        if result.get("ok") and result.get("json"):
            parsed = result.get("json", [])
            if isinstance(parsed, list):
                # Normalize: lowercase, strip, deduplicate
                keywords = []
                seen = set()
                for kw in parsed:
                    if isinstance(kw, str):
                        kw_lower = kw.lower().strip()
                        if kw_lower and kw_lower not in seen:
                            keywords.append(kw_lower)
                            seen.add(kw_lower)
                return keywords
            elif isinstance(parsed, dict) and "keywords" in parsed:
                # Handle dict format
                kw_list = parsed.get("keywords", [])
                if isinstance(kw_list, list):
                    keywords = []
                    seen = set()
                    for kw in kw_list:
                        if isinstance(kw, str):
                            kw_lower = kw.lower().strip()
                            if kw_lower and kw_lower not in seen:
                                keywords.append(kw_lower)
                                seen.add(kw_lower)
                    return keywords
        
        log.warning(f"LLM keyword extraction failed: {result.get('error')}")
        return []
    
    except Exception as e:
        log.error(f"Error in extract_keywords_llm: {e}")
        return []


async def extract_skills_llm(
    text: str,
    client: Optional[Any] = None
) -> List[str]:
    """
    Extract skills from response using LLM.
    
    Args:
        text: Response text to extract skills from
        client: Optional custom LLM client
        
    Returns:
        List of lowercase, normalized skills
    """
    if not text or not text.strip():
        return []
    
    prompt = f"""Extract skills and technologies from this interview response.

RESPONSE TEXT:
"{text.strip()}"

Return STRICT JSON array of skills (lowercase, no duplicates):
["skill1", "skill2", "skill3"]

Focus on:
- Programming languages
- Frameworks
- Tools
- Technologies
- Methodologies"""
    
    try:
        result = await invoke_llm(
            prompt=prompt,
            model=None,
            enforce_json=True,
            client=client
        )
        
        if result.get("ok") and result.get("json"):
            parsed = result.get("json", [])
            if isinstance(parsed, list):
                # Normalize: lowercase, strip, deduplicate
                skills = []
                seen = set()
                for skill in parsed:
                    if isinstance(skill, str):
                        skill_lower = skill.lower().strip()
                        if skill_lower and skill_lower not in seen:
                            skills.append(skill_lower)
                            seen.add(skill_lower)
                return skills
            elif isinstance(parsed, dict) and "skills" in parsed:
                # Handle dict format
                skills_list = parsed.get("skills", [])
                if isinstance(skills_list, list):
                    skills = []
                    seen = set()
                    for skill in skills_list:
                        if isinstance(skill, str):
                            skill_lower = skill.lower().strip()
                            if skill_lower and skill_lower not in seen:
                                skills.append(skill_lower)
                                seen.add(skill_lower)
                    return skills
        
        log.warning(f"LLM skill extraction failed: {result.get('error')}")
        return []
    
    except Exception as e:
        log.error(f"Error in extract_skills_llm: {e}")
        return []
