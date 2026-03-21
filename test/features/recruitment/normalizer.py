"""
Normalizer

Uses Groq LLM to standardize messy inputs like locations, skills, and seniority levels.
Ensures consistent values for database filtering.
"""

import asyncio
import json
import logging
import os
from typing import Optional, List, Dict, Any

from langchain_groq import ChatGroq

from settings import settings

log = logging.getLogger(__name__)

# Async-layer cap (ChatGroq also sets httpx timeout=30)
_NORMALIZER_LLM_TIMEOUT = float(os.getenv("NORMALIZER_LLM_TIMEOUT_SECONDS", "45"))


class Normalizer:
    """
    LLM-based normalizer using Groq for recruitment-specific value normalization.
    
    Example:
        Input: "NYC" → Output: "New York"
        Input: "React" → Output: ["React", "ReactJS", "React.js"]
        Input: "Sr" → Output: "senior"
    """
    
    def __init__(self):
        """Initialize Normalizer with Groq model"""
        self.groq_model = ChatGroq(
            model=settings.GROQ_MODEL,
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0.0,
            max_retries=2,
            timeout=30,
            model_kwargs={
                "top_p": 0.0,
                "max_completion_tokens": 500,
            },
        )
        log.info("Groq-based Normalizer initialized")
    
    async def normalize_location(self, location: Optional[str]) -> Optional[str]:
        """
        Normalize location to standard format using Groq.
        
        Args:
            location: Raw location string (e.g., "NYC", "Dubai", "Bay Area")
            
        Returns:
            Normalized location (e.g., "New York", "Dubai", "San Francisco")
        """
        if not location:
            return None
        
        prompt = f"""Normalize this location to a standard city/region name. Return ONLY the normalized location name, nothing else.

Examples:
- "NYC" → "New York"
- "DXB" → "Dubai"
- "Bay Area" → "San Francisco"
- "NCR" → "Delhi"

Input: {location}
Normalized:"""
        
        try:
            response = await asyncio.wait_for(
                self.groq_model.ainvoke(prompt), timeout=_NORMALIZER_LLM_TIMEOUT
            )
            normalized = response.content.strip().strip('"').strip("'")
            log.debug(f"Location normalized: {location} → {normalized}")
            return normalized if normalized else location.title()
        except Exception as e:
            log.warning(f"Groq normalization failed for location '{location}': {e}, using title case")
            return location.title()
    
    async def normalize_skill(self, skill: str) -> List[str]:
        """
        Normalize skill to include synonyms using Groq.
        
        Args:
            skill: Raw skill string (e.g., "React", "nodejs", "k8s")
            
        Returns:
            List of skill synonyms (e.g., ["React", "ReactJS", "React.js"])
        """
        if not skill:
            return []
        
        prompt = f"""Given a technical skill name, return a JSON array of common synonyms/variations.

Examples:
- "react" → ["React", "ReactJS", "React.js"]
- "nodejs" → ["Node.js", "NodeJS", "Node"]
- "k8s" → ["Kubernetes", "K8s"]

Return ONLY a valid JSON array, nothing else.

Input: {skill}
JSON array:"""
        
        try:
            response = await asyncio.wait_for(
                self.groq_model.ainvoke(prompt), timeout=_NORMALIZER_LLM_TIMEOUT
            )
            content = response.content.strip()
            # Try to extract JSON array
            if content.startswith('['):
                synonyms = json.loads(content)
            elif '[' in content:
                start = content.index('[')
                end = content.rindex(']') + 1
                synonyms = json.loads(content[start:end])
            else:
                synonyms = [skill.title()]
            
            if not isinstance(synonyms, list) or len(synonyms) == 0:
                synonyms = [skill.title()]
            
            log.debug(f"Skill normalized: {skill} → {synonyms}")
            return synonyms
        except Exception as e:
            log.warning(f"Groq normalization failed for skill '{skill}': {e}, using title case")
            return [skill.title()]
    
    async def normalize_skills(self, skills: List[str]) -> List[str]:
        """
        Normalize multiple skills and flatten synonyms.
        
        Args:
            skills: List of raw skill strings
            
        Returns:
            Flattened list of normalized skills with synonyms
        """
        if not skills:
            return []
        
        # Normalize all skills in parallel
        tasks = [self.normalize_skill(skill) for skill in skills]
        normalized_lists = await asyncio.gather(*tasks)
        
        # Flatten and deduplicate
        normalized = []
        seen = set()
        for skill_list in normalized_lists:
            for skill in skill_list:
                if skill.lower() not in seen:
                    seen.add(skill.lower())
                    normalized.append(skill)
        
        log.debug(f"Skills normalized: {skills} → {normalized}")
        return normalized
    
    async def normalize_seniority(self, seniority: Optional[str]) -> Optional[str]:
        """
        Normalize seniority level to standard format using Groq.
        
        Args:
            seniority: Raw seniority string (e.g., "Sr", "Lead", "Entry-level")
            
        Returns:
            Normalized seniority (e.g., "senior", "senior", "junior")
        """
        if not seniority:
            return None
        
        prompt = f"""Normalize this seniority level to one of: "junior", "mid", "senior", or "lead". Return ONLY the normalized value.

Examples:
- "Sr" → "senior"
- "Lead" → "senior"
- "Entry-level" → "junior"
- "Principal" → "senior"
- "Architect" → "lead"

Input: {seniority}
Normalized:"""
        
        try:
            response = await asyncio.wait_for(
                self.groq_model.ainvoke(prompt), timeout=_NORMALIZER_LLM_TIMEOUT
            )
            normalized = response.content.strip().lower().strip('"').strip("'")
            # Validate
            valid_levels = {"junior", "mid", "senior", "lead"}
            if normalized not in valid_levels:
                # Try to extract from response
                for level in valid_levels:
                    if level in normalized.lower():
                        normalized = level
                        break
                else:
                    normalized = None
            
            log.debug(f"Seniority normalized: {seniority} → {normalized}")
            return normalized
        except Exception as e:
            log.warning(f"Groq normalization failed for seniority '{seniority}': {e}")
            return None
    
    async def normalize_education(self, education: Optional[str]) -> Optional[str]:
        """
        Normalize education level to standard format using Groq.
        
        Args:
            education: Raw education string (e.g., "Bachelor's", "MS", "PhD")
            
        Returns:
            Normalized education (e.g., "bachelors", "masters", "phd")
        """
        if not education:
            return None
        
        prompt = f"""Normalize this education level to one of: "high_school", "bachelors", "masters", or "phd". Return ONLY the normalized value.

Examples:
- "Bachelor's" → "bachelors"
- "MS" → "masters"
- "PhD" → "phd"
- "BSc" → "bachelors"

Input: {education}
Normalized:"""
        
        try:
            response = await asyncio.wait_for(
                self.groq_model.ainvoke(prompt), timeout=_NORMALIZER_LLM_TIMEOUT
            )
            normalized = response.content.strip().lower().strip('"').strip("'")
            # Normalize underscores
            normalized = normalized.replace(" ", "_")
            # Validate
            valid_levels = {"high_school", "bachelors", "masters", "phd"}
            if normalized not in valid_levels:
                # Try to extract from response
                for level in valid_levels:
                    if level.replace("_", " ") in normalized or normalized in level:
                        normalized = level
                        break
                else:
                    normalized = None
            
            log.debug(f"Education normalized: {education} → {normalized}")
            return normalized
        except Exception as e:
            log.warning(f"Groq normalization failed for education '{education}': {e}")
            return None
    
    async def normalize_all(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize all fields in a dictionary using Groq.
        
        Args:
            data: Dictionary with raw values
            
        Returns:
            Dictionary with normalized values
        """
        normalized = {}
        tasks = []
        fields = []
        
        if "location" in data:
            tasks.append(self.normalize_location(data["location"]))
            fields.append("location")
        else:
            tasks.append(None)
            fields.append(None)
        
        if "skills" in data:
            tasks.append(self.normalize_skills(data["skills"]))
            fields.append("skills")
        else:
            tasks.append(None)
            fields.append(None)
        
        if "seniority" in data:
            tasks.append(self.normalize_seniority(data["seniority"]))
            fields.append("seniority")
        else:
            tasks.append(None)
            fields.append(None)
        
        if "education" in data:
            tasks.append(self.normalize_education(data["education"]))
            fields.append("education")
        else:
            tasks.append(None)
            fields.append(None)
        
        # Execute all normalizations in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (field, result) in enumerate(zip(fields, results)):
            if field and not isinstance(result, Exception):
                normalized[field] = result
        
        # Copy other fields as-is
        for key, value in data.items():
            if key not in normalized:
                normalized[key] = value
        
        log.info(f"Normalized all fields: {len(normalized)} fields processed")
        return normalized
