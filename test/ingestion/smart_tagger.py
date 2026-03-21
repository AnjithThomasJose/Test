"""
Smart Tagger - LLM-based Metadata Extraction

This module uses LLM (Gemini Flash) to extract searchable metadata from resumes.
The extracted metadata is used for hybrid search (vector + metadata filtering).

Extracted Metadata:
- skills: List of technical and soft skills
- total_years_exp: Total years of experience (float)
- current_city: Current location/city
- seniority_level: junior, mid, senior, lead
- education_level: high_school, bachelors, masters, phd
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llm_invoker import invoke_llm
from core.model_registry import TaskType

log = logging.getLogger(__name__)


@dataclass
class ResumeMetadata:
    """
    Structured metadata extracted from resume.
    
    This metadata is used for filtering in hybrid search operations.
    """
    skills: List[str]
    total_years_exp: float
    current_city: str
    seniority_level: str  # junior, mid, senior, lead
    education_level: str  # high_school, bachelors, masters, phd
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ChromaDB metadata"""
        return {
            "skills": ", ".join(self.skills),  # ChromaDB doesn't support lists
            "total_years_exp": self.total_years_exp,
            "current_city": self.current_city,
            "seniority_level": self.seniority_level,
            "education_level": self.education_level
        }


class SmartTagger:
    """
    LLM-based metadata extractor for resumes.
    
    Uses Gemini Flash for fast, cost-effective metadata extraction.
    The extracted metadata enables hybrid search (vector + filters).
    
    Usage:
        tagger = SmartTagger()
        metadata = await tagger.extract_metadata(structured_resume)
        
        # Use metadata for ChromaDB filtering
        filters = {
            "seniority_level": metadata.seniority_level,
            "current_city": metadata.current_city
        }
    """
    
    def __init__(self, model: str = "gemini-2.5-flash"):
        """
        Initialize Smart Tagger.
        
        Args:
            model: Gemini model to use (default: gemini-2.5-flash)
        """
        self.model = model
        log.info(f"SmartTagger initialized with model: {model}")
    
    def _create_extraction_prompt(self, structured_resume: Dict[str, Any]) -> str:
        """
        Create prompt for metadata extraction.
        
        Args:
            structured_resume: Parsed resume data
            
        Returns:
            Prompt string for LLM
        """
        # Extract key information from resume
        name = structured_resume.get("Name") or structured_resume.get("name", "")
        
        # Skills
        skills_data = structured_resume.get("Skills") or structured_resume.get("skills", [])
        skills_list = []
        if isinstance(skills_data, list):
            for skill in skills_data[:20]:  # Limit to top 20
                if isinstance(skill, dict):
                    skill_name = skill.get("SkillName") or skill.get("Name", "")
                    if skill_name:
                        skills_list.append(skill_name)
                elif isinstance(skill, str):
                    skills_list.append(skill)
        
        # Work Experience
        work_exp = structured_resume.get("WorkExperience") or structured_resume.get("work_experience") or structured_resume.get("experience", [])
        work_summary = []
        if isinstance(work_exp, list):
            for exp in work_exp[:3]:  # Top 3 experiences
                if isinstance(exp, dict):
                    title = exp.get("job_title") or exp.get("title", "")
                    company = exp.get("company", "")
                    if title:
                        work_summary.append(f"{title} at {company}")
        
        # Education
        education = structured_resume.get("Education") or structured_resume.get("education", [])
        education_summary = []
        if isinstance(education, list):
            for edu in education:
                if isinstance(edu, dict):
                    degree = edu.get("degree", "")
                    if degree:
                        education_summary.append(degree)
        
        # Total experience
        total_exp = structured_resume.get("total_experience_years", "0 years")
        
        # Location
        location = structured_resume.get("Location") or ""
        if not location:
            contact = structured_resume.get("ContactDetails", {})
            if isinstance(contact, dict):
                location = contact.get("Location", "")
        
        prompt = f"""Extract metadata from this resume for search filtering.

RESUME INFORMATION:
Name: {name}
Skills: {', '.join(skills_list) if skills_list else 'Not specified'}
Work Experience: {'; '.join(work_summary) if work_summary else 'Not specified'}
Education: {'; '.join(education_summary) if education_summary else 'Not specified'}
Total Experience: {total_exp}
Location: {location or 'Not specified'}

TASK: Extract the following metadata in JSON format:

1. **skills**: List of top 10 technical and professional skills (e.g., ["Python", "React", "Project Management"])
   - Extract from Skills section
   - Include both technical and soft skills
   - Limit to 10 most important skills

2. **total_years_exp**: Total years of professional experience as a float (e.g., 5.5)
   - Parse from "Total Experience" field
   - Convert strings like "5 years 6 months" to 5.5
   - If unclear, estimate from work history
   - Minimum: 0.0

3. **current_city**: Current city/location (e.g., "Dubai", "New York", "Remote")
   - Extract from Location field
   - Use city name only (not full address)
   - If not specified, use "Unknown"

4. **seniority_level**: Career level (MUST be one of: "junior", "mid", "senior", "lead")
   - junior: 0-2 years experience
   - mid: 2-5 years experience
   - senior: 5-10 years experience
   - lead: 10+ years experience
   - Base on total_years_exp and job titles

5. **education_level**: Highest education (MUST be one of: "high_school", "bachelors", "masters", "phd")
   - high_school: High school diploma or equivalent
   - bachelors: Bachelor's degree (BS, BA, B.Tech, etc.)
   - masters: Master's degree (MS, MA, MBA, etc.)
   - phd: Doctoral degree (PhD, etc.)
   - If not specified, use "bachelors" as default

CRITICAL RULES:
- Return ONLY valid JSON, no markdown or extra text
- Use exact field names as specified
- seniority_level MUST be one of: junior, mid, senior, lead
- education_level MUST be one of: high_school, bachelors, masters, phd
- skills must be a list of strings (max 10 items)
- total_years_exp must be a number (float)
- current_city must be a string

RESPONSE FORMAT:
{{
  "skills": ["skill1", "skill2", ...],
  "total_years_exp": 5.5,
  "current_city": "Dubai",
  "seniority_level": "senior",
  "education_level": "bachelors"
}}

Return ONLY the JSON object, nothing else."""
        
        return prompt
    
    async def extract_metadata(self, structured_resume: Dict[str, Any]) -> ResumeMetadata:
        """
        Extract metadata from structured resume using LLM.
        
        Args:
            structured_resume: Parsed resume data from groq_resume_parser
            
        Returns:
            ResumeMetadata object with extracted fields
            
        Raises:
            ValueError: If extraction fails
        """
        try:
            # Create prompt
            prompt = self._create_extraction_prompt(structured_resume)
            
            log.info("Calling LLM for metadata extraction...")
            raw = await asyncio.wait_for(
                invoke_llm(
                    prompt=prompt,
                    task_type=TaskType.RESUME_ANALYSIS,
                    preferred_model=self.model,
                    agent_name="smart_tagger",
                    response_mime_type="application/json",
                ),
                timeout=30.0,
            )
            
            content = (raw or "").strip()
            
            # Remove markdown code blocks if present
            import re
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            
            # Parse JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                log.error(f"Failed to parse LLM response as JSON: {e}")
                log.debug(f"Response content: {content[:500]}")
                raise ValueError(f"Invalid JSON response from LLM: {e}")
            
            # Validate and create metadata object
            metadata = ResumeMetadata(
                skills=data.get("skills", [])[:10],  # Limit to 10
                total_years_exp=float(data.get("total_years_exp", 0.0)),
                current_city=data.get("current_city", "Unknown"),
                seniority_level=data.get("seniority_level", "mid"),
                education_level=data.get("education_level", "bachelors")
            )
            
            # Validate seniority_level
            valid_seniority = ["junior", "mid", "senior", "lead"]
            if metadata.seniority_level not in valid_seniority:
                log.warning(f"Invalid seniority_level: {metadata.seniority_level}, defaulting to 'mid'")
                metadata.seniority_level = "mid"
            
            # Validate education_level
            valid_education = ["high_school", "bachelors", "masters", "phd"]
            if metadata.education_level not in valid_education:
                log.warning(f"Invalid education_level: {metadata.education_level}, defaulting to 'bachelors'")
                metadata.education_level = "bachelors"
            
            log.info(
                f"Metadata extracted: {len(metadata.skills)} skills, "
                f"{metadata.total_years_exp} years exp, "
                f"{metadata.seniority_level} level, "
                f"{metadata.education_level} education"
            )
            
            return metadata
            
        except asyncio.TimeoutError:
            log.error("LLM call timed out")
            raise ValueError("Metadata extraction timed out")
        except Exception as e:
            log.error(f"Metadata extraction failed: {e}", exc_info=True)
            raise ValueError(f"Failed to extract metadata: {str(e)}")
    
    def extract_metadata_fallback(self, structured_resume: Dict[str, Any]) -> ResumeMetadata:
        """
        Fallback metadata extraction without LLM (rule-based).
        
        Used when LLM is unavailable or fails.
        
        Args:
            structured_resume: Parsed resume data
            
        Returns:
            ResumeMetadata with basic extracted fields
        """
        log.info("Using fallback metadata extraction (rule-based)")
        
        # Extract skills
        skills_data = structured_resume.get("Skills") or structured_resume.get("skills", [])
        skills = []
        if isinstance(skills_data, list):
            for skill in skills_data[:10]:
                if isinstance(skill, dict):
                    skill_name = skill.get("SkillName") or skill.get("Name", "")
                    if skill_name:
                        skills.append(skill_name)
                elif isinstance(skill, str):
                    skills.append(skill)
        
        # Extract total experience
        total_exp_str = structured_resume.get("total_experience_years", "0 years")
        total_exp = 0.0
        if isinstance(total_exp_str, (int, float)):
            total_exp = float(total_exp_str)
        elif isinstance(total_exp_str, str):
            # Parse "5 years 6 months" or "5.5 years"
            import re
            years_match = re.search(r'(\d+\.?\d*)\s*years?', total_exp_str.lower())
            if years_match:
                total_exp = float(years_match.group(1))
            months_match = re.search(r'(\d+)\s*months?', total_exp_str.lower())
            if months_match:
                total_exp += float(months_match.group(1)) / 12.0
        
        # Extract location
        location = structured_resume.get("Location", "Unknown")
        if not location or location == "Unknown":
            contact = structured_resume.get("ContactDetails", {})
            if isinstance(contact, dict):
                location = contact.get("Location", "Unknown")
        
        # Determine seniority level
        if total_exp < 2:
            seniority = "junior"
        elif total_exp < 5:
            seniority = "mid"
        elif total_exp < 10:
            seniority = "senior"
        else:
            seniority = "lead"
        
        # Determine education level
        education = structured_resume.get("Education") or structured_resume.get("education", [])
        education_level = "bachelors"  # Default
        if isinstance(education, list) and education:
            # Check highest degree
            for edu in education:
                if isinstance(edu, dict):
                    degree = edu.get("degree", "").lower()
                    if "phd" in degree or "doctor" in degree:
                        education_level = "phd"
                        break
                    elif "master" in degree or "mba" in degree or "ms" in degree or "ma" in degree:
                        education_level = "masters"
                    elif "bachelor" in degree or "bs" in degree or "ba" in degree or "b.tech" in degree:
                        education_level = "bachelors"
                    elif "high school" in degree or "diploma" in degree:
                        education_level = "high_school"
        
        metadata = ResumeMetadata(
            skills=skills,
            total_years_exp=total_exp,
            current_city=location,
            seniority_level=seniority,
            education_level=education_level
        )
        
        log.info(f"Fallback metadata: {len(metadata.skills)} skills, {metadata.total_years_exp} years")
        
        return metadata


# Convenience function
async def extract_metadata_tags(structured_resume: Dict[str, Any], use_llm: bool = True) -> ResumeMetadata:
    """
    Extract metadata tags from structured resume.
    
    Args:
        structured_resume: Parsed resume data
        use_llm: Whether to use LLM (True) or fallback (False)
        
    Returns:
        ResumeMetadata object
    """
    tagger = SmartTagger()
    
    if use_llm:
        try:
            return await tagger.extract_metadata(structured_resume)
        except Exception as e:
            log.warning(f"LLM extraction failed, using fallback: {e}")
            return tagger.extract_metadata_fallback(structured_resume)
    else:
        return tagger.extract_metadata_fallback(structured_resume)
