"""
Query Parser

Converts recruiter's natural language queries into structured entities.
Uses LLM to extract: role, location, experience level, skills, education.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional
import uuid

from models.llm_invoker import invoke_llm
from core.model_registry import TaskType

from .models import RecruitmentQuery

log = logging.getLogger(__name__)


class QueryParser:
    """
    Parses recruiter queries using LLM to extract structured entities.
    
    Example:
        Input: "Senior Python Dev in Dubai with 5+ years experience"
        Output: RecruitmentQuery(
            role="Python Developer",
            location="Dubai",
            experience="Senior",
            parsed_skills=["Python"]
        )
    """
    
    def __init__(self, model: str = "gemini-2.5-flash"):
        """
        Initialize Query Parser.
        
        Args:
            model: Gemini model to use (default: gemini-2.5-flash)
        """
        self.model = model
        log.info(f"QueryParser initialized with model: {model}")
    
    async def parse_query(self, raw_query: str) -> RecruitmentQuery:
        """
        Parse recruiter query into structured entities.
        
        Args:
            raw_query: Natural language query from recruiter
            
        Returns:
            RecruitmentQuery with extracted entities
            
        Example:
            >>> parser = QueryParser()
            >>> query = await parser.parse_query("Senior Python Dev in Dubai")
            >>> print(query.parsed_role)  # "Python Developer"
            >>> print(query.parsed_location)  # "Dubai"
            >>> print(query.parsed_experience)  # "Senior"
        """
        log.info(f"Parsing query: {raw_query}")
        
        try:
            # Try LLM-based parsing
            result = await self._parse_with_llm(raw_query)
            log.info(f"LLM parsing successful: role={result.parsed_role}, location={result.parsed_location}")
            return result
        except Exception as e:
            log.warning(f"LLM parsing failed: {e}, using fallback")
            # Fallback to rule-based parsing
            return self._parse_with_rules(raw_query)
    
    async def _parse_with_llm(self, raw_query: str) -> RecruitmentQuery:
        """Parse query using LLM"""
        system_prompt = """You are a recruitment query parser. Extract structured information from recruiter queries AND normalize key fields in one step.

Extract the following fields:
- role: Job title/role (e.g., "Python Developer", "Frontend Engineer")
- location: City or country (e.g., "Dubai", "New York", "Remote")
- experience: Seniority level (e.g., "Senior", "Junior", "Mid-level")
- skills: List of technical skills mentioned (e.g., ["Python", "Django", "AWS"])
- education: Education requirement (e.g., "Bachelor's", "Master's", "PhD")
- company_type: Type of company if mentioned (e.g., "Startup", "Enterprise", "Agency")
- name: Candidate name if the query is searching for a specific person (e.g., "John Smith", "Anjith Thomas")
- university: University or institution name if the query mentions a specific school (e.g., "Stanford", "Amity University", "MIT", "candidates from IIT"). Only set when the query is clearly about institution/school, not degree level. Use null otherwise.

Also provide these NORMALIZED fields:
- location_normalized: Standard city/region name (e.g., "NYC"→"New York", "DXB"→"Dubai", "Bay Area"→"San Francisco"). If location is already standard, repeat it. null if no location.
- seniority_normalized: One of "junior", "mid", "senior", or "lead". Map: Sr/Senior/Staff→"senior", Lead/Principal/Architect→"lead", Mid/Intermediate→"mid", Jr/Junior/Entry-level→"junior". null if no experience level.
- education_normalized: One of "high_school", "bachelors", "masters", or "phd". Map: Bachelor's/BSc/BA/B.Tech→"bachelors", Master's/MS/MSc/MBA/MA→"masters", PhD/Doctorate→"phd", High School/Diploma→"high_school". null if no education.

- limit: Number of candidates to return when the query specifies a quantity. Extract from phrases like "top 3", "give me 5", "first 10", "20 candidates", "top 3 react developers". Use a single integer (e.g. 3, 10). Use null if no number or quantity is mentioned.

IMPORTANT: Only extract name if the query explicitly mentions a person's name. If the query is about a role/location/skills, name should be null.
IMPORTANT: Only extract university if the query explicitly mentions a school/institution name (e.g. "from Amity", "Amity University", "candidates from Stanford"). Do not set for degree level (Bachelor's, Master's).

Return ONLY a valid JSON object with these fields. Use null for missing fields.
Example: {"role": "Python Developer", "location": "Dubai", "experience": "Senior", "skills": ["Python"], "education": null, "company_type": null, "name": null, "university": null, "location_normalized": "Dubai", "seniority_normalized": "senior", "education_normalized": null, "limit": null}
Example with limit: {"role": "React Developer", "location": null, "experience": null, "skills": ["React"], "education": null, "company_type": null, "name": null, "university": null, "location_normalized": null, "seniority_normalized": null, "education_normalized": null, "limit": 3}
Example with name: {"role": null, "location": null, "experience": null, "skills": [], "education": null, "company_type": null, "name": "John Smith", "university": null, "location_normalized": null, "seniority_normalized": null, "education_normalized": null, "limit": null}
Example with abbreviation: {"role": "Data Analyst", "location": "NYC", "experience": "Mid-level", "skills": ["SQL", "Python"], "education": "MS", "company_type": null, "name": null, "university": null, "location_normalized": "New York", "seniority_normalized": "mid", "education_normalized": "masters", "limit": null}"""

        user_prompt = f"Parse this recruiter query: {raw_query}"
        
        log.info("Calling LLM for query parsing...")
        
        try:
            content = await asyncio.wait_for(
                invoke_llm(
                    prompt=user_prompt,
                    task_type=TaskType.CLASSIFICATION,
                    preferred_model=self.model,
                    agent_name="recruitment_query_parser",
                    response_mime_type="application/json",
                    system_instruction=system_prompt,
                ),
                timeout=30.0,
            )
            content = content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            # Parse JSON
            parsed = json.loads(content)
            
            # Validate normalized seniority/education against allowed values
            seniority_norm = parsed.get("seniority_normalized")
            if seniority_norm and seniority_norm not in ("junior", "mid", "senior", "lead"):
                seniority_norm = None
            education_norm = parsed.get("education_normalized")
            if education_norm and education_norm not in ("high_school", "bachelors", "masters", "phd"):
                education_norm = None

            # Validate limit: integer in [1, RECRUITMENT_MAX_TOP_K], ignore otherwise
            parsed_limit = None
            try:
                from settings import settings
                raw_limit = parsed.get("limit")
                if raw_limit is not None:
                    n = int(raw_limit) if not isinstance(raw_limit, int) else raw_limit
                    max_k = getattr(settings, "RECRUITMENT_MAX_TOP_K", 100)
                    if 1 <= n <= max_k:
                        parsed_limit = n
            except (TypeError, ValueError):
                pass

            # Create RecruitmentQuery
            query = RecruitmentQuery(
                raw_query=raw_query,
                parsed_role=parsed.get("role"),
                parsed_location=parsed.get("location"),
                parsed_experience=parsed.get("experience"),
                parsed_skills=parsed.get("skills", []),
                parsed_education=parsed.get("education"),
                parsed_company_type=parsed.get("company_type"),
                parsed_name=parsed.get("name"),
                parsed_university=parsed.get("university"),
                normalized_location=parsed.get("location_normalized") or None,
                normalized_seniority=seniority_norm,
                normalized_education=education_norm,
                parsed_limit=parsed_limit,
                query_id=str(uuid.uuid4())
            )
            
            log.info(f"LLM parsing complete: {query.to_dict()}")
            return query
            
        except asyncio.TimeoutError:
            log.error("LLM parsing timed out")
            raise
        except json.JSONDecodeError as e:
            log.error(f"Failed to parse LLM response as JSON: {e}")
            raise
        except Exception as e:
            log.error(f"LLM parsing error: {e}")
            raise
    
    def _parse_with_rules(self, raw_query: str) -> RecruitmentQuery:
        """
        Fallback rule-based parsing.
        Uses simple pattern matching and keyword detection.
        """
        log.info("Using rule-based parsing (fallback)")
        
        query_lower = raw_query.lower()
        
        # Extract role (common job titles)
        role = None
        role_keywords = [
            "developer", "engineer", "designer", "manager", "analyst",
            "architect", "consultant", "specialist", "lead", "director",
            "python", "java", "javascript", "react", "frontend", "backend",
            "fullstack", "devops", "data scientist", "ml engineer"
        ]
        for keyword in role_keywords:
            if keyword in query_lower:
                # Capitalize first letter of each word
                role = keyword.title()
                if "developer" not in role and "engineer" not in role:
                    role += " Developer"
                break
        
        # Extract location (common cities/countries)
        location = None
        location_keywords = [
            "dubai", "abu dhabi", "uae", "saudi", "riyadh", "jeddah",
            "new york", "london", "singapore", "bangalore", "mumbai",
            "remote", "hybrid"
        ]
        for keyword in location_keywords:
            if keyword in query_lower:
                location = keyword.title()
                break
        
        # Extract experience level
        experience = None
        if any(word in query_lower for word in ["senior", "sr", "lead", "principal"]):
            experience = "Senior"
        elif any(word in query_lower for word in ["junior", "jr", "entry"]):
            experience = "Junior"
        elif any(word in query_lower for word in ["mid", "intermediate"]):
            experience = "Mid-level"
        
        # Extract skills (common technologies)
        skills = []
        skill_keywords = [
            "python", "java", "javascript", "typescript", "react", "angular",
            "vue", "node", "django", "flask", "spring", "aws", "azure", "gcp",
            "docker", "kubernetes", "sql", "nosql", "mongodb", "postgresql"
        ]
        for keyword in skill_keywords:
            if keyword in query_lower:
                skills.append(keyword.title())
        
        # Extract education
        education = None
        if any(word in query_lower for word in ["phd", "doctorate"]):
            education = "PhD"
        elif any(word in query_lower for word in ["master", "msc", "ms"]):
            education = "Master's"
        elif any(word in query_lower for word in ["bachelor", "bsc", "bs"]):
            education = "Bachelor's"
        
        # Extract name (simple heuristic: if query looks like a name and doesn't contain role keywords)
        # Names typically have 2-3 capitalized words and don't contain job-related keywords
        name = None
        words = raw_query.split()
        if len(words) >= 2 and len(words) <= 4:
            # Check if words are capitalized (likely a name)
            if all(word[0].isupper() if word else False for word in words):
                # Check if it doesn't contain role keywords (likely a name, not a role)
                if not any(keyword in query_lower for keyword in role_keywords + ["developer", "engineer", "manager"]):
                    name = " ".join(words)
        
        # Extract university (query mentions university/college/institute or "from X")
        university = None
        if "university" in query_lower or "college" in query_lower or "institute" in query_lower or " from " in query_lower:
            # Try "candidates from X" or "X University" pattern
            import re
            from_match = re.search(r"\bfrom\s+([^.?,]+?)(?:\s+university|\s+college|\s+institute)?$", raw_query, re.I)
            if from_match:
                university = from_match.group(1).strip()
            else:
                # Take phrase containing university/college/institute
                for part in raw_query.split(","):
                    if "university" in part.lower() or "college" in part.lower() or "institute" in part.lower():
                        university = part.strip()
                        break
            if university and len(university) > 50:
                university = university[:50].strip()

        # Extract limit from "top N", "give me N", "first N", or standalone number (e.g. "10 react developers")
        parsed_limit = None
        try:
            import re
            from settings import settings
            max_k = getattr(settings, "RECRUITMENT_MAX_TOP_K", 100)
            # "top 3", "first 5", "give me 10", "20 candidates"
            for pattern in [r"\btop\s+(\d+)\b", r"\bfirst\s+(\d+)\b", r"\bgive\s+me\s+(\d+)\b", r"\b(\d+)\s+candidates?\b", r"\b(\d+)\s+developers?\b", r"\b(\d+)\s+engineers?\b"]:
                m = re.search(pattern, query_lower, re.I)
                if m:
                    n = int(m.group(1))
                    if 1 <= n <= max_k:
                        parsed_limit = n
                        break
        except Exception:
            pass

        query = RecruitmentQuery(
            raw_query=raw_query,
            parsed_role=role,
            parsed_location=location,
            parsed_experience=experience,
            parsed_skills=skills,
            parsed_education=education,
            parsed_name=name,
            parsed_university=university,
            parsed_limit=parsed_limit,
            query_id=str(uuid.uuid4())
        )
        
        log.info(f"Rule-based parsing complete: role={role}, location={location}, experience={experience}, parsed_limit={parsed_limit}")
        return query
    
    def parse_query_sync(self, raw_query: str) -> RecruitmentQuery:
        """
        Synchronous version of parse_query (uses fallback only).
        
        Args:
            raw_query: Natural language query from recruiter
            
        Returns:
            RecruitmentQuery with extracted entities
        """
        return self._parse_with_rules(raw_query)
