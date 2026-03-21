"""
Skill extraction utilities - shared across ranker and job_matcher.
Extracted from skill_matcher.py to remove dependency on semantic matching logic.
"""
import re
import logging
from typing import List, Set, Dict, Any

log = logging.getLogger(__name__)


def normalize_skill(skill: str) -> str:
    """
    Normalize a skill string for consistent matching.
    
    Enhanced normalization steps:
    1. Convert to lowercase
    2. Strip leading/trailing whitespace
    3. Replace multiple spaces with single space
    4. Handle common tech skill variations (JS, .js, etc.)
    5. Remove common prefixes/suffixes that don't affect matching
    6. Handle parenthetical expansions (e.g., "Vector databases (Pinecone, Weaviate)")
    7. Handle compound skills (e.g., "React/NextJS")
    
    Args:
        skill: Raw skill string
        
    Returns:
        Normalized skill string
    """
    if not skill or not isinstance(skill, str):
        return ""
    
    # Convert to lowercase and strip
    normalized = skill.lower().strip()
    
    # Remove parenthetical content for base normalization (e.g., "Vector databases (Pinecone, Weaviate)" -> "vector databases")
    # But we'll handle parenthetical content separately in variant generation
    normalized = re.sub(r'\s*\([^)]*\)', '', normalized)
    
    # Remove common prefixes that don't affect skill matching
    prefixes_to_remove = [
        r'^proficient in\s+',
        r'^experience with\s+',
        r'^knowledge of\s+',
        r'^expertise in\s+',
        r'^skilled in\s+',
        r'^familiar with\s+',
        r'^systems?\s+',  # Remove "system" or "systems" prefix
        r'^pipelines?\s+',  # Remove "pipeline" or "pipelines" prefix
        r'^databases?\s+',  # Remove "database" or "databases" prefix
        r'^frameworks?\s+',  # Remove "framework" or "frameworks" prefix
        r'^libraries?\s+',  # Remove "library" or "libraries" prefix
        r'^tools?\s+',  # Remove "tool" or "tools" prefix
    ]
    for prefix in prefixes_to_remove:
        normalized = re.sub(prefix, '', normalized, flags=re.IGNORECASE)
    
    # Normalize common tech skill patterns
    # Handle .js, .jsx, .ts, .tsx, .py, etc. - normalize to just the extension
    normalized = re.sub(r'\.(js|jsx|ts|tsx|py|java|cpp|c|go|rs|rb|php|swift|kt)$', r'.\1', normalized)
    
    # Handle common variations
    # "React.js" -> "reactjs", "ReactJS" -> "reactjs", "React JS" -> "reactjs"
    normalized = re.sub(r'\s*\.\s*js\b', 'js', normalized)
    normalized = re.sub(r'\s+js\b', 'js', normalized)
    
    # Handle "C++", "C#", etc.
    normalized = re.sub(r'c\s*\+\+', 'c++', normalized)
    normalized = re.sub(r'c\s*#', 'c#', normalized)
    
    # Handle common abbreviations and expansions
    abbreviations = {
        r'\bnext\.?js\b': 'nextjs',
        r'\bnext js\b': 'nextjs',
        r'\bnode\.?js\b': 'nodejs',
        r'\bnode js\b': 'nodejs',
        r'\breact\.?js\b': 'reactjs',
        r'\breact js\b': 'reactjs',
    }
    for pattern, replacement in abbreviations.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    
    # Replace multiple spaces with single space
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Remove common noise characters but keep important ones
    # Keep: letters, numbers, spaces, +, #, ., -
    normalized = re.sub(r'[^\w\s+#.\-]', '', normalized)
    
    # Remove trailing/leading dots, dashes, etc.
    normalized = normalized.strip('.-+ ')
    
    return normalized


def _extract_skills_with_parentheses(skill_text: str) -> List[str]:
    """
    Extract skills from text that may contain parenthetical expansions.
    Example: "Vector databases (Pinecone, Weaviate)" -> ["Vector databases", "Pinecone", "Weaviate"]
    
    Args:
        skill_text: Original skill text
        
    Returns:
        List of extracted skill components
    """
    if not skill_text:
        return []
    
    skills = []
    
    # Extract parenthetical content
    paren_pattern = r'\(([^)]+)\)'
    paren_matches = re.findall(paren_pattern, skill_text)
    
    # Get base skill (without parentheses)
    base_skill = re.sub(paren_pattern, '', skill_text).strip()
    if base_skill:
        skills.append(base_skill)
    
    # Add parenthetical content as separate skills
    for match in paren_matches:
        # Split by common separators
        for item in re.split(r'[,;|&]', match):
            item = item.strip()
            if item:
                skills.append(item)
    
    return skills if skills else [skill_text]


def extract_primary_skills(resume_data: Dict[str, Any]) -> Set[str]:
    """
    Extract PRIMARY skills only from the skills section (not from experience/education/certifications).
    Used for preliminary matching.
    
    Args:
        resume_data: Structured resume data
        
    Returns:
        Set of normalized primary skills
    """
    log.debug(f"🚀 Extracting primary skills from resume with keys: {list(resume_data.keys())[:20]}")
    candidate_skills = set()
    
    # PRIORITY 1: Check main skills array
    skills_array = resume_data.get("skills") or resume_data.get("Skills")
    if isinstance(skills_array, list):
        log.debug(f"Found primary skills array with {len(skills_array)} entries")
        
        for skill_item in skills_array:
            if isinstance(skill_item, dict):
                skill_name = (
                    skill_item.get("SkillName") or 
                    skill_item.get("Name") or 
                    skill_item.get("name") or
                    skill_item.get("skillName") or
                    skill_item.get("skill")
                )
                
                if skill_name and isinstance(skill_name, str):
                    extracted_skills = _extract_skills_with_parentheses(skill_name)
                    for skill in extracted_skills:
                        normalized = normalize_skill(skill)
                        if normalized:
                            candidate_skills.add(normalized)
            
            elif isinstance(skill_item, str):
                extracted_skills = _extract_skills_with_parentheses(skill_item)
                for skill in extracted_skills:
                    normalized = normalize_skill(skill)
                    if normalized:
                        candidate_skills.add(normalized)

    # PRIORITY 2: Check personalInformation.skills    
    personal_info = resume_data.get("personalInformation", {})
    if isinstance(personal_info, dict):
        pi_skills = personal_info.get("skills", [])
        
        if isinstance(pi_skills, list):
            for skill in pi_skills:
                if isinstance(skill, str):
                    extracted_skills = _extract_skills_with_parentheses(skill)
                    for s in extracted_skills:
                        normalized = normalize_skill(s)
                        if normalized:
                            candidate_skills.add(normalized)
    
    # PRIORITY 3: Check technical_skills field
    tech_skills = resume_data.get("technical_skills") or resume_data.get("technicalSkills")
    if isinstance(tech_skills, list):
        for skill in tech_skills:
            if isinstance(skill, str):
                extracted_skills = _extract_skills_with_parentheses(skill)
                for s in extracted_skills:
                    normalized = normalize_skill(s)
                    if normalized:
                        candidate_skills.add(normalized)
            elif isinstance(skill, dict):
                skill_name = skill.get("name") or skill.get("SkillName") or skill.get("skillName")
                if skill_name and isinstance(skill_name, str):
                    extracted_skills = _extract_skills_with_parentheses(skill_name)
                    for s in extracted_skills:
                        normalized = normalize_skill(s)
                        if normalized:
                            candidate_skills.add(normalized)
    
    # NOTE: Education is NOT extracted as skills - it should be matched separately
    # against JD's required education field, not as part of skills matching.
    
    log.debug(f"✅ Extracted {len(candidate_skills)} primary skills")
    return candidate_skills
