"""
LLM-Based Resume Section Detector

This module uses LLM intelligence to detect and segment resume sections,
providing superior flexibility for handling diverse resume formats.
"""

import json
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field
from models.llm_invoker import invoke_llm, invoke_structured_llm
from core.model_registry import TaskType
from core.security import sanitize_text_for_llm, PII_PATTERNS, INJECTION_FILTERS
from core.config import get_agent_config

log = logging.getLogger(__name__)

@dataclass
class ResumeSection:
    """Represents a detected section of a resume."""
    name: str
    content: str
    start_line: int
    end_line: int
    confidence: float
    entities: Dict[str, List[str]] = None

class ResumeSegmentation(BaseModel):
    """Structured response for resume segmentation using semantic text chunks."""
    personal_info: Optional[str] = Field(description="Raw text of personal info section (name, contact, links)", default=None)
    experience: Optional[str] = Field(description="Raw text of work experience section", default=None)
    education: Optional[str] = Field(description="Raw text of education section", default=None)
    skills: Optional[str] = Field(description="Raw text of skills section", default=None)
    projects: Optional[str] = Field(description="Raw text of projects section", default=None)
    certifications: Optional[str] = Field(description="Raw text of certifications section", default=None)

class LLMResumeSectionDetector:
    """LLM-powered resume section detector with superior format flexibility."""
    
    def __init__(self):
        self.config = get_agent_config("resume_section_detector")
        self.max_prompt_chars = self.config.max_prompt_chars
        self.timeout_seconds = self.config.timeout_seconds
        
        # Agent section mapping
        self.agent_section_mapping = {
            'personal_info_parser': ['personal_info', 'contact', 'profile'],
            'experience_parser': ['experience', 'work_experience', 'employment', 'projects', 'certifications'],
            'education_parser': ['education', 'academic', 'qualifications', 'degrees', 'university', 'college', 'school'],
            'skills_parser': ['skills', 'technical_skills', 'competencies', 'additional_skills']
        }
    
    def _generate_segmentation_prompt(self, resume_text: str) -> str:
        """Generate improved prompt that searches entire document for each section."""
        return f"""Your task is to act as a comprehensive resume analyzer. Read the ENTIRE resume text and find ALL instances of information for each section type, regardless of where they appear in the document.

CRITICAL INSTRUCTIONS:
- Search the ENTIRE document for each section type
- Information for a section might appear in multiple places or unexpected locations
- Extract and combine ALL relevant content from anywhere in the document
- Include content even if it's embedded within other sections
- Look for implicit information (e.g., skills mentioned in job descriptions)

SECTION EXTRACTION GUIDELINES:

1. personal_info: 
   - Name, contact details, email, phone numbers
   - LinkedIn, GitHub, portfolio URLs
   - Location, address
   - Professional summary/objective (if present)

2. experience:
   - ALL work experience, internships, employment history
   - Job titles, company names, dates
   - Responsibilities, achievements, technologies used
   - Include experience mentioned anywhere in the document

3. education:
   - ALL educational background, degrees, diplomas
   - University/college names, locations, graduation years
   - Academic achievements, GPA (if mentioned)
   - Relevant coursework, academic projects

4. skills:
   - ALL technical skills, programming languages, frameworks
   - Tools, technologies, methodologies
   - Soft skills, competencies
   - Include skills mentioned in job descriptions or project descriptions

5. projects:
   - ALL projects, portfolio items, notable work
   - Personal projects, academic projects, professional projects
   - Project descriptions, technologies used, achievements

6. certifications:
   - ALL certifications, licenses, awards
   - Professional certifications, online courses
   - Achievements, honors, specializations

OUTPUT FORMAT:
Return a single JSON object with the section names as keys and the extracted content as values.
If a section is not found, use null as the value.
Combine all relevant information for each section into a single comprehensive text block.

Resume Text:
---
{resume_text}
---"""

    async def detect_sections(self, resume_text: str) -> Dict[str, ResumeSection]:
        """Detect and extract sections using comprehensive multi-pass analysis."""
        log.info(f"Starting comprehensive LLM-based section detection for resume of {len(resume_text)} characters")
        
        # Sanitize input - preserve PII for personal info detection
        sanitized_text = sanitize_text_for_llm(resume_text, redact_pii_flag=False, filter_injection_flag=True)
        
        # Truncate if too long
        if len(sanitized_text) > self.max_prompt_chars:
            log.warning(f"Resume text truncated from {len(sanitized_text)} to {self.max_prompt_chars} characters")
            sanitized_text = sanitized_text[:self.max_prompt_chars]
        
        try:
            # First pass: Comprehensive extraction
            sections = await self._perform_comprehensive_extraction(sanitized_text, resume_text)
            
            # Second pass: Cross-section analysis for missed information
            sections = await self._perform_cross_section_analysis(sections, sanitized_text, resume_text)
            
            # Third pass: Content aggregation and validation
            sections = self._aggregate_and_validate_content(sections)
            
            log.info(f"✅ Comprehensive section detection completed. Found {len(sections)} sections")
            return sections
            
        except Exception as e:
            log.error(f"❌ Comprehensive LLM section detection failed: {e}")
            # Use enhanced fallback detection
            return self._fallback_section_detection(resume_text)
    
    async def _perform_comprehensive_extraction(self, sanitized_text: str, resume_text: str) -> Dict[str, ResumeSection]:
        """Perform comprehensive extraction using LLM."""
        log.info("🔍 First pass: Performing comprehensive extraction...")
        
        # Generate prompt
        prompt = self._generate_segmentation_prompt(sanitized_text)
        
        # Call LLM with structured output
        log.info("🤖 Calling LLM for comprehensive section extraction...")
        llm_result = await invoke_structured_llm(
            prompt,
            ResumeSegmentation,
            task_type=TaskType.RESUME_ANALYSIS,
            agent_name="llm_resume_section_detector",
            temperature=0.0,
            max_output_tokens=1500,
            timeout=float(self.timeout_seconds),
            raise_on_fallback=False,
        )
        
        log.info(f"✅ LLM completed comprehensive extraction")
        
        # Convert LLM result to ResumeSection objects
        sections = {}
        lines = resume_text.split('\n')
        
        # Map section names to their content
        section_content_map = {
            'personal_info': llm_result.personal_info,
            'experience': llm_result.experience,
            'education': llm_result.education,
            'skills': llm_result.skills,
            'projects': llm_result.projects,
            'certifications': llm_result.certifications
        }
        
        for section_name, content in section_content_map.items():
            if content and content.strip():
                # Calculate line numbers deterministically
                start_line, end_line = self._calculate_line_numbers(content, lines)
                
                # Create ResumeSection object
                section = ResumeSection(
                    name=section_name,
                    content=content.strip(),
                    start_line=start_line,
                    end_line=end_line,
                    confidence=0.9,  # High confidence for comprehensive extraction
                    entities={}
                )
                
                # Add debug logging for personal_info
                if section_name == 'personal_info':
                    log.info(f"🔍 DEBUG: Found personal_info content: '{content[:100]}...'")
                    log.info(f"🔍 DEBUG: Validation result: {self._validate_section_content(section_name, content)}")
                
                # Validate content relevance
                if self._validate_section_content(section_name, content):
                    sections[section_name] = section
                    log.info(f"📄 Section '{section_name}': lines {start_line}-{end_line}, {len(content)} chars")
                else:
                    log.warning(f"⚠️ Section '{section_name}' failed content validation, skipping")
                    if section_name == 'personal_info':
                        log.warning(f"🔍 DEBUG: Personal info validation failed for content: '{content[:200]}...'")
        
        return sections
    
    async def _perform_cross_section_analysis(self, sections: Dict[str, ResumeSection], sanitized_text: str, resume_text: str) -> Dict[str, ResumeSection]:
        """Perform cross-section analysis to find missed information."""
        log.info("🔍 Second pass: Performing cross-section analysis...")
        
        # Look for skills mentioned in experience descriptions
        if 'experience' in sections and 'skills' in sections:
            experience_content = sections['experience'].content.lower()
            skills_content = sections['skills'].content.lower()
            
            # Extract additional skills from experience
            additional_skills = self._extract_skills_from_text(experience_content)
            existing_skills = self._extract_skills_from_text(skills_content)
            
            # Find new skills not already captured
            new_skills = []
            for skill in additional_skills:
                if not any(existing_skill in skill or skill in existing_skill for existing_skill in existing_skills):
                    new_skills.append(skill)
            
            if new_skills:
                log.info(f"🔍 Found {len(new_skills)} additional skills in experience section")
                # Update skills section with additional skills
                sections['skills'].content += f"\n\nAdditional skills from experience:\n" + ", ".join(new_skills)
        
        return sections
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract technical skills from text using pattern matching."""
        skill_patterns = [
            r'\b(?:react|javascript|python|java|c\+\+|sql|aws|azure|docker|kubernetes)\b',
            r'\b(?:node\.?js|angular|vue|typescript|html|css|bootstrap)\b',
            r'\b(?:machine learning|artificial intelligence|data science|analytics)\b',
            r'\b(?:agile|scrum|devops|ci\/cd|git|github)\b'
        ]
        
        skills = []
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.extend(matches)
        
        return list(set(skills))  # Remove duplicates
    
    def _aggregate_and_validate_content(self, sections: Dict[str, ResumeSection]) -> Dict[str, ResumeSection]:
        """Aggregate and validate content from all sections."""
        log.info("🔍 Third pass: Aggregating and validating content...")
        
        # Remove duplicate content and validate sections
        validated_sections = {}
        
        for section_name, section in sections.items():
            if section.content and len(section.content.strip()) > 10:  # Minimum content length
                # Clean up content
                cleaned_content = self._clean_section_content(section.content)
                
                if cleaned_content:
                    section.content = cleaned_content
                    validated_sections[section_name] = section
                    log.info(f"✅ Validated section '{section_name}': {len(cleaned_content)} chars")
                else:
                    log.warning(f"⚠️ Section '{section_name}' content too short after cleaning")
            else:
                log.warning(f"⚠️ Section '{section_name}' has insufficient content")
        
        return validated_sections
    
    def _clean_section_content(self, content: str) -> str:
        """Clean and normalize section content."""
        if not content:
            return ""
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', content.strip())
        
        # Remove duplicate lines
        lines = cleaned.split('\n')
        unique_lines = []
        seen = set()
        
        for line in lines:
            line_clean = line.strip()
            if line_clean and line_clean not in seen:
                unique_lines.append(line_clean)
                seen.add(line_clean)
        
        return '\n'.join(unique_lines)
    
    def _calculate_line_numbers(self, content: str, lines: List[str]) -> Tuple[int, int]:
        if not content or not content.strip():
            return 0, 0
        
        # Clean the content for matching
        content_clean = content.strip()
        
        # Find the best match in the original text
        best_match_start = -1
        best_match_length = 0
        
        # Try to find the content as a substring in the original text
        original_text = '\n'.join(lines)
        
        # Look for exact match first
        if content_clean in original_text:
            start_pos = original_text.find(content_clean)
            end_pos = start_pos + len(content_clean)
            
            # Convert character positions to line numbers
            start_line = original_text[:start_pos].count('\n')
            end_line = original_text[:end_pos].count('\n')
            
            return start_line, end_line
        
        # If exact match fails, try to find by matching individual lines
        content_lines = content_clean.split('\n')
        if len(content_lines) >= 2:
            # Find the first line of content
            first_line = content_lines[0].strip()
            last_line = content_lines[-1].strip()
            
            start_line = -1
            end_line = -1
            
            # Find start line
            for i, line in enumerate(lines):
                if first_line in line.strip():
                    start_line = i
                    break
            
            # Find end line
            for i in range(len(lines) - 1, -1, -1):
                if last_line in lines[i].strip():
                    end_line = i
                    break
            
            if start_line != -1 and end_line != -1:
                return start_line, end_line
        
        # Fallback: estimate based on content length
        estimated_lines = max(1, len(content_clean.split('\n')))
        return 0, estimated_lines - 1
    
    def _validate_section_content(self, section_name: str, content: str) -> bool:
        """Validate that section content is relevant to the section type."""
        content_lower = content.lower()
        
        if section_name == 'personal_info':
            # More flexible validation - accept if it has name-like content OR contact info
            contact_patterns = [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone (US format)
                r'\+\d{1,3}\s*\d{3,}',  # International phone
                r'\b\d{10}\b',  # 10-digit phone
                r'linkedin\.com',  # LinkedIn
                r'github\.com',  # GitHub
                r'www\.',  # Website
                r'\.com|\.org|\.net',  # Generic URLs
            ]
            
            # Check for contact info
            has_contact = any(re.search(pattern, content_lower) for pattern in contact_patterns)
            
            # Check for name-like content (2+ words, proper case)
            name_patterns = [
                r'^[A-Z][a-z]+ [A-Z][a-z]+',  # First Last
                r'^[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+',  # First M. Last
                r'[A-Z][a-z]+ [A-Z][a-z]+',  # Name anywhere in content
            ]
            has_name = any(re.search(pattern, content) for pattern in name_patterns)
            
            # Accept if it has contact info OR name-like content
            return has_contact or has_name
        
        elif section_name == 'experience':
            # Must contain job-related content
            experience_patterns = [
                r'\b(?:software engineer|developer|manager|analyst|consultant|specialist|lead|senior|junior)\b',
                r'\b(?:amazon|google|microsoft|apple|facebook|meta|walmart|makeMyTrip)\b',
                r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b',
                r'\b\d{4}\s*[-–]\s*(?:present|\d{4})\b',
                r'\b(?:years?|months?)\b',
            ]
            return any(re.search(pattern, content_lower) for pattern in experience_patterns)
        
        elif section_name == 'education':
            # Must contain education-related content
            education_patterns = [
                r'\b(?:bachelor|master|phd|doctorate|degree|diploma|certificate)\b',
                r'\b(?:university|college|school|institute|academy)\b',
                r'\b(?:psg|mit|stanford|harvard|berkeley|caltech)\b',
                r'\b(?:computer science|engineering|information technology|business|management)\b',
                r'\b(?:2016|2017|2018|2019|2020|2021|2022|2023|2024|2025)\b',
            ]
            # Additional check: must NOT contain work-related terms
            work_patterns = [
                r'\b(?:software engineer|developer|intern|trainee|employee|job|work|company)\b',
                r'\b(?:amazon|google|microsoft|walmart|makeMyTrip)\b',
            ]
            has_education_content = any(re.search(pattern, content_lower) for pattern in education_patterns)
            has_work_content = any(re.search(pattern, content_lower) for pattern in work_patterns)
            
            # If it has work content but no education content, it's not a valid education section
            if has_work_content and not has_education_content:
                return False
            
            return has_education_content
        
        elif section_name == 'skills':
            # Must contain technical skills
            skills_patterns = [
                r'\b(?:react|javascript|python|java|c\+\+|sql|aws|azure|docker|kubernetes)\b',
                r'\b(?:node\.?js|angular|vue|typescript|html|css|bootstrap)\b',
                r'\b(?:machine learning|artificial intelligence|data science|analytics)\b',
                r'\b(?:agile|scrum|devops|ci\/cd|git|github)\b',
            ]
            return any(re.search(pattern, content_lower) for pattern in skills_patterns)
        
        # For other sections, accept any content
        return True
    
    def _fallback_section_detection(self, resume_text: str) -> Dict[str, ResumeSection]:
        """Enhanced fallback section detection using comprehensive document scanning."""
        log.info("🔄 Using enhanced comprehensive fallback section detection")
        
        lines = resume_text.split('\n')
        sections = {}
        
        # Comprehensive scanning approach - search entire document for each section type
        log.info("🔍 Scanning entire document for all section types...")
        
        # 1. Personal Info Detection - search entire document
        personal_info_content = self._scan_entire_document_for_personal_info(lines)
        log.info(f"🔍 DEBUG: Fallback personal_info scan result: '{personal_info_content[:100]}...'")
        if personal_info_content:
            sections['personal_info'] = ResumeSection(
                name='personal_info',
                content=personal_info_content,
                start_line=0,
                end_line=len(personal_info_content.split('\n')) - 1,
                confidence=0.8
            )
            log.info(f"📋 Found personal_info section: {len(personal_info_content)} chars")
        else:
            log.warning("🔍 DEBUG: No personal_info found in fallback detection")
        
        # 2. Experience Detection - search entire document
        experience_content = self._scan_entire_document_for_experience(lines)
        if experience_content:
            sections['experience'] = ResumeSection(
                name='experience',
                content=experience_content,
                start_line=0,
                end_line=len(experience_content.split('\n')) - 1,
                confidence=0.8
            )
            log.info(f"📋 Found experience section: {len(experience_content)} chars")
        
        # 3. Education Detection - search entire document
        education_content = self._scan_entire_document_for_education(lines)
        if education_content:
            sections['education'] = ResumeSection(
                name='education',
                content=education_content,
                start_line=0,
                end_line=len(education_content.split('\n')) - 1,
                confidence=0.8
            )
            log.info(f"📋 Found education section: {len(education_content)} chars")
        
        # 4. Skills Detection - search entire document
        skills_content = self._scan_entire_document_for_skills(lines)
        if skills_content:
            sections['skills'] = ResumeSection(
                name='skills',
                content=skills_content,
                start_line=0,
                end_line=len(skills_content.split('\n')) - 1,
                confidence=0.8
            )
            log.info(f"📋 Found skills section: {len(skills_content)} chars")
        
        # 5. Projects Detection - search entire document
        projects_content = self._scan_entire_document_for_projects(lines)
        if projects_content:
            sections['projects'] = ResumeSection(
                name='projects',
                content=projects_content,
                start_line=0,
                end_line=len(projects_content.split('\n')) - 1,
                confidence=0.8
            )
            log.info(f"📋 Found projects section: {len(projects_content)} chars")
        
        # 6. Certifications Detection - search entire document
        certifications_content = self._scan_entire_document_for_certifications(lines)
        if certifications_content:
            sections['certifications'] = ResumeSection(
                name='certifications',
                content=certifications_content,
                start_line=0,
                end_line=len(certifications_content.split('\n')) - 1,
                confidence=0.8
            )
            log.info(f"📋 Found certifications section: {len(certifications_content)} chars")
        
        log.info(f"✅ Enhanced fallback detection completed. Found {len(sections)} sections")
        return sections
    
    def _scan_entire_document_for_personal_info(self, lines: List[str]) -> str:
        """Scan entire document for personal information."""
        personal_info_lines = []
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Enhanced contact patterns
            if (re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', line) or
                re.search(r'\+\d{1,3}\s*\d{3,}', line) or  # International
                re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', line) or  # US format
                re.search(r'\b\d{10}\b', line) or  # 10-digit
                re.search(r'linkedin\.com', line_lower) or
                re.search(r'github\.com', line_lower) or
                re.search(r'www\.', line_lower) or
                re.search(r'\.com|\.org|\.net', line_lower)):
                
                # Look backwards for the name (extended range)
                for j in range(max(0, i-5), i):  # Increased from 3 to 5 lines
                    if (lines[j].strip() and 
                        len(lines[j].strip().split()) >= 2 and  # At least 2 words
                        not re.search(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b', lines[j].lower()) and
                        not re.search(r'\b(?:education|experience|projects|certifications|skills|work|employment|career)\b', lines[j].lower()) and
                        not re.search(r'\b(?:bachelor|master|phd|degree|university|college)\b', lines[j].lower())):
                        personal_info_lines.append(lines[j].strip())
                        break
                
                personal_info_lines.append(line.strip())
        
        # If no contact info found, look for name patterns in first few lines
        if not personal_info_lines:
            for i, line in enumerate(lines[:10]):  # Check first 10 lines
                line_stripped = line.strip()
                if (line_stripped and 
                    len(line_stripped.split()) >= 2 and  # At least 2 words
                    re.search(r'^[A-Z][a-z]+ [A-Z][a-z]+', line_stripped) and  # Name pattern
                    not re.search(r'\b(?:education|experience|projects|certifications|skills|work|employment|career)\b', line_stripped.lower()) and
                    not re.search(r'\b(?:bachelor|master|phd|degree|university|college)\b', line_stripped.lower())):
                    personal_info_lines.append(line_stripped)
                    break
        
        return '\n'.join(personal_info_lines) if personal_info_lines else ""
    
    def _scan_entire_document_for_experience(self, lines: List[str]) -> str:
        """Scan entire document for work experience."""
        experience_lines = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Look for experience patterns
            if (re.search(r'\b(?:software engineer|developer|manager|analyst|consultant|specialist|lead|senior|junior|intern|trainee)\b', line_lower) or
                re.search(r'\b(?:amazon|google|microsoft|apple|facebook|meta|walmart|makeMyTrip)\b', line_lower) or
                re.search(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b', line_lower) or
                re.search(r'\b\d{4}\s*[-–]\s*(?:present|\d{4})\b', line_lower) or
                re.search(r'\b(?:years?|months?)\b', line_lower)):
                experience_lines.append(line.strip())
        
        return '\n'.join(experience_lines) if experience_lines else ""
    
    def _scan_entire_document_for_education(self, lines: List[str]) -> str:
        """Scan entire document for education information."""
        education_lines = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Look for education patterns
            if (re.search(r'\b(?:bachelor|master|phd|doctorate|degree|diploma|certificate)\b', line_lower) or
                re.search(r'\b(?:university|college|school|institute|academy)\b', line_lower) or
                re.search(r'\b(?:psg|mit|stanford|harvard|berkeley|caltech)\b', line_lower) or
                re.search(r'\b(?:computer science|engineering|information technology|business|management)\b', line_lower) or
                re.search(r'\b(?:2016|2017|2018|2019|2020|2021|2022|2023|2024|2025)\b', line_lower)):
                
                # Additional check: must NOT contain work-related terms
                work_patterns = [
                    r'\b(?:software engineer|developer|intern|trainee|employee|job|work|company)\b',
                    r'\b(?:amazon|google|microsoft|walmart|makeMyTrip)\b',
                ]
                has_work_content = any(re.search(pattern, line_lower) for pattern in work_patterns)
                
                if not has_work_content:
                    education_lines.append(line.strip())
        
        return '\n'.join(education_lines) if education_lines else ""
    
    def _scan_entire_document_for_skills(self, lines: List[str]) -> str:
        """Scan entire document for skills."""
        skills_lines = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Look for skills patterns
            if (re.search(r'\b(?:react|javascript|python|java|c\+\+|sql|aws|azure|docker|kubernetes)\b', line_lower) or
                re.search(r'\b(?:node\.?js|angular|vue|typescript|html|css|bootstrap)\b', line_lower) or
                re.search(r'\b(?:machine learning|artificial intelligence|data science|analytics)\b', line_lower) or
                re.search(r'\b(?:agile|scrum|devops|ci\/cd|git|github)\b', line_lower)):
                skills_lines.append(line.strip())
        
        return '\n'.join(skills_lines) if skills_lines else ""
    
    def _scan_entire_document_for_projects(self, lines: List[str]) -> str:
        """Scan entire document for projects."""
        projects_lines = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Look for project patterns
            if (re.search(r'\b(?:project|application|system|website|app|software|tool|platform)\b', line_lower) or
                re.search(r'\b(?:developed|built|created|designed|implemented|programmed)\b', line_lower)):
                projects_lines.append(line.strip())
        
        return '\n'.join(projects_lines) if projects_lines else ""
    
    def _scan_entire_document_for_certifications(self, lines: List[str]) -> str:
        """Scan entire document for certifications."""
        certifications_lines = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Look for certification patterns
            if (re.search(r'\b(?:certified|certification|license|award|honor|achievement|specialization)\b', line_lower) or
                re.search(r'\b(?:aws|google|microsoft|oracle|cisco|comptia|pmp|agile|scrum)\b', line_lower)):
                certifications_lines.append(line.strip())
        
        return '\n'.join(certifications_lines) if certifications_lines else ""
    
    def _detect_sections_by_content(self, lines: List[str], section_boundaries: Dict[str, int]) -> None:
        """Detect sections based on content patterns when headers are not found."""
        log.info("🔍 Performing content-based section detection...")
        
        # Education detection by content
        if 'education' not in section_boundaries:
            for i, line in enumerate(lines):
                line_lower = line.lower().strip()
                # Look for education content patterns
                if (re.search(r'\b(?:bachelor|master|phd|doctorate|degree|diploma|certificate|university|college|school|institute|academy)\b', line_lower) and
                    re.search(r'\b(?:computer science|engineering|information technology|business|management|mathematics|physics|chemistry|biology)\b', line_lower)):
                    # Look backwards for a potential header
                    for j in range(max(0, i-3), i):
                        if re.search(r'(?i)^(?:education|academic|qualifications)', lines[j].strip()):
                            section_boundaries['education'] = j
                            log.info(f"📋 Found education section at line {j} (content pattern)")
                            break
                    if 'education' not in section_boundaries:
                        section_boundaries['education'] = max(0, i-1)
                        log.info(f"📋 Found education section at line {i-1} (content pattern)")
                    break
        
        # Experience detection by content
        if 'experience' not in section_boundaries:
            for i, line in enumerate(lines):
                line_lower = line.lower().strip()
                # Look for experience content patterns
                if (re.search(r'\b(?:software engineer|developer|manager|analyst|consultant|specialist|lead|senior|junior|intern|trainee)\b', line_lower) and
                    re.search(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b', line_lower)):
                    # Look backwards for a potential header
                    for j in range(max(0, i-3), i):
                        if re.search(r'(?i)^(?:experience|work|employment|career)', lines[j].strip()):
                            section_boundaries['experience'] = j
                            log.info(f"📋 Found experience section at line {j} (content pattern)")
                            break
                    if 'experience' not in section_boundaries:
                        section_boundaries['experience'] = max(0, i-1)
                        log.info(f"📋 Found experience section at line {i-1} (content pattern)")
                    break
        
        # Skills detection by content
        if 'skills' not in section_boundaries:
            for i, line in enumerate(lines):
                line_lower = line.lower().strip()
                # Look for skills content patterns
                if (re.search(r'\b(?:react|javascript|python|java|c\+\+|sql|aws|azure|docker|kubernetes|html|css|bootstrap|node\.?js|angular|vue|typescript)\b', line_lower) and
                    len(line.split(',')) >= 3):  # Multiple skills separated by commas
                    # Look backwards for a potential header
                    for j in range(max(0, i-3), i):
                        if re.search(r'(?i)^(?:skills|technical|programming|tools|competencies|expertise|proficiencies|capabilities|strengths|specializations|interpersonal|communication|language)', lines[j].strip()):
                            section_boundaries['skills'] = j
                            log.info(f"📋 Found skills section at line {j} (content pattern)")
                            break
                    if 'skills' not in section_boundaries:
                        section_boundaries['skills'] = max(0, i-1)
                        log.info(f"📋 Found skills section at line {i-1} (content pattern)")
                    break
        
        # Projects detection by content
        if 'projects' not in section_boundaries:
            for i, line in enumerate(lines):
                line_lower = line.lower().strip()
                # Look for project content patterns
                if (re.search(r'\b(?:project|application|system|website|app|software|tool|platform)\b', line_lower) and
                    re.search(r'\b(?:developed|built|created|designed|implemented|programmed)\b', line_lower)):
                    # Look backwards for a potential header
                    for j in range(max(0, i-3), i):
                        if re.search(r'(?i)^(?:projects|portfolio|work)', lines[j].strip()):
                            section_boundaries['projects'] = j
                            log.info(f"📋 Found projects section at line {j} (content pattern)")
                            break
                    if 'projects' not in section_boundaries:
                        section_boundaries['projects'] = max(0, i-1)
                        log.info(f"📋 Found projects section at line {i-1} (content pattern)")
                    break
        
        # Certifications detection by content
        if 'certifications' not in section_boundaries:
            for i, line in enumerate(lines):
                line_lower = line.lower().strip()
                # Look for certification content patterns
                if (re.search(r'\b(?:certified|certification|license|award|honor|achievement|specialization)\b', line_lower) and
                    re.search(r'\b(?:aws|google|microsoft|oracle|cisco|comptia|pmp|agile|scrum)\b', line_lower)):
                    # Look backwards for a potential header
                    for j in range(max(0, i-3), i):
                        if re.search(r'(?i)^(?:certifications|awards|honors|achievements)', lines[j].strip()):
                            section_boundaries['certifications'] = j
                            log.info(f"📋 Found certifications section at line {j} (content pattern)")
                            break
                    if 'certifications' not in section_boundaries:
                        section_boundaries['certifications'] = max(0, i-1)
                        log.info(f"📋 Found certifications section at line {i-1} (content pattern)")
                    break
    
    def get_section_for_agent(self, sections: Dict[str, ResumeSection], agent_name: str) -> str:
        """Get the most relevant section content for a specific agent."""
        relevant_section_names = self.agent_section_mapping.get(agent_name, [])
        
        # Special handling for skills parser
        if agent_name == 'skills_parser':
            skills_sections = [name for name in relevant_section_names if name in sections]
            if not skills_sections:
                log.warning("No skills sections detected by LLM; using full text fallback for skills")
                # Return a placeholder that indicates full text should be used
                return "FULL_TEXT_FALLBACK"
        
        # Combine relevant sections
        combined_content = []
        for section_name in relevant_section_names:
            if section_name in sections:
                section = sections[section_name]
                combined_content.append(f"=== {section.name.upper()} SECTION ===")
                combined_content.append(f"Confidence: {section.confidence:.2f}")
                combined_content.append(section.content)
                combined_content.append("")  # Empty line separator
        
        return '\n'.join(combined_content) if combined_content else ""
    
    def get_segmentation_metadata(self, sections: Dict[str, ResumeSection]) -> Dict[str, Any]:
        """Get metadata about the segmentation process."""
        return {
            "total_sections": len(sections),
            "section_names": list(sections.keys()),
            "average_confidence": sum(s.confidence for s in sections.values()) / len(sections) if sections else 0.0,
            "detection_method": "llm_based",
            "format_flexibility": "high"
        }
