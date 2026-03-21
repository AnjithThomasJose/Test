"""
Resume Chunker - Semantic Section Splitting

This module splits resumes into semantic sections for better embedding quality.
Instead of arbitrary page splits, we chunk by resume sections (Work, Education, etc.).

Chunking Strategy:
- Section-based: Split by Work History, Education, Skills, Projects
- Context preservation: Each chunk includes section header
- Size limits: Respect ChromaDB's document size limits
"""

import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class ResumeChunk:
    """
    A semantic chunk of a resume.
    
    Attributes:
        section_type: Type of section (work, education, skills, projects, summary)
        content: Text content of the chunk
        metadata: Additional metadata about the chunk
        chunk_index: Index of this chunk (for ordering)
    """
    section_type: str
    content: str
    metadata: Dict[str, Any]
    chunk_index: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "section_type": self.section_type,
            "content": self.content,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index
        }


class ResumeChunker:
    """
    Semantic resume chunker.
    
    Splits resumes by sections rather than arbitrary character limits.
    This preserves semantic meaning and improves embedding quality.
    
    Usage:
        chunker = ResumeChunker()
        chunks = chunker.chunk_resume(structured_resume)
        
        for chunk in chunks:
            print(f"Section: {chunk.section_type}")
            print(f"Content: {chunk.content[:100]}...")
    """
    
    def __init__(self, max_chunk_size: int = 2000):
        """
        Initialize chunker.
        
        Args:
            max_chunk_size: Maximum characters per chunk
        """
        self.max_chunk_size = max_chunk_size
        log.info(f"ResumeChunker initialized with max_chunk_size={max_chunk_size}")
    
    def chunk_resume(self, structured_resume: Dict[str, Any]) -> List[ResumeChunk]:
        """
        Chunk resume into semantic sections.
        
        Args:
            structured_resume: Parsed resume data from groq_resume_parser
            
        Returns:
            List of ResumeChunk objects
        """
        chunks = []
        chunk_index = 0
        
        # 1. Professional Summary / Objective
        summary = self._extract_summary(structured_resume)
        if summary:
            chunks.append(ResumeChunk(
                section_type="summary",
                content=summary,
                metadata={"section": "Professional Summary"},
                chunk_index=chunk_index
            ))
            chunk_index += 1
        
        # 2. Work Experience (most important for matching)
        work_chunks = self._chunk_work_experience(structured_resume, chunk_index)
        chunks.extend(work_chunks)
        chunk_index += len(work_chunks)
        
        # 3. Education
        education_chunks = self._chunk_education(structured_resume, chunk_index)
        chunks.extend(education_chunks)
        chunk_index += len(education_chunks)
        
        # 4. Skills
        skills_chunk = self._chunk_skills(structured_resume, chunk_index)
        if skills_chunk:
            chunks.append(skills_chunk)
            chunk_index += 1
        
        # 5. Projects
        project_chunks = self._chunk_projects(structured_resume, chunk_index)
        chunks.extend(project_chunks)
        chunk_index += len(project_chunks)
        
        # 6. Certifications
        cert_chunk = self._chunk_certifications(structured_resume, chunk_index)
        if cert_chunk:
            chunks.append(cert_chunk)
            chunk_index += 1
        
        log.info(f"Resume chunked into {len(chunks)} sections")
        return chunks
    
    def _extract_summary(self, resume: Dict[str, Any]) -> Optional[str]:
        """Extract professional summary/objective"""
        summary = resume.get("professional_summary") or resume.get("Summary") or resume.get("Objective", "")
        
        if summary and len(summary.strip()) > 20:
            return f"Professional Summary:\n{summary.strip()}"
        return None
    
    def _chunk_work_experience(self, resume: Dict[str, Any], start_index: int) -> List[ResumeChunk]:
        """Chunk work experience section"""
        work_exp = resume.get("WorkExperience") or resume.get("work_experience") or resume.get("experience", [])
        
        if not isinstance(work_exp, list):
            return []
        
        chunks = []
        
        for i, exp in enumerate(work_exp):
            if not isinstance(exp, dict):
                continue
            
            # Extract experience details
            title = exp.get("job_title") or exp.get("title", "")
            company = exp.get("company", "")
            dates = exp.get("dates") or exp.get("duration", "")
            location = exp.get("location", "")
            responsibilities = exp.get("responsibilities", [])
            
            if not title:
                continue
            
            # Build content
            content_parts = [f"Work Experience #{i+1}:"]
            content_parts.append(f"Position: {title}")
            if company:
                content_parts.append(f"Company: {company}")
            if dates:
                content_parts.append(f"Duration: {dates}")
            if location:
                content_parts.append(f"Location: {location}")
            
            if responsibilities:
                content_parts.append("\nResponsibilities:")
                if isinstance(responsibilities, list):
                    for resp in responsibilities[:10]:  # Limit to 10
                        if isinstance(resp, str) and resp.strip():
                            content_parts.append(f"- {resp.strip()}")
                elif isinstance(responsibilities, str):
                    content_parts.append(responsibilities)
            
            content = "\n".join(content_parts)
            
            # Check size and split if needed
            if len(content) > self.max_chunk_size:
                content = content[:self.max_chunk_size] + "..."
            
            chunks.append(ResumeChunk(
                section_type="work",
                content=content,
                metadata={
                    "section": "Work Experience",
                    "position": title,
                    "company": company,
                    "experience_index": i
                },
                chunk_index=start_index + i
            ))
        
        return chunks
    
    def _chunk_education(self, resume: Dict[str, Any], start_index: int) -> List[ResumeChunk]:
        """Chunk education section"""
        education = resume.get("Education") or resume.get("education", [])
        
        if not isinstance(education, list):
            return []
        
        chunks = []
        
        for i, edu in enumerate(education):
            if not isinstance(edu, dict):
                continue
            
            degree = edu.get("degree", "")
            major = edu.get("major") or edu.get("field", "")
            university = edu.get("university") or edu.get("institution", "")
            years = edu.get("years") or edu.get("duration", "")
            gpa = edu.get("gpa", "")
            
            if not degree and not university:
                continue
            
            content_parts = [f"Education #{i+1}:"]
            if degree:
                content_parts.append(f"Degree: {degree}")
            if major:
                content_parts.append(f"Major: {major}")
            if university:
                content_parts.append(f"University: {university}")
            if years:
                content_parts.append(f"Years: {years}")
            if gpa:
                content_parts.append(f"GPA: {gpa}")
            
            content = "\n".join(content_parts)
            
            chunks.append(ResumeChunk(
                section_type="education",
                content=content,
                metadata={
                    "section": "Education",
                    "degree": degree,
                    "university": university,
                    "education_index": i
                },
                chunk_index=start_index + i
            ))
        
        return chunks
    
    def _chunk_skills(self, resume: Dict[str, Any], chunk_index: int) -> Optional[ResumeChunk]:
        """Chunk skills section"""
        skills_data = resume.get("Skills") or resume.get("skills", [])
        
        if not skills_data:
            return None
        
        skills_list = []
        
        if isinstance(skills_data, list):
            for skill in skills_data:
                if isinstance(skill, dict):
                    skill_name = skill.get("SkillName") or skill.get("Name", "")
                    proficiency = skill.get("Proficiency", "")
                    if skill_name:
                        if proficiency:
                            skills_list.append(f"{skill_name} ({proficiency})")
                        else:
                            skills_list.append(skill_name)
                elif isinstance(skill, str):
                    skills_list.append(skill)
        
        if not skills_list:
            return None
        
        content = "Skills:\n" + "\n".join([f"- {skill}" for skill in skills_list[:50]])  # Limit to 50
        
        return ResumeChunk(
            section_type="skills",
            content=content,
            metadata={"section": "Skills", "skill_count": len(skills_list)},
            chunk_index=chunk_index
        )
    
    def _chunk_projects(self, resume: Dict[str, Any], start_index: int) -> List[ResumeChunk]:
        """Chunk projects section"""
        projects = resume.get("Projects") or resume.get("projects", [])
        
        if not isinstance(projects, list):
            return []
        
        chunks = []
        
        for i, project in enumerate(projects):
            if not isinstance(project, dict):
                continue
            
            name = project.get("project_name") or project.get("name", "")
            description = project.get("description", "")
            technologies = project.get("technologies", [])
            duration = project.get("duration", "")
            
            if not name:
                continue
            
            content_parts = [f"Project #{i+1}: {name}"]
            if description:
                content_parts.append(f"Description: {description}")
            if technologies:
                if isinstance(technologies, list):
                    content_parts.append(f"Technologies: {', '.join(technologies)}")
                elif isinstance(technologies, str):
                    content_parts.append(f"Technologies: {technologies}")
            if duration:
                content_parts.append(f"Duration: {duration}")
            
            content = "\n".join(content_parts)
            
            if len(content) > self.max_chunk_size:
                content = content[:self.max_chunk_size] + "..."
            
            chunks.append(ResumeChunk(
                section_type="projects",
                content=content,
                metadata={
                    "section": "Projects",
                    "project_name": name,
                    "project_index": i
                },
                chunk_index=start_index + i
            ))
        
        return chunks
    
    def _chunk_certifications(self, resume: Dict[str, Any], chunk_index: int) -> Optional[ResumeChunk]:
        """Chunk certifications section"""
        certifications = resume.get("Certifications") or resume.get("certifications", [])
        
        if not isinstance(certifications, list) or not certifications:
            return None
        
        cert_list = []
        
        for cert in certifications:
            if isinstance(cert, dict):
                name = cert.get("certification_name") or cert.get("name", "")
                org = cert.get("issuing_organization") or cert.get("organization", "")
                year = cert.get("year", "")
                
                if name:
                    if org and year:
                        cert_list.append(f"{name} - {org} ({year})")
                    elif org:
                        cert_list.append(f"{name} - {org}")
                    else:
                        cert_list.append(name)
            elif isinstance(cert, str):
                cert_list.append(cert)
        
        if not cert_list:
            return None
        
        content = "Certifications:\n" + "\n".join([f"- {cert}" for cert in cert_list])
        
        return ResumeChunk(
            section_type="certifications",
            content=content,
            metadata={"section": "Certifications", "cert_count": len(cert_list)},
            chunk_index=chunk_index
        )
    
    def merge_chunks_if_small(self, chunks: List[ResumeChunk], min_size: int = 500) -> List[ResumeChunk]:
        """
        Merge small chunks to improve embedding quality.
        
        Args:
            chunks: List of chunks to potentially merge
            min_size: Minimum size for a chunk (characters)
            
        Returns:
            List of merged chunks
        """
        if not chunks:
            return []
        
        merged = []
        current_merge = None
        
        for chunk in chunks:
            if len(chunk.content) < min_size and current_merge is None:
                # Start a merge
                current_merge = chunk
            elif len(chunk.content) < min_size and current_merge is not None:
                # Continue merging
                current_merge.content += "\n\n" + chunk.content
                current_merge.metadata["merged_sections"] = current_merge.metadata.get("merged_sections", []) + [chunk.section_type]
            else:
                # Chunk is large enough or we have a merge to finalize
                if current_merge is not None:
                    merged.append(current_merge)
                    current_merge = None
                merged.append(chunk)
        
        # Add final merge if exists
        if current_merge is not None:
            merged.append(current_merge)
        
        log.info(f"Merged {len(chunks)} chunks into {len(merged)} chunks")
        return merged


# Convenience function
def chunk_resume(structured_resume: Dict[str, Any], max_chunk_size: int = 2000) -> List[ResumeChunk]:
    """
    Chunk resume into semantic sections.
    
    Args:
        structured_resume: Parsed resume data
        max_chunk_size: Maximum characters per chunk
        
    Returns:
        List of ResumeChunk objects
    """
    chunker = ResumeChunker(max_chunk_size=max_chunk_size)
    return chunker.chunk_resume(structured_resume)
