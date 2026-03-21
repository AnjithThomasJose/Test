"""
FactExtractor - Structured facts extraction with confidence scoring
Extracts and validates structured facts from multi-source candidate data
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import uuid

from .candidate_matching_models import (
    StructuredFact, ConfidenceScore, Provenance, FactType, DataSource,
    CandidateProfile, QualityMetrics
)

log = logging.getLogger(__name__)


@dataclass
class ExtractionRule:
    """Rule for fact extraction"""
    fact_type: FactType
    pattern: str
    confidence_base: float
    validation_func: Optional[callable] = None
    required_fields: List[str] = None


@dataclass
class ExtractionContext:
    """Context for fact extraction"""
    source_data: Dict[str, Any]
    source_type: DataSource
    candidate_id: str
    extraction_timestamp: datetime
    metadata: Dict[str, Any] = None


class FactExtractor:
    """
    Extracts structured facts from candidate data with confidence scoring
    Implements validation, deduplication, and quality assessment
    """
    
    def __init__(self):
        self.extraction_rules = self._initialize_extraction_rules()
        self.validation_patterns = self._initialize_validation_patterns()
        self.confidence_weights = self._initialize_confidence_weights()
        
    def _initialize_extraction_rules(self) -> Dict[FactType, List[ExtractionRule]]:
        """Initialize extraction rules for different fact types"""
        return {
            FactType.SKILL: [
                ExtractionRule(
                    fact_type=FactType.SKILL,
                    pattern=r"(?i)(python|java|javascript|react|node\.?js|sql|aws|docker|kubernetes)",
                    confidence_base=0.9,
                    required_fields=["skill_name"]
                ),
                ExtractionRule(
                    fact_type=FactType.SKILL,
                    pattern=r"(?i)(leadership|communication|teamwork|problem.solving)",
                    confidence_base=0.8,
                    required_fields=["skill_name"]
                )
            ],
            FactType.EXPERIENCE: [
                ExtractionRule(
                    fact_type=FactType.EXPERIENCE,
                    pattern=r"(?i)(software engineer|developer|manager|analyst|consultant)",
                    confidence_base=0.85,
                    required_fields=["title", "company"]
                )
            ],
            FactType.EDUCATION: [
                ExtractionRule(
                    fact_type=FactType.EDUCATION,
                    pattern=r"(?i)(bachelor|master|phd|degree|certificate)",
                    confidence_base=0.9,
                    required_fields=["degree", "institution"]
                )
            ],
            FactType.CERTIFICATION: [
                ExtractionRule(
                    fact_type=FactType.CERTIFICATION,
                    pattern=r"(?i)(aws|certified|pmp|scrum|agile)",
                    confidence_base=0.8,
                    required_fields=["certification_name"]
                )
            ],
            FactType.PROJECT: [
                ExtractionRule(
                    fact_type=FactType.PROJECT,
                    pattern=r"(?i)(project|portfolio|github|repository)",
                    confidence_base=0.7,
                    required_fields=["project_name"]
                )
            ],
            FactType.ACHIEVEMENT: [
                ExtractionRule(
                    fact_type=FactType.ACHIEVEMENT,
                    pattern=r"(?i)(award|recognition|achievement|accomplishment)",
                    confidence_base=0.75,
                    required_fields=["achievement_title"]
                )
            ],
            FactType.INTEREST: [
                ExtractionRule(
                    fact_type=FactType.INTEREST,
                    pattern=r"(?i)(interest|passion|hobby|enthusiasm)",
                    confidence_base=0.6,
                    required_fields=["interest_name"]
                )
            ],
            FactType.LOCATION: [
                ExtractionRule(
                    fact_type=FactType.LOCATION,
                    pattern=r"(?i)(san francisco|new york|seattle|austin|remote|hybrid)",
                    confidence_base=0.8,
                    required_fields=["location_name"]
                )
            ],
            FactType.LANGUAGE: [
                ExtractionRule(
                    fact_type=FactType.LANGUAGE,
                    pattern=r"(?i)(english|spanish|french|mandarin|fluent|native)",
                    confidence_base=0.85,
                    required_fields=["language_name", "proficiency"]
                )
            ],
            FactType.SOFT_SKILL: [
                ExtractionRule(
                    fact_type=FactType.SOFT_SKILL,
                    pattern=r"(?i)(collaboration|adaptability|creativity|critical thinking)",
                    confidence_base=0.7,
                    required_fields=["skill_name"]
                )
            ]
        }
    
    def _initialize_validation_patterns(self) -> Dict[str, str]:
        """Initialize validation patterns for different data types"""
        return {
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "phone": r"^[\+]?[1-9][\d]{0,15}$",
            "url": r"^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$",
            "date": r"^\d{4}-\d{2}-\d{2}$",
            "year": r"^\d{4}$",
            "linkedin": r"^https?:\/\/(www\.)?linkedin\.com\/in\/[a-zA-Z0-9-]+\/?$",
            "github": r"^https?:\/\/(www\.)?github\.com\/[a-zA-Z0-9-]+\/?$"
        }
    
    def _initialize_confidence_weights(self) -> Dict[str, float]:
        """Initialize confidence weights for different factors"""
        return {
            "source_reliability": {
                DataSource.RESUME: 0.9,
                DataSource.ASSESSMENT: 0.8,
                DataSource.INTERVIEW: 0.75,
                DataSource.CHAT_SESSION: 0.6,
                DataSource.LINKEDIN: 0.7,
                DataSource.PORTFOLIO: 0.8
            },
            "extraction_method": {
                "structured_parser": 0.9,
                "ai_extraction": 0.8,
                "manual_entry": 0.7,
                "pattern_matching": 0.6,
                "inference": 0.5
            },
            "validation_passed": 0.1,
            "completeness": 0.1,
            "consistency": 0.1
        }
    
    async def extract_facts(self, context: ExtractionContext) -> List[StructuredFact]:
        """
        Extract structured facts from candidate data
        """
        log.info(f"Extracting facts for candidate {context.candidate_id} from {context.source_type}")
        
        facts = []
        
        # Extract facts based on source type
        if context.source_type == DataSource.RESUME:
            facts.extend(await self._extract_from_resume(context))
        elif context.source_type == DataSource.CHAT_SESSION:
            facts.extend(await self._extract_from_chat(context))
        elif context.source_type == DataSource.ASSESSMENT:
            facts.extend(await self._extract_from_assessment(context))
        elif context.source_type == DataSource.INTERVIEW:
            facts.extend(await self._extract_from_interview(context))
        elif context.source_type == DataSource.LINKEDIN:
            facts.extend(await self._extract_from_linkedin(context))
        elif context.source_type == DataSource.PORTFOLIO:
            facts.extend(await self._extract_from_portfolio(context))
        
        # Validate and score facts
        validated_facts = []
        for fact in facts:
            validated_fact = await self._validate_and_score_fact(fact, context)
            if validated_fact:
                validated_facts.append(validated_fact)
        
        # Deduplicate facts
        deduplicated_facts = await self._deduplicate_facts(validated_facts)
        
        log.info(f"Extracted {len(deduplicated_facts)} facts for candidate {context.candidate_id}")
        return deduplicated_facts
    
    async def _extract_from_resume(self, context: ExtractionContext) -> List[StructuredFact]:
        """Extract facts from resume data"""
        facts = []
        resume_data = context.source_data
        
        # Extract skills
        skills = resume_data.get("Skills", [])
        for skill in skills:
            if isinstance(skill, dict):
                skill_name = skill.get("skill", skill.get("name", ""))
                if skill_name:
                    fact = StructuredFact(
                        fact_type=FactType.SKILL,
                        content={
                            "skill_name": skill_name,
                            "skill_level": skill.get("level", "intermediate"),
                            "years_experience": skill.get("years", 0),
                            "skill_data": skill
                        },
                        confidence=ConfidenceScore(value=0.9, reasoning="Directly extracted from resume"),
                        provenance=Provenance(
                            source=DataSource.RESUME,
                            source_id=context.candidate_id,
                            extraction_method="structured_parser"
                        )
                    )
                    facts.append(fact)
        
        # Extract work experience
        work_exp = resume_data.get("WorkExperience", [])
        for i, exp in enumerate(work_exp):
            if isinstance(exp, dict):
                fact = StructuredFact(
                    fact_type=FactType.EXPERIENCE,
                    content={
                        "title": exp.get("title", ""),
                        "company": exp.get("company", ""),
                        "duration": exp.get("duration", ""),
                        "description": exp.get("description", ""),
                        "location": exp.get("location", ""),
                        "start_date": exp.get("start_date", ""),
                        "end_date": exp.get("end_date", ""),
                        "experience_data": exp
                    },
                    confidence=ConfidenceScore(value=0.85, reasoning="Structured experience data"),
                    provenance=Provenance(
                        source=DataSource.RESUME,
                        source_id=context.candidate_id,
                        extraction_method="structured_parser"
                    )
                )
                facts.append(fact)
        
        # Extract education
        education = resume_data.get("Education", [])
        for edu in education:
            if isinstance(edu, dict):
                fact = StructuredFact(
                    fact_type=FactType.EDUCATION,
                    content={
                        "degree": edu.get("degree", ""),
                        "major": edu.get("major", ""),
                        "institution": edu.get("institution", ""),
                        "graduation_year": edu.get("graduation_year", ""),
                        "gpa": edu.get("gpa", ""),
                        "education_data": edu
                    },
                    confidence=ConfidenceScore(value=0.9, reasoning="Structured education data"),
                    provenance=Provenance(
                        source=DataSource.RESUME,
                        source_id=context.candidate_id,
                        extraction_method="structured_parser"
                    )
                )
                facts.append(fact)
        
        # Extract certifications
        certifications = resume_data.get("certifications", [])
        for cert in certifications:
            if isinstance(cert, dict):
                fact = StructuredFact(
                    fact_type=FactType.CERTIFICATION,
                    content={
                        "certification_name": cert.get("name", ""),
                        "issuing_organization": cert.get("issuer", ""),
                        "issue_date": cert.get("issue_date", ""),
                        "expiry_date": cert.get("expiry_date", ""),
                        "certification_data": cert
                    },
                    confidence=ConfidenceScore(value=0.8, reasoning="Structured certification data"),
                    provenance=Provenance(
                        source=DataSource.RESUME,
                        source_id=context.candidate_id,
                        extraction_method="structured_parser"
                    )
                )
                facts.append(fact)
        
        # Extract projects
        projects = resume_data.get("projects", [])
        for project in projects:
            if isinstance(project, dict):
                fact = StructuredFact(
                    fact_type=FactType.PROJECT,
                    content={
                        "project_name": project.get("name", ""),
                        "description": project.get("description", ""),
                        "technologies": project.get("technologies", []),
                        "url": project.get("url", ""),
                        "project_data": project
                    },
                    confidence=ConfidenceScore(value=0.7, reasoning="Project information"),
                    provenance=Provenance(
                        source=DataSource.RESUME,
                        source_id=context.candidate_id,
                        extraction_method="structured_parser"
                    )
                )
                facts.append(fact)
        
        return facts
    
    async def _extract_from_chat(self, context: ExtractionContext) -> List[StructuredFact]:
        """Extract facts from chat session data"""
        facts = []
        chat_data = context.source_data
        
        # Extract interests and preferences
        interests = chat_data.get("interests", [])
        for interest in interests:
            fact = StructuredFact(
                fact_type=FactType.INTEREST,
                content={
                    "interest_name": interest,
                    "source": "chat_conversation",
                    "context": chat_data.get("context", "")
                },
                confidence=ConfidenceScore(value=0.7, reasoning="Expressed in chat conversation"),
                provenance=Provenance(
                    source=DataSource.CHAT_SESSION,
                    source_id=context.candidate_id,
                    extraction_method="ai_extraction"
                )
            )
            facts.append(fact)
        
        # Extract soft skills from conversation
        conversation_text = chat_data.get("conversation_text", "")
        soft_skills = self._extract_soft_skills_from_text(conversation_text)
        for skill in soft_skills:
            fact = StructuredFact(
                fact_type=FactType.SOFT_SKILL,
                content={
                    "skill_name": skill,
                    "source": "chat_conversation",
                    "context": conversation_text[:200] + "..." if len(conversation_text) > 200 else conversation_text
                },
                confidence=ConfidenceScore(value=0.6, reasoning="Inferred from conversation"),
                provenance=Provenance(
                    source=DataSource.CHAT_SESSION,
                    source_id=context.candidate_id,
                    extraction_method="ai_extraction"
                )
            )
            facts.append(fact)
        
        return facts
    
    async def _extract_from_assessment(self, context: ExtractionContext) -> List[StructuredFact]:
        """Extract facts from assessment data"""
        facts = []
        assessment_data = context.source_data
        
        # Extract skill assessments
        skill_scores = assessment_data.get("skill_scores", {})
        for skill, score in skill_scores.items():
            fact = StructuredFact(
                fact_type=FactType.SKILL,
                content={
                    "skill_name": skill,
                    "assessment_score": score,
                    "assessment_type": assessment_data.get("assessment_type", ""),
                    "assessment_date": assessment_data.get("completed_at", ""),
                    "skill_data": {"assessment_score": score}
                },
                confidence=ConfidenceScore(value=0.8, reasoning="Validated through assessment"),
                provenance=Provenance(
                    source=DataSource.ASSESSMENT,
                    source_id=context.candidate_id,
                    extraction_method="assessment_scoring"
                )
            )
            facts.append(fact)
        
        # Extract personality traits
        personality_traits = assessment_data.get("personality_traits", {})
        for trait, value in personality_traits.items():
            fact = StructuredFact(
                fact_type=FactType.SOFT_SKILL,
                content={
                    "skill_name": trait,
                    "trait_value": value,
                    "assessment_type": "personality",
                    "skill_data": {"trait_value": value}
                },
                confidence=ConfidenceScore(value=0.75, reasoning="Personality assessment"),
                provenance=Provenance(
                    source=DataSource.ASSESSMENT,
                    source_id=context.candidate_id,
                    extraction_method="assessment_scoring"
                )
            )
            facts.append(fact)
        
        return facts
    
    async def _extract_from_interview(self, context: ExtractionContext) -> List[StructuredFact]:
        """Extract facts from interview data"""
        facts = []
        interview_data = context.source_data
        
        # Extract soft skills from interview responses
        responses = interview_data.get("responses", [])
        for response in responses:
            if response.get("question_type") == "behavioral":
                # Analyze response for soft skills
                response_text = response.get("response", "")
                soft_skills = self._extract_soft_skills_from_text(response_text)
                
                for skill in soft_skills:
                    fact = StructuredFact(
                        fact_type=FactType.SOFT_SKILL,
                        content={
                            "skill_name": skill,
                            "source": "interview_response",
                            "question": response.get("question", ""),
                            "response_context": response_text[:200] + "..." if len(response_text) > 200 else response_text
                        },
                        confidence=ConfidenceScore(value=0.75, reasoning="Inferred from interview response"),
                        provenance=Provenance(
                            source=DataSource.INTERVIEW,
                            source_id=context.candidate_id,
                            extraction_method="ai_extraction"
                        )
                    )
                    facts.append(fact)
        
        return facts
    
    async def _extract_from_linkedin(self, context: ExtractionContext) -> List[StructuredFact]:
        """Extract facts from LinkedIn data"""
        facts = []
        linkedin_data = context.source_data
        
        # Extract skills from LinkedIn
        skills = linkedin_data.get("skills", [])
        for skill in skills:
            fact = StructuredFact(
                fact_type=FactType.SKILL,
                content={
                    "skill_name": skill,
                    "source": "linkedin",
                    "skill_data": {"linkedin_skill": skill}
                },
                confidence=ConfidenceScore(value=0.7, reasoning="LinkedIn profile data"),
                provenance=Provenance(
                    source=DataSource.LINKEDIN,
                    source_id=context.candidate_id,
                    extraction_method="api_extraction"
                )
            )
            facts.append(fact)
        
        return facts
    
    async def _extract_from_portfolio(self, context: ExtractionContext) -> List[StructuredFact]:
        """Extract facts from portfolio data"""
        facts = []
        portfolio_data = context.source_data
        
        # Extract projects from portfolio
        projects = portfolio_data.get("projects", [])
        for project in projects:
            fact = StructuredFact(
                fact_type=FactType.PROJECT,
                content={
                    "project_name": project.get("name", ""),
                    "description": project.get("description", ""),
                    "technologies": project.get("technologies", []),
                    "url": project.get("url", ""),
                    "source": "portfolio"
                },
                confidence=ConfidenceScore(value=0.8, reasoning="Portfolio project data"),
                provenance=Provenance(
                    source=DataSource.PORTFOLIO,
                    source_id=context.candidate_id,
                    extraction_method="api_extraction"
                )
            )
            facts.append(fact)
        
        return facts
    
    def _extract_soft_skills_from_text(self, text: str) -> List[str]:
        """Extract soft skills from text using pattern matching"""
        soft_skills = []
        text_lower = text.lower()
        
        skill_patterns = {
            "leadership": ["lead", "manage", "team", "direct", "guide"],
            "communication": ["communicate", "present", "explain", "articulate"],
            "problem_solving": ["solve", "problem", "challenge", "solution"],
            "collaboration": ["collaborate", "work together", "teamwork", "cooperate"],
            "adaptability": ["adapt", "flexible", "change", "adjust"],
            "creativity": ["creative", "innovative", "design", "imagine"],
            "critical_thinking": ["analyze", "evaluate", "assess", "critique"]
        }
        
        for skill, patterns in skill_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                soft_skills.append(skill)
        
        return soft_skills
    
    async def _validate_and_score_fact(self, fact: StructuredFact, context: ExtractionContext) -> Optional[StructuredFact]:
        """Validate and score a fact"""
        # Calculate confidence score
        confidence_score = await self._calculate_confidence_score(fact, context)
        
        # Update confidence
        fact.confidence = confidence_score
        
        # Validate fact content
        if not await self._validate_fact_content(fact):
            log.warning(f"Fact validation failed for {fact.fact_id}")
            return None
        
        return fact
    
    async def _calculate_confidence_score(self, fact: StructuredFact, context: ExtractionContext) -> ConfidenceScore:
        """Calculate confidence score for a fact"""
        base_score = fact.confidence.value
        
        # Adjust based on source reliability
        source_weight = self.confidence_weights["source_reliability"].get(
            context.source_type, 0.5
        )
        
        # Adjust based on extraction method
        method_weight = self.confidence_weights["extraction_method"].get(
            fact.provenance.extraction_method, 0.5
        )
        
        # Adjust based on validation
        validation_bonus = 0.0
        if await self._validate_fact_content(fact):
            validation_bonus = self.confidence_weights["validation_passed"]
        
        # Adjust based on completeness
        completeness_bonus = await self._calculate_completeness_bonus(fact)
        
        # Calculate final score
        final_score = min(1.0, base_score * source_weight * method_weight + validation_bonus + completeness_bonus)
        
        return ConfidenceScore(
            value=final_score,
            reasoning=f"Adjusted from {base_score:.2f} based on source ({source_weight:.2f}), method ({method_weight:.2f}), validation (+{validation_bonus:.2f}), completeness (+{completeness_bonus:.2f})"
        )
    
    async def _validate_fact_content(self, fact: StructuredFact) -> bool:
        """Validate fact content based on type"""
        content = fact.content
        
        if fact.fact_type == FactType.SKILL:
            return bool(content.get("skill_name"))
        elif fact.fact_type == FactType.EXPERIENCE:
            return bool(content.get("title") and content.get("company"))
        elif fact.fact_type == FactType.EDUCATION:
            return bool(content.get("degree") and content.get("institution"))
        elif fact.fact_type == FactType.CERTIFICATION:
            return bool(content.get("certification_name"))
        elif fact.fact_type == FactType.PROJECT:
            return bool(content.get("project_name"))
        elif fact.fact_type == FactType.ACHIEVEMENT:
            return bool(content.get("achievement_title"))
        elif fact.fact_type == FactType.INTEREST:
            return bool(content.get("interest_name"))
        elif fact.fact_type == FactType.LOCATION:
            return bool(content.get("location_name"))
        elif fact.fact_type == FactType.LANGUAGE:
            return bool(content.get("language_name") and content.get("proficiency"))
        elif fact.fact_type == FactType.SOFT_SKILL:
            return bool(content.get("skill_name"))
        
        return True
    
    async def _calculate_completeness_bonus(self, fact: StructuredFact) -> float:
        """Calculate completeness bonus for a fact"""
        content = fact.content
        required_fields = self.extraction_rules.get(fact.fact_type, [{}])[0].required_fields or []
        
        if not required_fields:
            return 0.0
        
        completed_fields = sum(1 for field in required_fields if content.get(field))
        completeness_ratio = completed_fields / len(required_fields)
        
        return completeness_ratio * self.confidence_weights["completeness"]
    
    async def _deduplicate_facts(self, facts: List[StructuredFact]) -> List[StructuredFact]:
        """Remove duplicate facts based on content similarity"""
        if not facts:
            return facts
        
        deduplicated = []
        seen_facts = set()
        
        for fact in facts:
            # Create a signature for the fact
            signature = self._create_fact_signature(fact)
            
            if signature not in seen_facts:
                seen_facts.add(signature)
                deduplicated.append(fact)
            else:
                # If duplicate, keep the one with higher confidence
                existing_fact = next(f for f in deduplicated if self._create_fact_signature(f) == signature)
                if fact.confidence.value > existing_fact.confidence.value:
                    deduplicated.remove(existing_fact)
                    deduplicated.append(fact)
        
        return deduplicated
    
    def _create_fact_signature(self, fact: StructuredFact) -> str:
        """Create a signature for fact deduplication"""
        if fact.fact_type == FactType.SKILL:
            return f"skill:{fact.content.get('skill_name', '').lower()}"
        elif fact.fact_type == FactType.EXPERIENCE:
            return f"exp:{fact.content.get('title', '').lower()}:{fact.content.get('company', '').lower()}"
        elif fact.fact_type == FactType.EDUCATION:
            return f"edu:{fact.content.get('degree', '').lower()}:{fact.content.get('institution', '').lower()}"
        elif fact.fact_type == FactType.CERTIFICATION:
            return f"cert:{fact.content.get('certification_name', '').lower()}"
        elif fact.fact_type == FactType.PROJECT:
            return f"proj:{fact.content.get('project_name', '').lower()}"
        elif fact.fact_type == FactType.ACHIEVEMENT:
            return f"ach:{fact.content.get('achievement_title', '').lower()}"
        elif fact.fact_type == FactType.INTEREST:
            return f"int:{fact.content.get('interest_name', '').lower()}"
        elif fact.fact_type == FactType.LOCATION:
            return f"loc:{fact.content.get('location_name', '').lower()}"
        elif fact.fact_type == FactType.LANGUAGE:
            return f"lang:{fact.content.get('language_name', '').lower()}"
        elif fact.fact_type == FactType.SOFT_SKILL:
            return f"soft:{fact.content.get('skill_name', '').lower()}"
        
        return f"other:{fact.fact_id}"
    
    async def merge_facts_from_multiple_sources(self, fact_lists: List[List[StructuredFact]]) -> List[StructuredFact]:
        """Merge facts from multiple sources, handling conflicts"""
        all_facts = []
        for fact_list in fact_lists:
            all_facts.extend(fact_list)
        
        # Group facts by signature
        fact_groups = {}
        for fact in all_facts:
            signature = self._create_fact_signature(fact)
            if signature not in fact_groups:
                fact_groups[signature] = []
            fact_groups[signature].append(fact)
        
        # Merge facts in each group
        merged_facts = []
        for signature, facts in fact_groups.items():
            if len(facts) == 1:
                merged_facts.append(facts[0])
            else:
                # Merge multiple facts with same signature
                merged_fact = await self._merge_duplicate_facts(facts)
                merged_facts.append(merged_fact)
        
        return merged_facts
    
    async def _merge_duplicate_facts(self, facts: List[StructuredFact]) -> StructuredFact:
        """Merge duplicate facts from different sources"""
        # Use the fact with highest confidence as base
        base_fact = max(facts, key=lambda f: f.confidence.value)
        
        # Merge content from all facts
        merged_content = base_fact.content.copy()
        
        for fact in facts:
            if fact != base_fact:
                # Merge additional fields
                for key, value in fact.content.items():
                    if key not in merged_content or not merged_content[key]:
                        merged_content[key] = value
                    elif isinstance(value, list) and isinstance(merged_content[key], list):
                        # Merge lists
                        merged_content[key] = list(set(merged_content[key] + value))
        
        # Calculate merged confidence
        avg_confidence = sum(f.confidence.value for f in facts) / len(facts)
        merged_confidence = ConfidenceScore(
            value=avg_confidence,
            reasoning=f"Merged from {len(facts)} sources with average confidence {avg_confidence:.2f}"
        )
        
        # Create merged fact
        merged_fact = StructuredFact(
            fact_id=base_fact.fact_id,
            fact_type=base_fact.fact_type,
            content=merged_content,
            confidence=merged_confidence,
            provenance=base_fact.provenance,
            tags=list(set(tag for fact in facts for tag in fact.tags)),
            metadata={
                "merged_from": [f.fact_id for f in facts],
                "source_count": len(facts),
                "merge_timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return merged_fact
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return {
            "extraction_rules_count": sum(len(rules) for rules in self.extraction_rules.values()),
            "validation_patterns_count": len(self.validation_patterns),
            "confidence_weights": self.confidence_weights,
            "supported_fact_types": [ft.value for ft in FactType],
            "supported_sources": [ds.value for ds in DataSource]
        }


# ==================== SINGLETON INSTANCE ====================

# Global instance for the application
fact_extractor = FactExtractor()
