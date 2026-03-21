"""
TagGenerator - Dynamic tag generation based on content analysis
Generates contextual tags with confidence scoring for candidate profiles
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from dataclasses import dataclass
import re

from .candidate_matching_models import (
    DynamicTag, ConfidenceScore, Provenance, TagCategory, DataSource,
    StructuredFact, FactType, CandidateProfile
)

log = logging.getLogger(__name__)


@dataclass
class TagRule:
    """Rule for tag generation"""
    category: TagCategory
    pattern: str
    confidence_base: float
    required_facts: List[FactType] = None
    exclusion_patterns: List[str] = None


@dataclass
class TagContext:
    """Context for tag generation"""
    candidate_id: str
    facts: List[StructuredFact]
    profile_data: Dict[str, Any]
    source_data: Dict[str, Any]
    generation_timestamp: datetime


class TagGenerator:
    """
    Generates dynamic tags based on content analysis
    Implements contextual tagging with confidence scoring
    """
    
    def __init__(self):
        self.tag_rules = self._initialize_tag_rules()
        self.tag_patterns = self._initialize_tag_patterns()
        self.confidence_weights = self._initialize_confidence_weights()
        
    def _initialize_tag_rules(self) -> Dict[TagCategory, List[TagRule]]:
        """Initialize tag generation rules"""
        return {
            TagCategory.TECHNICAL_SKILL: [
                TagRule(
                    category=TagCategory.TECHNICAL_SKILL,
                    pattern=r"(?i)(python|java|javascript|react|node\.?js)",
                    confidence_base=0.9,
                    required_facts=[FactType.SKILL]
                ),
                TagRule(
                    category=TagCategory.TECHNICAL_SKILL,
                    pattern=r"(?i)(aws|azure|gcp|cloud)",
                    confidence_base=0.85,
                    required_facts=[FactType.SKILL]
                ),
                TagRule(
                    category=TagCategory.TECHNICAL_SKILL,
                    pattern=r"(?i)(docker|kubernetes|devops)",
                    confidence_base=0.8,
                    required_facts=[FactType.SKILL]
                ),
                TagRule(
                    category=TagCategory.TECHNICAL_SKILL,
                    pattern=r"(?i)(machine learning|ai|data science)",
                    confidence_base=0.85,
                    required_facts=[FactType.SKILL]
                )
            ],
            TagCategory.INDUSTRY: [
                TagRule(
                    category=TagCategory.INDUSTRY,
                    pattern=r"(?i)(fintech|financial|banking)",
                    confidence_base=0.8,
                    required_facts=[FactType.EXPERIENCE]
                ),
                TagRule(
                    category=TagCategory.INDUSTRY,
                    pattern=r"(?i)(healthcare|medical|pharma)",
                    confidence_base=0.8,
                    required_facts=[FactType.EXPERIENCE]
                ),
                TagRule(
                    category=TagCategory.INDUSTRY,
                    pattern=r"(?i)(ecommerce|retail|shopping)",
                    confidence_base=0.8,
                    required_facts=[FactType.EXPERIENCE]
                ),
                TagRule(
                    category=TagCategory.INDUSTRY,
                    pattern=r"(?i)(saas|software|technology)",
                    confidence_base=0.8,
                    required_facts=[FactType.EXPERIENCE]
                )
            ],
            TagCategory.SENIORITY: [
                TagRule(
                    category=TagCategory.SENIORITY,
                    pattern=r"(?i)(senior|lead|principal|architect)",
                    confidence_base=0.9,
                    required_facts=[FactType.EXPERIENCE]
                ),
                TagRule(
                    category=TagCategory.SENIORITY,
                    pattern=r"(?i)(junior|entry|associate)",
                    confidence_base=0.9,
                    required_facts=[FactType.EXPERIENCE]
                ),
                TagRule(
                    category=TagCategory.SENIORITY,
                    pattern=r"(?i)(mid|intermediate|experienced)",
                    confidence_base=0.8,
                    required_facts=[FactType.EXPERIENCE]
                )
            ],
            TagCategory.ROLE_TYPE: [
                TagRule(
                    category=TagCategory.ROLE_TYPE,
                    pattern=r"(?i)(frontend|front-end|ui|ux)",
                    confidence_base=0.85,
                    required_facts=[FactType.SKILL, FactType.EXPERIENCE]
                ),
                TagRule(
                    category=TagCategory.ROLE_TYPE,
                    pattern=r"(?i)(backend|back-end|api|server)",
                    confidence_base=0.85,
                    required_facts=[FactType.SKILL, FactType.EXPERIENCE]
                ),
                TagRule(
                    category=TagCategory.ROLE_TYPE,
                    pattern=r"(?i)(fullstack|full-stack|full stack)",
                    confidence_base=0.8,
                    required_facts=[FactType.SKILL, FactType.EXPERIENCE]
                ),
                TagRule(
                    category=TagCategory.ROLE_TYPE,
                    pattern=r"(?i)(data engineer|data scientist|analyst)",
                    confidence_base=0.85,
                    required_facts=[FactType.SKILL, FactType.EXPERIENCE]
                ),
                TagRule(
                    category=TagCategory.ROLE_TYPE,
                    pattern=r"(?i)(devops|sre|infrastructure)",
                    confidence_base=0.85,
                    required_facts=[FactType.SKILL, FactType.EXPERIENCE]
                )
            ],
            TagCategory.LOCATION_PREFERENCE: [
                TagRule(
                    category=TagCategory.LOCATION_PREFERENCE,
                    pattern=r"(?i)(remote|work from home|wfh)",
                    confidence_base=0.9,
                    required_facts=[FactType.LOCATION]
                ),
                TagRule(
                    category=TagCategory.LOCATION_PREFERENCE,
                    pattern=r"(?i)(hybrid|flexible|onsite)",
                    confidence_base=0.8,
                    required_facts=[FactType.LOCATION]
                ),
                TagRule(
                    category=TagCategory.LOCATION_PREFERENCE,
                    pattern=r"(?i)(san francisco|sf|bay area)",
                    confidence_base=0.9,
                    required_facts=[FactType.LOCATION]
                ),
                TagRule(
                    category=TagCategory.LOCATION_PREFERENCE,
                    pattern=r"(?i)(new york|nyc|manhattan)",
                    confidence_base=0.9,
                    required_facts=[FactType.LOCATION]
                )
            ],
            TagCategory.WORK_STYLE: [
                TagRule(
                    category=TagCategory.WORK_STYLE,
                    pattern=r"(?i)(startup|early stage|fast paced)",
                    confidence_base=0.8,
                    required_facts=[FactType.EXPERIENCE]
                ),
                TagRule(
                    category=TagCategory.WORK_STYLE,
                    pattern=r"(?i)(enterprise|large company|corporate)",
                    confidence_base=0.8,
                    required_facts=[FactType.EXPERIENCE]
                ),
                TagRule(
                    category=TagCategory.WORK_STYLE,
                    pattern=r"(?i)(agile|scrum|kanban)",
                    confidence_base=0.7,
                    required_facts=[FactType.SKILL]
                )
            ],
            TagCategory.CAREER_STAGE: [
                TagRule(
                    category=TagCategory.CAREER_STAGE,
                    pattern=r"(?i)(recent graduate|new grad|entry level)",
                    confidence_base=0.9,
                    required_facts=[FactType.EDUCATION]
                ),
                TagRule(
                    category=TagCategory.CAREER_STAGE,
                    pattern=r"(?i)(career change|transition|pivot)",
                    confidence_base=0.8,
                    required_facts=[FactType.EXPERIENCE]
                ),
                TagRule(
                    category=TagCategory.CAREER_STAGE,
                    pattern=r"(?i)(experienced|veteran|seasoned)",
                    confidence_base=0.8,
                    required_facts=[FactType.EXPERIENCE]
                )
            ],
            TagCategory.SPECIALIZATION: [
                TagRule(
                    category=TagCategory.SPECIALIZATION,
                    pattern=r"(?i)(mobile|ios|android)",
                    confidence_base=0.85,
                    required_facts=[FactType.SKILL]
                ),
                TagRule(
                    category=TagCategory.SPECIALIZATION,
                    pattern=r"(?i)(web|frontend|backend)",
                    confidence_base=0.8,
                    required_facts=[FactType.SKILL]
                ),
                TagRule(
                    category=TagCategory.SPECIALIZATION,
                    pattern=r"(?i)(security|cybersecurity|infosec)",
                    confidence_base=0.85,
                    required_facts=[FactType.SKILL]
                ),
                TagRule(
                    category=TagCategory.SPECIALIZATION,
                    pattern=r"(?i)(blockchain|crypto|web3)",
                    confidence_base=0.85,
                    required_facts=[FactType.SKILL]
                )
            ]
        }
    
    def _initialize_tag_patterns(self) -> Dict[str, List[str]]:
        """Initialize tag patterns for different categories"""
        return {
            "technical_skills": [
                "python_developer", "java_developer", "javascript_developer",
                "react_developer", "nodejs_developer", "fullstack_developer",
                "frontend_developer", "backend_developer", "mobile_developer",
                "devops_engineer", "data_scientist", "ml_engineer",
                "cloud_engineer", "security_engineer", "blockchain_developer"
            ],
            "industries": [
                "fintech", "healthcare", "ecommerce", "saas", "edtech",
                "gaming", "media", "automotive", "aerospace", "retail"
            ],
            "seniority_levels": [
                "junior", "mid_level", "senior", "lead", "principal", "architect"
            ],
            "work_styles": [
                "startup_experience", "enterprise_experience", "remote_work",
                "agile_methodology", "fast_paced", "collaborative"
            ],
            "specializations": [
                "ai_ml", "cloud_native", "microservices", "api_design",
                "database_design", "performance_optimization", "scalability"
            ]
        }
    
    def _initialize_confidence_weights(self) -> Dict[str, float]:
        """Initialize confidence weights for tag generation"""
        return {
            "fact_confidence_weight": 0.4,
            "pattern_match_weight": 0.3,
            "context_weight": 0.2,
            "consistency_weight": 0.1
        }
    
    async def generate_tags(self, context: TagContext) -> List[DynamicTag]:
        """
        Generate dynamic tags based on candidate context
        """
        log.info(f"Generating tags for candidate {context.candidate_id}")
        
        tags = []
        
        # Generate tags based on facts
        fact_tags = await self._generate_tags_from_facts(context)
        tags.extend(fact_tags)
        
        # Generate tags based on profile data
        profile_tags = await self._generate_tags_from_profile(context)
        tags.extend(profile_tags)
        
        # Generate tags based on source data
        source_tags = await self._generate_tags_from_source(context)
        tags.extend(source_tags)
        
        # Generate contextual tags
        contextual_tags = await self._generate_contextual_tags(context)
        tags.extend(contextual_tags)
        
        # Validate and score tags
        validated_tags = []
        for tag in tags:
            validated_tag = await self._validate_and_score_tag(tag, context)
            if validated_tag:
                validated_tags.append(validated_tag)
        
        # Deduplicate tags
        deduplicated_tags = await self._deduplicate_tags(validated_tags)
        
        log.info(f"Generated {len(deduplicated_tags)} tags for candidate {context.candidate_id}")
        return deduplicated_tags
    
    async def _generate_tags_from_facts(self, context: TagContext) -> List[DynamicTag]:
        """Generate tags from structured facts"""
        tags = []
        
        for fact in context.facts:
            if fact.fact_type == FactType.SKILL:
                skill_tags = await self._generate_skill_tags(fact, context)
                tags.extend(skill_tags)
            elif fact.fact_type == FactType.EXPERIENCE:
                experience_tags = await self._generate_experience_tags(fact, context)
                tags.extend(experience_tags)
            elif fact.fact_type == FactType.EDUCATION:
                education_tags = await self._generate_education_tags(fact, context)
                tags.extend(education_tags)
            elif fact.fact_type == FactType.CERTIFICATION:
                certification_tags = await self._generate_certification_tags(fact, context)
                tags.extend(certification_tags)
        
        return tags
    
    async def _generate_skill_tags(self, fact: StructuredFact, context: TagContext) -> List[DynamicTag]:
        """Generate tags from skill facts"""
        tags = []
        skill_name = fact.content.get("skill_name", "").lower()
        
        # Check technical skill rules
        for rule in self.tag_rules.get(TagCategory.TECHNICAL_SKILL, []):
            if re.search(rule.pattern, skill_name):
                tag = DynamicTag(
                    category=TagCategory.TECHNICAL_SKILL,
                    value=self._normalize_tag_value(skill_name),
                    confidence=ConfidenceScore(
                        value=rule.confidence_base * fact.confidence.value,
                        reasoning=f"Generated from skill fact: {skill_name}"
                    ),
                    provenance=Provenance(
                        source=fact.provenance.source,
                        source_id=context.candidate_id,
                        extraction_method="skill_analysis"
                    ),
                    related_facts=[fact.fact_id]
                )
                tags.append(tag)
        
        # Generate role type tags based on skill combinations
        role_tags = await self._generate_role_type_tags(fact, context)
        tags.extend(role_tags)
        
        return tags
    
    async def _generate_experience_tags(self, fact: StructuredFact, context: TagContext) -> List[DynamicTag]:
        """Generate tags from experience facts"""
        tags = []
        title = fact.content.get("title", "").lower()
        company = fact.content.get("company", "").lower()
        description = fact.content.get("description", "").lower()
        
        # Generate seniority tags
        for rule in self.tag_rules.get(TagCategory.SENIORITY, []):
            if re.search(rule.pattern, title) or re.search(rule.pattern, description):
                tag = DynamicTag(
                    category=TagCategory.SENIORITY,
                    value=self._extract_seniority_level(title, description),
                    confidence=ConfidenceScore(
                        value=rule.confidence_base * fact.confidence.value,
                        reasoning=f"Generated from experience: {title}"
                    ),
                    provenance=Provenance(
                        source=fact.provenance.source,
                        source_id=context.candidate_id,
                        extraction_method="experience_analysis"
                    ),
                    related_facts=[fact.fact_id]
                )
                tags.append(tag)
        
        # Generate industry tags
        industry_tags = await self._generate_industry_tags(fact, context)
        tags.extend(industry_tags)
        
        # Generate work style tags
        work_style_tags = await self._generate_work_style_tags(fact, context)
        tags.extend(work_style_tags)
        
        return tags
    
    async def _generate_education_tags(self, fact: StructuredFact, context: TagContext) -> List[DynamicTag]:
        """Generate tags from education facts"""
        tags = []
        degree = fact.content.get("degree", "").lower()
        major = fact.content.get("major", "").lower()
        
        # Generate career stage tags
        if "graduate" in degree or "recent" in degree:
            tag = DynamicTag(
                category=TagCategory.CAREER_STAGE,
                value="recent_graduate",
                confidence=ConfidenceScore(
                    value=0.9 * fact.confidence.value,
                    reasoning=f"Generated from education: {degree}"
                ),
                provenance=Provenance(
                    source=fact.provenance.source,
                    source_id=context.candidate_id,
                    extraction_method="education_analysis"
                ),
                related_facts=[fact.fact_id]
            )
            tags.append(tag)
        
        # Generate specialization tags based on major
        if any(term in major for term in ["computer", "software", "engineering", "science"]):
            tag = DynamicTag(
                category=TagCategory.SPECIALIZATION,
                value="technical_background",
                confidence=ConfidenceScore(
                    value=0.8 * fact.confidence.value,
                    reasoning=f"Generated from major: {major}"
                ),
                provenance=Provenance(
                    source=fact.provenance.source,
                    source_id=context.candidate_id,
                    extraction_method="education_analysis"
                ),
                related_facts=[fact.fact_id]
            )
            tags.append(tag)
        
        return tags
    
    async def _generate_certification_tags(self, fact: StructuredFact, context: TagContext) -> List[DynamicTag]:
        """Generate tags from certification facts"""
        tags = []
        cert_name = fact.content.get("certification_name", "").lower()
        
        # Generate technical skill tags from certifications
        if any(term in cert_name for term in ["aws", "azure", "gcp", "cloud"]):
            tag = DynamicTag(
                category=TagCategory.TECHNICAL_SKILL,
                value="cloud_certified",
                confidence=ConfidenceScore(
                    value=0.85 * fact.confidence.value,
                    reasoning=f"Generated from certification: {cert_name}"
                ),
                provenance=Provenance(
                    source=fact.provenance.source,
                    source_id=context.candidate_id,
                    extraction_method="certification_analysis"
                ),
                related_facts=[fact.fact_id]
            )
            tags.append(tag)
        
        return tags
    
    async def _generate_role_type_tags(self, fact: StructuredFact, context: TagContext) -> List[DynamicTag]:
        """Generate role type tags based on skill combinations"""
        tags = []
        skill_name = fact.content.get("skill_name", "").lower()
        
        # Check role type rules
        for rule in self.tag_rules.get(TagCategory.ROLE_TYPE, []):
            if re.search(rule.pattern, skill_name):
                tag = DynamicTag(
                    category=TagCategory.ROLE_TYPE,
                    value=self._extract_role_type(skill_name),
                    confidence=ConfidenceScore(
                        value=rule.confidence_base * fact.confidence.value,
                        reasoning=f"Generated from skill: {skill_name}"
                    ),
                    provenance=Provenance(
                        source=fact.provenance.source,
                        source_id=context.candidate_id,
                        extraction_method="skill_analysis"
                    ),
                    related_facts=[fact.fact_id]
                )
                tags.append(tag)
        
        return tags
    
    async def _generate_industry_tags(self, fact: StructuredFact, context: TagContext) -> List[DynamicTag]:
        """Generate industry tags from experience"""
        tags = []
        company = fact.content.get("company", "").lower()
        description = fact.content.get("description", "").lower()
        
        # Check industry rules
        for rule in self.tag_rules.get(TagCategory.INDUSTRY, []):
            if re.search(rule.pattern, company) or re.search(rule.pattern, description):
                tag = DynamicTag(
                    category=TagCategory.INDUSTRY,
                    value=self._extract_industry(company, description),
                    confidence=ConfidenceScore(
                        value=rule.confidence_base * fact.confidence.value,
                        reasoning=f"Generated from experience at: {company}"
                    ),
                    provenance=Provenance(
                        source=fact.provenance.source,
                        source_id=context.candidate_id,
                        extraction_method="experience_analysis"
                    ),
                    related_facts=[fact.fact_id]
                )
                tags.append(tag)
        
        return tags
    
    async def _generate_work_style_tags(self, fact: StructuredFact, context: TagContext) -> List[DynamicTag]:
        """Generate work style tags from experience"""
        tags = []
        description = fact.content.get("description", "").lower()
        
        # Check work style rules
        for rule in self.tag_rules.get(TagCategory.WORK_STYLE, []):
            if re.search(rule.pattern, description):
                tag = DynamicTag(
                    category=TagCategory.WORK_STYLE,
                    value=self._extract_work_style(description),
                    confidence=ConfidenceScore(
                        value=rule.confidence_base * fact.confidence.value,
                        reasoning=f"Generated from experience description"
                    ),
                    provenance=Provenance(
                        source=fact.provenance.source,
                        source_id=context.candidate_id,
                        extraction_method="experience_analysis"
                    ),
                    related_facts=[fact.fact_id]
                )
                tags.append(tag)
        
        return tags
    
    async def _generate_tags_from_profile(self, context: TagContext) -> List[DynamicTag]:
        """Generate tags from profile data"""
        tags = []
        profile_data = context.profile_data
        
        # Generate location preference tags
        location = profile_data.get("location", "").lower()
        if location:
            if "remote" in location or "work from home" in location:
                tag = DynamicTag(
                    category=TagCategory.LOCATION_PREFERENCE,
                    value="remote_preference",
                    confidence=ConfidenceScore(
                        value=0.9,
                        reasoning="Location preference from profile"
                    ),
                    provenance=Provenance(
                        source=DataSource.RESUME,
                        source_id=context.candidate_id,
                        extraction_method="profile_analysis"
                    )
                )
                tags.append(tag)
        
        return tags
    
    async def _generate_tags_from_source(self, context: TagContext) -> List[DynamicTag]:
        """Generate tags from source data"""
        tags = []
        source_data = context.source_data
        
        # Generate tags based on source type
        if "interests" in source_data:
            interests = source_data.get("interests", [])
            for interest in interests:
                if isinstance(interest, str):
                    tag = DynamicTag(
                        category=TagCategory.SPECIALIZATION,
                        value=f"interest_{interest.lower().replace(' ', '_')}",
                        confidence=ConfidenceScore(
                            value=0.6,
                            reasoning="Interest expressed in source data"
                        ),
                        provenance=Provenance(
                            source=DataSource.CHAT_SESSION,
                            source_id=context.candidate_id,
                            extraction_method="interest_analysis"
                        )
                    )
                    tags.append(tag)
        
        return tags
    
    async def _generate_contextual_tags(self, context: TagContext) -> List[DynamicTag]:
        """Generate contextual tags based on overall profile"""
        tags = []
        
        # Analyze skill combinations for role type
        skill_facts = [f for f in context.facts if f.fact_type == FactType.SKILL]
        if skill_facts:
            role_type = await self._analyze_role_type_from_skills(skill_facts)
            if role_type:
                tag = DynamicTag(
                    category=TagCategory.ROLE_TYPE,
                    value=role_type,
                    confidence=ConfidenceScore(
                        value=0.8,
                        reasoning="Analyzed from skill combination"
                    ),
                    provenance=Provenance(
                        source=DataSource.RESUME,
                        source_id=context.candidate_id,
                        extraction_method="contextual_analysis"
                    ),
                    related_facts=[f.fact_id for f in skill_facts]
                )
                tags.append(tag)
        
        # Analyze experience for career stage
        experience_facts = [f for f in context.facts if f.fact_type == FactType.EXPERIENCE]
        if experience_facts:
            career_stage = await self._analyze_career_stage_from_experience(experience_facts)
            if career_stage:
                tag = DynamicTag(
                    category=TagCategory.CAREER_STAGE,
                    value=career_stage,
                    confidence=ConfidenceScore(
                        value=0.8,
                        reasoning="Analyzed from experience pattern"
                    ),
                    provenance=Provenance(
                        source=DataSource.RESUME,
                        source_id=context.candidate_id,
                        extraction_method="contextual_analysis"
                    ),
                    related_facts=[f.fact_id for f in experience_facts]
                )
                tags.append(tag)
        
        return tags
    
    async def _analyze_role_type_from_skills(self, skill_facts: List[StructuredFact]) -> Optional[str]:
        """Analyze role type from skill combinations"""
        skills = [f.content.get("skill_name", "").lower() for f in skill_facts]
        skill_text = " ".join(skills)
        
        # Frontend skills
        frontend_skills = ["react", "vue", "angular", "javascript", "typescript", "html", "css"]
        if any(skill in skill_text for skill in frontend_skills):
            return "frontend_developer"
        
        # Backend skills
        backend_skills = ["python", "java", "node", "spring", "django", "flask", "express"]
        if any(skill in skill_text for skill in backend_skills):
            return "backend_developer"
        
        # Full-stack indicators
        if len([s for s in skills if s in frontend_skills + backend_skills]) >= 3:
            return "fullstack_developer"
        
        # Data skills
        data_skills = ["python", "r", "sql", "pandas", "numpy", "tensorflow", "pytorch"]
        if any(skill in skill_text for skill in data_skills):
            return "data_scientist"
        
        # DevOps skills
        devops_skills = ["docker", "kubernetes", "aws", "azure", "jenkins", "terraform"]
        if any(skill in skill_text for skill in devops_skills):
            return "devops_engineer"
        
        return None
    
    async def _analyze_career_stage_from_experience(self, experience_facts: List[StructuredFact]) -> Optional[str]:
        """Analyze career stage from experience patterns"""
        if not experience_facts:
            return "entry_level"
        
        # Count years of experience
        total_years = 0
        for exp in experience_facts:
            duration = exp.content.get("duration", "")
            years = self._extract_years_from_duration(duration)
            total_years += years
        
        if total_years < 2:
            return "junior"
        elif total_years < 5:
            return "mid_level"
        elif total_years < 10:
            return "senior"
        else:
            return "lead"
    
    def _extract_years_from_duration(self, duration: str) -> int:
        """Extract years from duration string"""
        if not duration:
            return 0
        
        # Look for year patterns
        year_match = re.search(r'(\d+)\s*year', duration.lower())
        if year_match:
            return int(year_match.group(1))
        
        # Look for month patterns and convert to years
        month_match = re.search(r'(\d+)\s*month', duration.lower())
        if month_match:
            return int(month_match.group(1)) // 12
        
        return 0
    
    def _normalize_tag_value(self, value: str) -> str:
        """Normalize tag value"""
        return value.lower().replace(" ", "_").replace("-", "_")
    
    def _extract_seniority_level(self, title: str, description: str) -> str:
        """Extract seniority level from title and description"""
        text = f"{title} {description}".lower()
        
        if any(term in text for term in ["senior", "lead", "principal", "architect"]):
            return "senior_level"
        elif any(term in text for term in ["junior", "entry", "associate"]):
            return "junior_level"
        else:
            return "mid_level"
    
    def _extract_role_type(self, skill_name: str) -> str:
        """Extract role type from skill name"""
        skill_lower = skill_name.lower()
        
        if any(term in skill_lower for term in ["frontend", "react", "vue", "angular"]):
            return "frontend_developer"
        elif any(term in skill_lower for term in ["backend", "api", "server"]):
            return "backend_developer"
        elif any(term in skill_lower for term in ["fullstack", "full-stack"]):
            return "fullstack_developer"
        elif any(term in skill_lower for term in ["mobile", "ios", "android"]):
            return "mobile_developer"
        elif any(term in skill_lower for term in ["data", "ml", "ai"]):
            return "data_scientist"
        elif any(term in skill_lower for term in ["devops", "cloud", "aws"]):
            return "devops_engineer"
        else:
            return "software_developer"
    
    def _extract_industry(self, company: str, description: str) -> str:
        """Extract industry from company and description"""
        text = f"{company} {description}".lower()
        
        if any(term in text for term in ["fintech", "financial", "banking", "fintech"]):
            return "fintech"
        elif any(term in text for term in ["healthcare", "medical", "pharma"]):
            return "healthcare"
        elif any(term in text for term in ["ecommerce", "retail", "shopping"]):
            return "ecommerce"
        elif any(term in text for term in ["saas", "software", "technology"]):
            return "saas"
        else:
            return "technology"
    
    def _extract_work_style(self, description: str) -> str:
        """Extract work style from description"""
        desc_lower = description.lower()
        
        if any(term in desc_lower for term in ["startup", "early stage", "fast paced"]):
            return "startup_experience"
        elif any(term in desc_lower for term in ["enterprise", "large company", "corporate"]):
            return "enterprise_experience"
        elif any(term in desc_lower for term in ["agile", "scrum", "kanban"]):
            return "agile_methodology"
        else:
            return "collaborative"
    
    async def _validate_and_score_tag(self, tag: DynamicTag, context: TagContext) -> Optional[DynamicTag]:
        """Validate and score a tag"""
        # Calculate confidence score
        confidence_score = await self._calculate_tag_confidence(tag, context)
        tag.confidence = confidence_score
        
        # Validate tag
        if not await self._validate_tag(tag):
            log.warning(f"Tag validation failed for {tag.tag_id}")
            return None
        
        return tag
    
    async def _calculate_tag_confidence(self, tag: DynamicTag, context: TagContext) -> ConfidenceScore:
        """Calculate confidence score for a tag"""
        base_score = tag.confidence.value
        
        # Adjust based on related facts confidence
        if tag.related_facts:
            related_facts = [f for f in context.facts if f.fact_id in tag.related_facts]
            if related_facts:
                avg_fact_confidence = sum(f.confidence.value for f in related_facts) / len(related_facts)
                adjusted_score = base_score * (0.5 + 0.5 * avg_fact_confidence)
            else:
                adjusted_score = base_score * 0.7  # Lower confidence if facts not found
        else:
            adjusted_score = base_score * 0.8  # Lower confidence if no related facts
        
        return ConfidenceScore(
            value=min(1.0, adjusted_score),
            reasoning=f"Adjusted from {base_score:.2f} based on related facts confidence"
        )
    
    async def _validate_tag(self, tag: DynamicTag) -> bool:
        """Validate a tag"""
        # Check if tag value is not empty
        if not tag.value or not tag.value.strip():
            return False
        
        # Check if tag category is valid
        if not tag.category:
            return False
        
        # Check if confidence is reasonable
        if tag.confidence.value < 0.1:
            return False
        
        return True
    
    async def _deduplicate_tags(self, tags: List[DynamicTag]) -> List[DynamicTag]:
        """Remove duplicate tags"""
        if not tags:
            return tags
        
        deduplicated = []
        seen_tags = set()
        
        for tag in tags:
            # Create a signature for the tag
            signature = f"{tag.category.value}:{tag.value}"
            
            if signature not in seen_tags:
                seen_tags.add(signature)
                deduplicated.append(tag)
            else:
                # If duplicate, keep the one with higher confidence
                existing_tag = next(t for t in deduplicated if f"{t.category.value}:{t.value}" == signature)
                if tag.confidence.value > existing_tag.confidence.value:
                    deduplicated.remove(existing_tag)
                    deduplicated.append(tag)
        
        return deduplicated
    
    def get_tag_generation_stats(self) -> Dict[str, Any]:
        """Get tag generation statistics"""
        return {
            "tag_rules_count": sum(len(rules) for rules in self.tag_rules.values()),
            "tag_patterns_count": sum(len(patterns) for patterns in self.tag_patterns.values()),
            "confidence_weights": self.confidence_weights,
            "supported_categories": [tc.value for tc in TagCategory],
            "supported_sources": [ds.value for ds in DataSource]
        }


# ==================== SINGLETON INSTANCE ====================

# Global instance for the application
tag_generator = TagGenerator()
