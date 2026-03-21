"""
QualityGate - Data validation and completeness checks
Implements comprehensive quality assessment and validation for candidate data
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .candidate_matching_models import (
    CandidateProfile, StructuredFact, DynamicTag, MultiViewEmbedding,
    ConfidenceScore, Provenance, DataSource, FactType, TagCategory,
    EmbeddingView, QualityMetrics, ProcessingStage
)

log = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Validation levels"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"


class QualityThreshold(str, Enum):
    """Quality thresholds"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXCELLENT = "excellent"


@dataclass
class ValidationRule:
    """Validation rule configuration"""
    rule_id: str
    rule_type: str
    field_path: str
    validation_func: callable
    error_message: str
    warning_threshold: float
    error_threshold: float
    required: bool = False


@dataclass
class QualityCheck:
    """Quality check configuration"""
    check_id: str
    check_name: str
    check_func: callable
    weight: float
    threshold: float
    description: str


class QualityGate:
    """
    Comprehensive quality assessment and validation for candidate data
    Implements multi-level validation with configurable thresholds
    """
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
        self.quality_checks = self._initialize_quality_checks()
        self.quality_thresholds = self._initialize_quality_thresholds()
        self.validation_levels = self._initialize_validation_levels()
        
    def _initialize_validation_rules(self) -> List[ValidationRule]:
        """Initialize validation rules"""
        return [
            # Basic field validation
            ValidationRule(
                rule_id="candidate_id_required",
                rule_type="required_field",
                field_path="candidate_id",
                validation_func=self._validate_candidate_id,
                error_message="Candidate ID is required",
                warning_threshold=0.0,
                error_threshold=0.0,
                required=True
            ),
            ValidationRule(
                rule_id="uid_required",
                rule_type="required_field",
                field_path="uid",
                validation_func=self._validate_uid,
                error_message="UID is required",
                warning_threshold=0.0,
                error_threshold=0.0,
                required=True
            ),
            ValidationRule(
                rule_id="tenant_id_required",
                rule_type="required_field",
                field_path="tenant_id",
                validation_func=self._validate_tenant_id,
                error_message="Tenant ID is required",
                warning_threshold=0.0,
                error_threshold=0.0,
                required=True
            ),
            
            # Data completeness validation
            ValidationRule(
                rule_id="facts_present",
                rule_type="completeness",
                field_path="facts",
                validation_func=self._validate_facts_present,
                error_message="At least one fact is required",
                warning_threshold=0.5,
                error_threshold=0.0,
                required=True
            ),
            ValidationRule(
                rule_id="tags_present",
                rule_type="completeness",
                field_path="tags",
                validation_func=self._validate_tags_present,
                error_message="At least one tag is recommended",
                warning_threshold=0.3,
                error_threshold=0.0,
                required=False
            ),
            ValidationRule(
                rule_id="embeddings_present",
                rule_type="completeness",
                field_path="embeddings",
                validation_func=self._validate_embeddings_present,
                error_message="At least one embedding is required",
                warning_threshold=0.5,
                error_threshold=0.0,
                required=True
            ),
            
            # Confidence validation
            ValidationRule(
                rule_id="fact_confidence",
                rule_type="confidence",
                field_path="facts.confidence",
                validation_func=self._validate_fact_confidence,
                error_message="Fact confidence too low",
                warning_threshold=0.3,
                error_threshold=0.1,
                required=True
            ),
            ValidationRule(
                rule_id="tag_confidence",
                rule_type="confidence",
                field_path="tags.confidence",
                validation_func=self._validate_tag_confidence,
                error_message="Tag confidence too low",
                warning_threshold=0.2,
                error_threshold=0.1,
                required=False
            ),
            ValidationRule(
                rule_id="embedding_confidence",
                rule_type="confidence",
                field_path="embeddings.confidence",
                validation_func=self._validate_embedding_confidence,
                error_message="Embedding confidence too low",
                warning_threshold=0.4,
                error_threshold=0.2,
                required=True
            ),
            
            # Data quality validation
            ValidationRule(
                rule_id="fact_content_quality",
                rule_type="content_quality",
                field_path="facts.content",
                validation_func=self._validate_fact_content_quality,
                error_message="Fact content quality issues",
                warning_threshold=0.6,
                error_threshold=0.3,
                required=True
            ),
            ValidationRule(
                rule_id="tag_content_quality",
                rule_type="content_quality",
                field_path="tags.content",
                validation_func=self._validate_tag_content_quality,
                error_message="Tag content quality issues",
                warning_threshold=0.5,
                error_threshold=0.2,
                required=False
            ),
            ValidationRule(
                rule_id="embedding_vector_quality",
                rule_type="vector_quality",
                field_path="embeddings.vector",
                validation_func=self._validate_embedding_vector_quality,
                error_message="Embedding vector quality issues",
                warning_threshold=0.7,
                error_threshold=0.4,
                required=True
            )
        ]
    
    def _initialize_quality_checks(self) -> List[QualityCheck]:
        """Initialize quality checks"""
        return [
            QualityCheck(
                check_id="completeness_check",
                check_name="Data Completeness",
                check_func=self._check_completeness,
                weight=0.3,
                threshold=0.7,
                description="Checks if all required data components are present"
            ),
            QualityCheck(
                check_id="confidence_check",
                check_name="Confidence Assessment",
                check_func=self._check_confidence,
                weight=0.25,
                threshold=0.6,
                description="Assesses confidence levels across all data"
            ),
            QualityCheck(
                check_id="consistency_check",
                check_name="Data Consistency",
                check_func=self._check_consistency,
                weight=0.2,
                threshold=0.8,
                description="Checks consistency between different data sources"
            ),
            QualityCheck(
                check_id="freshness_check",
                check_name="Data Freshness",
                check_func=self._check_freshness,
                weight=0.15,
                threshold=0.5,
                description="Assesses how recent the data is"
            ),
            QualityCheck(
                check_id="accuracy_check",
                check_name="Data Accuracy",
                check_func=self._check_accuracy,
                weight=0.1,
                threshold=0.7,
                description="Validates accuracy of data content"
            )
        ]
    
    def _initialize_quality_thresholds(self) -> Dict[QualityThreshold, Dict[str, float]]:
        """Initialize quality thresholds"""
        return {
            QualityThreshold.LOW: {
                "overall_quality": 0.3,
                "completeness": 0.4,
                "confidence": 0.3,
                "consistency": 0.5,
                "freshness": 0.3,
                "accuracy": 0.4
            },
            QualityThreshold.MEDIUM: {
                "overall_quality": 0.5,
                "completeness": 0.6,
                "confidence": 0.5,
                "consistency": 0.7,
                "freshness": 0.5,
                "accuracy": 0.6
            },
            QualityThreshold.HIGH: {
                "overall_quality": 0.7,
                "completeness": 0.8,
                "confidence": 0.7,
                "consistency": 0.8,
                "freshness": 0.7,
                "accuracy": 0.8
            },
            QualityThreshold.EXCELLENT: {
                "overall_quality": 0.9,
                "completeness": 0.9,
                "confidence": 0.8,
                "consistency": 0.9,
                "freshness": 0.8,
                "accuracy": 0.9
            }
        }
    
    def _initialize_validation_levels(self) -> Dict[ValidationLevel, List[str]]:
        """Initialize validation levels"""
        return {
            ValidationLevel.BASIC: [
                "candidate_id_required",
                "uid_required",
                "tenant_id_required"
            ],
            ValidationLevel.STANDARD: [
                "candidate_id_required",
                "uid_required",
                "tenant_id_required",
                "facts_present",
                "embeddings_present",
                "fact_confidence",
                "embedding_confidence"
            ],
            ValidationLevel.STRICT: [
                "candidate_id_required",
                "uid_required",
                "tenant_id_required",
                "facts_present",
                "tags_present",
                "embeddings_present",
                "fact_confidence",
                "tag_confidence",
                "embedding_confidence",
                "fact_content_quality",
                "embedding_vector_quality"
            ],
            ValidationLevel.COMPREHENSIVE: [
                rule.rule_id for rule in self.validation_rules
            ]
        }
    
    async def validate_candidate_profile(self, profile: CandidateProfile, 
                                       validation_level: ValidationLevel = ValidationLevel.STANDARD) -> QualityMetrics:
        """
        Validate candidate profile and return quality metrics
        """
        log.info(f"Validating candidate profile {profile.candidate_id} with level {validation_level}")
        
        # Get applicable validation rules
        applicable_rules = self.validation_levels[validation_level]
        rules_to_apply = [rule for rule in self.validation_rules if rule.rule_id in applicable_rules]
        
        # Run validation rules
        validation_results = await self._run_validation_rules(profile, rules_to_apply)
        
        # Run quality checks
        quality_results = await self._run_quality_checks(profile)
        
        # Calculate overall quality metrics
        quality_metrics = await self._calculate_quality_metrics(profile, validation_results, quality_results)
        
        log.info(f"Validation completed for candidate {profile.candidate_id}: {quality_metrics.overall_quality:.2f}")
        return quality_metrics
    
    async def _run_validation_rules(self, profile: CandidateProfile, rules: List[ValidationRule]) -> Dict[str, Any]:
        """Run validation rules"""
        results = {
            "passed": [],
            "warnings": [],
            "errors": [],
            "skipped": []
        }
        
        for rule in rules:
            try:
                # Get field value
                field_value = self._get_field_value(profile, rule.field_path)
                
                # Run validation
                validation_result = await rule.validation_func(field_value, rule)
                
                if validation_result["status"] == "passed":
                    results["passed"].append({
                        "rule_id": rule.rule_id,
                        "message": validation_result["message"]
                    })
                elif validation_result["status"] == "warning":
                    results["warnings"].append({
                        "rule_id": rule.rule_id,
                        "message": validation_result["message"],
                        "score": validation_result["score"]
                    })
                elif validation_result["status"] == "error":
                    results["errors"].append({
                        "rule_id": rule.rule_id,
                        "message": validation_result["message"],
                        "score": validation_result["score"]
                    })
                else:
                    results["skipped"].append({
                        "rule_id": rule.rule_id,
                        "message": validation_result["message"]
                    })
                    
            except Exception as e:
                log.error(f"Error running validation rule {rule.rule_id}: {e}")
                results["errors"].append({
                    "rule_id": rule.rule_id,
                    "message": f"Validation error: {str(e)}",
                    "score": 0.0
                })
        
        return results
    
    async def _run_quality_checks(self, profile: CandidateProfile) -> Dict[str, Any]:
        """Run quality checks"""
        results = {}
        
        for check in self.quality_checks:
            try:
                check_result = await check.check_func(profile)
                results[check.check_id] = {
                    "name": check.check_name,
                    "score": check_result["score"],
                    "passed": check_result["score"] >= check.threshold,
                    "weight": check.weight,
                    "threshold": check.threshold,
                    "details": check_result.get("details", {})
                }
            except Exception as e:
                log.error(f"Error running quality check {check.check_id}: {e}")
                results[check.check_id] = {
                    "name": check.check_name,
                    "score": 0.0,
                    "passed": False,
                    "weight": check.weight,
                    "threshold": check.threshold,
                    "details": {"error": str(e)}
                }
        
        return results
    
    async def _calculate_quality_metrics(self, profile: CandidateProfile, 
                                       validation_results: Dict[str, Any],
                                       quality_results: Dict[str, Any]) -> QualityMetrics:
        """Calculate overall quality metrics"""
        
        # Calculate completeness score
        completeness_score = await self._check_completeness(profile)["score"]
        
        # Calculate accuracy score
        accuracy_score = await self._check_accuracy(profile)["score"]
        
        # Calculate consistency score
        consistency_score = await self._check_consistency(profile)["score"]
        
        # Calculate freshness score
        freshness_score = await self._check_freshness(profile)["score"]
        
        # Calculate overall quality
        overall_quality = (
            completeness_score * 0.3 +
            accuracy_score * 0.25 +
            consistency_score * 0.2 +
            freshness_score * 0.15 +
            quality_results.get("confidence_check", {}).get("score", 0.0) * 0.1
        )
        
        # Identify issues
        missing_fields = await self._identify_missing_fields(profile)
        inconsistent_fields = await self._identify_inconsistent_fields(profile, validation_results)
        low_confidence_facts = [f.fact_id for f in profile.facts if f.confidence.value < 0.5]
        
        # Generate improvement suggestions
        improvement_suggestions = await self._generate_improvement_suggestions(
            profile, validation_results, quality_results
        )
        
        return QualityMetrics(
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            freshness_score=freshness_score,
            overall_quality=overall_quality,
            missing_fields=missing_fields,
            inconsistent_fields=inconsistent_fields,
            low_confidence_facts=low_confidence_facts,
            improvement_suggestions=improvement_suggestions
        )
    
    def _get_field_value(self, profile: CandidateProfile, field_path: str) -> Any:
        """Get field value from profile using dot notation"""
        parts = field_path.split('.')
        value = profile
        
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            elif isinstance(value, dict) and part in value:
                value = value[part]
            elif isinstance(value, list) and part.isdigit():
                value = value[int(part)]
            else:
                return None
        
        return value
    
    # ==================== VALIDATION FUNCTIONS ====================
    
    async def _validate_candidate_id(self, value: Any, rule: ValidationRule) -> Dict[str, Any]:
        """Validate candidate ID"""
        if not value or not isinstance(value, str) or not value.strip():
            return {"status": "error", "message": rule.error_message, "score": 0.0}
        
        return {"status": "passed", "message": "Candidate ID is valid", "score": 1.0}
    
    async def _validate_uid(self, value: Any, rule: ValidationRule) -> Dict[str, Any]:
        """Validate UID"""
        if not value or not isinstance(value, str) or not value.strip():
            return {"status": "error", "message": rule.error_message, "score": 0.0}
        
        return {"status": "passed", "message": "UID is valid", "score": 1.0}
    
    async def _validate_tenant_id(self, value: Any, rule: ValidationRule) -> Dict[str, Any]:
        """Validate tenant ID"""
        if not value or not isinstance(value, str) or not value.strip():
            return {"status": "error", "message": rule.error_message, "score": 0.0}
        
        return {"status": "passed", "message": "Tenant ID is valid", "score": 1.0}
    
    async def _validate_facts_present(self, value: Any, rule: ValidationRule) -> Dict[str, Any]:
        """Validate facts are present"""
        if not value or not isinstance(value, list) or len(value) == 0:
            return {"status": "error", "message": rule.error_message, "score": 0.0}
        
        fact_count = len(value)
        if fact_count < 3:
            return {"status": "warning", "message": f"Only {fact_count} facts present", "score": fact_count / 5.0}
        
        return {"status": "passed", "message": f"{fact_count} facts present", "score": min(1.0, fact_count / 10.0)}
    
    async def _validate_tags_present(self, value: Any, rule: ValidationRule) -> Dict[str, Any]:
        """Validate tags are present"""
        if not value or not isinstance(value, list) or len(value) == 0:
            return {"status": "warning", "message": rule.error_message, "score": 0.0}
        
        tag_count = len(value)
        return {"status": "passed", "message": f"{tag_count} tags present", "score": min(1.0, tag_count / 5.0)}
    
    async def _validate_embeddings_present(self, value: Any, rule: ValidationRule) -> Dict[str, Any]:
        """Validate embeddings are present"""
        if not value or not isinstance(value, list) or len(value) == 0:
            return {"status": "error", "message": rule.error_message, "score": 0.0}
        
        embedding_count = len(value)
        return {"status": "passed", "message": f"{embedding_count} embeddings present", "score": min(1.0, embedding_count / 3.0)}
    
    async def _validate_fact_confidence(self, value: Any, rule: ValidationRule) -> Dict[str, Any]:
        """Validate fact confidence"""
        if not value or not isinstance(value, list):
            return {"status": "error", "message": rule.error_message, "score": 0.0}
        
        low_confidence_count = sum(1 for fact in value if fact.confidence.value < rule.warning_threshold)
        total_count = len(value)
        
        if low_confidence_count == total_count:
            return {"status": "error", "message": rule.error_message, "score": 0.0}
        elif low_confidence_count > total_count * 0.5:
            return {"status": "warning", "message": f"{low_confidence_count} facts have low confidence", "score": 0.5}
        
        avg_confidence = sum(fact.confidence.value for fact in value) / total_count
        return {"status": "passed", "message": f"Average fact confidence: {avg_confidence:.2f}", "score": avg_confidence}
    
    async def _validate_tag_confidence(self, value: Any, rule: ValidationRule) -> Dict[str, Any]:
        """Validate tag confidence"""
        if not value or not isinstance(value, list):
            return {"status": "warning", "message": rule.error_message, "score": 0.0}
        
        low_confidence_count = sum(1 for tag in value if tag.confidence.value < rule.warning_threshold)
        total_count = len(value)
        
        if low_confidence_count > total_count * 0.7:
            return {"status": "warning", "message": f"{low_confidence_count} tags have low confidence", "score": 0.3}
        
        avg_confidence = sum(tag.confidence.value for tag in value) / total_count
        return {"status": "passed", "message": f"Average tag confidence: {avg_confidence:.2f}", "score": avg_confidence}
    
    async def _validate_embedding_confidence(self, value: Any, rule: ValidationRule) -> Dict[str, Any]:
        """Validate embedding confidence"""
        if not value or not isinstance(value, list):
            return {"status": "error", "message": rule.error_message, "score": 0.0}
        
        low_confidence_count = sum(1 for emb in value if emb.confidence.value < rule.warning_threshold)
        total_count = len(value)
        
        if low_confidence_count == total_count:
            return {"status": "error", "message": rule.error_message, "score": 0.0}
        elif low_confidence_count > total_count * 0.5:
            return {"status": "warning", "message": f"{low_confidence_count} embeddings have low confidence", "score": 0.4}
        
        avg_confidence = sum(emb.confidence.value for emb in value) / total_count
        return {"status": "passed", "message": f"Average embedding confidence: {avg_confidence:.2f}", "score": avg_confidence}
    
    async def _validate_fact_content_quality(self, value: Any, rule: ValidationRule) -> Dict[str, Any]:
        """Validate fact content quality"""
        if not value or not isinstance(value, list):
            return {"status": "error", "message": rule.error_message, "score": 0.0}
        
        quality_scores = []
        for fact in value:
            content = fact.content
            if not content:
                quality_scores.append(0.0)
                continue
            
            # Check if required fields are present
            required_fields = self._get_required_fields_for_fact_type(fact.fact_type)
            present_fields = sum(1 for field in required_fields if content.get(field))
            field_score = present_fields / len(required_fields) if required_fields else 1.0
            
            quality_scores.append(field_score)
        
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        if avg_quality < rule.error_threshold:
            return {"status": "error", "message": rule.error_message, "score": avg_quality}
        elif avg_quality < rule.warning_threshold:
            return {"status": "warning", "message": f"Fact content quality: {avg_quality:.2f}", "score": avg_quality}
        
        return {"status": "passed", "message": f"Fact content quality: {avg_quality:.2f}", "score": avg_quality}
    
    async def _validate_tag_content_quality(self, value: Any, rule: ValidationRule) -> Dict[str, Any]:
        """Validate tag content quality"""
        if not value or not isinstance(value, list):
            return {"status": "warning", "message": rule.error_message, "score": 0.0}
        
        quality_scores = []
        for tag in value:
            if not tag.value or not tag.value.strip():
                quality_scores.append(0.0)
            else:
                quality_scores.append(1.0)
        
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        if avg_quality < rule.error_threshold:
            return {"status": "error", "message": rule.error_message, "score": avg_quality}
        elif avg_quality < rule.warning_threshold:
            return {"status": "warning", "message": f"Tag content quality: {avg_quality:.2f}", "score": avg_quality}
        
        return {"status": "passed", "message": f"Tag content quality: {avg_quality:.2f}", "score": avg_quality}
    
    async def _validate_embedding_vector_quality(self, value: Any, rule: ValidationRule) -> Dict[str, Any]:
        """Validate embedding vector quality"""
        if not value or not isinstance(value, list):
            return {"status": "error", "message": rule.error_message, "score": 0.0}
        
        quality_scores = []
        for embedding in value:
            vector = embedding.vector
            if not vector or len(vector) == 0:
                quality_scores.append(0.0)
                continue
            
            # Check vector properties
            vector_norm = sum(x**2 for x in vector)**0.5
            if vector_norm == 0:
                quality_scores.append(0.0)
            else:
                # Normalize and check for reasonable values
                normalized_vector = [x / vector_norm for x in vector]
                quality_score = min(1.0, vector_norm / 10.0)  # Simple quality metric
                quality_scores.append(quality_score)
        
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        if avg_quality < rule.error_threshold:
            return {"status": "error", "message": rule.error_message, "score": avg_quality}
        elif avg_quality < rule.warning_threshold:
            return {"status": "warning", "message": f"Embedding vector quality: {avg_quality:.2f}", "score": avg_quality}
        
        return {"status": "passed", "message": f"Embedding vector quality: {avg_quality:.2f}", "score": avg_quality}
    
    def _get_required_fields_for_fact_type(self, fact_type: FactType) -> List[str]:
        """Get required fields for a fact type"""
        field_mapping = {
            FactType.SKILL: ["skill_name"],
            FactType.EXPERIENCE: ["title", "company"],
            FactType.EDUCATION: ["degree", "university"],
            FactType.CERTIFICATION: ["certification_name"],
            FactType.PROJECT: ["project_name"],
            FactType.ACHIEVEMENT: ["achievement_title"],
            FactType.INTEREST: ["interest_name"],
            FactType.LOCATION: ["location_name"],
            FactType.LANGUAGE: ["language_name", "proficiency"],
            FactType.SOFT_SKILL: ["skill_name"]
        }
        
        return field_mapping.get(fact_type, [])
    
    # ==================== QUALITY CHECK FUNCTIONS ====================
    
    async def _check_completeness(self, profile: CandidateProfile) -> Dict[str, Any]:
        """Check data completeness"""
        components = {
            "facts": len(profile.facts) > 0,
            "tags": len(profile.tags) > 0,
            "embeddings": len(profile.embeddings) > 0,
            "profile_data": bool(profile.profile_data)
        }
        
        completeness_score = sum(components.values()) / len(components)
        
        return {
            "score": completeness_score,
            "details": {
                "components": components,
                "missing_components": [k for k, v in components.items() if not v]
            }
        }
    
    async def _check_confidence(self, profile: CandidateProfile) -> Dict[str, Any]:
        """Check confidence levels"""
        all_items = profile.facts + profile.tags + profile.embeddings
        
        if not all_items:
            return {"score": 0.0, "details": {"error": "No data items to check"}}
        
        confidence_scores = [item.confidence.value for item in all_items]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        low_confidence_count = sum(1 for score in confidence_scores if score < 0.5)
        
        return {
            "score": avg_confidence,
            "details": {
                "average_confidence": avg_confidence,
                "low_confidence_count": low_confidence_count,
                "total_items": len(all_items)
            }
        }
    
    async def _check_consistency(self, profile: CandidateProfile) -> Dict[str, Any]:
        """Check data consistency"""
        consistency_checks = []
        
        # Check fact consistency
        if profile.facts:
            fact_consistency = await self._check_fact_consistency(profile.facts)
            consistency_checks.append(fact_consistency)
        
        # Check tag consistency
        if profile.tags:
            tag_consistency = await self._check_tag_consistency(profile.tags)
            consistency_checks.append(tag_consistency)
        
        if not consistency_checks:
            return {"score": 1.0, "details": {"message": "No consistency checks available"}}
        
        avg_consistency = sum(consistency_checks) / len(consistency_checks)
        
        return {
            "score": avg_consistency,
            "details": {
                "consistency_checks": consistency_checks,
                "average_consistency": avg_consistency
            }
        }
    
    async def _check_freshness(self, profile: CandidateProfile) -> Dict[str, Any]:
        """Check data freshness"""
        now = datetime.utcnow()
        freshness_scores = []
        
        for fact in profile.facts:
            age_hours = (now - fact.provenance.timestamp).total_seconds() / 3600
            freshness = max(0.0, 1.0 - (age_hours / (24 * 30)))  # Decay over 30 days
            freshness_scores.append(freshness)
        
        if not freshness_scores:
            return {"score": 1.0, "details": {"message": "No timestamp data available"}}
        
        avg_freshness = sum(freshness_scores) / len(freshness_scores)
        
        return {
            "score": avg_freshness,
            "details": {
                "average_freshness": avg_freshness,
                "oldest_data_hours": max((now - fact.provenance.timestamp).total_seconds() / 3600 for fact in profile.facts) if profile.facts else 0
            }
        }
    
    async def _check_accuracy(self, profile: CandidateProfile) -> Dict[str, Any]:
        """Check data accuracy"""
        accuracy_checks = []
        
        # Check fact accuracy
        for fact in profile.facts:
            accuracy_score = await self._check_fact_accuracy(fact)
            accuracy_checks.append(accuracy_score)
        
        if not accuracy_checks:
            return {"score": 1.0, "details": {"message": "No accuracy checks available"}}
        
        avg_accuracy = sum(accuracy_checks) / len(accuracy_checks)
        
        return {
            "score": avg_accuracy,
            "details": {
                "accuracy_checks": accuracy_checks,
                "average_accuracy": avg_accuracy
            }
        }
    
    async def _check_fact_consistency(self, facts: List[StructuredFact]) -> float:
        """Check consistency between facts"""
        # Simple consistency check - can be enhanced
        return 0.8  # Placeholder
    
    async def _check_tag_consistency(self, tags: List[DynamicTag]) -> float:
        """Check consistency between tags"""
        # Simple consistency check - can be enhanced
        return 0.8  # Placeholder
    
    async def _check_fact_accuracy(self, fact: StructuredFact) -> float:
        """Check accuracy of a single fact"""
        # Simple accuracy check - can be enhanced
        return fact.confidence.value  # Use confidence as proxy for accuracy
    
    async def _identify_missing_fields(self, profile: CandidateProfile) -> List[str]:
        """Identify missing fields"""
        missing_fields = []
        
        if not profile.facts:
            missing_fields.append("facts")
        if not profile.tags:
            missing_fields.append("tags")
        if not profile.embeddings:
            missing_fields.append("embeddings")
        if not profile.profile_data:
            missing_fields.append("profile_data")
        
        return missing_fields
    
    async def _identify_inconsistent_fields(self, profile: CandidateProfile, validation_results: Dict[str, Any]) -> List[str]:
        """Identify inconsistent fields"""
        inconsistent_fields = []
        
        # Check for consistency warnings
        for warning in validation_results.get("warnings", []):
            if "consistency" in warning["rule_id"]:
                inconsistent_fields.append(warning["rule_id"])
        
        return inconsistent_fields
    
    async def _generate_improvement_suggestions(self, profile: CandidateProfile, 
                                              validation_results: Dict[str, Any],
                                              quality_results: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        # Based on validation results
        if validation_results.get("errors"):
            suggestions.append("Fix validation errors to improve data quality")
        
        if validation_results.get("warnings"):
            suggestions.append("Address validation warnings for better data quality")
        
        # Based on quality checks
        for check_id, result in quality_results.items():
            if not result["passed"]:
                suggestions.append(f"Improve {result['name'].lower()} (current: {result['score']:.2f}, target: {result['threshold']:.2f})")
        
        # Based on missing fields
        missing_fields = await self._identify_missing_fields(profile)
        if missing_fields:
            suggestions.append(f"Add missing data components: {', '.join(missing_fields)}")
        
        # Based on low confidence
        low_confidence_facts = [f for f in profile.facts if f.confidence.value < 0.5]
        if low_confidence_facts:
            suggestions.append(f"Improve confidence for {len(low_confidence_facts)} low-confidence facts")
        
        return suggestions
    
    def get_quality_gate_stats(self) -> Dict[str, Any]:
        """Get quality gate statistics"""
        return {
            "validation_rules_count": len(self.validation_rules),
            "quality_checks_count": len(self.quality_checks),
            "validation_levels": {level.value: len(rules) for level, rules in self.validation_levels.items()},
            "quality_thresholds": {threshold.value: thresholds for threshold, thresholds in self.quality_thresholds.items()}
        }


# ==================== SINGLETON INSTANCE ====================

# Global instance for the application
quality_gate = QualityGate()
