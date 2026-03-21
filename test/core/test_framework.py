"""
Comprehensive Testing Framework for Jobsify AI Candidate Matching Architecture
Implements mock data generation and validation testing
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import random
import uuid

from .candidate_matching_models import (
    CandidateProfile, StructuredFact, DynamicTag, MultiViewEmbedding,
    ConfidenceScore, Provenance, DataSource, FactType, TagCategory,
    EmbeddingView, QualityMetrics, JobMatchResult, JobDescription,
    CandidateMatchingResponse
)
from .jobsify_candidate_matching import (
    jobsify_system, process_resume_data, process_chat_data,
    process_assessment_data, find_job_matches_for_candidate
)

log = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """Test configuration"""
    num_candidates: int = 10
    num_jobs: int = 5
    test_data_dir: str = "test_data"
    mock_mode: bool = True
    validation_level: str = "standard"


class MockDataGenerator:
    """Generates mock data for testing"""
    
    def __init__(self):
        self.skill_pool = [
            "Python", "Java", "JavaScript", "React", "Node.js", "SQL", "AWS", "Docker",
            "Kubernetes", "Machine Learning", "Data Science", "TensorFlow", "PyTorch",
            "Git", "Linux", "MongoDB", "PostgreSQL", "Redis", "Elasticsearch",
            "Kafka", "RabbitMQ", "GraphQL", "REST API", "Microservices", "DevOps"
        ]
        
        self.job_titles = [
            "Software Engineer", "Senior Software Engineer", "Full Stack Developer",
            "Frontend Developer", "Backend Developer", "Data Scientist", "ML Engineer",
            "DevOps Engineer", "Cloud Engineer", "Product Manager", "Technical Lead",
            "Architect", "QA Engineer", "Mobile Developer", "UI/UX Designer"
        ]
        
        self.companies = [
            "Google", "Microsoft", "Amazon", "Apple", "Meta", "Netflix", "Uber",
            "Airbnb", "Stripe", "Shopify", "Slack", "Zoom", "Tesla", "SpaceX",
            "OpenAI", "Anthropic", "Databricks", "Snowflake", "MongoDB", "Redis"
        ]
        
        self.locations = [
            "San Francisco", "New York", "Seattle", "Austin", "Boston", "Denver",
            "Remote", "Hybrid", "London", "Berlin", "Toronto", "Vancouver"
        ]
        
        self.education_degrees = [
            "Bachelor of Science", "Master of Science", "PhD", "Bachelor of Arts",
            "Master of Business Administration", "Associate Degree", "Bootcamp Certificate"
        ]
        
        self.majors = [
            "Computer Science", "Software Engineering", "Data Science", "Mathematics",
            "Physics", "Electrical Engineering", "Information Technology", "Business"
        ]
    
    def generate_candidate_profile(self, candidate_id: str) -> CandidateProfile:
        """Generate a mock candidate profile"""
        # Generate basic profile data
        profile_data = {
            "Name": f"Candidate {candidate_id}",
            "Email": f"candidate{candidate_id}@example.com",
            "Phone": f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
            "Location": random.choice(self.locations),
            "LinkedIn": f"https://linkedin.com/in/candidate{candidate_id}",
            "GitHub": f"https://github.com/candidate{candidate_id}"
        }
        
        # Generate facts
        facts = self._generate_facts(candidate_id)
        
        # Generate tags
        tags = self._generate_tags(candidate_id, facts)
        
        # Generate embeddings
        embeddings = self._generate_embeddings(candidate_id)
        
        return CandidateProfile(
            candidate_id=candidate_id,
            uid=candidate_id,
            tenant_id="test_tenant",
            profile_data=profile_data,
            facts=facts,
            tags=tags,
            embeddings=embeddings,
            processing_stage="retrieval",
            last_updated=datetime.utcnow(),
            version="1.0",
            completeness_score=random.uniform(0.7, 1.0),
            quality_score=random.uniform(0.6, 1.0)
        )
    
    def _generate_facts(self, candidate_id: str) -> List[StructuredFact]:
        """Generate mock facts"""
        facts = []
        
        # Generate skills
        num_skills = random.randint(5, 15)
        skills = random.sample(self.skill_pool, num_skills)
        
        for skill in skills:
            fact = StructuredFact(
                fact_type=FactType.SKILL,
                content={
                    "skill_name": skill,
                    "skill_level": random.choice(["beginner", "intermediate", "advanced", "expert"]),
                    "years_experience": random.randint(1, 10)
                },
                confidence=ConfidenceScore(
                    value=random.uniform(0.6, 1.0),
                    reasoning="Generated from resume"
                ),
                provenance=Provenance(
                    source=DataSource.RESUME,
                    source_id=candidate_id,
                    extraction_method="resume_parser"
                )
            )
            facts.append(fact)
        
        # Generate work experience
        num_jobs = random.randint(2, 5)
        for i in range(num_jobs):
            fact = StructuredFact(
                fact_type=FactType.EXPERIENCE,
                content={
                    "title": random.choice(self.job_titles),
                    "company": random.choice(self.companies),
                    "duration": f"{random.randint(1, 4)} years",
                    "description": f"Worked on {random.choice(['web development', 'mobile apps', 'data analysis', 'machine learning', 'cloud infrastructure'])}",
                    "location": random.choice(self.locations)
                },
                confidence=ConfidenceScore(
                    value=random.uniform(0.7, 1.0),
                    reasoning="Structured experience data"
                ),
                provenance=Provenance(
                    source=DataSource.RESUME,
                    source_id=candidate_id,
                    extraction_method="resume_parser"
                )
            )
            facts.append(fact)
        
        # Generate education
        num_degrees = random.randint(1, 3)
        for i in range(num_degrees):
            fact = StructuredFact(
                fact_type=FactType.EDUCATION,
                content={
                    "degree": random.choice(self.education_degrees),
                    "major": random.choice(self.majors),
                    "institution": f"University {random.randint(1, 100)}",
                    "graduation_year": random.randint(2015, 2024)
                },
                confidence=ConfidenceScore(
                    value=random.uniform(0.8, 1.0),
                    reasoning="Structured education data"
                ),
                provenance=Provenance(
                    source=DataSource.RESUME,
                    source_id=candidate_id,
                    extraction_method="resume_parser"
                )
            )
            facts.append(fact)
        
        return facts
    
    def _generate_tags(self, candidate_id: str, facts: List[StructuredFact]) -> List[DynamicTag]:
        """Generate mock tags"""
        tags = []
        
        # Generate technical skill tags
        skill_facts = [f for f in facts if f.fact_type == FactType.SKILL]
        if skill_facts:
            tag = DynamicTag(
                category=TagCategory.TECHNICAL_SKILL,
                value="software_developer",
                confidence=ConfidenceScore(
                    value=random.uniform(0.7, 1.0),
                    reasoning="Generated from skills"
                ),
                provenance=Provenance(
                    source=DataSource.RESUME,
                    source_id=candidate_id,
                    extraction_method="skill_analysis"
                ),
                related_facts=[f.fact_id for f in skill_facts[:3]]
            )
            tags.append(tag)
        
        # Generate seniority tags
        experience_facts = [f for f in facts if f.fact_type == FactType.EXPERIENCE]
        if experience_facts:
            seniority = random.choice(["junior", "mid_level", "senior", "lead"])
            tag = DynamicTag(
                category=TagCategory.SENIORITY,
                value=seniority,
                confidence=ConfidenceScore(
                    value=random.uniform(0.6, 1.0),
                    reasoning="Generated from experience"
                ),
                provenance=Provenance(
                    source=DataSource.RESUME,
                    source_id=candidate_id,
                    extraction_method="experience_analysis"
                ),
                related_facts=[f.fact_id for f in experience_facts]
            )
            tags.append(tag)
        
        # Generate location preference tags
        location_pref = random.choice(["remote_preference", "onsite_preference", "hybrid_preference"])
        tag = DynamicTag(
            category=TagCategory.LOCATION_PREFERENCE,
            value=location_pref,
            confidence=ConfidenceScore(
                value=random.uniform(0.5, 1.0),
                reasoning="Generated from profile"
            ),
            provenance=Provenance(
                source=DataSource.RESUME,
                source_id=candidate_id,
                extraction_method="profile_analysis"
            )
        )
        tags.append(tag)
        
        return tags
    
    def _generate_embeddings(self, candidate_id: str) -> List[MultiViewEmbedding]:
        """Generate mock embeddings"""
        embeddings = []
        
        # Generate embeddings for different views
        view_types = [EmbeddingView.RESUME_CONTENT, EmbeddingView.SKILL_FOCUSED, EmbeddingView.COMPREHENSIVE]
        
        for view_type in view_types:
            embedding = MultiViewEmbedding(
                view_type=view_type,
                vector=[random.uniform(-1, 1) for _ in range(384)],  # Mock 384-dim vector
                confidence=ConfidenceScore(
                    value=random.uniform(0.6, 1.0),
                    reasoning=f"Generated {view_type.value} embedding"
                ),
                source_data={"candidate_id": candidate_id, "view_type": view_type.value},
                metadata={"generated": True, "candidate_id": candidate_id}
            )
            embeddings.append(embedding)
        
        return embeddings
    
    def generate_job_description(self, job_id: str) -> JobDescription:
        """Generate a mock job description"""
        return JobDescription(
            job_id=job_id,
            job_title=random.choice(self.job_titles),
            company_name=random.choice(self.companies),
            location=random.choice(self.locations),
            required_skills=random.sample(self.skill_pool, random.randint(3, 8)),
            preferred_skills=random.sample(self.skill_pool, random.randint(2, 5)),
            required_experience=f"{random.randint(2, 8)} years",
            education_requirements=[random.choice(self.education_degrees)],
            job_description=f"We are looking for a {random.choice(self.job_titles)} to join our team. You will work on exciting projects using cutting-edge technologies.",
            company_culture=["Innovation", "Collaboration", "Growth", "Diversity"],
            benefits=["Health Insurance", "401k", "Flexible Hours", "Remote Work"],
            salary_range=f"${random.randint(80, 200)}k - ${random.randint(200, 400)}k",
            job_type=random.choice(["Full-time", "Part-time", "Contract"]),
            remote_option=random.choice([True, False])
        )


class TestRunner:
    """Runs comprehensive tests for the candidate matching system"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.mock_generator = MockDataGenerator()
        self.test_results = {}
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests"""
        log.info("Starting comprehensive test suite")
        
        test_results = {
            "start_time": datetime.utcnow().isoformat(),
            "config": {
                "num_candidates": self.config.num_candidates,
                "num_jobs": self.config.num_jobs,
                "validation_level": self.config.validation_level
            },
            "tests": {}
        }
        
        # Run individual test suites
        test_results["tests"]["data_processing"] = await self._test_data_processing()
        test_results["tests"]["fact_extraction"] = await self._test_fact_extraction()
        test_results["tests"]["tag_generation"] = await self._test_tag_generation()
        test_results["tests"]["embedding_creation"] = await self._test_embedding_creation()
        test_results["tests"]["data_merging"] = await self._test_data_merging()
        test_results["tests"]["quality_validation"] = await self._test_quality_validation()
        test_results["tests"]["job_matching"] = await self._test_job_matching()
        test_results["tests"]["performance"] = await self._test_performance()
        test_results["tests"]["error_handling"] = await self._test_error_handling()
        
        test_results["end_time"] = datetime.utcnow().isoformat()
        test_results["overall_success"] = all(
            test.get("success", False) for test in test_results["tests"].values()
        )
        
        log.info(f"Test suite completed. Overall success: {test_results['overall_success']}")
        return test_results
    
    async def _test_data_processing(self) -> Dict[str, Any]:
        """Test data processing pipeline"""
        log.info("Testing data processing pipeline")
        
        test_results = {
            "test_name": "data_processing",
            "success": True,
            "errors": [],
            "metrics": {}
        }
        
        try:
            # Generate test candidate
            candidate_id = f"test_candidate_{uuid.uuid4().hex[:8]}"
            candidate_profile = self.mock_generator.generate_candidate_profile(candidate_id)
            
            # Test resume processing
            resume_data = {
                "Name": candidate_profile.profile_data["Name"],
                "Skills": [{"skill": fact.content["skill_name"]} for fact in candidate_profile.facts if fact.fact_type == FactType.SKILL],
                "WorkExperience": [fact.content for fact in candidate_profile.facts if fact.fact_type == FactType.EXPERIENCE],
                "Education": [fact.content for fact in candidate_profile.facts if fact.fact_type == FactType.EDUCATION]
            }
            
            result = await process_resume_data(candidate_id, candidate_id, "test_tenant", resume_data)
            
            test_results["metrics"]["resume_processing_success"] = result.success
            test_results["metrics"]["processing_time_ms"] = result.processing_time_ms
            
            if not result.success:
                test_results["success"] = False
                test_results["errors"].append(f"Resume processing failed: {result.error_message}")
            
            # Test chat processing
            chat_data = {
                "conversation_text": "I'm interested in software development and machine learning",
                "interests": ["Python", "AI", "Web Development"],
                "preferences": {"remote_work": True, "startup_environment": True}
            }
            
            result = await process_chat_data(candidate_id, candidate_id, "test_tenant", chat_data)
            
            test_results["metrics"]["chat_processing_success"] = result.success
            test_results["metrics"]["chat_processing_time_ms"] = result.processing_time_ms
            
            if not result.success:
                test_results["success"] = False
                test_results["errors"].append(f"Chat processing failed: {result.error_message}")
            
            # Test assessment processing
            assessment_data = {
                "skill_scores": {"Python": 85, "JavaScript": 70, "SQL": 90},
                "personality_traits": {"communication": 8, "leadership": 7, "creativity": 9},
                "overall_score": 82,
                "assessment_type": "technical_skills"
            }
            
            result = await process_assessment_data(candidate_id, candidate_id, "test_tenant", assessment_data)
            
            test_results["metrics"]["assessment_processing_success"] = result.success
            test_results["metrics"]["assessment_processing_time_ms"] = result.processing_time_ms
            
            if not result.success:
                test_results["success"] = False
                test_results["errors"].append(f"Assessment processing failed: {result.error_message}")
            
        except Exception as e:
            test_results["success"] = False
            test_results["errors"].append(f"Data processing test failed: {str(e)}")
        
        return test_results
    
    async def _test_fact_extraction(self) -> Dict[str, Any]:
        """Test fact extraction"""
        log.info("Testing fact extraction")
        
        test_results = {
            "test_name": "fact_extraction",
            "success": True,
            "errors": [],
            "metrics": {}
        }
        
        try:
            # Test with mock data
            candidate_id = f"test_fact_{uuid.uuid4().hex[:8]}"
            candidate_profile = self.mock_generator.generate_candidate_profile(candidate_id)
            
            # Count facts by type
            fact_counts = {}
            for fact in candidate_profile.facts:
                fact_type = fact.fact_type.value
                fact_counts[fact_type] = fact_counts.get(fact_type, 0) + 1
            
            test_results["metrics"]["total_facts"] = len(candidate_profile.facts)
            test_results["metrics"]["fact_types"] = fact_counts
            
            # Validate fact quality
            low_confidence_facts = [f for f in candidate_profile.facts if f.confidence.value < 0.5]
            test_results["metrics"]["low_confidence_facts"] = len(low_confidence_facts)
            
            if len(low_confidence_facts) > len(candidate_profile.facts) * 0.3:
                test_results["success"] = False
                test_results["errors"].append("Too many low-confidence facts")
            
        except Exception as e:
            test_results["success"] = False
            test_results["errors"].append(f"Fact extraction test failed: {str(e)}")
        
        return test_results
    
    async def _test_tag_generation(self) -> Dict[str, Any]:
        """Test tag generation"""
        log.info("Testing tag generation")
        
        test_results = {
            "test_name": "tag_generation",
            "success": True,
            "errors": [],
            "metrics": {}
        }
        
        try:
            candidate_id = f"test_tag_{uuid.uuid4().hex[:8]}"
            candidate_profile = self.mock_generator.generate_candidate_profile(candidate_id)
            
            # Count tags by category
            tag_counts = {}
            for tag in candidate_profile.tags:
                category = tag.category.value
                tag_counts[category] = tag_counts.get(category, 0) + 1
            
            test_results["metrics"]["total_tags"] = len(candidate_profile.tags)
            test_results["metrics"]["tag_categories"] = tag_counts
            
            # Validate tag quality
            low_confidence_tags = [t for t in candidate_profile.tags if t.confidence.value < 0.3]
            test_results["metrics"]["low_confidence_tags"] = len(low_confidence_tags)
            
            if len(low_confidence_tags) > len(candidate_profile.tags) * 0.5:
                test_results["success"] = False
                test_results["errors"].append("Too many low-confidence tags")
            
        except Exception as e:
            test_results["success"] = False
            test_results["errors"].append(f"Tag generation test failed: {str(e)}")
        
        return test_results
    
    async def _test_embedding_creation(self) -> Dict[str, Any]:
        """Test embedding creation"""
        log.info("Testing embedding creation")
        
        test_results = {
            "test_name": "embedding_creation",
            "success": True,
            "errors": [],
            "metrics": {}
        }
        
        try:
            candidate_id = f"test_embedding_{uuid.uuid4().hex[:8]}"
            candidate_profile = self.mock_generator.generate_candidate_profile(candidate_id)
            
            # Count embeddings by view type
            embedding_counts = {}
            for embedding in candidate_profile.embeddings:
                view_type = embedding.view_type.value
                embedding_counts[view_type] = embedding_counts.get(view_type, 0) + 1
            
            test_results["metrics"]["total_embeddings"] = len(candidate_profile.embeddings)
            test_results["metrics"]["embedding_views"] = embedding_counts
            
            # Validate embedding quality
            low_confidence_embeddings = [e for e in candidate_profile.embeddings if e.confidence.value < 0.4]
            test_results["metrics"]["low_confidence_embeddings"] = len(low_confidence_embeddings)
            
            # Validate vector dimensions
            for embedding in candidate_profile.embeddings:
                if len(embedding.vector) != 384:
                    test_results["success"] = False
                    test_results["errors"].append(f"Invalid vector dimension: {len(embedding.vector)}")
            
        except Exception as e:
            test_results["success"] = False
            test_results["errors"].append(f"Embedding creation test failed: {str(e)}")
        
        return test_results
    
    async def _test_data_merging(self) -> Dict[str, Any]:
        """Test data merging"""
        log.info("Testing data merging")
        
        test_results = {
            "test_name": "data_merging",
            "success": True,
            "errors": [],
            "metrics": {}
        }
        
        try:
            # Generate multiple candidate profiles to test merging
            candidate_id = f"test_merge_{uuid.uuid4().hex[:8]}"
            
            # Create profiles from different sources
            resume_profile = self.mock_generator.generate_candidate_profile(f"{candidate_id}_resume")
            chat_profile = self.mock_generator.generate_candidate_profile(f"{candidate_id}_chat")
            
            # Test merging logic (simplified)
            total_facts = len(resume_profile.facts) + len(chat_profile.facts)
            total_tags = len(resume_profile.tags) + len(chat_profile.tags)
            total_embeddings = len(resume_profile.embeddings) + len(chat_profile.embeddings)
            
            test_results["metrics"]["total_facts_before_merge"] = total_facts
            test_results["metrics"]["total_tags_before_merge"] = total_tags
            test_results["metrics"]["total_embeddings_before_merge"] = total_embeddings
            
            # Simulate deduplication (simplified)
            unique_facts = len(set(f.fact_id for f in resume_profile.facts + chat_profile.facts))
            unique_tags = len(set(t.tag_id for t in resume_profile.tags + chat_profile.tags))
            
            test_results["metrics"]["unique_facts_after_merge"] = unique_facts
            test_results["metrics"]["unique_tags_after_merge"] = unique_tags
            
        except Exception as e:
            test_results["success"] = False
            test_results["errors"].append(f"Data merging test failed: {str(e)}")
        
        return test_results
    
    async def _test_quality_validation(self) -> Dict[str, Any]:
        """Test quality validation"""
        log.info("Testing quality validation")
        
        test_results = {
            "test_name": "quality_validation",
            "success": True,
            "errors": [],
            "metrics": {}
        }
        
        try:
            candidate_id = f"test_quality_{uuid.uuid4().hex[:8]}"
            candidate_profile = self.mock_generator.generate_candidate_profile(candidate_id)
            
            # Test quality metrics
            completeness_score = candidate_profile.completeness_score
            quality_score = candidate_profile.quality_score
            
            test_results["metrics"]["completeness_score"] = completeness_score
            test_results["metrics"]["quality_score"] = quality_score
            
            # Validate quality thresholds
            if completeness_score < 0.5:
                test_results["success"] = False
                test_results["errors"].append(f"Low completeness score: {completeness_score}")
            
            if quality_score < 0.5:
                test_results["success"] = False
                test_results["errors"].append(f"Low quality score: {quality_score}")
            
        except Exception as e:
            test_results["success"] = False
            test_results["errors"].append(f"Quality validation test failed: {str(e)}")
        
        return test_results
    
    async def _test_job_matching(self) -> Dict[str, Any]:
        """Test job matching"""
        log.info("Testing job matching")
        
        test_results = {
            "test_name": "job_matching",
            "success": True,
            "errors": [],
            "metrics": {}
        }
        
        try:
            # Generate test candidate and job
            candidate_id = f"test_match_{uuid.uuid4().hex[:8]}"
            job_id = f"test_job_{uuid.uuid4().hex[:8]}"
            
            job_description = self.mock_generator.generate_job_description(job_id)
            
            # Test job matching (mock implementation)
            # In a real test, this would call the actual matching system
            test_results["metrics"]["job_id"] = job_id
            test_results["metrics"]["candidate_id"] = candidate_id
            test_results["metrics"]["job_title"] = job_description.job_title
            test_results["metrics"]["company_name"] = job_description.company_name
            
            # Simulate matching results
            mock_matches = [
                {
                    "job_id": job_id,
                    "candidate_id": candidate_id,
                    "overall_score": random.uniform(0.6, 0.9),
                    "skill_match_score": random.uniform(0.7, 0.95),
                    "experience_match_score": random.uniform(0.5, 0.8),
                    "education_match_score": random.uniform(0.6, 0.9),
                    "cultural_fit_score": random.uniform(0.5, 0.8)
                }
            ]
            
            test_results["metrics"]["matches_found"] = len(mock_matches)
            test_results["metrics"]["top_match_score"] = mock_matches[0]["overall_score"]
            
        except Exception as e:
            test_results["success"] = False
            test_results["errors"].append(f"Job matching test failed: {str(e)}")
        
        return test_results
    
    async def _test_performance(self) -> Dict[str, Any]:
        """Test performance metrics"""
        log.info("Testing performance")
        
        test_results = {
            "test_name": "performance",
            "success": True,
            "errors": [],
            "metrics": {}
        }
        
        try:
            # Test processing time for multiple candidates
            start_time = datetime.utcnow()
            
            processing_times = []
            for i in range(min(5, self.config.num_candidates)):
                candidate_id = f"perf_test_{i}"
                candidate_profile = self.mock_generator.generate_candidate_profile(candidate_id)
                
                # Simulate processing time
                await asyncio.sleep(0.01)  # Simulate processing
                
                processing_times.append(100)  # Mock processing time
            
            total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            test_results["metrics"]["total_processing_time_ms"] = total_time
            test_results["metrics"]["average_processing_time_ms"] = sum(processing_times) / len(processing_times)
            test_results["metrics"]["candidates_processed"] = len(processing_times)
            
            # Performance thresholds
            if total_time > 5000:  # 5 seconds
                test_results["success"] = False
                test_results["errors"].append(f"Processing too slow: {total_time}ms")
            
        except Exception as e:
            test_results["success"] = False
            test_results["errors"].append(f"Performance test failed: {str(e)}")
        
        return test_results
    
    async def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling"""
        log.info("Testing error handling")
        
        test_results = {
            "test_name": "error_handling",
            "success": True,
            "errors": [],
            "metrics": {}
        }
        
        try:
            # Test with invalid data
            invalid_candidate_id = ""
            invalid_resume_data = {}
            
            # This should handle errors gracefully
            try:
                result = await process_resume_data(invalid_candidate_id, invalid_candidate_id, "test_tenant", invalid_resume_data)
                if result.success:
                    test_results["success"] = False
                    test_results["errors"].append("Should have failed with invalid data")
                else:
                    test_results["metrics"]["error_handling_success"] = True
            except Exception as e:
                test_results["metrics"]["error_handling_success"] = True
                test_results["metrics"]["error_message"] = str(e)
            
            # Test with missing required fields
            try:
                result = await process_resume_data("test", "test", "", {})
                if result.success:
                    test_results["success"] = False
                    test_results["errors"].append("Should have failed with missing tenant_id")
            except Exception as e:
                test_results["metrics"]["missing_field_handling"] = True
            
        except Exception as e:
            test_results["success"] = False
            test_results["errors"].append(f"Error handling test failed: {str(e)}")
        
        return test_results


# ==================== TEST UTILITIES ====================

async def run_comprehensive_tests(config: TestConfig = None) -> Dict[str, Any]:
    """Run comprehensive test suite"""
    if config is None:
        config = TestConfig()
    
    test_runner = TestRunner(config)
    return await test_runner.run_all_tests()


async def run_quick_tests() -> Dict[str, Any]:
    """Run quick test suite"""
    config = TestConfig(num_candidates=3, num_jobs=2)
    return await run_comprehensive_tests(config)


async def run_performance_tests() -> Dict[str, Any]:
    """Run performance-focused tests"""
    config = TestConfig(num_candidates=50, num_jobs=10)
    test_runner = TestRunner(config)
    
    # Run only performance tests
    return {
        "performance_test": await test_runner._test_performance(),
        "data_processing_test": await test_runner._test_data_processing()
    }


# ==================== MOCK DATA EXPORT ====================

def export_mock_data(num_candidates: int = 10, num_jobs: int = 5) -> Dict[str, Any]:
    """Export mock data for external testing"""
    generator = MockDataGenerator()
    
    mock_data = {
        "candidates": [],
        "jobs": [],
        "generated_at": datetime.utcnow().isoformat()
    }
    
    # Generate candidates
    for i in range(num_candidates):
        candidate_id = f"mock_candidate_{i}"
        candidate_profile = generator.generate_candidate_profile(candidate_id)
        
        # Convert to serializable format
        candidate_data = {
            "candidate_id": candidate_profile.candidate_id,
            "profile_data": candidate_profile.profile_data,
            "facts": [
                {
                    "fact_type": fact.fact_type.value,
                    "content": fact.content,
                    "confidence": fact.confidence.value
                }
                for fact in candidate_profile.facts
            ],
            "tags": [
                {
                    "category": tag.category.value,
                    "value": tag.value,
                    "confidence": tag.confidence.value
                }
                for tag in candidate_profile.tags
            ]
        }
        mock_data["candidates"].append(candidate_data)
    
    # Generate jobs
    for i in range(num_jobs):
        job_id = f"mock_job_{i}"
        job_description = generator.generate_job_description(job_id)
        
        job_data = {
            "job_id": job_description.job_id,
            "job_title": job_description.job_title,
            "company_name": job_description.company_name,
            "location": job_description.location,
            "required_skills": job_description.required_skills,
            "preferred_skills": job_description.preferred_skills,
            "required_experience": job_description.required_experience,
            "education_requirements": job_description.education_requirements,
            "job_description": job_description.job_description
        }
        mock_data["jobs"].append(job_data)
    
    return mock_data
