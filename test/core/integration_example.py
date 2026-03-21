"""
Jobsify AI Candidate Matching Architecture - Integration Example
Demonstrates how to use the new architecture for candidate matching
"""

import asyncio
import json
import logging
from typing import Dict, List, Any
from datetime import datetime

# Import the new architecture components
from .candidate_matching_models import (
    DataSource, ProcessingStage, ValidationLevel, RetrievalConfig
)
from .jobsify_candidate_matching import (
    process_resume_data, process_chat_data, process_assessment_data,
    find_job_matches_for_candidate, get_system_status, perform_health_check
)
from .retrieval_gateway import JobDescription
from .test_framework import run_comprehensive_tests, export_mock_data

log = logging.getLogger(__name__)


class JobsifyCandidateMatchingExample:
    """
    Example implementation showing how to use the Jobsify AI Candidate Matching Architecture
    """
    
    def __init__(self):
        self.candidate_id = "example_candidate_123"
        self.uid = "user_456"
        self.tenant_id = "tenant_789"
        
    async def run_complete_example(self):
        """Run a complete example of the candidate matching system"""
        log.info("Starting Jobsify Candidate Matching Example")
        
        # Step 1: Check system health
        await self._check_system_health()
        
        # Step 2: Process candidate data from multiple sources
        await self._process_candidate_data()
        
        # Step 3: Find job matches
        await self._find_job_matches()
        
        # Step 4: Display results
        await self._display_results()
        
        log.info("Example completed successfully")
    
    async def _check_system_health(self):
        """Check system health"""
        log.info("Checking system health...")
        
        health_status = await perform_health_check()
        
        if health_status["overall_status"] == "healthy":
            log.info("✅ System is healthy")
        else:
            log.error("❌ System health issues detected")
            for component, status in health_status["components"].items():
                if status["status"] != "healthy":
                    log.error(f"  - {component}: {status['details']}")
    
    async def _process_candidate_data(self):
        """Process candidate data from multiple sources"""
        log.info("Processing candidate data from multiple sources...")
        
        # Process resume data
        resume_data = {
            "Name": "John Doe",
            "Email": "john.doe@example.com",
            "Phone": "+1-555-123-4567",
            "Location": "San Francisco, CA",
            "LinkedIn": "https://linkedin.com/in/johndoe",
            "GitHub": "https://github.com/johndoe",
            "Skills": [
                {"skill": "Python", "level": "advanced", "years": 5},
                {"skill": "Machine Learning", "level": "intermediate", "years": 3},
                {"skill": "AWS", "level": "intermediate", "years": 2},
                {"skill": "Docker", "level": "beginner", "years": 1},
                {"skill": "SQL", "level": "advanced", "years": 4}
            ],
            "WorkExperience": [
                {
                    "title": "Senior Software Engineer",
                    "company": "Tech Corp",
                    "duration": "2 years",
                    "description": "Developed machine learning models for recommendation systems using Python and TensorFlow",
                    "location": "San Francisco, CA"
                },
                {
                    "title": "Software Engineer",
                    "company": "StartupXYZ",
                    "duration": "3 years", 
                    "description": "Built web applications using Python, React, and AWS",
                    "location": "Remote"
                }
            ],
            "Education": [
                {
                    "degree": "Bachelor of Science",
                    "major": "Computer Science",
                    "institution": "University of California, Berkeley",
                    "graduation_year": "2018"
                }
            ],
            "Projects": [
                {
                    "name": "ML Recommendation Engine",
                    "description": "Built a recommendation system using collaborative filtering",
                    "technologies": ["Python", "TensorFlow", "Pandas", "Scikit-learn"],
                    "url": "https://github.com/johndoe/ml-recommendation-engine"
                }
            ]
        }
        
        resume_result = await process_resume_data(
            self.candidate_id, self.uid, self.tenant_id, resume_data
        )
        
        if resume_result.success:
            log.info(f"✅ Resume processed successfully")
            log.info(f"   Quality score: {resume_result.quality_metrics.overall_quality:.2f}")
            log.info(f"   Processing time: {resume_result.processing_time_ms}ms")
        else:
            log.error(f"❌ Resume processing failed: {resume_result.error_message}")
        
        # Process chat session data
        chat_data = {
            "conversation_text": "I'm passionate about machine learning and AI. I enjoy working on projects that have real-world impact. I prefer working in collaborative environments where I can learn from others and contribute to meaningful projects.",
            "interests": ["Machine Learning", "Artificial Intelligence", "Data Science", "Open Source"],
            "preferences": {
                "remote_work": True,
                "startup_environment": True,
                "team_size": "small_to_medium",
                "work_style": "collaborative"
            },
            "career_goals": [
                "Lead ML engineering teams",
                "Contribute to open source AI projects",
                "Work on cutting-edge AI research"
            ]
        }
        
        chat_result = await process_chat_data(
            self.candidate_id, self.uid, self.tenant_id, chat_data
        )
        
        if chat_result.success:
            log.info(f"✅ Chat session processed successfully")
            log.info(f"   Processing time: {chat_result.processing_time_ms}ms")
        else:
            log.error(f"❌ Chat processing failed: {chat_result.error_message}")
        
        # Process assessment data
        assessment_data = {
            "skill_scores": {
                "Python": 92,
                "Machine Learning": 85,
                "AWS": 78,
                "Docker": 65,
                "SQL": 88,
                "Problem Solving": 90,
                "Communication": 85,
                "Leadership": 75
            },
            "personality_traits": {
                "openness": 8.5,
                "conscientiousness": 9.0,
                "extraversion": 7.0,
                "agreeableness": 8.5,
                "neuroticism": 3.0
            },
            "technical_assessment": {
                "coding_challenge_score": 88,
                "system_design_score": 82,
                "algorithms_score": 90
            },
            "overall_score": 87,
            "assessment_type": "comprehensive_technical",
            "completed_at": "2024-01-15T14:30:00Z"
        }
        
        assessment_result = await process_assessment_data(
            self.candidate_id, self.uid, self.tenant_id, assessment_data
        )
        
        if assessment_result.success:
            log.info(f"✅ Assessment processed successfully")
            log.info(f"   Processing time: {assessment_result.processing_time_ms}ms")
        else:
            log.error(f"❌ Assessment processing failed: {assessment_result.error_message}")
    
    async def _find_job_matches(self):
        """Find job matches for the candidate"""
        log.info("Finding job matches...")
        
        # Create sample job descriptions
        jobs = [
            JobDescription(
                job_id="job_ml_engineer_001",
                job_title="Senior Machine Learning Engineer",
                company_name="AI Innovations Inc",
                location="San Francisco, CA",
                required_skills=["Python", "Machine Learning", "TensorFlow", "AWS"],
                preferred_skills=["PyTorch", "Docker", "Kubernetes", "MLOps"],
                required_experience="3+ years",
                education_requirements=["Bachelor's Degree in Computer Science or related field"],
                job_description="We're looking for a Senior ML Engineer to join our team and work on cutting-edge AI products. You'll be responsible for developing and deploying machine learning models at scale.",
                company_culture=["Innovation", "Collaboration", "Growth", "Diversity"],
                benefits=["Health Insurance", "401k", "Stock Options", "Flexible Hours", "Remote Work"],
                salary_range="$150k - $200k",
                job_type="Full-time",
                remote_option=True
            ),
            JobDescription(
                job_id="job_python_dev_002",
                job_title="Python Developer",
                company_name="Tech Startup Co",
                location="Remote",
                required_skills=["Python", "SQL", "AWS"],
                preferred_skills=["Docker", "React", "FastAPI"],
                required_experience="2+ years",
                education_requirements=["Bachelor's Degree or equivalent experience"],
                job_description="Join our fast-growing startup as a Python Developer. You'll work on building scalable web applications and data processing pipelines.",
                company_culture=["Startup", "Fast-paced", "Innovation", "Learning"],
                benefits=["Health Insurance", "401k", "Equity", "Unlimited PTO"],
                salary_range="$120k - $160k",
                job_type="Full-time",
                remote_option=True
            ),
            JobDescription(
                job_id="job_data_scientist_003",
                job_title="Data Scientist",
                company_name="Data Corp",
                location="New York, NY",
                required_skills=["Python", "Machine Learning", "SQL", "Statistics"],
                preferred_skills=["R", "Tableau", "Spark", "Hadoop"],
                required_experience="2+ years",
                education_requirements=["Master's Degree in Data Science, Statistics, or related field"],
                job_description="We're seeking a Data Scientist to analyze large datasets and build predictive models. You'll work closely with product and engineering teams.",
                company_culture=["Data-driven", "Collaborative", "Professional", "Growth"],
                benefits=["Health Insurance", "401k", "Professional Development", "Gym Membership"],
                salary_range="$130k - $170k",
                job_type="Full-time",
                remote_option=False
            )
        ]
        
        # Find matches for each job
        for job in jobs:
            log.info(f"Finding matches for: {job.job_title} at {job.company_name}")
            
            matches = await find_job_matches_for_candidate(
                candidate_id=self.candidate_id,
                job_description=job,
                config_name="default"
            )
            
            if matches.total_matches_found > 0:
                log.info(f"✅ Found {matches.total_matches_found} matches")
                log.info(f"   Overall confidence: {matches.overall_confidence.value:.2f}")
                log.info(f"   Processing time: {matches.processing_time_ms}ms")
                
                # Display top match details
                if matches.top_matches:
                    top_match = matches.top_matches[0]
                    log.info(f"   Top match score: {top_match.overall_score:.2f}")
                    log.info(f"   Skill match: {top_match.skill_match_score:.2f}")
                    log.info(f"   Experience match: {top_match.experience_match_score:.2f}")
                    log.info(f"   Education match: {top_match.education_match_score:.2f}")
                    log.info(f"   Cultural fit: {top_match.cultural_fit_score:.2f}")
                    log.info(f"   Rationale: {top_match.rationale}")
            else:
                log.info("❌ No matches found")
    
    async def _display_results(self):
        """Display final results"""
        log.info("Displaying final results...")
        
        # Get system status
        status = await get_system_status()
        
        log.info("📊 System Status:")
        log.info(f"   System: {status['system_name']}")
        log.info(f"   Version: {status['version']}")
        log.info(f"   Timestamp: {status['timestamp']}")
        
        # Display component status
        log.info("🔧 Component Status:")
        for component_name, component_status in status["components"].items():
            if "status" in component_status:
                status_icon = "✅" if component_status["status"] == "active" else "❌"
                log.info(f"   {status_icon} {component_name}: {component_status['status']}")
            else:
                log.info(f"   📊 {component_name}: {component_status}")


async def run_example():
    """Run the complete example"""
    example = JobsifyCandidateMatchingExample()
    await example.run_complete_example()


async def run_test_suite():
    """Run the comprehensive test suite"""
    log.info("Running comprehensive test suite...")
    
    test_results = await run_comprehensive_tests()
    
    log.info("🧪 Test Results:")
    log.info(f"   Overall success: {test_results['overall_success']}")
    log.info(f"   Tests run: {len(test_results['tests'])}")
    
    for test_name, test_result in test_results["tests"].items():
        status_icon = "✅" if test_result["success"] else "❌"
        log.info(f"   {status_icon} {test_name}: {test_result['success']}")
        
        if not test_result["success"] and test_result.get("errors"):
            for error in test_result["errors"]:
                log.error(f"      Error: {error}")


async def generate_mock_data():
    """Generate mock data for testing"""
    log.info("Generating mock data...")
    
    mock_data = export_mock_data(num_candidates=5, num_jobs=3)
    
    log.info("📁 Mock Data Generated:")
    log.info(f"   Candidates: {len(mock_data['candidates'])}")
    log.info(f"   Jobs: {len(mock_data['jobs'])}")
    log.info(f"   Generated at: {mock_data['generated_at']}")
    
    # Save to file
    with open("mock_data.json", "w") as f:
        json.dump(mock_data, f, indent=2)
    
    log.info("   Saved to: mock_data.json")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the example
    asyncio.run(run_example())
    
    # Uncomment to run tests
    # asyncio.run(run_test_suite())
    
    # Uncomment to generate mock data
    # asyncio.run(generate_mock_data())
