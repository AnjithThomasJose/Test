#!/usr/bin/env python3
"""
Script to populate the knowledge base with soft skills and career courses.
Adds 60 courses across 6 categories:
- Communication Skills (10 courses)
- Leadership (10 courses)
- Resume & Interview Prep (10 courses)
- Public Speaking (10 courses)
- Negotiation (10 courses)
- Critical Thinking (10 courses)

Usage:
    python scripts/populate_soft_skills_courses.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.course_knowledge_base import CourseKnowledgeBase
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Populate knowledge base with soft skills and career courses"""
    try:
        logger.info("🚀 Starting soft skills and career courses population...")
        logger.info("📚 Adding 60 courses across 6 categories...")
        
        # Initialize knowledge base
        kb = CourseKnowledgeBase()
        
        # Get current course count
        initial_count = kb.get_course_count()
        logger.info(f"📊 Current courses in knowledge base: {initial_count}")
        
        # Track results
        results = {}
        total_added = 0
        
        # Add all categories
        categories = [
            ("Communication Skills", kb.add_communication_skills_courses),
            ("Leadership", kb.add_leadership_courses),
            ("Resume & Interview Prep", kb.add_resume_interview_courses),
            ("Public Speaking", kb.add_public_speaking_courses),
            ("Negotiation", kb.add_negotiation_courses),
            ("Critical Thinking", kb.add_critical_thinking_courses),
        ]
        
        for category_name, add_method in categories:
            logger.info(f"\n📦 Adding {category_name} courses...")
            success = add_method()
            if success:
                results[category_name] = "✅ Success"
                total_added += 10
            else:
                results[category_name] = "❌ Failed"
        
        # Get final course count
        final_count = kb.get_course_count()
        added_count = final_count - initial_count
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("📊 POPULATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Initial courses: {initial_count}")
        logger.info(f"Final courses: {final_count}")
        logger.info(f"Total added: {added_count}")
        logger.info("\nCategory Results:")
        for category, status in results.items():
            logger.info(f"  {category}: {status}")
        
        # Verify by searching
        logger.info("\n🔍 Verifying courses by searching...")
        test_queries = [
            ("communication skills", "Communication Skills"),
            ("leadership management", "Leadership"),
            ("resume interview", "Resume & Interview"),
            ("public speaking", "Public Speaking"),
            ("negotiation", "Negotiation"),
            ("critical thinking", "Critical Thinking"),
        ]
        
        for query, category in test_queries:
            results = kb.search_courses(query, top_k=3)
            logger.info(f"\n  {category} ({len(results)} results):")
            for i, course in enumerate(results, 1):
                logger.info(f"    {i}. {course['title']} ({course['provider']}) - Score: {course.get('similarity_score', 0):.3f}")
        
        if added_count >= 50:  # Allow some flexibility for duplicates
            logger.info(f"\n✅ Successfully added {added_count} soft skills and career courses!")
            return 0
        else:
            logger.warning(f"\n⚠️ Expected ~60 courses, but added {added_count}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error populating courses: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())

