#!/usr/bin/env python3
"""
Script to populate the knowledge base with cloud platform courses.
Adds 50 courses across 5 categories:
- AWS (10 courses)
- Microsoft Azure (10 courses)
- Google Cloud Platform (10 courses)
- Cloud Architecture (10 courses)
- Cloud Security (10 courses)

Usage:
    python scripts/populate_cloud_courses.py
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
    """Populate knowledge base with cloud platform courses"""
    try:
        logger.info("🚀 Starting cloud courses population...")
        logger.info("📚 Adding 50 courses across 5 cloud categories...")
        
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
            ("AWS", kb.add_aws_courses),
            ("Microsoft Azure", kb.add_azure_courses),
            ("Google Cloud Platform", kb.add_gcp_courses),
            ("Cloud Architecture", kb.add_cloud_architecture_courses),
            ("Cloud Security", kb.add_cloud_security_courses),
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
            ("AWS", "AWS Courses"),
            ("Azure", "Azure Courses"),
            ("Google Cloud", "GCP Courses"),
            ("cloud architecture", "Cloud Architecture"),
            ("cloud security", "Cloud Security"),
        ]
        
        for query, category in test_queries:
            results = kb.search_courses(query, top_k=3)
            logger.info(f"\n  {category} ({len(results)} results):")
            for i, course in enumerate(results, 1):
                logger.info(f"    {i}. {course['title']} ({course['provider']}) - Score: {course.get('similarity_score', 0):.3f}")
        
        if added_count == 50:
            logger.info("\n✅ Successfully added all 50 cloud courses!")
            return 0
        else:
            logger.warning(f"\n⚠️ Expected 50 courses, but added {added_count}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error populating courses: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())

