#!/usr/bin/env python3
"""
Script to populate the knowledge base with business and management courses.
Adds 70 courses across 7 categories:
- Product Management (10 courses)
- Project Management (10 courses)
- Digital Marketing (10 courses)
- Business Analytics (10 courses)
- Entrepreneurship (10 courses)
- Finance / FinTech (10 courses)
- HR Analytics (10 courses)

Usage:
    python scripts/populate_business_courses.py
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
    """Populate knowledge base with business and management courses"""
    try:
        logger.info("🚀 Starting business and management courses population...")
        logger.info("📚 Adding 70 courses across 7 business categories...")
        
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
            ("Product Management", kb.add_product_management_courses),
            ("Project Management", kb.add_project_management_courses),
            ("Digital Marketing", kb.add_digital_marketing_courses),
            ("Business Analytics", kb.add_business_analytics_courses),
            ("Entrepreneurship", kb.add_entrepreneurship_courses),
            ("Finance / FinTech", kb.add_finance_fintech_courses),
            ("HR Analytics", kb.add_hr_analytics_courses),
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
            ("product management", "Product Management"),
            ("project management", "Project Management"),
            ("digital marketing", "Digital Marketing"),
            ("business analytics", "Business Analytics"),
            ("entrepreneurship", "Entrepreneurship"),
            ("fintech", "Finance/FinTech"),
            ("hr analytics", "HR Analytics"),
        ]
        
        for query, category in test_queries:
            results = kb.search_courses(query, top_k=3)
            logger.info(f"\n  {category} ({len(results)} results):")
            for i, course in enumerate(results, 1):
                logger.info(f"    {i}. {course['title']} ({course['provider']}) - Score: {course.get('similarity_score', 0):.3f}")
        
        if added_count == 70:
            logger.info("\n✅ Successfully added all 70 business and management courses!")
            return 0
        else:
            logger.warning(f"\n⚠️ Expected 70 courses, but added {added_count}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error populating courses: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())

