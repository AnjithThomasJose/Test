#!/usr/bin/env python3
"""
Script to populate the knowledge base with programming language courses.
Adds 40 courses across 4 categories:
- Python (10 courses)
- Java (10 courses)
- C/C++ (10 courses)
- JavaScript (10 courses)

Usage:
    python scripts/populate_programming_languages.py
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
    """Populate knowledge base with programming language courses"""
    try:
        logger.info("🚀 Starting programming language courses population...")
        logger.info("📚 Adding 40 courses across 4 programming languages...")
        
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
            ("Python", kb.add_python_courses),
            ("Java", kb.add_java_courses),
            ("C/C++", kb.add_c_cpp_courses),
            ("JavaScript", kb.add_javascript_courses),
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
            ("Python", "Python Courses"),
            ("Java", "Java Courses"),
            ("C++", "C/C++ Courses"),
            ("JavaScript", "JavaScript Courses"),
        ]
        
        for query, category in test_queries:
            results = kb.search_courses(query, top_k=3)
            logger.info(f"\n  {category} ({len(results)} results):")
            for i, course in enumerate(results, 1):
                logger.info(f"    {i}. {course['title']} ({course['provider']}) - Score: {course.get('similarity_score', 0):.3f}")
        
        if added_count == 40:
            logger.info("\n✅ Successfully added all 40 programming language courses!")
            return 0
        else:
            logger.warning(f"\n⚠️ Expected 40 courses, but added {added_count}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error populating courses: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())

