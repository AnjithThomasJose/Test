#!/usr/bin/env python3
"""
Script to populate the knowledge base with engineering courses.
Adds 80 courses across 4 categories (Set 1 + Set 2):
- Mechanical Engineering (20 courses: 10 Set 1 + 10 Set 2)
- Electrical Engineering (20 courses: 10 Set 1 + 10 Set 2)
- Electronics (VLSI, FPGA, PCB Design) (20 courses: 10 Set 1 + 10 Set 2)
- Cloud Hardware (Server Systems & Data Centers) (20 courses: 10 Set 1 + 10 Set 2)

Usage:
    python scripts/populate_engineering_courses.py
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
    """Populate knowledge base with engineering courses"""
    try:
        logger.info("🚀 Starting engineering courses population...")
        logger.info("📚 Adding 80 courses across 4 engineering categories (Set 1 + Set 2)...")
        
        # Initialize knowledge base
        kb = CourseKnowledgeBase()
        
        # Get current course count
        initial_count = kb.get_course_count()
        logger.info(f"📊 Current courses in knowledge base: {initial_count}")
        
        # Track results
        results = {}
        total_added = 0
        
        # Add all categories (Set 1)
        categories_set1 = [
            ("Mechanical Engineering (Set 1)", kb.add_mechanical_engineering_courses),
            ("Electrical Engineering (Set 1)", kb.add_electrical_engineering_courses),
            ("Electronics (Set 1)", kb.add_electronics_courses),
            ("Cloud Hardware (Set 1)", kb.add_cloud_hardware_courses),
        ]
        
        # Add all categories (Set 2)
        categories_set2 = [
            ("Mechanical Engineering (Set 2)", kb.add_mechanical_engineering_courses_set2),
            ("Electrical Engineering (Set 2)", kb.add_electrical_engineering_courses_set2),
            ("Electronics (Set 2)", kb.add_electronics_courses_set2),
            ("Cloud Hardware (Set 2)", kb.add_cloud_hardware_courses_set2),
        ]
        
        # Add Set 1
        logger.info("\n" + "="*60)
        logger.info("📦 Adding Set 1 courses...")
        logger.info("="*60)
        for category_name, add_method in categories_set1:
            logger.info(f"\n📦 Adding {category_name} courses...")
            success = add_method()
            if success:
                results[category_name] = "✅ Success"
                total_added += 10
            else:
                results[category_name] = "❌ Failed"
        
        # Add Set 2
        logger.info("\n" + "="*60)
        logger.info("📦 Adding Set 2 courses...")
        logger.info("="*60)
        for category_name, add_method in categories_set2:
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
            ("mechanical engineering CAD", "Mechanical Engineering"),
            ("electrical engineering smart grid", "Electrical Engineering"),
            ("FPGA VLSI SystemVerilog", "Electronics"),
            ("server administration virtualization", "Cloud Hardware"),
        ]
        
        for query, category in test_queries:
            results = kb.search_courses(query, top_k=3)
            logger.info(f"\n  {category} ({len(results)} results):")
            for i, course in enumerate(results, 1):
                logger.info(f"    {i}. {course['title']} ({course['provider']}) - Score: {course.get('similarity_score', 0):.3f}")
        
        if added_count == 80:
            logger.info("\n✅ Successfully added all 80 engineering courses!")
            return 0
        else:
            logger.warning(f"\n⚠️ Expected 80 courses, but added {added_count}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error populating courses: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())

