#!/usr/bin/env python3
"""
Script to populate the knowledge base with emerging technology courses.
Adds 60 courses across 6 categories:
- Internet of Things (IoT) (10 courses)
- Embedded Systems (10 courses)
- Robotics (10 courses)
- AR/VR Development (10 courses)
- Quantum Computing (10 courses)
- Networking (CCNA/CCNP) (10 courses)

Usage:
    python scripts/populate_emerging_tech_courses.py
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
    """Populate knowledge base with emerging technology courses"""
    try:
        logger.info("🚀 Starting emerging technology courses population...")
        logger.info("📚 Adding 60 courses across 6 emerging tech categories...")
        
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
            ("Internet of Things (IoT)", kb.add_iot_courses),
            ("Embedded Systems", kb.add_embedded_systems_courses),
            ("Robotics", kb.add_robotics_courses),
            ("AR/VR Development", kb.add_ar_vr_courses),
            ("Quantum Computing", kb.add_quantum_computing_courses),
            ("Networking (CCNA/CCNP)", kb.add_networking_courses),
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
            ("IoT", "Internet of Things"),
            ("embedded systems", "Embedded Systems"),
            ("robotics", "Robotics"),
            ("AR VR", "AR/VR Development"),
            ("quantum computing", "Quantum Computing"),
            ("CCNA networking", "Networking"),
        ]
        
        for query, category in test_queries:
            results = kb.search_courses(query, top_k=3)
            logger.info(f"\n  {category} ({len(results)} results):")
            for i, course in enumerate(results, 1):
                logger.info(f"    {i}. {course['title']} ({course['provider']}) - Score: {course.get('similarity_score', 0):.3f}")
        
        if added_count == 60:
            logger.info("\n✅ Successfully added all 60 emerging technology courses!")
            return 0
        else:
            logger.warning(f"\n⚠️ Expected 60 courses, but added {added_count}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error populating courses: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())

