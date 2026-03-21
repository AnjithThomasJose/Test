#!/usr/bin/env python3
"""
Script to populate the knowledge base with blockchain courses.

Usage:
    python scripts/populate_blockchain_courses.py
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
    """Populate knowledge base with blockchain courses"""
    try:
        logger.info("🚀 Starting blockchain courses population...")
        
        # Initialize knowledge base
        kb = CourseKnowledgeBase()
        
        # Get current course count
        initial_count = kb.get_course_count()
        logger.info(f"📊 Current courses in knowledge base: {initial_count}")
        
        # Add blockchain courses
        success = kb.add_blockchain_courses()
        
        if success:
            # Get updated course count
            final_count = kb.get_course_count()
            added_count = final_count - initial_count
            logger.info(f"✅ Successfully added {added_count} blockchain courses!")
            logger.info(f"📊 Total courses in knowledge base: {final_count}")
            
            # Verify by searching
            logger.info("\n🔍 Verifying courses by searching for 'blockchain'...")
            results = kb.search_courses("blockchain", top_k=5)
            logger.info(f"Found {len(results)} courses matching 'blockchain':")
            for i, course in enumerate(results, 1):
                logger.info(f"  {i}. {course['title']} ({course['provider']}) - Score: {course.get('similarity_score', 0):.3f}")
            
            return 0
        else:
            logger.error("❌ Failed to add blockchain courses")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error populating courses: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())

