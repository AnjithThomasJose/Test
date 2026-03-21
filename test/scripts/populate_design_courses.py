#!/usr/bin/env python3
"""
Script to populate the knowledge base with design and creative courses.
Adds 60 courses across 6 categories:
- UI/UX Design (10 courses)
- Graphic Design (10 courses)
- Animation (10 courses)
- Game Development (10 courses)
- Video Editing (10 courses)
- 3D Modeling (10 courses)

Usage:
    python scripts/populate_design_courses.py
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
    """Populate knowledge base with design and creative courses"""
    try:
        logger.info("🚀 Starting design and creative courses population...")
        logger.info("📚 Adding 60 courses across 6 design categories...")
        
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
            ("UI/UX Design", kb.add_ui_ux_design_courses),
            ("Graphic Design", kb.add_graphic_design_courses),
            ("Animation", kb.add_animation_courses),
            ("Game Development", kb.add_game_development_courses),
            ("Video Editing", kb.add_video_editing_courses),
            ("3D Modeling", kb.add_3d_modeling_courses),
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
            ("UX design", "UI/UX Design"),
            ("graphic design", "Graphic Design"),
            ("animation", "Animation"),
            ("game development", "Game Development"),
            ("video editing", "Video Editing"),
            ("3D modeling", "3D Modeling"),
        ]
        
        for query, category in test_queries:
            results = kb.search_courses(query, top_k=3)
            logger.info(f"\n  {category} ({len(results)} results):")
            for i, course in enumerate(results, 1):
                logger.info(f"    {i}. {course['title']} ({course['provider']}) - Score: {course.get('similarity_score', 0):.3f}")
        
        if added_count == 60:
            logger.info("\n✅ Successfully added all 60 design and creative courses!")
            return 0
        else:
            logger.warning(f"\n⚠️ Expected 60 courses, but added {added_count}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error populating courses: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())

