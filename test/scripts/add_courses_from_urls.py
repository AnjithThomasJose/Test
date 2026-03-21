#!/usr/bin/env python3
"""
Script to add courses to the knowledge base from URLs
"""

import sys
import os
import hashlib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.course_knowledge_base import CourseKnowledgeBase
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# List of course URLs to add with their metadata
COURSES_TO_ADD = [
    {
        "url": "https://www.udemy.com/course/online-course-creation-introduction-to-instructional-design/?couponCode=KEEPLEARNING",
        "title": "Online Course Creation: Introduction to Instructional Design",
        "provider": "Udemy",
        "difficulty": "Beginner",
        "duration": "Self-paced",
        "type": "course",
        "skills": ["Instructional Design", "Online Course Creation", "Course Development", "E-Learning"],
        "description": "Introduction to instructional design for creating effective online courses"
    },
    {
        "url": "https://www.udemy.com/course/curriculum-structural-design/?couponCode=KEEPLEARNING",
        "title": "Curriculum Structural Design",
        "provider": "Udemy",
        "difficulty": "Intermediate",
        "duration": "Self-paced",
        "type": "course",
        "skills": ["Curriculum Design", "Structural Design", "Curriculum Development", "Education"],
        "description": "Learn how to design and structure effective curricula"
    },
    {
        "url": "https://www.udemy.com/course/instructional-design-for-elearning/?couponCode=KEEPLEARNING",
        "title": "Instructional Design for eLearning",
        "provider": "Udemy",
        "difficulty": "Intermediate",
        "duration": "Self-paced",
        "type": "course",
        "skills": ["Instructional Design", "E-Learning", "Online Learning", "Course Design"],
        "description": "Comprehensive guide to instructional design principles for eLearning"
    },
    {
        "url": "https://www.udemy.com/course/classroom-management/?couponCode=KEEPLEARNING",
        "title": "Classroom Management",
        "provider": "Udemy",
        "difficulty": "Beginner",
        "duration": "Self-paced",
        "type": "course",
        "skills": ["Classroom Management", "Teaching", "Education", "Teacher Training"],
        "description": "Essential strategies and techniques for effective classroom management"
    },
    {
        "url": "https://www.udemy.com/course/onlineeducationstartups/",
        "title": "Online Education Startups",
        "provider": "Udemy",
        "difficulty": "Intermediate",
        "duration": "Self-paced",
        "type": "course",
        "skills": ["Online Education", "EdTech", "Startups", "Entrepreneurship", "Education Technology"],
        "description": "Learn how to build and scale online education startups"
    },
    {
        "url": "https://www.udemy.com/course/master-child-psychology-certification/?couponCode=KEEPLEARNING",
        "title": "Master Child Psychology Certification",
        "provider": "Udemy",
        "difficulty": "Intermediate",
        "duration": "Self-paced",
        "type": "course",
        "skills": ["Child Psychology", "Psychology", "Education", "Child Development", "Certification"],
        "description": "Comprehensive certification course in child psychology"
    },
]

def generate_course_id(course_data: dict) -> str:
    """Generate unique course ID from URL, provider, and title"""
    url = course_data.get("url", "")
    provider = course_data.get("provider", "")
    title = course_data.get("title", "")
    
    # Create a unique identifier combining all three
    unique_string = f"{provider}|{url}|{title}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def main():
    log.info("🚀 Starting addition of courses to knowledge base...")
    log.info(f"📋 Courses to add: {len(COURSES_TO_ADD)}")
    
    kb = CourseKnowledgeBase()
    
    total_added = 0
    total_skipped = 0
    total_errors = 0
    
    for i, course_data in enumerate(COURSES_TO_ADD, 1):
        url = course_data.get("url", "")
        title = course_data.get("title", "")
        
        log.info(f"\n{'='*60}")
        log.info(f"📝 Processing course {i}/{len(COURSES_TO_ADD)}: {title}")
        log.info(f"🔗 URL: {url}")
        log.info(f"{'='*60}")
        
        try:
            # Check if course already exists
            results = kb.collection.get(where={"url": url})
            if results.get("ids") and results["ids"]:
                log.warning(f"⚠️ Course already exists in KB (ID: {results['ids'][0]})")
                total_skipped += 1
                continue
            
            # Generate course ID
            course_id = generate_course_id(course_data)
            
            # Add course to knowledge base
            if kb.add_course(course_id, course_data):
                total_added += 1
                log.info(f"✅ Successfully added course: {title}")
            else:
                total_errors += 1
                log.error(f"❌ Failed to add course: {title}")
                
        except Exception as e:
            total_errors += 1
            log.error(f"❌ Error processing course '{title}': {e}")
            import traceback
            log.error(traceback.format_exc())
    
    log.info(f"\n{'='*60}")
    log.info(f"📊 Summary:")
    log.info(f"   Total courses processed: {len(COURSES_TO_ADD)}")
    log.info(f"   ✅ Successfully added: {total_added}")
    log.info(f"   ⏭️  Skipped (already exists): {total_skipped}")
    log.info(f"   ❌ Errors: {total_errors}")
    log.info(f"{'='*60}")
    
    if total_added > 0:
        log.info("✅ Course addition completed successfully")
    else:
        log.warning("⚠️ No courses were added")

if __name__ == "__main__":
    main()

