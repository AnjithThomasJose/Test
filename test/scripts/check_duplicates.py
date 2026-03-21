#!/usr/bin/env python3
"""
Script to check for duplicate courses in the knowledge base.
Identifies courses that would have the same ID based on provider, URL, and title.

Usage:
    python scripts/check_duplicates.py
"""

import sys
import os
import hashlib

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

def generate_course_id(course: dict) -> str:
    """Generate unique ID from course URL, provider, and title (same logic as CourseKnowledgeBase)"""
    url = course.get("url", "")
    provider = course.get("provider", "")
    title = course.get("title", "")
    unique_string = f"{provider}|{url}|{title}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def check_duplicates_in_new_courses():
    """Check for duplicates within the new courses we're trying to add"""
    kb = CourseKnowledgeBase()
    
    # Collect all courses from both sets
    all_courses = []
    
    # Set 1 courses
    logger.info("📋 Collecting Set 1 courses...")
    mech_set1 = [
        {"title": "Introduction to Engineering Mechanics", "provider": "Coursera", "url": "https://www.coursera.org/learn/engineering-mechanics-statics"},
        {"title": "Applications in Engineering Mechanics", "provider": "Coursera", "url": "https://www.coursera.org/learn/engineering-mechanics-applications"},
        {"title": "Mechanics of Materials I: Fundamentals", "provider": "Coursera", "url": "https://www.coursera.org/learn/mechanics-1"},
        {"title": "Introduction to Thermodynamics", "provider": "Coursera", "url": "https://www.coursera.org/learn/thermodynamics-intro"},
        {"title": "Modern Robotics: Mechanics, Planning, and Control", "provider": "Coursera", "url": "https://www.coursera.org/specializations/modernrobotics"},
        {"title": "Engineering Systems in Motion: Dynamics", "provider": "Coursera", "url": "https://www.coursera.org/learn/engineering-systems-in-motion"},
        {"title": "Material Behavior", "provider": "Coursera", "url": "https://www.coursera.org/learn/material-behavior"},
        {"title": "Finite Element Method (FEM) for Engineers", "provider": "Coursera", "url": "https://www.coursera.org/learn/finite-element-method"},
        {"title": "Machine Design Part I", "provider": "Coursera", "url": "https://www.coursera.org/learn/machine-design1"},
        {"title": "Advanced Manufacturing Enterprise", "provider": "Coursera", "url": "https://www.coursera.org/learn/advanced-manufacturing-enterprise"},
    ]
    
    elec_set1 = [
        {"title": "Introduction to Electronics", "provider": "Coursera", "url": "https://www.coursera.org/learn/electronics"},
        {"title": "Electric Power Systems", "provider": "Coursera", "url": "https://www.coursera.org/learn/electric-power-systems"},
        {"title": "Power Electronics Specialization", "provider": "Coursera", "url": "https://www.coursera.org/specializations/power-electronics"},
        {"title": "Linear Circuits 1: DC Analysis", "provider": "Coursera", "url": "https://www.coursera.org/learn/linear-circuits-dcanalysis"},
        {"title": "Linear Circuits 2: AC Analysis", "provider": "Coursera", "url": "https://www.coursera.org/learn/linear-circuits-ac-analysis"},
        {"title": "Introduction to Electricity and Magnetism", "provider": "Coursera", "url": "https://www.coursera.org/learn/electricity-magnetism"},
        {"title": "Solar Energy Basics", "provider": "Coursera", "url": "https://www.coursera.org/learn/solar-energy-basics"},
        {"title": "Electrical Power Distribution", "provider": "Coursera", "url": "https://www.coursera.org/learn/electrical-power-distribution"},
        {"title": "Battery Management Systems", "provider": "Coursera", "url": "https://www.coursera.org/specializations/battery-management-systems"},
        {"title": "Wind Energy", "provider": "Coursera", "url": "https://www.coursera.org/learn/wind-energy"},
    ]
    
    # Set 2 courses
    logger.info("📋 Collecting Set 2 courses...")
    mech_set2 = [
        {"title": "Autodesk CAD/CAM/CAE for Mechanical Engineering", "provider": "Coursera", "url": "https://www.coursera.org/specializations/autodesk-cad-cam-cae-mechanical-engineering"},
        {"title": "Computational Fluid Dynamics (CFD) Fundamentals", "provider": "Udemy", "url": "https://www.udemy.com/course/fluid-mechanics-and-cfd-fundamentals/"},
        {"title": "Introduction to Engineering (Mechanics Focus)", "provider": "edX", "url": "https://www.edx.org/course/a-hands-on-introduction-to-engineering-simulations"},
        {"title": "Six Sigma Green Belt Specialization", "provider": "Coursera", "url": "https://www.coursera.org/specializations/six-sigma-green-belt"},
        {"title": "MATLAB Programming for Engineers", "provider": "Coursera", "url": "https://www.coursera.org/specializations/matlab-programming-engineers-scientists"},
        {"title": "Fundamentals of Fluid Power", "provider": "Coursera", "url": "https://www.coursera.org/learn/fluid-power"},
        {"title": "SolidWorks: Become a Certified Associate (CSWA)", "provider": "Udemy", "url": "https://www.udemy.com/course/solidworks-cswa/"},
        {"title": "Introduction to Geometric Dimensioning (GD&T)", "provider": "Udemy", "url": "https://www.udemy.com/course/geometric-dimensioning-and-tolerancing-gdt-fundamentals/"},
        {"title": "Ferrous Technology I (Steel Metallurgy)", "provider": "Coursera", "url": "https://www.coursera.org/learn/ferrous-technology"},
        {"title": "Control of Mobile Robots", "provider": "Coursera", "url": "https://www.coursera.org/learn/mobile-robot"},
    ]
    
    elec_set2 = [
        {"title": "Introduction to Electronics: Semiconductors", "provider": "Coursera", "url": "https://www.coursera.org/learn/electronics"},
        {"title": "Electric Industry Operations and Markets", "provider": "Coursera", "url": "https://www.coursera.org/learn/electric-industry-operations-markets"},
        {"title": "Fundamentals of Electrical Controls", "provider": "Udemy", "url": "https://www.udemy.com/course/fundamentals-of-electrical-controls/"},
        {"title": "Smart Grid: Fundamentals and Technologies", "provider": "edX", "url": "https://www.edx.org/learn/smart-grids/delft-university-of-technology-smart-grids-fundamentals-and-technologies"},
        {"title": "Electrical Engineering: Circuit Analysis", "provider": "Udemy", "url": "https://www.udemy.com/course/electrical-engineering-circuit-analysis/"},
        {"title": "Electric Vehicles and Mobility", "provider": "Coursera", "url": "https://www.coursera.org/learn/electric-vehicles-mobility"},
        {"title": "Power System Protection", "provider": "Udemy", "url": "https://www.udemy.com/course/power-system-protection-fundamentals/"},
        {"title": "Introduction to Satellite Communications", "provider": "Coursera", "url": "https://www.coursera.org/learn/satellite-communications"},
        {"title": "Electricity & Magnetism: Fields & Forces", "provider": "Coursera", "url": "https://www.coursera.org/learn/electricity-magnetism-fields-forces"},
        {"title": "Solar Energy System Design", "provider": "Udemy", "url": "https://www.udemy.com/course/solar-energy-system-design/"},
    ]
    
    all_courses = mech_set1 + elec_set1 + mech_set2 + elec_set2
    
    # Check for duplicates within the new courses
    logger.info(f"\n🔍 Checking {len(all_courses)} courses for duplicates...")
    seen_ids = {}
    duplicates = []
    
    for course in all_courses:
        course_id = generate_course_id(course)
        if course_id in seen_ids:
            duplicates.append({
                "course": course,
                "duplicate_of": seen_ids[course_id],
                "id": course_id
            })
        else:
            seen_ids[course_id] = course
    
    if duplicates:
        logger.warning(f"\n⚠️ Found {len(duplicates)} duplicates within new courses:")
        for dup in duplicates:
            logger.warning(f"  - '{dup['course']['title']}' (Set 2)")
            logger.warning(f"    duplicates '{dup['duplicate_of']['title']}' (Set 1)")
            logger.warning(f"    ID: {dup['id']}")
    else:
        logger.info("✅ No duplicates found within new courses")
    
    # Now check against existing courses in database
    logger.info("\n🔍 Checking against existing courses in database...")
    existing_count = kb.get_course_count()
    logger.info(f"📊 Current courses in database: {existing_count}")
    
    # Search for potential matches
    potential_matches = 0
    for course in all_courses:
        # Try searching by title
        results = kb.search_courses(course['title'], top_k=5)
        for result in results:
            if result.get('url') == course['url'] or result.get('title', '').lower() == course['title'].lower():
                potential_matches += 1
                logger.info(f"  ⚠️ Potential match: '{course['title']}' might already exist")
                logger.info(f"     Existing: '{result.get('title')}' - {result.get('url')}")
                break
    
    logger.info(f"\n📊 Summary:")
    logger.info(f"  Total new courses to add: {len(all_courses)}")
    logger.info(f"  Duplicates within new courses: {len(duplicates)}")
    logger.info(f"  Potential matches with existing: {potential_matches}")
    logger.info(f"  Expected unique additions: {len(all_courses) - len(duplicates) - potential_matches}")
    
    return duplicates, potential_matches

if __name__ == "__main__":
    duplicates, matches = check_duplicates_in_new_courses()

