#!/usr/bin/env python3
"""
Populate KB with additional materials (books, papers, videos, tutorials) 
for all existing course topics in the knowledge base.

This script:
1. Gets all courses from KB
2. Extracts unique topics/skills from course titles and skills
3. For each topic, discovers books, papers, videos, tutorials
4. Adds discovered materials to KB

Usage:
    python scripts/populate_materials_for_all_topics.py [--limit N] [--material-types book,paper,video,tutorial]
"""

import sys
import os
import argparse
import asyncio
import re
from typing import List, Dict, Any, Set
from collections import Counter

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.course_knowledge_base import CourseKnowledgeBase
from agents.market_and_course_recommender import _check_and_discover_materials
from core.logging_helpers import AgentLogger, create_log_context
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def extract_topics_from_courses(courses: List[Dict[str, Any]]) -> Set[str]:
    """
    Extract unique topics/skills from courses.
    
    Sources:
    1. Course skills field (primary source)
    2. Course title (extract known tech keywords only)
    
    Args:
        courses: List of course dictionaries
        
    Returns:
        Set of unique topic strings
    """
    topics = set()
    
    # Common tech keywords to extract from titles (only well-known tech terms)
    tech_keywords = [
        # Languages
        "python", "javascript", "java", "typescript", "go", "golang", "rust", "c++", "c#", "csharp",
        "swift", "kotlin", "php", "ruby", "scala", "r", "matlab", "dart",
        # Frameworks & Libraries
        "react", "angular", "vue", "node", "nodejs", "express", "django", "flask", "spring",
        "laravel", "nextjs", "next.js", "nuxt", "svelte", "ember",
        # Databases
        "sql", "mysql", "postgresql", "postgres", "mongodb", "mongo", "redis", "elasticsearch",
        "cassandra", "oracle", "sqlite",
        # Cloud & DevOps
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s", "terraform",
        "ansible", "jenkins", "ci/cd", "cicd", "devops", "sre", "mlops",
        # Domains
        "machine learning", "ml", "deep learning", "data science", "ai", "artificial intelligence",
        "blockchain", "cybersecurity", "web development", "mobile development", "backend",
        "frontend", "full stack", "fullstack",
        # Testing
        "testing", "qa", "selenium", "cypress", "jest", "pytest", "junit",
        # APIs & Protocols
        "api", "rest", "graphql", "grpc", "microservices",
        # Methodologies
        "agile", "scrum", "kanban"
    ]
    
    # Common words to exclude
    exclude_words = {
        "the", "and", "for", "with", "from", "course", "learn", "complete", "guide",
        "tutorial", "bootcamp", "masterclass", "certification", "certified", "training",
        "beginner", "intermediate", "advanced", "expert", "level", "fundamentals",
        "introduction", "intro", "basics", "essentials", "specialty", "specialization",
        "path", "track", "program", "series", "part", "volume", "edition", "version"
    }
    
    for course in courses:
        # PRIMARY: Extract from skills field (most reliable)
        skills = course.get("skills", [])
        if isinstance(skills, list):
            for skill in skills:
                if skill and isinstance(skill, str):
                    skill_clean = skill.strip().lower()
                    # Only add if it's a reasonable length and not excluded
                    if 2 < len(skill_clean) < 50 and skill_clean not in exclude_words:
                        topics.add(skill_clean)
        elif isinstance(skills, str):
            for skill in skills.split(","):
                skill_clean = skill.strip().lower()
                if skill_clean and 2 < len(skill_clean) < 50 and skill_clean not in exclude_words:
                    topics.add(skill_clean)
        
        # SECONDARY: Extract known tech keywords from title
        title = course.get("title", "").lower()
        if title:
            # Check for tech keywords in title
            for keyword in tech_keywords:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, title):
                    topics.add(keyword)
    
    # Final filtering
    filtered_topics = set()
    for topic in topics:
        # Skip if too short, too long, or contains excluded words
        if len(topic) < 2 or len(topic) > 50:
            continue
        
        # Skip if it's just common words
        words = topic.split()
        if all(word in exclude_words for word in words):
            continue
        
        # Skip if it contains common non-topic phrases
        if any(phrase in topic for phrase in ["specialty courses", "engineering path", "analytics specialty"]):
            continue
        
        # Only keep topics that look like real tech terms
        # Must be: single word, or 2-3 words max, and not all common words
        if len(words) <= 3:
            filtered_topics.add(topic)
    
    return filtered_topics


async def populate_materials_for_topics(
    topics: List[str],
    material_types: List[str],
    max_per_topic: int = 5,
    log_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Populate materials for a list of topics.
    
    Args:
        topics: List of topics to populate
        material_types: List of material types (book, paper, video, tutorial)
        max_per_topic: Maximum materials to discover per topic
        log_context: Logging context
        
    Returns:
        Dict with results
    """
    if not log_context:
        log_context = create_log_context("populate_materials", "system")
    
    results = {
        "topics_processed": 0,
        "materials_added": 0,
        "topics_skipped": 0,
        "errors": []
    }
    
    kb = CourseKnowledgeBase()
    
    for i, topic in enumerate(topics, 1):
        try:
            logger.info(f"\n[{i}/{len(topics)}] Processing topic: {topic}")
            
            # Check if we already have materials for this topic
            existing_materials = []
            for material_type in material_types:
                try:
                    filters = {"type": material_type}
                    found = kb.search_courses(
                        query=f"{topic} {material_type}",
                        top_k=5,
                        filters=filters
                    )
                    existing_materials.extend(found)
                except Exception as e:
                    logger.debug(f"Search failed for {material_type}: {e}")
            
            # If we already have enough materials, skip
            if len(existing_materials) >= max_per_topic:
                logger.info(f"  ⏭️  Skipping '{topic}' - already has {len(existing_materials)} materials")
                results["topics_skipped"] += 1
                continue
            
            # Discover materials (excluding courses since we already have those)
            logger.info(f"  🔍 Discovering {', '.join(material_types)} for '{topic}'...")
            
            discovered = await _check_and_discover_materials(
                topic=topic,
                material_types=material_types,
                min_results_required=3,
                min_similarity_threshold=0.4,
                max_results_per_type=3,
                log_context=log_context
            )
            
            if discovered:
                results["materials_added"] += len(discovered)
                logger.info(f"  ✅ Added {len(discovered)} materials for '{topic}'")
            else:
                logger.info(f"  ⚠️  No materials found for '{topic}'")
            
            results["topics_processed"] += 1
            
            # Rate limiting - small delay between topics
            await asyncio.sleep(1)
            
        except Exception as e:
            error_msg = f"Error processing topic '{topic}': {e}"
            results["errors"].append(error_msg)
            logger.error(f"  ❌ {error_msg}")
            continue
    
    return results


async def main_async():
    parser = argparse.ArgumentParser(
        description="Populate KB with additional materials for all course topics"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of topics to process (default: all)"
    )
    parser.add_argument(
        "--material-types",
        type=str,
        default="book,paper,video,tutorial",
        help="Comma-separated list of material types (default: book,paper,video,tutorial)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually adding materials"
    )
    parser.add_argument(
        "--min-courses",
        type=int,
        default=1,
        help="Minimum number of courses a topic must have to be included (default: 1)"
    )
    
    args = parser.parse_args()
    
    try:
        log_context = create_log_context("populate_materials", "system")
        
        # Parse material types
        material_types = [t.strip() for t in args.material_types.split(",") if t.strip()]
        # Ensure we don't include "course" since we're adding other materials
        material_types = [t for t in material_types if t != "course"]
        
        if not material_types:
            logger.error("❌ No valid material types specified (excluding 'course')")
            return 1
        
        logger.info("="*60)
        logger.info("📚 POPULATE MATERIALS FOR ALL COURSE TOPICS")
        logger.info("="*60)
        logger.info(f"Material types: {', '.join(material_types)}")
        logger.info(f"Dry run: {args.dry_run}")
        logger.info("")
        
        # Get all courses from KB
        logger.info("🔍 Fetching all courses from knowledge base...")
        kb = CourseKnowledgeBase()
        all_courses = kb.get_all_courses()
        
        if not all_courses:
            logger.error("❌ No courses found in knowledge base")
            return 1
        
        logger.info(f"✅ Found {len(all_courses)} courses in KB")
        
        # Extract unique topics
        logger.info("\n📊 Extracting topics from courses...")
        topics = extract_topics_from_courses(all_courses)
        logger.info(f"✅ Extracted {len(topics)} unique topics")
        
        # Filter topics that have at least min_courses
        if args.min_courses > 1:
            topic_counts = Counter()
            for course in all_courses:
                skills = course.get("skills", [])
                if isinstance(skills, list):
                    for skill in skills:
                        if skill:
                            topic_counts[skill.lower()] += 1
                elif isinstance(skills, str):
                    for skill in skills.split(","):
                        if skill.strip():
                            topic_counts[skill.strip().lower()] += 1
            
            topics = {t for t in topics if topic_counts.get(t, 0) >= args.min_courses}
            logger.info(f"✅ Filtered to {len(topics)} topics with ≥{args.min_courses} courses")
        
        # Limit topics if specified
        if args.limit:
            topics = list(topics)[:args.limit]
            logger.info(f"📌 Limited to {len(topics)} topics")
        else:
            topics = list(topics)
        
        if not topics:
            logger.warning("⚠️  No topics to process")
            return 0
        
        logger.info(f"\n📋 Topics to process: {len(topics)}")
        logger.info("\n" + "="*60)
        logger.info("📝 COMPLETE TOPIC LIST")
        logger.info("="*60)
        
        # Sort topics alphabetically for easier reading
        sorted_topics = sorted(topics)
        
        # Display in columns for better readability
        cols = 3
        for i in range(0, len(sorted_topics), cols):
            row_topics = sorted_topics[i:i+cols]
            # Format with padding for alignment
            formatted = " | ".join(f"{t:<25}" for t in row_topics)
            logger.info(f"  {formatted}")
        
        logger.info("="*60)
        
        if args.dry_run:
            logger.info("\n🔍 DRY RUN MODE - No materials will be added")
            logger.info(f"Would process {len(topics)} topics")
            logger.info(f"Material types: {', '.join(material_types)}")
            return 0
        
        # Confirm before proceeding
        logger.info("\n" + "="*60)
        response = input(f"Proceed with populating materials for {len(topics)} topics? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            logger.info("❌ Cancelled by user")
            return 0
        
        # Populate materials
        logger.info("\n🚀 Starting material population...")
        results = await populate_materials_for_topics(
            topics=topics,
            material_types=material_types,
            max_per_topic=5,
            log_context=log_context
        )
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("📊 POPULATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Topics processed: {results['topics_processed']}")
        logger.info(f"Topics skipped: {results['topics_skipped']}")
        logger.info(f"Materials added: {results['materials_added']}")
        if results.get('errors'):
            logger.warning(f"Errors: {len(results['errors'])}")
            for error in results['errors'][:5]:
                logger.warning(f"  - {error}")
        
        logger.info("\n✅ Material population complete!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1


def main():
    return asyncio.run(main_async())


if __name__ == "__main__":
    exit(main())


Populate KB with additional materials (books, papers, videos, tutorials) 
for all existing course topics in the knowledge base.

This script:
1. Gets all courses from KB
2. Extracts unique topics/skills from course titles and skills
3. For each topic, discovers books, papers, videos, tutorials
4. Adds discovered materials to KB

Usage:
    python scripts/populate_materials_for_all_topics.py [--limit N] [--material-types book,paper,video,tutorial]
"""

import sys
import os
import argparse
import asyncio
import re
from typing import List, Dict, Any, Set
from collections import Counter

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.course_knowledge_base import CourseKnowledgeBase
from agents.market_and_course_recommender import _check_and_discover_materials
from core.logging_helpers import AgentLogger, create_log_context
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def extract_topics_from_courses(courses: List[Dict[str, Any]]) -> Set[str]:
    """
    Extract unique topics/skills from courses.
    
    Sources:
    1. Course skills field (primary source)
    2. Course title (extract known tech keywords only)
    
    Args:
        courses: List of course dictionaries
        
    Returns:
        Set of unique topic strings
    """
    topics = set()
    
    # Common tech keywords to extract from titles (only well-known tech terms)
    tech_keywords = [
        # Languages
        "python", "javascript", "java", "typescript", "go", "golang", "rust", "c++", "c#", "csharp",
        "swift", "kotlin", "php", "ruby", "scala", "r", "matlab", "dart",
        # Frameworks & Libraries
        "react", "angular", "vue", "node", "nodejs", "express", "django", "flask", "spring",
        "laravel", "nextjs", "next.js", "nuxt", "svelte", "ember",
        # Databases
        "sql", "mysql", "postgresql", "postgres", "mongodb", "mongo", "redis", "elasticsearch",
        "cassandra", "oracle", "sqlite",
        # Cloud & DevOps
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s", "terraform",
        "ansible", "jenkins", "ci/cd", "cicd", "devops", "sre", "mlops",
        # Domains
        "machine learning", "ml", "deep learning", "data science", "ai", "artificial intelligence",
        "blockchain", "cybersecurity", "web development", "mobile development", "backend",
        "frontend", "full stack", "fullstack",
        # Testing
        "testing", "qa", "selenium", "cypress", "jest", "pytest", "junit",
        # APIs & Protocols
        "api", "rest", "graphql", "grpc", "microservices",
        # Methodologies
        "agile", "scrum", "kanban"
    ]
    
    # Common words to exclude
    exclude_words = {
        "the", "and", "for", "with", "from", "course", "learn", "complete", "guide",
        "tutorial", "bootcamp", "masterclass", "certification", "certified", "training",
        "beginner", "intermediate", "advanced", "expert", "level", "fundamentals",
        "introduction", "intro", "basics", "essentials", "specialty", "specialization",
        "path", "track", "program", "series", "part", "volume", "edition", "version"
    }
    
    for course in courses:
        # PRIMARY: Extract from skills field (most reliable)
        skills = course.get("skills", [])
        if isinstance(skills, list):
            for skill in skills:
                if skill and isinstance(skill, str):
                    skill_clean = skill.strip().lower()
                    # Only add if it's a reasonable length and not excluded
                    if 2 < len(skill_clean) < 50 and skill_clean not in exclude_words:
                        topics.add(skill_clean)
        elif isinstance(skills, str):
            for skill in skills.split(","):
                skill_clean = skill.strip().lower()
                if skill_clean and 2 < len(skill_clean) < 50 and skill_clean not in exclude_words:
                    topics.add(skill_clean)
        
        # SECONDARY: Extract known tech keywords from title
        title = course.get("title", "").lower()
        if title:
            # Check for tech keywords in title
            for keyword in tech_keywords:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, title):
                    topics.add(keyword)
    
    # Final filtering
    filtered_topics = set()
    for topic in topics:
        # Skip if too short, too long, or contains excluded words
        if len(topic) < 2 or len(topic) > 50:
            continue
        
        # Skip if it's just common words
        words = topic.split()
        if all(word in exclude_words for word in words):
            continue
        
        # Skip if it contains common non-topic phrases
        if any(phrase in topic for phrase in ["specialty courses", "engineering path", "analytics specialty"]):
            continue
        
        # Only keep topics that look like real tech terms
        # Must be: single word, or 2-3 words max, and not all common words
        if len(words) <= 3:
            filtered_topics.add(topic)
    
    return filtered_topics


async def populate_materials_for_topics(
    topics: List[str],
    material_types: List[str],
    max_per_topic: int = 5,
    log_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Populate materials for a list of topics.
    
    Args:
        topics: List of topics to populate
        material_types: List of material types (book, paper, video, tutorial)
        max_per_topic: Maximum materials to discover per topic
        log_context: Logging context
        
    Returns:
        Dict with results
    """
    if not log_context:
        log_context = create_log_context("populate_materials", "system")
    
    results = {
        "topics_processed": 0,
        "materials_added": 0,
        "topics_skipped": 0,
        "errors": []
    }
    
    kb = CourseKnowledgeBase()
    
    for i, topic in enumerate(topics, 1):
        try:
            logger.info(f"\n[{i}/{len(topics)}] Processing topic: {topic}")
            
            # Check if we already have materials for this topic
            existing_materials = []
            for material_type in material_types:
                try:
                    filters = {"type": material_type}
                    found = kb.search_courses(
                        query=f"{topic} {material_type}",
                        top_k=5,
                        filters=filters
                    )
                    existing_materials.extend(found)
                except Exception as e:
                    logger.debug(f"Search failed for {material_type}: {e}")
            
            # If we already have enough materials, skip
            if len(existing_materials) >= max_per_topic:
                logger.info(f"  ⏭️  Skipping '{topic}' - already has {len(existing_materials)} materials")
                results["topics_skipped"] += 1
                continue
            
            # Discover materials (excluding courses since we already have those)
            logger.info(f"  🔍 Discovering {', '.join(material_types)} for '{topic}'...")
            
            discovered = await _check_and_discover_materials(
                topic=topic,
                material_types=material_types,
                min_results_required=3,
                min_similarity_threshold=0.4,
                max_results_per_type=3,
                log_context=log_context
            )
            
            if discovered:
                results["materials_added"] += len(discovered)
                logger.info(f"  ✅ Added {len(discovered)} materials for '{topic}'")
            else:
                logger.info(f"  ⚠️  No materials found for '{topic}'")
            
            results["topics_processed"] += 1
            
            # Rate limiting - small delay between topics
            await asyncio.sleep(1)
            
        except Exception as e:
            error_msg = f"Error processing topic '{topic}': {e}"
            results["errors"].append(error_msg)
            logger.error(f"  ❌ {error_msg}")
            continue
    
    return results


async def main_async():
    parser = argparse.ArgumentParser(
        description="Populate KB with additional materials for all course topics"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of topics to process (default: all)"
    )
    parser.add_argument(
        "--material-types",
        type=str,
        default="book,paper,video,tutorial",
        help="Comma-separated list of material types (default: book,paper,video,tutorial)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually adding materials"
    )
    parser.add_argument(
        "--min-courses",
        type=int,
        default=1,
        help="Minimum number of courses a topic must have to be included (default: 1)"
    )
    
    args = parser.parse_args()
    
    try:
        log_context = create_log_context("populate_materials", "system")
        
        # Parse material types
        material_types = [t.strip() for t in args.material_types.split(",") if t.strip()]
        # Ensure we don't include "course" since we're adding other materials
        material_types = [t for t in material_types if t != "course"]
        
        if not material_types:
            logger.error("❌ No valid material types specified (excluding 'course')")
            return 1
        
        logger.info("="*60)
        logger.info("📚 POPULATE MATERIALS FOR ALL COURSE TOPICS")
        logger.info("="*60)
        logger.info(f"Material types: {', '.join(material_types)}")
        logger.info(f"Dry run: {args.dry_run}")
        logger.info("")
        
        # Get all courses from KB
        logger.info("🔍 Fetching all courses from knowledge base...")
        kb = CourseKnowledgeBase()
        all_courses = kb.get_all_courses()
        
        if not all_courses:
            logger.error("❌ No courses found in knowledge base")
            return 1
        
        logger.info(f"✅ Found {len(all_courses)} courses in KB")
        
        # Extract unique topics
        logger.info("\n📊 Extracting topics from courses...")
        topics = extract_topics_from_courses(all_courses)
        logger.info(f"✅ Extracted {len(topics)} unique topics")
        
        # Filter topics that have at least min_courses
        if args.min_courses > 1:
            topic_counts = Counter()
            for course in all_courses:
                skills = course.get("skills", [])
                if isinstance(skills, list):
                    for skill in skills:
                        if skill:
                            topic_counts[skill.lower()] += 1
                elif isinstance(skills, str):
                    for skill in skills.split(","):
                        if skill.strip():
                            topic_counts[skill.strip().lower()] += 1
            
            topics = {t for t in topics if topic_counts.get(t, 0) >= args.min_courses}
            logger.info(f"✅ Filtered to {len(topics)} topics with ≥{args.min_courses} courses")
        
        # Limit topics if specified
        if args.limit:
            topics = list(topics)[:args.limit]
            logger.info(f"📌 Limited to {len(topics)} topics")
        else:
            topics = list(topics)
        
        if not topics:
            logger.warning("⚠️  No topics to process")
            return 0
        
        logger.info(f"\n📋 Topics to process: {len(topics)}")
        logger.info("\n" + "="*60)
        logger.info("📝 COMPLETE TOPIC LIST")
        logger.info("="*60)
        
        # Sort topics alphabetically for easier reading
        sorted_topics = sorted(topics)
        
        # Display in columns for better readability
        cols = 3
        for i in range(0, len(sorted_topics), cols):
            row_topics = sorted_topics[i:i+cols]
            # Format with padding for alignment
            formatted = " | ".join(f"{t:<25}" for t in row_topics)
            logger.info(f"  {formatted}")
        
        logger.info("="*60)
        
        if args.dry_run:
            logger.info("\n🔍 DRY RUN MODE - No materials will be added")
            logger.info(f"Would process {len(topics)} topics")
            logger.info(f"Material types: {', '.join(material_types)}")
            return 0
        
        # Confirm before proceeding
        logger.info("\n" + "="*60)
        response = input(f"Proceed with populating materials for {len(topics)} topics? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            logger.info("❌ Cancelled by user")
            return 0
        
        # Populate materials
        logger.info("\n🚀 Starting material population...")
        results = await populate_materials_for_topics(
            topics=topics,
            material_types=material_types,
            max_per_topic=5,
            log_context=log_context
        )
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("📊 POPULATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Topics processed: {results['topics_processed']}")
        logger.info(f"Topics skipped: {results['topics_skipped']}")
        logger.info(f"Materials added: {results['materials_added']}")
        if results.get('errors'):
            logger.warning(f"Errors: {len(results['errors'])}")
            for error in results['errors'][:5]:
                logger.warning(f"  - {error}")
        
        logger.info("\n✅ Material population complete!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1


def main():
    return asyncio.run(main_async())


if __name__ == "__main__":
    exit(main())


Populate KB with additional materials (books, papers, videos, tutorials) 
for all existing course topics in the knowledge base.

This script:
1. Gets all courses from KB
2. Extracts unique topics/skills from course titles and skills
3. For each topic, discovers books, papers, videos, tutorials
4. Adds discovered materials to KB

Usage:
    python scripts/populate_materials_for_all_topics.py [--limit N] [--material-types book,paper,video,tutorial]
"""

import sys
import os
import argparse
import asyncio
import re
from typing import List, Dict, Any, Set
from collections import Counter

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.course_knowledge_base import CourseKnowledgeBase
from agents.market_and_course_recommender import _check_and_discover_materials
from core.logging_helpers import AgentLogger, create_log_context
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def extract_topics_from_courses(courses: List[Dict[str, Any]]) -> Set[str]:
    """
    Extract unique topics/skills from courses.
    
    Sources:
    1. Course skills field (primary source)
    2. Course title (extract known tech keywords only)
    
    Args:
        courses: List of course dictionaries
        
    Returns:
        Set of unique topic strings
    """
    topics = set()
    
    # Common tech keywords to extract from titles (only well-known tech terms)
    tech_keywords = [
        # Languages
        "python", "javascript", "java", "typescript", "go", "golang", "rust", "c++", "c#", "csharp",
        "swift", "kotlin", "php", "ruby", "scala", "r", "matlab", "dart",
        # Frameworks & Libraries
        "react", "angular", "vue", "node", "nodejs", "express", "django", "flask", "spring",
        "laravel", "nextjs", "next.js", "nuxt", "svelte", "ember",
        # Databases
        "sql", "mysql", "postgresql", "postgres", "mongodb", "mongo", "redis", "elasticsearch",
        "cassandra", "oracle", "sqlite",
        # Cloud & DevOps
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s", "terraform",
        "ansible", "jenkins", "ci/cd", "cicd", "devops", "sre", "mlops",
        # Domains
        "machine learning", "ml", "deep learning", "data science", "ai", "artificial intelligence",
        "blockchain", "cybersecurity", "web development", "mobile development", "backend",
        "frontend", "full stack", "fullstack",
        # Testing
        "testing", "qa", "selenium", "cypress", "jest", "pytest", "junit",
        # APIs & Protocols
        "api", "rest", "graphql", "grpc", "microservices",
        # Methodologies
        "agile", "scrum", "kanban"
    ]
    
    # Common words to exclude
    exclude_words = {
        "the", "and", "for", "with", "from", "course", "learn", "complete", "guide",
        "tutorial", "bootcamp", "masterclass", "certification", "certified", "training",
        "beginner", "intermediate", "advanced", "expert", "level", "fundamentals",
        "introduction", "intro", "basics", "essentials", "specialty", "specialization",
        "path", "track", "program", "series", "part", "volume", "edition", "version"
    }
    
    for course in courses:
        # PRIMARY: Extract from skills field (most reliable)
        skills = course.get("skills", [])
        if isinstance(skills, list):
            for skill in skills:
                if skill and isinstance(skill, str):
                    skill_clean = skill.strip().lower()
                    # Only add if it's a reasonable length and not excluded
                    if 2 < len(skill_clean) < 50 and skill_clean not in exclude_words:
                        topics.add(skill_clean)
        elif isinstance(skills, str):
            for skill in skills.split(","):
                skill_clean = skill.strip().lower()
                if skill_clean and 2 < len(skill_clean) < 50 and skill_clean not in exclude_words:
                    topics.add(skill_clean)
        
        # SECONDARY: Extract known tech keywords from title
        title = course.get("title", "").lower()
        if title:
            # Check for tech keywords in title
            for keyword in tech_keywords:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, title):
                    topics.add(keyword)
    
    # Final filtering
    filtered_topics = set()
    for topic in topics:
        # Skip if too short, too long, or contains excluded words
        if len(topic) < 2 or len(topic) > 50:
            continue
        
        # Skip if it's just common words
        words = topic.split()
        if all(word in exclude_words for word in words):
            continue
        
        # Skip if it contains common non-topic phrases
        if any(phrase in topic for phrase in ["specialty courses", "engineering path", "analytics specialty"]):
            continue
        
        # Only keep topics that look like real tech terms
        # Must be: single word, or 2-3 words max, and not all common words
        if len(words) <= 3:
            filtered_topics.add(topic)
    
    return filtered_topics


async def populate_materials_for_topics(
    topics: List[str],
    material_types: List[str],
    max_per_topic: int = 5,
    log_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Populate materials for a list of topics.
    
    Args:
        topics: List of topics to populate
        material_types: List of material types (book, paper, video, tutorial)
        max_per_topic: Maximum materials to discover per topic
        log_context: Logging context
        
    Returns:
        Dict with results
    """
    if not log_context:
        log_context = create_log_context("populate_materials", "system")
    
    results = {
        "topics_processed": 0,
        "materials_added": 0,
        "topics_skipped": 0,
        "errors": []
    }
    
    kb = CourseKnowledgeBase()
    
    for i, topic in enumerate(topics, 1):
        try:
            logger.info(f"\n[{i}/{len(topics)}] Processing topic: {topic}")
            
            # Check if we already have materials for this topic
            existing_materials = []
            for material_type in material_types:
                try:
                    filters = {"type": material_type}
                    found = kb.search_courses(
                        query=f"{topic} {material_type}",
                        top_k=5,
                        filters=filters
                    )
                    existing_materials.extend(found)
                except Exception as e:
                    logger.debug(f"Search failed for {material_type}: {e}")
            
            # If we already have enough materials, skip
            if len(existing_materials) >= max_per_topic:
                logger.info(f"  ⏭️  Skipping '{topic}' - already has {len(existing_materials)} materials")
                results["topics_skipped"] += 1
                continue
            
            # Discover materials (excluding courses since we already have those)
            logger.info(f"  🔍 Discovering {', '.join(material_types)} for '{topic}'...")
            
            discovered = await _check_and_discover_materials(
                topic=topic,
                material_types=material_types,
                min_results_required=3,
                min_similarity_threshold=0.4,
                max_results_per_type=3,
                log_context=log_context
            )
            
            if discovered:
                results["materials_added"] += len(discovered)
                logger.info(f"  ✅ Added {len(discovered)} materials for '{topic}'")
            else:
                logger.info(f"  ⚠️  No materials found for '{topic}'")
            
            results["topics_processed"] += 1
            
            # Rate limiting - small delay between topics
            await asyncio.sleep(1)
            
        except Exception as e:
            error_msg = f"Error processing topic '{topic}': {e}"
            results["errors"].append(error_msg)
            logger.error(f"  ❌ {error_msg}")
            continue
    
    return results


async def main_async():
    parser = argparse.ArgumentParser(
        description="Populate KB with additional materials for all course topics"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of topics to process (default: all)"
    )
    parser.add_argument(
        "--material-types",
        type=str,
        default="book,paper,video,tutorial",
        help="Comma-separated list of material types (default: book,paper,video,tutorial)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually adding materials"
    )
    parser.add_argument(
        "--min-courses",
        type=int,
        default=1,
        help="Minimum number of courses a topic must have to be included (default: 1)"
    )
    
    args = parser.parse_args()
    
    try:
        log_context = create_log_context("populate_materials", "system")
        
        # Parse material types
        material_types = [t.strip() for t in args.material_types.split(",") if t.strip()]
        # Ensure we don't include "course" since we're adding other materials
        material_types = [t for t in material_types if t != "course"]
        
        if not material_types:
            logger.error("❌ No valid material types specified (excluding 'course')")
            return 1
        
        logger.info("="*60)
        logger.info("📚 POPULATE MATERIALS FOR ALL COURSE TOPICS")
        logger.info("="*60)
        logger.info(f"Material types: {', '.join(material_types)}")
        logger.info(f"Dry run: {args.dry_run}")
        logger.info("")
        
        # Get all courses from KB
        logger.info("🔍 Fetching all courses from knowledge base...")
        kb = CourseKnowledgeBase()
        all_courses = kb.get_all_courses()
        
        if not all_courses:
            logger.error("❌ No courses found in knowledge base")
            return 1
        
        logger.info(f"✅ Found {len(all_courses)} courses in KB")
        
        # Extract unique topics
        logger.info("\n📊 Extracting topics from courses...")
        topics = extract_topics_from_courses(all_courses)
        logger.info(f"✅ Extracted {len(topics)} unique topics")
        
        # Filter topics that have at least min_courses
        if args.min_courses > 1:
            topic_counts = Counter()
            for course in all_courses:
                skills = course.get("skills", [])
                if isinstance(skills, list):
                    for skill in skills:
                        if skill:
                            topic_counts[skill.lower()] += 1
                elif isinstance(skills, str):
                    for skill in skills.split(","):
                        if skill.strip():
                            topic_counts[skill.strip().lower()] += 1
            
            topics = {t for t in topics if topic_counts.get(t, 0) >= args.min_courses}
            logger.info(f"✅ Filtered to {len(topics)} topics with ≥{args.min_courses} courses")
        
        # Limit topics if specified
        if args.limit:
            topics = list(topics)[:args.limit]
            logger.info(f"📌 Limited to {len(topics)} topics")
        else:
            topics = list(topics)
        
        if not topics:
            logger.warning("⚠️  No topics to process")
            return 0
        
        logger.info(f"\n📋 Topics to process: {len(topics)}")
        logger.info("\n" + "="*60)
        logger.info("📝 COMPLETE TOPIC LIST")
        logger.info("="*60)
        
        # Sort topics alphabetically for easier reading
        sorted_topics = sorted(topics)
        
        # Display in columns for better readability
        cols = 3
        for i in range(0, len(sorted_topics), cols):
            row_topics = sorted_topics[i:i+cols]
            # Format with padding for alignment
            formatted = " | ".join(f"{t:<25}" for t in row_topics)
            logger.info(f"  {formatted}")
        
        logger.info("="*60)
        
        if args.dry_run:
            logger.info("\n🔍 DRY RUN MODE - No materials will be added")
            logger.info(f"Would process {len(topics)} topics")
            logger.info(f"Material types: {', '.join(material_types)}")
            return 0
        
        # Confirm before proceeding
        logger.info("\n" + "="*60)
        response = input(f"Proceed with populating materials for {len(topics)} topics? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            logger.info("❌ Cancelled by user")
            return 0
        
        # Populate materials
        logger.info("\n🚀 Starting material population...")
        results = await populate_materials_for_topics(
            topics=topics,
            material_types=material_types,
            max_per_topic=5,
            log_context=log_context
        )
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("📊 POPULATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Topics processed: {results['topics_processed']}")
        logger.info(f"Topics skipped: {results['topics_skipped']}")
        logger.info(f"Materials added: {results['materials_added']}")
        if results.get('errors'):
            logger.warning(f"Errors: {len(results['errors'])}")
            for error in results['errors'][:5]:
                logger.warning(f"  - {error}")
        
        logger.info("\n✅ Material population complete!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1


def main():
    return asyncio.run(main_async())


if __name__ == "__main__":
    exit(main())


Populate KB with additional materials (books, papers, videos, tutorials) 
for all existing course topics in the knowledge base.

This script:
1. Gets all courses from KB
2. Extracts unique topics/skills from course titles and skills
3. For each topic, discovers books, papers, videos, tutorials
4. Adds discovered materials to KB

Usage:
    python scripts/populate_materials_for_all_topics.py [--limit N] [--material-types book,paper,video,tutorial]
"""

import sys
import os
import argparse
import asyncio
import re
from typing import List, Dict, Any, Set
from collections import Counter

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.course_knowledge_base import CourseKnowledgeBase
from agents.market_and_course_recommender import _check_and_discover_materials
from core.logging_helpers import AgentLogger, create_log_context
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def extract_topics_from_courses(courses: List[Dict[str, Any]]) -> Set[str]:
    """
    Extract unique topics/skills from courses.
    
    Sources:
    1. Course skills field (primary source)
    2. Course title (extract known tech keywords only)
    
    Args:
        courses: List of course dictionaries
        
    Returns:
        Set of unique topic strings
    """
    topics = set()
    
    # Common tech keywords to extract from titles (only well-known tech terms)
    tech_keywords = [
        # Languages
        "python", "javascript", "java", "typescript", "go", "golang", "rust", "c++", "c#", "csharp",
        "swift", "kotlin", "php", "ruby", "scala", "r", "matlab", "dart",
        # Frameworks & Libraries
        "react", "angular", "vue", "node", "nodejs", "express", "django", "flask", "spring",
        "laravel", "nextjs", "next.js", "nuxt", "svelte", "ember",
        # Databases
        "sql", "mysql", "postgresql", "postgres", "mongodb", "mongo", "redis", "elasticsearch",
        "cassandra", "oracle", "sqlite",
        # Cloud & DevOps
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s", "terraform",
        "ansible", "jenkins", "ci/cd", "cicd", "devops", "sre", "mlops",
        # Domains
        "machine learning", "ml", "deep learning", "data science", "ai", "artificial intelligence",
        "blockchain", "cybersecurity", "web development", "mobile development", "backend",
        "frontend", "full stack", "fullstack",
        # Testing
        "testing", "qa", "selenium", "cypress", "jest", "pytest", "junit",
        # APIs & Protocols
        "api", "rest", "graphql", "grpc", "microservices",
        # Methodologies
        "agile", "scrum", "kanban"
    ]
    
    # Common words to exclude
    exclude_words = {
        "the", "and", "for", "with", "from", "course", "learn", "complete", "guide",
        "tutorial", "bootcamp", "masterclass", "certification", "certified", "training",
        "beginner", "intermediate", "advanced", "expert", "level", "fundamentals",
        "introduction", "intro", "basics", "essentials", "specialty", "specialization",
        "path", "track", "program", "series", "part", "volume", "edition", "version"
    }
    
    for course in courses:
        # PRIMARY: Extract from skills field (most reliable)
        skills = course.get("skills", [])
        if isinstance(skills, list):
            for skill in skills:
                if skill and isinstance(skill, str):
                    skill_clean = skill.strip().lower()
                    # Only add if it's a reasonable length and not excluded
                    if 2 < len(skill_clean) < 50 and skill_clean not in exclude_words:
                        topics.add(skill_clean)
        elif isinstance(skills, str):
            for skill in skills.split(","):
                skill_clean = skill.strip().lower()
                if skill_clean and 2 < len(skill_clean) < 50 and skill_clean not in exclude_words:
                    topics.add(skill_clean)
        
        # SECONDARY: Extract known tech keywords from title
        title = course.get("title", "").lower()
        if title:
            # Check for tech keywords in title
            for keyword in tech_keywords:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, title):
                    topics.add(keyword)
    
    # Final filtering
    filtered_topics = set()
    for topic in topics:
        # Skip if too short, too long, or contains excluded words
        if len(topic) < 2 or len(topic) > 50:
            continue
        
        # Skip if it's just common words
        words = topic.split()
        if all(word in exclude_words for word in words):
            continue
        
        # Skip if it contains common non-topic phrases
        if any(phrase in topic for phrase in ["specialty courses", "engineering path", "analytics specialty"]):
            continue
        
        # Only keep topics that look like real tech terms
        # Must be: single word, or 2-3 words max, and not all common words
        if len(words) <= 3:
            filtered_topics.add(topic)
    
    return filtered_topics


async def populate_materials_for_topics(
    topics: List[str],
    material_types: List[str],
    max_per_topic: int = 5,
    log_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Populate materials for a list of topics.
    
    Args:
        topics: List of topics to populate
        material_types: List of material types (book, paper, video, tutorial)
        max_per_topic: Maximum materials to discover per topic
        log_context: Logging context
        
    Returns:
        Dict with results
    """
    if not log_context:
        log_context = create_log_context("populate_materials", "system")
    
    results = {
        "topics_processed": 0,
        "materials_added": 0,
        "topics_skipped": 0,
        "errors": []
    }
    
    kb = CourseKnowledgeBase()
    
    for i, topic in enumerate(topics, 1):
        try:
            logger.info(f"\n[{i}/{len(topics)}] Processing topic: {topic}")
            
            # Check if we already have materials for this topic
            existing_materials = []
            for material_type in material_types:
                try:
                    filters = {"type": material_type}
                    found = kb.search_courses(
                        query=f"{topic} {material_type}",
                        top_k=5,
                        filters=filters
                    )
                    existing_materials.extend(found)
                except Exception as e:
                    logger.debug(f"Search failed for {material_type}: {e}")
            
            # If we already have enough materials, skip
            if len(existing_materials) >= max_per_topic:
                logger.info(f"  ⏭️  Skipping '{topic}' - already has {len(existing_materials)} materials")
                results["topics_skipped"] += 1
                continue
            
            # Discover materials (excluding courses since we already have those)
            logger.info(f"  🔍 Discovering {', '.join(material_types)} for '{topic}'...")
            
            discovered = await _check_and_discover_materials(
                topic=topic,
                material_types=material_types,
                min_results_required=3,
                min_similarity_threshold=0.4,
                max_results_per_type=3,
                log_context=log_context
            )
            
            if discovered:
                results["materials_added"] += len(discovered)
                logger.info(f"  ✅ Added {len(discovered)} materials for '{topic}'")
            else:
                logger.info(f"  ⚠️  No materials found for '{topic}'")
            
            results["topics_processed"] += 1
            
            # Rate limiting - small delay between topics
            await asyncio.sleep(1)
            
        except Exception as e:
            error_msg = f"Error processing topic '{topic}': {e}"
            results["errors"].append(error_msg)
            logger.error(f"  ❌ {error_msg}")
            continue
    
    return results


async def main_async():
    parser = argparse.ArgumentParser(
        description="Populate KB with additional materials for all course topics"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of topics to process (default: all)"
    )
    parser.add_argument(
        "--material-types",
        type=str,
        default="book,paper,video,tutorial",
        help="Comma-separated list of material types (default: book,paper,video,tutorial)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually adding materials"
    )
    parser.add_argument(
        "--min-courses",
        type=int,
        default=1,
        help="Minimum number of courses a topic must have to be included (default: 1)"
    )
    
    args = parser.parse_args()
    
    try:
        log_context = create_log_context("populate_materials", "system")
        
        # Parse material types
        material_types = [t.strip() for t in args.material_types.split(",") if t.strip()]
        # Ensure we don't include "course" since we're adding other materials
        material_types = [t for t in material_types if t != "course"]
        
        if not material_types:
            logger.error("❌ No valid material types specified (excluding 'course')")
            return 1
        
        logger.info("="*60)
        logger.info("📚 POPULATE MATERIALS FOR ALL COURSE TOPICS")
        logger.info("="*60)
        logger.info(f"Material types: {', '.join(material_types)}")
        logger.info(f"Dry run: {args.dry_run}")
        logger.info("")
        
        # Get all courses from KB
        logger.info("🔍 Fetching all courses from knowledge base...")
        kb = CourseKnowledgeBase()
        all_courses = kb.get_all_courses()
        
        if not all_courses:
            logger.error("❌ No courses found in knowledge base")
            return 1
        
        logger.info(f"✅ Found {len(all_courses)} courses in KB")
        
        # Extract unique topics
        logger.info("\n📊 Extracting topics from courses...")
        topics = extract_topics_from_courses(all_courses)
        logger.info(f"✅ Extracted {len(topics)} unique topics")
        
        # Filter topics that have at least min_courses
        if args.min_courses > 1:
            topic_counts = Counter()
            for course in all_courses:
                skills = course.get("skills", [])
                if isinstance(skills, list):
                    for skill in skills:
                        if skill:
                            topic_counts[skill.lower()] += 1
                elif isinstance(skills, str):
                    for skill in skills.split(","):
                        if skill.strip():
                            topic_counts[skill.strip().lower()] += 1
            
            topics = {t for t in topics if topic_counts.get(t, 0) >= args.min_courses}
            logger.info(f"✅ Filtered to {len(topics)} topics with ≥{args.min_courses} courses")
        
        # Limit topics if specified
        if args.limit:
            topics = list(topics)[:args.limit]
            logger.info(f"📌 Limited to {len(topics)} topics")
        else:
            topics = list(topics)
        
        if not topics:
            logger.warning("⚠️  No topics to process")
            return 0
        
        logger.info(f"\n📋 Topics to process: {len(topics)}")
        logger.info("\n" + "="*60)
        logger.info("📝 COMPLETE TOPIC LIST")
        logger.info("="*60)
        
        # Sort topics alphabetically for easier reading
        sorted_topics = sorted(topics)
        
        # Display in columns for better readability
        cols = 3
        for i in range(0, len(sorted_topics), cols):
            row_topics = sorted_topics[i:i+cols]
            # Format with padding for alignment
            formatted = " | ".join(f"{t:<25}" for t in row_topics)
            logger.info(f"  {formatted}")
        
        logger.info("="*60)
        
        if args.dry_run:
            logger.info("\n🔍 DRY RUN MODE - No materials will be added")
            logger.info(f"Would process {len(topics)} topics")
            logger.info(f"Material types: {', '.join(material_types)}")
            return 0
        
        # Confirm before proceeding
        logger.info("\n" + "="*60)
        response = input(f"Proceed with populating materials for {len(topics)} topics? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            logger.info("❌ Cancelled by user")
            return 0
        
        # Populate materials
        logger.info("\n🚀 Starting material population...")
        results = await populate_materials_for_topics(
            topics=topics,
            material_types=material_types,
            max_per_topic=5,
            log_context=log_context
        )
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("📊 POPULATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Topics processed: {results['topics_processed']}")
        logger.info(f"Topics skipped: {results['topics_skipped']}")
        logger.info(f"Materials added: {results['materials_added']}")
        if results.get('errors'):
            logger.warning(f"Errors: {len(results['errors'])}")
            for error in results['errors'][:5]:
                logger.warning(f"  - {error}")
        
        logger.info("\n✅ Material population complete!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1


def main():
    return asyncio.run(main_async())


if __name__ == "__main__":
    exit(main())

