#!/usr/bin/env python3
"""
Script to remove invalid course URLs from the knowledge base.

Usage:
    # Remove specific URLs
    python scripts/remove_invalid_courses.py --urls "https://url1.com" "https://url2.com"
    
    # Validate all courses and remove invalid ones
    python scripts/remove_invalid_courses.py --validate-all
    
    # Validate courses from a specific provider
    python scripts/remove_invalid_courses.py --validate-provider "Udemy"
"""

import sys
import os
import argparse
import asyncio
import aiohttp

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

async def check_url_accessibility(url: str, timeout: int = 5) -> tuple[bool, str, int | None]:
    """
    Check if a URL is accessible.
    
    Returns:
        (is_accessible, message, status_code)
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout),
            headers=headers
        ) as session:
            async with session.get(url, allow_redirects=True) as response:
                status = response.status
                if status == 200:
                    return True, f"URL accessible (HTTP {status})", status
                if status in [301, 302, 303, 307, 308]:
                    final_url = str(response.url)
                    return True, f"URL accessible with redirect (HTTP {status})", status
                return False, f"URL not accessible (HTTP {status})", status
                
    except aiohttp.ClientTimeout:
        return False, "URL check timed out", None
    except aiohttp.ClientError as e:
        return False, f"Connection error: {str(e)}", None
    except Exception as e:
        return False, f"Error: {str(e)}", None

async def validate_courses_async(courses: list[dict], max_concurrent: int = 10) -> list[dict]:
    """
    Validate multiple courses concurrently.
    
    Returns:
        List of invalid courses (courses that failed validation)
    """
    invalid_courses = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def validate_one(course: dict) -> tuple[dict, bool]:
        async with semaphore:
            url = course.get("url", "")
            if not url:
                return course, False  # No URL to validate
            
            is_accessible, message, status_code = await check_url_accessibility(url)
            
            if not is_accessible:
                logger.warning(
                    f"❌ Invalid: {course.get('title', 'Unknown')} - {message} "
                    f"(Status: {status_code})"
                )
                return course, True  # Invalid
            else:
                logger.debug(f"✅ Valid: {course.get('title', 'Unknown')}")
                return course, False  # Valid
    
    tasks = [validate_one(course) for course in courses]
    results = await asyncio.gather(*tasks)
    
    for course, is_invalid in results:
        if is_invalid:
            invalid_courses.append(course)
    
    return invalid_courses

def remove_specific_urls(urls: list[str]) -> int:
    """Remove specific URLs from knowledge base"""
    kb = CourseKnowledgeBase()
    deleted_count = kb.delete_courses_by_urls(urls)
    return deleted_count

async def validate_and_remove_invalid(provider: str | None = None) -> dict:
    """
    Validate all courses (or from a specific provider) and remove invalid ones.
    
    Returns:
        Dict with validation results
    """
    kb = CourseKnowledgeBase()
    
    # Get courses to validate
    if provider:
        logger.info(f"📚 Getting courses from provider: {provider}")
        courses = kb.get_courses_by_provider(provider)
    else:
        logger.info("📚 Getting all courses from knowledge base...")
        courses = kb.get_all_courses()
    
    total_courses = len(courses)
    logger.info(f"📊 Found {total_courses} courses to validate")
    
    if total_courses == 0:
        return {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "deleted": 0
        }
    
    # Validate courses
    logger.info("🔍 Validating course URLs...")
    invalid_courses = await validate_courses_async(courses)
    
    invalid_count = len(invalid_courses)
    valid_count = total_courses - invalid_count
    
    # Delete invalid courses
    deleted_count = 0
    if invalid_courses:
        logger.info(f"\n🗑️  Removing {invalid_count} invalid courses...")
        for course in invalid_courses:
            url = course.get("url", "")
            if url:
                deleted = kb.delete_courses_by_url(url)
                if deleted > 0:
                    deleted_count += deleted
                    logger.info(f"  ✅ Deleted: {course.get('title', 'Unknown')}")
    
    return {
        "total": total_courses,
        "valid": valid_count,
        "invalid": invalid_count,
        "deleted": deleted_count
    }

def main():
    parser = argparse.ArgumentParser(
        description="Remove invalid course URLs from knowledge base"
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        help="Specific URLs to remove"
    )
    parser.add_argument(
        "--validate-all",
        action="store_true",
        help="Validate all courses and remove invalid ones"
    )
    parser.add_argument(
        "--validate-provider",
        type=str,
        help="Validate courses from a specific provider (e.g., 'Udemy', 'Coursera')"
    )
    
    args = parser.parse_args()
    
    try:
        if args.urls:
            # Remove specific URLs
            logger.info(f"🚀 Removing {len(args.urls)} specific URLs...")
            for url in args.urls:
                logger.info(f"  - {url}")
            
            deleted_count = remove_specific_urls(args.urls)
            logger.info(f"\n✅ Successfully removed {deleted_count} course(s)")
            return 0
            
        elif args.validate_all:
            # Validate all courses
            logger.info("🚀 Starting validation of all courses...")
            results = asyncio.run(validate_and_remove_invalid())
            
            logger.info("\n" + "="*60)
            logger.info("📊 VALIDATION SUMMARY")
            logger.info("="*60)
            logger.info(f"Total courses: {results['total']}")
            logger.info(f"Valid courses: {results['valid']}")
            logger.info(f"Invalid courses: {results['invalid']}")
            logger.info(f"Deleted courses: {results['deleted']}")
            
            return 0
            
        elif args.validate_provider:
            # Validate courses from specific provider
            logger.info(f"🚀 Starting validation of {args.validate_provider} courses...")
            results = asyncio.run(validate_and_remove_invalid(provider=args.validate_provider))
            
            logger.info("\n" + "="*60)
            logger.info(f"📊 VALIDATION SUMMARY ({args.validate_provider})")
            logger.info("="*60)
            logger.info(f"Total courses: {results['total']}")
            logger.info(f"Valid courses: {results['valid']}")
            logger.info(f"Invalid courses: {results['invalid']}")
            logger.info(f"Deleted courses: {results['deleted']}")
            
            return 0
            
        else:
            parser.print_help()
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())
