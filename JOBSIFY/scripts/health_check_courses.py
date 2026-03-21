#!/usr/bin/env python3
"""
Health check script for course knowledge base.
Validates all course URLs and generates a report of invalid courses.

This can be run periodically (e.g., via cron) to maintain course quality.

Usage:
    # Full health check
    python scripts/health_check_courses.py
    
    # Check specific provider
    python scripts/health_check_courses.py --provider "Udemy"
    
    # Generate report only (don't delete)
    python scripts/health_check_courses.py --report-only
    
    # Auto-remove invalid courses
    python scripts/health_check_courses.py --auto-remove
"""

import sys
import os
import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.course_knowledge_base import CourseKnowledgeBase
from scripts.remove_invalid_courses import validate_courses_async
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def generate_report(invalid_courses: list, output_file: str = None) -> dict:
    """
    Generate a health check report.
    
    Args:
        invalid_courses: List of invalid courses
        output_file: Optional file path to save report
        
    Returns:
        Report dictionary
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_invalid": len(invalid_courses),
        "invalid_courses": [
            {
                "course_id": c.get("course_id", ""),
                "title": c.get("title", ""),
                "provider": c.get("provider", ""),
                "url": c.get("url", ""),
                "difficulty": c.get("difficulty", ""),
                "type": c.get("type", "")
            }
            for c in invalid_courses
        ],
        "summary_by_provider": {}
    }
    
    # Group by provider
    for course in invalid_courses:
        provider = course.get("provider", "Unknown")
        report["summary_by_provider"][provider] = report["summary_by_provider"].get(provider, 0) + 1
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"📄 Report saved to: {output_file}")
    
    return report

async def run_health_check(
    provider: str = None,
    auto_remove: bool = False,
    report_only: bool = False,
    report_file: str = None
) -> dict:
    """
    Run health check on course knowledge base.
    
    Args:
        provider: Optional provider to check (None = all)
        auto_remove: If True, automatically remove invalid courses
        report_only: If True, only generate report, don't remove
        report_file: Optional path to save report
        
    Returns:
        Health check results
    """
    kb = CourseKnowledgeBase()
    
    # Get courses to check
    if provider:
        logger.info(f"📚 Checking courses from provider: {provider}")
        courses = kb.get_courses_by_provider(provider)
    else:
        logger.info("📚 Checking all courses in knowledge base...")
        courses = kb.get_all_courses()
    
    total_courses = len(courses)
    logger.info(f"📊 Total courses to check: {total_courses}")
    
    if total_courses == 0:
        logger.warning("⚠️  No courses found to check")
        return {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "deleted": 0
        }
    
    # Validate courses
    logger.info("🔍 Validating course URLs (this may take a while)...")
    invalid_courses = await validate_courses_async(courses)
    
    invalid_count = len(invalid_courses)
    valid_count = total_courses - invalid_count
    
    # Generate report
    report = generate_report(invalid_courses, report_file)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("📊 HEALTH CHECK SUMMARY")
    logger.info("="*60)
    logger.info(f"Total courses checked: {total_courses}")
    logger.info(f"✅ Valid courses: {valid_count}")
    logger.info(f"❌ Invalid courses: {invalid_count}")
    
    if invalid_count > 0:
        logger.info("\n📋 Invalid courses by provider:")
        for provider, count in report["summary_by_provider"].items():
            logger.info(f"  - {provider}: {count}")
    
    # Remove invalid courses if requested
    deleted_count = 0
    if invalid_courses and auto_remove and not report_only:
        logger.info(f"\n🗑️  Auto-removing {invalid_count} invalid courses...")
        for course in invalid_courses:
            url = course.get("url", "")
            if url:
                deleted = kb.delete_courses_by_url(url)
                if deleted > 0:
                    deleted_count += deleted
                    logger.info(f"  ✅ Deleted: {course.get('title', 'Unknown')}")
        logger.info(f"✅ Removed {deleted_count} invalid courses")
    elif invalid_courses and report_only:
        logger.info("\nℹ️  Report-only mode: Invalid courses not removed")
        logger.info("   Run with --auto-remove to remove invalid courses")
    
    return {
        "total": total_courses,
        "valid": valid_count,
        "invalid": invalid_count,
        "deleted": deleted_count,
        "report": report
    }

def main():
    parser = argparse.ArgumentParser(
        description="Health check for course knowledge base"
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="Check courses from specific provider (e.g., 'Udemy', 'Coursera')"
    )
    parser.add_argument(
        "--auto-remove",
        action="store_true",
        help="Automatically remove invalid courses"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report only, don't remove courses"
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default="course_health_report.json",
        help="Path to save health check report (default: course_health_report.json)"
    )
    
    args = parser.parse_args()
    
    try:
        results = asyncio.run(run_health_check(
            provider=args.provider,
            auto_remove=args.auto_remove,
            report_only=args.report_only,
            report_file=args.report_file
        ))
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())


Health check script for course knowledge base.
Validates all course URLs and generates a report of invalid courses.

This can be run periodically (e.g., via cron) to maintain course quality.

Usage:
    # Full health check
    python scripts/health_check_courses.py
    
    # Check specific provider
    python scripts/health_check_courses.py --provider "Udemy"
    
    # Generate report only (don't delete)
    python scripts/health_check_courses.py --report-only
    
    # Auto-remove invalid courses
    python scripts/health_check_courses.py --auto-remove
"""

import sys
import os
import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.course_knowledge_base import CourseKnowledgeBase
from scripts.remove_invalid_courses import validate_courses_async
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def generate_report(invalid_courses: list, output_file: str = None) -> dict:
    """
    Generate a health check report.
    
    Args:
        invalid_courses: List of invalid courses
        output_file: Optional file path to save report
        
    Returns:
        Report dictionary
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_invalid": len(invalid_courses),
        "invalid_courses": [
            {
                "course_id": c.get("course_id", ""),
                "title": c.get("title", ""),
                "provider": c.get("provider", ""),
                "url": c.get("url", ""),
                "difficulty": c.get("difficulty", ""),
                "type": c.get("type", "")
            }
            for c in invalid_courses
        ],
        "summary_by_provider": {}
    }
    
    # Group by provider
    for course in invalid_courses:
        provider = course.get("provider", "Unknown")
        report["summary_by_provider"][provider] = report["summary_by_provider"].get(provider, 0) + 1
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"📄 Report saved to: {output_file}")
    
    return report

async def run_health_check(
    provider: str = None,
    auto_remove: bool = False,
    report_only: bool = False,
    report_file: str = None
) -> dict:
    """
    Run health check on course knowledge base.
    
    Args:
        provider: Optional provider to check (None = all)
        auto_remove: If True, automatically remove invalid courses
        report_only: If True, only generate report, don't remove
        report_file: Optional path to save report
        
    Returns:
        Health check results
    """
    kb = CourseKnowledgeBase()
    
    # Get courses to check
    if provider:
        logger.info(f"📚 Checking courses from provider: {provider}")
        courses = kb.get_courses_by_provider(provider)
    else:
        logger.info("📚 Checking all courses in knowledge base...")
        courses = kb.get_all_courses()
    
    total_courses = len(courses)
    logger.info(f"📊 Total courses to check: {total_courses}")
    
    if total_courses == 0:
        logger.warning("⚠️  No courses found to check")
        return {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "deleted": 0
        }
    
    # Validate courses
    logger.info("🔍 Validating course URLs (this may take a while)...")
    invalid_courses = await validate_courses_async(courses)
    
    invalid_count = len(invalid_courses)
    valid_count = total_courses - invalid_count
    
    # Generate report
    report = generate_report(invalid_courses, report_file)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("📊 HEALTH CHECK SUMMARY")
    logger.info("="*60)
    logger.info(f"Total courses checked: {total_courses}")
    logger.info(f"✅ Valid courses: {valid_count}")
    logger.info(f"❌ Invalid courses: {invalid_count}")
    
    if invalid_count > 0:
        logger.info("\n📋 Invalid courses by provider:")
        for provider, count in report["summary_by_provider"].items():
            logger.info(f"  - {provider}: {count}")
    
    # Remove invalid courses if requested
    deleted_count = 0
    if invalid_courses and auto_remove and not report_only:
        logger.info(f"\n🗑️  Auto-removing {invalid_count} invalid courses...")
        for course in invalid_courses:
            url = course.get("url", "")
            if url:
                deleted = kb.delete_courses_by_url(url)
                if deleted > 0:
                    deleted_count += deleted
                    logger.info(f"  ✅ Deleted: {course.get('title', 'Unknown')}")
        logger.info(f"✅ Removed {deleted_count} invalid courses")
    elif invalid_courses and report_only:
        logger.info("\nℹ️  Report-only mode: Invalid courses not removed")
        logger.info("   Run with --auto-remove to remove invalid courses")
    
    return {
        "total": total_courses,
        "valid": valid_count,
        "invalid": invalid_count,
        "deleted": deleted_count,
        "report": report
    }

def main():
    parser = argparse.ArgumentParser(
        description="Health check for course knowledge base"
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="Check courses from specific provider (e.g., 'Udemy', 'Coursera')"
    )
    parser.add_argument(
        "--auto-remove",
        action="store_true",
        help="Automatically remove invalid courses"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report only, don't remove courses"
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default="course_health_report.json",
        help="Path to save health check report (default: course_health_report.json)"
    )
    
    args = parser.parse_args()
    
    try:
        results = asyncio.run(run_health_check(
            provider=args.provider,
            auto_remove=args.auto_remove,
            report_only=args.report_only,
            report_file=args.report_file
        ))
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())


Health check script for course knowledge base.
Validates all course URLs and generates a report of invalid courses.

This can be run periodically (e.g., via cron) to maintain course quality.

Usage:
    # Full health check
    python scripts/health_check_courses.py
    
    # Check specific provider
    python scripts/health_check_courses.py --provider "Udemy"
    
    # Generate report only (don't delete)
    python scripts/health_check_courses.py --report-only
    
    # Auto-remove invalid courses
    python scripts/health_check_courses.py --auto-remove
"""

import sys
import os
import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.course_knowledge_base import CourseKnowledgeBase
from scripts.remove_invalid_courses import validate_courses_async
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def generate_report(invalid_courses: list, output_file: str = None) -> dict:
    """
    Generate a health check report.
    
    Args:
        invalid_courses: List of invalid courses
        output_file: Optional file path to save report
        
    Returns:
        Report dictionary
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_invalid": len(invalid_courses),
        "invalid_courses": [
            {
                "course_id": c.get("course_id", ""),
                "title": c.get("title", ""),
                "provider": c.get("provider", ""),
                "url": c.get("url", ""),
                "difficulty": c.get("difficulty", ""),
                "type": c.get("type", "")
            }
            for c in invalid_courses
        ],
        "summary_by_provider": {}
    }
    
    # Group by provider
    for course in invalid_courses:
        provider = course.get("provider", "Unknown")
        report["summary_by_provider"][provider] = report["summary_by_provider"].get(provider, 0) + 1
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"📄 Report saved to: {output_file}")
    
    return report

async def run_health_check(
    provider: str = None,
    auto_remove: bool = False,
    report_only: bool = False,
    report_file: str = None
) -> dict:
    """
    Run health check on course knowledge base.
    
    Args:
        provider: Optional provider to check (None = all)
        auto_remove: If True, automatically remove invalid courses
        report_only: If True, only generate report, don't remove
        report_file: Optional path to save report
        
    Returns:
        Health check results
    """
    kb = CourseKnowledgeBase()
    
    # Get courses to check
    if provider:
        logger.info(f"📚 Checking courses from provider: {provider}")
        courses = kb.get_courses_by_provider(provider)
    else:
        logger.info("📚 Checking all courses in knowledge base...")
        courses = kb.get_all_courses()
    
    total_courses = len(courses)
    logger.info(f"📊 Total courses to check: {total_courses}")
    
    if total_courses == 0:
        logger.warning("⚠️  No courses found to check")
        return {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "deleted": 0
        }
    
    # Validate courses
    logger.info("🔍 Validating course URLs (this may take a while)...")
    invalid_courses = await validate_courses_async(courses)
    
    invalid_count = len(invalid_courses)
    valid_count = total_courses - invalid_count
    
    # Generate report
    report = generate_report(invalid_courses, report_file)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("📊 HEALTH CHECK SUMMARY")
    logger.info("="*60)
    logger.info(f"Total courses checked: {total_courses}")
    logger.info(f"✅ Valid courses: {valid_count}")
    logger.info(f"❌ Invalid courses: {invalid_count}")
    
    if invalid_count > 0:
        logger.info("\n📋 Invalid courses by provider:")
        for provider, count in report["summary_by_provider"].items():
            logger.info(f"  - {provider}: {count}")
    
    # Remove invalid courses if requested
    deleted_count = 0
    if invalid_courses and auto_remove and not report_only:
        logger.info(f"\n🗑️  Auto-removing {invalid_count} invalid courses...")
        for course in invalid_courses:
            url = course.get("url", "")
            if url:
                deleted = kb.delete_courses_by_url(url)
                if deleted > 0:
                    deleted_count += deleted
                    logger.info(f"  ✅ Deleted: {course.get('title', 'Unknown')}")
        logger.info(f"✅ Removed {deleted_count} invalid courses")
    elif invalid_courses and report_only:
        logger.info("\nℹ️  Report-only mode: Invalid courses not removed")
        logger.info("   Run with --auto-remove to remove invalid courses")
    
    return {
        "total": total_courses,
        "valid": valid_count,
        "invalid": invalid_count,
        "deleted": deleted_count,
        "report": report
    }

def main():
    parser = argparse.ArgumentParser(
        description="Health check for course knowledge base"
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="Check courses from specific provider (e.g., 'Udemy', 'Coursera')"
    )
    parser.add_argument(
        "--auto-remove",
        action="store_true",
        help="Automatically remove invalid courses"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report only, don't remove courses"
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default="course_health_report.json",
        help="Path to save health check report (default: course_health_report.json)"
    )
    
    args = parser.parse_args()
    
    try:
        results = asyncio.run(run_health_check(
            provider=args.provider,
            auto_remove=args.auto_remove,
            report_only=args.report_only,
            report_file=args.report_file
        ))
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())


Health check script for course knowledge base.
Validates all course URLs and generates a report of invalid courses.

This can be run periodically (e.g., via cron) to maintain course quality.

Usage:
    # Full health check
    python scripts/health_check_courses.py
    
    # Check specific provider
    python scripts/health_check_courses.py --provider "Udemy"
    
    # Generate report only (don't delete)
    python scripts/health_check_courses.py --report-only
    
    # Auto-remove invalid courses
    python scripts/health_check_courses.py --auto-remove
"""

import sys
import os
import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.course_knowledge_base import CourseKnowledgeBase
from scripts.remove_invalid_courses import validate_courses_async
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def generate_report(invalid_courses: list, output_file: str = None) -> dict:
    """
    Generate a health check report.
    
    Args:
        invalid_courses: List of invalid courses
        output_file: Optional file path to save report
        
    Returns:
        Report dictionary
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_invalid": len(invalid_courses),
        "invalid_courses": [
            {
                "course_id": c.get("course_id", ""),
                "title": c.get("title", ""),
                "provider": c.get("provider", ""),
                "url": c.get("url", ""),
                "difficulty": c.get("difficulty", ""),
                "type": c.get("type", "")
            }
            for c in invalid_courses
        ],
        "summary_by_provider": {}
    }
    
    # Group by provider
    for course in invalid_courses:
        provider = course.get("provider", "Unknown")
        report["summary_by_provider"][provider] = report["summary_by_provider"].get(provider, 0) + 1
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"📄 Report saved to: {output_file}")
    
    return report

async def run_health_check(
    provider: str = None,
    auto_remove: bool = False,
    report_only: bool = False,
    report_file: str = None
) -> dict:
    """
    Run health check on course knowledge base.
    
    Args:
        provider: Optional provider to check (None = all)
        auto_remove: If True, automatically remove invalid courses
        report_only: If True, only generate report, don't remove
        report_file: Optional path to save report
        
    Returns:
        Health check results
    """
    kb = CourseKnowledgeBase()
    
    # Get courses to check
    if provider:
        logger.info(f"📚 Checking courses from provider: {provider}")
        courses = kb.get_courses_by_provider(provider)
    else:
        logger.info("📚 Checking all courses in knowledge base...")
        courses = kb.get_all_courses()
    
    total_courses = len(courses)
    logger.info(f"📊 Total courses to check: {total_courses}")
    
    if total_courses == 0:
        logger.warning("⚠️  No courses found to check")
        return {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "deleted": 0
        }
    
    # Validate courses
    logger.info("🔍 Validating course URLs (this may take a while)...")
    invalid_courses = await validate_courses_async(courses)
    
    invalid_count = len(invalid_courses)
    valid_count = total_courses - invalid_count
    
    # Generate report
    report = generate_report(invalid_courses, report_file)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("📊 HEALTH CHECK SUMMARY")
    logger.info("="*60)
    logger.info(f"Total courses checked: {total_courses}")
    logger.info(f"✅ Valid courses: {valid_count}")
    logger.info(f"❌ Invalid courses: {invalid_count}")
    
    if invalid_count > 0:
        logger.info("\n📋 Invalid courses by provider:")
        for provider, count in report["summary_by_provider"].items():
            logger.info(f"  - {provider}: {count}")
    
    # Remove invalid courses if requested
    deleted_count = 0
    if invalid_courses and auto_remove and not report_only:
        logger.info(f"\n🗑️  Auto-removing {invalid_count} invalid courses...")
        for course in invalid_courses:
            url = course.get("url", "")
            if url:
                deleted = kb.delete_courses_by_url(url)
                if deleted > 0:
                    deleted_count += deleted
                    logger.info(f"  ✅ Deleted: {course.get('title', 'Unknown')}")
        logger.info(f"✅ Removed {deleted_count} invalid courses")
    elif invalid_courses and report_only:
        logger.info("\nℹ️  Report-only mode: Invalid courses not removed")
        logger.info("   Run with --auto-remove to remove invalid courses")
    
    return {
        "total": total_courses,
        "valid": valid_count,
        "invalid": invalid_count,
        "deleted": deleted_count,
        "report": report
    }

def main():
    parser = argparse.ArgumentParser(
        description="Health check for course knowledge base"
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="Check courses from specific provider (e.g., 'Udemy', 'Coursera')"
    )
    parser.add_argument(
        "--auto-remove",
        action="store_true",
        help="Automatically remove invalid courses"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report only, don't remove courses"
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default="course_health_report.json",
        help="Path to save health check report (default: course_health_report.json)"
    )
    
    args = parser.parse_args()
    
    try:
        results = asyncio.run(run_health_check(
            provider=args.provider,
            auto_remove=args.auto_remove,
            report_only=args.report_only,
            report_file=args.report_file
        ))
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())

