#!/usr/bin/env python3
"""
Script to remove specific course URLs from the knowledge base
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.course_knowledge_base import CourseKnowledgeBase
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# List of URLs to remove
URLS_TO_REMOVE = [
    "https://www.udemy.com/course/curriculum-development/",
    "https://www.edx.org/course/adult-learning-theories-and-principles",
    "https://emeritus.org/university-certificate-programs/instructional-design/",
]

def main():
    log.info("🔍 Starting removal of specific course URLs from knowledge base...")
    log.info(f"📋 URLs to remove: {len(URLS_TO_REMOVE)}")
    
    kb = CourseKnowledgeBase()
    
    total_found = 0
    total_deleted = 0
    
    for url in URLS_TO_REMOVE:
        log.info(f"\n{'='*60}")
        log.info(f"🔍 Processing URL: {url}")
        log.info(f"{'='*60}")
        
        # Check if course exists first
        try:
            results = kb.collection.get(where={"url": url})
            if results.get("ids") and results["ids"]:
                total_found += len(results["ids"])
                log.info(f"📋 Found {len(results['ids'])} course(s) with this URL:")
                for i, course_id in enumerate(results["ids"]):
                    metadata = results.get("metadatas", [[]])[i] if results.get("metadatas") else {}
                    log.info(f"   {i+1}. ID: {course_id}")
                    log.info(f"      Title: {metadata.get('title', 'N/A')}")
                    log.info(f"      Provider: {metadata.get('provider', 'N/A')}")
                
                # Delete the course
                if kb.delete_course_by_url(url):
                    total_deleted += len(results["ids"])
                    log.info(f"✅ Successfully removed {len(results['ids'])} course(s)")
                else:
                    log.error(f"❌ Failed to remove course(s)")
            else:
                log.warning(f"⚠️ No course found with URL: {url}")
        except Exception as e:
            log.error(f"❌ Error processing URL {url}: {e}")
            import traceback
            log.error(traceback.format_exc())
    
    log.info(f"\n{'='*60}")
    log.info(f"📊 Summary:")
    log.info(f"   Total URLs processed: {len(URLS_TO_REMOVE)}")
    log.info(f"   Total courses found: {total_found}")
    log.info(f"   Total courses deleted: {total_deleted}")
    log.info(f"{'='*60}")
    
    if total_deleted > 0:
        log.info("✅ Removal completed successfully")
    else:
        log.warning("⚠️ No courses were deleted")

if __name__ == "__main__":
    main()
