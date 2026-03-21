#!/usr/bin/env python3
"""
Script to copy all courses from QA database to dev database.
This script reads all courses from the QA environment and adds them to the dev environment.

Usage:
    # Set QA environment variables first
    export APP_ENV=qa
    python scripts/copy_courses_qa_to_dev.py --qa-db qa-jobsify-agent --dev-db dev-jobsify-agent
"""

import sys
import os
import argparse
import logging
import hashlib
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chroma import normalize_metadata

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def get_all_courses_from_collection(collection) -> List[Dict[str, Any]]:
    """
    Get all courses from a ChromaDB collection.
    
    Args:
        collection: ChromaDB collection object
        
    Returns:
        List of course dictionaries
    """
    try:
        # Get all documents from collection
        # ChromaDB's get() without parameters returns all documents
        results = collection.get()
        
        if not results or not results.get("ids"):
            logger.warning("No courses found in collection")
            return []
        
        courses = []
        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])
        documents = results.get("documents", [])
        
        logger.info(f"Found {len(ids)} courses in collection")
        
        for i, course_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            document = documents[i] if i < len(documents) else ""
            
            # Convert skills string back to list
            skills_str = metadata.get("skills", "")
            skills_list = [s.strip() for s in skills_str.split(",") if s.strip()] if skills_str else []
            
            # Reconstruct course dict
            course = {
                "course_id": course_id,
                "title": metadata.get("title", ""),
                "provider": metadata.get("provider", ""),
                "url": metadata.get("url", ""),
                "difficulty": metadata.get("difficulty", "All Levels"),
                "duration": metadata.get("duration", ""),
                "type": metadata.get("type", "course"),
                "author": metadata.get("author", ""),
                "skills": skills_list,
                "description": document
            }
            
            courses.append(course)
        
        return courses
    
    except Exception as e:
        logger.error(f"Error getting courses from collection: {e}")
        raise

def create_chromadb_client(api_key: str, tenant: str, database: str):
    """Create a ChromaDB Cloud client"""
    try:
        # Try creating client with database directly
        client = chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database
        )
        # Test connection by listing collections
        client.list_collections()
        return client
    except Exception as e:
        logger.error(f"Failed to connect to database '{database}': {e}")
        logger.error("This might be a permission issue. Make sure:")
        logger.error("  1. The API key has access to the database")
        logger.error("  2. The database name is correct")
        logger.error("  3. The tenant ID is correct")
        raise

def _generate_course_id(course: Dict[str, Any]) -> str:
    """Generate unique ID from course URL, provider, and title (same as CourseKnowledgeBase)"""
    url = course.get("url", "")
    provider = course.get("provider", "")
    title = course.get("title", "")
    unique_string = f"{provider}|{url}|{title}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def _create_searchable_text(course_data: Dict[str, Any]) -> str:
    """Create searchable text from course data (same as CourseKnowledgeBase)"""
    parts = []
    
    if course_data.get("title"):
        parts.append(f"Title: {course_data['title']}")
    
    if course_data.get("description"):
        parts.append(f"Description: {course_data['description']}")
    
    if course_data.get("skills"):
        skills = course_data["skills"]
        if isinstance(skills, list):
            skills_str = ", ".join(skills)
        else:
            skills_str = str(skills)
        parts.append(f"Skills: {skills_str}")
    
    if course_data.get("provider"):
        parts.append(f"Platform: {course_data['provider']}")
    
    if course_data.get("author"):
        parts.append(f"Author: {course_data['author']}")
    
    return " | ".join(parts)

def copy_courses_qa_to_dev(
    qa_api_key: str,
    qa_tenant: str,
    qa_database: str,
    dev_api_key: str,
    dev_tenant: str,
    dev_database: str
):
    """
    Copy all courses from QA database to dev database.
    
    Args:
        qa_api_key: ChromaDB API key for QA
        qa_tenant: ChromaDB tenant for QA
        qa_database: ChromaDB database name for QA
        dev_api_key: ChromaDB API key for dev
        dev_tenant: ChromaDB tenant for dev
        dev_database: ChromaDB database name for dev
    """
    try:
        logger.info("="*60)
        logger.info("🚀 Starting course copy from QA to dev")
        logger.info("="*60)
        
        # Step 1: Connect to QA database
        logger.info(f"\n📊 Connecting to QA database: {qa_database}")
        qa_client = create_chromadb_client(qa_api_key, qa_tenant, qa_database)
        
        # Get QA collection
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        try:
            qa_collection = qa_client.get_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        except Exception as e:
            logger.warning(f"Could not get QA collection with embedding function: {e}")
            logger.info("Trying without embedding function...")
            qa_collection = qa_client.get_collection(name="courses_knowledge_base")
        
        qa_count = qa_collection.count()
        logger.info(f"✅ Connected to QA. Found {qa_count} courses")
        
        if qa_count == 0:
            logger.warning("⚠️ No courses found in QA database. Nothing to copy.")
            return
        
        # Step 2: Get all courses from QA
        logger.info(f"\n📥 Fetching all courses from QA...")
        qa_courses = get_all_courses_from_collection(qa_collection)
        logger.info(f"✅ Retrieved {len(qa_courses)} courses from QA")
        
        # Step 3: Connect to dev database
        logger.info(f"\n📊 Connecting to dev database: {dev_database}")
        dev_client = create_chromadb_client(dev_api_key, dev_tenant, dev_database)
        
        # Get or create dev collection
        try:
            dev_collection = dev_client.get_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        except Exception as e:
            logger.info(f"Creating dev collection (error: {e})...")
            dev_collection = dev_client.get_or_create_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        
        dev_count_before = dev_collection.count()
        logger.info(f"✅ Connected to dev. Current courses: {dev_count_before}")
        
        # Step 4: Prepare courses for batch add
        courses_to_add = []
        for course in qa_courses:
            course_dict = {
                "title": course.get("title", ""),
                "provider": course.get("provider", ""),
                "url": course.get("url", ""),
                "difficulty": course.get("difficulty", "All Levels"),
                "duration": course.get("duration", ""),
                "type": course.get("type", "course"),
                "author": course.get("author", ""),
                "skills": course.get("skills", []),
                "description": course.get("description", "")
            }
            courses_to_add.append(course_dict)
        
        # Step 5: Add courses to dev in batches
        logger.info(f"\n📤 Adding {len(courses_to_add)} courses to dev database...")
        
        batch_size = 100
        total_added = 0
        
        for i in range(0, len(courses_to_add), batch_size):
            batch = courses_to_add[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(courses_to_add) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} courses)...")
            
            ids = []
            documents = []
            metadatas = []
            
            for course in batch:
                # Generate course ID (same logic as CourseKnowledgeBase)
                course_id = _generate_course_id(course)
                
                # Create searchable text
                searchable_text = _create_searchable_text(course)
                
                # Prepare metadata
                skills_value = course.get("skills", [])
                if isinstance(skills_value, list):
                    skills_str = ", ".join(str(s) for s in skills_value)[:500]
                else:
                    skills_str = str(skills_value)[:500]
                
                metadata = normalize_metadata({
                    "course_id": course_id,
                    "title": course.get("title", "")[:200],
                    "provider": course.get("provider", ""),
                    "url": course.get("url", ""),
                    "difficulty": course.get("difficulty", "All Levels"),
                    "duration": course.get("duration", ""),
                    "type": course.get("type", "course"),
                    "author": course.get("author", ""),
                    "skills": skills_str,
                })
                
                ids.append(course_id)
                documents.append(searchable_text)
                metadatas.append(metadata)
            
            # Upsert batch to dev collection
            try:
                dev_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                total_added += len(batch)
                logger.info(f"✅ Added batch {batch_num}/{total_batches} ({len(batch)} courses)")
            except Exception as e:
                logger.error(f"❌ Error adding batch {batch_num}: {e}")
                raise
        
        # Step 6: Verify
        dev_count_after = dev_collection.count()
        logger.info("\n" + "="*60)
        logger.info("📊 COPY SUMMARY")
        logger.info("="*60)
        logger.info(f"QA courses: {qa_count}")
        logger.info(f"Dev courses (before): {dev_count_before}")
        logger.info(f"Dev courses (after): {dev_count_after}")
        logger.info(f"Courses added: {dev_count_after - dev_count_before}")
        logger.info(f"Total processed: {total_added}")
        logger.info("="*60)
        
        if dev_count_after >= qa_count:
            logger.info("✅ Successfully copied all courses from QA to dev!")
        else:
            logger.warning(f"⚠️ Some courses may not have been copied. Expected at least {qa_count}, got {dev_count_after}")
        
    except Exception as e:
        logger.error(f"❌ Error copying courses: {e}", exc_info=True)
        raise

def main():
    parser = argparse.ArgumentParser(description="Copy courses from QA database to dev database")
    parser.add_argument(
        "--qa-db",
        type=str,
        default="qa-jobsify-agent",
        help="QA database name (default: qa-jobsify-agent)"
    )
    parser.add_argument(
        "--dev-db",
        type=str,
        default="dev-jobsify-agent",
        help="Dev database name (default: dev-jobsify-agent)"
    )
    parser.add_argument(
        "--qa-api-key",
        type=str,
        help="QA ChromaDB API key. If not provided, uses CHROMA_API_KEY from environment"
    )
    parser.add_argument(
        "--qa-tenant",
        type=str,
        help="QA ChromaDB tenant. If not provided, uses CHROMA_TENANT from environment"
    )
    parser.add_argument(
        "--dev-api-key",
        type=str,
        help="Dev ChromaDB API key. If not provided, uses CHROMA_API_KEY from environment"
    )
    parser.add_argument(
        "--dev-tenant",
        type=str,
        help="Dev ChromaDB tenant. If not provided, uses CHROMA_TENANT from environment"
    )
    
    args = parser.parse_args()
    
    # Get credentials from args or environment
    # For QA - you'll need to set these or pass as args
    qa_api_key = args.qa_api_key or os.getenv("CHROMA_API_KEY")
    qa_tenant = args.qa_tenant or os.getenv("CHROMA_TENANT")
    qa_database = args.qa_db
    
    # For dev - uses current environment settings
    dev_api_key = args.dev_api_key or os.getenv("CHROMA_API_KEY")
    dev_tenant = args.dev_tenant or os.getenv("CHROMA_TENANT")
    dev_database = args.dev_db
    
    # Validate
    if not qa_api_key or not qa_tenant:
        logger.error("❌ QA ChromaDB credentials not provided.")
        logger.error("   Set CHROMA_API_KEY and CHROMA_TENANT environment variables")
        logger.error("   Or use --qa-api-key and --qa-tenant arguments")
        logger.error("")
        logger.error("   NOTE: If QA and dev use the same credentials, you can:")
        logger.error("   export CHROMA_API_KEY='your-key'")
        logger.error("   export CHROMA_TENANT='your-tenant'")
        logger.error("   Then run: python scripts/copy_courses_qa_to_dev.py --qa-db qa-jobsify-agent --dev-db dev-jobsify-agent")
        return 1
    
    if not dev_api_key or not dev_tenant:
        logger.error("❌ Dev ChromaDB credentials not provided.")
        logger.error("   Set CHROMA_API_KEY and CHROMA_TENANT environment variables")
        logger.error("   Or use --dev-api-key and --dev-tenant arguments")
        return 1
    
    # Check if QA and dev use same credentials (common case)
    if qa_api_key == dev_api_key and qa_tenant == dev_tenant:
        logger.info("ℹ️  Using same credentials for QA and dev (same tenant)")
    else:
        logger.info("ℹ️  Using different credentials for QA and dev")
    
    logger.info(f"📋 Configuration:")
    logger.info(f"   QA Database: {qa_database}")
    logger.info(f"   Dev Database: {dev_database}")
    logger.info(f"   QA Tenant: {qa_tenant}")
    logger.info(f"   Dev Tenant: {dev_tenant}")
    
    try:
        copy_courses_qa_to_dev(
            qa_api_key=qa_api_key,
            qa_tenant=qa_tenant,
            qa_database=qa_database,
            dev_api_key=dev_api_key,
            dev_tenant=dev_tenant,
            dev_database=dev_database
        )
        return 0
    except Exception as e:
        logger.error(f"❌ Failed to copy courses: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())


Script to copy all courses from QA database to dev database.
This script reads all courses from the QA environment and adds them to the dev environment.

Usage:
    # Set QA environment variables first
    export APP_ENV=qa
    python scripts/copy_courses_qa_to_dev.py --qa-db qa-jobsify-agent --dev-db dev-jobsify-agent
"""

import sys
import os
import argparse
import logging
import hashlib
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chroma import normalize_metadata

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def get_all_courses_from_collection(collection) -> List[Dict[str, Any]]:
    """
    Get all courses from a ChromaDB collection.
    
    Args:
        collection: ChromaDB collection object
        
    Returns:
        List of course dictionaries
    """
    try:
        # Get all documents from collection
        # ChromaDB's get() without parameters returns all documents
        results = collection.get()
        
        if not results or not results.get("ids"):
            logger.warning("No courses found in collection")
            return []
        
        courses = []
        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])
        documents = results.get("documents", [])
        
        logger.info(f"Found {len(ids)} courses in collection")
        
        for i, course_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            document = documents[i] if i < len(documents) else ""
            
            # Convert skills string back to list
            skills_str = metadata.get("skills", "")
            skills_list = [s.strip() for s in skills_str.split(",") if s.strip()] if skills_str else []
            
            # Reconstruct course dict
            course = {
                "course_id": course_id,
                "title": metadata.get("title", ""),
                "provider": metadata.get("provider", ""),
                "url": metadata.get("url", ""),
                "difficulty": metadata.get("difficulty", "All Levels"),
                "duration": metadata.get("duration", ""),
                "type": metadata.get("type", "course"),
                "author": metadata.get("author", ""),
                "skills": skills_list,
                "description": document
            }
            
            courses.append(course)
        
        return courses
    
    except Exception as e:
        logger.error(f"Error getting courses from collection: {e}")
        raise

def create_chromadb_client(api_key: str, tenant: str, database: str):
    """Create a ChromaDB Cloud client"""
    try:
        # Try creating client with database directly
        client = chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database
        )
        # Test connection by listing collections
        client.list_collections()
        return client
    except Exception as e:
        logger.error(f"Failed to connect to database '{database}': {e}")
        logger.error("This might be a permission issue. Make sure:")
        logger.error("  1. The API key has access to the database")
        logger.error("  2. The database name is correct")
        logger.error("  3. The tenant ID is correct")
        raise

def _generate_course_id(course: Dict[str, Any]) -> str:
    """Generate unique ID from course URL, provider, and title (same as CourseKnowledgeBase)"""
    url = course.get("url", "")
    provider = course.get("provider", "")
    title = course.get("title", "")
    unique_string = f"{provider}|{url}|{title}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def _create_searchable_text(course_data: Dict[str, Any]) -> str:
    """Create searchable text from course data (same as CourseKnowledgeBase)"""
    parts = []
    
    if course_data.get("title"):
        parts.append(f"Title: {course_data['title']}")
    
    if course_data.get("description"):
        parts.append(f"Description: {course_data['description']}")
    
    if course_data.get("skills"):
        skills = course_data["skills"]
        if isinstance(skills, list):
            skills_str = ", ".join(skills)
        else:
            skills_str = str(skills)
        parts.append(f"Skills: {skills_str}")
    
    if course_data.get("provider"):
        parts.append(f"Platform: {course_data['provider']}")
    
    if course_data.get("author"):
        parts.append(f"Author: {course_data['author']}")
    
    return " | ".join(parts)

def copy_courses_qa_to_dev(
    qa_api_key: str,
    qa_tenant: str,
    qa_database: str,
    dev_api_key: str,
    dev_tenant: str,
    dev_database: str
):
    """
    Copy all courses from QA database to dev database.
    
    Args:
        qa_api_key: ChromaDB API key for QA
        qa_tenant: ChromaDB tenant for QA
        qa_database: ChromaDB database name for QA
        dev_api_key: ChromaDB API key for dev
        dev_tenant: ChromaDB tenant for dev
        dev_database: ChromaDB database name for dev
    """
    try:
        logger.info("="*60)
        logger.info("🚀 Starting course copy from QA to dev")
        logger.info("="*60)
        
        # Step 1: Connect to QA database
        logger.info(f"\n📊 Connecting to QA database: {qa_database}")
        qa_client = create_chromadb_client(qa_api_key, qa_tenant, qa_database)
        
        # Get QA collection
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        try:
            qa_collection = qa_client.get_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        except Exception as e:
            logger.warning(f"Could not get QA collection with embedding function: {e}")
            logger.info("Trying without embedding function...")
            qa_collection = qa_client.get_collection(name="courses_knowledge_base")
        
        qa_count = qa_collection.count()
        logger.info(f"✅ Connected to QA. Found {qa_count} courses")
        
        if qa_count == 0:
            logger.warning("⚠️ No courses found in QA database. Nothing to copy.")
            return
        
        # Step 2: Get all courses from QA
        logger.info(f"\n📥 Fetching all courses from QA...")
        qa_courses = get_all_courses_from_collection(qa_collection)
        logger.info(f"✅ Retrieved {len(qa_courses)} courses from QA")
        
        # Step 3: Connect to dev database
        logger.info(f"\n📊 Connecting to dev database: {dev_database}")
        dev_client = create_chromadb_client(dev_api_key, dev_tenant, dev_database)
        
        # Get or create dev collection
        try:
            dev_collection = dev_client.get_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        except Exception as e:
            logger.info(f"Creating dev collection (error: {e})...")
            dev_collection = dev_client.get_or_create_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        
        dev_count_before = dev_collection.count()
        logger.info(f"✅ Connected to dev. Current courses: {dev_count_before}")
        
        # Step 4: Prepare courses for batch add
        courses_to_add = []
        for course in qa_courses:
            course_dict = {
                "title": course.get("title", ""),
                "provider": course.get("provider", ""),
                "url": course.get("url", ""),
                "difficulty": course.get("difficulty", "All Levels"),
                "duration": course.get("duration", ""),
                "type": course.get("type", "course"),
                "author": course.get("author", ""),
                "skills": course.get("skills", []),
                "description": course.get("description", "")
            }
            courses_to_add.append(course_dict)
        
        # Step 5: Add courses to dev in batches
        logger.info(f"\n📤 Adding {len(courses_to_add)} courses to dev database...")
        
        batch_size = 100
        total_added = 0
        
        for i in range(0, len(courses_to_add), batch_size):
            batch = courses_to_add[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(courses_to_add) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} courses)...")
            
            ids = []
            documents = []
            metadatas = []
            
            for course in batch:
                # Generate course ID (same logic as CourseKnowledgeBase)
                course_id = _generate_course_id(course)
                
                # Create searchable text
                searchable_text = _create_searchable_text(course)
                
                # Prepare metadata
                skills_value = course.get("skills", [])
                if isinstance(skills_value, list):
                    skills_str = ", ".join(str(s) for s in skills_value)[:500]
                else:
                    skills_str = str(skills_value)[:500]
                
                metadata = normalize_metadata({
                    "course_id": course_id,
                    "title": course.get("title", "")[:200],
                    "provider": course.get("provider", ""),
                    "url": course.get("url", ""),
                    "difficulty": course.get("difficulty", "All Levels"),
                    "duration": course.get("duration", ""),
                    "type": course.get("type", "course"),
                    "author": course.get("author", ""),
                    "skills": skills_str,
                })
                
                ids.append(course_id)
                documents.append(searchable_text)
                metadatas.append(metadata)
            
            # Upsert batch to dev collection
            try:
                dev_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                total_added += len(batch)
                logger.info(f"✅ Added batch {batch_num}/{total_batches} ({len(batch)} courses)")
            except Exception as e:
                logger.error(f"❌ Error adding batch {batch_num}: {e}")
                raise
        
        # Step 6: Verify
        dev_count_after = dev_collection.count()
        logger.info("\n" + "="*60)
        logger.info("📊 COPY SUMMARY")
        logger.info("="*60)
        logger.info(f"QA courses: {qa_count}")
        logger.info(f"Dev courses (before): {dev_count_before}")
        logger.info(f"Dev courses (after): {dev_count_after}")
        logger.info(f"Courses added: {dev_count_after - dev_count_before}")
        logger.info(f"Total processed: {total_added}")
        logger.info("="*60)
        
        if dev_count_after >= qa_count:
            logger.info("✅ Successfully copied all courses from QA to dev!")
        else:
            logger.warning(f"⚠️ Some courses may not have been copied. Expected at least {qa_count}, got {dev_count_after}")
        
    except Exception as e:
        logger.error(f"❌ Error copying courses: {e}", exc_info=True)
        raise

def main():
    parser = argparse.ArgumentParser(description="Copy courses from QA database to dev database")
    parser.add_argument(
        "--qa-db",
        type=str,
        default="qa-jobsify-agent",
        help="QA database name (default: qa-jobsify-agent)"
    )
    parser.add_argument(
        "--dev-db",
        type=str,
        default="dev-jobsify-agent",
        help="Dev database name (default: dev-jobsify-agent)"
    )
    parser.add_argument(
        "--qa-api-key",
        type=str,
        help="QA ChromaDB API key. If not provided, uses CHROMA_API_KEY from environment"
    )
    parser.add_argument(
        "--qa-tenant",
        type=str,
        help="QA ChromaDB tenant. If not provided, uses CHROMA_TENANT from environment"
    )
    parser.add_argument(
        "--dev-api-key",
        type=str,
        help="Dev ChromaDB API key. If not provided, uses CHROMA_API_KEY from environment"
    )
    parser.add_argument(
        "--dev-tenant",
        type=str,
        help="Dev ChromaDB tenant. If not provided, uses CHROMA_TENANT from environment"
    )
    
    args = parser.parse_args()
    
    # Get credentials from args or environment
    # For QA - you'll need to set these or pass as args
    qa_api_key = args.qa_api_key or os.getenv("CHROMA_API_KEY")
    qa_tenant = args.qa_tenant or os.getenv("CHROMA_TENANT")
    qa_database = args.qa_db
    
    # For dev - uses current environment settings
    dev_api_key = args.dev_api_key or os.getenv("CHROMA_API_KEY")
    dev_tenant = args.dev_tenant or os.getenv("CHROMA_TENANT")
    dev_database = args.dev_db
    
    # Validate
    if not qa_api_key or not qa_tenant:
        logger.error("❌ QA ChromaDB credentials not provided.")
        logger.error("   Set CHROMA_API_KEY and CHROMA_TENANT environment variables")
        logger.error("   Or use --qa-api-key and --qa-tenant arguments")
        logger.error("")
        logger.error("   NOTE: If QA and dev use the same credentials, you can:")
        logger.error("   export CHROMA_API_KEY='your-key'")
        logger.error("   export CHROMA_TENANT='your-tenant'")
        logger.error("   Then run: python scripts/copy_courses_qa_to_dev.py --qa-db qa-jobsify-agent --dev-db dev-jobsify-agent")
        return 1
    
    if not dev_api_key or not dev_tenant:
        logger.error("❌ Dev ChromaDB credentials not provided.")
        logger.error("   Set CHROMA_API_KEY and CHROMA_TENANT environment variables")
        logger.error("   Or use --dev-api-key and --dev-tenant arguments")
        return 1
    
    # Check if QA and dev use same credentials (common case)
    if qa_api_key == dev_api_key and qa_tenant == dev_tenant:
        logger.info("ℹ️  Using same credentials for QA and dev (same tenant)")
    else:
        logger.info("ℹ️  Using different credentials for QA and dev")
    
    logger.info(f"📋 Configuration:")
    logger.info(f"   QA Database: {qa_database}")
    logger.info(f"   Dev Database: {dev_database}")
    logger.info(f"   QA Tenant: {qa_tenant}")
    logger.info(f"   Dev Tenant: {dev_tenant}")
    
    try:
        copy_courses_qa_to_dev(
            qa_api_key=qa_api_key,
            qa_tenant=qa_tenant,
            qa_database=qa_database,
            dev_api_key=dev_api_key,
            dev_tenant=dev_tenant,
            dev_database=dev_database
        )
        return 0
    except Exception as e:
        logger.error(f"❌ Failed to copy courses: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())


Script to copy all courses from QA database to dev database.
This script reads all courses from the QA environment and adds them to the dev environment.

Usage:
    # Set QA environment variables first
    export APP_ENV=qa
    python scripts/copy_courses_qa_to_dev.py --qa-db qa-jobsify-agent --dev-db dev-jobsify-agent
"""

import sys
import os
import argparse
import logging
import hashlib
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chroma import normalize_metadata

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def get_all_courses_from_collection(collection) -> List[Dict[str, Any]]:
    """
    Get all courses from a ChromaDB collection.
    
    Args:
        collection: ChromaDB collection object
        
    Returns:
        List of course dictionaries
    """
    try:
        # Get all documents from collection
        # ChromaDB's get() without parameters returns all documents
        results = collection.get()
        
        if not results or not results.get("ids"):
            logger.warning("No courses found in collection")
            return []
        
        courses = []
        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])
        documents = results.get("documents", [])
        
        logger.info(f"Found {len(ids)} courses in collection")
        
        for i, course_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            document = documents[i] if i < len(documents) else ""
            
            # Convert skills string back to list
            skills_str = metadata.get("skills", "")
            skills_list = [s.strip() for s in skills_str.split(",") if s.strip()] if skills_str else []
            
            # Reconstruct course dict
            course = {
                "course_id": course_id,
                "title": metadata.get("title", ""),
                "provider": metadata.get("provider", ""),
                "url": metadata.get("url", ""),
                "difficulty": metadata.get("difficulty", "All Levels"),
                "duration": metadata.get("duration", ""),
                "type": metadata.get("type", "course"),
                "author": metadata.get("author", ""),
                "skills": skills_list,
                "description": document
            }
            
            courses.append(course)
        
        return courses
    
    except Exception as e:
        logger.error(f"Error getting courses from collection: {e}")
        raise

def create_chromadb_client(api_key: str, tenant: str, database: str):
    """Create a ChromaDB Cloud client"""
    try:
        # Try creating client with database directly
        client = chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database
        )
        # Test connection by listing collections
        client.list_collections()
        return client
    except Exception as e:
        logger.error(f"Failed to connect to database '{database}': {e}")
        logger.error("This might be a permission issue. Make sure:")
        logger.error("  1. The API key has access to the database")
        logger.error("  2. The database name is correct")
        logger.error("  3. The tenant ID is correct")
        raise

def _generate_course_id(course: Dict[str, Any]) -> str:
    """Generate unique ID from course URL, provider, and title (same as CourseKnowledgeBase)"""
    url = course.get("url", "")
    provider = course.get("provider", "")
    title = course.get("title", "")
    unique_string = f"{provider}|{url}|{title}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def _create_searchable_text(course_data: Dict[str, Any]) -> str:
    """Create searchable text from course data (same as CourseKnowledgeBase)"""
    parts = []
    
    if course_data.get("title"):
        parts.append(f"Title: {course_data['title']}")
    
    if course_data.get("description"):
        parts.append(f"Description: {course_data['description']}")
    
    if course_data.get("skills"):
        skills = course_data["skills"]
        if isinstance(skills, list):
            skills_str = ", ".join(skills)
        else:
            skills_str = str(skills)
        parts.append(f"Skills: {skills_str}")
    
    if course_data.get("provider"):
        parts.append(f"Platform: {course_data['provider']}")
    
    if course_data.get("author"):
        parts.append(f"Author: {course_data['author']}")
    
    return " | ".join(parts)

def copy_courses_qa_to_dev(
    qa_api_key: str,
    qa_tenant: str,
    qa_database: str,
    dev_api_key: str,
    dev_tenant: str,
    dev_database: str
):
    """
    Copy all courses from QA database to dev database.
    
    Args:
        qa_api_key: ChromaDB API key for QA
        qa_tenant: ChromaDB tenant for QA
        qa_database: ChromaDB database name for QA
        dev_api_key: ChromaDB API key for dev
        dev_tenant: ChromaDB tenant for dev
        dev_database: ChromaDB database name for dev
    """
    try:
        logger.info("="*60)
        logger.info("🚀 Starting course copy from QA to dev")
        logger.info("="*60)
        
        # Step 1: Connect to QA database
        logger.info(f"\n📊 Connecting to QA database: {qa_database}")
        qa_client = create_chromadb_client(qa_api_key, qa_tenant, qa_database)
        
        # Get QA collection
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        try:
            qa_collection = qa_client.get_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        except Exception as e:
            logger.warning(f"Could not get QA collection with embedding function: {e}")
            logger.info("Trying without embedding function...")
            qa_collection = qa_client.get_collection(name="courses_knowledge_base")
        
        qa_count = qa_collection.count()
        logger.info(f"✅ Connected to QA. Found {qa_count} courses")
        
        if qa_count == 0:
            logger.warning("⚠️ No courses found in QA database. Nothing to copy.")
            return
        
        # Step 2: Get all courses from QA
        logger.info(f"\n📥 Fetching all courses from QA...")
        qa_courses = get_all_courses_from_collection(qa_collection)
        logger.info(f"✅ Retrieved {len(qa_courses)} courses from QA")
        
        # Step 3: Connect to dev database
        logger.info(f"\n📊 Connecting to dev database: {dev_database}")
        dev_client = create_chromadb_client(dev_api_key, dev_tenant, dev_database)
        
        # Get or create dev collection
        try:
            dev_collection = dev_client.get_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        except Exception as e:
            logger.info(f"Creating dev collection (error: {e})...")
            dev_collection = dev_client.get_or_create_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        
        dev_count_before = dev_collection.count()
        logger.info(f"✅ Connected to dev. Current courses: {dev_count_before}")
        
        # Step 4: Prepare courses for batch add
        courses_to_add = []
        for course in qa_courses:
            course_dict = {
                "title": course.get("title", ""),
                "provider": course.get("provider", ""),
                "url": course.get("url", ""),
                "difficulty": course.get("difficulty", "All Levels"),
                "duration": course.get("duration", ""),
                "type": course.get("type", "course"),
                "author": course.get("author", ""),
                "skills": course.get("skills", []),
                "description": course.get("description", "")
            }
            courses_to_add.append(course_dict)
        
        # Step 5: Add courses to dev in batches
        logger.info(f"\n📤 Adding {len(courses_to_add)} courses to dev database...")
        
        batch_size = 100
        total_added = 0
        
        for i in range(0, len(courses_to_add), batch_size):
            batch = courses_to_add[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(courses_to_add) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} courses)...")
            
            ids = []
            documents = []
            metadatas = []
            
            for course in batch:
                # Generate course ID (same logic as CourseKnowledgeBase)
                course_id = _generate_course_id(course)
                
                # Create searchable text
                searchable_text = _create_searchable_text(course)
                
                # Prepare metadata
                skills_value = course.get("skills", [])
                if isinstance(skills_value, list):
                    skills_str = ", ".join(str(s) for s in skills_value)[:500]
                else:
                    skills_str = str(skills_value)[:500]
                
                metadata = normalize_metadata({
                    "course_id": course_id,
                    "title": course.get("title", "")[:200],
                    "provider": course.get("provider", ""),
                    "url": course.get("url", ""),
                    "difficulty": course.get("difficulty", "All Levels"),
                    "duration": course.get("duration", ""),
                    "type": course.get("type", "course"),
                    "author": course.get("author", ""),
                    "skills": skills_str,
                })
                
                ids.append(course_id)
                documents.append(searchable_text)
                metadatas.append(metadata)
            
            # Upsert batch to dev collection
            try:
                dev_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                total_added += len(batch)
                logger.info(f"✅ Added batch {batch_num}/{total_batches} ({len(batch)} courses)")
            except Exception as e:
                logger.error(f"❌ Error adding batch {batch_num}: {e}")
                raise
        
        # Step 6: Verify
        dev_count_after = dev_collection.count()
        logger.info("\n" + "="*60)
        logger.info("📊 COPY SUMMARY")
        logger.info("="*60)
        logger.info(f"QA courses: {qa_count}")
        logger.info(f"Dev courses (before): {dev_count_before}")
        logger.info(f"Dev courses (after): {dev_count_after}")
        logger.info(f"Courses added: {dev_count_after - dev_count_before}")
        logger.info(f"Total processed: {total_added}")
        logger.info("="*60)
        
        if dev_count_after >= qa_count:
            logger.info("✅ Successfully copied all courses from QA to dev!")
        else:
            logger.warning(f"⚠️ Some courses may not have been copied. Expected at least {qa_count}, got {dev_count_after}")
        
    except Exception as e:
        logger.error(f"❌ Error copying courses: {e}", exc_info=True)
        raise

def main():
    parser = argparse.ArgumentParser(description="Copy courses from QA database to dev database")
    parser.add_argument(
        "--qa-db",
        type=str,
        default="qa-jobsify-agent",
        help="QA database name (default: qa-jobsify-agent)"
    )
    parser.add_argument(
        "--dev-db",
        type=str,
        default="dev-jobsify-agent",
        help="Dev database name (default: dev-jobsify-agent)"
    )
    parser.add_argument(
        "--qa-api-key",
        type=str,
        help="QA ChromaDB API key. If not provided, uses CHROMA_API_KEY from environment"
    )
    parser.add_argument(
        "--qa-tenant",
        type=str,
        help="QA ChromaDB tenant. If not provided, uses CHROMA_TENANT from environment"
    )
    parser.add_argument(
        "--dev-api-key",
        type=str,
        help="Dev ChromaDB API key. If not provided, uses CHROMA_API_KEY from environment"
    )
    parser.add_argument(
        "--dev-tenant",
        type=str,
        help="Dev ChromaDB tenant. If not provided, uses CHROMA_TENANT from environment"
    )
    
    args = parser.parse_args()
    
    # Get credentials from args or environment
    # For QA - you'll need to set these or pass as args
    qa_api_key = args.qa_api_key or os.getenv("CHROMA_API_KEY")
    qa_tenant = args.qa_tenant or os.getenv("CHROMA_TENANT")
    qa_database = args.qa_db
    
    # For dev - uses current environment settings
    dev_api_key = args.dev_api_key or os.getenv("CHROMA_API_KEY")
    dev_tenant = args.dev_tenant or os.getenv("CHROMA_TENANT")
    dev_database = args.dev_db
    
    # Validate
    if not qa_api_key or not qa_tenant:
        logger.error("❌ QA ChromaDB credentials not provided.")
        logger.error("   Set CHROMA_API_KEY and CHROMA_TENANT environment variables")
        logger.error("   Or use --qa-api-key and --qa-tenant arguments")
        logger.error("")
        logger.error("   NOTE: If QA and dev use the same credentials, you can:")
        logger.error("   export CHROMA_API_KEY='your-key'")
        logger.error("   export CHROMA_TENANT='your-tenant'")
        logger.error("   Then run: python scripts/copy_courses_qa_to_dev.py --qa-db qa-jobsify-agent --dev-db dev-jobsify-agent")
        return 1
    
    if not dev_api_key or not dev_tenant:
        logger.error("❌ Dev ChromaDB credentials not provided.")
        logger.error("   Set CHROMA_API_KEY and CHROMA_TENANT environment variables")
        logger.error("   Or use --dev-api-key and --dev-tenant arguments")
        return 1
    
    # Check if QA and dev use same credentials (common case)
    if qa_api_key == dev_api_key and qa_tenant == dev_tenant:
        logger.info("ℹ️  Using same credentials for QA and dev (same tenant)")
    else:
        logger.info("ℹ️  Using different credentials for QA and dev")
    
    logger.info(f"📋 Configuration:")
    logger.info(f"   QA Database: {qa_database}")
    logger.info(f"   Dev Database: {dev_database}")
    logger.info(f"   QA Tenant: {qa_tenant}")
    logger.info(f"   Dev Tenant: {dev_tenant}")
    
    try:
        copy_courses_qa_to_dev(
            qa_api_key=qa_api_key,
            qa_tenant=qa_tenant,
            qa_database=qa_database,
            dev_api_key=dev_api_key,
            dev_tenant=dev_tenant,
            dev_database=dev_database
        )
        return 0
    except Exception as e:
        logger.error(f"❌ Failed to copy courses: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())


Script to copy all courses from QA database to dev database.
This script reads all courses from the QA environment and adds them to the dev environment.

Usage:
    # Set QA environment variables first
    export APP_ENV=qa
    python scripts/copy_courses_qa_to_dev.py --qa-db qa-jobsify-agent --dev-db dev-jobsify-agent
"""

import sys
import os
import argparse
import logging
import hashlib
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chroma import normalize_metadata

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def get_all_courses_from_collection(collection) -> List[Dict[str, Any]]:
    """
    Get all courses from a ChromaDB collection.
    
    Args:
        collection: ChromaDB collection object
        
    Returns:
        List of course dictionaries
    """
    try:
        # Get all documents from collection
        # ChromaDB's get() without parameters returns all documents
        results = collection.get()
        
        if not results or not results.get("ids"):
            logger.warning("No courses found in collection")
            return []
        
        courses = []
        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])
        documents = results.get("documents", [])
        
        logger.info(f"Found {len(ids)} courses in collection")
        
        for i, course_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            document = documents[i] if i < len(documents) else ""
            
            # Convert skills string back to list
            skills_str = metadata.get("skills", "")
            skills_list = [s.strip() for s in skills_str.split(",") if s.strip()] if skills_str else []
            
            # Reconstruct course dict
            course = {
                "course_id": course_id,
                "title": metadata.get("title", ""),
                "provider": metadata.get("provider", ""),
                "url": metadata.get("url", ""),
                "difficulty": metadata.get("difficulty", "All Levels"),
                "duration": metadata.get("duration", ""),
                "type": metadata.get("type", "course"),
                "author": metadata.get("author", ""),
                "skills": skills_list,
                "description": document
            }
            
            courses.append(course)
        
        return courses
    
    except Exception as e:
        logger.error(f"Error getting courses from collection: {e}")
        raise

def create_chromadb_client(api_key: str, tenant: str, database: str):
    """Create a ChromaDB Cloud client"""
    try:
        # Try creating client with database directly
        client = chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database
        )
        # Test connection by listing collections
        client.list_collections()
        return client
    except Exception as e:
        logger.error(f"Failed to connect to database '{database}': {e}")
        logger.error("This might be a permission issue. Make sure:")
        logger.error("  1. The API key has access to the database")
        logger.error("  2. The database name is correct")
        logger.error("  3. The tenant ID is correct")
        raise

def _generate_course_id(course: Dict[str, Any]) -> str:
    """Generate unique ID from course URL, provider, and title (same as CourseKnowledgeBase)"""
    url = course.get("url", "")
    provider = course.get("provider", "")
    title = course.get("title", "")
    unique_string = f"{provider}|{url}|{title}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def _create_searchable_text(course_data: Dict[str, Any]) -> str:
    """Create searchable text from course data (same as CourseKnowledgeBase)"""
    parts = []
    
    if course_data.get("title"):
        parts.append(f"Title: {course_data['title']}")
    
    if course_data.get("description"):
        parts.append(f"Description: {course_data['description']}")
    
    if course_data.get("skills"):
        skills = course_data["skills"]
        if isinstance(skills, list):
            skills_str = ", ".join(skills)
        else:
            skills_str = str(skills)
        parts.append(f"Skills: {skills_str}")
    
    if course_data.get("provider"):
        parts.append(f"Platform: {course_data['provider']}")
    
    if course_data.get("author"):
        parts.append(f"Author: {course_data['author']}")
    
    return " | ".join(parts)

def copy_courses_qa_to_dev(
    qa_api_key: str,
    qa_tenant: str,
    qa_database: str,
    dev_api_key: str,
    dev_tenant: str,
    dev_database: str
):
    """
    Copy all courses from QA database to dev database.
    
    Args:
        qa_api_key: ChromaDB API key for QA
        qa_tenant: ChromaDB tenant for QA
        qa_database: ChromaDB database name for QA
        dev_api_key: ChromaDB API key for dev
        dev_tenant: ChromaDB tenant for dev
        dev_database: ChromaDB database name for dev
    """
    try:
        logger.info("="*60)
        logger.info("🚀 Starting course copy from QA to dev")
        logger.info("="*60)
        
        # Step 1: Connect to QA database
        logger.info(f"\n📊 Connecting to QA database: {qa_database}")
        qa_client = create_chromadb_client(qa_api_key, qa_tenant, qa_database)
        
        # Get QA collection
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        try:
            qa_collection = qa_client.get_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        except Exception as e:
            logger.warning(f"Could not get QA collection with embedding function: {e}")
            logger.info("Trying without embedding function...")
            qa_collection = qa_client.get_collection(name="courses_knowledge_base")
        
        qa_count = qa_collection.count()
        logger.info(f"✅ Connected to QA. Found {qa_count} courses")
        
        if qa_count == 0:
            logger.warning("⚠️ No courses found in QA database. Nothing to copy.")
            return
        
        # Step 2: Get all courses from QA
        logger.info(f"\n📥 Fetching all courses from QA...")
        qa_courses = get_all_courses_from_collection(qa_collection)
        logger.info(f"✅ Retrieved {len(qa_courses)} courses from QA")
        
        # Step 3: Connect to dev database
        logger.info(f"\n📊 Connecting to dev database: {dev_database}")
        dev_client = create_chromadb_client(dev_api_key, dev_tenant, dev_database)
        
        # Get or create dev collection
        try:
            dev_collection = dev_client.get_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        except Exception as e:
            logger.info(f"Creating dev collection (error: {e})...")
            dev_collection = dev_client.get_or_create_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        
        dev_count_before = dev_collection.count()
        logger.info(f"✅ Connected to dev. Current courses: {dev_count_before}")
        
        # Step 4: Prepare courses for batch add
        courses_to_add = []
        for course in qa_courses:
            course_dict = {
                "title": course.get("title", ""),
                "provider": course.get("provider", ""),
                "url": course.get("url", ""),
                "difficulty": course.get("difficulty", "All Levels"),
                "duration": course.get("duration", ""),
                "type": course.get("type", "course"),
                "author": course.get("author", ""),
                "skills": course.get("skills", []),
                "description": course.get("description", "")
            }
            courses_to_add.append(course_dict)
        
        # Step 5: Add courses to dev in batches
        logger.info(f"\n📤 Adding {len(courses_to_add)} courses to dev database...")
        
        batch_size = 100
        total_added = 0
        
        for i in range(0, len(courses_to_add), batch_size):
            batch = courses_to_add[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(courses_to_add) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} courses)...")
            
            ids = []
            documents = []
            metadatas = []
            
            for course in batch:
                # Generate course ID (same logic as CourseKnowledgeBase)
                course_id = _generate_course_id(course)
                
                # Create searchable text
                searchable_text = _create_searchable_text(course)
                
                # Prepare metadata
                skills_value = course.get("skills", [])
                if isinstance(skills_value, list):
                    skills_str = ", ".join(str(s) for s in skills_value)[:500]
                else:
                    skills_str = str(skills_value)[:500]
                
                metadata = normalize_metadata({
                    "course_id": course_id,
                    "title": course.get("title", "")[:200],
                    "provider": course.get("provider", ""),
                    "url": course.get("url", ""),
                    "difficulty": course.get("difficulty", "All Levels"),
                    "duration": course.get("duration", ""),
                    "type": course.get("type", "course"),
                    "author": course.get("author", ""),
                    "skills": skills_str,
                })
                
                ids.append(course_id)
                documents.append(searchable_text)
                metadatas.append(metadata)
            
            # Upsert batch to dev collection
            try:
                dev_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                total_added += len(batch)
                logger.info(f"✅ Added batch {batch_num}/{total_batches} ({len(batch)} courses)")
            except Exception as e:
                logger.error(f"❌ Error adding batch {batch_num}: {e}")
                raise
        
        # Step 6: Verify
        dev_count_after = dev_collection.count()
        logger.info("\n" + "="*60)
        logger.info("📊 COPY SUMMARY")
        logger.info("="*60)
        logger.info(f"QA courses: {qa_count}")
        logger.info(f"Dev courses (before): {dev_count_before}")
        logger.info(f"Dev courses (after): {dev_count_after}")
        logger.info(f"Courses added: {dev_count_after - dev_count_before}")
        logger.info(f"Total processed: {total_added}")
        logger.info("="*60)
        
        if dev_count_after >= qa_count:
            logger.info("✅ Successfully copied all courses from QA to dev!")
        else:
            logger.warning(f"⚠️ Some courses may not have been copied. Expected at least {qa_count}, got {dev_count_after}")
        
    except Exception as e:
        logger.error(f"❌ Error copying courses: {e}", exc_info=True)
        raise

def main():
    parser = argparse.ArgumentParser(description="Copy courses from QA database to dev database")
    parser.add_argument(
        "--qa-db",
        type=str,
        default="qa-jobsify-agent",
        help="QA database name (default: qa-jobsify-agent)"
    )
    parser.add_argument(
        "--dev-db",
        type=str,
        default="dev-jobsify-agent",
        help="Dev database name (default: dev-jobsify-agent)"
    )
    parser.add_argument(
        "--qa-api-key",
        type=str,
        help="QA ChromaDB API key. If not provided, uses CHROMA_API_KEY from environment"
    )
    parser.add_argument(
        "--qa-tenant",
        type=str,
        help="QA ChromaDB tenant. If not provided, uses CHROMA_TENANT from environment"
    )
    parser.add_argument(
        "--dev-api-key",
        type=str,
        help="Dev ChromaDB API key. If not provided, uses CHROMA_API_KEY from environment"
    )
    parser.add_argument(
        "--dev-tenant",
        type=str,
        help="Dev ChromaDB tenant. If not provided, uses CHROMA_TENANT from environment"
    )
    
    args = parser.parse_args()
    
    # Get credentials from args or environment
    # For QA - you'll need to set these or pass as args
    qa_api_key = args.qa_api_key or os.getenv("CHROMA_API_KEY")
    qa_tenant = args.qa_tenant or os.getenv("CHROMA_TENANT")
    qa_database = args.qa_db
    
    # For dev - uses current environment settings
    dev_api_key = args.dev_api_key or os.getenv("CHROMA_API_KEY")
    dev_tenant = args.dev_tenant or os.getenv("CHROMA_TENANT")
    dev_database = args.dev_db
    
    # Validate
    if not qa_api_key or not qa_tenant:
        logger.error("❌ QA ChromaDB credentials not provided.")
        logger.error("   Set CHROMA_API_KEY and CHROMA_TENANT environment variables")
        logger.error("   Or use --qa-api-key and --qa-tenant arguments")
        logger.error("")
        logger.error("   NOTE: If QA and dev use the same credentials, you can:")
        logger.error("   export CHROMA_API_KEY='your-key'")
        logger.error("   export CHROMA_TENANT='your-tenant'")
        logger.error("   Then run: python scripts/copy_courses_qa_to_dev.py --qa-db qa-jobsify-agent --dev-db dev-jobsify-agent")
        return 1
    
    if not dev_api_key or not dev_tenant:
        logger.error("❌ Dev ChromaDB credentials not provided.")
        logger.error("   Set CHROMA_API_KEY and CHROMA_TENANT environment variables")
        logger.error("   Or use --dev-api-key and --dev-tenant arguments")
        return 1
    
    # Check if QA and dev use same credentials (common case)
    if qa_api_key == dev_api_key and qa_tenant == dev_tenant:
        logger.info("ℹ️  Using same credentials for QA and dev (same tenant)")
    else:
        logger.info("ℹ️  Using different credentials for QA and dev")
    
    logger.info(f"📋 Configuration:")
    logger.info(f"   QA Database: {qa_database}")
    logger.info(f"   Dev Database: {dev_database}")
    logger.info(f"   QA Tenant: {qa_tenant}")
    logger.info(f"   Dev Tenant: {dev_tenant}")
    
    try:
        copy_courses_qa_to_dev(
            qa_api_key=qa_api_key,
            qa_tenant=qa_tenant,
            qa_database=qa_database,
            dev_api_key=dev_api_key,
            dev_tenant=dev_tenant,
            dev_database=dev_database
        )
        return 0
    except Exception as e:
        logger.error(f"❌ Failed to copy courses: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())

