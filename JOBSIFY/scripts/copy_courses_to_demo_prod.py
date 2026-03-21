#!/usr/bin/env python3
"""
Script to copy all courses from a source database (dev/qa) to demo and production databases.
This script reads all courses from the source environment and adds them to demo and production.

Usage:
    # Copy from dev to demo and production
    python scripts/copy_courses_to_demo_prod.py --source-db dev-jobsify-agent --target-envs demo,prod
    
    # Copy from QA to demo and production
    python scripts/copy_courses_to_demo_prod.py --source-db qa-jobsify-agent --target-envs demo,prod
    
    # Copy to specific databases
    python scripts/copy_courses_to_demo_prod.py --source-db dev-jobsify-agent --demo-db demo-jobsify-agent --prod-db prod-jobsify-agent
"""

import sys
import os
import argparse
import logging
import hashlib
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.errors import ChromaAuthError
from chroma import normalize_metadata

# Try to load python-dotenv for .env file support
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

def load_env_file(env_file: str) -> Dict[str, str]:
    """
    Load environment variables from a .env file.
    Returns a dict of key-value pairs.
    """
    env_vars = {}
    if not os.path.exists(env_file):
        return env_vars
    
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value
    except Exception as e:
        logger.warning(f"Could not load {env_file}: {e}")
    
    return env_vars

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
    """Create a ChromaDB Cloud client (same logic as copy_courses_qa_to_dev.py)"""
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

def copy_courses_to_target(
    source_courses: List[Dict[str, Any]],
    target_api_key: str,
    target_tenant: str,
    target_database: str,
    target_name: str
) -> Dict[str, Any]:
    """
    Copy courses to a target database.
    
    Args:
        source_courses: List of course dictionaries to copy
        target_api_key: ChromaDB API key for target
        target_tenant: ChromaDB tenant for target
        target_database: ChromaDB database name for target
        target_name: Human-readable name for target (for logging)
        
    Returns:
        Dict with copy statistics
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Copying to {target_name.upper()} database: {target_database}")
        logger.info(f"{'='*60}")
        
        # Connect to target database
        target_client = create_chromadb_client(target_api_key, target_tenant, target_database)
        
        # Get or create target collection
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        try:
            target_collection = target_client.get_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        except Exception as e:
            logger.info(f"Creating {target_name} collection (error: {e})...")
            target_collection = target_client.get_or_create_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        
        target_count_before = target_collection.count()
        logger.info(f"✅ Connected to {target_name}. Current courses: {target_count_before}")
        
        # Prepare courses for batch add
        courses_to_add = []
        for course in source_courses:
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
        
        # Add courses in batches
        logger.info(f"📤 Adding {len(courses_to_add)} courses to {target_name} database...")
        
        batch_size = 100
        total_added = 0
        
        for i in range(0, len(courses_to_add), batch_size):
            batch = courses_to_add[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(courses_to_add) + batch_size - 1) // batch_size
            
            logger.info(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} courses)...")
            
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
            
            # Upsert batch to target collection
            try:
                target_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                total_added += len(batch)
                logger.info(f"  ✅ Added batch {batch_num}/{total_batches} ({len(batch)} courses)")
            except Exception as e:
                logger.error(f"  ❌ Error adding batch {batch_num}: {e}")
                raise
        
        # Verify
        target_count_after = target_collection.count()
        
        result = {
            "target_name": target_name,
            "target_database": target_database,
            "courses_before": target_count_before,
            "courses_after": target_count_after,
            "courses_added": target_count_after - target_count_before,
            "total_processed": total_added,
            "success": target_count_after >= target_count_before
        }
        
        logger.info(f"✅ {target_name.upper()} copy complete:")
        logger.info(f"   Before: {target_count_before} courses")
        logger.info(f"   After: {target_count_after} courses")
        logger.info(f"   Added: {target_count_after - target_count_before} courses")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error copying courses to {target_name}: {e}", exc_info=True)
        return {
            "target_name": target_name,
            "target_database": target_database,
            "success": False,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(
        description="Copy courses from source database to demo and/or production databases"
    )
    parser.add_argument(
        "--source-db",
        type=str,
        default="dev-jobsify-agent",
        help="Source database name (default: dev-jobsify-agent)"
    )
    parser.add_argument(
        "--demo-db",
        type=str,
        default="demo-jobsify-agent",
        help="Demo database name (default: demo-jobsify-agent)"
    )
    parser.add_argument(
        "--prod-db",
        type=str,
        default="prod-jobsify-agent",
        help="Production database name (default: prod-jobsify-agent)"
    )
    parser.add_argument(
        "--target-envs",
        type=str,
        help="Comma-separated list of target environments (demo,prod). If not provided, copies to both."
    )
    parser.add_argument(
        "--source-api-key",
        type=str,
        help="Source ChromaDB API key. If not provided, uses CHROMA_API_KEY from environment"
    )
    parser.add_argument(
        "--source-tenant",
        type=str,
        help="Source ChromaDB tenant. If not provided, uses CHROMA_TENANT from environment"
    )
    parser.add_argument(
        "--target-api-key",
        type=str,
        help="Target ChromaDB API key (for demo/prod). If not provided, uses CHROMA_API_KEY from environment"
    )
    parser.add_argument(
        "--target-tenant",
        type=str,
        help="Target ChromaDB tenant (for demo/prod). If not provided, uses CHROMA_TENANT from environment"
    )
    
    args = parser.parse_args()
    
    # Determine target environments first
    if args.target_envs:
        target_envs = [env.strip().lower() for env in args.target_envs.split(",")]
    else:
        target_envs = ["demo", "prod"]
    
    # Get root directory (KA)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    # Load environment files for target environments
    demo_env_vars = {}
    prod_env_vars = {}
    
    if "demo" in target_envs:
        demo_env_file = os.path.join(root_dir, ".env.demo")
        if os.path.exists(demo_env_file):
            logger.info(f"📄 Loading demo credentials from {demo_env_file}")
            demo_env_vars = load_env_file(demo_env_file)
            if DOTENV_AVAILABLE:
                load_dotenv(demo_env_file, override=False)
            # Log what we found
            if demo_env_vars.get("CHROMA_API_KEY"):
                logger.info(f"   ✅ Found CHROMA_API_KEY in .env.demo (length: {len(demo_env_vars['CHROMA_API_KEY'])})")
            else:
                logger.warning(f"   ⚠️  CHROMA_API_KEY not found in .env.demo")
            if demo_env_vars.get("CHROMA_TENANT"):
                logger.info(f"   ✅ Found CHROMA_TENANT in .env.demo")
            else:
                logger.warning(f"   ⚠️  CHROMA_TENANT not found in .env.demo")
        else:
            logger.warning(f"   ⚠️  .env.demo file not found at {demo_env_file}")
    
    if "prod" in target_envs:
        prod_env_file = os.path.join(root_dir, ".env.prod")
        if os.path.exists(prod_env_file):
            logger.info(f"📄 Loading prod credentials from {prod_env_file}")
            prod_env_vars = load_env_file(prod_env_file)
            if DOTENV_AVAILABLE:
                load_dotenv(prod_env_file, override=False)
            # Log what we found
            if prod_env_vars.get("CHROMA_API_KEY"):
                logger.info(f"   ✅ Found CHROMA_API_KEY in .env.prod (length: {len(prod_env_vars['CHROMA_API_KEY'])})")
            else:
                logger.warning(f"   ⚠️  CHROMA_API_KEY not found in .env.prod")
            if prod_env_vars.get("CHROMA_TENANT"):
                logger.info(f"   ✅ Found CHROMA_TENANT in .env.prod")
            else:
                logger.warning(f"   ⚠️  CHROMA_TENANT not found in .env.prod")
        else:
            logger.warning(f"   ⚠️  .env.prod file not found at {prod_env_file}")
    
    # Get source credentials
    source_api_key = args.source_api_key or os.getenv("CHROMA_API_KEY")
    source_tenant = args.source_tenant or os.getenv("CHROMA_TENANT")
    source_database = args.source_db
    
    # Get demo credentials (priority: args > .env.demo > source > env)
    demo_api_key = None
    demo_tenant = None
    if "demo" in target_envs:
        # Explicitly use .env.demo credentials first (don't fallback to source)
        demo_api_key = args.target_api_key or demo_env_vars.get("CHROMA_API_KEY")
        demo_tenant = args.target_tenant or demo_env_vars.get("CHROMA_TENANT")
        # Only fallback if not found in .env.demo
        if not demo_api_key:
            logger.warning("   ⚠️  No demo API key found in .env.demo, using source credentials")
            demo_api_key = source_api_key or os.getenv("CHROMA_API_KEY")
        if not demo_tenant:
            logger.warning("   ⚠️  No demo tenant found in .env.demo, using source tenant")
            demo_tenant = source_tenant or os.getenv("CHROMA_TENANT")
        logger.info(f"   🔑 Using demo API key: {'From .env.demo' if demo_env_vars.get('CHROMA_API_KEY') else 'From source/default'} (first 10 chars: {demo_api_key[:10] if demo_api_key else 'None'}...)")
    
    # Get prod credentials (priority: args > .env.prod > source > env)
    prod_api_key = None
    prod_tenant = None
    if "prod" in target_envs:
        # Explicitly use .env.prod credentials first (don't fallback to source)
        prod_api_key = args.target_api_key or prod_env_vars.get("CHROMA_API_KEY")
        prod_tenant = args.target_tenant or prod_env_vars.get("CHROMA_TENANT")
        # Only fallback if not found in .env.prod
        if not prod_api_key:
            logger.warning("   ⚠️  No prod API key found in .env.prod, using source credentials")
            prod_api_key = source_api_key or os.getenv("CHROMA_API_KEY")
        if not prod_tenant:
            logger.warning("   ⚠️  No prod tenant found in .env.prod, using source tenant")
            prod_tenant = source_tenant or os.getenv("CHROMA_TENANT")
        logger.info(f"   🔑 Using prod API key: {'From .env.prod' if prod_env_vars.get('CHROMA_API_KEY') else 'From source/default'} (first 10 chars: {prod_api_key[:10] if prod_api_key else 'None'}...)")
    
    # Fallback target credentials (for backward compatibility)
    target_api_key = args.target_api_key or source_api_key or os.getenv("CHROMA_API_KEY")
    target_tenant = args.target_tenant or source_tenant or os.getenv("CHROMA_TENANT")
    
    # Validate source credentials
    if not source_api_key or not source_tenant:
        logger.error("❌ Source ChromaDB credentials not provided.")
        logger.error("   Set CHROMA_API_KEY and CHROMA_TENANT environment variables")
        logger.error("   Or use --source-api-key and --source-tenant arguments")
        return 1
    
    # Validate target credentials
    if not target_api_key or not target_tenant:
        logger.error("❌ Target ChromaDB credentials not provided.")
        logger.error("   Set CHROMA_API_KEY and CHROMA_TENANT environment variables")
        logger.error("   Or use --target-api-key and --target-tenant arguments")
        return 1
    
    logger.info("="*60)
    logger.info("🚀 COURSE COPY: Source → Demo/Production")
    logger.info("="*60)
    logger.info(f"📋 Configuration:")
    logger.info(f"   Source Database: {source_database}")
    logger.info(f"   Source Tenant: {source_tenant}")
    logger.info(f"   Target Environments: {', '.join(target_envs)}")
    if "demo" in target_envs:
        logger.info(f"   Demo API Key: {'✅ Loaded from .env.demo' if demo_env_vars.get('CHROMA_API_KEY') else '⚠️  Using source/default'}")
        logger.info(f"   Demo Tenant: {demo_tenant}")
    if "prod" in target_envs:
        logger.info(f"   Prod API Key: {'✅ Loaded from .env.prod' if prod_env_vars.get('CHROMA_API_KEY') else '⚠️  Using source/default'}")
        logger.info(f"   Prod Tenant: {prod_tenant}")
    logger.info("="*60)
    logger.info("")
    logger.info("⚠️  NOTE: If you get 'Database does not match' errors, you need:")
    logger.info("   - API keys that have access to demo/prod databases")
    logger.info("   - Or use --target-api-key and --target-tenant for demo/prod credentials")
    logger.info("")
    
    try:
        # Step 1: Connect to source database and get courses
        logger.info(f"\n📥 Step 1: Fetching courses from source database: {source_database}")
        source_client = create_chromadb_client(source_api_key, source_tenant, source_database)
        
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        try:
            source_collection = source_client.get_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        except Exception as e:
            logger.warning(f"Could not get source collection with embedding function: {e}")
            logger.info("Trying without embedding function...")
            source_collection = source_client.get_collection(name="courses_knowledge_base")
        
        source_count = source_collection.count()
        logger.info(f"✅ Connected to source. Found {source_count} courses")
        
        if source_count == 0:
            logger.warning("⚠️ No courses found in source database. Nothing to copy.")
            return 1
        
        # Get all courses from source
        source_courses = get_all_courses_from_collection(source_collection)
        logger.info(f"✅ Retrieved {len(source_courses)} courses from source")
        
        # Step 2: Copy to target environments
        results = []
        
        if "demo" in target_envs:
            # Use demo-specific credentials from .env.demo if available
            demo_result = copy_courses_to_target(
                source_courses=source_courses,
                target_api_key=demo_api_key,
                target_tenant=demo_tenant,
                target_database=args.demo_db,
                target_name="demo"
            )
            results.append(demo_result)
        
        if "prod" in target_envs:
            # Use prod-specific credentials from .env.prod if available
            prod_result = copy_courses_to_target(
                source_courses=source_courses,
                target_api_key=prod_api_key,
                target_tenant=prod_tenant,
                target_database=args.prod_db,
                target_name="production"
            )
            results.append(prod_result)
        
        # Step 3: Summary
        logger.info("\n" + "="*60)
        logger.info("📊 COPY SUMMARY")
        logger.info("="*60)
        logger.info(f"Source database: {source_database}")
        logger.info(f"Source courses: {source_count}")
        logger.info("")
        
        for result in results:
            if result.get("success"):
                logger.info(f"✅ {result['target_name'].upper()}: {result['courses_after']} courses (added {result['courses_added']})")
            else:
                logger.error(f"❌ {result['target_name'].upper()}: Failed - {result.get('error', 'Unknown error')}")
        
        logger.info("="*60)
        
        # Check if all copies succeeded
        all_success = all(r.get("success", False) for r in results)
        if all_success:
            logger.info("✅ Successfully copied courses to all target environments!")
            return 0
        else:
            logger.error("⚠️ Some copies failed. Check logs above for details.")
            return 1
        
    except Exception as e:
        logger.error(f"❌ Failed to copy courses: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())


Script to copy all courses from a source database (dev/qa) to demo and production databases.
This script reads all courses from the source environment and adds them to demo and production.

Usage:
    # Copy from dev to demo and production
    python scripts/copy_courses_to_demo_prod.py --source-db dev-jobsify-agent --target-envs demo,prod
    
    # Copy from QA to demo and production
    python scripts/copy_courses_to_demo_prod.py --source-db qa-jobsify-agent --target-envs demo,prod
    
    # Copy to specific databases
    python scripts/copy_courses_to_demo_prod.py --source-db dev-jobsify-agent --demo-db demo-jobsify-agent --prod-db prod-jobsify-agent
"""

import sys
import os
import argparse
import logging
import hashlib
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.errors import ChromaAuthError
from chroma import normalize_metadata

# Try to load python-dotenv for .env file support
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

def load_env_file(env_file: str) -> Dict[str, str]:
    """
    Load environment variables from a .env file.
    Returns a dict of key-value pairs.
    """
    env_vars = {}
    if not os.path.exists(env_file):
        return env_vars
    
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value
    except Exception as e:
        logger.warning(f"Could not load {env_file}: {e}")
    
    return env_vars

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
    """Create a ChromaDB Cloud client (same logic as copy_courses_qa_to_dev.py)"""
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

def copy_courses_to_target(
    source_courses: List[Dict[str, Any]],
    target_api_key: str,
    target_tenant: str,
    target_database: str,
    target_name: str
) -> Dict[str, Any]:
    """
    Copy courses to a target database.
    
    Args:
        source_courses: List of course dictionaries to copy
        target_api_key: ChromaDB API key for target
        target_tenant: ChromaDB tenant for target
        target_database: ChromaDB database name for target
        target_name: Human-readable name for target (for logging)
        
    Returns:
        Dict with copy statistics
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Copying to {target_name.upper()} database: {target_database}")
        logger.info(f"{'='*60}")
        
        # Connect to target database
        target_client = create_chromadb_client(target_api_key, target_tenant, target_database)
        
        # Get or create target collection
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        try:
            target_collection = target_client.get_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        except Exception as e:
            logger.info(f"Creating {target_name} collection (error: {e})...")
            target_collection = target_client.get_or_create_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        
        target_count_before = target_collection.count()
        logger.info(f"✅ Connected to {target_name}. Current courses: {target_count_before}")
        
        # Prepare courses for batch add
        courses_to_add = []
        for course in source_courses:
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
        
        # Add courses in batches
        logger.info(f"📤 Adding {len(courses_to_add)} courses to {target_name} database...")
        
        batch_size = 100
        total_added = 0
        
        for i in range(0, len(courses_to_add), batch_size):
            batch = courses_to_add[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(courses_to_add) + batch_size - 1) // batch_size
            
            logger.info(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} courses)...")
            
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
            
            # Upsert batch to target collection
            try:
                target_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                total_added += len(batch)
                logger.info(f"  ✅ Added batch {batch_num}/{total_batches} ({len(batch)} courses)")
            except Exception as e:
                logger.error(f"  ❌ Error adding batch {batch_num}: {e}")
                raise
        
        # Verify
        target_count_after = target_collection.count()
        
        result = {
            "target_name": target_name,
            "target_database": target_database,
            "courses_before": target_count_before,
            "courses_after": target_count_after,
            "courses_added": target_count_after - target_count_before,
            "total_processed": total_added,
            "success": target_count_after >= target_count_before
        }
        
        logger.info(f"✅ {target_name.upper()} copy complete:")
        logger.info(f"   Before: {target_count_before} courses")
        logger.info(f"   After: {target_count_after} courses")
        logger.info(f"   Added: {target_count_after - target_count_before} courses")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error copying courses to {target_name}: {e}", exc_info=True)
        return {
            "target_name": target_name,
            "target_database": target_database,
            "success": False,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(
        description="Copy courses from source database to demo and/or production databases"
    )
    parser.add_argument(
        "--source-db",
        type=str,
        default="dev-jobsify-agent",
        help="Source database name (default: dev-jobsify-agent)"
    )
    parser.add_argument(
        "--demo-db",
        type=str,
        default="demo-jobsify-agent",
        help="Demo database name (default: demo-jobsify-agent)"
    )
    parser.add_argument(
        "--prod-db",
        type=str,
        default="prod-jobsify-agent",
        help="Production database name (default: prod-jobsify-agent)"
    )
    parser.add_argument(
        "--target-envs",
        type=str,
        help="Comma-separated list of target environments (demo,prod). If not provided, copies to both."
    )
    parser.add_argument(
        "--source-api-key",
        type=str,
        help="Source ChromaDB API key. If not provided, uses CHROMA_API_KEY from environment"
    )
    parser.add_argument(
        "--source-tenant",
        type=str,
        help="Source ChromaDB tenant. If not provided, uses CHROMA_TENANT from environment"
    )
    parser.add_argument(
        "--target-api-key",
        type=str,
        help="Target ChromaDB API key (for demo/prod). If not provided, uses CHROMA_API_KEY from environment"
    )
    parser.add_argument(
        "--target-tenant",
        type=str,
        help="Target ChromaDB tenant (for demo/prod). If not provided, uses CHROMA_TENANT from environment"
    )
    
    args = parser.parse_args()
    
    # Determine target environments first
    if args.target_envs:
        target_envs = [env.strip().lower() for env in args.target_envs.split(",")]
    else:
        target_envs = ["demo", "prod"]
    
    # Get root directory (KA)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    # Load environment files for target environments
    demo_env_vars = {}
    prod_env_vars = {}
    
    if "demo" in target_envs:
        demo_env_file = os.path.join(root_dir, ".env.demo")
        if os.path.exists(demo_env_file):
            logger.info(f"📄 Loading demo credentials from {demo_env_file}")
            demo_env_vars = load_env_file(demo_env_file)
            if DOTENV_AVAILABLE:
                load_dotenv(demo_env_file, override=False)
            # Log what we found
            if demo_env_vars.get("CHROMA_API_KEY"):
                logger.info(f"   ✅ Found CHROMA_API_KEY in .env.demo (length: {len(demo_env_vars['CHROMA_API_KEY'])})")
            else:
                logger.warning(f"   ⚠️  CHROMA_API_KEY not found in .env.demo")
            if demo_env_vars.get("CHROMA_TENANT"):
                logger.info(f"   ✅ Found CHROMA_TENANT in .env.demo")
            else:
                logger.warning(f"   ⚠️  CHROMA_TENANT not found in .env.demo")
        else:
            logger.warning(f"   ⚠️  .env.demo file not found at {demo_env_file}")
    
    if "prod" in target_envs:
        prod_env_file = os.path.join(root_dir, ".env.prod")
        if os.path.exists(prod_env_file):
            logger.info(f"📄 Loading prod credentials from {prod_env_file}")
            prod_env_vars = load_env_file(prod_env_file)
            if DOTENV_AVAILABLE:
                load_dotenv(prod_env_file, override=False)
            # Log what we found
            if prod_env_vars.get("CHROMA_API_KEY"):
                logger.info(f"   ✅ Found CHROMA_API_KEY in .env.prod (length: {len(prod_env_vars['CHROMA_API_KEY'])})")
            else:
                logger.warning(f"   ⚠️  CHROMA_API_KEY not found in .env.prod")
            if prod_env_vars.get("CHROMA_TENANT"):
                logger.info(f"   ✅ Found CHROMA_TENANT in .env.prod")
            else:
                logger.warning(f"   ⚠️  CHROMA_TENANT not found in .env.prod")
        else:
            logger.warning(f"   ⚠️  .env.prod file not found at {prod_env_file}")
    
    # Get source credentials
    source_api_key = args.source_api_key or os.getenv("CHROMA_API_KEY")
    source_tenant = args.source_tenant or os.getenv("CHROMA_TENANT")
    source_database = args.source_db
    
    # Get demo credentials (priority: args > .env.demo > source > env)
    demo_api_key = None
    demo_tenant = None
    if "demo" in target_envs:
        # Explicitly use .env.demo credentials first (don't fallback to source)
        demo_api_key = args.target_api_key or demo_env_vars.get("CHROMA_API_KEY")
        demo_tenant = args.target_tenant or demo_env_vars.get("CHROMA_TENANT")
        # Only fallback if not found in .env.demo
        if not demo_api_key:
            logger.warning("   ⚠️  No demo API key found in .env.demo, using source credentials")
            demo_api_key = source_api_key or os.getenv("CHROMA_API_KEY")
        if not demo_tenant:
            logger.warning("   ⚠️  No demo tenant found in .env.demo, using source tenant")
            demo_tenant = source_tenant or os.getenv("CHROMA_TENANT")
        logger.info(f"   🔑 Using demo API key: {'From .env.demo' if demo_env_vars.get('CHROMA_API_KEY') else 'From source/default'} (first 10 chars: {demo_api_key[:10] if demo_api_key else 'None'}...)")
    
    # Get prod credentials (priority: args > .env.prod > source > env)
    prod_api_key = None
    prod_tenant = None
    if "prod" in target_envs:
        # Explicitly use .env.prod credentials first (don't fallback to source)
        prod_api_key = args.target_api_key or prod_env_vars.get("CHROMA_API_KEY")
        prod_tenant = args.target_tenant or prod_env_vars.get("CHROMA_TENANT")
        # Only fallback if not found in .env.prod
        if not prod_api_key:
            logger.warning("   ⚠️  No prod API key found in .env.prod, using source credentials")
            prod_api_key = source_api_key or os.getenv("CHROMA_API_KEY")
        if not prod_tenant:
            logger.warning("   ⚠️  No prod tenant found in .env.prod, using source tenant")
            prod_tenant = source_tenant or os.getenv("CHROMA_TENANT")
        logger.info(f"   🔑 Using prod API key: {'From .env.prod' if prod_env_vars.get('CHROMA_API_KEY') else 'From source/default'} (first 10 chars: {prod_api_key[:10] if prod_api_key else 'None'}...)")
    
    # Fallback target credentials (for backward compatibility)
    target_api_key = args.target_api_key or source_api_key or os.getenv("CHROMA_API_KEY")
    target_tenant = args.target_tenant or source_tenant or os.getenv("CHROMA_TENANT")
    
    # Validate source credentials
    if not source_api_key or not source_tenant:
        logger.error("❌ Source ChromaDB credentials not provided.")
        logger.error("   Set CHROMA_API_KEY and CHROMA_TENANT environment variables")
        logger.error("   Or use --source-api-key and --source-tenant arguments")
        return 1
    
    # Validate target credentials
    if not target_api_key or not target_tenant:
        logger.error("❌ Target ChromaDB credentials not provided.")
        logger.error("   Set CHROMA_API_KEY and CHROMA_TENANT environment variables")
        logger.error("   Or use --target-api-key and --target-tenant arguments")
        return 1
    
    logger.info("="*60)
    logger.info("🚀 COURSE COPY: Source → Demo/Production")
    logger.info("="*60)
    logger.info(f"📋 Configuration:")
    logger.info(f"   Source Database: {source_database}")
    logger.info(f"   Source Tenant: {source_tenant}")
    logger.info(f"   Target Environments: {', '.join(target_envs)}")
    if "demo" in target_envs:
        logger.info(f"   Demo API Key: {'✅ Loaded from .env.demo' if demo_env_vars.get('CHROMA_API_KEY') else '⚠️  Using source/default'}")
        logger.info(f"   Demo Tenant: {demo_tenant}")
    if "prod" in target_envs:
        logger.info(f"   Prod API Key: {'✅ Loaded from .env.prod' if prod_env_vars.get('CHROMA_API_KEY') else '⚠️  Using source/default'}")
        logger.info(f"   Prod Tenant: {prod_tenant}")
    logger.info("="*60)
    logger.info("")
    logger.info("⚠️  NOTE: If you get 'Database does not match' errors, you need:")
    logger.info("   - API keys that have access to demo/prod databases")
    logger.info("   - Or use --target-api-key and --target-tenant for demo/prod credentials")
    logger.info("")
    
    try:
        # Step 1: Connect to source database and get courses
        logger.info(f"\n📥 Step 1: Fetching courses from source database: {source_database}")
        source_client = create_chromadb_client(source_api_key, source_tenant, source_database)
        
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        try:
            source_collection = source_client.get_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        except Exception as e:
            logger.warning(f"Could not get source collection with embedding function: {e}")
            logger.info("Trying without embedding function...")
            source_collection = source_client.get_collection(name="courses_knowledge_base")
        
        source_count = source_collection.count()
        logger.info(f"✅ Connected to source. Found {source_count} courses")
        
        if source_count == 0:
            logger.warning("⚠️ No courses found in source database. Nothing to copy.")
            return 1
        
        # Get all courses from source
        source_courses = get_all_courses_from_collection(source_collection)
        logger.info(f"✅ Retrieved {len(source_courses)} courses from source")
        
        # Step 2: Copy to target environments
        results = []
        
        if "demo" in target_envs:
            # Use demo-specific credentials from .env.demo if available
            demo_result = copy_courses_to_target(
                source_courses=source_courses,
                target_api_key=demo_api_key,
                target_tenant=demo_tenant,
                target_database=args.demo_db,
                target_name="demo"
            )
            results.append(demo_result)
        
        if "prod" in target_envs:
            # Use prod-specific credentials from .env.prod if available
            prod_result = copy_courses_to_target(
                source_courses=source_courses,
                target_api_key=prod_api_key,
                target_tenant=prod_tenant,
                target_database=args.prod_db,
                target_name="production"
            )
            results.append(prod_result)
        
        # Step 3: Summary
        logger.info("\n" + "="*60)
        logger.info("📊 COPY SUMMARY")
        logger.info("="*60)
        logger.info(f"Source database: {source_database}")
        logger.info(f"Source courses: {source_count}")
        logger.info("")
        
        for result in results:
            if result.get("success"):
                logger.info(f"✅ {result['target_name'].upper()}: {result['courses_after']} courses (added {result['courses_added']})")
            else:
                logger.error(f"❌ {result['target_name'].upper()}: Failed - {result.get('error', 'Unknown error')}")
        
        logger.info("="*60)
        
        # Check if all copies succeeded
        all_success = all(r.get("success", False) for r in results)
        if all_success:
            logger.info("✅ Successfully copied courses to all target environments!")
            return 0
        else:
            logger.error("⚠️ Some copies failed. Check logs above for details.")
            return 1
        
    except Exception as e:
        logger.error(f"❌ Failed to copy courses: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())


Script to copy all courses from a source database (dev/qa) to demo and production databases.
This script reads all courses from the source environment and adds them to demo and production.

Usage:
    # Copy from dev to demo and production
    python scripts/copy_courses_to_demo_prod.py --source-db dev-jobsify-agent --target-envs demo,prod
    
    # Copy from QA to demo and production
    python scripts/copy_courses_to_demo_prod.py --source-db qa-jobsify-agent --target-envs demo,prod
    
    # Copy to specific databases
    python scripts/copy_courses_to_demo_prod.py --source-db dev-jobsify-agent --demo-db demo-jobsify-agent --prod-db prod-jobsify-agent
"""

import sys
import os
import argparse
import logging
import hashlib
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.errors import ChromaAuthError
from chroma import normalize_metadata

# Try to load python-dotenv for .env file support
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

def load_env_file(env_file: str) -> Dict[str, str]:
    """
    Load environment variables from a .env file.
    Returns a dict of key-value pairs.
    """
    env_vars = {}
    if not os.path.exists(env_file):
        return env_vars
    
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value
    except Exception as e:
        logger.warning(f"Could not load {env_file}: {e}")
    
    return env_vars

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
    """Create a ChromaDB Cloud client (same logic as copy_courses_qa_to_dev.py)"""
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

def copy_courses_to_target(
    source_courses: List[Dict[str, Any]],
    target_api_key: str,
    target_tenant: str,
    target_database: str,
    target_name: str
) -> Dict[str, Any]:
    """
    Copy courses to a target database.
    
    Args:
        source_courses: List of course dictionaries to copy
        target_api_key: ChromaDB API key for target
        target_tenant: ChromaDB tenant for target
        target_database: ChromaDB database name for target
        target_name: Human-readable name for target (for logging)
        
    Returns:
        Dict with copy statistics
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Copying to {target_name.upper()} database: {target_database}")
        logger.info(f"{'='*60}")
        
        # Connect to target database
        target_client = create_chromadb_client(target_api_key, target_tenant, target_database)
        
        # Get or create target collection
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        try:
            target_collection = target_client.get_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        except Exception as e:
            logger.info(f"Creating {target_name} collection (error: {e})...")
            target_collection = target_client.get_or_create_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        
        target_count_before = target_collection.count()
        logger.info(f"✅ Connected to {target_name}. Current courses: {target_count_before}")
        
        # Prepare courses for batch add
        courses_to_add = []
        for course in source_courses:
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
        
        # Add courses in batches
        logger.info(f"📤 Adding {len(courses_to_add)} courses to {target_name} database...")
        
        batch_size = 100
        total_added = 0
        
        for i in range(0, len(courses_to_add), batch_size):
            batch = courses_to_add[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(courses_to_add) + batch_size - 1) // batch_size
            
            logger.info(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} courses)...")
            
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
            
            # Upsert batch to target collection
            try:
                target_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                total_added += len(batch)
                logger.info(f"  ✅ Added batch {batch_num}/{total_batches} ({len(batch)} courses)")
            except Exception as e:
                logger.error(f"  ❌ Error adding batch {batch_num}: {e}")
                raise
        
        # Verify
        target_count_after = target_collection.count()
        
        result = {
            "target_name": target_name,
            "target_database": target_database,
            "courses_before": target_count_before,
            "courses_after": target_count_after,
            "courses_added": target_count_after - target_count_before,
            "total_processed": total_added,
            "success": target_count_after >= target_count_before
        }
        
        logger.info(f"✅ {target_name.upper()} copy complete:")
        logger.info(f"   Before: {target_count_before} courses")
        logger.info(f"   After: {target_count_after} courses")
        logger.info(f"   Added: {target_count_after - target_count_before} courses")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error copying courses to {target_name}: {e}", exc_info=True)
        return {
            "target_name": target_name,
            "target_database": target_database,
            "success": False,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(
        description="Copy courses from source database to demo and/or production databases"
    )
    parser.add_argument(
        "--source-db",
        type=str,
        default="dev-jobsify-agent",
        help="Source database name (default: dev-jobsify-agent)"
    )
    parser.add_argument(
        "--demo-db",
        type=str,
        default="demo-jobsify-agent",
        help="Demo database name (default: demo-jobsify-agent)"
    )
    parser.add_argument(
        "--prod-db",
        type=str,
        default="prod-jobsify-agent",
        help="Production database name (default: prod-jobsify-agent)"
    )
    parser.add_argument(
        "--target-envs",
        type=str,
        help="Comma-separated list of target environments (demo,prod). If not provided, copies to both."
    )
    parser.add_argument(
        "--source-api-key",
        type=str,
        help="Source ChromaDB API key. If not provided, uses CHROMA_API_KEY from environment"
    )
    parser.add_argument(
        "--source-tenant",
        type=str,
        help="Source ChromaDB tenant. If not provided, uses CHROMA_TENANT from environment"
    )
    parser.add_argument(
        "--target-api-key",
        type=str,
        help="Target ChromaDB API key (for demo/prod). If not provided, uses CHROMA_API_KEY from environment"
    )
    parser.add_argument(
        "--target-tenant",
        type=str,
        help="Target ChromaDB tenant (for demo/prod). If not provided, uses CHROMA_TENANT from environment"
    )
    
    args = parser.parse_args()
    
    # Determine target environments first
    if args.target_envs:
        target_envs = [env.strip().lower() for env in args.target_envs.split(",")]
    else:
        target_envs = ["demo", "prod"]
    
    # Get root directory (KA)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    # Load environment files for target environments
    demo_env_vars = {}
    prod_env_vars = {}
    
    if "demo" in target_envs:
        demo_env_file = os.path.join(root_dir, ".env.demo")
        if os.path.exists(demo_env_file):
            logger.info(f"📄 Loading demo credentials from {demo_env_file}")
            demo_env_vars = load_env_file(demo_env_file)
            if DOTENV_AVAILABLE:
                load_dotenv(demo_env_file, override=False)
            # Log what we found
            if demo_env_vars.get("CHROMA_API_KEY"):
                logger.info(f"   ✅ Found CHROMA_API_KEY in .env.demo (length: {len(demo_env_vars['CHROMA_API_KEY'])})")
            else:
                logger.warning(f"   ⚠️  CHROMA_API_KEY not found in .env.demo")
            if demo_env_vars.get("CHROMA_TENANT"):
                logger.info(f"   ✅ Found CHROMA_TENANT in .env.demo")
            else:
                logger.warning(f"   ⚠️  CHROMA_TENANT not found in .env.demo")
        else:
            logger.warning(f"   ⚠️  .env.demo file not found at {demo_env_file}")
    
    if "prod" in target_envs:
        prod_env_file = os.path.join(root_dir, ".env.prod")
        if os.path.exists(prod_env_file):
            logger.info(f"📄 Loading prod credentials from {prod_env_file}")
            prod_env_vars = load_env_file(prod_env_file)
            if DOTENV_AVAILABLE:
                load_dotenv(prod_env_file, override=False)
            # Log what we found
            if prod_env_vars.get("CHROMA_API_KEY"):
                logger.info(f"   ✅ Found CHROMA_API_KEY in .env.prod (length: {len(prod_env_vars['CHROMA_API_KEY'])})")
            else:
                logger.warning(f"   ⚠️  CHROMA_API_KEY not found in .env.prod")
            if prod_env_vars.get("CHROMA_TENANT"):
                logger.info(f"   ✅ Found CHROMA_TENANT in .env.prod")
            else:
                logger.warning(f"   ⚠️  CHROMA_TENANT not found in .env.prod")
        else:
            logger.warning(f"   ⚠️  .env.prod file not found at {prod_env_file}")
    
    # Get source credentials
    source_api_key = args.source_api_key or os.getenv("CHROMA_API_KEY")
    source_tenant = args.source_tenant or os.getenv("CHROMA_TENANT")
    source_database = args.source_db
    
    # Get demo credentials (priority: args > .env.demo > source > env)
    demo_api_key = None
    demo_tenant = None
    if "demo" in target_envs:
        # Explicitly use .env.demo credentials first (don't fallback to source)
        demo_api_key = args.target_api_key or demo_env_vars.get("CHROMA_API_KEY")
        demo_tenant = args.target_tenant or demo_env_vars.get("CHROMA_TENANT")
        # Only fallback if not found in .env.demo
        if not demo_api_key:
            logger.warning("   ⚠️  No demo API key found in .env.demo, using source credentials")
            demo_api_key = source_api_key or os.getenv("CHROMA_API_KEY")
        if not demo_tenant:
            logger.warning("   ⚠️  No demo tenant found in .env.demo, using source tenant")
            demo_tenant = source_tenant or os.getenv("CHROMA_TENANT")
        logger.info(f"   🔑 Using demo API key: {'From .env.demo' if demo_env_vars.get('CHROMA_API_KEY') else 'From source/default'} (first 10 chars: {demo_api_key[:10] if demo_api_key else 'None'}...)")
    
    # Get prod credentials (priority: args > .env.prod > source > env)
    prod_api_key = None
    prod_tenant = None
    if "prod" in target_envs:
        # Explicitly use .env.prod credentials first (don't fallback to source)
        prod_api_key = args.target_api_key or prod_env_vars.get("CHROMA_API_KEY")
        prod_tenant = args.target_tenant or prod_env_vars.get("CHROMA_TENANT")
        # Only fallback if not found in .env.prod
        if not prod_api_key:
            logger.warning("   ⚠️  No prod API key found in .env.prod, using source credentials")
            prod_api_key = source_api_key or os.getenv("CHROMA_API_KEY")
        if not prod_tenant:
            logger.warning("   ⚠️  No prod tenant found in .env.prod, using source tenant")
            prod_tenant = source_tenant or os.getenv("CHROMA_TENANT")
        logger.info(f"   🔑 Using prod API key: {'From .env.prod' if prod_env_vars.get('CHROMA_API_KEY') else 'From source/default'} (first 10 chars: {prod_api_key[:10] if prod_api_key else 'None'}...)")
    
    # Fallback target credentials (for backward compatibility)
    target_api_key = args.target_api_key or source_api_key or os.getenv("CHROMA_API_KEY")
    target_tenant = args.target_tenant or source_tenant or os.getenv("CHROMA_TENANT")
    
    # Validate source credentials
    if not source_api_key or not source_tenant:
        logger.error("❌ Source ChromaDB credentials not provided.")
        logger.error("   Set CHROMA_API_KEY and CHROMA_TENANT environment variables")
        logger.error("   Or use --source-api-key and --source-tenant arguments")
        return 1
    
    # Validate target credentials
    if not target_api_key or not target_tenant:
        logger.error("❌ Target ChromaDB credentials not provided.")
        logger.error("   Set CHROMA_API_KEY and CHROMA_TENANT environment variables")
        logger.error("   Or use --target-api-key and --target-tenant arguments")
        return 1
    
    logger.info("="*60)
    logger.info("🚀 COURSE COPY: Source → Demo/Production")
    logger.info("="*60)
    logger.info(f"📋 Configuration:")
    logger.info(f"   Source Database: {source_database}")
    logger.info(f"   Source Tenant: {source_tenant}")
    logger.info(f"   Target Environments: {', '.join(target_envs)}")
    if "demo" in target_envs:
        logger.info(f"   Demo API Key: {'✅ Loaded from .env.demo' if demo_env_vars.get('CHROMA_API_KEY') else '⚠️  Using source/default'}")
        logger.info(f"   Demo Tenant: {demo_tenant}")
    if "prod" in target_envs:
        logger.info(f"   Prod API Key: {'✅ Loaded from .env.prod' if prod_env_vars.get('CHROMA_API_KEY') else '⚠️  Using source/default'}")
        logger.info(f"   Prod Tenant: {prod_tenant}")
    logger.info("="*60)
    logger.info("")
    logger.info("⚠️  NOTE: If you get 'Database does not match' errors, you need:")
    logger.info("   - API keys that have access to demo/prod databases")
    logger.info("   - Or use --target-api-key and --target-tenant for demo/prod credentials")
    logger.info("")
    
    try:
        # Step 1: Connect to source database and get courses
        logger.info(f"\n📥 Step 1: Fetching courses from source database: {source_database}")
        source_client = create_chromadb_client(source_api_key, source_tenant, source_database)
        
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        try:
            source_collection = source_client.get_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        except Exception as e:
            logger.warning(f"Could not get source collection with embedding function: {e}")
            logger.info("Trying without embedding function...")
            source_collection = source_client.get_collection(name="courses_knowledge_base")
        
        source_count = source_collection.count()
        logger.info(f"✅ Connected to source. Found {source_count} courses")
        
        if source_count == 0:
            logger.warning("⚠️ No courses found in source database. Nothing to copy.")
            return 1
        
        # Get all courses from source
        source_courses = get_all_courses_from_collection(source_collection)
        logger.info(f"✅ Retrieved {len(source_courses)} courses from source")
        
        # Step 2: Copy to target environments
        results = []
        
        if "demo" in target_envs:
            # Use demo-specific credentials from .env.demo if available
            demo_result = copy_courses_to_target(
                source_courses=source_courses,
                target_api_key=demo_api_key,
                target_tenant=demo_tenant,
                target_database=args.demo_db,
                target_name="demo"
            )
            results.append(demo_result)
        
        if "prod" in target_envs:
            # Use prod-specific credentials from .env.prod if available
            prod_result = copy_courses_to_target(
                source_courses=source_courses,
                target_api_key=prod_api_key,
                target_tenant=prod_tenant,
                target_database=args.prod_db,
                target_name="production"
            )
            results.append(prod_result)
        
        # Step 3: Summary
        logger.info("\n" + "="*60)
        logger.info("📊 COPY SUMMARY")
        logger.info("="*60)
        logger.info(f"Source database: {source_database}")
        logger.info(f"Source courses: {source_count}")
        logger.info("")
        
        for result in results:
            if result.get("success"):
                logger.info(f"✅ {result['target_name'].upper()}: {result['courses_after']} courses (added {result['courses_added']})")
            else:
                logger.error(f"❌ {result['target_name'].upper()}: Failed - {result.get('error', 'Unknown error')}")
        
        logger.info("="*60)
        
        # Check if all copies succeeded
        all_success = all(r.get("success", False) for r in results)
        if all_success:
            logger.info("✅ Successfully copied courses to all target environments!")
            return 0
        else:
            logger.error("⚠️ Some copies failed. Check logs above for details.")
            return 1
        
    except Exception as e:
        logger.error(f"❌ Failed to copy courses: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())


Script to copy all courses from a source database (dev/qa) to demo and production databases.
This script reads all courses from the source environment and adds them to demo and production.

Usage:
    # Copy from dev to demo and production
    python scripts/copy_courses_to_demo_prod.py --source-db dev-jobsify-agent --target-envs demo,prod
    
    # Copy from QA to demo and production
    python scripts/copy_courses_to_demo_prod.py --source-db qa-jobsify-agent --target-envs demo,prod
    
    # Copy to specific databases
    python scripts/copy_courses_to_demo_prod.py --source-db dev-jobsify-agent --demo-db demo-jobsify-agent --prod-db prod-jobsify-agent
"""

import sys
import os
import argparse
import logging
import hashlib
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.errors import ChromaAuthError
from chroma import normalize_metadata

# Try to load python-dotenv for .env file support
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

def load_env_file(env_file: str) -> Dict[str, str]:
    """
    Load environment variables from a .env file.
    Returns a dict of key-value pairs.
    """
    env_vars = {}
    if not os.path.exists(env_file):
        return env_vars
    
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value
    except Exception as e:
        logger.warning(f"Could not load {env_file}: {e}")
    
    return env_vars

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
    """Create a ChromaDB Cloud client (same logic as copy_courses_qa_to_dev.py)"""
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

def copy_courses_to_target(
    source_courses: List[Dict[str, Any]],
    target_api_key: str,
    target_tenant: str,
    target_database: str,
    target_name: str
) -> Dict[str, Any]:
    """
    Copy courses to a target database.
    
    Args:
        source_courses: List of course dictionaries to copy
        target_api_key: ChromaDB API key for target
        target_tenant: ChromaDB tenant for target
        target_database: ChromaDB database name for target
        target_name: Human-readable name for target (for logging)
        
    Returns:
        Dict with copy statistics
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Copying to {target_name.upper()} database: {target_database}")
        logger.info(f"{'='*60}")
        
        # Connect to target database
        target_client = create_chromadb_client(target_api_key, target_tenant, target_database)
        
        # Get or create target collection
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        try:
            target_collection = target_client.get_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        except Exception as e:
            logger.info(f"Creating {target_name} collection (error: {e})...")
            target_collection = target_client.get_or_create_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        
        target_count_before = target_collection.count()
        logger.info(f"✅ Connected to {target_name}. Current courses: {target_count_before}")
        
        # Prepare courses for batch add
        courses_to_add = []
        for course in source_courses:
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
        
        # Add courses in batches
        logger.info(f"📤 Adding {len(courses_to_add)} courses to {target_name} database...")
        
        batch_size = 100
        total_added = 0
        
        for i in range(0, len(courses_to_add), batch_size):
            batch = courses_to_add[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(courses_to_add) + batch_size - 1) // batch_size
            
            logger.info(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} courses)...")
            
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
            
            # Upsert batch to target collection
            try:
                target_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                total_added += len(batch)
                logger.info(f"  ✅ Added batch {batch_num}/{total_batches} ({len(batch)} courses)")
            except Exception as e:
                logger.error(f"  ❌ Error adding batch {batch_num}: {e}")
                raise
        
        # Verify
        target_count_after = target_collection.count()
        
        result = {
            "target_name": target_name,
            "target_database": target_database,
            "courses_before": target_count_before,
            "courses_after": target_count_after,
            "courses_added": target_count_after - target_count_before,
            "total_processed": total_added,
            "success": target_count_after >= target_count_before
        }
        
        logger.info(f"✅ {target_name.upper()} copy complete:")
        logger.info(f"   Before: {target_count_before} courses")
        logger.info(f"   After: {target_count_after} courses")
        logger.info(f"   Added: {target_count_after - target_count_before} courses")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error copying courses to {target_name}: {e}", exc_info=True)
        return {
            "target_name": target_name,
            "target_database": target_database,
            "success": False,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(
        description="Copy courses from source database to demo and/or production databases"
    )
    parser.add_argument(
        "--source-db",
        type=str,
        default="dev-jobsify-agent",
        help="Source database name (default: dev-jobsify-agent)"
    )
    parser.add_argument(
        "--demo-db",
        type=str,
        default="demo-jobsify-agent",
        help="Demo database name (default: demo-jobsify-agent)"
    )
    parser.add_argument(
        "--prod-db",
        type=str,
        default="prod-jobsify-agent",
        help="Production database name (default: prod-jobsify-agent)"
    )
    parser.add_argument(
        "--target-envs",
        type=str,
        help="Comma-separated list of target environments (demo,prod). If not provided, copies to both."
    )
    parser.add_argument(
        "--source-api-key",
        type=str,
        help="Source ChromaDB API key. If not provided, uses CHROMA_API_KEY from environment"
    )
    parser.add_argument(
        "--source-tenant",
        type=str,
        help="Source ChromaDB tenant. If not provided, uses CHROMA_TENANT from environment"
    )
    parser.add_argument(
        "--target-api-key",
        type=str,
        help="Target ChromaDB API key (for demo/prod). If not provided, uses CHROMA_API_KEY from environment"
    )
    parser.add_argument(
        "--target-tenant",
        type=str,
        help="Target ChromaDB tenant (for demo/prod). If not provided, uses CHROMA_TENANT from environment"
    )
    
    args = parser.parse_args()
    
    # Determine target environments first
    if args.target_envs:
        target_envs = [env.strip().lower() for env in args.target_envs.split(",")]
    else:
        target_envs = ["demo", "prod"]
    
    # Get root directory (KA)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    # Load environment files for target environments
    demo_env_vars = {}
    prod_env_vars = {}
    
    if "demo" in target_envs:
        demo_env_file = os.path.join(root_dir, ".env.demo")
        if os.path.exists(demo_env_file):
            logger.info(f"📄 Loading demo credentials from {demo_env_file}")
            demo_env_vars = load_env_file(demo_env_file)
            if DOTENV_AVAILABLE:
                load_dotenv(demo_env_file, override=False)
            # Log what we found
            if demo_env_vars.get("CHROMA_API_KEY"):
                logger.info(f"   ✅ Found CHROMA_API_KEY in .env.demo (length: {len(demo_env_vars['CHROMA_API_KEY'])})")
            else:
                logger.warning(f"   ⚠️  CHROMA_API_KEY not found in .env.demo")
            if demo_env_vars.get("CHROMA_TENANT"):
                logger.info(f"   ✅ Found CHROMA_TENANT in .env.demo")
            else:
                logger.warning(f"   ⚠️  CHROMA_TENANT not found in .env.demo")
        else:
            logger.warning(f"   ⚠️  .env.demo file not found at {demo_env_file}")
    
    if "prod" in target_envs:
        prod_env_file = os.path.join(root_dir, ".env.prod")
        if os.path.exists(prod_env_file):
            logger.info(f"📄 Loading prod credentials from {prod_env_file}")
            prod_env_vars = load_env_file(prod_env_file)
            if DOTENV_AVAILABLE:
                load_dotenv(prod_env_file, override=False)
            # Log what we found
            if prod_env_vars.get("CHROMA_API_KEY"):
                logger.info(f"   ✅ Found CHROMA_API_KEY in .env.prod (length: {len(prod_env_vars['CHROMA_API_KEY'])})")
            else:
                logger.warning(f"   ⚠️  CHROMA_API_KEY not found in .env.prod")
            if prod_env_vars.get("CHROMA_TENANT"):
                logger.info(f"   ✅ Found CHROMA_TENANT in .env.prod")
            else:
                logger.warning(f"   ⚠️  CHROMA_TENANT not found in .env.prod")
        else:
            logger.warning(f"   ⚠️  .env.prod file not found at {prod_env_file}")
    
    # Get source credentials
    source_api_key = args.source_api_key or os.getenv("CHROMA_API_KEY")
    source_tenant = args.source_tenant or os.getenv("CHROMA_TENANT")
    source_database = args.source_db
    
    # Get demo credentials (priority: args > .env.demo > source > env)
    demo_api_key = None
    demo_tenant = None
    if "demo" in target_envs:
        # Explicitly use .env.demo credentials first (don't fallback to source)
        demo_api_key = args.target_api_key or demo_env_vars.get("CHROMA_API_KEY")
        demo_tenant = args.target_tenant or demo_env_vars.get("CHROMA_TENANT")
        # Only fallback if not found in .env.demo
        if not demo_api_key:
            logger.warning("   ⚠️  No demo API key found in .env.demo, using source credentials")
            demo_api_key = source_api_key or os.getenv("CHROMA_API_KEY")
        if not demo_tenant:
            logger.warning("   ⚠️  No demo tenant found in .env.demo, using source tenant")
            demo_tenant = source_tenant or os.getenv("CHROMA_TENANT")
        logger.info(f"   🔑 Using demo API key: {'From .env.demo' if demo_env_vars.get('CHROMA_API_KEY') else 'From source/default'} (first 10 chars: {demo_api_key[:10] if demo_api_key else 'None'}...)")
    
    # Get prod credentials (priority: args > .env.prod > source > env)
    prod_api_key = None
    prod_tenant = None
    if "prod" in target_envs:
        # Explicitly use .env.prod credentials first (don't fallback to source)
        prod_api_key = args.target_api_key or prod_env_vars.get("CHROMA_API_KEY")
        prod_tenant = args.target_tenant or prod_env_vars.get("CHROMA_TENANT")
        # Only fallback if not found in .env.prod
        if not prod_api_key:
            logger.warning("   ⚠️  No prod API key found in .env.prod, using source credentials")
            prod_api_key = source_api_key or os.getenv("CHROMA_API_KEY")
        if not prod_tenant:
            logger.warning("   ⚠️  No prod tenant found in .env.prod, using source tenant")
            prod_tenant = source_tenant or os.getenv("CHROMA_TENANT")
        logger.info(f"   🔑 Using prod API key: {'From .env.prod' if prod_env_vars.get('CHROMA_API_KEY') else 'From source/default'} (first 10 chars: {prod_api_key[:10] if prod_api_key else 'None'}...)")
    
    # Fallback target credentials (for backward compatibility)
    target_api_key = args.target_api_key or source_api_key or os.getenv("CHROMA_API_KEY")
    target_tenant = args.target_tenant or source_tenant or os.getenv("CHROMA_TENANT")
    
    # Validate source credentials
    if not source_api_key or not source_tenant:
        logger.error("❌ Source ChromaDB credentials not provided.")
        logger.error("   Set CHROMA_API_KEY and CHROMA_TENANT environment variables")
        logger.error("   Or use --source-api-key and --source-tenant arguments")
        return 1
    
    # Validate target credentials
    if not target_api_key or not target_tenant:
        logger.error("❌ Target ChromaDB credentials not provided.")
        logger.error("   Set CHROMA_API_KEY and CHROMA_TENANT environment variables")
        logger.error("   Or use --target-api-key and --target-tenant arguments")
        return 1
    
    logger.info("="*60)
    logger.info("🚀 COURSE COPY: Source → Demo/Production")
    logger.info("="*60)
    logger.info(f"📋 Configuration:")
    logger.info(f"   Source Database: {source_database}")
    logger.info(f"   Source Tenant: {source_tenant}")
    logger.info(f"   Target Environments: {', '.join(target_envs)}")
    if "demo" in target_envs:
        logger.info(f"   Demo API Key: {'✅ Loaded from .env.demo' if demo_env_vars.get('CHROMA_API_KEY') else '⚠️  Using source/default'}")
        logger.info(f"   Demo Tenant: {demo_tenant}")
    if "prod" in target_envs:
        logger.info(f"   Prod API Key: {'✅ Loaded from .env.prod' if prod_env_vars.get('CHROMA_API_KEY') else '⚠️  Using source/default'}")
        logger.info(f"   Prod Tenant: {prod_tenant}")
    logger.info("="*60)
    logger.info("")
    logger.info("⚠️  NOTE: If you get 'Database does not match' errors, you need:")
    logger.info("   - API keys that have access to demo/prod databases")
    logger.info("   - Or use --target-api-key and --target-tenant for demo/prod credentials")
    logger.info("")
    
    try:
        # Step 1: Connect to source database and get courses
        logger.info(f"\n📥 Step 1: Fetching courses from source database: {source_database}")
        source_client = create_chromadb_client(source_api_key, source_tenant, source_database)
        
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        try:
            source_collection = source_client.get_collection(
                name="courses_knowledge_base",
                embedding_function=embedding_fn
            )
        except Exception as e:
            logger.warning(f"Could not get source collection with embedding function: {e}")
            logger.info("Trying without embedding function...")
            source_collection = source_client.get_collection(name="courses_knowledge_base")
        
        source_count = source_collection.count()
        logger.info(f"✅ Connected to source. Found {source_count} courses")
        
        if source_count == 0:
            logger.warning("⚠️ No courses found in source database. Nothing to copy.")
            return 1
        
        # Get all courses from source
        source_courses = get_all_courses_from_collection(source_collection)
        logger.info(f"✅ Retrieved {len(source_courses)} courses from source")
        
        # Step 2: Copy to target environments
        results = []
        
        if "demo" in target_envs:
            # Use demo-specific credentials from .env.demo if available
            demo_result = copy_courses_to_target(
                source_courses=source_courses,
                target_api_key=demo_api_key,
                target_tenant=demo_tenant,
                target_database=args.demo_db,
                target_name="demo"
            )
            results.append(demo_result)
        
        if "prod" in target_envs:
            # Use prod-specific credentials from .env.prod if available
            prod_result = copy_courses_to_target(
                source_courses=source_courses,
                target_api_key=prod_api_key,
                target_tenant=prod_tenant,
                target_database=args.prod_db,
                target_name="production"
            )
            results.append(prod_result)
        
        # Step 3: Summary
        logger.info("\n" + "="*60)
        logger.info("📊 COPY SUMMARY")
        logger.info("="*60)
        logger.info(f"Source database: {source_database}")
        logger.info(f"Source courses: {source_count}")
        logger.info("")
        
        for result in results:
            if result.get("success"):
                logger.info(f"✅ {result['target_name'].upper()}: {result['courses_after']} courses (added {result['courses_added']})")
            else:
                logger.error(f"❌ {result['target_name'].upper()}: Failed - {result.get('error', 'Unknown error')}")
        
        logger.info("="*60)
        
        # Check if all copies succeeded
        all_success = all(r.get("success", False) for r in results)
        if all_success:
            logger.info("✅ Successfully copied courses to all target environments!")
            return 0
        else:
            logger.error("⚠️ Some copies failed. Check logs above for details.")
            return 1
        
    except Exception as e:
        logger.error(f"❌ Failed to copy courses: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())

