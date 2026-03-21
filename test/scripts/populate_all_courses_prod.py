#!/usr/bin/env python3
"""
Master script to populate all courses in production database.
Runs all population scripts to add courses to the production environment.

Usage:
    # Load production environment variables
    source ../.env.prod
    python scripts/populate_all_courses_prod.py
    
    OR
    
    # Set environment variables manually
    export CHROMA_DATABASE=prod-jobsify-agent
    export CHROMA_API_KEY=your-prod-api-key
    export CHROMA_TENANT=your-prod-tenant
    python scripts/populate_all_courses_prod.py
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# List of all population scripts (same as dev)
POPULATION_SCRIPTS = [
    "populate_ai_ml_courses.py",
    "populate_blockchain_courses.py",
    "populate_business_courses.py",
    "populate_cloud_courses.py",
    "populate_comprehensive_courses.py",
    "populate_cybersecurity_courses.py",
    "populate_data_science_courses.py",
    "populate_frontend_courses.py",
    "populate_design_courses.py",
    "populate_education_training_courses.py",
    "populate_emerging_tech_courses.py",
    "populate_engineering_courses.py",
    "populate_health_wellness_courses.py",
    "populate_language_courses.py",
    "populate_legal_compliance_courses.py",
    "populate_mathematics_courses.py",
    "populate_media_communications_courses.py",
    "populate_operations_management_udemy_courses.py",
    "populate_personal_finance_courses.py",
    "populate_programming_languages.py",
    "populate_sales_courses.py",
    "populate_soft_skills_courses.py",
    "populate_supply_chain_udemy_courses.py",
    "populate_sustainability_csr_courses.py",
    "populate_warehouse_inventory_udemy_courses.py",
    "populate_software_testing_courses.py",
    "populate_power_bi_data_testing_courses.py",
    "populate_hairstylist_courses.py",
]

def load_env_file(env_file: str) -> dict:
    """Load environment variables from a .env file"""
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
                # Handle export KEY=VALUE format
                if line.startswith('export '):
                    line = line[7:]  # Remove 'export '
                # Parse KEY=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value
                    # Also set in environment
                    os.environ[key] = value
    except Exception as e:
        logger.warning(f"Could not load {env_file}: {e}")
    
    return env_vars

def run_script(script_path: str, env_vars: dict = None) -> tuple[bool, str]:
    """
    Run a population script and return success status and output.
    
    Args:
        script_path: Path to the script to run
        env_vars: Additional environment variables to set
        
    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        logger.info(f"📝 Running {script_path}...")
        
        # Prepare environment
        script_env = os.environ.copy()
        if env_vars:
            script_env.update(env_vars)
        
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.dirname(script_path)),  # Run from agents directory
            env=script_env,  # Pass environment variables
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per script
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {script_path} completed successfully")
            return True, result.stdout
        else:
            logger.error(f"❌ {script_path} failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏱️ {script_path} timed out after 5 minutes")
        return False, "Timeout"
    except Exception as e:
        logger.error(f"❌ Error running {script_path}: {e}")
        return False, str(e)

def main():
    """Run all population scripts for production environment"""
    # Get scripts directory
    scripts_dir = Path(__file__).parent
    agents_dir = scripts_dir.parent
    
    # Try to load .env.prod file from agents directory (where it actually is)
    prod_env_file = agents_dir / ".env.prod"
    env_vars = {}
    if prod_env_file.exists():
        logger.info(f"📄 Loading environment from {prod_env_file}")
        env_vars = load_env_file(str(prod_env_file))
        logger.info(f"✅ Loaded {len(env_vars)} environment variables from .env.prod")
    else:
        logger.warning(f"⚠️  .env.prod file not found at {prod_env_file}")
        logger.info("   Using environment variables from current shell")
    
    # Check current database
    current_db = os.getenv("CHROMA_DATABASE", "prod-jobsify-agent")
    logger.info("="*60)
    logger.info("🚀 Starting population of all courses in PRODUCTION database")
    logger.info("="*60)
    logger.info(f"📊 Target database: {current_db}")
    logger.info(f"📊 Target tenant: {os.getenv('CHROMA_TENANT', 'Not set')}")
    logger.info("="*60)
    logger.info("")
    
    # Verify credentials
    if not os.getenv("CHROMA_API_KEY"):
        logger.error("❌ CHROMA_API_KEY not set. Please set it in .env.prod or export it.")
        return 1
    
    if not os.getenv("CHROMA_TENANT"):
        logger.error("❌ CHROMA_TENANT not set. Please set it in .env.prod or export it.")
        return 1
    
    # Run all population scripts
    scripts_dir_path = scripts_dir
    successful = 0
    failed = 0
    
    for script_name in POPULATION_SCRIPTS:
        script_path = scripts_dir_path / script_name
        
        if not script_path.exists():
            logger.warning(f"⚠️  Script not found: {script_path}")
            failed += 1
            continue
        
        success, output = run_script(str(script_path), env_vars)
        
        if success:
            successful += 1
        else:
            failed += 1
            # Log last few lines of output for debugging
            if output:
                output_lines = output.strip().split('\n')
                logger.error(f"   Last output lines: {output_lines[-3:]}")
    
    # Summary
    logger.info("")
    logger.info("="*60)
    logger.info("📊 POPULATION SUMMARY")
    logger.info("="*60)
    logger.info(f"✅ Successful: {successful}/{len(POPULATION_SCRIPTS)}")
    logger.info(f"❌ Failed: {failed}/{len(POPULATION_SCRIPTS)}")
    logger.info("="*60)
    
    if failed == 0:
        logger.info("🎉 All courses successfully populated in PRODUCTION database!")
        return 0
    else:
        logger.warning(f"⚠️  {failed} scripts failed. Check logs above for details.")
        return 1

if __name__ == "__main__":
    exit(main())


Master script to populate all courses in production database.
Runs all population scripts to add courses to the production environment.

Usage:
    # Load production environment variables
    source ../.env.prod
    python scripts/populate_all_courses_prod.py
    
    OR
    
    # Set environment variables manually
    export CHROMA_DATABASE=prod-jobsify-agent
    export CHROMA_API_KEY=your-prod-api-key
    export CHROMA_TENANT=your-prod-tenant
    python scripts/populate_all_courses_prod.py
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# List of all population scripts (same as dev)
POPULATION_SCRIPTS = [
    "populate_ai_ml_courses.py",
    "populate_blockchain_courses.py",
    "populate_business_courses.py",
    "populate_cloud_courses.py",
    "populate_comprehensive_courses.py",
    "populate_cybersecurity_courses.py",
    "populate_data_science_courses.py",
    "populate_frontend_courses.py",
    "populate_design_courses.py",
    "populate_education_training_courses.py",
    "populate_emerging_tech_courses.py",
    "populate_engineering_courses.py",
    "populate_health_wellness_courses.py",
    "populate_language_courses.py",
    "populate_legal_compliance_courses.py",
    "populate_mathematics_courses.py",
    "populate_media_communications_courses.py",
    "populate_operations_management_udemy_courses.py",
    "populate_personal_finance_courses.py",
    "populate_programming_languages.py",
    "populate_sales_courses.py",
    "populate_soft_skills_courses.py",
    "populate_supply_chain_udemy_courses.py",
    "populate_sustainability_csr_courses.py",
    "populate_warehouse_inventory_udemy_courses.py",
    "populate_software_testing_courses.py",
    "populate_power_bi_data_testing_courses.py",
    "populate_hairstylist_courses.py",
]

def load_env_file(env_file: str) -> dict:
    """Load environment variables from a .env file"""
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
                # Handle export KEY=VALUE format
                if line.startswith('export '):
                    line = line[7:]  # Remove 'export '
                # Parse KEY=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value
                    # Also set in environment
                    os.environ[key] = value
    except Exception as e:
        logger.warning(f"Could not load {env_file}: {e}")
    
    return env_vars

def run_script(script_path: str, env_vars: dict = None) -> tuple[bool, str]:
    """
    Run a population script and return success status and output.
    
    Args:
        script_path: Path to the script to run
        env_vars: Additional environment variables to set
        
    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        logger.info(f"📝 Running {script_path}...")
        
        # Prepare environment
        script_env = os.environ.copy()
        if env_vars:
            script_env.update(env_vars)
        
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.dirname(script_path)),  # Run from agents directory
            env=script_env,  # Pass environment variables
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per script
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {script_path} completed successfully")
            return True, result.stdout
        else:
            logger.error(f"❌ {script_path} failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏱️ {script_path} timed out after 5 minutes")
        return False, "Timeout"
    except Exception as e:
        logger.error(f"❌ Error running {script_path}: {e}")
        return False, str(e)

def main():
    """Run all population scripts for production environment"""
    # Get scripts directory
    scripts_dir = Path(__file__).parent
    agents_dir = scripts_dir.parent
    
    # Try to load .env.prod file from agents directory (where it actually is)
    prod_env_file = agents_dir / ".env.prod"
    env_vars = {}
    if prod_env_file.exists():
        logger.info(f"📄 Loading environment from {prod_env_file}")
        env_vars = load_env_file(str(prod_env_file))
        logger.info(f"✅ Loaded {len(env_vars)} environment variables from .env.prod")
    else:
        logger.warning(f"⚠️  .env.prod file not found at {prod_env_file}")
        logger.info("   Using environment variables from current shell")
    
    # Check current database
    current_db = os.getenv("CHROMA_DATABASE", "prod-jobsify-agent")
    logger.info("="*60)
    logger.info("🚀 Starting population of all courses in PRODUCTION database")
    logger.info("="*60)
    logger.info(f"📊 Target database: {current_db}")
    logger.info(f"📊 Target tenant: {os.getenv('CHROMA_TENANT', 'Not set')}")
    logger.info("="*60)
    logger.info("")
    
    # Verify credentials
    if not os.getenv("CHROMA_API_KEY"):
        logger.error("❌ CHROMA_API_KEY not set. Please set it in .env.prod or export it.")
        return 1
    
    if not os.getenv("CHROMA_TENANT"):
        logger.error("❌ CHROMA_TENANT not set. Please set it in .env.prod or export it.")
        return 1
    
    # Run all population scripts
    scripts_dir_path = scripts_dir
    successful = 0
    failed = 0
    
    for script_name in POPULATION_SCRIPTS:
        script_path = scripts_dir_path / script_name
        
        if not script_path.exists():
            logger.warning(f"⚠️  Script not found: {script_path}")
            failed += 1
            continue
        
        success, output = run_script(str(script_path), env_vars)
        
        if success:
            successful += 1
        else:
            failed += 1
            # Log last few lines of output for debugging
            if output:
                output_lines = output.strip().split('\n')
                logger.error(f"   Last output lines: {output_lines[-3:]}")
    
    # Summary
    logger.info("")
    logger.info("="*60)
    logger.info("📊 POPULATION SUMMARY")
    logger.info("="*60)
    logger.info(f"✅ Successful: {successful}/{len(POPULATION_SCRIPTS)}")
    logger.info(f"❌ Failed: {failed}/{len(POPULATION_SCRIPTS)}")
    logger.info("="*60)
    
    if failed == 0:
        logger.info("🎉 All courses successfully populated in PRODUCTION database!")
        return 0
    else:
        logger.warning(f"⚠️  {failed} scripts failed. Check logs above for details.")
        return 1

if __name__ == "__main__":
    exit(main())


Master script to populate all courses in production database.
Runs all population scripts to add courses to the production environment.

Usage:
    # Load production environment variables
    source ../.env.prod
    python scripts/populate_all_courses_prod.py
    
    OR
    
    # Set environment variables manually
    export CHROMA_DATABASE=prod-jobsify-agent
    export CHROMA_API_KEY=your-prod-api-key
    export CHROMA_TENANT=your-prod-tenant
    python scripts/populate_all_courses_prod.py
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# List of all population scripts (same as dev)
POPULATION_SCRIPTS = [
    "populate_ai_ml_courses.py",
    "populate_blockchain_courses.py",
    "populate_business_courses.py",
    "populate_cloud_courses.py",
    "populate_comprehensive_courses.py",
    "populate_cybersecurity_courses.py",
    "populate_data_science_courses.py",
    "populate_frontend_courses.py",
    "populate_design_courses.py",
    "populate_education_training_courses.py",
    "populate_emerging_tech_courses.py",
    "populate_engineering_courses.py",
    "populate_health_wellness_courses.py",
    "populate_language_courses.py",
    "populate_legal_compliance_courses.py",
    "populate_mathematics_courses.py",
    "populate_media_communications_courses.py",
    "populate_operations_management_udemy_courses.py",
    "populate_personal_finance_courses.py",
    "populate_programming_languages.py",
    "populate_sales_courses.py",
    "populate_soft_skills_courses.py",
    "populate_supply_chain_udemy_courses.py",
    "populate_sustainability_csr_courses.py",
    "populate_warehouse_inventory_udemy_courses.py",
    "populate_software_testing_courses.py",
    "populate_power_bi_data_testing_courses.py",
    "populate_hairstylist_courses.py",
]

def load_env_file(env_file: str) -> dict:
    """Load environment variables from a .env file"""
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
                # Handle export KEY=VALUE format
                if line.startswith('export '):
                    line = line[7:]  # Remove 'export '
                # Parse KEY=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value
                    # Also set in environment
                    os.environ[key] = value
    except Exception as e:
        logger.warning(f"Could not load {env_file}: {e}")
    
    return env_vars

def run_script(script_path: str, env_vars: dict = None) -> tuple[bool, str]:
    """
    Run a population script and return success status and output.
    
    Args:
        script_path: Path to the script to run
        env_vars: Additional environment variables to set
        
    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        logger.info(f"📝 Running {script_path}...")
        
        # Prepare environment
        script_env = os.environ.copy()
        if env_vars:
            script_env.update(env_vars)
        
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.dirname(script_path)),  # Run from agents directory
            env=script_env,  # Pass environment variables
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per script
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {script_path} completed successfully")
            return True, result.stdout
        else:
            logger.error(f"❌ {script_path} failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏱️ {script_path} timed out after 5 minutes")
        return False, "Timeout"
    except Exception as e:
        logger.error(f"❌ Error running {script_path}: {e}")
        return False, str(e)

def main():
    """Run all population scripts for production environment"""
    # Get scripts directory
    scripts_dir = Path(__file__).parent
    agents_dir = scripts_dir.parent
    
    # Try to load .env.prod file from agents directory (where it actually is)
    prod_env_file = agents_dir / ".env.prod"
    env_vars = {}
    if prod_env_file.exists():
        logger.info(f"📄 Loading environment from {prod_env_file}")
        env_vars = load_env_file(str(prod_env_file))
        logger.info(f"✅ Loaded {len(env_vars)} environment variables from .env.prod")
    else:
        logger.warning(f"⚠️  .env.prod file not found at {prod_env_file}")
        logger.info("   Using environment variables from current shell")
    
    # Check current database
    current_db = os.getenv("CHROMA_DATABASE", "prod-jobsify-agent")
    logger.info("="*60)
    logger.info("🚀 Starting population of all courses in PRODUCTION database")
    logger.info("="*60)
    logger.info(f"📊 Target database: {current_db}")
    logger.info(f"📊 Target tenant: {os.getenv('CHROMA_TENANT', 'Not set')}")
    logger.info("="*60)
    logger.info("")
    
    # Verify credentials
    if not os.getenv("CHROMA_API_KEY"):
        logger.error("❌ CHROMA_API_KEY not set. Please set it in .env.prod or export it.")
        return 1
    
    if not os.getenv("CHROMA_TENANT"):
        logger.error("❌ CHROMA_TENANT not set. Please set it in .env.prod or export it.")
        return 1
    
    # Run all population scripts
    scripts_dir_path = scripts_dir
    successful = 0
    failed = 0
    
    for script_name in POPULATION_SCRIPTS:
        script_path = scripts_dir_path / script_name
        
        if not script_path.exists():
            logger.warning(f"⚠️  Script not found: {script_path}")
            failed += 1
            continue
        
        success, output = run_script(str(script_path), env_vars)
        
        if success:
            successful += 1
        else:
            failed += 1
            # Log last few lines of output for debugging
            if output:
                output_lines = output.strip().split('\n')
                logger.error(f"   Last output lines: {output_lines[-3:]}")
    
    # Summary
    logger.info("")
    logger.info("="*60)
    logger.info("📊 POPULATION SUMMARY")
    logger.info("="*60)
    logger.info(f"✅ Successful: {successful}/{len(POPULATION_SCRIPTS)}")
    logger.info(f"❌ Failed: {failed}/{len(POPULATION_SCRIPTS)}")
    logger.info("="*60)
    
    if failed == 0:
        logger.info("🎉 All courses successfully populated in PRODUCTION database!")
        return 0
    else:
        logger.warning(f"⚠️  {failed} scripts failed. Check logs above for details.")
        return 1

if __name__ == "__main__":
    exit(main())


Master script to populate all courses in production database.
Runs all population scripts to add courses to the production environment.

Usage:
    # Load production environment variables
    source ../.env.prod
    python scripts/populate_all_courses_prod.py
    
    OR
    
    # Set environment variables manually
    export CHROMA_DATABASE=prod-jobsify-agent
    export CHROMA_API_KEY=your-prod-api-key
    export CHROMA_TENANT=your-prod-tenant
    python scripts/populate_all_courses_prod.py
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# List of all population scripts (same as dev)
POPULATION_SCRIPTS = [
    "populate_ai_ml_courses.py",
    "populate_blockchain_courses.py",
    "populate_business_courses.py",
    "populate_cloud_courses.py",
    "populate_comprehensive_courses.py",
    "populate_cybersecurity_courses.py",
    "populate_data_science_courses.py",
    "populate_frontend_courses.py",
    "populate_design_courses.py",
    "populate_education_training_courses.py",
    "populate_emerging_tech_courses.py",
    "populate_engineering_courses.py",
    "populate_health_wellness_courses.py",
    "populate_language_courses.py",
    "populate_legal_compliance_courses.py",
    "populate_mathematics_courses.py",
    "populate_media_communications_courses.py",
    "populate_operations_management_udemy_courses.py",
    "populate_personal_finance_courses.py",
    "populate_programming_languages.py",
    "populate_sales_courses.py",
    "populate_soft_skills_courses.py",
    "populate_supply_chain_udemy_courses.py",
    "populate_sustainability_csr_courses.py",
    "populate_warehouse_inventory_udemy_courses.py",
    "populate_software_testing_courses.py",
    "populate_power_bi_data_testing_courses.py",
    "populate_hairstylist_courses.py",
]

def load_env_file(env_file: str) -> dict:
    """Load environment variables from a .env file"""
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
                # Handle export KEY=VALUE format
                if line.startswith('export '):
                    line = line[7:]  # Remove 'export '
                # Parse KEY=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value
                    # Also set in environment
                    os.environ[key] = value
    except Exception as e:
        logger.warning(f"Could not load {env_file}: {e}")
    
    return env_vars

def run_script(script_path: str, env_vars: dict = None) -> tuple[bool, str]:
    """
    Run a population script and return success status and output.
    
    Args:
        script_path: Path to the script to run
        env_vars: Additional environment variables to set
        
    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        logger.info(f"📝 Running {script_path}...")
        
        # Prepare environment
        script_env = os.environ.copy()
        if env_vars:
            script_env.update(env_vars)
        
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.dirname(script_path)),  # Run from agents directory
            env=script_env,  # Pass environment variables
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per script
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {script_path} completed successfully")
            return True, result.stdout
        else:
            logger.error(f"❌ {script_path} failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏱️ {script_path} timed out after 5 minutes")
        return False, "Timeout"
    except Exception as e:
        logger.error(f"❌ Error running {script_path}: {e}")
        return False, str(e)

def main():
    """Run all population scripts for production environment"""
    # Get scripts directory
    scripts_dir = Path(__file__).parent
    agents_dir = scripts_dir.parent
    
    # Try to load .env.prod file from agents directory (where it actually is)
    prod_env_file = agents_dir / ".env.prod"
    env_vars = {}
    if prod_env_file.exists():
        logger.info(f"📄 Loading environment from {prod_env_file}")
        env_vars = load_env_file(str(prod_env_file))
        logger.info(f"✅ Loaded {len(env_vars)} environment variables from .env.prod")
    else:
        logger.warning(f"⚠️  .env.prod file not found at {prod_env_file}")
        logger.info("   Using environment variables from current shell")
    
    # Check current database
    current_db = os.getenv("CHROMA_DATABASE", "prod-jobsify-agent")
    logger.info("="*60)
    logger.info("🚀 Starting population of all courses in PRODUCTION database")
    logger.info("="*60)
    logger.info(f"📊 Target database: {current_db}")
    logger.info(f"📊 Target tenant: {os.getenv('CHROMA_TENANT', 'Not set')}")
    logger.info("="*60)
    logger.info("")
    
    # Verify credentials
    if not os.getenv("CHROMA_API_KEY"):
        logger.error("❌ CHROMA_API_KEY not set. Please set it in .env.prod or export it.")
        return 1
    
    if not os.getenv("CHROMA_TENANT"):
        logger.error("❌ CHROMA_TENANT not set. Please set it in .env.prod or export it.")
        return 1
    
    # Run all population scripts
    scripts_dir_path = scripts_dir
    successful = 0
    failed = 0
    
    for script_name in POPULATION_SCRIPTS:
        script_path = scripts_dir_path / script_name
        
        if not script_path.exists():
            logger.warning(f"⚠️  Script not found: {script_path}")
            failed += 1
            continue
        
        success, output = run_script(str(script_path), env_vars)
        
        if success:
            successful += 1
        else:
            failed += 1
            # Log last few lines of output for debugging
            if output:
                output_lines = output.strip().split('\n')
                logger.error(f"   Last output lines: {output_lines[-3:]}")
    
    # Summary
    logger.info("")
    logger.info("="*60)
    logger.info("📊 POPULATION SUMMARY")
    logger.info("="*60)
    logger.info(f"✅ Successful: {successful}/{len(POPULATION_SCRIPTS)}")
    logger.info(f"❌ Failed: {failed}/{len(POPULATION_SCRIPTS)}")
    logger.info("="*60)
    
    if failed == 0:
        logger.info("🎉 All courses successfully populated in PRODUCTION database!")
        return 0
    else:
        logger.warning(f"⚠️  {failed} scripts failed. Check logs above for details.")
        return 1

if __name__ == "__main__":
    exit(main())

