#!/usr/bin/env python3
"""
Master script to populate all courses in dev database.
Runs all population scripts to add courses to the dev environment.

Usage:
    # Make sure you're in dev environment
    export CHROMA_DATABASE=dev-jobsify-agent
    python scripts/populate_all_courses_dev.py
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

# List of all population scripts (excluding the ones we removed)
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
    "populate_medical_education_courses.py",
    "populate_database_administration_courses.py",
    "populate_system_administration_courses.py",
    "populate_sre_courses.py",
    "populate_data_engineering_courses.py",
    "populate_api_development_courses.py",
    "populate_container_orchestration_courses.py",
    "populate_infrastructure_as_code_courses.py",
    "populate_observability_monitoring_courses.py",
    "populate_microservices_courses.py",
    "populate_mlops_courses.py",
    "populate_hairstylist_courses.py",
    # Note: populate_supply_chain_operations_courses.py, populate_civil_engineering_courses.py,
    # populate_biotech_pharma_courses.py were removed as those categories were deleted
]

def run_script(script_path: str) -> tuple[bool, str]:
    """
    Run a population script and return success status and output.
    
    Args:
        script_path: Path to the script to run
        
    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        logger.info(f"📝 Running {script_path}...")
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.dirname(script_path)),  # Run from agents directory
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
    """Run all population scripts"""
    # Get scripts directory
    scripts_dir = Path(__file__).parent
    agents_dir = scripts_dir.parent
    
    # Check current database
    current_db = os.getenv("CHROMA_DATABASE", "dev-jobsify-agent")
    logger.info("="*60)
    logger.info("🚀 Starting population of all courses in dev database")
    logger.info("="*60)
    logger.info(f"📊 Target database: {current_db}")
    logger.info(f"📁 Scripts directory: {scripts_dir}")
    logger.info("")
    
    # Verify we're targeting dev
    if "dev" not in current_db.lower() and "qa" not in current_db.lower():
        logger.warning(f"⚠️  Warning: Database '{current_db}' doesn't look like dev or qa")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Aborted by user")
            return 1
    
    # Run all scripts
    results = []
    successful = 0
    failed = 0
    
    for script_name in POPULATION_SCRIPTS:
        script_path = scripts_dir / script_name
        
        if not script_path.exists():
            logger.warning(f"⚠️  Script not found: {script_name}, skipping...")
            continue
        
        success, output = run_script(str(script_path))
        results.append((script_name, success, output))
        
        if success:
            successful += 1
        else:
            failed += 1
        
        logger.info("")  # Blank line between scripts
    
    # Summary
    logger.info("="*60)
    logger.info("📊 POPULATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total scripts: {len(POPULATION_SCRIPTS)}")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"⏭️  Skipped: {len(POPULATION_SCRIPTS) - successful - failed}")
    logger.info("")
    
    if failed > 0:
        logger.info("Failed scripts:")
        for script_name, success, output in results:
            if not success:
                logger.info(f"  - {script_name}")
    
    # Final course count
    try:
        from agents.course_knowledge_base import CourseKnowledgeBase
        kb = CourseKnowledgeBase()
        final_count = kb.get_course_count()
        logger.info(f"📚 Final course count in {current_db}: {final_count}")
    except Exception as e:
        logger.warning(f"Could not get final course count: {e}")
    
    logger.info("="*60)
    
    if failed == 0:
        logger.info("✅ All scripts completed successfully!")
        return 0
    else:
        logger.warning(f"⚠️  {failed} script(s) failed. Check logs above for details.")
        return 1

if __name__ == "__main__":
    exit(main())


Master script to populate all courses in dev database.
Runs all population scripts to add courses to the dev environment.

Usage:
    # Make sure you're in dev environment
    export CHROMA_DATABASE=dev-jobsify-agent
    python scripts/populate_all_courses_dev.py
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

# List of all population scripts (excluding the ones we removed)
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
    "populate_medical_education_courses.py",
    "populate_database_administration_courses.py",
    "populate_system_administration_courses.py",
    "populate_sre_courses.py",
    "populate_data_engineering_courses.py",
    "populate_api_development_courses.py",
    "populate_container_orchestration_courses.py",
    "populate_infrastructure_as_code_courses.py",
    "populate_observability_monitoring_courses.py",
    "populate_microservices_courses.py",
    "populate_mlops_courses.py",
    "populate_hairstylist_courses.py",
    # Note: populate_supply_chain_operations_courses.py, populate_civil_engineering_courses.py,
    # populate_biotech_pharma_courses.py were removed as those categories were deleted
]

def run_script(script_path: str) -> tuple[bool, str]:
    """
    Run a population script and return success status and output.
    
    Args:
        script_path: Path to the script to run
        
    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        logger.info(f"📝 Running {script_path}...")
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.dirname(script_path)),  # Run from agents directory
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
    """Run all population scripts"""
    # Get scripts directory
    scripts_dir = Path(__file__).parent
    agents_dir = scripts_dir.parent
    
    # Check current database
    current_db = os.getenv("CHROMA_DATABASE", "dev-jobsify-agent")
    logger.info("="*60)
    logger.info("🚀 Starting population of all courses in dev database")
    logger.info("="*60)
    logger.info(f"📊 Target database: {current_db}")
    logger.info(f"📁 Scripts directory: {scripts_dir}")
    logger.info("")
    
    # Verify we're targeting dev
    if "dev" not in current_db.lower() and "qa" not in current_db.lower():
        logger.warning(f"⚠️  Warning: Database '{current_db}' doesn't look like dev or qa")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Aborted by user")
            return 1
    
    # Run all scripts
    results = []
    successful = 0
    failed = 0
    
    for script_name in POPULATION_SCRIPTS:
        script_path = scripts_dir / script_name
        
        if not script_path.exists():
            logger.warning(f"⚠️  Script not found: {script_name}, skipping...")
            continue
        
        success, output = run_script(str(script_path))
        results.append((script_name, success, output))
        
        if success:
            successful += 1
        else:
            failed += 1
        
        logger.info("")  # Blank line between scripts
    
    # Summary
    logger.info("="*60)
    logger.info("📊 POPULATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total scripts: {len(POPULATION_SCRIPTS)}")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"⏭️  Skipped: {len(POPULATION_SCRIPTS) - successful - failed}")
    logger.info("")
    
    if failed > 0:
        logger.info("Failed scripts:")
        for script_name, success, output in results:
            if not success:
                logger.info(f"  - {script_name}")
    
    # Final course count
    try:
        from agents.course_knowledge_base import CourseKnowledgeBase
        kb = CourseKnowledgeBase()
        final_count = kb.get_course_count()
        logger.info(f"📚 Final course count in {current_db}: {final_count}")
    except Exception as e:
        logger.warning(f"Could not get final course count: {e}")
    
    logger.info("="*60)
    
    if failed == 0:
        logger.info("✅ All scripts completed successfully!")
        return 0
    else:
        logger.warning(f"⚠️  {failed} script(s) failed. Check logs above for details.")
        return 1

if __name__ == "__main__":
    exit(main())


Master script to populate all courses in dev database.
Runs all population scripts to add courses to the dev environment.

Usage:
    # Make sure you're in dev environment
    export CHROMA_DATABASE=dev-jobsify-agent
    python scripts/populate_all_courses_dev.py
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

# List of all population scripts (excluding the ones we removed)
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
    "populate_medical_education_courses.py",
    "populate_database_administration_courses.py",
    "populate_system_administration_courses.py",
    "populate_sre_courses.py",
    "populate_data_engineering_courses.py",
    "populate_api_development_courses.py",
    "populate_container_orchestration_courses.py",
    "populate_infrastructure_as_code_courses.py",
    "populate_observability_monitoring_courses.py",
    "populate_microservices_courses.py",
    "populate_mlops_courses.py",
    "populate_hairstylist_courses.py",
    # Note: populate_supply_chain_operations_courses.py, populate_civil_engineering_courses.py,
    # populate_biotech_pharma_courses.py were removed as those categories were deleted
]

def run_script(script_path: str) -> tuple[bool, str]:
    """
    Run a population script and return success status and output.
    
    Args:
        script_path: Path to the script to run
        
    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        logger.info(f"📝 Running {script_path}...")
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.dirname(script_path)),  # Run from agents directory
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
    """Run all population scripts"""
    # Get scripts directory
    scripts_dir = Path(__file__).parent
    agents_dir = scripts_dir.parent
    
    # Check current database
    current_db = os.getenv("CHROMA_DATABASE", "dev-jobsify-agent")
    logger.info("="*60)
    logger.info("🚀 Starting population of all courses in dev database")
    logger.info("="*60)
    logger.info(f"📊 Target database: {current_db}")
    logger.info(f"📁 Scripts directory: {scripts_dir}")
    logger.info("")
    
    # Verify we're targeting dev
    if "dev" not in current_db.lower() and "qa" not in current_db.lower():
        logger.warning(f"⚠️  Warning: Database '{current_db}' doesn't look like dev or qa")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Aborted by user")
            return 1
    
    # Run all scripts
    results = []
    successful = 0
    failed = 0
    
    for script_name in POPULATION_SCRIPTS:
        script_path = scripts_dir / script_name
        
        if not script_path.exists():
            logger.warning(f"⚠️  Script not found: {script_name}, skipping...")
            continue
        
        success, output = run_script(str(script_path))
        results.append((script_name, success, output))
        
        if success:
            successful += 1
        else:
            failed += 1
        
        logger.info("")  # Blank line between scripts
    
    # Summary
    logger.info("="*60)
    logger.info("📊 POPULATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total scripts: {len(POPULATION_SCRIPTS)}")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"⏭️  Skipped: {len(POPULATION_SCRIPTS) - successful - failed}")
    logger.info("")
    
    if failed > 0:
        logger.info("Failed scripts:")
        for script_name, success, output in results:
            if not success:
                logger.info(f"  - {script_name}")
    
    # Final course count
    try:
        from agents.course_knowledge_base import CourseKnowledgeBase
        kb = CourseKnowledgeBase()
        final_count = kb.get_course_count()
        logger.info(f"📚 Final course count in {current_db}: {final_count}")
    except Exception as e:
        logger.warning(f"Could not get final course count: {e}")
    
    logger.info("="*60)
    
    if failed == 0:
        logger.info("✅ All scripts completed successfully!")
        return 0
    else:
        logger.warning(f"⚠️  {failed} script(s) failed. Check logs above for details.")
        return 1

if __name__ == "__main__":
    exit(main())


Master script to populate all courses in dev database.
Runs all population scripts to add courses to the dev environment.

Usage:
    # Make sure you're in dev environment
    export CHROMA_DATABASE=dev-jobsify-agent
    python scripts/populate_all_courses_dev.py
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

# List of all population scripts (excluding the ones we removed)
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
    "populate_medical_education_courses.py",
    "populate_database_administration_courses.py",
    "populate_system_administration_courses.py",
    "populate_sre_courses.py",
    "populate_data_engineering_courses.py",
    "populate_api_development_courses.py",
    "populate_container_orchestration_courses.py",
    "populate_infrastructure_as_code_courses.py",
    "populate_observability_monitoring_courses.py",
    "populate_microservices_courses.py",
    "populate_mlops_courses.py",
    "populate_hairstylist_courses.py",
    # Note: populate_supply_chain_operations_courses.py, populate_civil_engineering_courses.py,
    # populate_biotech_pharma_courses.py were removed as those categories were deleted
]

def run_script(script_path: str) -> tuple[bool, str]:
    """
    Run a population script and return success status and output.
    
    Args:
        script_path: Path to the script to run
        
    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        logger.info(f"📝 Running {script_path}...")
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.dirname(script_path)),  # Run from agents directory
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
    """Run all population scripts"""
    # Get scripts directory
    scripts_dir = Path(__file__).parent
    agents_dir = scripts_dir.parent
    
    # Check current database
    current_db = os.getenv("CHROMA_DATABASE", "dev-jobsify-agent")
    logger.info("="*60)
    logger.info("🚀 Starting population of all courses in dev database")
    logger.info("="*60)
    logger.info(f"📊 Target database: {current_db}")
    logger.info(f"📁 Scripts directory: {scripts_dir}")
    logger.info("")
    
    # Verify we're targeting dev
    if "dev" not in current_db.lower() and "qa" not in current_db.lower():
        logger.warning(f"⚠️  Warning: Database '{current_db}' doesn't look like dev or qa")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Aborted by user")
            return 1
    
    # Run all scripts
    results = []
    successful = 0
    failed = 0
    
    for script_name in POPULATION_SCRIPTS:
        script_path = scripts_dir / script_name
        
        if not script_path.exists():
            logger.warning(f"⚠️  Script not found: {script_name}, skipping...")
            continue
        
        success, output = run_script(str(script_path))
        results.append((script_name, success, output))
        
        if success:
            successful += 1
        else:
            failed += 1
        
        logger.info("")  # Blank line between scripts
    
    # Summary
    logger.info("="*60)
    logger.info("📊 POPULATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total scripts: {len(POPULATION_SCRIPTS)}")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"⏭️  Skipped: {len(POPULATION_SCRIPTS) - successful - failed}")
    logger.info("")
    
    if failed > 0:
        logger.info("Failed scripts:")
        for script_name, success, output in results:
            if not success:
                logger.info(f"  - {script_name}")
    
    # Final course count
    try:
        from agents.course_knowledge_base import CourseKnowledgeBase
        kb = CourseKnowledgeBase()
        final_count = kb.get_course_count()
        logger.info(f"📚 Final course count in {current_db}: {final_count}")
    except Exception as e:
        logger.warning(f"Could not get final course count: {e}")
    
    logger.info("="*60)
    
    if failed == 0:
        logger.info("✅ All scripts completed successfully!")
        return 0
    else:
        logger.warning(f"⚠️  {failed} script(s) failed. Check logs above for details.")
        return 1

if __name__ == "__main__":
    exit(main())

