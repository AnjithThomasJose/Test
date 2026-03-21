# import requests
# from google.cloud import firestore
# from google.oauth2 import service_account
# from settings import settings

# # Load service account JSON from URL
# # url = "https://firebasestorage.googleapis.com/v0/b/jobsify-5d910.firebasestorage.app/o/serviceAccountKey.json?alt=media&token=bf19ea03-8b09-431f-a992-a0e082144be7"
# # url = "https://firebasestorage.googleapis.com/v0/b/qa-jobsify.firebasestorage.app/o/serviceAccountKey.json?alt=media&token=820b7a41-a32f-4d10-a8bf-ce3564f33015"
# url = settings.DB_URL
# response = requests.get(url)
# service_account_info = response.json()

# # Create credentials
# credentials = service_account.Credentials.from_service_account_info(service_account_info)

# # Firestore client with credentials
# # Connect to Firestore `mem-transaction` database
# db = firestore.Client(
#     project=service_account_info["project_id"],
#     credentials=credentials,
#     database="mem-transaction"
# )





import requests
import logging
from google.cloud import firestore
from google.oauth2 import service_account
from settings import settings
from typing import Dict, Any, List, Optional

# Initialize logger
log = logging.getLogger(__name__)

# Determine database name based on environment
# Default to "(default)" for dev, "mem-transaction" for prod
def get_firestore_database_name() -> str:
    """Get Firestore database name based on environment"""
    env = settings.APP_ENV.lower()
    if env in ["development", "dev"]:
        # Use default database for dev (or try mem-transaction if it exists)
        return "(default)"  # Default Firestore database
    elif env == "production" or env == "prod":
        return "mem-transaction"
    else:
        # For QA or other environments, try mem-transaction first
        return "mem-transaction"

# Initialize Firestore client with error handling (Section 4 Issue 1: timeout to avoid startup hang)
db = None
try:
    import os as _os
    _timeout = int(_os.getenv("FIREBASE_INIT_TIMEOUT_SECONDS", "10"))
    url = settings.DB_URL
    response = requests.get(url, timeout=_timeout)
    service_account_info = response.json()
    
    # Create credentials
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    
    # Get database name based on environment
    database_name = get_firestore_database_name()
    
    # Firestore client with credentials
    db = firestore.Client(
        project=service_account_info["project_id"],
        credentials=credentials,
        database=database_name
    )
    log.info(f"✅ Firestore client initialized for database: {database_name} (env: {settings.APP_ENV})")
except Exception as e:
    log.error(f"❌ Failed to initialize Firestore client: {e}")
    log.warning("⚠️  Firebase operations will be skipped. User interests will only be stored in ChromaDB.")
    db = None

def fetch_structured_resume(uid: str) -> Optional[Dict[str, Any]]:
    """
    Fetches the structured_resume from the 'users' collection in Firestore.
    """
    if db is None:
        log.warning("Firestore client not initialized, skipping fetch_structured_resume")
        return None
    try:
        doc_ref = db.collection("users").document(uid)
        doc = doc_ref.get()
        if doc.exists:
            user_data = doc.to_dict()
            # The structured resume is stored under the 'resume_parser' key
            return user_data.get("resume_parser")
    except Exception as e:
        log.error(f"Error fetching structured_resume for UID {uid}: {e}")
        return None

def save_personal_insights(uid: str, answers: List[str]):
    """
    Saves the 10 personal insight answers to the 'personal_insight' collection.
    Gracefully handles missing database or connection issues.
    """
    if db is None:
        log.warning(f"Firestore client not initialized, skipping save_personal_insights for UID {uid}")
        log.info(f"💡 User interests are still stored in ChromaDB (structured_resume.user_interests_summary)")
        return
    
    try:
        doc_ref = db.collection("personal_insight").document(uid)
        # The document will have a single field 'answers' which is an array of strings
        doc_ref.set({"answers": answers})
        log.info(f"✅ Personal insights saved to Firebase for UID {uid}")
    except Exception as e:
        error_msg = str(e)
        # Check if it's a database not found error
        if "does not exist" in error_msg or "404" in error_msg:
            log.warning(f"⚠️  Firebase database not found for UID {uid}. This is expected in dev if database doesn't exist.")
            log.info(f"💡 User interests are still stored in ChromaDB (structured_resume.user_interests_summary)")
        else:
            log.error(f"Error saving personal insights to Firebase for UID {uid}: {e}")
            log.info(f"💡 User interests are still stored in ChromaDB (structured_resume.user_interests_summary)")