import os
import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

def store_callback(agent_name: str, uid: str, data: dict) -> str:
    """
    Store callback data as a JSON file in local_callbacks/.
    Returns the file path. Gracefully handles permission errors.
    """
    try:
        base_dir = Path("local_callbacks")
        # Try to create directory with proper permissions
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            log.warning(f"⚠️  Permission denied creating local_callbacks directory: {e}")
            log.info("💡 Callback data will not be stored locally, but will still be sent to callback URL")
            raise
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{agent_name}_{uid}_{timestamp}.json"
        filepath = base_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    except PermissionError as e:
        log.error(f"❌ Permission denied writing callback file: {e}")
        log.info("💡 Callback data will not be stored locally, but will still be sent to callback URL")
        # Return empty string to indicate failure, but don't raise to avoid breaking the flow
        return ""
    except Exception as e:
        log.error(f"❌ Error storing callback file: {e}")
        log.info("💡 Callback data will not be stored locally, but will still be sent to callback URL")
        # Return empty string to indicate failure, but don't raise to avoid breaking the flow
        return ""
