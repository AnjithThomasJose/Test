import json
import logging
from typing import Dict, Any, Optional

log = logging.getLogger("main")
def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from a text string that may contain markdown or other formatting.
    """
    # Try to find JSON in markdown code blocks
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0]
    elif '```' in text:
        text = text.split('```')[1].split('```')[0]
    
    # Find the first { and last } to extract the JSON
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in text")
            
        json_str = text[start:end]
        return json.loads(json_str)
        
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse JSON: {e}")
        log.debug(f"Problematic text: {text}")
        raise