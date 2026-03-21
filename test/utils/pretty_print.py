import json
import logging
from rich import print as rprint  # pip install rich

log = logging.getLogger("main")

def pretty_print_json(data, title=None):
    """Pretty print JSON or string in terminal."""
    if title:
        log.info(f"=== {title} ===")

    try:
        if isinstance(data, str):
            parsed = json.loads(data)
            rprint(parsed)
        else:
            # Already a dict or AIMessage.content is dict-like
            rprint(data)
    except Exception:
        # Fallback: print raw if JSON parse fails
        rprint(data)
