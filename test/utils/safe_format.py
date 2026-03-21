"""
Small utility to safely .format() templates that may contain literal braces.

It escapes unmatched braces and replaces missing placeholders with empty strings.
"""

import re
import logging
from typing import Dict, Any

log = logging.getLogger(__name__)


def safe_format(template: str, placeholders: Dict[str, Any]) -> str:
    """
    Safely format a template string, handling missing placeholders and unmatched braces.
    
    Args:
        template: Template string that may contain {placeholders}
        placeholders: Dictionary of placeholder values
        
    Returns:
        Formatted string with missing placeholders replaced by empty strings
    """
    if not template:
        return template
    
    # First, if no braces present, fast-return
    if "{" not in template and "}" not in template:
        return template
    
    # Detect format specifiers in template (e.g., {key:.2f}, {key:d}, {key:>10})
    import re
    format_spec_pattern = r"\{([^}:]+)(:[^}]+)?\}"
    format_specifiers = {}
    for match in re.finditer(format_spec_pattern, template):
        key = match.group(1)
        spec = match.group(2)  # e.g., ":.2f", ":d", etc.
        if spec and spec != ":s":  # If there's a format specifier (and it's not string)
            format_specifiers[key] = spec
    
    # Build safe mapping: preserve types for format specifiers, convert others to strings
    safe = {}
    for k, v in (placeholders or {}).items():
        if v is None:
            safe[k] = ""
        elif k in format_specifiers:
            # Preserve original type for format specifiers (e.g., float for .2f, int for :d)
            safe[k] = v
        else:
            # Convert to string for simple placeholders
            safe[k] = str(v)
    
    # Replace unknown fields with empty string to avoid KeyError
    try:
        return template.format(**safe)
    except KeyError as e:
        missing = str(e).strip("'\"")
        log.debug("safe_format: missing placeholder %s, replacing with empty string", missing)
        # Replace unknown placeholder occurrences with empty braces then format
        # naive but effective: remove the token occurrences
        try:
            pattern = r"\{[^}]*" + re.escape(missing) + r"[^}]*\}"
            template2 = re.sub(pattern, "", template)
            return template2.format(**safe)
        except Exception:
            # Last resort: escape all braces and format only known placeholders
            try:
                esc = template.replace("{", "{{").replace("}", "}}")
                for k, v in safe.items():
                    esc = esc.replace("{{" + k + "}}", str(v))
                return esc
            except Exception:
                log.exception("safe_format fallback failed")
                return template
    except Exception:
        # On any other formatting error, return original template
        log.exception("safe_format unexpected formatting error")
        return template

