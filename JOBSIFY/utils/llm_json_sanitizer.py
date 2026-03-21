"""
LLM JSON Sanitizer - Robust JSON extraction from LLM text responses.

This module provides best-effort extraction of JSON objects/arrays from raw LLM
output, handling common formatting issues like trailing commas, extra text, and
markdown wrapping.
"""

import json
import re
from typing import Any, Optional


def extract_json(raw: str) -> Optional[Any]:
    """
    Best-effort extraction of the first valid JSON object or array from raw LLM text.
    
    Strategies:
    - Find first { ... } or [ ... ] substring with balanced braces using stack scan
    - If direct json.loads fails, try to fix common issues:
        - Remove leading/trailing garbage
        - Fix trailing commas
        - Handle markdown code blocks
    - Return parsed object/array or None
    
    Args:
        raw: Raw text string that may contain JSON
        
    Returns:
        Parsed JSON object/array or None if extraction fails
    """
    if not raw or not isinstance(raw, str):
        return None
    
    # Try direct JSON parse first (fast path)
    try:
        return json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Find JSON-like substrings using balanced brace matching
    candidates = []
    
    # Look for object { ... }
    for match in re.finditer(r'\{', raw):
        start = match.start()
        depth = 0
        in_string = False
        escape_next = False
        
        for i in range(start, len(raw)):
            char = raw[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        candidates.append((start, i + 1, raw[start:i+1]))
                        break
    
    # Look for array [ ... ]
    for match in re.finditer(r'\[', raw):
        start = match.start()
        depth = 0
        in_string = False
        escape_next = False
        
        for i in range(start, len(raw)):
            char = raw[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '[':
                    depth += 1
                elif char == ']':
                    depth -= 1
                    if depth == 0:
                        candidates.append((start, i + 1, raw[start:i+1]))
                        break
    
    # Try to parse each candidate
    for start, end, candidate in candidates:
        # Remove markdown code blocks if present
        candidate = re.sub(r'^```(?:json)?\s*\n?', '', candidate, flags=re.IGNORECASE)
        candidate = re.sub(r'\n?```\s*$', '', candidate)
        candidate = candidate.strip()
        
        # Try direct parse
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Try fixing trailing commas
        fixed = re.sub(r',(\s*[}\]])', r'\1', candidate)
        if fixed != candidate:
            try:
                return json.loads(fixed)
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Try removing comments (JSON doesn't support comments, but LLMs sometimes add them)
        fixed = re.sub(r'//.*?$', '', fixed, flags=re.MULTILINE)
        fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
        if fixed != candidate:
            try:
                return json.loads(fixed.strip())
            except (json.JSONDecodeError, ValueError):
                pass
    
    # Last resort: try to extract from triple-quoted strings
    triple_quote_match = re.search(r'"""(.*?)"""', raw, re.DOTALL)
    if triple_quote_match:
        try:
            return json.loads(triple_quote_match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass
    
    return None

