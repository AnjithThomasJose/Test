from __future__ import annotations

import re
from typing import Dict, Any


PII_REGEXES = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b(?:\d[ -]*?){13,19}\b"),  # Credit card
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),  # Email
    re.compile(r"\b(?:\+?\d[\d(). -]{8,}\d)\b"),  # Phone
]


def evaluate_agent_output_for_pii(state: Dict[str, Any]) -> None:
    """Raise a warning via annotation if potential PII is detected in agent outputs."""
    if state is None:
        return
        
    suspect_fields = []
    for key, value in state.items():
        if isinstance(value, str):
            for rx in PII_REGEXES:
                if rx.search(value):
                    suspect_fields.append(key)
                    break
    if suspect_fields:
        state["pii_suspect_fields"] = suspect_fields


