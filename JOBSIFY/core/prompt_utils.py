"""
Centralized prompt size limits and truncation (Section 8 Issue 3).
Use truncate_prompt() so behavior is consistent and JSON-safe when over limit.
"""
import logging
from typing import Literal

log = logging.getLogger(__name__)

Strategy = Literal["json_aware", "sentence"]


def truncate_prompt(
    text: str,
    max_chars: int,
    strategy: Strategy = "json_aware",
) -> str:
    """
    Truncate prompt text to max_chars with a safe strategy.
    - json_aware: preserve instructions, truncate payload at valid JSON boundary where possible.
    - sentence: truncate at last sentence or word boundary in the tail.
    """
    if not text or len(text) <= max_chars:
        return text or ""

    if strategy == "json_aware":
        # Try to keep instructions and truncate JSON payload
        json_marker = "**Full Resume Data:**"
        idx = text.find(json_marker)
        if idx >= 0:
            instructions = text[: idx + len(json_marker) + 1]
            payload = text[idx + len(json_marker) + 1 :]
            space_for_payload = max_chars - len(instructions)
            if space_for_payload > 1000:
                truncated = payload[:space_for_payload]
                for sep in ("}\n", "},", "]", "}"):
                    last = truncated.rfind(sep)
                    if last > space_for_payload * 0.7:
                        truncated = truncated[: last + len(sep)]
                        break
                else:
                    truncated = truncated[: space_for_payload - 10] + "\n}"
                return instructions + truncated
        # Fallback: truncate at last safe character
        return text[: max_chars - 20] + "\n...[truncated]"

    if strategy == "sentence":
        truncated = text[:max_chars]
        for sep in (". ", "\n", " "):
            last = truncated.rfind(sep)
            if last > max_chars * 0.8:
                return truncated[: last + len(sep)] + "...[truncated]"
        return truncated + "...[truncated]"

    return text[:max_chars]
