from __future__ import annotations

import json
import re
from typing import Union


JSON_MIME_RX = re.compile(r"^application/(?:json|.*\+json)(?:;|$)", re.IGNORECASE)


def is_valid_content_type(content_type: str) -> bool:
    return bool(JSON_MIME_RX.search(content_type or ""))


def is_valid_json_size(body: Union[bytes, bytearray], max_bytes: int) -> bool:
    if body is None:
        return True
    return len(body) <= max_bytes


