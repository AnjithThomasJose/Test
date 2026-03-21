# utils/date_utils.py

from datetime import datetime
import re
import logging

log = logging.getLogger(__name__)

MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

def parse_date(date_str: str):
    """
    Parse various date formats into datetime objects.
    Returns None if parsing fails.
    """
    if not date_str:
        return None

    original = date_str

    date_str = (
        date_str.strip()
        .lower()
        .replace(",", "")
        .replace("-", " ")
        .replace("/", " ")
        .replace(".", " ")
    )

    # =====================================================
    # ✅ NEED TO ADD: handle present / till date keywords
    # =====================================================
    if date_str in {"present", "current", "now", "till date", "till"}:
        result = datetime.today()
        log.debug(f"parse_date('{original}') -> {result} [present]")
        return result

    # =====================================================
    # EXISTING: Month Year (Jan 2020)
    # =====================================================
    match = re.match(r"^([a-z]+)\s+(\d{4})$", date_str)
    if match:
        month = MONTH_MAP.get(match.group(1))
        year = int(match.group(2))
        if month:
            return datetime(year, month, 1)

    # =====================================================
    # EXISTING: MonthYear (Jan2020)
    # =====================================================
    match = re.match(r"^([a-z]+)(\d{4})$", date_str)
    if match:
        month = MONTH_MAP.get(match.group(1))
        year = int(match.group(2))
        if month:
            return datetime(year, month, 1)

    # =====================================================
    # EXISTING: Year only (2020)
    # =====================================================
    match = re.match(r"^(\d{4})$", date_str)
    if match:
        return datetime(int(match.group(1)), 1, 1)

    # =====================================================
    # ✅ NEED TO ADD: MM YYYY or MM/YYYY
    # Example: 03/2025 → 03 2025
    # =====================================================
    match = re.match(r"^(\d{1,2})\s+(\d{4})$", date_str)
    if match:
        month, year = int(match.group(1)), int(match.group(2))
        return datetime(year, month, 1)

    # =====================================================
    # ✅ NEED TO ADD: DD MM YYYY
    # Example: 30.06.2021 → 30 06 2021
    # =====================================================
    match = re.match(r"^(\d{1,2})\s+(\d{1,2})\s+(\d{4})$", date_str)
    if match:
        day, month, year = map(int, match.groups())
        return datetime(year, month, day)

    log.warning(
        f"parse_date('{original}') -> None [unmatched format: '{date_str}']"
    )
    return None
