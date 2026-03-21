# utils/experience_utils.py

from datetime import datetime
from utils.date_utils import parse_date
import logging

log = logging.getLogger(__name__)

def calculate_total_experience_from_models(work_experience):
    """
    Calculate total experience by merging overlapping ranges.
    Returns human-readable string (e.g., '5 years 3 months')
    """

    if not work_experience:
        return "0 months"

    ranges = []
    today = datetime.today()

    for idx, exp in enumerate(work_experience):

        # Support dict & model object
        if isinstance(exp, dict):
            dates = exp.get("dates", "")
        else:
            dates = getattr(exp, "dates", "")

        if not dates:
            continue

        original_dates = dates

        # =====================================================
        # ✅ FIXED: normalize ALL resume date formats
        # =====================================================
        dates = (
            dates.lower()
            .replace("–", "-")
            .replace("—", "-")
            .replace("→", "-")
            .replace(" to ", "-")
            .replace(" till ", "-")
            .replace("till date", "present")
            .replace("till", "present")
        ).strip()

        log.debug(f"Normalized dates: '{original_dates}' → '{dates}'")

        if "-" not in dates:
            continue

        start_str, end_str = [d.strip() for d in dates.split("-", 1)]

        start = parse_date(start_str)
        end = parse_date(end_str) or today

        if not start or end <= start:
            continue

        ranges.append((start, min(end, today)))

    if not ranges:
        return "0 months"

    # =====================================================
    # Merge overlapping date ranges
    # =====================================================
    ranges.sort(key=lambda x: x[0])
    merged = [list(ranges[0])]

    for start, end in ranges[1:]:
        if start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    total_months = sum(
        (e.year - s.year) * 12 + (e.month - s.month)
        for s, e in merged
    )

    years = total_months // 12
    months = total_months % 12

    if years and months:
        return f"{years} years {months} months"
    elif years:
        return f"{years} years"
    else:
        return f"{months} months"
