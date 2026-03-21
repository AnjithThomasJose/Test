import re
import datetime
from typing import Any, Dict, List, Optional, Tuple


MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _parse_month_year(token: str) -> Optional[Tuple[int, int]]:
    s = token.strip().lower()
    # Formats:
    # - MMM YYYY (Jan 2020)
    # - MMM-YYYY (Jan-2020)
    # - MMM.YYYY (Jan.2020)
    # - YYYY-MM (2020-01)
    # - MM/YYYY (01/2020)
    # - YYYY (2020)
    # - Present/Now/Current
    if not s:
        return None
    if s in {"present", "now", "current"}:
        today = datetime.date.today()
        return today.month, today.year

    # 01/2020 or 1/2020
    m = re.match(r"^(\d{1,2})\s*[\-/]\s*(\d{4})$", s)
    if m:
        month = int(m.group(1))
        year = int(m.group(2))
        if 1 <= month <= 12:
            return month, year

    # 2020-01 or 2020/01
    m = re.match(r"^(\d{4})\s*[\-/]\s*(\d{1,2})$", s)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        if 1 <= month <= 12:
            return month, year

    # Jan 2020, January 2020, Jan-2020, Jan.2020
    m = re.match(r"^([A-Za-z]{3,9})[\s\-.]+(\d{4})$", s)
    if m:
        mon = MONTHS.get(m.group(1).lower())
        year = int(m.group(2))
        if mon:
            return mon, year

    # 2020 only -> assume mid-year for conservative calc
    m = re.match(r"^(\d{4})$", s)
    if m:
        year = int(m.group(1))
        return 7, year  # July as midpoint

    return None


def _parse_date_range(text: str) -> Optional[Tuple[datetime.date, datetime.date]]:
    if not text:
        return None
    s = text.replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("to", "-").replace("→", "-")
    s = re.sub(r"\s+", " ", s).strip()

    # Common patterns split by a dash
    parts = [p.strip() for p in re.split(r"\s*-\s*", s) if p.strip()]
    if len(parts) == 2:
        start_my = _parse_month_year(parts[0])
        end_my = _parse_month_year(parts[1])
        if start_my and end_my:
            sm, sy = start_my
            em, ey = end_my
            start = datetime.date(sy, sm, 1)
            # end -> last day of month
            if em == 12:
                end = datetime.date(ey, 12, 31)
            else:
                end = datetime.date(ey, em + 1, 1) - datetime.timedelta(days=1)
            if end >= start:
                return start, end
    return None


def _intervals_union(intervals: List[Tuple[datetime.date, datetime.date]]) -> List[Tuple[datetime.date, datetime.date]]:
    if not intervals:
        return []
    intervals_sorted = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[datetime.date, datetime.date]] = []
    cur_start, cur_end = intervals_sorted[0]
    for s, e in intervals_sorted[1:]:
        if s <= (cur_end + datetime.timedelta(days=1)):
            if e > cur_end:
                cur_end = e
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))
    return merged


def _extract_work_intervals(work_experience: List[Dict[str, Any]]) -> List[Tuple[datetime.date, datetime.date]]:
    intervals: List[Tuple[datetime.date, datetime.date]] = []
    today = datetime.date.today()

    for item in work_experience or []:
        # Prefer explicit start/end if present
        start_token = None
        end_token = None

        for key in ("start_date", "start", "from"):
            if isinstance(item.get(key), str) and item.get(key).strip():
                start_token = item.get(key).strip()
                break
        for key in ("end_date", "end", "to"):
            if isinstance(item.get(key), str) and item.get(key).strip():
                end_token = item.get(key).strip()
                break

        if start_token and end_token:
            start_my = _parse_month_year(start_token)
            end_my = _parse_month_year(end_token)
            if start_my and end_my:
                sm, sy = start_my
                em, ey = end_my
                start = datetime.date(sy, sm, 1)
                if em == 12:
                    end = datetime.date(ey, 12, 31)
                else:
                    end = datetime.date(ey, em + 1, 1) - datetime.timedelta(days=1)
                if end >= start:
                    intervals.append((start, end))
                continue

        # Fallback to parsing combined dates string
        dates_field = None
        for key in ("dates", "duration", "years"):
            if isinstance(item.get(key), str) and item.get(key).strip():
                dates_field = item.get(key).strip()
                break

        rng = _parse_date_range(dates_field) if dates_field else None
        if rng:
            intervals.append(rng)
            continue

        # If still nothing, but we have a single year, assume a 6-month stint
        single_year = None
        for key in ("dates", "duration", "years"):
            val = item.get(key)
            if isinstance(val, str):
                m = re.search(r"(\d{4})", val)
                if m:
                    single_year = int(m.group(1))
                    break
        if single_year:
            start = datetime.date(single_year, 1, 1)
            end = min(datetime.date(single_year, 12, 31), today)
            intervals.append((start, end))

    return intervals


def _years_between(start: datetime.date, end: datetime.date) -> float:
    days = (end - start).days + 1
    return round(days / 365.0, 1)


def _calculate_experience_years(work_experience: List[Dict[str, Any]]) -> float:
    intervals = _extract_work_intervals(work_experience)
    if not intervals:
        return 0.0
    merged = _intervals_union(intervals)
    total_years = 0.0
    for s, e in merged:
        total_years += _years_between(s, e)
    # Round to 1 decimal as convention
    return round(total_years, 1)


def calculate_total_experience_years(work_experience: List[Dict[str, Any]]) -> float:
    """
    Public API for computing total years of experience from a resume's work history.

    Accepts list of dicts with fields like:
    - dates: "Jan 2020 - Present", "2021 - 2023", etc.
    - start_date/end_date or start/end or from/to as strings

    Returns float years rounded to 1 decimal.
    """
    return _calculate_experience_years(work_experience)


