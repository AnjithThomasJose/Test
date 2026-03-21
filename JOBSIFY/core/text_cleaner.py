import re
from typing import Optional, List


def clean_full_description(text: str) -> str:
    """Clean extracted job description text from various document formats."""
    if not text:
        return ""
    
    # Step 1: Decode HTML entities first (before stripping tags)
    # import html
    # text = html.unescape(text)
    
    # # Step 2: Strip HTML tags if present
    # if '<' in text and '>' in text:
    #     # Remove HTML tags using regex
    #     text = re.sub(r'<[^>]+>', ' ', text)
    
    # Encode/decode to handle unicode issues
    text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    
    # Replace common unicode characters
    replacements = {
        '\u2013': '-', '\u2014': '-',
        '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"',
        '\u2026': '...', '\u2022': '-',
        '\\': '', '|': '', '/': ' '  # 🔥 FIX BUG 1: Remove extra slashes and pipes
    }
    
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    # 🔥 FIX BUG 1: Remove ALL control characters including \u0000, \n, \r, \t
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
    
    # 🔥 FIX BUG 1: Remove non-ASCII characters more aggressively
    text = re.sub(r'[^\x20-\x7E]+', ' ', text)
    
    # 🔥 FIX BUG 1: Remove any remaining unicode escape sequences like \u
    text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
    text = re.sub(r'\\x[0-9a-fA-F]{2}', '', text)
    
    # Clean formatting
    text = re.sub(r'\s+', ' ', text)  # Collapse all whitespace to single space
    text = re.sub(r'\s*\.\s*', '. ', text)  # Normalize periods
    text = re.sub(r'\s*,\s*', ', ', text)  # Normalize commas
    
    # Remove extra spaces around parentheses and brackets
    text = re.sub(r'\s*\(\s*', ' (', text)
    text = re.sub(r'\s*\)\s*', ') ', text)
    
    # Final cleanup
    text = ' '.join(text.split())  # Remove all extra spaces
    text = text.strip()
    
    return text


# ============================================================================
# FIELD-SPECIFIC CLEANERS (for extracted short fields, not full descriptions)
# ============================================================================

def normalize_whitespace(text: Optional[str]) -> Optional[str]:
    """
    Normalize whitespace by removing ALL newlines and collapsing spaces.
    For short extracted fields only (experience, salary, location, etc.)
    """
    if not text:
        return None
    
    # Replace all newlines, tabs, and carriage returns with space
    cleaned = re.sub(r'[\n\r\t]+', ' ', text)
    
    # Replace multiple spaces with single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()
    
    # Return None if empty after cleaning
    return cleaned if cleaned else None


def extract_numbers_only(text: str) -> str:
    """🔥 FIX BUG 3: Extract only numbers, ranges, and basic symbols."""
    if not text:
        return ""
    
    # Keep only digits, ranges, decimals, and plus signs
    cleaned = re.sub(r'[^\d\-+.,\s]', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


def clean_experience(experience: Optional[str]) -> str:
    """
    Clean experience field - preserve "years" text.
    Examples:
        "Minimum 35 years of experience..." -> "35 years"
        "0-5 years" -> "0-5 years"
        "3-5 years" -> "3-5 years"
        "15+ years" -> "15+ years"
        "Minimum 5 years" -> "5 years"
    """
    if not experience:
        return ""
    
    cleaned = normalize_whitespace(experience)
    if not cleaned:
        return ""
    
    # Extract number patterns: "35", "0-5", "3-5", "15+"
    number_part = ""
    
    # Try to find range (0-5 years, 3-5 years, 3 to 5 years)
    # Pattern matches: digit(s), optional whitespace, dash/en-dash/em-dash or "to", optional whitespace, digit(s)
    range_match = re.search(r'(\d+)\s*[-–—to]+\s*(\d+)', cleaned)
    if range_match:
        number_part = f"{range_match.group(1)}-{range_match.group(2)}"
    else:
        # Try to find number with plus (15+)
        plus_match = re.search(r'(\d+)\s*\+', cleaned)
        if plus_match:
            number_part = f"{plus_match.group(1)}+"
        else:
            # Check if "05" might be a misformatted "0-5"
            # Heuristic: If we see exactly 2 digits starting with 0 and second digit is 1-9,
            # and the original experience string contains "0" and "-" or "to", it's likely "0-5"
            two_digit_match = re.search(r'\b0(\d)\b', cleaned)
            if two_digit_match and int(two_digit_match.group(1)) <= 9:
                # Check if original experience (before normalize_whitespace) might have had a range indicator
                # Look for patterns like "0" followed by "-" or "to" in the original string
                if experience and (re.search(r'0\s*[-–—to]', experience, re.IGNORECASE) or 
                                   re.search(r'[-–—to]\s*5', experience, re.IGNORECASE)):
                    number_part = f"0-{two_digit_match.group(1)}"
                else:
                    # Just extract as single number (remove leading zero)
                    number_part = two_digit_match.group(1)
            else:
                # Just a single number
                single_match = re.search(r'(\d+)', cleaned)
                if single_match:
                    number_part = single_match.group(1)
    
    if not number_part:
        return ""
    
    # Check if "years" is mentioned
    if re.search(r'\b(years?|yrs?)\b', cleaned, re.IGNORECASE):
        return f"{number_part} years"
    
    return number_part


def clean_salary(salary: Optional[str]) -> str:
    """
    Clean salary field - preserve currency symbol and units.
    Examples:
        "₹1540 Lakhs per film..." -> "₹15-40 Lakhs per"
        "INR 50000-60000 per month" -> "INR 50000-60000 per month"
        "₹5 LPA" -> "₹5 LPA"
        "Rs. 15,00,000 - 20,00,000 per annum" -> "Rs. 15,00,000-20,00,000 per annum"
    """
    if not salary:
        return ""
    
    cleaned = normalize_whitespace(salary)
    if not cleaned:
        return ""
    
    # Extract currency symbol if present
    currency = ""
    currency_match = re.search(r'^(₹|INR|Rs\.?|USD|\$|EUR|€)\s*', cleaned)
    if currency_match:
        currency = currency_match.group(1) + " "
        cleaned = cleaned[len(currency_match.group(0)):]
    
    # Extract number range or single number
    range_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*[-–—to]+\s*(\d+(?:,\d+)*(?:\.\d+)?)', cleaned)
    if range_match:
        num1 = range_match.group(1).replace(',', '')
        num2 = range_match.group(2).replace(',', '')
        number_part = f"{num1}-{num2}"
    else:
        single_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)', cleaned)
        if single_match:
            number_part = single_match.group(1).replace(',', '')
        else:
            return ""
    
    # Extract unit/period text (Lakhs, LPA, per month, per annum, etc.)
    unit_text = ""
    unit_match = re.search(r'(Lakhs?|LPA|Crores?|K|per\s+(?:month|annum|year|hour|day|film|project))', cleaned, re.IGNORECASE)
    if unit_match:
        unit_text = " " + unit_match.group(1)
    
    # Combine all parts
    result = f"{currency}{number_part}{unit_text}".strip()
    
    return result


def clean_location(location: Optional[str]) -> str:
    """
    Clean location field.
    
    Examples:
        "Bangalore,\\n \\nIndia" -> "Bangalore, India"
        "Remote\\n/\\nHybrid" -> "Remote/Hybrid"
    """
    if not location:
        return ""
    
    cleaned = normalize_whitespace(location)
    if not cleaned:
        return ""
    
    # Normalize comma spacing
    cleaned = re.sub(r'\s*,\s*', ', ', cleaned)
    
    # Normalize slash spacing for work modes
    cleaned = re.sub(r'\s*/\s*', '/', cleaned)
    
    return cleaned.strip()


def clean_job_type(job_type: Optional[str]) -> str:
    """
    Clean job type field.
    
    Examples:
        "Full-time,\\nContract" -> "Full-time, Contract"
        "Part\\n-\\ntime" -> "Part-time"
    """
    if not job_type:
        return ""
    
    cleaned = normalize_whitespace(job_type)
    if not cleaned:
        return ""
    
    # Fix hyphenation (Full - time -> Full-time)
    cleaned = re.sub(r'(Full|Part)\s*-\s*time', r'\1-time', cleaned, flags=re.IGNORECASE)
    
    # Normalize comma spacing
    cleaned = re.sub(r'\s*,\s*', ', ', cleaned)
    
    return cleaned.strip()


def clean_job_title(job_title: Optional[str]) -> str:
    """
    Clean job title field.
    
    Examples:
        "Senior\\nSoftware\\nEngineer" -> "Senior Software Engineer"
        "Lead   Developer" -> "Lead Developer"
    """
    if not job_title:
        return ""
    
    cleaned = normalize_whitespace(job_title)
    return cleaned or ""


def clean_skills_list(skills: List[str]) -> List[str]:
    """
    Clean a list of skills by removing newlines and normalizing whitespace.
    
    Args:
        skills: List of skill strings that may contain newlines
        
    Returns:
        Cleaned list of skills with duplicates removed
    """
    cleaned_skills = []
    seen = set()
    
    for skill in skills:
        cleaned = normalize_whitespace(skill)
        if cleaned:
            # Avoid duplicates (case-insensitive)
            if cleaned.lower() not in seen:
                cleaned_skills.append(cleaned)
                seen.add(cleaned.lower())
    
    return cleaned_skills


def clean_jd_fields(jd_data: dict) -> dict:
    """
    Clean extracted field values in a job description data dictionary.
    This cleans SHORT FIELDS ONLY - NOT fullJobDescription.
    """
    cleaned = jd_data.copy()
    
    # Clean string fields with specific cleaners
    if "experience" in cleaned:
        cleaned["experience"] = clean_experience(cleaned["experience"])
    
    if "salary" in cleaned:
        cleaned["salary"] = clean_salary(cleaned["salary"])
    
    if "location" in cleaned:
        cleaned["location"] = clean_location(cleaned["location"])
    
    if "jobType" in cleaned:
        cleaned["jobType"] = clean_job_type(cleaned["jobType"])
    
    if "jobTitle" in cleaned:
        cleaned["jobTitle"] = clean_job_title(cleaned["jobTitle"])
    
    # Clean skills list
    if "requiredSkills" in cleaned and isinstance(cleaned["requiredSkills"], list):
        cleaned["requiredSkills"] = clean_skills_list(cleaned["requiredSkills"])
    
    # Clean company name (if present)
    if "company" in cleaned:
        cleaned["company"] = normalize_whitespace(cleaned["company"]) or ""
    
    # DO NOT clean fullJobDescription here - use clean_full_description() separately
    
    return cleaned


def clean_and_normalize_jd_text(raw_text: str) -> str:
    """Clean and normalize JD text before sending to Groq."""
    if not raw_text:
        return ""

    cleaned = raw_text

    # Remove page numbers and artifacts
    lines = cleaned.split('\n')
    filtered_lines = []
    for line in lines:
        line_stripped = line.strip()
        if re.match(r'^\d+$', line_stripped):
            continue
        if len(line_stripped) > 0:
            filtered_lines.append(line)
    cleaned = '\n'.join(filtered_lines)

    # Normalize whitespace
    cleaned = re.sub(r' +', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    lines = [line.rstrip() for line in cleaned.split('\n')]
    cleaned = '\n'.join(lines)

    # Remove special characters (keep basic punctuation)
    cleaned = re.sub(r'[^\w\s\-.,;:()\[\]{}\'"/@+&$%#!?=*<>]', '', cleaned)

    # Normalize section headers
    section_headers = {
        r'(?i)^\s*(job\s*description|position\s*summary)\s*:?\s*$': 'JOB DESCRIPTION:',
        r'(?i)^\s*(responsibilities|duties|key\s*responsibilities)\s*:?\s*$': 'RESPONSIBILITIES:',
        r'(?i)^\s*(requirements|qualifications|required\s*skills)\s*:?\s*$': 'REQUIREMENTS:',
        r'(?i)^\s*(preferred|nice\s*to\s*have)\s*:?\s*$': 'PREFERRED:',
        r'(?i)^\s*(benefits|perks|what\s*we\s*offer)\s*:?\s*$': 'BENEFITS:',
    }

    for pattern, replacement in section_headers.items():
        cleaned = re.sub(pattern, f'\n{replacement}\n', cleaned)

    # Normalize bullet points
    cleaned = re.sub(r'[•●○▪▫■□►→-]\s*', '- ', cleaned)
    cleaned = re.sub(r'-\s{2,}', '- ', cleaned)

    cleaned = cleaned.strip()
    log = logging.getLogger("main")
    log.debug(
        "🧹 Cleaned JD text: %d -> %d characters", len(raw_text), len(cleaned)
    )

    return cleaned

def clean_for_llm(raw_text: str) -> str:
    """
    🔥 NEW: Clean JD text for LLM - adapted from resume parser.
    Handles ANY text length by intelligently truncating to max chars.
    """
    try:
        from utils.normalize_text import normalize
        s = normalize(raw_text or "")
    except ImportError:
        # Fallback if normalize isn't available
        s = raw_text or ""
        s = s.strip()
    
    # Remove emojis and decorative Unicode characters
    s = re.sub(r'[\U0001F300-\U0001F9FF]', '', s)  # Emojis
    s = re.sub(r'[\U0001FA00-\U0001FAFF]', '', s)  # Extended emojis
    s = re.sub(r'[💸🚀💰🔥📊✅🌱🔒🛠🎯🎓🗣🔥⚙]', '', s)  # Common emojis
    
    # Remove null bytes and null-like tokens
    s = re.sub(r'[\u0000]', '', s)
    null_tokens = {"null", "n/a", "na", "none"}
    cleaned_lines: List[str] = []
    for line in s.splitlines():
        words = [w for w in line.split() if w.lower() not in null_tokens]
        cleaned_lines.append(" ".join(words))
    s = "\n".join(cleaned_lines).strip()
    
    # Strip bullet prefixes
    bullet_chars = ("- ", "• ", "▸ ", "► ", "· ")
    debulleted: List[str] = []
    for line in s.splitlines():
        stripped = line
        for b in bullet_chars:
            if stripped.startswith(b):
                stripped = stripped[len(b):]
                break
        debulleted.append(stripped)
    s = "\n".join(debulleted)

    # Deduplicate consecutive identical lines
    deduped: List[str] = []
    last_line = None
    for line in s.splitlines():
        if line != last_line:
            deduped.append(line)
        last_line = line
    s = "\n".join(deduped)

    # 🔥 ADAPTIVE: Handles ANY length by truncating to max chars
    if len(s) > JD_LLM_INPUT_MAX_CHARS:
        log = logging.getLogger("main")
        log.debug(
            "📏 JD text truncated: %d → %d chars", len(s), JD_LLM_INPUT_MAX_CHARS
        )
        s = s[:JD_LLM_INPUT_MAX_CHARS]
    
    return s

