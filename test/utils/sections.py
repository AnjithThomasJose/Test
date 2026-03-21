import re

ANCHORS = [
    r"summary|profile|objective",
    r"skills?",
    r"experience|employment|work history",
    r"projects?|portfolio",
    r"education|academics|qualifications",
    r"certifications?|licenses?",
    r"awards?|honors?",
    r"publications?|research",
    r"languages?",
    r"contact|contact information|details",
    r"affiliations|memberships|associations",
]

HEADER_RE = re.compile(rf"^({'|'.join(ANCHORS)})\s*$", re.IGNORECASE)


def split_sections(txt: str):
    sections = {}
    current = "unknown"
    buff = []
    for ln in txt.splitlines():
        if HEADER_RE.match(ln.strip()):
            if buff:
                sections.setdefault(current, []).append("\n".join(buff).strip())
                buff = []
            current = ln.strip().lower()
        else:
            buff.append(ln)
    if buff:
        sections.setdefault(current, []).append("\n".join(buff).strip())
    return sections


