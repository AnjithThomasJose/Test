from typing import Optional, Dict
from string import Template

def generate_comprehensive_resume_prompt(resume_text: str, segmented_sections: Optional[Dict[str, str]] = None) -> str:
    tmpl = Template("""RESUME TEXT (raw, unmodified):
${resume_text}

TASK:
Extract information from the resume text above and return it as a single JSON object that matches the schema below.
Parse the entire text (no pre-segmentation is provided).
Identify sections semantically (e.g., headings like "Experience", "Education", "Projects", "Skills", "Certifications", etc. — including variants: "Professional Experience", "Work History", "Academic Background", "Technical Skills", "Licenses", "Portfolio", etc.).
Handle any layout (single/multi-column, tables, bullets, decorative icons, headers/footers, page numbers).
Copy text exactly where asked (don’t paraphrase).
If something is missing, use null for optional scalars, [] for arrays, "" for empty strings.

CRITICAL RULES:
1) JSON ONLY: Start with { and end with }. No extra text or code fences.
2) No hallucinations: Extract ONLY what’s explicitly present. Do not invent emails, dates, companies, degrees, skills, or metrics.
3) Keep original formatting where applicable (e.g., date ranges "Sep 2020 – May 2024"; degree names; company names; job titles).
4) Normalization:
   - Deduplicate repeated items across pages/headers.
   - Strip decorative symbols (•, ▸, ►) from bullet text.
   - Split comma/semicolon/pipe-separated skill lists into individual items.
5) Contact details validation:
   - Email must contain "@".
   - LinkedIn/GitHub must be valid URLs.
5) Experience computation:
   - total_experience_years: sum non-overlapping employment ranges.
   - Round to 1 decimal place.
6) Skills extraction policy:
   - Extract skills ONLY from explicit Skills-like sections.
   - Do NOT infer skills from narrative text.
""")
    return tmpl.substitute(resume_text=resume_text)

def generate_simplified_resume_prompt(resume_text: str, segmented_sections: Optional[Dict[str, str]] = None) -> str:
    tmpl = Template("""You are a JSON API. Return ONLY valid JSON. Your response must start with { and end with }. No other text.

RESUME TEXT:
${resume_text}

Extract information from the resume above and return it as a JSON object with this exact structure:

{
  "name": "extract full name",
  "contact_details": {
    "Email": "extract email or null",
    "Phone": "extract phone or null",
    "Location": "extract location or null",
    "LinkedIn": "extract LinkedIn URL or null",
    "GitHub": "extract GitHub URL or null",
    "Website": "extract website URL or null"
  },
  "education": [
    {
      "degree": "extract degree",
      "major": "extract major",
      "university": "extract university",
      "location": "extract location",
      "years": "extract years",
      "gpa": "extract gpa or empty string",
      "capstone": {}
    }
  ],
  "work_experience": [
    {
      "job_title": "extract job title",
      "company": "extract company",
      "dates": "extract dates",
      "location": "extract location",
      "responsibilities": ["extract responsibility 1", "extract responsibility 2"]
    }
  ],
  "skills": [
    {
      "SkillName": "extract skill name",
      "Proficiency": "",
      "Rationale": ""
    }
  ],
  "certifications": [
    {
      "certification_name": "extract certification name",
      "issuing_organization": "extract organization",
      "year": "extract year or null"
    }
  ],
  "projects": [
    {
      "project_name": "extract project name",
      "description": "extract description",
      "technologies": ["extract technology"],
      "duration": "extract duration",
      "achievements": ["extract achievement"]
    }
  ],
  "extras": [],
  "total_experience_years": 0.0
}

IMPORTANT: Replace the example text above with actual data from the resume. Use null for missing optional fields, [] for empty arrays, "" for empty strings. Return ONLY the JSON object starting with { and ending with }.
""")
    return tmpl.substitute(resume_text=resume_text)

def generate_basic_resume_prompt(resume_text: str, segmented_sections: Optional[Dict[str, str]] = None) -> str:
    tmpl = Template("""Extract resume data to JSON.

CRITICAL: Extract ONLY what is written. DO NOT invent data. Use null/[]/"" for missing fields.

RESUME:
${resume_text}

Extract: name, contact_details, education, work_experience, skills (from skills sections), certifications, projects, total_experience_years.

Return valid JSON only. Copy text exactly as written. Use null/[] for missing data.
""")
    return tmpl.substitute(resume_text=resume_text)