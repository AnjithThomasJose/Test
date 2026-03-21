import json

DOMAIN_TAXONOMY_LIST = [
    "software_engineering", "data_analytics", "data_science", "ai_ml",
    "technical_infrastructure", "traditional_engineering", "business_strategic",
    "sales_business", "marketing", "hr", "finance", "legal", "management",
    "healthcare", "ui_ux_design", "creative_design", "manufacturing",
    "supply_chain", "operations", "customer_support", "customer_success",
    "admin", "education", "science", "construction", "consulting",
    "hospitality", "retail", "trades", "real_estate", "fitness", "beauty",
    "aviation", "transportation", "military", "nonprofit", "public_sector",
    "arts", "sports", "environmental", "security", "cleaning_janitorial",
    "personal_care", "childcare", "other"
]

CANONICAL_SCHEMA = {
    "jobTitle": "",
    "company": "",
    "location": "",
    "jobType": "",
    "experience": "",
    "educationRequired": "",
    "salary": "",
    "requiredSkills": [],
    "fullJobDescription": ""
}

def generate_comprehensive_jd_prompt(jd_text: str) -> str:
    """Generate comprehensive prompt for JD parsing with individual skill extraction and domain classification."""
    domain_list = ", ".join(DOMAIN_TAXONOMY_LIST)
    return f"""Extract structured information from this job description and return as JSON.

**JOB DESCRIPTION TEXT:**
---
{jd_text}
---

Extract the following information EXACTLY as specified:

{{
  "jobTitle": "Job title or role name (e.g., Software Engineer, Data Analyst, Frontend Developer)",
  "company": "Company name",
  "location": "Location",
  "jobType": "Full-Time/Part-Time/Contract/etc.",
  "experience": "Experience with number and unit (e.g., '0-5 years' or '3-5 years' or '35 years' or '15+ years')",
  "educationRequired": "Education requirement (e.g., 'B.Com', 'BSc', 'M.Com', 'Bachelor of Commerce', 'Master of Science')",
  "salary": "Salary with currency and unit (e.g., '₹15-40 Lakhs per' or 'INR 50000-60000 per month')",
  "requiredSkills": ["JavaScript", "React", "Python", "Communication Skills"],
  "jobDomains": ["primary_domain", "secondary_domain", "tertiary_domain"],
  "fullJobDescription": "Complete original text",
  "fileUrl": ""
}}

CRITICAL RULES:

1. **jobTitle**: Extract the job title or role name as stated in the JD (e.g., "Software Engineer", "Data Analyst", "Frontend Developer"). Use "" only if not mentioned.
2. **company, location**: Extract EXACTLY as written

2. **experience**: Include number AND "years" 
   - Examples: "0-5 years", "3-5 years", "35 years", "15+ years"
   - IMPORTANT: Preserve ranges exactly as written (e.g., "0-5 years" NOT "05 years")

3. **salary**: Include currency, number/range, and unit
   - Examples: "₹15-40 Lakhs per", "INR 50000-60000 per month", "$80000-100000 per year"

4. **educationRequired**: Extract education degree/qualification requirements
   - Examples: "B.Com", "BSc", "M.Com", "Bachelor of Commerce", "Master of Science", "MBA"
   - Include degree level (Bachelor's, Master's, PhD) and field/major if specified
   - If no education requirement is mentioned, use empty string ""

5. **requiredSkills**: Extract ONLY individual skill names as separate array items, NEVER as one long sentence or phrase
   - ⚠️ DO NOT include education degrees/majors in requiredSkills (e.g., "B.Com", "M.Com", "BSc")
   - ⚠️ Education should go in "educationRequired" field, NOT in requiredSkills
   - ⚠️ NEVER put multiple skills in one string. Each skill MUST be its own array element.
   - CORRECT: ["JavaScript", "React", "SDLC", "STLC", "Test Case Design", "Functional Testing", "Regression Testing", "Jira", "SQL", "Agile methodology"]
   - WRONG: ["Good knowledge of SDLC and STLC Experience in writing test cases Knowledge of Jira"] → this is a sentence, not a list
   - WRONG: ["B.Com", "JavaScript", "React"] → "B.Com" should be in educationRequired, not requiredSkills
   - Extract: programming languages, frameworks, tools, databases, methodologies, testing types, soft skills — each as a separate short string (1-4 words)
   - If you see "Experience with X, Y, and Z" → extract ["X", "Y", "Z"] (three items)
   - If you see "Good knowledge of A. Experience in B. Familiar with C" → extract ["A", "B", "C"] (three items)
   - Soft skills: one per item, e.g. "Communication Skills", "Problem-solving", "Leadership"

6. **fullJobDescription**: Include the complete original text

7. **jobDomains**: Classify this job into its top 3 professional domains from this list:
   {domain_list}
   - Pick the 3 most relevant domains based on job title, required skills, and responsibilities
   - Order from most relevant to least relevant
   - Use ONLY values from the list above
   - Example: A "Full Stack AI Developer" role → ["ai_ml", "software_engineering", "data_science"]
   - Example: A "Sales Manager" role → ["sales_business", "management", "marketing"]
   - If fewer than 3 domains apply, include "other" to fill remaining slots

8. Use empty strings "" for missing text fields, empty arrays [] for missing lists

EXTRACTION EXAMPLES:
- "Proficiency in React.js, Redux, and modern JavaScript" → ["React.js", "Redux", "JavaScript"]
- "Experience with Git and version control" → ["Git", "Version Control"]
- "Strong Python and Django skills" → ["Python", "Django"]
- "Good communication and teamwork" → ["Communication Skills", "Teamwork"]
- "Knowledge of AWS, Docker, Kubernetes" → ["AWS", "Docker", "Kubernetes"]
- "Requires B.Com or equivalent" → educationRequired: "B.Com", requiredSkills: [] (NOT ["B.Com"])
- "Bachelor's in Computer Science required" → educationRequired: "Bachelor of Science in Computer Science", requiredSkills: [] (NOT ["Computer Science"])
- "B.Com/M.Com with knowledge of Python" → educationRequired: "B.Com or M.Com", requiredSkills: ["Python"]"""


