"""
LLM-Augmented Skill Framework

Hybrid approach combining deterministic base with LLM expansion:
- Static INDUSTRY_SKILLS as deterministic foundation
- LLM-powered expansion for long-tail industries/roles
- Caching and validation for consistent results
- Backward compatibility for existing agents

Pipeline: static → cache → store → LLM → validate → store

This design reduces schema sprawl while adding coverage "from students to presidents to scientists".
"""

from typing import Dict, Set, List, Optional, Callable, Protocol, TypedDict
import re
import json
import os
from functools import lru_cache

# Universal contextual skill frameworks
CONTEXT_SKILLS = {
    "career_switch": {
        # Core transferable skills
        "communication", "project management", "problem solving", "analytical thinking",
        "leadership", "teamwork", "time management", "adaptability",
        # Basic digital literacy (universal now)
        "microsoft office", "data analysis", "presentation skills", "email management",
        # Common professional tools
        "crm software", "collaboration tools", "basic accounting", "customer service"
    },
    "skill_upgrade": {
        # Professional development essentials
        "digital marketing", "social media", "data visualization", "process improvement",
        "quality management", "risk management", "compliance", "automation tools",
        "advanced excel", "reporting", "stakeholder management", "vendor management",
        # Industry 4.0 basics
        "ai awareness", "cloud basics", "cybersecurity awareness", "remote work tools"
    },
    "entry_level": {
        # Foundation skills for any industry
        "professional communication", "time management", "basic computer skills",
        "research skills", "critical thinking", "attention to detail", "customer focus",
        "initiative", "reliability", "learning agility", "basic math", "writing skills",
        # Modern workplace essentials
        "video conferencing", "online collaboration", "basic data entry", "file management"
    },
    "senior_advancement": {
        # Leadership and strategic skills
        "strategic planning", "team leadership", "budget management", "performance management",
        "change management", "stakeholder engagement", "decision making", "mentoring",
        "conflict resolution", "cross-functional collaboration", "business acumen",
        # Advanced professional skills
        "executive communication", "board reporting", "market analysis", "competitive intelligence"
    },
    "industry_transition": {
        # Transition facilitation skills
        "industry research", "networking", "personal branding", "interview skills",
        "salary negotiation", "regulatory awareness", "compliance understanding",
        "cultural adaptation", "market trends", "competitive landscape",
        # Transferable skill identification
        "skill mapping", "experience translation", "value proposition", "portfolio development"
    }
}

# Comprehensive industry-specific skills (38+ industries)
INDUSTRY_SKILLS = {
    # Technology & IT - Software Development
    "technology": {
        "keywords": ["software", "programming", "developer", "tech", "api", "database", "cloud", "agile", "coding", "it support", "web development", "mobile development"],
        "skills": {"python", "sql", "git", "docker", "cloud", "systems design", "agile", "apis", "cybersecurity", "devops"},
        "tools": ["python", "javascript", "react", "docker", "kubernetes", "aws", "git", "vscode", "jenkins"],
        "certifications": ["aws certified", "google cloud", "microsoft azure", "cissp", "pmp", "scrum master"]
    },
    
    # Electronics & Computer Engineering
    "electronics_engineering": {
        "keywords": ["ece", "electronics", "embedded", "microcontroller", "circuit design", "pcb", "analog", "digital", "vlsi", "microprocessor", "electronics engineer", "ece student"],
        "skills": {"circuit design", "pcb design", "microcontroller programming", "embedded systems", "analog electronics", "digital electronics", "signal processing", "control systems", "vlsi design", "fpga programming", "power electronics", "communication systems"},
        "tools": ["matlab", "simulink", "pspice", "ltspice", "altium designer", "eagle pcb", "keil", "arduino ide", "vivado", "quartus", "cadence", "proteus"],
        "certifications": ["embedded systems certification", "vlsi certification", "pcb design certification", "ieee membership", "engineering license"]
    },
    
    # Electrical Engineering
    "electrical_engineering": {
        "keywords": ["electrical engineer", "power systems", "electrical design", "motor control", "power electronics", "electrical maintenance", "substation", "transmission"],
        "skills": {"power systems", "electrical machines", "motor control", "power electronics", "electrical design", "electrical safety", "plc programming", "scada systems", "electrical codes", "protection systems"},
        "tools": ["etap", "pscad", "matlab", "autocad electrical", "eplan", "schneider electric", "siemens", "abb drives", "rockwell automation"],
        "certifications": ["pe license", "electrical license", "ieee membership", "nfpa certification", "arc flash training"]
    },
    
    # Healthcare & Medical
    "healthcare": {
        "keywords": ["medical", "hospital", "patient", "clinical", "healthcare", "nurse", "doctor", "therapy", "medical assistant", "radiology"],
        "skills": {"hipaa compliance", "medical terminology", "patient care", "clinical documentation", "ehr systems", "cpr certification", "medical ethics", "infection control"},
        "tools": ["ehr systems", "medical devices", "diagnostic equipment", "electronic health records", "patient monitors"],
        "certifications": ["medical license", "nursing license", "board certification", "cme credits", "bls", "acls"]
    },
    
    # Finance & Accounting
    "finance": {
        "keywords": ["finance", "banking", "investment", "trading", "analyst", "accounting", "audit", "credit", "financial advisor"],
        "skills": {"financial modeling", "risk assessment", "regulatory compliance", "bloomberg terminal", "trading systems", "tax preparation", "auditing", "quickbooks"},
        "tools": ["excel", "bloomberg terminal", "quickbooks", "sap", "tableau", "python", "r", "sql"],
        "certifications": ["cpa", "cfa", "frm", "pmp", "mba", "chartered accountant"]
    },
    
    "accounting": {
        "keywords": ["accountant", "bookkeeper", "cpa", "tax", "payroll", "accounts", "financial reporting", "quickbooks", "gaap"],
        "skills": {"gaap knowledge", "tax compliance", "financial reporting", "bookkeeping", "accounts payable", "accounts receivable", "payroll processing", "cost accounting"},
        "tools": ["quickbooks", "excel", "sap", "sage", "xero", "peachtree", "tax software"],
        "certifications": ["cpa", "cma", "ea", "quickbooks certified", "tax preparer license"]
    },
    
    # Marketing & Communications
    "marketing": {
        "keywords": ["marketing", "advertising", "brand", "campaign", "seo", "digital marketing", "social media", "content creator"],
        "skills": {"seo", "sem", "content marketing", "brand management", "campaign management", "analytics tools", "social media marketing", "market research"},
        "tools": ["google analytics", "hubspot", "mailchimp", "canva", "adobe creative", "hootsuite", "facebook ads"],
        "certifications": ["google ads", "hubspot", "facebook blueprint", "google analytics", "digital marketing"]
    },
    
    # Sales & Business Development
    "sales": {
        "keywords": ["sales", "business development", "account manager", "territory", "quota", "pipeline", "sales rep", "business sales"],
        "skills": {"crm", "lead generation", "pipeline management", "negotiation", "territory management", "sales forecasting", "cold calling", "relationship building"},
        "tools": ["salesforce", "hubspot", "pipedrive", "zoho", "linkedin", "crm software", "email automation"],
        "certifications": ["salesforce certified", "hubspot sales", "sales methodology", "negotiation training"]
    },
    
    # Manufacturing & Production
    "manufacturing": {
        "keywords": ["manufacturing", "production", "factory", "assembly", "quality", "lean", "operations", "plant", "industrial"],
        "skills": {"lean manufacturing", "six sigma", "quality control", "supply chain", "production planning", "machinery operation", "safety protocols", "inventory management"},
        "tools": ["erp systems", "mrp software", "quality management systems", "plc programming", "scada"],
        "certifications": ["six sigma", "lean certification", "quality management", "osha safety", "iso certification"]
    },
    
    # Education & Training
    "education": {
        "keywords": ["education", "teacher", "professor", "curriculum", "learning", "academic", "school", "instructor", "tutor"],
        "skills": {"curriculum development", "learning management systems", "assessment design", "classroom management", "educational technology", "special needs support"},
        "tools": ["lms", "interactive whiteboard", "educational software", "assessment tools", "google classroom", "canvas"],
        "certifications": ["teaching license", "educational leadership", "subject certifications", "special education"]
    },
    
    # Retail & Customer Service
    "retail": {
        "keywords": ["retail", "store", "customer service", "merchandising", "inventory", "sales associate", "cashier", "store manager"],
        "skills": {"inventory management", "pos systems", "merchandising", "customer experience", "supply chain", "loss prevention", "visual merchandising"},
        "tools": ["pos systems", "inventory software", "cash registers", "barcode scanners", "retail management"],
        "certifications": ["retail management", "customer service", "visual merchandising", "loss prevention"]
    },
    
    # Consulting & Professional Services
    "consulting": {
        "keywords": ["consulting", "consultant", "advisory", "strategy", "business analyst", "implementation", "project manager"],
        "skills": {"business analysis", "process mapping", "change management", "stakeholder management", "presentation skills", "project management", "client relations"},
        "tools": ["powerpoint", "excel", "visio", "project management software", "collaboration tools"],
        "certifications": ["pmp", "business analysis", "change management", "six sigma", "agile certification"]
    },
    
    # Nonprofit & Social Services
    "nonprofit": {
        "keywords": ["nonprofit", "ngo", "volunteer", "fundraising", "grant", "community", "social impact", "charity"],
        "skills": {"grant writing", "donor relations", "volunteer management", "program evaluation", "fundraising", "community outreach", "social impact measurement"},
        "tools": ["donor management software", "grant management", "volunteer platforms", "fundraising tools"],
        "certifications": ["fundraising certification", "grant writing", "nonprofit management", "volunteer coordination"]
    },
    
    # SKILLED TRADES & TECHNICAL PROFESSIONS
    "electrical": {
        "keywords": ["electrician", "electrical", "wiring", "circuits", "electrical contractor", "lineman", "electrical maintenance"],
        "skills": {"electrical codes", "wiring installation", "circuit design", "electrical safety", "troubleshooting", "motor controls", "plc programming", "electrical testing"},
        "tools": ["multimeter", "wire strippers", "conduit bender", "oscilloscope", "power tools", "safety equipment"],
        "certifications": ["electrical license", "nec certification", "osha safety", "electrical safety", "journeyman electrician"]
    },
    
    "plumbing": {
        "keywords": ["plumber", "plumbing", "pipes", "water", "sewer", "drain", "plumbing contractor", "pipefitter"],
        "skills": {"pipe installation", "plumbing codes", "leak detection", "water systems", "sewer systems", "pipe fitting", "soldering", "hydro jetting"},
        "tools": ["pipe wrench", "drain snake", "soldering torch", "pipe cutter", "pressure gauge", "inspection camera"],
        "certifications": ["plumbing license", "backflow certification", "gas line certification", "green plumbing"]
    },
    
    "carpentry": {
        "keywords": ["carpenter", "carpentry", "framing", "construction", "woodworking", "cabinetmaker", "finish carpenter"],
        "skills": {"framing", "finish carpentry", "blueprint reading", "wood joinery", "power tools", "building codes", "measurement accuracy", "safety procedures"},
        "tools": ["circular saw", "drill", "level", "measuring tape", "hammer", "nail gun", "router"],
        "certifications": ["carpentry certification", "construction safety", "blueprint reading", "finish carpentry"]
    },
    
    "welding": {
        "keywords": ["welder", "welding", "fabrication", "metalworking", "arc welding", "mig", "tig", "structural welding"],
        "skills": {"arc welding", "mig welding", "tig welding", "blueprint reading", "metal fabrication", "welding safety", "quality inspection", "material properties"},
        "tools": ["welding torch", "grinding wheel", "welding helmet", "plasma cutter", "measuring tools"],
        "certifications": ["welding certification", "aws certification", "structural welding", "pipe welding"]
    },
    
    "automotive": {
        "keywords": ["mechanic", "automotive", "car repair", "auto technician", "diesel mechanic", "automotive service"],
        "skills": {"engine diagnostics", "brake systems", "electrical systems", "transmission repair", "computer diagnostics", "safety inspections", "customer service"},
        "tools": ["diagnostic scanner", "socket set", "jack", "multimeter", "brake tools", "engine hoist"],
        "certifications": ["ase certification", "automotive technician", "brake certification", "emissions inspector"]
    },
    
    "hvac": {
        "keywords": ["hvac", "heating", "cooling", "air conditioning", "ventilation", "hvac technician", "refrigeration"],
        "skills": {"heating systems", "cooling systems", "ventilation", "refrigeration", "ductwork", "electrical controls", "energy efficiency", "safety protocols"},
        "tools": ["refrigerant gauges", "ductwork tools", "electrical meters", "brazing torch", "vacuum pump"],
        "certifications": ["hvac license", "epa certification", "refrigeration license", "energy efficiency"]
    },
    
    "construction": {
        "keywords": ["construction", "builder", "contractor", "foreman", "heavy equipment", "construction worker"],
        "skills": {"heavy machinery", "construction safety", "blueprint reading", "project coordination", "material management", "quality control", "osha compliance"},
        "tools": ["hard hat", "safety equipment", "measuring tools", "hand tools", "power tools", "heavy machinery"],
        "certifications": ["osha safety", "construction management", "heavy equipment", "project management"]
    },
    
    # SPECIALIZED & UNIQUE PROFESSIONS
    "diving": {
        "keywords": ["scuba", "diver", "underwater", "diving instructor", "marine", "dive master", "commercial diving"],
        "skills": {"scuba certification", "underwater welding", "dive safety", "equipment maintenance", "emergency procedures", "marine biology", "underwater photography", "decompression protocols"},
        "tools": ["scuba gear", "underwater tools", "dive computer", "underwater camera", "safety equipment"],
        "certifications": ["padi", "naui", "commercial diving", "underwater welding", "dive master"]
    },
    
    "aviation": {
        "keywords": ["pilot", "aviation", "aircraft", "flight", "air traffic", "aviation maintenance", "airline"],
        "skills": {"flight operations", "aircraft systems", "navigation", "weather analysis", "safety protocols", "emergency procedures", "flight planning", "communication systems"},
        "tools": ["flight instruments", "navigation equipment", "radio systems", "flight planning software"],
        "certifications": ["pilot license", "instrument rating", "commercial pilot", "atp", "flight instructor"]
    },
    
    "aerospace": {
        "keywords": ["astronaut", "aerospace", "spacecraft", "nasa", "space", "rocket", "mission specialist"],
        "skills": {"spacecraft systems", "orbital mechanics", "life support systems", "mission planning", "emergency protocols", "scientific research", "zero gravity operations"},
        "tools": ["spacecraft controls", "life support equipment", "scientific instruments", "communication systems"],
        "certifications": ["astronaut training", "spacecraft systems", "mission specialist", "scientific research"]
    },
    
    "marine": {
        "keywords": ["maritime", "boat", "ship", "marine", "naval", "coast guard", "maritime officer", "deck hand"],
        "skills": {"boat operation", "navigation", "marine safety", "weather analysis", "equipment maintenance", "emergency procedures", "maritime law", "cargo handling"},
        "tools": ["navigation equipment", "marine radio", "safety equipment", "deck tools", "navigation charts"],
        "certifications": ["captain's license", "maritime certification", "safety training", "navigation certification"]
    },
    
    "agriculture": {
        "keywords": ["farmer", "agriculture", "farming", "ranch", "livestock", "agricultural", "crop", "farm worker"],
        "skills": {"crop management", "livestock care", "farm equipment", "soil analysis", "pest control", "irrigation systems", "food safety", "sustainable farming"},
        "tools": ["tractors", "farm equipment", "irrigation systems", "soil testing", "livestock equipment"],
        "certifications": ["pesticide applicator", "organic certification", "livestock management", "agricultural safety"]
    },
    
    "culinary": {
        "keywords": ["chef", "cook", "culinary", "kitchen", "restaurant", "food service", "pastry chef", "line cook"],
        "skills": {"food preparation", "food safety", "menu planning", "inventory management", "kitchen management", "sanitation", "cost control", "customer service"},
        "tools": ["kitchen knives", "cooking equipment", "food processors", "ovens", "refrigeration", "pos systems"],
        "certifications": ["food safety", "culinary degree", "sommelier", "food handling", "kitchen management"]
    },
    
    # GOVERNMENT & PUBLIC SERVICE
    "government": {
        "keywords": ["government", "public service", "civil service", "federal", "state", "municipal", "public sector"],
        "skills": {"public policy", "regulatory compliance", "constituent services", "budget management", "public speaking", "legislative process", "ethics compliance"},
        "tools": ["government software", "document management", "case management", "budget software"],
        "certifications": ["public administration", "government ethics", "budget management", "policy analysis"]
    },
    
    "politics": {
        "keywords": ["politician", "political", "campaign", "elected", "legislative", "policy", "political advisor", "council", "government", "mayor", "senator"],
        "skills": {"campaign management", "public speaking", "policy analysis", "constituent relations", "fundraising", "media relations", "coalition building", "ethics compliance"},
        "tools": ["campaign software", "fundraising platforms", "social media", "polling software", "voter databases"],
        "certifications": ["campaign management", "political communications", "public administration", "ethics training"]
    },
    
    "law_enforcement": {
        "keywords": ["police", "officer", "law enforcement", "detective", "security", "sheriff", "corrections"],
        "skills": {"criminal law", "investigation techniques", "report writing", "community policing", "crisis management", "firearms training", "legal procedures"},
        "tools": ["patrol equipment", "investigation tools", "communication devices", "safety equipment", "forensic tools"],
        "certifications": ["police academy", "firearms certification", "defensive tactics", "crisis intervention"]
    },
    
    "firefighting": {
        "keywords": ["firefighter", "fire department", "ems", "paramedic", "emergency services", "fire inspector"],
        "skills": {"fire suppression", "emergency medical", "rescue operations", "hazardous materials", "equipment maintenance", "physical fitness", "team coordination"},
        "tools": ["fire equipment", "rescue tools", "medical equipment", "safety gear", "communication devices"],
        "certifications": ["firefighter certification", "emt", "paramedic", "hazmat", "rescue technician"]
    },
    
    # CREATIVE & ARTS
    "creative": {
        "keywords": ["designer", "artist", "creative", "graphic", "photographer", "writer", "creative director"],
        "skills": {"graphic design", "video editing", "photography", "creative writing", "project management", "client communication", "software proficiency", "portfolio development"},
        "tools": ["adobe creative", "camera equipment", "design software", "video editing", "drawing tablets"],
        "certifications": ["adobe certification", "photography", "graphic design", "creative writing"]
    },
    
    "entertainment": {
        "keywords": ["actor", "performer", "entertainment", "musician", "stage", "theater", "production"],
        "skills": {"performance skills", "stage management", "audio/visual equipment", "audience engagement", "rehearsal coordination", "contract negotiation"},
        "tools": ["sound equipment", "lighting", "musical instruments", "stage equipment", "recording software"],
        "certifications": ["performance training", "stage management", "audio engineering", "music theory"]
    },
    
    # TRANSPORTATION & LOGISTICS
    "transportation": {
        "keywords": ["driver", "transportation", "truck", "delivery", "logistics", "shipping", "warehouse"],
        "skills": {"vehicle operation", "route planning", "safety regulations", "cargo handling", "equipment maintenance", "customer service", "time management"},
        "tools": ["gps systems", "delivery equipment", "safety equipment", "vehicle maintenance", "tracking systems"],
        "certifications": ["cdl", "dot certification", "hazmat endorsement", "defensive driving", "transportation safety"]
    },
    
    "logistics": {
        "keywords": ["logistics", "supply chain", "warehouse", "distribution", "inventory", "procurement"],
        "skills": {"supply chain management", "inventory control", "warehouse operations", "shipping coordination", "data analysis", "vendor relations"},
        "tools": ["warehouse management", "inventory systems", "tracking software", "logistics software", "barcode scanners"],
        "certifications": ["supply chain management", "logistics certification", "warehouse management", "inventory control"]
    },
    
    # FOOD SERVICE & HOSPITALITY
    "foodservice": {
        "keywords": ["server", "bartender", "food service", "restaurant", "catering", "food prep"],
        "skills": {"food preparation", "food safety", "customer service", "inventory management", "pos systems", "team coordination", "sanitation protocols"},
        "tools": ["pos systems", "kitchen equipment", "serving equipment", "cleaning supplies", "food storage"],
        "certifications": ["food safety", "alcohol service", "customer service", "food handling"]
    },
    
    "hospitality": {
        "keywords": ["hotel", "hospitality", "guest services", "event planning", "tourism", "concierge"],
        "skills": {"guest services", "reservation systems", "event planning", "facility management", "customer relations", "problem solving"},
        "tools": ["reservation systems", "event planning software", "guest services", "facility management"],
        "certifications": ["hospitality management", "event planning", "customer service", "tourism"]
    },
    
    # STUDENTS & ACADEMIC CONTEXTS
    "student_highschool": {
        "keywords": ["high school", "student", "grade 9", "grade 10", "grade 11", "grade 12", "teenager"],
        "skills": {"study skills", "time management", "research methods", "presentation skills", "critical thinking", "collaboration", "digital literacy"},
        "tools": ["laptops", "educational software", "online platforms", "calculators", "research databases"],
        "certifications": ["academic achievements", "subject certifications", "skill certifications", "volunteer recognition"]
    },
    
    "student_undergraduate": {
        "keywords": ["undergraduate", "college student", "bachelor", "university", "freshman", "sophomore", "junior", "senior"],
        "skills": {"academic research", "citation methods", "project management", "internship seeking", "networking", "resume building", "career exploration"},
        "tools": ["research databases", "citation tools", "presentation software", "collaboration platforms", "career platforms"],
        "certifications": ["dean's list", "academic honors", "research participation", "leadership roles"]
    },
    
    "student_graduate": {
        "keywords": ["graduate student", "masters", "phd", "doctoral", "thesis", "research assistant"],
        "skills": {"research methodology", "thesis writing", "conference presentation", "academic publishing", "teaching assistance", "grant writing", "professional networking"},
        "tools": ["research software", "statistical analysis", "reference management", "presentation tools", "academic databases"],
        "certifications": ["research ethics", "teaching certification", "conference presentations", "published research"]
    },
    
    "student_vocational": {
        "keywords": ["trade school", "vocational", "apprentice", "technical college", "certification program"],
        "skills": {"hands-on training", "certification preparation", "apprenticeship readiness", "safety procedures", "technical skills", "workplace readiness"},
        "tools": ["trade tools", "safety equipment", "technical equipment", "measurement tools", "specialized software"],
        "certifications": ["trade certifications", "safety training", "apprenticeship completion", "skill certifications"]
    },
}

# Type definitions
IndustryProfile = TypedDict("IndustryProfile", {
    "industry": str,
    "keywords": List[str], 
    "skills": List[str],
    "tools": List[str],
    "certifications": List[str],
})

class SkillStore(Protocol):
    """Protocol for skill profile storage"""
    def get(self, key: str) -> Optional[Dict]: ...
    def set(self, key: str, profile: Dict) -> None: ...

class DefaultInMemoryStore:
    """Default in-memory store implementation"""
    def __init__(self):
        self._store: Dict[str, Dict] = {}
    
    def get(self, key: str) -> Optional[Dict]:
        return self._store.get(key)
    
    def set(self, key: str, profile: Dict) -> None:
        self._store[key] = profile

class ApiStore:
    """API-backed store with HTTP client"""
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, timeout: float = 4.0):
        self.base_url = (base_url or "https://api.example.com").rstrip("/")
        self.api_key = api_key or os.getenv("SKILL_API_KEY", "")
        self.timeout = timeout

    def _hdr(self):
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def get(self, key: str) -> Optional[Dict]:
        """GET /skills/{key} - requires requests library"""
        try:
            import requests
            r = requests.get(f"{self.base_url}/skills/{key}", headers=self._hdr(), timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            return None
        return None

    def set(self, key: str, profile: Dict) -> None:
        """PUT /skills/{key} - requires requests library"""
        try:
            import requests
            requests.put(f"{self.base_url}/skills/{key}", headers=self._hdr(),
                         data=json.dumps(profile), timeout=self.timeout)
        except Exception:
            pass

# --- helpers: safe JSON extraction for LLM strings ---
_JSON_OBJ_RE = re.compile(r'\{[\s\S]*\}')

def _parse_llm_json(resp) -> Dict:
    """Parse LLM response (string or dict) into JSON dict safely"""
    if isinstance(resp, dict):
        return resp
    if not isinstance(resp, str):
        return {}
    # Try fenced JSON first
    m = re.search(r'```json\s*([\s\S]*?)\s*```', resp, re.IGNORECASE)
    txt = m.group(1) if m else resp
    # Fallback: first balanced-ish {...} slice
    m2 = _JSON_OBJ_RE.search(txt)
    if not m2:
        return {}
    try:
        candidate = m2.group(0)
        # Remove trailing commas
        candidate = re.sub(r',(\s*[}\]])', r'\1', candidate)
        return json.loads(candidate)
    except Exception:
        return {}

def normalize_key(name: str) -> str:
    """Normalize industry/role name for cache/store keys"""
    return re.sub(r'[^a-z0-9_]+', '_', name.lower().strip()).strip('_')

def _word_in(text: str, kw: str) -> bool:
    """Word-boundary keyword match (case-insensitive) to avoid substring false-positives."""
    return re.search(rf"\b{re.escape(kw.lower())}\b", text.lower()) is not None

def _merge_lists(base: List[str], extra: List[str], max_items=15, lowercase=False) -> List[str]:
    """Merge two lists, deduplicating and capping at max_items"""
    seen, out = set(), []
    for lst in (base, extra):
        for it in lst or []:
            s = it.strip()
            if not s: continue
            if lowercase: s = s.lower()
            if s not in seen:
                seen.add(s); out.append(s)
            if len(out) >= max_items: return out
    return out

# Alias map for long-tail titles (no schema changes)
ALIASES = {
    # government / politics leadership
    "president": "politics",
    "prime minister": "politics", 
    "head of state": "politics",
    "civil servant": "government",
    "diplomat": "government",
    "chief of staff": "government",
    # research & science (fan out to closest buckets)
    "research scientist": "technology",
    "scientist": "technology",
    "physicist": "technology",
    "chemist": "technology",
    "biologist": "healthcare",
    "biomedical scientist": "healthcare", 
    "biostatistician": "healthcare",
    "marine biologist": "diving",
    # academia variants
    "lecturer": "education",
    "postdoc": "education",
    "professor": "education",
    # generic student handles already exist; add a couple more:
    "phd student": "student_graduate",
    "mba student": "student_graduate",
    "doctoral student": "student_graduate",
}

def _alias_lookup(name: str) -> Optional[str]:
    """Look up alias for role/industry name"""
    n = name.lower().strip()
    if n in ALIASES:
        return ALIASES[n]
    # word-boundary alias match
    for k, v in ALIASES.items():
        if _word_in(n, k):
            return v
    return None

def validate_profile(profile: Dict) -> IndustryProfile:
    """Validate and normalize industry profile from LLM or API"""
    # Safe defaults by type (don't put [] into 'industry')
    defaults = {
        "industry": "general",
        "keywords": [],
        "skills": [],
        "tools": [],
        "certifications": [],
    }
    data = {**defaults, **(profile or {})}

    def clean_list(items, max_items: int = 15, lowercase: bool = False) -> List[str]:
        if not isinstance(items, list):
            items = []
        out: List[str] = []
        for itm in items:
            if isinstance(itm, str):
                s = itm.strip()
                if not s:
                    continue
                if lowercase:
                    s = s.lower()
                if s not in out:
                    out.append(s)
        return out[:max_items]

    # Normalize industry string (lower/underscore to keep keys uniform)
    industry_str = str(data.get("industry") or "").strip()
    if not industry_str:
        industry_str = "general"
    industry_norm = normalize_key(industry_str)

    return IndustryProfile(
        industry=industry_norm,
        keywords=clean_list(data.get("keywords"), lowercase=True),
        skills=clean_list(data.get("skills"), lowercase=True),
        tools=clean_list(data.get("tools")),
        certifications=clean_list(data.get("certifications")),
    )

def build_profile_prompt(industry_or_role: str) -> str:
    """Build LLM prompt for industry/role profile generation"""
    return f"""Generate a comprehensive skill profile for the industry/role: "{industry_or_role}"

Return ONLY valid JSON in this exact format:
{{
  "industry": "{industry_or_role}",
  "keywords": ["keyword1", "keyword2", ...],
  "skills": ["skill1", "skill2", ...], 
  "tools": ["tool1", "tool2", ...],
  "certifications": ["cert1", "cert2", ...]
}}

Requirements:
- Max 15 items per list
- Use globally-recognized, real terms only
- Keywords and skills should be lowercase
- Tools and certifications keep original casing
- No fictional or made-up content
- Do not include "etc", "misc", "n/a", or placeholders
- Focus on practical, industry-standard items

Return only the JSON object, no other text."""

def build_industry_classify_prompt(resume_text: str) -> str:
    """Build LLM prompt for industry classification from resume"""
    # Truncate very long resume text
    truncated_text = resume_text[:2000] if len(resume_text) > 2000 else resume_text
    
    return f"""Classify the primary industry for this resume content:

"{truncated_text}"

Return ONLY a JSON object:
{{
  "industry": "primary_industry_name",
  "confidence": 0.85
}}

Choose from common industries like: technology, healthcare, finance, education, manufacturing, retail, consulting, government, creative, or others.
Use lowercase, underscore-separated names (e.g., "electronics_engineering", "software_development").
Confidence should be 0.0-1.0 where 1.0 is completely certain.

Return only the JSON object, no other text.
If unsure, pick the closest common industry and set confidence < 0.7."""

class SkillFrameworkManager:
    """
    LLM-augmented skill framework manager.
    
    Pipeline: static → cache → store → LLM → validate → store
    """
    
    def __init__(self, llm_call: Optional[Callable[[str], Dict]] = None, store: Optional[SkillStore] = None):
        self.llm_call = llm_call
        self.store = store or DefaultInMemoryStore()
    
    def clear_cache(self) -> None:
        """Clear LRU cache for get_profile"""
        try:
            self.get_profile.cache_clear()  # type: ignore[attr-defined]
        except Exception:
            pass
    
    @lru_cache(maxsize=256)
    def get_profile(self, industry_or_role: str, augment_if_static: bool = True) -> IndustryProfile:
        """
        Get industry profile through hybrid pipeline:
        1) Try alias lookup first
        2) Try static INDUSTRY_SKILLS by key or keyword match (with optional LLM augmentation)
        3) Try cache (handled by @lru_cache)
        4) Try store (API/database)
        5) If not found: call LLM → validate → store
        """
        # Check aliases first
        alias = _alias_lookup(industry_or_role)
        if alias and alias in INDUSTRY_SKILLS:
            return self.get_profile(alias, augment_if_static)
        
        normalized_key = normalize_key(industry_or_role)

        # 1) Static by normalized key (with optional LLM augmentation)
        if normalized_key in INDUSTRY_SKILLS:
            static_data = INDUSTRY_SKILLS[normalized_key]
            base = IndustryProfile(
                industry=normalized_key,
                keywords=[k.lower() for k in static_data.get("keywords", [])],
                skills=[s.lower() for s in list(static_data.get("skills", []))],
                tools=static_data.get("tools", []),
                certifications=static_data.get("certifications", []),
            )
            if augment_if_static and self.llm_call:
                try:
                    llm_resp = self.llm_call(build_profile_prompt(industry_or_role))
                    llm_data = _parse_llm_json(llm_resp)
                    v = validate_profile(llm_data)
                    return IndustryProfile(
                        industry=base["industry"],
                        keywords=_merge_lists(base["keywords"], v["keywords"], lowercase=True),
                        skills=_merge_lists(base["skills"], v["skills"], lowercase=True),
                        tools=_merge_lists(base["tools"], v["tools"]),
                        certifications=_merge_lists(base["certifications"], v["certifications"]),
                    )
                except Exception:
                    pass
            return base

        # 1b) Static by keyword (word-boundary match)
        for industry, config in INDUSTRY_SKILLS.items():
            kws = [k.lower() for k in config.get("keywords", [])]
            if any(_word_in(industry_or_role, kw) for kw in kws):
                return IndustryProfile(
                    industry=industry,
                    keywords=kws,
                    skills=[s.lower() for s in list(config.get("skills", []))],
                    tools=config.get("tools", []),
                    certifications=config.get("certifications", []),
                )

        # 3) Store
        stored_profile = self.store.get(normalized_key)
        if stored_profile:
            return validate_profile(stored_profile)

        # 4) LLM fallback
        if self.llm_call:
            try:
                prompt = build_profile_prompt(industry_or_role)
                llm_resp = self.llm_call(prompt)
                validated = validate_profile(_parse_llm_json(llm_resp))
                self.store.set(normalized_key, validated)
                return validated
            except Exception:
                pass

        # Final fallback
        return IndustryProfile(
            industry="general",
            keywords=[normalized_key],
            skills=["communication", "problem solving", "time management"],
            tools=["microsoft office", "email", "collaboration tools"],
            certifications=["professional development", "industry training"],
        )
    
    def determine_industry_from_resume(self, resume: Dict) -> str:
        """
        Determine industry from resume with LLM fallback for low confidence matches.
        """
        resume_text = " ".join([
            " ".join(resume.get("skills", [])),
            " ".join(
                f"{exp.get('company','')} {exp.get('role','')} " +
                " ".join(exp.get('responsibilities', []))
                for exp in resume.get("experience", [])
            ),
            " ".join(f"{edu.get('degree','')} {edu.get('field','')}" for edu in resume.get("education", [])),
        ])

        # Word-boundary scoring to avoid substring hits
        industry_scores = {}
        for industry, config in INDUSTRY_SKILLS.items():
            score = sum(1 for kw in config.get("keywords", []) if _word_in(resume_text, kw))
            if score:
                industry_scores[industry] = score

        if industry_scores:
            top_industry = max(industry_scores, key=industry_scores.get)
            if industry_scores[top_industry] >= 2:
                return top_industry

        if self.llm_call and resume_text.strip():
            try:
                prompt = build_industry_classify_prompt(resume_text)
                resp = self.llm_call(prompt)
                obj = _parse_llm_json(resp)
                if isinstance(obj, dict):
                    ind = normalize_key(str(obj.get("industry", "")).strip())
                    conf = float(obj.get("confidence", 0.0) or 0.0)
                    if ind and conf >= 0.7:
                        return ind
            except Exception:
                pass

        return max(industry_scores, key=industry_scores.get) if industry_scores else "general"

# Module-level singleton manager
_default_manager: Optional[SkillFrameworkManager] = None

def _get_default_manager() -> SkillFrameworkManager:
    """Get or create default manager instance"""
    global _default_manager
    if _default_manager is None:
        _default_manager = SkillFrameworkManager()
    return _default_manager

# Backward compatibility functions
def determine_industry_from_resume(resume: Dict) -> str:
    """
    Determine the most likely industry from resume content.
    Uses hybrid approach: deterministic + LLM fallback for low confidence.
    """
    return _get_default_manager().determine_industry_from_resume(resume)

def get_industry_skills(industry: str) -> Set[str]:
    """Get skills for a specific industry"""
    profile = _get_default_manager().get_profile(industry)
    return set(profile["skills"])

def get_industry_tools(industry: str) -> List[str]:
    """Get tools for a specific industry"""
    profile = _get_default_manager().get_profile(industry)
    return profile["tools"]

def get_industry_certifications(industry: str) -> List[str]:
    """Get certifications for a specific industry"""
    profile = _get_default_manager().get_profile(industry)
    return profile["certifications"]

def get_all_industries() -> List[str]:
    """Get list of all supported industries"""
    return list(INDUSTRY_SKILLS.keys())

def get_profile(industry_or_role: str) -> IndustryProfile:
    """Get complete industry profile (new public API)"""
    return _get_default_manager().get_profile(industry_or_role)