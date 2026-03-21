from typing import Dict, List

ROLE_SPECIFIC_SCORING = {
    "TECHNICAL_CODING": {
        "categories": {
            "functional": {"weight": 0.25, "max_score": 10},
            "problem_solving": {"weight": 0.25, "max_score": 10},
            "behavioral": {"weight": 0.20, "max_score": 10},
            "fitment": {"weight": 0.20, "max_score": 10},
            "professionalism": {"weight": 0.10, "max_score": 10}
        },
        "keywords": {
            "functional": ["code", "algorithm", "architecture", "api", "database", "design pattern", "programming", "development", "implementation", "framework", "library", "syntax", "debugging", "testing", "deployment"],
            "problem_solving": ["debug", "optimize", "solve", "approach", "trade-off", "analysis", "troubleshoot", "investigate", "fix", "resolve", "improve", "enhance", "refactor", "optimization", "performance"],
            "behavioral": ["team", "collaborate", "code review", "mentor", "pair programming", "agile", "scrum", "communication", "feedback", "knowledge sharing", "leadership", "mentoring"],
            "fitment": ["learn", "grow", "passion", "goal", "career", "development", "challenge", "innovation", "technology", "excitement", "motivation", "aspiration"],
            "professionalism": ["clear", "structured", "concise", "professional", "organized", "thorough", "detailed", "precise", "accurate", "reliable"]
        }
    },
    "TECHNICAL_DATA": {
        "categories": {
            "ml_architecture": {"weight": 0.25, "max_score": 10},
            "data_pipelines": {"weight": 0.20, "max_score": 10},
            "evaluation_monitoring": {"weight": 0.20, "max_score": 10},
            "problem_solving": {"weight": 0.20, "max_score": 10},
            "communication_fitment": {"weight": 0.15, "max_score": 10}
        },
        "keywords": {
            "ml_architecture": ["model", "embedding", "transformer", "rag", "vector", "llm", "neural network", "deep learning", "machine learning", "ai", "algorithm", "training", "inference", "pipeline"],
            "data_pipelines": ["pipeline", "etl", "airflow", "orchestration", "feature engineering", "data processing", "batch", "streaming", "kafka", "spark", "hadoop", "data lake", "warehouse"],
            "evaluation_monitoring": ["metrics", "monitoring", "drift", "mlflow", "evaluation", "validation", "testing", "accuracy", "precision", "recall", "f1", "auc", "performance", "tracking"],
            "problem_solving": ["optimize", "debug", "analyze", "experiment", "hypothesis", "testing", "validation", "improvement", "enhancement", "troubleshoot", "investigate"],
            "communication_fitment": ["explain", "stakeholder", "vision", "communicate", "presentation", "documentation", "insights", "findings", "business impact", "value", "storytelling"]
        }
    },
    "TECHNICAL_INFRASTRUCTURE": {
        "categories": {
            "functional_design": {"weight": 0.25, "max_score": 10},
            "scalability_reliability": {"weight": 0.25, "max_score": 10},
            "problem_solving": {"weight": 0.20, "max_score": 10},
            "communication": {"weight": 0.15, "max_score": 10},
            "leadership_fitment": {"weight": 0.15, "max_score": 10}
        },
        "keywords": {
            "functional_design": ["architecture", "component", "service", "api", "infrastructure", "system design", "microservices", "distributed", "cloud", "aws", "azure", "gcp", "kubernetes", "docker"],
            "scalability_reliability": ["scale", "shard", "replica", "cache", "availability", "fault tolerance", "redundancy", "load balancing", "auto-scaling", "monitoring", "alerting", "disaster recovery"],
            "problem_solving": ["trade-off", "optimize", "debug", "analyze", "troubleshoot", "incident", "root cause", "solution", "improvement", "enhancement", "performance"],
            "communication": ["explain", "diagram", "document", "presentation", "stakeholder", "team", "collaboration", "knowledge sharing", "mentoring"],
            "leadership_fitment": ["mentor", "lead", "collaborate", "team", "leadership", "guidance", "coaching", "development", "growth", "vision"]
        }
    },
    "OPERATIONS_MANAGEMENT": {
        "categories": {
            "functional": {"weight": 0.20, "max_score": 10},
            "problem_solving": {"weight": 0.25, "max_score": 10},
            "behavioral": {"weight": 0.20, "max_score": 10},
            "fitment": {"weight": 0.20, "max_score": 10},
            "professionalism": {"weight": 0.15, "max_score": 10}
        },
        "keywords": {
            "functional": ["operations", "process", "efficiency", "metrics", "kpi", "governance", "compliance", "standards", "procedures", "workflow", "automation", "optimization"],
            "problem_solving": ["challenge", "solution", "improve", "optimize", "resolve", "troubleshoot", "analysis", "root cause", "fix", "enhancement", "innovation", "transformation"],
            "behavioral": ["team", "leadership", "stakeholder", "conflict", "communication", "collaboration", "management", "mentoring", "coaching", "influence", "negotiation"],
            "fitment": ["goal", "aspiration", "culture", "vision", "growth", "development", "career", "passion", "motivation", "challenge", "impact", "contribution"],
            "professionalism": ["structured", "clear", "concise", "professional", "organized", "thorough", "reliable", "accountable", "transparent", "ethical"]
        }
    },
    "TECHNICAL_DESIGN": {
        "categories": {
            "functional": {"weight": 0.25, "max_score": 10},
            "problem_solving": {"weight": 0.20, "max_score": 10},
            "behavioral": {"weight": 0.20, "max_score": 10},
            "fitment": {"weight": 0.20, "max_score": 10},
            "professionalism": {"weight": 0.15, "max_score": 10}
        },
        "keywords": {
            "functional": ["design", "ui", "ux", "user research", "wireframe", "prototype", "usability", "accessibility", "interaction", "visual", "interface", "experience", "user journey"],
            "problem_solving": ["iterate", "test", "validate", "improve", "optimize", "analyze", "research", "experiment", "hypothesis", "solution", "innovation", "creativity"],
            "behavioral": ["collaborate", "feedback", "stakeholder", "user", "team", "communication", "presentation", "critique", "iteration", "adaptability", "empathy"],
            "fitment": ["creativity", "empathy", "vision", "passion", "innovation", "aesthetics", "user-centered", "problem-solving", "growth", "learning", "challenge"],
            "professionalism": ["clear", "structured", "process", "methodology", "documentation", "presentation", "communication", "collaboration", "professional", "organized"]
        }
    },
    "BUSINESS_STRATEGIC": {
        "categories": {
            "functional": {"weight": 0.25, "max_score": 10},
            "problem_solving": {"weight": 0.25, "max_score": 10},
            "behavioral": {"weight": 0.20, "max_score": 10},
            "fitment": {"weight": 0.20, "max_score": 10},
            "professionalism": {"weight": 0.10, "max_score": 10}
        },
        "keywords": {
            "functional": ["strategy", "product", "market", "roi", "roadmap", "business model", "value proposition", "competitive analysis", "market research", "positioning", "go-to-market"],
            "problem_solving": ["analyze", "prioritize", "trade-off", "decision", "evaluation", "assessment", "optimization", "improvement", "innovation", "transformation", "change management"],
            "behavioral": ["stakeholder", "negotiate", "influence", "communicate", "leadership", "collaboration", "team", "presentation", "persuasion", "relationship building"],
            "fitment": ["vision", "growth", "leadership", "impact", "contribution", "passion", "motivation", "challenge", "development", "career", "aspiration"],
            "professionalism": ["data-driven", "structured", "clear", "analytical", "strategic", "professional", "organized", "thorough", "reliable", "accountable"]
        }
    },
    "CREATIVE_MARKETING": {
        "categories": {
            "functional": {"weight": 0.25, "max_score": 10},
            "problem_solving": {"weight": 0.20, "max_score": 10},
            "behavioral": {"weight": 0.20, "max_score": 10},
            "fitment": {"weight": 0.20, "max_score": 10},
            "professionalism": {"weight": 0.15, "max_score": 10}
        },
        "keywords": {
            "functional": ["campaign", "brand", "content", "seo", "analytics", "conversion", "marketing", "advertising", "social media", "digital", "strategy", "creative", "messaging"],
            "problem_solving": ["optimize", "test", "analyze", "improve", "experiment", "a/b testing", "performance", "metrics", "roi", "efficiency", "innovation", "creative solution"],
            "behavioral": ["collaborate", "stakeholder", "creative", "team", "communication", "presentation", "feedback", "iteration", "adaptability", "creativity", "brainstorming"],
            "fitment": ["passion", "creativity", "growth", "innovation", "aesthetics", "brand", "marketing", "communication", "storytelling", "impact", "motivation"],
            "professionalism": ["data-driven", "results-oriented", "clear", "organized", "professional", "creative", "strategic", "analytical", "collaborative", "reliable"]
        }
    },
    "SALES_BUSINESS": {
        "categories": {
            "functional": {"weight": 0.25, "max_score": 10},
            "problem_solving": {"weight": 0.20, "max_score": 10},
            "behavioral": {"weight": 0.25, "max_score": 10},
            "fitment": {"weight": 0.20, "max_score": 10},
            "professionalism": {"weight": 0.10, "max_score": 10}
        },
        "keywords": {
            "functional": ["sales", "pipeline", "revenue", "quota", "crm", "prospecting", "lead generation", "conversion", "deal", "negotiation", "closing", "relationship"],
            "problem_solving": ["negotiate", "close", "overcome objection", "strategy", "solution", "challenge", "opportunity", "analysis", "improvement", "optimization", "innovation"],
            "behavioral": ["relationship", "trust", "communicate", "listen", "empathy", "persuasion", "influence", "collaboration", "team", "leadership", "mentoring", "coaching"],
            "fitment": ["goal-oriented", "resilient", "growth", "achievement", "success", "motivation", "passion", "challenge", "development", "career", "aspiration"],
            "professionalism": ["professional", "reliable", "organized", "ethical", "transparent", "accountable", "results-driven", "persistent", "dedicated", "committed"]
        }
    }
}

# Core categories as fallback
CORE_SCORING_CATEGORIES = {
    "functional": {"weight": 0.25, "max_score": 10},
    "problem_solving": {"weight": 0.25, "max_score": 10},
    "behavioral": {"weight": 0.20, "max_score": 10},
    "fitment": {"weight": 0.20, "max_score": 10},
    "professionalism": {"weight": 0.10, "max_score": 10}
}
