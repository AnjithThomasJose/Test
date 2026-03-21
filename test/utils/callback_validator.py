import json
import os
from jsonschema import validate, ValidationError

# Load schemas from markdown files (for now, hardcode schemas for agents up to assessment_recommender)

# Example schemas (should be expanded for all agents as per the markdown)
SCHEMAS = {
    "is_valid_resume": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["completed"]},
            "node": {"type": "string", "enum": ["is_valid_resume", "validate_resume"]},
            "output": {
                "type": "object",
                "properties": {
                    "is_valid_resume": {"type": "boolean"}
                },
                "required": ["is_valid_resume"]
            }
        },
        "required": ["status", "node", "output"]
    },
    "is_valid_jd": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["completed"]},
            "node": {"type": "string", "enum": ["is_valid_jd"]},
            "output": {
                "type": "object",
                "properties": {
                    "is_valid_jd": {"type": "boolean"}
                },
                "required": ["is_valid_jd"]
            }
        },
        "required": ["status", "node", "output"]
    },
    "assessment_question_generator": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["completed"]},
            "node": {"type": "string", "enum": ["assessment_question_generator"]},
            "output": {"type": "object"}
        },
        "required": ["status", "node", "output"]
    },
    "assessment_evaluator": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["completed"]},
            "node": {"type": "string", "enum": ["assessment_evaluator"]},
            "output": {"type": "object"}
        },
        "required": ["status", "node", "output"]
    },
    "report_generator": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["completed"]},
            "node": {"type": "string", "enum": ["report_generator"]},
            "output": {"type": "object"}
        },
        "required": ["status", "node", "output"]
    },
    "personal_info_parser": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["completed"]},
            "node": {"type": "string", "enum": ["personal_info_parser"]},
            "output": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "contact_details": {
                        "type": "object",
                        "properties": {
                            "Email": {"type": "string"},
                            "Phone": {"type": "string"},
                            "Location": {"type": "string"}
                        }
                    }
                },
                "required": ["name", "contact_details"]
            }
        },
        "required": ["status", "node", "output"]
    },
    "education_parser": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["completed"]},
            "node": {"type": "string", "enum": ["education_parser"]},
            "output": {
                "type": "object",
                "properties": {
                    "education": {"type": "array"}
                },
                "required": ["education"]
            }
        },
        "required": ["status", "node", "output"]
    },
    "experience_parser": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["completed"]},
            "node": {"type": "string", "enum": ["experience_parser"]},
            "output": {
                "type": "object",
                "properties": {
                    "work_experience": {"type": "array"},
                    "certifications": {"type": "array"},
                    "projects": {"type": "object"},
                    "extras": {"type": "object"}
                },
                "required": ["work_experience"]
            }
        },
        "required": ["status", "node", "output"]
    },
    "skills_parser": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["completed"]},
            "node": {"type": "string", "enum": ["skills_parser"]},
            "output": {
                "type": "object",
                "properties": {
                    "skills": {"type": "array"}
                },
                "required": ["skills"]
            }
        },
        "required": ["status", "node", "output"]
    },
    "resume_scorer": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["completed"]},
            "node": {"type": "string", "enum": ["resume_scorer"]},
            "output": {"type": "object"}
        },
        "required": ["status", "node", "output"]
    },
    "interest_filler": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["completed"]},
            "node": {"type": "string", "enum": ["interest_filler"]},
            "output": {
                "type": "object",
                "properties": {
                    "user_interests": {"type": "array"}
                },
                "required": ["user_interests"]
            }
        },
        "required": ["status", "node", "output"]
    },
    "career_advisor": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["completed"]},
            "node": {"type": "string", "enum": ["career_advisor"]},
            "output": {"type": "object"}
        },
        "required": ["status", "node", "output"]
    },
    "market_and_course_recommender": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["completed"]},
            "node": {"type": "string", "enum": ["market_and_course_recommender"]},
            "output": {"type": "object"}
        },
        "required": ["status", "node", "output"]
    },
    "assessment_recommender": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["completed"]},
            "node": {"type": "string", "enum": ["assessment_recommender"]},
            "output": {"type": "object"}
        },
        "required": ["status", "node", "output"]
    },
    "groq_resume_parser": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["completed"]},
            "node": {"type": "string", "enum": ["groq_resume_parser"]},
            "output": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "contact_details": {"type": "object"},
                    "education": {"type": "array"},
                    "work_experience": {"type": "array"},
                    "skills": {"type": "array"},
                    "certifications": {"type": "array"},
                    "projects": {"type": "array"},
                    "extras": {"type": "array"},
                    "professional_affiliations": {"type": "array"},
                    "total_experience_years": {"type": ["string", "number"]},  # Accept both string (e.g., "1 years 5 months") and number
                    "professional_summary": {"type": "string"},
                    "structured_resume": {"type": "object"}
                }
            }
        },
        "required": ["status", "node", "output"]
    }
}

def validate_callback(agent_name: str, data: dict) -> None:
    """
    Validate callback data for the given agent name. Raises ValidationError if invalid.
    """
    # Map alternative agent names to schema keys
    agent_schema_map = {
        "validate_resume": "is_valid_resume",
        "validate_jd": "is_valid_jd",
        # Add other mappings if needed
    }
    
    # Get the schema key (either direct match or from the mapping)
    schema_key = agent_schema_map.get(agent_name, agent_name)
    
    schema = SCHEMAS.get(schema_key)
    if not schema:
        raise ValueError(f"No schema defined for agent: {agent_name} (schema key: {schema_key})")
    
    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        raise ValidationError(f"Callback data for {agent_name} does not match schema: {e.message}")
