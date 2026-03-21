import json
import re
import asyncio
import hashlib
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
from pydantic import BaseModel, Field, validator
from models.llm_invoker import invoke_llm, invoke_structured_llm, rewrite_steps_with_context
from langsmith.run_helpers import traceable
from chroma import get_chat_session, update_chat_session
from utils.session_manager import session_manager
from core.security import validate_tenant_id, redact_pii
from core.utils import run_blocking_io
from urllib.parse import urlparse
from core.model_registry import TaskType
from langchain_google_genai import ChatGoogleGenerativeAI
from settings import settings as _settings

log = logging.getLogger(__name__)

# ============================================================================
# URL VALIDATION UTILITIES
# ============================================================================

def validate_url(url: str) -> bool:
    """Validate if a URL is well-formed and potentially accessible."""
    if not isinstance(url, str) or not url.strip():
        return False
    
    try:
        result = urlparse(url.strip())
        # Check if it has a valid scheme (http, https)
        if result.scheme not in ['http', 'https']:
            return False
        # Check if it has a netloc (domain)
        if not result.netloc:
            return False
        # Basic format validation - must start with http:// or https://
        if not re.match(r'^https?://', url.strip()):
            return False
        # Check for basic domain format (at least one dot or localhost)
        if '.' not in result.netloc and result.netloc != 'localhost':
            return False
        return True
    except Exception:
        return False

def filter_valid_resources(resources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out resources with invalid links, keeping only those with at least one valid link."""
    if not isinstance(resources, list):
        return []
    
    filtered = []
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        
        links = resource.get("links", [])
        if not isinstance(links, list):
            continue
        
        # Filter valid links
        valid_links = [link for link in links if isinstance(link, str) and validate_url(link)]
        
        # Only include resource if it has at least one valid link
        if valid_links:
            filtered.append({
                "subtopic": resource.get("subtopic", ""),
                "links": valid_links
            })
    
    return filtered


def _to_optional_int(value: Any) -> Optional[int]:
    """Best-effort conversion for optional count fields."""
    try:
        if value in (None, ""):
            return None
        return int(round(float(str(value).strip())))
    except (TypeError, ValueError):
        return None


def _format_score_value(value: Any) -> str:
    """Format score values without unnecessary trailing zeros."""
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _normalize_sentence(text: Any) -> str:
    """Normalize optional free-text explanations into a single sentence."""
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _build_weighted_proficiency_text(score: float, max_score: float, percentage: float) -> str:
    """Describe the canonical normalized weighted score consistently."""
    return (
        f"weighted proficiency score of {_format_score_value(score)} out of "
        f"{_format_score_value(max_score)} ({percentage:.1f}%)"
    )


def _build_assessment_count_sentence(assessment_results: Dict[str, Any]) -> str:
    """Build concise count-based scoring context when explicit question counts are available."""
    attempted = _to_optional_int(assessment_results.get("attempted_questions"))
    correct = _to_optional_int(assessment_results.get("fully_correct_questions"))
    credit_earning = _to_optional_int(assessment_results.get("correct_answers"))
    partially_correct = _to_optional_int(assessment_results.get("partially_correct_questions"))
    incorrect = _to_optional_int(assessment_results.get("incorrect_questions"))
    if incorrect is None:
        incorrect = _to_optional_int(assessment_results.get("incorrect_answers"))

    details: List[str] = []
    if correct is not None:
        if attempted is not None and attempted > 0:
            details.append(f"{correct} out of {attempted} questions were fully correct")
        else:
            details.append(f"{correct} questions were fully correct")
    elif credit_earning is not None:
        credit_label = "question earned some credit" if credit_earning == 1 else "questions earned some credit"
        details.append(f"{credit_earning} {credit_label}")
    if partially_correct is not None and partially_correct > 0:
        partial_label = "question was partially correct" if partially_correct == 1 else "questions were partially correct"
        details.append(f"{partially_correct} {partial_label}")
    if incorrect is not None:
        incorrect_label = "question was incorrect" if incorrect == 1 else "questions were incorrect"
        details.append(f"{incorrect} {incorrect_label}")

    if attempted is not None:
        attempted_label = "question was attempted" if attempted == 1 else "questions were attempted"
        if details:
            return f"{attempted} {attempted_label}; {' and '.join(details)}."
        return f"{attempted} {attempted_label}."

    if details:
        return f"{' and '.join(details)}."

    return ""

# ============================================================================
# SPELL CHECKING AND TEXT VALIDATION
# ============================================================================

class ReportSpellChecker:
    """Spell checking and text validation for report content"""
    
    def __init__(self):
        # Common technical terms and domain-specific words
        self.technical_terms = {
            # Programming languages and frameworks
            'javascript': 'javascript', 'typescript': 'typescript', 'python': 'python', 'java': 'java', 
            'csharp': 'csharp', 'c++': 'c++', 'php': 'php', 'ruby': 'ruby', 'go': 'go', 'rust': 'rust',
            'react': 'react', 'angular': 'angular', 'vue': 'vue', 'nodejs': 'nodejs', 
            'express': 'express', 'django': 'django', 'flask': 'flask', 'spring': 'spring', 'laravel': 'laravel',
            'mongodb': 'mongodb', 'postgresql': 'postgresql', 'mysql': 'mysql', 'redis': 'redis', 
            'elasticsearch': 'elasticsearch', 'kubernetes': 'kubernetes', 'docker': 'docker',
            
            # Technical concepts
            'api': 'api', 'rest': 'rest', 'graphql': 'graphql', 'microservices': 'microservices', 
            'devops': 'devops', 'ci/cd': 'ci/cd', 'agile': 'agile', 'scrum': 'scrum',
            'algorithms': 'algorithms', 'datastructures': 'datastructures', 'machinelearning': 'machinelearning', 
            'artificialintelligence': 'artificialintelligence', 'frontend': 'frontend', 'backend': 'backend', 
            'fullstack': 'fullstack', 'responsive': 'responsive', 'accessibility': 'accessibility',
            
            # Assessment and career terms
            'assessment': 'assessment', 'evaluation': 'evaluation', 'competency': 'competency', 
            'proficiency': 'proficiency', 'skillgap': 'skillgap', 'roadmap': 'roadmap',
            'certification': 'certification', 'portfolio': 'portfolio', 'resume': 'resume', 
            'interview': 'interview', 'recruitment': 'recruitment', 'onboarding': 'onboarding',
            
            # Common misspellings corrections
            'recieve': 'receive', 'seperate': 'separate', 'occured': 'occurred', 'definately': 'definitely',
            'accomodate': 'accommodate', 'begining': 'beginning', 'calender': 'calendar',
            'existance': 'existence', 'independant': 'independent', 'occassion': 'occasion',
            'priviledge': 'privilege', 'reccomend': 'recommend', 'succesful': 'successful',
            'teh': 'the', 'adn': 'and', 'taht': 'that', 'wih': 'with', 'ot': 'to',
            'programming': 'programming', 'developement': 'development', 'enviroment': 'environment',
            'perfomance': 'performance', 'achieve': 'achieve', 'achivement': 'achievement',
            'improve': 'improve', 'improvement': 'improvement', 'learn': 'learn', 'learning': 'learning'
        }
        
        # Common grammar patterns to fix
        self.grammar_patterns = [
            (r'\ba\s+([aeiou])', r'an \1'),  # a apple -> an apple
            (r'\ban\s+([bcdfghjklmnpqrstvwxyz])', r'a \1'),  # an book -> a book
            (r'\b(its)\s+([a-z])', r"it's \2"),  # its -> it's (context dependent)
            (r'\b(youre)\b', r"you're"),
            (r'\b(theyre)\b', r"they're"),
            (r'\b(were)\b', r"we're"),
            (r'\b(cant)\b', r"can't"),
            (r'\b(wont)\b', r"won't"),
            (r'\b(dont)\b', r"don't"),
            (r'\b(doesnt)\b', r"doesn't"),
            (r'\b(havent)\b', r"haven't"),
            (r'\b(hasnt)\b', r"hasn't"),
            (r'\b(hadnt)\b', r"hadn't"),
            (r'\b(wouldnt)\b', r"wouldn't"),
            (r'\b(couldnt)\b', r"couldn't"),
            (r'\b(shouldnt)\b', r"shouldn't"),
        ]
    
    def check_and_correct_text(self, text: str) -> str:
        """Check and correct spelling and grammar in text"""
        if not isinstance(text, str):
            return text
            
        corrected_text = text
        
        # Fix common grammar patterns
        for pattern, replacement in self.grammar_patterns:
            corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
        
        # Fix common misspellings
        words = corrected_text.split()
        corrected_words = []
        
        for word in words:
            # Clean word (remove punctuation for checking)
            clean_word = re.sub(r'[^\w]', '', word.lower())
            
            # Check if it's a known misspelling
            if clean_word in self.technical_terms:
                # Keep original capitalization and punctuation
                original_punctuation = re.sub(r'[\w]', '', word)
                corrected_word = self.technical_terms[clean_word] + original_punctuation
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def validate_report_content(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and correct spelling/grammar in report content"""
        corrected_report = report.copy()
        
        # Fields to check and correct
        text_fields = ['summary', 'feedback']
        list_fields = ['suggested_next_steps']
        
        # Correct text fields
        for field in text_fields:
            if field in corrected_report and isinstance(corrected_report[field], str):
                corrected_report[field] = self.check_and_correct_text(corrected_report[field])
        
        # Correct list fields
        for field in list_fields:
            if field in corrected_report and isinstance(corrected_report[field], list):
                corrected_report[field] = [
                    self.check_and_correct_text(item) if isinstance(item, str) else item
                    for item in corrected_report[field]
                ]
        
        # Correct topics if they exist
        if 'topics' in corrected_report and isinstance(corrected_report['topics'], list):
            for topic in corrected_report['topics']:
                if isinstance(topic, dict):
                    # Correct topic fields
                    for field in ['topic', 'feedback']:
                        if field in topic and isinstance(topic[field], str):
                            topic[field] = self.check_and_correct_text(topic[field])
                    
                    # Correct arrays within topics
                    for field in ['strengths', 'areas_for_improvement', 'recommendations']:
                        if field in topic and isinstance(topic[field], list):
                            topic[field] = [
                                self.check_and_correct_text(item) if isinstance(item, str) else item
                                for item in topic[field]
                            ]
        
        return corrected_report

# ============================================================================
# TOKEN-OPTIMIZED ENHANCED REPORT GENERATION SYSTEM
# ============================================================================

class TokenOptimizedReportAnalyzer:
    """Lightweight analysis for report generation with minimal token usage"""
    
    def __init__(self):
        self.max_topic_length = 50
        self.max_feedback_length = 800
        self.max_summary_length = 200
        # Domain keyword maps for inferring a more precise assessment topic from questions.
        # This is extensible at runtime via core.config.get_domain_keywords() if present.
        self.domain_keyword_map = self._load_domain_keyword_map()
        # Generic or non-descriptive topic labels that should trigger inference
        self._generic_topic_labels = {
            "assessment", "assessment test", "performance assessment", "performance test",
            "test", "exam", "quiz", "evaluation", "general assessment", "practice test"
        }
    
    def _load_domain_keyword_map(self) -> Dict[str, Dict[str, Any]]:
        """Load domain keyword map, optionally overriding/augmenting via core.config."""
        base_map: Dict[str, Dict[str, Any]] = {
            # Fashion/Draping domain (example provided by user report)
            "Draping": {
                "keywords": [
                    "draping", "dress form", "bodice", "muslin", "grainline",
                    "bias", "dart", "tuck", "ease", "hemline", "truing", "trueing",
                    "cowl", "ruffle", "waistline", "skirt", "sleeve cap", "a-line"
                ]
            },
            # Education Assessment Design (to distinguish from generic “Performance Assessment”)
            "Assessment Design": {
                "keywords": [
                    "formative", "summative", "diagnostic", "rubric", "feedback",
                    "validity", "reliability", "learning objectives", "assessment method",
                    "evaluation criteria"
                ]
            },
            # Software Testing
            "Software Testing": {
                "keywords": [
                    "unit test", "integration test", "e2e", "mock", "stub",
                    "coverage", "assertion", "fixture", "test case", "regression"
                ]
            },
            # Databases / SQL
            "SQL and Databases": {
                "keywords": [
                    "select", "join", "where", "group by", "index", "primary key",
                    "foreign key", "transaction", "isolation", "normalization"
                ]
            },
            # Networking
            "Computer Networking": {
                "keywords": [
                    "tcp", "udp", "ip address", "subnet", "routing", "dns",
                    "latency", "throughput", "bandwidth", "packet"
                ]
            },
        }
        # Optionally merge external map from configuration
        try:
            from core.config import get_domain_keywords  # type: ignore
            external = get_domain_keywords()
            if isinstance(external, dict):
                # Merge/override keys from external map
                for k, v in external.items():
                    if isinstance(v, dict):
                        base_map[k] = v
        except Exception:
            pass
        return base_map
    
    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        """Safely convert inputs (including None/str) to float with a default fallback."""
        try:
            if value is None:
                return default
            # Handle strings like "85%" or "85.0"
            if isinstance(value, str):
                cleaned = value.strip().replace('%', '')
                return float(cleaned)
            return float(value)
        except (ValueError, TypeError):
            return default

    def extract_key_insights(self, assessment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key insights with minimal token usage (synchronous version for backward compatibility)"""
        raw_score = assessment_results.get('total_score', 0)
        raw_max_score = assessment_results.get('max_score', 100)

        score = self._to_float(raw_score, 0.0)
        max_score = self._to_float(raw_max_score, 0.0)

        insights = {
            "score": score,
            "max_score": max_score,
            "topic": assessment_results.get('assessment_topic', 'Assessment')[:30],  # Truncate
            "sections": {},
            "topics": [],
            "performance_pattern": ""
        }
        
        # Analyze section scores efficiently
        section_scores = assessment_results.get('section_scores', {})
        if section_scores:
            # Coerce values to float for reliable comparisons
            numeric_section_scores = {k: self._to_float(v, 0.0) for k, v in section_scores.items()}
            if numeric_section_scores:
                best_section = max(numeric_section_scores.items(), key=lambda x: x[1])
                worst_section = min(numeric_section_scores.items(), key=lambda x: x[1])
            else:
                best_section = ("", 0.0)
                worst_section = ("", 0.0)
            insights["sections"] = {
                "best": f"{best_section[0]}({best_section[1]})",
                "worst": f"{worst_section[0]}({worst_section[1]})"
            }
        
        # Potentially refine the topic if the provided topic appears generic
        provided_topic = (assessment_results.get('assessment_topic') or '').lower()
        if self._is_generic_topic(provided_topic):
            refined = self._infer_topic_from_questions(assessment_results.get('per_question', {}))
            if refined:
                insights["topic"] = refined[:30]
        
        # Extract topics efficiently
        insights["topics"] = self._extract_topics_compact(assessment_results)
        
        # Determine performance pattern (rule-based for backward compatibility)
        percentage = (score / max_score * 100) if max_score and max_score > 0 else 0.0
        if percentage >= 80:
            insights["performance_pattern"] = "strong"
        elif percentage >= 60:
            insights["performance_pattern"] = "moderate"
        else:
            insights["performance_pattern"] = "needs_work"
        
        return insights
    
    async def extract_key_insights_enhanced(self, assessment_results: Dict[str, Any], candidate_profile: Dict[str, Any] = None, use_llm: bool = True) -> Dict[str, Any]:
        """Extract key insights with LLM-enhanced performance assessment.
        
        Args:
            assessment_results: Full assessment data
            candidate_profile: Optional candidate profile for context-aware assessment
            use_llm: If True, use LLM for performance assessment; otherwise use rules
        """
        # Start with base insights
        insights = self.extract_key_insights(assessment_results)
        
        if not use_llm or not candidate_profile:
            return insights
        
        # Enhance with LLM-based performance assessment
        try:
            score = insights["score"]
            max_score = insights["max_score"]
            test_type = assessment_results.get("test_type", "assessment")
            
            # Get domain and experience for context
            domain = candidate_profile.get("domain_expertise", [None])[0] if candidate_profile.get("domain_expertise") else None
            experience_level = self._infer_experience_level(candidate_profile.get("total_experience_years", 0))
            
            # Call LLM for performance assessment
            perf_result = await self._determine_performance_level_with_llm(
                score=score,
                max_score=max_score,
                test_type=test_type,
                test_difficulty="medium",  # Could be enhanced
                domain=domain,
                experience_level=experience_level,
                context={
                    "total_questions": len(assessment_results.get("per_question", {})),
                    "topics": insights["topics"][:3]
                }
            )
            
            # Add LLM results to insights
            insights["llm_performance_level"] = perf_result["performance_level"]
            insights["llm_reasoning"] = perf_result["reasoning"]
            insights["llm_percentile"] = perf_result["percentile_estimate"]
            
        except Exception as e:
            log.warning(f"LLM performance enhancement failed: {e}")
        
        return insights
    
    def _parse_experience_value(self, experience_value: Any) -> float:
        """Parse experience years from various formats (int, float, or string like '8 years 4 months')."""
        if experience_value is None:
            return 0.0
        
        # If already a number, convert to float
        if isinstance(experience_value, (int, float)):
            return float(experience_value)
        
        # If it's a string, try to parse it
        if isinstance(experience_value, str):
            experience_str = experience_value.strip()
            
            # Try to extract years using regex (handles "8 years 4 months", "8 years", "8", etc.)
            # Pattern matches: "8 years 4 months", "8 years", "8y", "8.5 years", etc.
            match = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?|y\b)', experience_str.lower())
            if match:
                return float(match.group(1))
            
            # Fallback: try to extract any number
            match = re.search(r'(\d+(?:\.\d+)?)', experience_str)
            if match:
                return float(match.group(1))
        
        return 0.0
    
    def _infer_experience_level(self, years: Any) -> str:
        """Infer experience level from years (simple rule-based). Handles int, float, or string formats."""
        years_num = self._parse_experience_value(years)
        if years_num < 2:
            return "entry"
        elif years_num < 5:
            return "mid"
        else:
            return "senior"
    
    def _is_generic_topic(self, topic_lower: str) -> bool:
        """Returns True if the provided topic looks generic or non-descriptive."""
        if not topic_lower:
            return True
        t = topic_lower.strip()
        return t in self._generic_topic_labels
    
    def _infer_topic_from_questions(self, per_question: Dict[str, Any]) -> Optional[str]:
        """Infer a subject-matter topic from question texts when the declared topic is generic."""
        if not isinstance(per_question, dict):
            return None
        text_blob = []
        try:
            for _qtype, questions in per_question.items():
                if isinstance(questions, list):
                    for q in questions:
                        if isinstance(q, dict):
                            qt = q.get("question") or ""
                            text_blob.append(str(qt).lower())
        except Exception:
            pass
        joined = " ".join(text_blob)
        if not joined:
            return None
        # Search domain maps for hits
        for domain, cfg in self.domain_keyword_map.items():
            hits = 0
            for kw in cfg.get("keywords", []):
                if kw in joined:
                    hits += 1
            # Require several high-signal matches to avoid misclassification
            if hits >= 3:
                return domain
        return None

    def _normalize_grounded_topic(self, topic: str) -> str:
        """Normalize extracted topic labels while preserving acronyms."""
        cleaned = re.sub(r"\s+", " ", str(topic or "")).strip(" .,:;!?-")
        if not cleaned:
            return ""
        if cleaned.isupper() and 2 <= len(cleaned) <= 10:
            return cleaned
        return " ".join(word if word.isupper() else word.capitalize() for word in cleaned.split())

    def _extract_question_topic_candidates(self, question_text: str) -> List[str]:
        """Extract likely subtopics directly from the asked question text."""
        text = re.sub(r"\s+", " ", str(question_text or "")).strip()
        if not text:
            return []

        candidates: List[str] = []

        def _add(candidate: str) -> None:
            normalized = self._normalize_grounded_topic(candidate)
            if not normalized:
                return
            generic_labels = {
                "Question", "Assessment", "Finance Assessment", "General Assessment",
                "Mcq", "Short", "Long", "Coding", "Following"
            }
            if normalized in generic_labels:
                return
            if normalized not in candidates:
                candidates.append(normalized)

        for match in re.findall(r"`([^`]{2,60})`|'([^']{2,60})'|\"([^\"]{2,60})\"", text):
            _add(next((group for group in match if group), ""))

        for acronym in re.findall(r"\b[A-Z]{2,10}\b", text):
            _add(acronym)

        patterns = [
            r"\bwhat is ([A-Za-z][A-Za-z0-9/&\-\s]{2,60}?)(?: primarily| mainly| typically| generally| used| responsible| important| in | for | to |\?)",
            r"\bhow does ([A-Za-z][A-Za-z0-9/&\-\s]{2,60}?)(?: affect| impact| influence| change| improve| support| help|\?)",
            r"\bwhy is ([A-Za-z][A-Za-z0-9/&\-\s]{2,60}?)(?: important| useful| significant| relevant|\?)",
            r"\bpurpose of ([A-Za-z][A-Za-z0-9/&\-\s]{2,60}?)(?: in | for |\?)",
            r"\bmeaning of ([A-Za-z][A-Za-z0-9/&\-\s]{2,60}?)(?: in | for |\?)",
            r"\brole of ([A-Za-z][A-Za-z0-9/&\-\s]{2,60}?)(?: in | for |\?)",
            r"\bdifference between ([A-Za-z][A-Za-z0-9/&\-\s]{2,60}?)(?: and |\?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                _add(match.group(1))

        return candidates[:3]
    
    def _extract_topics_compact(self, assessment_results: Dict[str, Any]) -> List[str]:
        """Extract topics with minimal processing"""
        topics = []
        per_question = assessment_results.get('per_question', {})
        
        for qtype, questions in per_question.items():
            if not isinstance(questions, list):
                continue
            for question in questions:
                if not isinstance(question, dict):
                    continue
                question_text = question.get('question', '')[:200]
                extracted_topics = self._extract_question_topic_candidates(question_text)
                if not extracted_topics:
                    fallback_topic = self._extract_topic_name(question_text, qtype)
                    extracted_topics = [fallback_topic] if fallback_topic else []
                for topic in extracted_topics:
                    if topic and topic not in topics:
                        topics.append(topic)
                    if len(topics) >= 3:
                        return topics[:3]
        
        return topics[:3]  # Limit to 3 topics
    
    def _extract_topic_name(self, question_text: str, qtype: str) -> str:
        """Extract topic name efficiently"""
        text_lower = question_text.lower()
        
        # Quick topic mapping
        if 'verilog' in text_lower or 'hdl' in text_lower:
            if 'data type' in text_lower:
                return "Data Types"
            elif 'always' in text_lower:
                return "Always Blocks"
            elif 'assignment' in text_lower:
                return "Assignments"
            else:
                return "Verilog Basics"
        # Fashion/Draping domain
        elif ('draping' in text_lower or 'dress form' in text_lower or 'muslin' in text_lower or
              'bodice' in text_lower or 'grainline' in text_lower or 'bias' in text_lower):
            if 'grainline' in text_lower:
                return "Grainline Alignment"
            if 'dart' in text_lower:
                return "Darts and Shaping"
            if 'tuck' in text_lower:
                return "Tucks and Controlled Fullness"
            if 'ease' in text_lower or 'easing' in text_lower:
                return "Easing and Fullness Control"
            if 'hemline' in text_lower:
                return "Hemline Establishment"
            if 'true' in text_lower or 'truing' in text_lower or 'trueing' in text_lower:
                return "Truing Pattern Pieces"
            if 'sleeve' in text_lower:
                return "Sleeve Cap Shaping"
            if 'a-line' in text_lower or 'skirt' in text_lower:
                return "A-line Skirt Draping"
            if 'cowl' in text_lower or 'ruffle' in text_lower:
                return "Complex Draped Features"
            return "Draping Fundamentals"
        elif 'python' in text_lower:
            return "Python"
        elif 'algorithm' in text_lower:
            return "Algorithms"
        else:
            return qtype.replace('_', ' ').title()


class EnhancedNextStepsGenerator:
    """Generates high-quality, specific, and actionable next steps"""
    
    def __init__(self):
        try:
            from core.config import get_learning_resources
            self.learning_resources = get_learning_resources()
        except Exception:
            self.learning_resources = {}
    
    def generate_enhanced_next_steps(
        self, 
        insights: Dict[str, Any], 
        assessment_results: Dict[str, Any]
    ) -> List[str]:
        """Generate specific, actionable next steps based on performance analysis"""
        
        # Normalize score values in case they are None/str
        def _to_float(value: Any, default: float = 0.0) -> float:
            try:
                if value is None:
                    return default
                if isinstance(value, str):
                    cleaned = value.strip().replace('%', '')
                    return float(cleaned)
                return float(value)
            except (ValueError, TypeError):
                return default

        score = _to_float(insights.get("score", 0.0), 0.0)
        max_score = _to_float(insights.get("max_score", 0.0), 0.0)
        percentage = (score / max_score * 100) if max_score and max_score > 0 else 0.0
        topic = insights["topic"].lower()
        performance_pattern = insights["performance_pattern"]
        topics = insights["topics"]
        
        next_steps = []
        
        # Step 1: Immediate remediation based on performance level
        if percentage < 50:
            next_steps.extend(self._get_remediation_steps(topic, percentage))
        elif percentage < 70:
            next_steps.extend(self._get_improvement_steps(topic, percentage))
        else:
            next_steps.extend(self._get_advancement_steps(topic, percentage))
        
        # Step 2: Topic-specific recommendations
        topic_steps = self._get_topic_specific_steps(topics, topic)
        next_steps.extend(topic_steps)
        
        # Step 3: Performance pattern-based recommendations
        pattern_steps = self._get_pattern_based_steps(performance_pattern, topic)
        next_steps.extend(pattern_steps)
        
        # Step 4: Learning resource recommendations
        resource_steps = self._get_resource_recommendations(topic, percentage)
        next_steps.extend(resource_steps)
        
        # Step 5: Timeline-based recommendations
        timeline_steps = self._get_timeline_recommendations(percentage, topic)
        next_steps.extend(timeline_steps)
        
        # Optional LLM enrichment using learning_resources (non-breaking)
        try:
            band = (
                "remediation" if percentage < 50 else
                "improvement" if percentage < 70 else
                "advancement"
            )
            resources = self.learning_resources.get(topic, {}) if hasattr(self, "learning_resources") else {}
            context = {
                "topic": topic,
                "band": band,
                "seniority": insights.get("seniority", "practitioner"),
                "timeline": insights.get("timeline", "2-4 weeks"),
                "resources": resources,
            }
            # Run enrichment; handle both sync and async contexts
            try:
                import asyncio
                try:
                    _loop = asyncio.get_running_loop()
                    # In async context; skip synchronous enrichment to avoid blocking.
                    # Callers running in async can enrich separately if needed.
                    pass
                except RuntimeError:
                    # No running loop; safe to use asyncio.run for enrichment.
                    next_steps = asyncio.run(rewrite_steps_with_context(next_steps[:6], context))
            except Exception:
                # Fall back silently if enrichment fails
                pass
        except Exception:
            pass

        # Validate, remove duplicates and limit to 5 best steps
        topic_resources = self.learning_resources.get(topic, {}) if hasattr(self, "learning_resources") else {}
        validated = self._validate_steps(next_steps, topic, topic_resources)
        unique_steps = list(dict.fromkeys(validated))  # Preserve order while removing duplicates
        return unique_steps[:5]

    def _validate_steps(self, steps: List[str], topic: str, resources: Dict[str, Any]) -> List[str]:
        """Apply basic safety/quality checks to steps."""
        if not isinstance(steps, list):
            return []
        filtered: List[str] = []
        allow_links = bool(resources)
        for s in steps:
            if not isinstance(s, str) or not s.strip():
                continue
            if not allow_links and ("http://" in s.lower() or "https://" in s.lower()):
                continue
            filtered.append(s.strip())
        return filtered
    
    def _get_remediation_steps(self, topic: str, percentage: float) -> List[str]:
        """Steps for low performance (below 50%)"""
        return [
            f"Review {topic} fundamentals: Start with introductory course materials",
            f"Practice basics: Focus on core concepts before advancing",
            f"Seek help: Join {topic} learning community or find a mentor",
            f"Build foundation: Complete beginner-level exercises and tutorials",
        ]
    
    def _get_improvement_steps(self, topic: str, percentage: float) -> List[str]:
        """Steps for moderate performance (50-70%)"""
        return [
            f"Deepen {topic} knowledge: Take intermediate-level courses",
            f"Practice regularly: Solve progressively challenging problems",
            f"Apply concepts: Work on real-world {topic} projects",
            f"Join community: Participate in {topic} forums and discussions",
        ]
    
    def _get_advancement_steps(self, topic: str, percentage: float) -> List[str]:
        """Steps for good performance (70%+)"""
        return [
            f"Master advanced {topic}: Take expert-level courses and certifications",
            f"Specialize: Choose a {topic} specialization area",
            f"Contribute: Share knowledge through teaching or open source",
            f"Lead projects: Take on leadership roles in {topic} initiatives",
        ]
    
    def _get_topic_specific_steps(self, topics: List[str], main_topic: str) -> List[str]:
        """Generate steps based on specific topics identified"""
        steps = []
        for topic in topics:
            topic_lower = topic.lower()
            if 'data type' in topic_lower or 'data types' in topic_lower:
                steps.append(f"Master {main_topic} data types: Review and practice advanced type usage")
            elif 'assignments' in topic_lower or 'assignment' in topic_lower:
                steps.append(f"Practice {main_topic} assignments: Focus on best practices and pitfalls")
            elif 'always' in topic_lower:
                steps.append("Master control constructs: Practice event-driven and sequential patterns")
            elif 'algorithms' in topic_lower or 'algorithm' in topic_lower:
                steps.append("Strengthen algorithms: Focus on time complexity analysis and optimization techniques")
        return steps
    
    def _get_pattern_based_steps(self, performance_pattern: str, topic: str) -> List[str]:
        """Generate steps based on performance patterns"""
        steps = []
        
        if performance_pattern == "needs_work":
            steps.extend([
                f"Focus on fundamentals: Review basic {topic} concepts thoroughly",
                f"Practice consistently: Dedicate 30 minutes daily to {topic} practice",
                f"Seek guidance: Find a mentor or join {topic} study group"
            ])
        elif performance_pattern == "moderate":
            steps.extend([
                f"Identify weak areas: Focus on specific {topic} topics that need improvement",
                f"Practice strategically: Target your weakest {topic} concepts",
                f"Apply knowledge: Work on practical {topic} projects"
            ])
        elif performance_pattern == "strong":
            steps.extend([
                f"Challenge yourself: Take advanced {topic} courses or certifications",
                f"Share knowledge: Help others learn {topic} concepts",
                f"Explore specialization: Choose a {topic} specialization area"
            ])
        
        return steps
    
    def _get_resource_recommendations(self, topic: str, percentage: float) -> List[str]:
        """Generate specific resource recommendations"""
        steps = []
        
        # Get topic-specific resources
        topic_key = None
        for key in self.learning_resources.keys():
            if key in topic.lower():
                topic_key = key
                break
        
        if topic_key and topic_key in self.learning_resources:
            resources = self.learning_resources[topic_key]
            
            if percentage < 60:
                # Focus on courses and practice sites
                steps.append(f"Take structured course: {resources['courses'][0]}")
                steps.append(f"Practice daily: Use {resources['practice_sites'][0]} for hands-on practice")
            else:
                # Focus on advanced resources
                steps.append(f"Read advanced book: {resources['books'][0]}")
                steps.append(f"Use professional tools: Set up {resources['tools'][0]} for development")
        
        return steps
    
    def _get_timeline_recommendations(self, percentage: float, topic: str) -> List[str]:
        """Generate timeline-based recommendations"""
        steps = []
        
        if percentage < 50:
            steps.extend([
                f"Short-term (1-2 weeks): Complete {topic} fundamentals review",
                f"Medium-term (1 month): Take structured {topic} course",
                f"Long-term (2-3 months): Build practical {topic} project"
            ])
        elif percentage < 70:
            steps.extend([
                f"Short-term (1 week): Focus on weakest {topic} areas",
                f"Medium-term (2-3 weeks): Complete intermediate {topic} course",
                f"Long-term (1-2 months): Apply {topic} skills in real project"
            ])
        else:
            steps.extend([
                f"Short-term (1 week): Take advanced {topic} assessment",
                f"Medium-term (2-4 weeks): Complete expert-level {topic} course",
                f"Long-term (1-3 months): Contribute to {topic} community or open source"
            ])
        
        return steps


class TokenOptimizedReportPrompts:
    """Ultra-compact prompts for report generation"""
    
    @staticmethod
    def get_compact_report_prompt(insights: Dict[str, Any]) -> str:
        """Generate compact report prompt"""
        return f"""Generate report JSON:
Weighted proficiency score: {insights['score']}/{insights['max_score']}
Topic: {insights['topic']}
Pattern: {insights['performance_pattern']}
Sections: {insights['sections'].get('best', '')} best, {insights['sections'].get('worst', '')} worst
Topics: {', '.join(insights['topics'])}

JSON:
{{
    "total_score": {insights['score']},
    "max_score": {insights['max_score']},
    "summary": "2 sentences max",
    "performance_level": "Excellent|Good|Satisfactory|Needs Improvement",
    "topics": [{{"topic": "name", "strengths": ["s1"], "areas_for_improvement": ["w1"], "recommendations": ["r1"]}}],
    "feedback": "3 paragraphs max",
    "suggested_next_steps": ["5 specific steps"]
}}"""
    
    @staticmethod
    def get_enhanced_report_prompt(insights: Dict[str, Any], assessment_data: str) -> str:
        """Enhanced but still compact prompt with explicit JSON formatting"""
        assessment_topic = insights.get('topic', 'General Assessment')
        grounded_topics = insights.get('topics', [])
        allowed_topics = ", ".join(grounded_topics) if grounded_topics else "Only use topics directly evidenced by the provided questions."
        return f"""You are an expert career coach and technical evaluator. Generate a detailed, topic-specific performance report in JSON format only.

Assessment Topic: {assessment_topic}
Weighted proficiency score: {insights['score']}/{insights['max_score']} ({insights['performance_pattern']})
Key Data: {assessment_data[:1000]}...

CRITICAL REQUIREMENTS:
- Return ONLY valid JSON. No explanations, no markdown, no additional text.
- Use proper spelling and grammar throughout.
- Write professionally and clearly.
- Be SPECIFIC to the assessment topic ({assessment_topic}) - reference actual subtopics, concepts, and principles from the questions.
- Use ONLY question-backed topics/subtopics supported by the provided assessment data. Do NOT invent adjacent domain topics that do not appear in the questions or evaluated answers.
- Topics should be SUBJECT-MATTER specific (e.g., "Prophet: Time Series Decomposition", "AXIS: Actuarial Valuation") NOT question-type generic (e.g., NOT "Multiple Choice Questions").
- Analyze the questions and answers to identify specific subtopics and principles being tested.
- SECOND-PERSON VOICE: Write all sections in second-person (use "Your", "You", "Your performance", etc.) - NEVER use "The Candidate" or third-person references. Sections should start with "Your" not "The Candidate."
- Allowed grounded topics: {allowed_topics}

Required JSON structure:
{{
    "total_score": {insights['score']},
    "max_score": {insights['max_score']},
    "summary": "2-3 sentence overview referencing specific subtopics from {assessment_topic} and performance on them",
    "performance_level": "Excellent|Good|Satisfactory|Needs Improvement",
    "topics": [
        {{
            "topic": "Subject-matter subtopic name (e.g., 'Prophet: Decomposable Model Components', 'AXIS: Liability Valuation')",
            "strengths": ["Specific strength tied to this subtopic with evidence from questions"],
            "areas_for_improvement": ["Specific subtopic/principle that was misunderstood or needs work"],
            "recommendations": ["Actionable recommendation specific to this subtopic"]
        }}
    ],
    "feedback": "3-4 paragraph analysis explicitly mentioning subtopics, principles, and concepts from {assessment_topic}",
    "suggested_next_steps": ["Topic-specific step 1", "Topic-specific step 2", "Topic-specific step 3", "Topic-specific step 4", "Topic-specific step 5"],
    "subtopic_breakdown": [{{"name": "<subtopic>", "score": 0.0, "notes": "<reason>"}}],
    "misconceptions": [{{"subtopic": "<subtopic>", "message": "<brief misconception>", "evidence": ["question_id"]}}],
    "rubric_scores": {{"short_answer": {{"clarity": 0.0, "correctness": 0.0, "terminology": 0.0}}, "long_answer": {{"structure": 0.0, "integration": 0.0, "specificity": 0.0}}}},
    "resources": [{{"subtopic": "<subtopic>", "links": ["https://valid-link.com"]}}]
}}

Focus on: {', '.join(insights.get('topics', []))}
Analyze the questions and answers to identify specific subtopics within {assessment_topic}. Make all feedback topic-specific, not generic."""


class ReportTokenBudgetManager:
    """Manages token usage for report generation"""
    
    def __init__(self, max_tokens: int = 3000):
        self.max_tokens = max_tokens
        self.current_budget = max_tokens
    
    def optimize_assessment_data(self, assessment_results: Dict[str, Any]) -> str:
        """Optimize assessment data for token efficiency"""
        
        # Extract only essential data
        essential_data = {
            "total_score": assessment_results.get('total_score', 0),
            "max_score": assessment_results.get('max_score', 100),
            "assessment_topic": assessment_results.get('assessment_topic', '')[:50],
            "section_scores": assessment_results.get('section_scores', {}),
            "question_scores": self._sample_question_scores(assessment_results.get('question_scores', {})),
            "per_question": self._sample_per_question(assessment_results.get('per_question', {}))
        }
        
        # Convert to compact string
        data_str = json.dumps(essential_data, separators=(',', ':'))
        
        # Truncate if too long
        if len(data_str) > 1000:
            data_str = data_str[:997] + "..."
        
        return data_str
    
    def _sample_question_scores(self, question_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Sample question scores to reduce token usage. Values may be int or 'score/max' strings."""
        if len(question_scores) <= 5:
            return question_scores

        # Take first 3 and last 2 for pattern analysis
        items = list(question_scores.items())
        sampled = dict(items[:3] + items[-2:])
        return sampled
    
    def _sample_per_question(self, per_question: Dict[str, Any]) -> Dict[str, Any]:
        """Sample per_question data efficiently"""
        sampled = {}
        
        for qtype, questions in per_question.items():
            if isinstance(questions, list):
                # Take first 2 questions as representative
                sampled[qtype] = questions[:2]
            else:
                sampled[qtype] = questions
        
        return sampled
    
    def can_afford_report(self, prompt: str) -> bool:
        """Check if we can afford this report generation"""
        estimated_tokens = len(prompt) // 4
        return estimated_tokens <= self.current_budget
    
    def optimize_prompt_for_budget(self, prompt: str) -> str:
        """Optimize prompt to fit within token budget"""
        estimated_tokens = len(prompt) // 4
        
        if estimated_tokens <= self.current_budget:
            return prompt
        
        # Calculate reduction factor
        reduction_factor = self.current_budget / estimated_tokens
        
        # Truncate proportionally
        target_length = int(len(prompt) * reduction_factor * 0.9)
        return prompt[:target_length] + "..."


class TokenOptimizedReportGenerator:
    """Enhanced report generator with token optimization"""
    
    def __init__(self):
        self.analyzer = TokenOptimizedReportAnalyzer()
        self.budget_manager = ReportTokenBudgetManager()
        self.prompts = TokenOptimizedReportPrompts()
        self.next_steps_generator = EnhancedNextStepsGenerator()

    def _is_topic_supported(self, topic_name: str, grounded_topics: List[str]) -> bool:
        """Return True when a report topic overlaps with evidence-backed topics."""
        topic_lower = str(topic_name or "").strip().lower()
        if not topic_lower:
            return False
        if not grounded_topics:
            return True

        topic_tokens = {token for token in re.findall(r"[a-z0-9]+", topic_lower) if len(token) > 2}
        for grounded_topic in grounded_topics:
            grounded_lower = str(grounded_topic or "").strip().lower()
            if not grounded_lower:
                continue
            if topic_lower == grounded_lower or topic_lower in grounded_lower or grounded_lower in topic_lower:
                return True
            grounded_tokens = {token for token in re.findall(r"[a-z0-9]+", grounded_lower) if len(token) > 2}
            if topic_tokens and grounded_tokens and topic_tokens.intersection(grounded_tokens):
                return True
        return False

    def _mentions_any_topic(self, text: str, topics: List[str]) -> bool:
        """Return True if text mentions any of the provided topics."""
        text_lower = str(text or "").lower()
        if not text_lower:
            return False
        for topic in topics:
            topic_lower = str(topic or "").strip().lower()
            if not topic_lower:
                continue
            if topic_lower in text_lower:
                return True
            topic_tokens = [token for token in re.findall(r"[a-z0-9]+", topic_lower) if len(token) > 2]
            if topic_tokens and all(token in text_lower for token in topic_tokens):
                return True
        return False

    def _strip_unsupported_topic_mentions(self, text: str, unsupported_topics: List[str]) -> str:
        """Remove unsupported topic mentions from free text fields."""
        cleaned = str(text or "")
        if not cleaned or not unsupported_topics:
            return cleaned

        lines = cleaned.splitlines()
        kept_lines = [line for line in lines if not self._mentions_any_topic(line, unsupported_topics)]
        if kept_lines:
            cleaned = "\n".join(kept_lines)
        for topic in unsupported_topics:
            cleaned = re.sub(rf"\b{re.escape(topic)}\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+,", ",", cleaned)
        cleaned = re.sub(r",\s*,", ", ", cleaned)
        cleaned = re.sub(r"\(\s*\)", "", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip(" ,.;:-")

    def _filter_report_to_grounded_topics(self, report: Dict[str, Any], assessment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Remove unsupported report topics and actions that cannot be traced to question evidence."""
        if not isinstance(report, dict):
            return report

        grounded_topics = self.analyzer._extract_topics_compact(assessment_results)
        if not grounded_topics:
            return report

        filtered_report = dict(report)
        original_topics = report.get("topics", []) if isinstance(report.get("topics"), list) else []
        kept_topics = []
        removed_topics = []

        for topic_entry in original_topics:
            if not isinstance(topic_entry, dict):
                continue
            topic_name = str(topic_entry.get("topic", "")).strip()
            if self._is_topic_supported(topic_name, grounded_topics):
                kept_topics.append(topic_entry)
            elif topic_name:
                removed_topics.append(topic_name)

        if kept_topics:
            filtered_report["topics"] = kept_topics

        if removed_topics:
            for field in ("summary", "feedback"):
                if isinstance(filtered_report.get(field), str):
                    filtered_report[field] = self._strip_unsupported_topic_mentions(filtered_report[field], removed_topics)

            if isinstance(filtered_report.get("suggested_next_steps"), list):
                filtered_report["suggested_next_steps"] = [
                    step for step in filtered_report["suggested_next_steps"]
                    if isinstance(step, str) and not self._mentions_any_topic(step, removed_topics)
                ]

            for list_field, item_key in (("subtopic_breakdown", "name"), ("misconceptions", "subtopic"), ("resources", "subtopic")):
                items = filtered_report.get(list_field)
                if isinstance(items, list):
                    filtered_report[list_field] = [
                        item for item in items
                        if not isinstance(item, dict) or not self._mentions_any_topic(item.get(item_key, ""), removed_topics)
                    ]

        return filtered_report
    
    async def generate_enhanced_report(
        self,
        assessment_results: Dict[str, Any],
        candidate_name: str,
        tenant_id: str
    ) -> Dict[str, Any]:
        """Generate enhanced report with token optimization"""
        
        # Step 1: Extract key insights efficiently (offloaded — heavy regex per question)
        insights = await asyncio.to_thread(self.analyzer.extract_key_insights, assessment_results)
        
        # Step 2: Optimize assessment data for token efficiency
        optimized_data = self.budget_manager.optimize_assessment_data(assessment_results)
        
        # Step 3: Create compact prompt
        prompt = self.prompts.get_enhanced_report_prompt(insights, optimized_data)
        
        # Step 4: Check and optimize for token budget
        if not self.budget_manager.can_afford_report(prompt):
            prompt = self.budget_manager.optimize_prompt_for_budget(prompt)
            log.debug(f"Optimized report prompt for token budget: {len(prompt)} chars")
        
        # Step 5: Generate report with LLM
        try:
            llm_response = await invoke_llm(
                prompt=prompt,
                task_type="report_generation",
                agent_name="report_generator"
            )
            
            if llm_response:
                content = getattr(llm_response, "content", str(llm_response)).strip()
                log.debug(f"📝 Raw LLM response: {content[:200]}...")
                
                # Parse JSON response with robust extraction
                try:
                    # Try to extract JSON from the response
                    json_content = self._extract_json_from_response(content)
                    if json_content:
                        report = json.loads(json_content)
                        
                        # Validate structure
                        if self._validate_report_structure(report):
                            log.info(f"✅ Successfully parsed JSON report with {len(report.get('topics', []))} topics")
                            
                            # Ensure new optional fields are always present with safe defaults
                            if "subtopic_breakdown" not in report or not isinstance(report.get("subtopic_breakdown"), list):
                                report["subtopic_breakdown"] = []
                            if "misconceptions" not in report or not isinstance(report.get("misconceptions"), list):
                                report["misconceptions"] = []
                            if "rubric_scores" not in report or not isinstance(report.get("rubric_scores"), dict):
                                report["rubric_scores"] = {}
                            if "resources" not in report or not isinstance(report.get("resources"), list):
                                report["resources"] = []
                            else:
                                # Filter invalid links from resources
                                report["resources"] = filter_valid_resources(report["resources"])
                            
                            # Apply spell checking and grammar correction
                            log.debug("🔍 Applying spell checking and grammar correction...")
                            spell_checker = ReportSpellChecker()
                            corrected_report = spell_checker.validate_report_content(report)
                            
                            return self._filter_report_to_grounded_topics(corrected_report, assessment_results)
                        else:
                            log.warning("Invalid report structure, using fallback")
                            return self._generate_fallback_report(insights, assessment_results)
                    else:
                        log.warning("No valid JSON found in response, using fallback")
                        return self._generate_fallback_report(insights, assessment_results)
                        
                except json.JSONDecodeError as e:
                    log.warning(f"Failed to parse JSON response: {e}, using fallback")
                    return self._generate_fallback_report(insights, assessment_results)
            
        except Exception as e:
            log.error(f"Enhanced report generation failed: {e}")
            return self._generate_fallback_report(insights, assessment_results)
        
        return self._generate_fallback_report(insights, assessment_results)
    
    def _extract_json_from_response(self, content: str) -> Optional[str]:
        """Extract JSON from LLM response, handling various formats."""
        import re
        
        # Remove any markdown formatting
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*$', '', content)
        content = content.strip()
        
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                # Validate it's valid JSON
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass
        
        # Try direct parsing
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError:
            pass
        
        return None

    def _validate_report_structure(self, report: Dict[str, Any]) -> bool:
        """Validate report has required structure (scores optional for generic tests)"""
        required_fields = ["summary", "performance_level", "topics", "feedback", "suggested_next_steps"]
        return all(field in report for field in required_fields)
    
    def _generate_fallback_report(self, insights: Dict[str, Any], assessment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback report with enhanced content"""
        
        score = insights["score"]
        max_score = insights["max_score"]
        percentage = (score / max_score * 100) if max_score > 0 else 0
        topic = insights["topic"]
        weighted_score_text = _build_weighted_proficiency_text(score, max_score, percentage)
        count_sentence = _build_assessment_count_sentence(assessment_results)
        score_explanation = _normalize_sentence(assessment_results.get("score_explanation"))
        attempted_questions = _to_optional_int(assessment_results.get("attempted_questions"))
        fully_correct_questions = _to_optional_int(assessment_results.get("fully_correct_questions"))
        partially_correct_questions = _to_optional_int(assessment_results.get("partially_correct_questions")) or 0
        incorrect_questions = _to_optional_int(assessment_results.get("incorrect_questions"))
        no_gap_assessment = (
            attempted_questions is not None
            and attempted_questions > 0
            and fully_correct_questions == attempted_questions
            and partially_correct_questions == 0
            and (incorrect_questions or 0) == 0
        )
        
        # Enhanced summary
        if percentage >= 80:
            level = "Excellent"
            summary = f"Demonstrated excellent mastery of {topic} concepts with a {weighted_score_text}."
        elif percentage >= 65:
            level = "Good"
            summary = f"Showed solid understanding of {topic} with a {weighted_score_text}, indicating good foundational knowledge."
        elif percentage >= 50:
            level = "Satisfactory"
            summary = f"Achieved a satisfactory {weighted_score_text} in {topic}, showing basic comprehension with room for improvement."
        else:
            level = "Needs Improvement"
            summary = f"Performance indicates a need for focused study in {topic} fundamentals based on a {weighted_score_text}."
        if count_sentence:
            summary = f"{summary} {count_sentence}"
        
        # Enhanced topics
        topics = []
        for topic_name in insights["topics"]:
            topics.append({
                "topic": topic_name,
                "strengths": [f"Demonstrated understanding of {topic_name} concepts"],
                "areas_for_improvement": [] if no_gap_assessment else [f"Practice more {topic_name} problems"],
                "recommendations": (
                    [f"Apply {topic_name} concepts in more advanced scenarios"]
                    if no_gap_assessment
                    else [f"Review {topic_name} fundamentals and practice"]
                ),
            })
        
        # Enhanced feedback
        feedback = f"Your {topic} assessment reflects a {weighted_score_text} and "
        if percentage >= 80:
            feedback += "excellent command of the material. You've mastered the core concepts and are ready for advanced topics."
        elif percentage >= 65:
            feedback += "solid foundational knowledge. Focus on strengthening weaker areas to achieve mastery."
        elif percentage >= 50:
            feedback += "basic understanding with significant gaps. Dedicated study and practice will help build stronger foundations."
        else:
            feedback += "a need for comprehensive review. Start with fundamental concepts before advancing to complex topics."
        if count_sentence:
            feedback = f"{feedback} {count_sentence}"
        if score_explanation:
            feedback = f"{feedback} {score_explanation}"
        
        # Enhanced next steps using the enhanced generator
        next_steps = self.next_steps_generator.generate_enhanced_next_steps(
            insights, assessment_results
        )
        
        fallback_resources = [{"subtopic": "<subtopic>", "links": ["https://facebook.github.io/prophet/docs/quick_start.html"]}]
        
        fallback_report = {
            "total_score": score,
            "max_score": max_score,
            "summary": summary,
            "performance_level": level,
            "topics": topics,
            "feedback": feedback,
            "suggested_next_steps": next_steps,
            "subtopic_breakdown": [] if no_gap_assessment else [{"name": "<subtopic>", "score": 0.0, "notes": "<reason>"}],
            "misconceptions": [] if no_gap_assessment else [{"subtopic": "<subtopic>", "message": "<brief misconception>", "evidence": ["mcq[2]"]}],
            "rubric_scores": {"short_answer": {"clarity": 0.0, "correctness": 0.0, "terminology": 0.0}, "long_answer": {"structure": 0.0, "integration": 0.0, "specificity": 0.0}},
            "resources": filter_valid_resources(fallback_resources)
        }
        return self._filter_report_to_grounded_topics(fallback_report, assessment_results)


# ============================================================================
# 1. PYDANTIC MODELS FOR VALIDATION
# ============================================================================

class Priority(str, Enum):
    """Priority levels for recommendations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TopicFeedback(BaseModel):
    """Structured feedback for each topic."""
    topic: str = Field(..., min_length=1, max_length=100)
    strengths: List[str] = Field(default_factory=list, max_items=10)
    areas_for_improvement: List[str] = Field(default_factory=list, max_items=10)
    recommendations: List[str] = Field(default_factory=list, max_items=10)

    @validator('strengths', 'areas_for_improvement', 'recommendations', pre=True, each_item=True)
    def validate_items(cls, v):
        if isinstance(v, str):
            return v[:500]  # Cap string length
        return v


class ReportResponse(BaseModel):
    """Structured report response model."""
    total_score: float = Field(..., ge=0)
    max_score: float = Field(..., ge=0)
    summary: str = Field(..., min_length=1, max_length=4000)
    performance_level: str = Field(..., description="A qualitative assessment of the score (e.g., Needs Improvement, Good, Excellent).")
    topics: List[TopicFeedback] = Field(default_factory=list, max_items=50)
    feedback: str = Field(..., min_length=1, max_length=6000)
    suggested_next_steps: List[str] = Field(default_factory=list, max_items=5, description="Actionable next steps for the candidate.")
    # New optional, topic-specific fields. These default to empty so existing
    # callers remain backward compatible when data is unavailable.
    subtopic_breakdown: List[Dict[str, Any]] = Field(default_factory=list, description="Per-subtopic scores/notes. e.g., [{name, score, notes}]")
    misconceptions: List[Dict[str, Any]] = Field(default_factory=list, description="Detected misconceptions with evidence.")
    rubric_scores: Dict[str, Any] = Field(default_factory=dict, description="Section rubric scores, e.g., short_answer/long_answer criteria.")
    resources: List[Dict[str, Any]] = Field(default_factory=list, description="Topic-specific learning resources with valid links.")

    @validator('total_score', 'max_score')
    def validate_scores(cls, v):
        return round(v, 2)


# Security, circuit breaker, and rate limiting functionality moved to centralized middleware





def validate_size(data: Any, max_size: int) -> bool:
    """Check if data size is within limits. Uses fast estimate first."""
    import sys
    rough = sys.getsizeof(str(data)) if not isinstance(data, str) else len(data)
    if rough > max_size * 3:
        return False
    if rough < max_size // 2:
        return True
    return len(json.dumps(data)) <= max_size



# ============================================================================
# 5. CACHE
# ============================================================================

class TenantAwareReportCache:
    """LRU cache with TTL for report generation."""

    def __init__(self, max_entries: int = 1000, ttl_minutes: int = 30):
        self.max_entries = max_entries
        self.ttl = timedelta(minutes=ttl_minutes)
        self.cache: Dict[str, tuple[Any, datetime]] = {}
        self.access_order: deque = deque(maxlen=max_entries)
        self._lock = asyncio.Lock()

    def _generate_key(self, tenant_id: str, data: Dict[str, Any]) -> str:
        """Generate cache key."""
        key_data = {
            "tenant_id": tenant_id,
            "assessment_results": data.get("assessment_results"),
            "candidate_name": data.get("candidate_name")
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    async def get(self, tenant_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached result if fresh."""
        async with self._lock:
            key = self._generate_key(tenant_id, data)
            if key in self.cache:
                result, timestamp = self.cache[key]
                if datetime.now() - timestamp < self.ttl:
                    # Move to end (most recently used)
                    self.access_order.remove(key)
                    self.access_order.append(key)
                    return result
                else:
                    del self.cache[key]
                    self.access_order.remove(key)
            return None

    async def set(self, tenant_id: str, data: Dict[str, Any], result: Dict[str, Any]):
        """Cache result."""
        async with self._lock:
            key = self._generate_key(tenant_id, data)

            # LRU eviction if at capacity
            if len(self.cache) >= self.max_entries and key not in self.cache:
                oldest = self.access_order.popleft()
                del self.cache[oldest]

            self.cache[key] = (result, datetime.now())
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)


# ============================================================================
# 6. METRICS
# ============================================================================

class PerformanceMetrics:
    """Track performance metrics."""

    def __init__(self, window_size: int = 5000):
        self.window_size = window_size
        self.response_times: deque = deque(maxlen=window_size)
        self.confidence_scores: deque = deque(maxlen=window_size)
        self.method_counts: Dict[str, int] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.timeouts = 0
        self.circuit_opens = 0
        self._lock = asyncio.Lock()

    async def record_response(self, time_seconds: float, confidence: float, method: str):
        """Record response metrics."""
        async with self._lock:
            self.response_times.append(time_seconds)
            self.confidence_scores.append(confidence)
            self.method_counts[method] = self.method_counts.get(method, 0) + 1

    async def get_p95_response_time(self) -> float:
        """Get 95th percentile response time."""
        async with self._lock:
            if not self.response_times:
                return 0.0
            sorted_times = sorted(self.response_times)
            idx = int(len(sorted_times) * 0.95)
            return sorted_times[idx]


# ============================================================================
# 7. JSON EXTRACTION
# ============================================================================

def extract_json_robust(text: str) -> Dict[str, Any]:
    """Extract JSON with multiple fallback strategies."""
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Find JSON block
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: Balanced brace extraction
    start = text.find('{')
    if start != -1:
        brace_count = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break

    # Fallback: return structured default
    return {
        "total_score": 0,
        "max_score": 0,
        "summary": text.strip()[:500] if text else "Report generation failed.",
        "topics": [],
        "feedback": "Unable to generate detailed feedback."
    }


# ============================================================================
# 8. MAIN REPORT GENERATOR
# ============================================================================

class ReportGeneratorAgent:
    """Production-ready report generator with all safety features."""

    def __init__(self):
        # Circuit breaker, rate limiter, and security validation now handled by centralized middleware
        self.cache = TenantAwareReportCache()
        self.metrics = PerformanceMetrics()

    @traceable(name="generate_report_with_llm")
    async def generate_report_with_llm(
        self,
        assessment_results: Dict[str, Any],
        candidate_name: str,
        tenant_id: str
    ) -> Dict[str, Any]:
        """Generate enhanced report using token-optimized LLM with fallback to original method."""
        candidate_name = redact_pii(candidate_name)

        # Try enhanced token-optimized report generation first
        try:
            log.info("🚀 Attempting enhanced token-optimized report generation...")
            enhanced_generator = TokenOptimizedReportGenerator()
            result = await enhanced_generator.generate_enhanced_report(
                assessment_results, candidate_name, tenant_id
            )
            
            log.debug(f"📊 Enhanced generator result: {result}")
            
            # Validate result has topics (non-empty) and proper structure
            if (result.get("topics") and len(result["topics"]) > 0 and 
                result.get("summary") and len(result["summary"]) > 50):
                log.info(f"✅ Enhanced report generated successfully with {len(result.get('topics', []))} topics")
                return result
            else:
                log.warning("⚠️ Enhanced report missing topics, falling back to original method")
                log.debug(f"   Topics: {result.get('topics', [])}")
                log.debug(f"   Summary length: {len(result.get('summary', ''))}")
                
        except Exception as e:
            log.warning(f"⚠️ Enhanced report generation failed: {e}, falling back to original method")
            import traceback
            log.debug(f"   Traceback: {traceback.format_exc()}")
        
        # Fallback to original enhanced method
        log.info("🔄 Falling back to original enhanced LLM method...")
        return await self._original_generate_report_with_llm(
            assessment_results, candidate_name, tenant_id
        )

    async def _original_generate_report_with_llm(
        self,
        assessment_results: Dict[str, Any],
        candidate_name: str,
        tenant_id: str
    ) -> Dict[str, Any]:
        """Enhanced report generation method with improved prompts and content quality."""
        candidate_name = redact_pii(candidate_name)
        log.info(f"🎯 Original LLM method: Generating report for {candidate_name}")

        # Extract experience data for context-aware reporting
        experience_data = await self._extract_experience_context(assessment_results, None)
        log.debug(f"📊 Experience data extracted: {experience_data}")
        
        # Generate adaptive, context-aware prompt
        prompt = self._generate_adaptive_prompt_variations(assessment_results, candidate_name, experience_data)
        log.debug(f"📝 Generated prompt length: {len(prompt)}")

        if not validate_size(prompt, 30000):  # MAX_PROMPT_SIZE
            return self._fallback_report("Prompt size exceeded")

        for attempt in range(3):
            try:
                # Enforce structured output for report generation
                from typing import List as _List, Optional as _Optional
                from settings import settings as _settings

                class ReportStructuredResponse(BaseModel):
                    total_score: int
                    max_score: int
                    summary: str
                    performance_level: str
                    topics: _List[dict] = Field(default_factory=list)  # Changed from _List[str] to _List[dict] to match prompt
                    feedback: str
                    suggested_next_steps: _List[str] = Field(default_factory=list)
                    # Optional topic-specific fields requested by caller
                    subtopic_breakdown: _List[dict] = Field(default_factory=list)
                    misconceptions: _List[dict] = Field(default_factory=list)
                    rubric_scores: dict = Field(default_factory=dict)
                    resources: _List[dict] = Field(default_factory=list)

                log.debug(f"🤖 Calling LLM (attempt {attempt + 1}/3)...")
                structured_result: ReportStructuredResponse = await invoke_structured_llm(
                    prompt,
                    ReportStructuredResponse,
                    task_type=TaskType.REPORT_GENERATION,
                    preferred_model=_settings.GEMINI_MODEL,
                    agent_name="report_generator",
                    temperature=0.3,
                    max_output_tokens=4000,
                    timeout=60.0,
                    raise_on_fallback=False,
                )
                
                log.debug(f"✅ LLM call successful, processing result...")
                result = structured_result.model_dump()
                log.debug(f"📊 Raw LLM result: {result}")
                
                # Normalize and validate field types before processing
                # Ensure string fields are strings, not lists
                if "summary" in result and not isinstance(result["summary"], str):
                    result["summary"] = str(result["summary"]) if result["summary"] else ""
                if "feedback" in result and not isinstance(result["feedback"], str):
                    result["feedback"] = str(result["feedback"]) if result["feedback"] else ""
                if "performance_level" in result and not isinstance(result["performance_level"], str):
                    result["performance_level"] = str(result["performance_level"]) if result["performance_level"] else "Satisfactory"
                
                # Ensure list fields are lists
                if "topics" in result and not isinstance(result["topics"], list):
                    result["topics"] = []
                if "suggested_next_steps" in result and not isinstance(result["suggested_next_steps"], list):
                    result["suggested_next_steps"] = []
                
                # Ensure new optional fields are correct types
                if "subtopic_breakdown" in result and not isinstance(result["subtopic_breakdown"], list):
                    result["subtopic_breakdown"] = []
                if "misconceptions" in result and not isinstance(result["misconceptions"], list):
                    result["misconceptions"] = []
                if "rubric_scores" in result and not isinstance(result["rubric_scores"], dict):
                    result["rubric_scores"] = {}
                if "resources" in result and not isinstance(result["resources"], list):
                    result["resources"] = []
                
                # Validate content quality
                quality_score = self._validate_report_quality(result, assessment_results)
                log.debug(f"📈 Quality score: {quality_score:.2f}")
                
                if quality_score < 0.7:  # Quality threshold
                    log.warning(f"⚠️ Report quality below threshold ({quality_score:.2f}), regenerating...")
                    if attempt < 2:  # Try again with improved prompt
                        prompt = self._enhance_prompt_for_quality(prompt, quality_score)
                        continue
                
                # Add quality metrics to result
                result["quality_score"] = quality_score
                result["content_validation"] = self._get_content_validation_summary(result)
                
                # Enrich content with additional insights and resources
                log.debug("🎨 Enriching report content...")
                enriched_result = await self._enrich_report_content(result, assessment_results, experience_data)
                
                # Apply spell checking and grammar correction
                log.debug("🔍 Applying spell checking and grammar correction...")
                spell_checker = ReportSpellChecker()
                final_result = spell_checker.validate_report_content(enriched_result)
                
                log.debug(f"✨ Final corrected result: {final_result}")
                # Ensure new optional fields are always present with safe defaults
                try:
                    if not isinstance(final_result.get("subtopic_breakdown"), list):
                        final_result["subtopic_breakdown"] = []
                    if not isinstance(final_result.get("misconceptions"), list):
                        final_result["misconceptions"] = []
                    if not isinstance(final_result.get("rubric_scores"), dict):
                        final_result["rubric_scores"] = {}
                    if not isinstance(final_result.get("resources"), list):
                        final_result["resources"] = []
                    else:
                        # Filter invalid links from resources
                        final_result["resources"] = filter_valid_resources(final_result["resources"])
                except Exception:
                    # On any failure, enforce defaults
                    final_result.update({
                        "subtopic_breakdown": final_result.get("subtopic_breakdown") or [],
                        "misconceptions": final_result.get("misconceptions") or [],
                        "rubric_scores": final_result.get("rubric_scores") or {},
                        "resources": filter_valid_resources(final_result.get("resources") or [])
                    })

                grounding_filter = TokenOptimizedReportGenerator()
                return grounding_filter._filter_report_to_grounded_topics(final_result, assessment_results)
            except asyncio.TimeoutError:
                await self.metrics.record_response(30, 0, "timeout")
                if attempt == 2:
                    # Circuit breaker failure recording now handled by centralized middleware
                    return self._fallback_report("LLM timeout")
                backoff = (0.4 * (2 ** attempt)) + (asyncio.create_task(asyncio.sleep(0)).get_coro().cr_frame.f_lasti % 100) / 100
                await asyncio.sleep(backoff)
            except Exception as e:
                # Circuit breaker failure recording now handled by centralized middleware
                if attempt == 2:
                    return self._fallback_report(f"LLM error: {str(e)[:100]}")
                await asyncio.sleep(0.4 * (2 ** attempt))
        return self._fallback_report("Max retries exceeded")

    async def _detect_domain_with_llm(self, job_title: str, company: str = "", responsibilities: List[str] = None) -> Dict[str, Any]:
        """Use LLM to intelligently detect domain and key skills from job information."""
        if not job_title or not isinstance(job_title, str):
            return {"domain": None, "skills": []}
        
        # Build context from available information
        context_parts = [f"Job Title: {job_title}"]
        if company:
            context_parts.append(f"Company: {company}")
        if responsibilities and isinstance(responsibilities, list):
            resp_text = "; ".join(responsibilities[:3])  # Top 3 responsibilities
            context_parts.append(f"Key Responsibilities: {resp_text}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""
Analyze the following job information and identify the professional domain and key skills.

{context}

Identify:
1. The primary professional domain/industry (e.g., technology, healthcare, education, beauty_services, hospitality, retail, finance, etc.)
2. Top 3-5 key skills demonstrated in this role

OUTPUT FORMAT (JSON):
{{
    "domain": "primary_domain_name",
    "skills": ["skill1", "skill2", "skill3"]
}}

Examples:
- "Hair Stylist" → {{"domain": "beauty_services", "skills": ["customer service", "styling", "client relations"]}}
- "Data Scientist" → {{"domain": "data_analytics", "skills": ["data analysis", "machine learning", "statistics"]}}
- "Registered Nurse" → {{"domain": "healthcare", "skills": ["patient care", "medical knowledge", "clinical skills"]}}
- "Elementary Teacher" → {{"domain": "education", "skills": ["teaching", "curriculum development", "student engagement"]}}

Provide the domain and skills for the given job information.
"""
        
        try:
            response = await invoke_llm(
                prompt=prompt,
                task_type=TaskType.CLASSIFICATION,  # Classifying job into domain
                agent_name="report_generator"
            )
            content = getattr(response, "content", str(response)).strip()
            
            if content:
                result = extract_json_robust(content)
                if result and isinstance(result, dict):
                    domain = result.get("domain", "").strip() if result.get("domain") else None
                    skills = result.get("skills", [])
                    if isinstance(skills, list):
                        skills = [str(s).strip() for s in skills if s][:5]  # Limit to 5
                    else:
                        skills = []
                    return {"domain": domain, "skills": skills}
        except Exception as e:
            log.warning(f"LLM domain detection failed: {e}")
        
        # Fallback to None if LLM fails
        return {"domain": None, "skills": []}
    
    async def _determine_performance_level_with_llm(
        self,
        score: float,
        max_score: float,
        test_type: str,
        test_difficulty: str = "medium",
        domain: str = None,
        experience_level: str = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Use LLM to determine performance level based on context, not just fixed thresholds.
        
        Args:
            score: Candidate's score
            max_score: Maximum possible score
            test_type: Type of test (e.g., "technical", "communication", "coding")
            test_difficulty: Estimated difficulty ("easy", "medium", "hard")
            domain: Candidate's professional domain
            experience_level: Candidate's experience level ("entry", "mid", "senior")
            context: Additional context about the assessment
            
        Returns:
            {
                "performance_level": "Excellent|Good|Satisfactory|Needs Development",
                "reasoning": "Explanation of the assessment",
                "percentile_estimate": 85 (estimated percentile)
            }
        """
        percentage = (score / max_score * 100) if max_score and max_score > 0 else 0.0
        
        # Build context for LLM
        context_parts = [
            f"Weighted proficiency score: {score}/{max_score} ({percentage:.1f}% normalized)",
            f"Test Type: {test_type}",
            f"Test Difficulty: {test_difficulty}"
        ]
        
        if domain:
            context_parts.append(f"Candidate Domain: {domain}")
        if experience_level:
            context_parts.append(f"Experience Level: {experience_level}")
        if context:
            if context.get("total_questions"):
                context_parts.append(f"Total Questions: {context['total_questions']}")
            if context.get("topics"):
                context_parts.append(f"Topics Covered: {', '.join(context['topics'][:3])}")
        
        context_str = "\n".join(context_parts)
        
        prompt = f"""
Determine the appropriate performance level for this assessment, considering ALL context factors:

{context_str}

**Important Context Considerations:**
1. Test Difficulty: A {test_difficulty} test requires different benchmarks than easy/hard tests
2. Experience Level: Expected performance varies by experience (entry vs senior)
3. Domain: Industry standards differ (healthcare vs tech vs hospitality)
4. Test Type: Technical tests vs behavioral tests have different passing standards

**Performance Levels:**
- **Excellent**: Outstanding performance, exceeds expectations
- **Good**: Solid performance, meets expectations well
- **Satisfactory**: Acceptable performance, room for improvement
- **Needs Development**: Below expectations, requires significant improvement

OUTPUT FORMAT (JSON):
{{
    "performance_level": "Excellent|Good|Satisfactory|Needs Development",
    "reasoning": "Brief explanation considering context",
    "percentile_estimate": 85
}}

Examples:
- 85% on a HARD coding test for MID-level → "Excellent" (high difficulty adjusted)
- 85% on an EASY communication test for ENTRY-level → "Good" (easy test adjusted)
- 60% on a MEDIUM technical test for SENIOR-level → "Needs Development" (high expectations)
- 75% on a MEDIUM assessment for ENTRY-level → "Good" (appropriate for level)

Provide fair, context-aware assessment.
"""
        
        try:
            response = await invoke_llm(
                prompt=prompt,
                task_type=TaskType.REPORT_GENERATION,  # Part of report generation
                agent_name="report_generator"
            )
            content = getattr(response, "content", str(response)).strip()
            
            if content:
                result = extract_json_robust(content)
                if result and isinstance(result, dict):
                    level = result.get("performance_level", "").strip()
                    reasoning = result.get("reasoning", "")
                    percentile = result.get("percentile_estimate", 0)
                    
                    # Validate level
                    valid_levels = ["Excellent", "Good", "Satisfactory", "Needs Development"]
                    if level not in valid_levels:
                        level = self._fallback_performance_level(percentage)
                    
                    return {
                        "performance_level": level,
                        "reasoning": reasoning,
                        "percentile_estimate": percentile,
                        "score_percentage": percentage
                    }
        except Exception as e:
            log.warning(f"LLM performance assessment failed: {e}")
        
        # Fallback to rule-based if LLM fails
        return {
            "performance_level": self._fallback_performance_level(percentage),
            "reasoning": "Rule-based assessment",
            "percentile_estimate": int(percentage),
            "score_percentage": percentage
        }
    
    def _fallback_performance_level(self, percentage: float) -> str:
        """Fallback rule-based performance level (original hard-coded logic)."""
        if percentage >= 80:
            return "Excellent"
        elif percentage >= 60:
            return "Good"
        elif percentage >= 40:
            return "Satisfactory"
        else:
            return "Needs Development"
    
    async def _generate_personalized_feedback_with_llm(
        self,
        assessment_results: Dict[str, Any],
        candidate_profile: Dict[str, Any],
        test_type: str = "assessment"
    ) -> Dict[str, Any]:
        """Generate personalized, domain-specific feedback using LLM.
        
        Args:
            assessment_results: Full assessment data including scores, topics, responses
            candidate_profile: Candidate's domain, experience, recent roles, skills
            test_type: Type of assessment
            
        Returns:
            {
                "feedback": "Personalized feedback text",
                "suggested_next_steps": ["Action 1", "Action 2", ...],
                "strengths_summary": "Summary of strengths",
                "improvement_areas": "Areas needing focus"
            }
        """
        # Extract key information
        score = assessment_results.get("score", 0)
        max_score = assessment_results.get("max_score", 100)
        percentage = (score / max_score * 100) if max_score and max_score > 0 else 0.0
        
        domain = candidate_profile.get("domain_expertise", ["general"])[0] if candidate_profile.get("domain_expertise") else "general"
        recent_role = candidate_profile.get("recent_roles", ["Professional"])[0] if candidate_profile.get("recent_roles") else "Professional"
        experience_years = candidate_profile.get("total_experience_years", 0)
        key_skills = candidate_profile.get("key_skills", [])[:5]
        
        # Get topics/strengths from assessment
        topics = []
        strengths = []
        weaknesses = []
        
        if "topics" in assessment_results and isinstance(assessment_results["topics"], list):
            for topic in assessment_results["topics"][:5]:
                if isinstance(topic, dict):
                    topic_name = topic.get("topic", "")
                    if topic_name:
                        topics.append(topic_name)
                    strengths.extend(topic.get("strengths", [])[:2])
                    weaknesses.extend(topic.get("areas_for_improvement", [])[:2])
        
        # Build context
        context_parts = [
            f"Domain: {domain}",
            f"Current Role: {recent_role}",
            f"Experience: {experience_years} years",
            f"Test Type: {test_type}",
            f"Normalized weighted proficiency: {percentage:.1f}%"
        ]
        
        if key_skills:
            context_parts.append(f"Key Skills: {', '.join(key_skills)}")
        if topics:
            context_parts.append(f"Assessment Areas: {', '.join(topics)}")
        if strengths:
            context_parts.append(f"Demonstrated Strengths: {'; '.join(strengths[:3])}")
        if weaknesses:
            context_parts.append(f"Development Areas: {'; '.join(weaknesses[:3])}")
        
        context_str = "\n".join(context_parts)
        
        prompt = f"""
Generate personalized, actionable feedback for this candidate:

{context_str}

**Requirements:**
1. Make feedback SPECIFIC to their domain ({domain}) and role ({recent_role})
2. Provide ACTIONABLE next steps ORGANIZED BY TOPIC/ASSESSMENT AREA
3. Each topic should have 2-3 specific, practical actions
4. Use domain-appropriate language and examples relevant to their profession
5. Focus on real-world application in their specific role
6. Be encouraging yet constructive
7. If assessment areas include specific topics (e.g., "Active Listening", "Code-switching"), address each topic directly

**Examples of Domain-Specific, Topic-Wise Next Steps:**
For a Hair Stylist assessed on "Communication Skills":
- Active Listening: "During consultations, repeat back clients' requests in your own words to confirm understanding before starting"
- Clarity: "Show visual references or style books to ensure you and your client are aligned on the desired outcome"

For a Software Engineer assessed on "Collaboration":
- Code Reviews: "Provide constructive feedback that explains the 'why' behind suggestions, not just the 'what'"
- Team Communication: "Use diagrams or screenshots when discussing technical solutions to improve clarity"

OUTPUT FORMAT (JSON):
{{
    "feedback": "2-3 paragraphs of personalized, domain-specific feedback",
    "suggested_next_steps": [
        {{"topic": "Topic/Area 1", "actions": ["Specific action 1 for this topic", "Specific action 2 for this topic"]}},
        {{"topic": "Topic/Area 2", "actions": ["Specific action 1 for this topic", "Specific action 2 for this topic"]}},
        {{"topic": "General", "actions": ["Overall development action 1", "Overall development action 2"]}}
    ],
    "strengths_summary": "1-2 sentences highlighting key strengths shown in assessment",
    "improvement_areas": "1-2 sentences on specific focus areas related to their role"
}}

Generate professional, empowering, domain-specific, and topic-organized feedback.
"""
        
        try:
            response = await invoke_llm(
                prompt=prompt,
                task_type=TaskType.REPORT_GENERATION,  # Part of report generation
                agent_name="report_generator"
            )
            content = getattr(response, "content", str(response)).strip()
            
            if content:
                result = extract_json_robust(content)
                if result and isinstance(result, dict):
                    feedback = result.get("feedback", "")
                    next_steps_raw = result.get("suggested_next_steps", [])
                    strengths_summary = result.get("strengths_summary", "")
                    improvement_areas = result.get("improvement_areas", "")
                    
                    # Process next_steps - support both topic-wise and flat formats
                    next_steps_processed = []
                    topic_wise_steps = []
                    
                    if isinstance(next_steps_raw, list):
                        for item in next_steps_raw:
                            if isinstance(item, dict) and "topic" in item and "actions" in item:
                                # Topic-wise format
                                topic_name = item.get("topic", "General")
                                actions = item.get("actions", [])
                                if isinstance(actions, list):
                                    topic_wise_steps.append({
                                        "topic": topic_name,
                                        "actions": [str(a) for a in actions if a][:3]  # Max 3 per topic
                                    })
                                    # Also create flat list for backward compatibility
                                    for action in actions[:3]:
                                        if action:
                                            next_steps_processed.append(f"{topic_name}: {action}")
                            elif isinstance(item, str) and item:
                                # Flat format
                                next_steps_processed.append(item)
                    
                    # If we got topic-wise steps, return both formats
                    if topic_wise_steps:
                        return {
                            "feedback": feedback,
                            "suggested_next_steps": next_steps_processed[:8],  # Flat format for compatibility
                            "suggested_next_steps_by_topic": topic_wise_steps,  # New topic-wise format
                            "strengths_summary": strengths_summary,
                            "improvement_areas": improvement_areas
                        }
                    else:
                        return {
                            "feedback": feedback,
                            "suggested_next_steps": next_steps_processed[:6],
                            "strengths_summary": strengths_summary,
                            "improvement_areas": improvement_areas
                        }
        except Exception as e:
            log.warning(f"LLM feedback generation failed: {e}")
        
        # Fallback to minimal generic feedback
        return {
            "feedback": f"Assessment completed with a normalized weighted proficiency score of {percentage:.1f}%. Review your performance across all areas and focus on continuous improvement.",
            "suggested_next_steps": [
                "Review areas where you scored lower",
                "Practice skills in real-world scenarios",
                "Seek feedback from peers or mentors",
                "Set specific improvement goals"
            ],
            "strengths_summary": "Demonstrated competency in assessed areas.",
            "improvement_areas": "Continue developing skills through practice and learning."
        }
    
    async def _extract_experience_context(self, assessment_results: Dict[str, Any], state: Dict[str, Any] | None = None, use_llm_detection: bool = True) -> Dict[str, Any]:
        """Extract relevant experience data for context-aware reporting from multiple sources.
        
        Args:
            assessment_results: Assessment data
            state: Full state containing resume and other context
            use_llm_detection: If True, use LLM for domain detection; otherwise use keyword matching (fallback)
        """
        experience_context = {
            "work_experience": [],
            "certifications": [],
            "projects": [],
            "total_experience_years": 0,
            "skill_level": "entry",  # entry, mid, senior
            "domain_expertise": [],
            "education_level": "unknown",
            "recent_roles": [],
            "key_skills": [],
            "is_generic_test": False  # Flag to indicate if this is a behavioral/generic test
        }
        
        # Check if this is a generic/behavioral test
        test_type = assessment_results.get("test_type", "")
        if isinstance(test_type, str):
            test_type_lower = test_type.lower()
            if any(keyword in test_type_lower for keyword in ["personality", "psychometric", "communication", "behavioral"]):
                experience_context["is_generic_test"] = True
        
        # Extract from assessment results if available
        if "work_experience" in assessment_results:
            experience_context["work_experience"] = assessment_results["work_experience"][:3]  # Top 3 most recent
        
        if "certifications" in assessment_results:
            experience_context["certifications"] = assessment_results["certifications"][:5]  # Top 5 certifications
        
        if "projects" in assessment_results:
            experience_context["projects"] = assessment_results["projects"][:3]  # Top 3 projects
        
        if "total_experience_years" in assessment_results:
            exp_years = assessment_results["total_experience_years"]
            # Parse experience years defensively (handles int, float, or string like "8 years 4 months")
            exp_years_num = self._parse_experience_value(exp_years)
            experience_context["total_experience_years"] = exp_years_num
            
            # Determine skill level based on experience
            if exp_years_num >= 5:
                experience_context["skill_level"] = "senior"
            elif exp_years_num >= 2:
                experience_context["skill_level"] = "mid"
            else:
                experience_context["skill_level"] = "entry"

        # Augment from state. Prefer resume-derived data over assessment where present
        if state and isinstance(state, dict):
            try:
                structured_resume = state.get("structured_resume", {}) or {}
                # Prefer resume total experience years if present and > 0
                resume_years = structured_resume.get("total_experience_years")
                if resume_years is not None:
                    resume_years_num = self._parse_experience_value(resume_years)
                    if resume_years_num > 0:
                        experience_context["total_experience_years"] = resume_years_num
                        if resume_years_num >= 5:
                            experience_context["skill_level"] = "senior"
                        elif resume_years_num >= 2:
                            experience_context["skill_level"] = "mid"
                        else:
                            experience_context["skill_level"] = "entry"

                # If work experience is empty, populate from resume
                if not experience_context["work_experience"]:
                    resume_experience = structured_resume.get("experience") or []
                    if isinstance(resume_experience, list) and resume_experience:
                        experience_context["work_experience"] = resume_experience[:3]

                # Derive domains, recent roles, and key skills from resume when available
                if structured_resume and not experience_context["domain_expertise"]:
                    domains_from_resume = set()
                    recent_roles_from_resume = []
                    key_skills_from_resume = set()
                    for job in (structured_resume.get("experience") or [])[:6]:
                        if isinstance(job, dict):
                            title = job.get("job_title") or job.get("title") or ""
                            company = job.get("company") or ""
                            if title and company:
                                recent_roles_from_resume.append(f"{title} at {company}")
                            
                            # Use LLM-based domain detection if enabled
                            if use_llm_detection and title:
                                responsibilities = job.get("responsibilities", [])
                                detection_result = await self._detect_domain_with_llm(title, company, responsibilities)
                                if detection_result.get("domain"):
                                    domains_from_resume.add(detection_result["domain"])
                                if detection_result.get("skills"):
                                    key_skills_from_resume.update(detection_result["skills"])
                    
                    if domains_from_resume:
                        experience_context["domain_expertise"] = list(domains_from_resume)
                    if recent_roles_from_resume:
                        experience_context["recent_roles"] = recent_roles_from_resume[:3]
                    if key_skills_from_resume:
                        experience_context["key_skills"] = list(key_skills_from_resume)[:10]
            except Exception:
                # Best-effort augmentation; ignore augmentation errors
                pass
        
        # Extract domain expertise from work experience
        domains = set()
        recent_roles = []
        key_skills = set()
        
        for job in experience_context["work_experience"]:
            if isinstance(job, dict):
                # Extract job title and company for recent roles
                title = job.get("job_title", "")
                company = job.get("company", "")
                if title and company:
                    recent_roles.append(f"{title} at {company}")
                
                # Use LLM-based domain detection if enabled
                if use_llm_detection and title:
                    responsibilities = job.get("responsibilities", [])
                    detection_result = await self._detect_domain_with_llm(title, company, responsibilities)
                    if detection_result.get("domain"):
                        domains.add(detection_result["domain"])
                    if detection_result.get("skills"):
                        key_skills.update(detection_result["skills"])
                
                # Extract skills from responsibilities
                responsibilities = job.get("responsibilities", [])
                if isinstance(responsibilities, list):
                    for resp in responsibilities:
                        resp_lower = resp.lower()
                        # Extract technical skills
                        if any(tech in resp_lower for tech in ["python", "java", "javascript", "react", "node"]):
                            key_skills.add("programming")
                        if any(tech in resp_lower for tech in ["sql", "database", "mysql", "postgresql"]):
                            key_skills.add("database management")
                        if any(tech in resp_lower for tech in ["aws", "azure", "cloud", "docker", "kubernetes"]):
                            key_skills.add("cloud computing")
                        if any(tech in resp_lower for tech in ["agile", "scrum", "project management"]):
                            key_skills.add("project management")
        
        experience_context["domain_expertise"] = list(domains)
        experience_context["recent_roles"] = recent_roles[:3]  # Top 3 recent roles
        experience_context["key_skills"] = list(key_skills)[:10]  # Top 10 key skills
        
        # Extract education level from certifications and projects
        education_indicators = []
        for cert in experience_context["certifications"]:
            if isinstance(cert, dict):
                cert_name = cert.get("certification_name", "").lower()
                if any(degree in cert_name for degree in ["bachelor", "master", "phd", "degree"]):
                    education_indicators.append("degree")
                elif any(prof in cert_name for prof in ["professional", "certified", "expert"]):
                    education_indicators.append("professional")
        
        if education_indicators:
            if "degree" in education_indicators:
                experience_context["education_level"] = "degree"
            elif "professional" in education_indicators:
                experience_context["education_level"] = "professional"
        
        return experience_context

    def _generate_enhanced_report_prompt(self, assessment_results: Dict[str, Any], candidate_name: str, experience_data: Dict[str, Any]) -> str:
        """Generate enhanced, context-aware report prompt with better instructions."""
        
        # Extract key assessment metrics
        total_score = assessment_results.get("total_score", 0)
        max_score = assessment_results.get("max_score", 100)
        percentage = (total_score / max_score * 100) if max_score > 0 else 0
        
        # Determine assessment type and topics
        assessment_type = assessment_results.get("assessment_type", "technical")
        topics = assessment_results.get("topics", [])
        if isinstance(topics, str):
            topics = [topics]
        
        # Build experience context string
        exp_context = ""
        exp_years = self._parse_experience_value(experience_data.get("total_experience_years", 0))
        if exp_years > 0:
            exp_context += f"Candidate has {exp_years:.1f} years of experience ({experience_data.get('skill_level', 'entry')} level). "
        
        if experience_data["domain_expertise"]:
            exp_context += f"Domain expertise: {', '.join(experience_data['domain_expertise'])}. "
        
        if experience_data["recent_roles"]:
            exp_context += f"Recent roles: {', '.join(experience_data['recent_roles'])}. "
        
        if experience_data["key_skills"]:
            exp_context += f"Key skills: {', '.join(experience_data['key_skills'][:5])}. "
        
        if experience_data["certifications"]:
            cert_names = [cert.get("certification_name", "") for cert in experience_data["certifications"] if isinstance(cert, dict)]
            if cert_names:
                exp_context += f"Relevant certifications: {', '.join(cert_names[:3])}. "
        
        if experience_data["education_level"] != "unknown":
            exp_context += f"Education level: {experience_data['education_level']}. "
        
        # Build detailed assessment data
        assessment_details = self._build_assessment_details(assessment_results)
        
        prompt = f"""You are an expert career coach and technical evaluator with deep industry knowledge. Generate a comprehensive, personalized performance report that provides actionable insights and specific recommendations.

CANDIDATE CONTEXT:
- Name: {candidate_name}
- {exp_context}
- Assessment Type: {assessment_type}
- Topics Evaluated: {', '.join(topics) if topics else 'General Skills'}

ASSESSMENT RESULTS:
- Weighted proficiency score: {total_score}/{max_score} ({percentage:.1f}% normalized)
- Performance Level: {self._determine_performance_level(percentage)}
{assessment_details}

REPORT REQUIREMENTS:
1. PERSONALIZATION: Tailor all feedback to the candidate's experience level and domain expertise
2. SPECIFICITY: Provide concrete examples and specific areas for improvement
3. ACTIONABILITY: Include detailed, implementable recommendations with resources
4. MOTIVATION: Maintain an encouraging tone while being honest about areas needing improvement
5. RELEVANCE: Connect feedback to real-world applications and career advancement
6. EXPERIENCE-AWARE: Reference the candidate's background, recent roles, and key skills when relevant
7. CAREER-FOCUSED: Align recommendations with the candidate's career trajectory and domain expertise
8. SECOND-PERSON VOICE: Write all sections in second-person (use "Your", "You", "Your performance", etc.) - NEVER use "The Candidate" or third-person references. Sections should start with "Your" not "The Candidate."

OUTPUT FORMAT (JSON only):
{{
    "total_score": {total_score},
    "max_score": {max_score},
    "summary": "2-3 sentence personalized overview highlighting key strengths and main area for growth",
    "performance_level": "Needs Improvement|Satisfactory|Good|Excellent",
    "topics": [
        {{
            "topic": "Specific topic name (e.g., 'Data Structures & Algorithms', 'Communication Skills')",
            "strengths": [
                "Specific strength with concrete example from assessment",
                "Another strength with context",
                "Third strength with evidence"
            ],
            "areas_for_improvement": [
                "Specific area with explanation of why it matters",
                "Another area with impact on career growth",
                "Third area with practical implications"
            ],
            "recommendations": [
                "Actionable recommendation with specific resources or steps",
                "Another recommendation with timeline and expected outcome",
                "Third recommendation with measurable progress indicators"
            ]
        }}
    ],
    "feedback": "3-4 paragraph detailed analysis covering: 1) Overall performance assessment, 2) Key strengths and their significance, 3) Critical areas for improvement with reasoning, 4) Career development pathway and next steps",
    "suggested_next_steps": [
        "Specific learning resource with timeline (e.g., 'Complete Coursera Machine Learning course within 6 weeks')",
        "Practical application suggestion (e.g., 'Build a portfolio project using React and Node.js')",
        "Skill development activity (e.g., 'Practice coding interviews on LeetCode, focusing on dynamic programming')",
        "Professional development action (e.g., 'Join local tech meetup groups to network and learn')",
        "Career advancement step (e.g., 'Apply for mid-level developer positions after completing 3 portfolio projects')"
    ],
    "subtopic_breakdown": [{"name": "<subtopic>", "score": 0.0, "notes": "<reason>"}],
    "misconceptions": [{"subtopic": "<subtopic>", "message": "<brief misconception>", "evidence": ["mcq[2]"]}],
    "rubric_scores": {"short_answer": {"clarity": 0.0, "correctness": 0.0, "terminology": 0.0}, "long_answer": {"structure": 0.0, "integration": 0.0, "specificity": 0.0}},
    "resources": [{"subtopic": "<subtopic>", "links": ["https://facebook.github.io/prophet/docs/quick_start.html"]}]
}}

QUALITY STANDARDS:
- Each strength must include specific evidence from the assessment
- Each improvement area must explain why it matters for career growth
- Each recommendation must be actionable with clear next steps
- Feedback must be personalized to the candidate's experience level
- All content must be directly relevant to the assessment results
- Avoid generic advice - provide specific, tailored guidance

Generate the report now:"""

        return prompt

    def _generate_adaptive_prompt_variations(self, assessment_results: Dict[str, Any], candidate_name: str, experience_data: Dict[str, Any]) -> str:
        """Generate adaptive prompt based on assessment type, experience level, and performance patterns."""
        
        # Extract key metrics
        def _to_float(value: Any, default: float = 0.0) -> float:
            try:
                if value is None:
                    return default
                if isinstance(value, str):
                    cleaned = value.strip().replace('%', '')
                    return float(cleaned)
                return float(value)
            except (ValueError, TypeError):
                return default

        total_score = _to_float(assessment_results.get("total_score", 0.0), 0.0)
        max_score = _to_float(assessment_results.get("max_score", 0.0), 0.0)
        percentage = (total_score / max_score * 100) if max_score and max_score > 0 else 0.0
        assessment_type = assessment_results.get("assessment_type", "technical")
        skill_level = experience_data.get("skill_level", "entry")
        
        # Determine prompt strategy based on performance and experience
        if percentage >= 85 and skill_level == "senior":
            strategy = "expert_mentoring"
        elif percentage >= 70 and skill_level in ["mid", "senior"]:
            strategy = "advanced_development"
        elif percentage >= 55 and skill_level == "entry":
            strategy = "foundational_building"
        elif percentage < 55 and skill_level == "entry":
            strategy = "remedial_focus"
        else:
            strategy = "balanced_improvement"
        
        # Generate strategy-specific instructions
        strategy_instructions = self._get_strategy_instructions(strategy, assessment_type, skill_level, percentage)
        
        # Build base prompt with strategy-specific enhancements
        base_prompt = self._generate_enhanced_report_prompt(assessment_results, candidate_name, experience_data)
        
        # Add strategy-specific enhancements
        enhanced_prompt = f"""{base_prompt}

STRATEGY-SPECIFIC INSTRUCTIONS ({strategy.upper()}):
{strategy_instructions}

ADAPTIVE GUIDANCE:
- Assessment Type: {assessment_type}
- Candidate Level: {skill_level}
- Normalized weighted proficiency: {percentage:.1f}%
- Strategy: {strategy.replace('_', ' ').title()}

Focus your recommendations and feedback according to the strategy above."""

        return enhanced_prompt

    def _get_strategy_instructions(self, strategy: str, assessment_type: str, skill_level: str, percentage: float) -> str:
        """Get strategy-specific instructions for prompt adaptation."""
        
        strategies = {
            "expert_mentoring": f"""
- This candidate demonstrates expert-level weighted proficiency ({percentage:.1f}%) with senior experience
- Focus on advanced career development, leadership opportunities, and mentoring others
- Recommend specialized certifications, advanced training, and thought leadership opportunities
- Suggest ways to share expertise through speaking, writing, or mentoring
- Emphasize career advancement to principal/staff/architect roles
- Include recommendations for industry recognition and professional networking""",

            "advanced_development": f"""
- This candidate shows strong weighted proficiency ({percentage:.1f}%) with {skill_level}-level experience
- Focus on skill deepening and career progression within their domain
- Recommend advanced certifications and specialized training programs
- Suggest taking on more complex projects and leadership responsibilities
- Include recommendations for building expertise in emerging technologies
- Emphasize preparation for senior-level roles and technical leadership""",

            "foundational_building": f"""
- This candidate shows satisfactory weighted proficiency ({percentage:.1f}%) as an entry-level professional
- Focus on building strong foundational skills and gaining practical experience
- Recommend fundamental courses, hands-on projects, and mentorship programs
- Suggest ways to gain real-world experience through internships or side projects
- Include recommendations for building a strong professional network
- Emphasize consistent practice and gradual skill development""",

            "remedial_focus": f"""
- This candidate needs significant improvement in weighted proficiency ({percentage:.1f}%) as an entry-level professional
- Focus on fundamental skill building and knowledge gaps
- Recommend basic courses, tutorials, and structured learning programs
- Suggest finding a mentor or study group for support
- Include recommendations for building confidence through small wins
- Emphasize the importance of consistent practice and not giving up""",

            "balanced_improvement": f"""
- This candidate shows mixed weighted proficiency ({percentage:.1f}%) with {skill_level}-level experience
- Focus on identifying specific strengths to build upon and weaknesses to address
- Recommend targeted skill development in areas of weakness
- Suggest leveraging existing strengths for career advancement
- Include recommendations for practical application of learned skills
- Emphasize balanced development across technical and soft skills"""
        }
        
        return strategies.get(strategy, strategies["balanced_improvement"])

    def _convert_to_second_person(self, text: str) -> str:
        """Convert third-person references to second-person for better personalization.
        
        Converts phrases like "The Candidate." to "Your" at the start of sections,
        and other third-person references to second-person throughout the text.
        """
        if not text or not isinstance(text, str):
            return text
        
        # Common third-person patterns to convert to second-person
        replacements = [
            # "The Candidate." at start of sentence/section -> "Your"
            (r'^The Candidate\.\s*', r'Your '),
            (r'\.\s*The Candidate\.\s*', r'. Your '),
            (r'\n\s*The Candidate\.\s*', r'\nYour '),
            # "The candidate" -> "You" or "Your" depending on context
            (r'\bThe candidate\s+', r'You '),
            (r'\bThe candidate\'?s\s+', r'Your '),
            (r'\bcandidate\'?s\s+', r'your '),
            # "The Candidate" (capitalized) -> "You" or "Your"
            (r'\bThe Candidate\s+', r'You '),
            (r'\bThe Candidate\'?s\s+', r'Your '),
        ]
        
        result = text
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE | re.MULTILINE)
        
        return result

    async def _enrich_report_content(self, result: Dict[str, Any], assessment_results: Dict[str, Any], experience_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich report content with additional insights, resources, and actionable details."""
        
        # Create a copy to avoid modifying the original
        enriched_result = result.copy()
        
        # Helper: build a canonical list of topic names for matching
        def _collect_topic_names() -> List[str]:
            names: List[str] = []
            topics = enriched_result.get("topics") or []
            for t in topics:
                if isinstance(t, dict):
                    name = str(t.get("topic", "")).strip()
                    if name:
                        names.append(name)
            # Also include subtopic_breakdown names from assessment_results
            for s in (assessment_results.get("subtopic_breakdown") or []):
                if isinstance(s, dict):
                    n = str(s.get("name", "")).strip()
                    if n:
                        names.append(n)
            # Deduplicate preserving order
            seen = set()
            ordered: List[str] = []
            for n in names:
                if n not in seen:
                    seen.add(n)
                    ordered.append(n)
            return ordered
        
        # Normalize field types to prevent type errors
        # Ensure string fields are strings
        if "summary" in enriched_result and not isinstance(enriched_result["summary"], str):
            enriched_result["summary"] = str(enriched_result["summary"]) if enriched_result["summary"] else ""
        if "feedback" in enriched_result and not isinstance(enriched_result["feedback"], str):
            enriched_result["feedback"] = str(enriched_result["feedback"]) if enriched_result["feedback"] else ""
        if "performance_level" in enriched_result and not isinstance(enriched_result["performance_level"], str):
            enriched_result["performance_level"] = str(enriched_result["performance_level"]) if enriched_result["performance_level"] else "Satisfactory"
        
        # Convert third-person references to second-person for better personalization
        if "summary" in enriched_result and enriched_result["summary"]:
            enriched_result["summary"] = self._convert_to_second_person(enriched_result["summary"])
        if "feedback" in enriched_result and enriched_result["feedback"]:
            enriched_result["feedback"] = self._convert_to_second_person(enriched_result["feedback"])
        
        # Ensure list fields are lists
        if "topics" in enriched_result and not isinstance(enriched_result["topics"], list):
            enriched_result["topics"] = []
        if "suggested_next_steps" in enriched_result and not isinstance(enriched_result["suggested_next_steps"], list):
            enriched_result["suggested_next_steps"] = []
        if "subtopic_breakdown" in enriched_result and not isinstance(enriched_result["subtopic_breakdown"], list):
            enriched_result["subtopic_breakdown"] = []
        if "misconceptions" in enriched_result and not isinstance(enriched_result["misconceptions"], list):
            enriched_result["misconceptions"] = []
        if "resources" in enriched_result and not isinstance(enriched_result["resources"], list):
            enriched_result["resources"] = []
        else:
            # Filter invalid links from resources
            if "resources" in enriched_result and isinstance(enriched_result["resources"], list):
                enriched_result["resources"] = filter_valid_resources(enriched_result["resources"])
        
        # Ensure dict fields are dicts
        if "rubric_scores" in enriched_result and not isinstance(enriched_result["rubric_scores"], dict):
            enriched_result["rubric_scores"] = {}
        
        # Add learning resources based on assessment topics and experience (async)
        try:
            learning_resources = await self._generate_learning_resources(assessment_results, experience_data)
            if learning_resources:
                enriched_result["learning_resources"] = learning_resources
        except Exception as e:
            log.warning(f"Learning resource generation failed: {e}")
            # Continue without resources rather than failing
        
        # Add career development insights
        career_insights = self._generate_career_insights(assessment_results, experience_data)
        if career_insights:
            enriched_result["career_insights"] = career_insights
        
        # Add skill gap analysis
        skill_gaps = self._analyze_skill_gaps(assessment_results, experience_data)
        if skill_gaps:
            enriched_result["skill_gap_analysis"] = skill_gaps
        
        # Add industry-specific recommendations
        industry_recommendations = self._generate_industry_recommendations(assessment_results, experience_data)
        if industry_recommendations:
            enriched_result["industry_recommendations"] = industry_recommendations
        
        # Enhance next steps with more specific timelines and resources
        enhanced_next_steps = self._enhance_next_steps(result.get("suggested_next_steps", []), experience_data)
        enriched_result["suggested_next_steps"] = enhanced_next_steps
        
        # Inject subject-specific feedback derived from question-level performance (universal across domains)
        subject_insights = await asyncio.to_thread(self._extract_subject_misunderstandings, assessment_results)
        if subject_insights:
            # Merge into topic sections where possible
            topics_section = enriched_result.get("topics") or []
            if isinstance(topics_section, list):
                updated_topics = []
                for topic_entry in topics_section:
                    if isinstance(topic_entry, dict):
                        topic_name = str(topic_entry.get("topic", "")).strip()
                        areas = list(topic_entry.get("areas_for_improvement", []) or [])
                        # Match subject insights to this topic by simple substring overlap (case-insensitive)
                        matched_keys = [k for k in subject_insights.keys() if k and (k.lower() in topic_name.lower() or topic_name.lower() in k.lower())]
                        merged_items = []
                        for k in matched_keys:
                            for principle in subject_insights.get(k, [])[:3]:
                                merged_items.append(f"Misunderstood: {principle}")
                        # If no direct match, attach top generic insights once at report level later
                        if merged_items:
                            # Deduplicate while preserving order
                            seen = set()
                            for it in merged_items:
                                if it not in seen:
                                    seen.add(it)
                                    areas.append(it)
                        topic_entry["areas_for_improvement"] = areas
                        updated_topics.append(topic_entry)
                    else:
                        updated_topics.append(topic_entry)
                enriched_result["topics"] = updated_topics

            # Add a concise subject-specific section to feedback
            feedback_text = enriched_result.get("feedback", "") or ""
            # Ensure feedback_text is a string, not a list
            if not isinstance(feedback_text, str):
                feedback_text = str(feedback_text) if feedback_text else ""
            lines = []
            for k, vals in subject_insights.items():
                if not vals:
                    continue
                # Limit to top 3 per subject key
                top_vals = ", ".join(vals[:3])
                lines.append(f"- {k}: {top_vals}")
            if lines:
                subject_block = "\n\nSubject-Specific Feedback:\n" + "\n".join(lines)
                enriched_result["feedback"] = (feedback_text + subject_block).strip()
                # Convert to second-person after adding subject block
                enriched_result["feedback"] = self._convert_to_second_person(enriched_result["feedback"])
        
        # Integrate subtopic_breakdown notes directly into feedback for topic specificity
        st_breakdown = assessment_results.get("subtopic_breakdown") or []
        topic_notes_lines: List[str] = []
        if isinstance(st_breakdown, list) and st_breakdown:
            for entry in st_breakdown[:10]:
                if isinstance(entry, dict):
                    nm = str(entry.get("name", "")).strip()
                    note = str(entry.get("notes", "")).strip()
                    if nm and note:
                        topic_notes_lines.append(f"- {nm}: {note}")
            if topic_notes_lines:
                fb_text = enriched_result.get("feedback", "") or ""
                if not isinstance(fb_text, str):
                    fb_text = str(fb_text)
                block = "\n\nTopic-Specific Notes:\n" + "\n".join(topic_notes_lines)
                enriched_result["feedback"] = (fb_text + block).strip()
                enriched_result["feedback"] = self._convert_to_second_person(enriched_result["feedback"])

        # Also convert topics and other text fields to second-person
        if "topics" in enriched_result and isinstance(enriched_result["topics"], list):
            for topic_entry in enriched_result["topics"]:
                if isinstance(topic_entry, dict):
                    for field in ["strengths", "areas_for_improvement", "recommendations"]:
                        if field in topic_entry and isinstance(topic_entry[field], list):
                            topic_entry[field] = [
                                self._convert_to_second_person(str(item)) if isinstance(item, str) else item
                                for item in topic_entry[field]
                            ]
        # If subtopic_breakdown notes exist, attach matching notes under the corresponding topic's areas_for_improvement
        topic_names = _collect_topic_names()
        if "topics" in enriched_result and isinstance(enriched_result["topics"], list) and isinstance(st_breakdown, list):
            name_to_notes: Dict[str, List[str]] = {}
            for entry in st_breakdown:
                if isinstance(entry, dict):
                    nm = str(entry.get("name", "")).strip()
                    note = str(entry.get("notes", "")).strip()
                    if nm and note:
                        name_to_notes.setdefault(nm, []).append(note)
            if name_to_notes:
                for topic_entry in enriched_result["topics"]:
                    if not isinstance(topic_entry, dict):
                        continue
                    topic_name = str(topic_entry.get("topic", "")).strip()
                    if not topic_name:
                        continue
                    # Match direct or substring overlap
                    matched = [nm for nm in name_to_notes.keys() if nm.lower() in topic_name.lower() or topic_name.lower() in nm.lower()]
                    if matched:
                        notes_to_add: List[str] = []
                        for m in matched:
                            notes_to_add.extend(name_to_notes.get(m, [])[:2])
                        if notes_to_add:
                            existing = list(topic_entry.get("areas_for_improvement", []) or [])
                            for nt in notes_to_add:
                                phr = self._convert_to_second_person(str(nt))
                                if phr not in existing:
                                    existing.append(phr)
                            topic_entry["areas_for_improvement"] = existing

        return enriched_result

    def _reinforce_topic_specificity(self, report: Dict[str, Any], assessment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure summary and feedback explicitly reference topic or subtopic names to avoid generic tone."""
        if not isinstance(report, dict):
            return report
        # Collect candidate names to reference
        topic_names: List[str] = []
        for t in (report.get("topics") or []):
            if isinstance(t, dict):
                nm = str(t.get("topic", "")).strip()
                if nm:
                    topic_names.append(nm)
        for s in (assessment_results.get("subtopic_breakdown") or []):
            if isinstance(s, dict):
                nm = str(s.get("name", "")).strip()
                if nm:
                    topic_names.append(nm)
        # Deduplicate
        seen = set()
        topic_names = [n for n in topic_names if not (n in seen or seen.add(n))]
        if not topic_names:
            return report
        def ensure_mentions(text: str) -> str:
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            lower_text = text.lower()
            present = any(name.lower() in lower_text for name in topic_names)
            if present:
                return text
            # Prepend a concise, topic-specific lead-in with up to 2 topics
            lead = ", ".join(topic_names[:2])
            prefix = f"Your performance in {lead} stood out in this assessment. "
            return prefix + text
        if "summary" in report:
            report["summary"] = self._convert_to_second_person(ensure_mentions(report.get("summary", "") or ""))
        if "feedback" in report:
            report["feedback"] = self._convert_to_second_person(ensure_mentions(report.get("feedback", "") or ""))
        return report

    def _extract_subject_misunderstandings(self, assessment_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract likely misunderstood subtopics/principles from per-question results.

        Universal heuristic that works across domains:
        - Uses question text + evaluator's correct_answer/explanation
        - Aggregates by inferred subject key from question text
        - Returns mapping: subject_key -> list of concise principles
        """
        insights: Dict[str, List[str]] = {}

        per_question = assessment_results.get("per_question", {}) or {}
        if not isinstance(per_question, dict):
            return insights

        # Helper to push a principle into a subject bucket with de-duplication
        def _add(subject_key: str, principle: str) -> None:
            key = subject_key.strip() or "General"
            val = principle.strip()
            if not val:
                return
            bucket = insights.setdefault(key, [])
            if val not in bucket:
                bucket.append(val)

        # Iterate across all question types
        for qtype, questions in per_question.items():
            if not isinstance(questions, list):
                continue
            for q in questions:
                if not isinstance(q, dict):
                    continue
                qtext = str(q.get("question", ""))
                eval_block = q.get("evaluation", {}) or {}
                # Consider misunderstandings when score is low or explicit wrongness is implied
                score = eval_block.get("score")
                try:
                    score_val = float(score) if score is not None else None
                except Exception:
                    score_val = None

                correct_answer = str(eval_block.get("correct_answer", ""))
                explanation = str(eval_block.get("explanation", ""))

                is_potential_issue = False
                if score_val is not None:
                    # Heuristic thresholds; works for most 0-10 scales
                    is_potential_issue = score_val < 7.5
                else:
                    # If no numeric score, still consider if explanation includes corrective language
                    lower_exp = explanation.lower()
                    is_potential_issue = any(w in lower_exp for w in ["incorrect", "should", "instead", "not", "however", "but "])

                if not is_potential_issue:
                    continue

                subject_key = self._infer_subject_key(qtext)
                # Derive a concise principle from question + correct answer/explanation
                principle = self._derive_concise_principle(qtext, correct_answer, explanation)
                if principle:
                    _add(subject_key, principle)

        # Cap list sizes for brevity
        for k in list(insights.keys()):
            insights[k] = insights[k][:5]

        return insights

    def _infer_subject_key(self, question_text: str) -> str:
        """Infer a subject key from the question text using generic cues (language-agnostic)."""
        text = (question_text or "").strip()
        if not text:
            return "General"

        # Prefer quoted/backticked phrases as subject keys
        import re
        m = re.search(r"`([^`]{3,60})`", text)
        if m:
            return m.group(1)
        m = re.search(r"'([^']{3,60})'|\"([^\"]{3,60})\"", text)
        if m:
            return (m.group(1) or m.group(2))

        # Fall back: pick first 2-3 capitalized words sequence (e.g., Value at Risk)
        m = re.search(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\b", text)
        if m:
            return m.group(1)

        # Last resort: use prominent noun-like token (long word)
        tokens = re.findall(r"[A-Za-z]{5,}", text)
        return tokens[0] if tokens else "General"

    def _derive_concise_principle(self, question_text: str, correct_answer: str, explanation: str) -> str:
        """Create a short, subject-specific principle statement from available fields."""
        qt = (question_text or "").strip()
        ca = (correct_answer or "").strip()
        ex = (explanation or "").strip()

        # Prefer structured patterns in questions (works universally)
        import re
        # Patterns like: Explain the concept of X, Difference between X and Y, Purpose of X, Formula for X
        patterns = [
            (r"[Ee]xplain the (?:core )?concept of ([^.?]+)", "Concept misunderstood: {x}"),
            (r"[Ww]hat is the difference between ([^?]+)\?", "Difference unclear: {x}"),
            (r"[Pp]urpose of ([^?]+)\?", "Purpose unclear: {x}"),
            (r"[Ff]ormula (?:for|of) ([^.?]+)", "Formula/relationship: {x}"),
        ]
        for pat, tmpl in patterns:
            m = re.search(pat, qt)
            if m:
                x = m.group(1).strip().rstrip('.')
                return tmpl.format(x=x)

        # Use correct answer cue if available
        if ca:
            # Keep it short
            ca_short = re.sub(r"\s+", " ", ca)[:120]
            return f"Key point: {ca_short}"

        # Fallback to explanation cue
        if ex:
            ex_short = re.sub(r"\s+", " ", ex)[:140]
            return f"Clarify: {ex_short}"

        # Last resort: a trimmed question cue
        qt_short = re.sub(r"\s+", " ", qt)[:120]
        return f"Review: {qt_short}"

    async def _generate_learning_resources(self, assessment_results: Dict[str, Any], experience_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific learning resources dynamically using LLM, with fallback to static/config resources.
        
        This method replaces static resource lists with an LLM-driven approach that works
        universally across all topics and professions, following agentic AI principles.
        """
        resources = []
        
        # Get assessment topics
        topics = assessment_results.get("topics", [])
        if isinstance(topics, str):
            topics = [topics]
        elif isinstance(topics, list):
            # Extract topic names from dict format if needed
            topics = [
                t.get("topic", t) if isinstance(t, dict) else str(t)
                for t in topics
            ]
        
        # Get skill level and domain expertise
        skill_level = experience_data.get("skill_level", "entry")
        domains = experience_data.get("domain_expertise", [])
        experience_years = experience_data.get("total_experience_years", 0)
        
        # Try config-based resources first (if available)
        try:
            from core.config import get_learning_resources
            config_resources = get_learning_resources()
            if config_resources and isinstance(config_resources, dict):
                for topic in topics[:3]:
                    topic_lower = str(topic).lower()
                    # Try to match topic to config resources
                    for key, resource_data in config_resources.items():
                        if key.lower() in topic_lower or topic_lower in key.lower():
                            if isinstance(resource_data, dict):
                                # Extract resources from config format
                                for resource_type in ["courses", "books", "tools", "practice_sites"]:
                                    items = resource_data.get(resource_type, [])
                                    if isinstance(items, list):
                                        for item in items[:1]:  # Take first item of each type
                                            if isinstance(item, str):
                                                resources.append({
                                                    "type": resource_type[:-1] if resource_type.endswith("s") else resource_type,
                                                    "name": item,
                                                    "platform": "Various",
                                                    "duration": "Varies",
                                                    "cost": "Varies"
                                                })
                            break
        except Exception:
            pass  # Fallback to LLM or static methods
        
        # If we have topics but no config resources, use LLM to generate dynamically
        if topics and len(resources) < 3:
            try:
                # Generate resources dynamically using LLM for remaining topics
                llm_resources = await self._generate_resources_with_llm(
                    topics=topics[:3],
                    skill_level=skill_level,
                    experience_years=experience_years,
                    domains=domains
                )
                resources.extend(llm_resources)
            except Exception as e:
                log.warning(f"LLM resource generation failed: {e}, using static fallback")
                # Fallback to static keyword-based matching
                resources.extend(self._get_static_resources_fallback(topics, skill_level))
        
        # If still no resources, use static fallback
        if not resources:
            resources = self._get_static_resources_fallback(topics, skill_level)
        
        return resources[:5]  # Limit to 5 resources

    async def _generate_resources_with_llm(
        self,
        topics: List[str],
        skill_level: str,
        experience_years: float,
        domains: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate learning resources dynamically using LLM for any profession.
        
        Args:
            topics: List of assessment topics
            skill_level: Candidate skill level (entry, mid, senior)
            experience_years: Years of experience
            domains: List of domain expertise areas
            
        Returns:
            List of learning resource dictionaries
        """
        if not topics:
            return []
        
        # Build prompt for LLM
        topics_str = ", ".join(topics[:3])
        domains_str = ", ".join(domains[:2]) if domains else "General"
        
        prompt = f"""Generate 3-5 specific, actionable learning resources for someone learning about: {topics_str}

Context:
- Skill Level: {skill_level}
- Experience: {experience_years:.1f} years
- Domain: {domains_str}

Requirements:
- Resources should be appropriate for the skill level and experience
- Include a mix of courses, books, practice platforms, and tools
- Make resources specific to the topics mentioned
- Ensure resources are profession-agnostic and universally applicable
- Include realistic costs and durations

Return ONLY a JSON array:
[
  {{
    "type": "course|book|practice|tool|certification",
    "name": "Specific resource name",
    "platform": "Platform/provider name",
    "duration": "Realistic duration (e.g., '4 weeks', '2-3 months')",
    "cost": "Cost information (e.g., 'Free', '$50', '$39/month')"
  }}
]

Topics: {topics_str}
Return JSON array only, no other text."""

        try:
            response = await invoke_llm(
                prompt=prompt,
                task_type=TaskType.REPORT_GENERATION,
                agent_name="report_generator"
            )
            content = getattr(response, "content", str(response)).strip()
            
            # Extract JSON from response
            result = extract_json_robust(content)
            
            # Validate and sanitize results
            if isinstance(result, list):
                validated_resources = []
                for item in result:
                    if isinstance(item, dict) and item.get("name"):
                        # Ensure required fields
                        resource = {
                            "type": item.get("type", "course"),
                            "name": str(item.get("name", ""))[:200],  # Limit length
                            "platform": str(item.get("platform", "Various"))[:100],
                            "duration": str(item.get("duration", "Varies"))[:50],
                            "cost": str(item.get("cost", "Varies"))[:50]
                        }
                        validated_resources.append(resource)
                
                return validated_resources[:5]  # Limit to 5 resources
            
        except Exception as e:
            log.warning(f"LLM resource generation error: {e}")
        
        return []
    
    def _get_static_resources_fallback(self, topics: List[str], skill_level: str) -> List[Dict[str, Any]]:
        """Fallback to static keyword-based resource matching when LLM/config fail.
        
        This consolidates the static resource methods as a last resort fallback.
        """
        resources = []
        
        for topic in topics[:3]:
            if not isinstance(topic, str):
                continue
                
            topic_lower = topic.lower()
            
            # Technical topics - use static methods as fallback
            if any(tech in topic_lower for tech in ["programming", "coding", "software", "development"]):
                resources.extend(self._get_programming_resources(skill_level))
            elif any(tech in topic_lower for tech in ["data", "analytics", "machine learning", "ai"]):
                resources.extend(self._get_data_science_resources(skill_level))
            elif any(tech in topic_lower for tech in ["design", "ui", "ux", "frontend"]):
                resources.extend(self._get_design_resources(skill_level))
            elif any(tech in topic_lower for tech in ["cloud", "aws", "azure", "devops"]):
                resources.extend(self._get_cloud_resources(skill_level))
            elif any(tech in topic_lower for tech in ["management", "leadership", "project"]):
                resources.extend(self._get_management_resources(skill_level))
        
        return resources[:5]  # Limit to 5 resources

    def _get_programming_resources(self, skill_level: str) -> List[Dict[str, Any]]:
        """Get programming learning resources based on skill level."""
        if skill_level == "entry":
            return [
                {"type": "course", "name": "Python for Beginners", "platform": "Coursera", "duration": "4 weeks", "cost": "Free"},
                {"type": "practice", "name": "LeetCode Easy Problems", "platform": "LeetCode", "duration": "Ongoing", "cost": "Free"},
                {"type": "book", "name": "Automate the Boring Stuff", "platform": "Book", "duration": "2-3 months", "cost": "Free online"}
            ]
        elif skill_level == "mid":
            return [
                {"type": "course", "name": "Advanced Python Programming", "platform": "Udemy", "duration": "6 weeks", "cost": "$50"},
                {"type": "certification", "name": "AWS Certified Developer", "platform": "AWS", "duration": "2-3 months", "cost": "$150"},
                {"type": "practice", "name": "System Design Interview Prep", "platform": "Educative", "duration": "1 month", "cost": "$30/month"}
            ]
        else:  # senior
            return [
                {"type": "certification", "name": "AWS Solutions Architect Professional", "platform": "AWS", "duration": "3-4 months", "cost": "$300"},
                {"type": "course", "name": "Distributed Systems Design", "platform": "MIT OpenCourseWare", "duration": "4 months", "cost": "Free"},
                {"type": "practice", "name": "Open Source Contributions", "platform": "GitHub", "duration": "Ongoing", "cost": "Free"}
            ]

    def _get_data_science_resources(self, skill_level: str) -> List[Dict[str, Any]]:
        """Get data science learning resources based on skill level."""
        if skill_level == "entry":
            return [
                {"type": "course", "name": "Introduction to Data Science", "platform": "Coursera", "duration": "6 weeks", "cost": "Free"},
                {"type": "practice", "name": "Kaggle Learn", "platform": "Kaggle", "duration": "2-3 months", "cost": "Free"},
                {"type": "book", "name": "Python for Data Analysis", "platform": "Book", "duration": "2 months", "cost": "$50"}
            ]
        elif skill_level == "mid":
            return [
                {"type": "certification", "name": "Google Data Analytics Certificate", "platform": "Google", "duration": "3 months", "cost": "$39/month"},
                {"type": "course", "name": "Machine Learning Specialization", "platform": "Coursera", "duration": "4 months", "cost": "$49/month"},
                {"type": "practice", "name": "ML Competition Participation", "platform": "Kaggle", "duration": "Ongoing", "cost": "Free"}
            ]
        else:  # senior
            return [
                {"type": "certification", "name": "AWS Machine Learning Specialty", "platform": "AWS", "duration": "2-3 months", "cost": "$300"},
                {"type": "course", "name": "Deep Learning Specialization", "platform": "Coursera", "duration": "5 months", "cost": "$49/month"},
                {"type": "practice", "name": "Research Paper Implementation", "platform": "ArXiv", "duration": "Ongoing", "cost": "Free"}
            ]

    def _get_design_resources(self, skill_level: str) -> List[Dict[str, Any]]:
        """Get design learning resources based on skill level."""
        if skill_level == "entry":
            return [
                {"type": "course", "name": "UI/UX Design Fundamentals", "platform": "Figma", "duration": "4 weeks", "cost": "Free"},
                {"type": "practice", "name": "Daily UI Challenge", "platform": "Daily UI", "duration": "100 days", "cost": "Free"},
                {"type": "book", "name": "Don't Make Me Think", "platform": "Book", "duration": "1 month", "cost": "$25"}
            ]
        elif skill_level == "mid":
            return [
                {"type": "certification", "name": "Google UX Design Certificate", "platform": "Google", "duration": "6 months", "cost": "$39/month"},
                {"type": "course", "name": "Advanced Prototyping", "platform": "Skillshare", "duration": "2 months", "cost": "$15/month"},
                {"type": "practice", "name": "Design System Creation", "platform": "Figma", "duration": "1 month", "cost": "Free"}
            ]
        else:  # senior
            return [
                {"type": "certification", "name": "Nielsen Norman UX Certification", "platform": "NN/g", "duration": "3 months", "cost": "$1,500"},
                {"type": "course", "name": "Design Leadership", "platform": "IDEO U", "duration": "2 months", "cost": "$500"},
                {"type": "practice", "name": "Design Mentorship", "platform": "ADPList", "duration": "Ongoing", "cost": "Free"}
            ]

    def _get_cloud_resources(self, skill_level: str) -> List[Dict[str, Any]]:
        """Get cloud computing learning resources based on skill level."""
        if skill_level == "entry":
            return [
                {"type": "course", "name": "AWS Cloud Practitioner", "platform": "AWS Training", "duration": "1 month", "cost": "Free"},
                {"type": "practice", "name": "AWS Free Tier Projects", "platform": "AWS", "duration": "2 months", "cost": "Free"},
                {"type": "book", "name": "Cloud Computing Basics", "platform": "Book", "duration": "1 month", "cost": "$30"}
            ]
        elif skill_level == "mid":
            return [
                {"type": "certification", "name": "AWS Solutions Architect Associate", "platform": "AWS", "duration": "2-3 months", "cost": "$150"},
                {"type": "course", "name": "Kubernetes Fundamentals", "platform": "Linux Foundation", "duration": "1 month", "cost": "$299"},
                {"type": "practice", "name": "Terraform Infrastructure", "platform": "HashiCorp", "duration": "1 month", "cost": "Free"}
            ]
        else:  # senior
            return [
                {"type": "certification", "name": "AWS Solutions Architect Professional", "platform": "AWS", "duration": "3-4 months", "cost": "$300"},
                {"type": "course", "name": "Advanced Kubernetes", "platform": "CNCF", "duration": "2 months", "cost": "$500"},
                {"type": "practice", "name": "Multi-Cloud Architecture", "platform": "Multiple", "duration": "Ongoing", "cost": "Variable"}
            ]

    def _get_management_resources(self, skill_level: str) -> List[Dict[str, Any]]:
        """Get management learning resources based on skill level."""
        if skill_level == "entry":
            return [
                {"type": "course", "name": "Project Management Fundamentals", "platform": "Coursera", "duration": "4 weeks", "cost": "Free"},
                {"type": "book", "name": "The Manager's Path", "platform": "Book", "duration": "1 month", "cost": "$25"},
                {"type": "practice", "name": "Volunteer Leadership", "platform": "Local Organizations", "duration": "Ongoing", "cost": "Free"}
            ]
        elif skill_level == "mid":
            return [
                {"type": "certification", "name": "PMP Certification", "platform": "PMI", "duration": "3 months", "cost": "$405"},
                {"type": "course", "name": "Agile Leadership", "platform": "Scrum Alliance", "duration": "2 weeks", "cost": "$1,200"},
                {"type": "practice", "name": "Team Lead Experience", "platform": "Workplace", "duration": "6 months", "cost": "Free"}
            ]
        else:  # senior
            return [
                {"type": "certification", "name": "Executive Leadership Program", "platform": "Harvard Business School", "duration": "6 months", "cost": "$15,000"},
                {"type": "course", "name": "Strategic Management", "platform": "Wharton", "duration": "3 months", "cost": "$2,500"},
                {"type": "practice", "name": "Board Advisory Roles", "platform": "Non-profits", "duration": "Ongoing", "cost": "Free"}
            ]

    def _generate_career_insights(self, assessment_results: Dict[str, Any], experience_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate career development insights based on assessment and experience."""
        insights = {
            "career_stage": experience_data.get("skill_level", "entry"),
            "growth_areas": [],
            "strength_leverage": [],
            "market_opportunities": []
        }
        
        # For generic tests, provide behavioral-focused insights
        is_generic_test = experience_data.get("is_generic_test", False)
        if is_generic_test:
            # Generic growth areas for behavioral assessments
            insights["growth_areas"] = [
                "Interpersonal skill development",
                "Self-awareness and reflection"
            ]
            # Keep strength_leverage and market_opportunities empty for generic tests
            return insights
        
        # Analyze performance to determine growth areas for technical tests
        total_score = assessment_results.get("total_score", 0)
        max_score = assessment_results.get("max_score", 100)
        percentage = (total_score / max_score * 100) if max_score > 0 else 0
        
        if percentage < 60:
            insights["growth_areas"].append("Fundamental skill building")
            insights["growth_areas"].append("Knowledge gap identification")
        elif percentage < 80:
            insights["growth_areas"].append("Skill deepening")
            insights["growth_areas"].append("Practical application")
        else:
            insights["growth_areas"].append("Advanced specialization")
            insights["growth_areas"].append("Leadership development")
        
        # Leverage domain expertise for career insights
        domains = experience_data.get("domain_expertise", [])
        if "software_development" in domains:
            insights["strength_leverage"].append("Technical problem-solving")
            insights["market_opportunities"].append("High demand for developers")
        elif "data_analytics" in domains:
            insights["strength_leverage"].append("Data-driven decision making")
            insights["market_opportunities"].append("Growing data science market")
        elif "management" in domains:
            insights["strength_leverage"].append("Team leadership")
            insights["market_opportunities"].append("Leadership roles in tech")
        
        return insights

    def _analyze_skill_gaps(self, assessment_results: Dict[str, Any], experience_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze skill gaps based on assessment performance and experience."""
        gaps = {
            "critical_gaps": [],
            "moderate_gaps": [],
            "minor_gaps": [],
            "recommended_focus": []
        }
        
        # For generic tests, provide behavioral-focused gaps
        is_generic_test = experience_data.get("is_generic_test", False)
        if is_generic_test:
            # No specific gaps for behavioral assessments - just general development areas
            gaps["recommended_focus"] = ["Self-awareness", "Communication skills", "Emotional intelligence"]
            return gaps
        
        # Analyze section scores if available for technical tests
        section_scores = assessment_results.get("section_scores", {})
        if section_scores:
            for section, score in section_scores.items():
                if isinstance(score, (int, float)) and score < 50:
                    gaps["critical_gaps"].append(section)
                elif score < 70:
                    gaps["moderate_gaps"].append(section)
                elif score < 85:
                    gaps["minor_gaps"].append(section)
        
        # Determine recommended focus based on experience level
        skill_level = experience_data.get("skill_level", "entry")
        if skill_level == "entry":
            gaps["recommended_focus"] = ["Fundamental concepts", "Basic skills", "Practical experience"]
        elif skill_level == "mid":
            gaps["recommended_focus"] = ["Advanced techniques", "Specialization", "Leadership skills"]
        else:  # senior
            gaps["recommended_focus"] = ["Strategic thinking", "Mentoring", "Industry expertise"]
        
        return gaps

    def _generate_industry_recommendations(self, assessment_results: Dict[str, Any], experience_data: Dict[str, Any]) -> List[str]:
        """Generate industry-specific recommendations based on domain expertise."""
        recommendations = []
        
        # Skip industry recommendations for generic/behavioral tests
        is_generic_test = experience_data.get("is_generic_test", False)
        if is_generic_test:
            return []  # No job-specific recommendations for behavioral assessments
        
        domains = experience_data.get("domain_expertise", [])
        skill_level = experience_data.get("skill_level", "entry")
        
        if "software_development" in domains:
            if skill_level == "entry":
                recommendations.append("Focus on building a strong portfolio with diverse projects")
                recommendations.append("Contribute to open source projects to gain experience")
            elif skill_level == "mid":
                recommendations.append("Specialize in a specific technology stack or domain")
                recommendations.append("Take on technical leadership responsibilities")
            else:  # senior
                recommendations.append("Mentor junior developers and share knowledge")
                recommendations.append("Consider architect or principal engineer roles")
        
        elif "data_analytics" in domains:
            if skill_level == "entry":
                recommendations.append("Build expertise in statistical analysis and visualization")
                recommendations.append("Work on real-world data projects")
            elif skill_level == "mid":
                recommendations.append("Develop machine learning and AI skills")
                recommendations.append("Focus on business impact and storytelling")
            else:  # senior
                recommendations.append("Lead data strategy and governance initiatives")
                recommendations.append("Mentor data science teams")
        
        elif "management" in domains:
            if skill_level == "entry":
                recommendations.append("Develop strong communication and leadership skills")
                recommendations.append("Take on small team leadership opportunities")
            elif skill_level == "mid":
                recommendations.append("Focus on strategic thinking and decision making")
                recommendations.append("Build cross-functional collaboration skills")
            else:  # senior
                recommendations.append("Develop executive presence and strategic vision")
                recommendations.append("Consider C-level or board positions")
        
        return recommendations[:3]  # Limit to 3 recommendations

    def _enhance_next_steps(self, next_steps: List[str], experience_data: Dict[str, Any]) -> List[str]:
        """Enhance next steps with more specific timelines and resources."""
        enhanced_steps = []
        
        # Ensure next_steps is a list
        if not isinstance(next_steps, list):
            return []
        
        # For generic tests, return steps without domain-specific context
        is_generic_test = experience_data.get("is_generic_test", False)
        if is_generic_test:
            return next_steps  # Return unmodified steps for behavioral assessments
        
        skill_level = experience_data.get("skill_level", "entry")
        domains = experience_data.get("domain_expertise", [])
        
        for step in next_steps:
            # Ensure step is a string, not a list
            if not isinstance(step, str):
                step = str(step) if step else ""
            if not step:
                continue
            
            # Add timeline and specificity based on experience level
            if skill_level == "entry":
                enhanced_step = f"{step} (Timeline: 3-6 months, Focus: Building foundation)"
            elif skill_level == "mid":
                enhanced_step = f"{step} (Timeline: 2-4 months, Focus: Skill deepening)"
            else:  # senior
                enhanced_step = f"{step} (Timeline: 1-3 months, Focus: Advanced application)"
            
            # Add domain-specific context only if domains are detected
            if domains:
                domain_context = f" Leverage your {', '.join(domains[:2])} background."
                enhanced_step += domain_context
            
            enhanced_steps.append(enhanced_step)
        
        return enhanced_steps

    def _build_assessment_details(self, assessment_results: Dict[str, Any]) -> str:
        """Build detailed assessment information for the prompt."""
        details = []
        
        # Add section scores if available
        if "section_scores" in assessment_results:
            section_scores = assessment_results["section_scores"]
            if isinstance(section_scores, dict):
                for section, score in section_scores.items():
                    details.append(f"- {section}: {score}")
        
        # Add question analysis if available
        if "question_analysis" in assessment_results:
            qa = assessment_results["question_analysis"]
            if isinstance(qa, dict):
                correct = qa.get("correct", 0)
                total = qa.get("total", 0)
                if total > 0:
                    details.append(f"- Questions answered correctly: {correct}/{total}")
        
        # Add time taken if available
        if "time_taken" in assessment_results:
            time_taken = assessment_results["time_taken"]
            details.append(f"- Time taken: {time_taken}")
        
        return "\n".join(details) if details else "- Detailed breakdown not available"

    def _determine_performance_level(self, percentage: float) -> str:
        """Determine performance level based on percentage."""
        if percentage >= 85:
            return "Excellent"
        elif percentage >= 70:
            return "Good"
        elif percentage >= 55:
            return "Satisfactory"
        else:
            return "Needs Improvement"

    def _validate_report_quality(self, result: Dict[str, Any], assessment_results: Dict[str, Any]) -> float:
        """Validate the quality of generated report content."""
        quality_checks = []
        
        # Check summary quality - ensure it's a string
        summary = result.get("summary", "")
        if not isinstance(summary, str):
            summary = str(summary) if summary else ""
        if len(summary) > 50 and not self._is_generic_text(summary):
            quality_checks.append(0.2)
        else:
            quality_checks.append(0.0)
        
        # Check topics quality
        topics = result.get("topics", [])
        if topics and isinstance(topics, list):
            topic_quality = 0
            for topic in topics:
                if isinstance(topic, dict):
                    # Check if topic has required fields
                    if all(key in topic for key in ["topic", "strengths", "areas_for_improvement", "recommendations"]):
                        # Check content specificity - ensure all items are strings before joining
                        strengths = topic.get("strengths", [])
                        improvements = topic.get("areas_for_improvement", [])
                        recommendations = topic.get("recommendations", [])
                        
                        # Convert all items to strings safely
                        strengths_str = " ".join(str(s) for s in strengths if s) if isinstance(strengths, list) else ""
                        improvements_str = " ".join(str(i) for i in improvements if i) if isinstance(improvements, list) else ""
                        recommendations_str = " ".join(str(r) for r in recommendations if r) if isinstance(recommendations, list) else ""
                        
                        if (len(strengths) >= 2 and len(improvements) >= 2 and len(recommendations) >= 2 and
                            not self._is_generic_text(strengths_str) and
                            not self._is_generic_text(improvements_str) and
                            not self._is_generic_text(recommendations_str)):
                            topic_quality += 1
            
            if topics:
                quality_checks.append(min(0.3, (topic_quality / len(topics)) * 0.3))
            else:
                quality_checks.append(0.0)
        else:
            quality_checks.append(0.0)
        
        # Check feedback quality - ensure it's a string
        feedback = result.get("feedback", "")
        if not isinstance(feedback, str):
            feedback = str(feedback) if feedback else ""
        if len(feedback) > 200 and not self._is_generic_text(feedback):
            quality_checks.append(0.2)
        else:
            quality_checks.append(0.0)
        
        # Check next steps quality - ensure all items are strings
        next_steps = result.get("suggested_next_steps", [])
        if isinstance(next_steps, list) and len(next_steps) >= 3:
            # Convert all steps to strings safely
            steps_str = [str(step) for step in next_steps if step]
            if all(not self._is_generic_text(step) for step in steps_str):
                quality_checks.append(0.2)
            else:
                quality_checks.append(0.0)
        else:
            quality_checks.append(0.0)
        
        # Check personalization (based on assessment results)
        personalization_score = self._check_personalization(result, assessment_results)
        quality_checks.append(personalization_score * 0.1)
        
        return sum(quality_checks)

    def _is_generic_text(self, text: str) -> bool:
        """Check if text contains generic, non-specific content."""
        generic_phrases = [
            "good job", "well done", "keep it up", "continue learning",
            "practice more", "study harder", "improve your skills",
            "work on your weaknesses", "build on your strengths",
            "focus on improvement", "keep practicing", "good work"
        ]
        
        text_lower = text.lower()
        generic_count = sum(1 for phrase in generic_phrases if phrase in text_lower)
        
        # If more than 20% of words are generic phrases, consider it generic
        word_count = len(text.split())
        return (generic_count / max(word_count, 1)) > 0.2

    def _check_personalization(self, result: Dict[str, Any], assessment_results: Dict[str, Any]) -> float:
        """Check how well the report is personalized to the assessment results."""
        personalization_score = 0
        
        # Check if feedback references specific scores or metrics - ensure feedback is a string
        feedback = result.get("feedback", "")
        if not isinstance(feedback, str):
            feedback = str(feedback) if feedback else ""
        total_score = assessment_results.get("total_score", 0)
        max_score = assessment_results.get("max_score", 100)
        
        if str(total_score) in feedback or str(max_score) in feedback:
            personalization_score += 1
        
        # Check if topics match assessment topics
        assessment_topics = assessment_results.get("topics", [])
        result_topics = result.get("topics", [])
        
        if assessment_topics and result_topics:
            topic_match = 0
            for result_topic in result_topics:
                if isinstance(result_topic, dict):
                    topic_name = result_topic.get("topic", "").lower()
                    for assessment_topic in assessment_topics:
                        if isinstance(assessment_topic, str) and assessment_topic.lower() in topic_name:
                            topic_match += 1
                            break
            
            if result_topics:
                personalization_score += (topic_match / len(result_topics)) * 2
        
        return min(personalization_score, 10) / 10  # Normalize to 0-1

    def _enhance_prompt_for_quality(self, original_prompt: str, quality_score: float) -> str:
        """Enhance prompt based on quality issues."""
        enhancement = """
        
QUALITY IMPROVEMENT INSTRUCTIONS:
- Provide SPECIFIC examples from the assessment data, not generic advice
- Include concrete details about what the candidate did well or poorly
- Make recommendations ACTIONABLE with clear steps and timelines
- Personalize feedback to the candidate's experience level and domain
- Avoid generic phrases like "good job" or "practice more"
- Connect feedback directly to career advancement opportunities
- Provide detailed explanations for why each area matters
"""
        
        return original_prompt + enhancement

    def _get_content_validation_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of content validation results."""
        # Safely extract string fields, ensuring they're strings
        summary = result.get("summary", "")
        if not isinstance(summary, str):
            summary = str(summary) if summary else ""
        
        feedback = result.get("feedback", "")
        if not isinstance(feedback, str):
            feedback = str(feedback) if feedback else ""
        
        return {
            "summary_length": len(summary),
            "topics_count": len(result.get("topics", [])) if isinstance(result.get("topics"), list) else 0,
            "feedback_length": len(feedback),
            "next_steps_count": len(result.get("suggested_next_steps", [])) if isinstance(result.get("suggested_next_steps"), list) else 0,
            "has_specific_examples": not self._is_generic_text(summary + " " + feedback),
            "validation_timestamp": time.time()
        }

    def _fallback_report(self, reason: str) -> Dict[str, Any]:
        """Generate deterministic fallback report for critical failures."""
        return {
            "total_score": 0,
            "max_score": 0,
            "summary": "Automated report generation encountered a technical issue.",
            "performance_level": "Incomplete",
            "topics": [],
            "feedback": f"Report generation failed due to: {reason}. Please try again or contact support.",
            "suggested_next_steps": ["Verify your connection and try submitting again."]
        }

    def _normalize_report_scores_from_assessment(
        self,
        result: Dict[str, Any],
        assessment_results: Dict[str, Any],
        is_generic_test: bool = False,
    ) -> Dict[str, Any]:
        """Overwrite report total_score, max_score, and performance_level from assessment_results.

        Ensures the Assessment Intelligence Report page shows the same score/level as the
        Results page (single source of truth: assessment evaluator output).
        Only applies when assessment_results has canonical total_score/max_score (traditional
        assessments); skips or applies gently for generic tests.
        """
        if not isinstance(result, dict) or not isinstance(assessment_results, dict):
            return result
        canonical_total = assessment_results.get("total_score")
        canonical_max = assessment_results.get("max_score", 100)
        if canonical_total is None and is_generic_test:
            return result
        if canonical_total is None:
            return result

        def _to_float(val: Any, default: float = 0.0) -> float:
            try:
                if val is None:
                    return default
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, str):
                    return float(val.strip().replace("%", ""))
                return default
            except Exception:
                return default

        total = _to_float(canonical_total, 0.0)
        max_score = _to_float(canonical_max, 100.0)
        if max_score <= 0:
            max_score = 100.0
        result["total_score"] = total
        result["max_score"] = max_score
        percentage = (total / max_score * 100.0) if max_score > 0 else 0.0
        # Same thresholds as _deterministic_report for consistent display
        if percentage >= 90:
            result["performance_level"] = "Excellent"
        elif percentage >= 75:
            result["performance_level"] = "Good"
        elif percentage >= 60:
            result["performance_level"] = "Satisfactory"
        else:
            result["performance_level"] = "Needs Improvement"
        # Preserve only compact-report-relevant evaluator metadata so downstream
        # rendering can distinguish refinement from remediation without bloating
        # cached/session-stored report payloads with full per-question details.
        compact_metadata_keys = [
            "weighted_score",
            "weighted_max_score",
            "weighted_score_percentage",
            "total_questions",
            "answered_questions",
            "attempted_questions",
            "fully_correct_questions",
            "partially_correct_questions",
            "incorrect_questions",
            "correct_answers",
            "incorrect_answers",
            "question_correctness_rate",
            "score_explanation",
        ]
        compact_assessment_results = {
            key: assessment_results[key]
            for key in compact_metadata_keys
            if key in assessment_results
        }
        if compact_assessment_results:
            result["assessment_results"] = compact_assessment_results
        log.debug(f"Normalized report scores from assessment: {total}/{max_score} ({percentage:.1f}%)")
        return result

    def _deterministic_report(self, assessment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced deterministic report from assessment results."""
        total = assessment_results.get("total_score", 0)
        max_score = assessment_results.get("max_score", 100)
        percentage = (total / max_score * 100) if max_score > 0 else 0
        weighted_score_text = _build_weighted_proficiency_text(total, max_score, percentage)
        count_sentence = _build_assessment_count_sentence(assessment_results)
        score_explanation = _normalize_sentence(assessment_results.get("score_explanation"))

        if percentage >= 90:
            level = "Excellent"
            summary = f"Demonstrated an excellent understanding with a {weighted_score_text}."
            feedback_text = "Exceptional work. You have a strong grasp of the material. Consider exploring advanced topics in this area to further deepen your expertise."
            next_steps = ["Explore advanced certifications.", "Mentor others in this subject."]
        elif percentage >= 75:
            level = "Good"
            summary = f"Showed a good grasp of the subject with a {weighted_score_text}."
            feedback_text = "Solid performance. You have a good foundation but there are some areas where you can strengthen your knowledge. Focus on the recommended topics for improvement."
            next_steps = ["Review the specific areas for improvement.", "Attempt more complex practice problems."]
        elif percentage >= 60:
            level = "Satisfactory"
            summary = f"Achieved a satisfactory {weighted_score_text}, indicating foundational knowledge."
            feedback_text = "You have a basic understanding, but there are significant gaps. A thorough review of the fundamental concepts is recommended to build a stronger base."
            next_steps = ["Revisit foundational course materials.", "Seek clarification on difficult topics."]
        else:
            level = "Needs Improvement"
            summary = f"Performance indicates a need for significant review based on a {weighted_score_text}."
            feedback_text = "There are critical gaps in your understanding of the core concepts. It is highly recommended to start by reviewing the fundamental materials to build a solid foundation before proceeding."
            next_steps = ["Schedule a review session with a mentor.", "Start with introductory courses on the subject."]
        if count_sentence:
            summary = f"{summary} {count_sentence}"
            feedback_text = f"{feedback_text} {count_sentence}"
        if score_explanation:
            feedback_text = f"{feedback_text} {score_explanation}"

        report = {
            "total_score": total,
            "max_score": max_score,
            "summary": summary,
            "performance_level": level,
            "topics": [],
            "feedback": feedback_text,
            "suggested_next_steps": next_steps,
        }
        
        # Apply spell checking to deterministic report
        spell_checker = ReportSpellChecker()
        corrected_report = spell_checker.validate_report_content(report)
        
        return corrected_report

    def _generate_generic_test_report(self, assessment_results: Dict[str, Any], candidate_name: str) -> Dict[str, Any]:
        """Generate behavioral assessment report for generic tests with constructive feedback."""
        test_type = assessment_results.get("test_type", "assessment")
        constructive_feedback = assessment_results.get("constructive_feedback", {})
        interpretations = assessment_results.get("interpretations", {})
        recommendations = assessment_results.get("recommendations", [])
        total_questions = assessment_results.get("total_questions", 0)
        answered_questions = assessment_results.get("answered_questions", 0)
        
        # Extract constructive feedback components
        summary = constructive_feedback.get("summary", f"{candidate_name} completed a {test_type.replace('_', ' ').title()} assessment.")
        strengths = constructive_feedback.get("strengths", [])
        areas_for_development = constructive_feedback.get("areas_for_development", [])
        behavioral_insights = constructive_feedback.get("behavioral_insights", [])
        development_focus = constructive_feedback.get("development_focus", [])
        
        # Determine performance level based on strengths vs areas for development
        if len(strengths) > len(areas_for_development) * 2:
            performance_level = "Excellent"
        elif len(strengths) > len(areas_for_development):
            performance_level = "Good"
        elif len(areas_for_development) > len(strengths) * 2:
            performance_level = "Needs Development"
        else:
            performance_level = "Satisfactory"
        
        # Generate detailed feedback
        feedback_parts = []
        
        # Add summary
        feedback_parts.append(f"**Assessment Summary:**\n{summary}")
        
        # Add behavioral insights
        if behavioral_insights:
            feedback_parts.append("\n**Behavioral Insights:**")
            for insight in behavioral_insights:
                feedback_parts.append(f"• {insight}")
        
        # Add strengths
        if strengths:
            feedback_parts.append("\n**Key Strengths:**")
            for strength in strengths:
                if isinstance(strength, dict):
                    trait = strength.get("trait", "")
                    insight = strength.get("insight", "")
                    feedback_parts.append(f"• **{trait}**: {insight}")
                else:
                    feedback_parts.append(f"• {strength}")
        
        # Add areas for development
        if areas_for_development:
            feedback_parts.append("\n**Areas for Development:**")
            for area in areas_for_development:
                if isinstance(area, dict):
                    trait = area.get("trait", "")
                    insight = area.get("insight", "")
                    feedback_parts.append(f"• **{trait}**: {insight}")
                else:
                    feedback_parts.append(f"• {area}")
        
        # Add development focus
        if development_focus:
            feedback_parts.append("\n**Development Focus:**")
            for focus in development_focus:
                feedback_parts.append(f"• {focus}")
        
        # Add recommendations
        if recommendations:
            feedback_parts.append("\n**Development Recommendations:**")
            for i, rec in enumerate(recommendations, 1):
                feedback_parts.append(f"{i}. {rec}")
        
        # Add completion information
        if total_questions > 0:
            completion_rate = (answered_questions / total_questions) * 100
            feedback_parts.append(f"\n**Assessment Completion**: {answered_questions}/{total_questions} questions answered ({completion_rate:.1f}%)")
        
        feedback_text = "\n".join(feedback_parts) if feedback_parts else "Behavioral assessment completed successfully."
        
        # Generate suggested next steps
        next_steps = []
        if development_focus:
            next_steps.extend(development_focus[:3])  # Take top 3 development focus areas
        
        if recommendations:
            next_steps.extend(recommendations[:2])  # Add top 2 recommendations
        
        # Add generic next steps based on performance level
        if performance_level == "Excellent":
            next_steps.extend([
                "Consider mentoring others in this area",
                "Explore advanced behavioral assessments"
            ])
        elif performance_level == "Good":
            next_steps.extend([
                "Continue developing identified strengths",
                "Focus on areas for improvement"
            ])
        else:
            next_steps.extend([
                "Focus on fundamental skill development",
                "Consider additional training or coaching"
            ])
        
        # Create topic feedback for each trait
        topics = []
        # Build topics purely from qualitative feedback (no numeric scores)
        strengths = constructive_feedback.get("strengths", []) or []
        areas_for_development = constructive_feedback.get("areas_for_development", []) or []
        for strength in strengths:
            if isinstance(strength, dict):
                trait = (strength.get("trait") or "").strip()
                insight = (strength.get("insight") or "").strip()
                if trait:
                    trait_display = trait.replace("_", " ").title()
                    topics.append({
                        "topic": trait_display,
                        "strengths": [insight] if insight else [f"Strength in {trait_display.lower()}"],
                        "areas_for_improvement": [],
                        "recommendations": [interpretations.get(trait, "")] if interpretations.get(trait) else []
                    })
        for area in areas_for_development:
            if isinstance(area, dict):
                trait = (area.get("trait") or "").strip()
                insight = (area.get("insight") or "").strip()
                if trait:
                    trait_display = trait.replace("_", " ").title()
                    topics.append({
                        "topic": trait_display,
                        "strengths": [],
                        "areas_for_improvement": [insight] if insight else [f"Focus on improving {trait_display.lower()}"],
                        "recommendations": [interpretations.get(trait, "")] if interpretations.get(trait) else []
                    })
        
        # Convert third-person references to second-person for better personalization
        summary = self._convert_to_second_person(summary) if summary else summary
        feedback_text = self._convert_to_second_person(feedback_text) if feedback_text else feedback_text
        
        # Also convert topics text fields
        for topic_entry in topics:
            if isinstance(topic_entry, dict):
                for field in ["strengths", "areas_for_improvement", "recommendations"]:
                    if field in topic_entry and isinstance(topic_entry[field], list):
                        topic_entry[field] = [
                            self._convert_to_second_person(str(item)) if isinstance(item, str) else item
                            for item in topic_entry[field]
                        ]
        
        return {
            "summary": summary,
            "performance_level": performance_level,
            "topics": topics,
            "feedback": feedback_text,
            "suggested_next_steps": next_steps,
            # Ensure new optional fields are always present with safe defaults
            "subtopic_breakdown": [],
            "misconceptions": [],
            "rubric_scores": {},
            "resources": [],
            "test_type": test_type,
            "constructive_feedback": constructive_feedback,
            "interpretations": interpretations,
            "evaluation_method": "behavioral_scoring"
        }

    async def _build_communication_mcq_report(self, assessment_results: Dict[str, Any], candidate_name: str, state: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Deterministic report for communication MCQ assessments using trait scores.

        Uses existing top-level keys only; content is tailored for MCQ-derived traits.
        Now with LLM-based personalized feedback integration.
        """
        # Normalize numeric fields
        def _to_float(value, default=0.0):
            try:
                if value is None:
                    return default
                if isinstance(value, str):
                    return float(value.strip().replace('%', ''))
                return float(value)
            except Exception:
                return default

        trait_scores = assessment_results.get("trait_scores", {}) or {}
        if not isinstance(trait_scores, dict):
            trait_scores = {}

        # Qualitative performance level based on trait distribution (no numeric totals)
        high_count = sum(1 for s in trait_scores.values() if _to_float(s, 0.0) >= 3.5)
        low_count = sum(1 for s in trait_scores.values() if _to_float(s, 0.0) <= 2.5)
        if high_count >= low_count * 2:
            performance_level = "Excellent"
        elif high_count > low_count:
            performance_level = "Good"
        elif low_count >= high_count * 2:
            performance_level = "Needs Development"
        else:
            performance_level = "Satisfactory"

        # Build topics from top traits
        sorted_traits = sorted(trait_scores.items(), key=lambda kv: _to_float(kv[1], 0.0), reverse=True)
        top_traits = sorted_traits[:3]
        topics = []
        for trait, score in top_traits:
            topics.append({
                "topic": str(trait).replace('_', ' ').title(),
                "strengths": [f"Evidence of strength in {trait.replace('_', ' ')} based on MCQ patterns"],
                "areas_for_improvement": [f"Practice scenarios to reinforce {trait.replace('_', ' ')} in meetings and written updates"],
                "recommendations": [
                    f"Apply {trait.replace('_', ' ')} in 2 real interactions this week and request feedback"
                ]
            })

        # Feedback text oriented to MCQ traits
        feedback = (
            f"This communication assessment summarizes your style based on multiple-choice patterns. "
            f"Top strengths reflect higher scores in traits like {', '.join([t for t, _ in top_traits])}. "
            f"Focus improvement on lower-scoring traits with targeted practice in real conversations."
        )

        # Default next steps (fallback)
        next_steps = [
            "Use the 'reflect-and-repeat' technique in 3 conversations this week",
            "Send one concise written update using BLUF (Bottom Line Up Front)",
            "Ask one clarifying question per meeting to confirm understanding",
            "Summarize decisions and owners at the end of each meeting",
            "Schedule a peer feedback check-in after applying these actions"
        ]
        
        # Try to generate personalized, topic-wise feedback using LLM
        use_llm_feedback = getattr(self.config, "USE_LLM_FEEDBACK", True)
        if use_llm_feedback:
            try:
                # Extract candidate profile
                candidate_profile = await self._extract_experience_context(assessment_results, state, use_llm_detection=True)
                
                # Prepare assessment results for LLM with topics
                assessment_for_llm = {
                    "test_type": assessment_results.get("test_type", "communication_assessment"),
                    "topics": topics,
                    "trait_scores": trait_scores,
                    "performance_level": performance_level
                }
                
                # Generate personalized feedback
                personalized = await self._generate_personalized_feedback_with_llm(
                    assessment_results=assessment_for_llm,
                    candidate_profile=candidate_profile,
                    test_type="communication_assessment"
                )
                
                # Use LLM-generated feedback and steps
                if personalized:
                    feedback = personalized.get("feedback", feedback)
                    next_steps = personalized.get("suggested_next_steps", next_steps)
                    
                    # If topic-wise steps are available, include them
                    result = {
                        "summary": f"Communication profile derived from MCQ responses for {candidate_name}.",
                        "performance_level": performance_level,
                        "topics": topics,
                        "feedback": feedback,
                        "suggested_next_steps": next_steps
                    }
                    
                    if "suggested_next_steps_by_topic" in personalized:
                        result["suggested_next_steps_by_topic"] = personalized["suggested_next_steps_by_topic"]
                    
                    return result
            except Exception as e:
                log.warning(f"LLM feedback generation failed, using default: {e}")

        return {
            "summary": f"Communication profile derived from MCQ responses for {candidate_name}.",
            "performance_level": performance_level,
            "topics": topics,
            "feedback": feedback,
            "suggested_next_steps": next_steps
        }

    async def _build_personality_mcq_report(self, assessment_results: Dict[str, Any], candidate_name: str, state: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Deterministic report for Personality MCQ assessments using trait scores.
        
        Now with LLM-based personalized feedback integration.
        """
        def _to_float(value, default=0.0):
            try:
                if value is None:
                    return default
                if isinstance(value, str):
                    return float(value.strip().replace('%', ''))
                return float(value)
            except Exception:
                return default

        trait_scores = assessment_results.get("trait_scores", {}) or {}
        if not isinstance(trait_scores, dict):
            trait_scores = {}

        total_score = _to_float(assessment_results.get("total_score", 0.0), 0.0)
        max_score = _to_float(assessment_results.get("max_score", 100.0), 100.0)
        if max_score <= 0:
            if trait_scores:
                avg = sum(_to_float(s, 0.0) for s in trait_scores.values()) / max(1, len(trait_scores))
                total_score = round(((avg - 1) / 3) * 100, 2)
                max_score = 100.0
            else:
                total_score, max_score = 0.0, 100.0

        percentage = (total_score / max_score * 100) if max_score > 0 else 0.0
        if percentage >= 85:
            performance_level = "Excellent"
        elif percentage >= 70:
            performance_level = "Good"
        elif percentage >= 55:
            performance_level = "Satisfactory"
        else:
            performance_level = "Needs Development"

        sorted_traits = sorted(trait_scores.items(), key=lambda kv: _to_float(kv[1], 0.0), reverse=True)
        top_traits = sorted_traits[:3]
        topics = []
        for trait, score in top_traits:
            readable = str(trait).replace('_', ' ').title()
            topics.append({
                "topic": readable,
                "strengths": [f"Prominent {readable.lower()} indicated by MCQ patterns"],
                "areas_for_improvement": [f"Balance {readable.lower()} with complementary behaviors on team tasks"],
                "recommendations": [f"Choose one weekly scenario to apply {readable.lower()} deliberately and capture outcomes"]
            })

        feedback = (
            "This personality profile is derived from MCQ responses. Use it to align work style, improve collaboration, "
            "and make role choices that leverage your top strengths."
        )

        # Default next steps (fallback)
        next_steps = [
            "Pick one strength to leverage and one to balance this week",
            "Share preferred collaboration norms with your team and align",
            "Do a 10‑minute weekly reflection on where traits helped/hindered",
            "Pair with a complementary teammate on one deliverable",
            "Map responsibilities to strengths where feasible"
        ]
        
        # Try to generate personalized, topic-wise feedback using LLM
        use_llm_feedback = getattr(self.config, "USE_LLM_FEEDBACK", True)
        if use_llm_feedback:
            try:
                candidate_profile = await self._extract_experience_context(assessment_results, state, use_llm_detection=True)
                assessment_for_llm = {
                    "test_type": assessment_results.get("test_type", "personality_assessment"),
                    "topics": topics,
                    "trait_scores": trait_scores,
                    "performance_level": performance_level
                }
                personalized = await self._generate_personalized_feedback_with_llm(
                    assessment_results=assessment_for_llm,
                    candidate_profile=candidate_profile,
                    test_type="personality_assessment"
                )
                if personalized:
                    feedback = personalized.get("feedback", feedback)
                    next_steps = personalized.get("suggested_next_steps", next_steps)
                    result = {
                        "summary": f"Personality profile derived from MCQ responses for {candidate_name}.",
                        "performance_level": performance_level,
                        "topics": topics,
                        "feedback": feedback,
                        "suggested_next_steps": next_steps
                    }
                    if "suggested_next_steps_by_topic" in personalized:
                        result["suggested_next_steps_by_topic"] = personalized["suggested_next_steps_by_topic"]
                    return result
            except Exception as e:
                log.warning(f"LLM feedback generation failed, using default: {e}")

        return {
            "summary": f"Personality profile derived from MCQ responses for {candidate_name}.",
            "performance_level": performance_level,
            "topics": topics,
            "feedback": feedback,
            "suggested_next_steps": next_steps
        }

        
    async def _build_psychometric_mcq_report(self, assessment_results: Dict[str, Any], candidate_name: str, state: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Deterministic report for Psychometric MCQ assessments using capability/trait scores.
        
        Now with LLM-based personalized feedback integration.
        """
        def _to_float(value, default=0.0):
            try:
                if value is None:
                    return default
                if isinstance(value, str):
                    return float(value.strip().replace('%', ''))
                return float(value)
            except Exception:
                return default

        trait_scores = assessment_results.get("trait_scores", {}) or {}
        if not isinstance(trait_scores, dict):
            trait_scores = {}

        total_score = _to_float(assessment_results.get("total_score", 0.0), 0.0)
        max_score = _to_float(assessment_results.get("max_score", 100.0), 100.0)
        if max_score <= 0:
            if trait_scores:
                avg = sum(_to_float(s, 0.0) for s in trait_scores.values()) / max(1, len(trait_scores))
                total_score = round(((avg - 1) / 3) * 100, 2)
                max_score = 100.0
            else:
                total_score, max_score = 0.0, 100.0

        percentage = (total_score / max_score * 100) if max_score > 0 else 0.0
        if percentage >= 85:
            performance_level = "Excellent"
        elif percentage >= 70:
            performance_level = "Good"
        elif percentage >= 55:
            performance_level = "Satisfactory"
        else:
            performance_level = "Needs Development"

        sorted_traits = sorted(trait_scores.items(), key=lambda kv: _to_float(kv[1], 0.0), reverse=True)
        top_traits = sorted_traits[:3]
        topics = []
        for trait, score in top_traits:
            readable = str(trait).replace('_', ' ').title()
            topics.append({
                "topic": readable,
                "strengths": [f"Strong {readable.lower()} under standard conditions"],
                "areas_for_improvement": [f"Increase {readable.lower()} consistency under time pressure"],
                "recommendations": [f"Run two timed drills for {readable.lower()} and log accuracy vs. speed"]
            })

        feedback = (
            "This psychometric snapshot reflects work‑style and cognitive tendencies from MCQ responses. "
            "Use targeted drills to improve reliability and pacing."
        )

        # Default next steps (fallback)
        next_steps = [
            "Do two 20‑minute timed practice sets in weaker areas",
            "Apply a pacing rule (e.g., 90s/item) and track adherence",
            "Classify each miss as knowledge/process/attention and address",
            "Add checkpoints every 5 items to recalibrate speed",
            "Schedule a weekly mock and track accuracy trend"
        ]
        
        # Try to generate personalized, topic-wise feedback using LLM
        use_llm_feedback = getattr(self.config, "USE_LLM_FEEDBACK", True)
        if use_llm_feedback:
            try:
                candidate_profile = await self._extract_experience_context(assessment_results, state, use_llm_detection=True)
                assessment_for_llm = {
                    "test_type": assessment_results.get("test_type", "psychometric_assessment"),
                    "topics": topics,
                    "trait_scores": trait_scores,
                    "performance_level": performance_level
                }
                personalized = await self._generate_personalized_feedback_with_llm(
                    assessment_results=assessment_for_llm,
                    candidate_profile=candidate_profile,
                    test_type="psychometric_assessment"
                )
                if personalized:
                    feedback = personalized.get("feedback", feedback)
                    next_steps = personalized.get("suggested_next_steps", next_steps)
                    result = {
                        "summary": f"Psychometric profile derived from MCQ responses for {candidate_name}.",
                        "performance_level": performance_level,
                        "topics": topics,
                        "feedback": feedback,
                        "suggested_next_steps": next_steps
                    }
                    if "suggested_next_steps_by_topic" in personalized:
                        result["suggested_next_steps_by_topic"] = personalized["suggested_next_steps_by_topic"]
                    return result
            except Exception as e:
                log.warning(f"LLM feedback generation failed, using default: {e}")

        return {
            "summary": f"Psychometric profile derived from MCQ responses for {candidate_name}.",
            "performance_level": performance_level,
            "topics": topics,
            "feedback": feedback,
            "suggested_next_steps": next_steps
        }

    @traceable(name="generate_enhanced_generic_test_report")
    async def _generate_enhanced_generic_test_report(self, assessment_results: Dict[str, Any], candidate_name: str, tenant_id: str, state: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Generate enhanced behavioral assessment report using LLM for deeper insights."""
        test_type = assessment_results.get("test_type", "assessment")
        # MCQ-only generic tests: ALWAYS route deterministic by type and enrich, regardless of free-text presence
        try:
            if isinstance(test_type, str):
                t = test_type.lower()
                if any(k in t for k in ["communication", "personality", "psychometric"]):
                    if "communication" in t:
                        base = await self._build_communication_mcq_report(assessment_results, candidate_name, state)
                    elif "personality" in t:
                        base = await self._build_personality_mcq_report(assessment_results, candidate_name, state)
                    else:
                        base = await self._build_psychometric_mcq_report(assessment_results, candidate_name, state)
                    try:
                        experience_data = await self._extract_experience_context(assessment_results, state)
                        base = await self._enrich_report_content(base, assessment_results, experience_data)
                    except Exception:
                        pass
                    return base
        except Exception:
            pass
        constructive_feedback = assessment_results.get("constructive_feedback", {})
        trait_scores = assessment_results.get("trait_scores", {})
        interpretations = assessment_results.get("interpretations", {})
        recommendations = assessment_results.get("recommendations", [])
        total_questions = assessment_results.get("total_questions", 0)
        answered_questions = assessment_results.get("answered_questions", 0)
        
        # Extract constructive feedback components
        summary = constructive_feedback.get("summary", f"{candidate_name} completed a {test_type.replace('_', ' ').title()} assessment.")
        strengths = constructive_feedback.get("strengths", [])
        areas_for_development = constructive_feedback.get("areas_for_development", [])
        behavioral_insights = constructive_feedback.get("behavioral_insights", [])
        development_focus = constructive_feedback.get("development_focus", [])
        
        # Create LLM prompt for enhanced report generation
        # Build topic context to force topic-based feedback
        topic_list = assessment_results.get("topics") or []
        normalized_topics = []
        for t in topic_list:
            if isinstance(t, dict):
                normalized_topics.append(t.get("topic"))
            else:
                normalized_topics.append(str(t))
        subtopic_breakdown = assessment_results.get("subtopic_breakdown", []) or []
        # Limit context size to avoid token bloat
        subtopic_context = [
            {"name": s.get("name"), "notes": s.get("notes")}
            for s in subtopic_breakdown if isinstance(s, dict)
        ][:10]

        prompt = f"""
You are an expert behavioral psychologist and career development specialist. Generate a comprehensive, professional behavioral assessment report for {candidate_name}.

ASSESSMENT TYPE: {test_type.replace('_', ' ').title()}

TOPIC CONTEXT:
- Topics (use these exact names when relevant): {json.dumps([n for n in normalized_topics if n])}
- Subtopic notes (weave into analysis concisely): {json.dumps(subtopic_context)}

BEHAVIORAL ANALYSIS DATA:
- Summary: {summary}
- Strengths: {json.dumps(strengths, indent=2)}
- Areas for Development: {json.dumps(areas_for_development, indent=2)}
- Behavioral Insights: {json.dumps(behavioral_insights, indent=2)}
- Development Focus: {json.dumps(development_focus, indent=2)}
- Recommendations: {json.dumps(recommendations, indent=2)}
- Trait Scores: {json.dumps(trait_scores, indent=2)}
- Interpretations: {json.dumps(interpretations, indent=2)}

REPORT REQUIREMENTS:
1. Create a professional, comprehensive behavioral assessment report
2. Focus on career development and professional growth
3. Provide actionable insights and recommendations
4. Use a positive, constructive tone
5. Include specific examples and practical advice
6. Structure the report for maximum impact and clarity
7. SECOND-PERSON VOICE: Write all sections in second-person (use "Your", "You", "Your performance", etc.) - NEVER use "The Candidate" or third-person references. Sections should start with "Your" not "The Candidate."
8. TOPIC-SPECIFICITY: Explicitly reference at least two items from TOPIC CONTEXT (topic or subtopic names) in BOTH the executive_summary and feedback. Avoid generic phrasing.

OUTPUT FORMAT (JSON):
{{
    "executive_summary": "Brief overview of the assessment results and key findings",
    "detailed_analysis": {{
        "strengths_analysis": "Comprehensive analysis of identified strengths with career implications",
        "development_areas": "Detailed analysis of areas for development with specific improvement strategies",
        "behavioral_patterns": "Analysis of behavioral patterns and their professional implications",
        "career_implications": "How these traits impact career success and development"
    }},
    "professional_recommendations": [
        "Specific, actionable professional development recommendations"
    ],
    "career_development_plan": [
        "Structured career development steps and milestones"
    ],
    "next_steps": [
        "Immediate actionable next steps for the candidate"
    ],
    "performance_level": "Overall performance assessment (Excellent/Good/Satisfactory/Needs Development)",
    "confidence_score": 0.95
}}

Generate a comprehensive, professional report that provides maximum value for career development.
"""
        
        try:
            # Call LLM for enhanced report generation
            llm_response = await invoke_llm(
            prompt=prompt,
            task_type="report_generation",
            agent_name="report_generator"
        )
            content = getattr(llm_response, "content", str(llm_response)).strip()
            
            if content:
                # Try to extract JSON from response
                try:
                    enhanced_report = extract_json_robust(content)
                    
                    # Merge with base report structure
                    base_report = self._generate_generic_test_report(assessment_results, candidate_name)
                    
                    # Enhance with LLM insights (excluding total_score and max_score for generic tests)
                    enhanced_report.update({
                        "summary": enhanced_report.get("executive_summary", base_report.get("summary", "")),
                        "performance_level": enhanced_report.get("performance_level", base_report.get("performance_level", "Satisfactory")),
                        "topics": base_report.get("topics", []),
                        "feedback": enhanced_report.get("detailed_analysis", {}).get("strengths_analysis", "") + "\n\n" + 
                                  enhanced_report.get("detailed_analysis", {}).get("development_areas", ""),
                        "suggested_next_steps": enhanced_report.get("next_steps", base_report.get("suggested_next_steps", [])),
                        # New optional fields with safe defaults
                        "subtopic_breakdown": enhanced_report.get("subtopic_breakdown", base_report.get("subtopic_breakdown", [])) or [],
                        "misconceptions": enhanced_report.get("misconceptions", base_report.get("misconceptions", [])) or [],
                        "rubric_scores": enhanced_report.get("rubric_scores", base_report.get("rubric_scores", {})) or {},
                        "resources": enhanced_report.get("resources", base_report.get("resources", [])) or [],
                        "test_type": test_type,
                        "constructive_feedback": constructive_feedback,
                        "trait_scores": trait_scores,
                        "interpretations": interpretations,
                        "evaluation_method": "llm_behavioral_analysis",
                        "enhanced_insights": enhanced_report.get("detailed_analysis", {}),
                        "professional_recommendations": enhanced_report.get("professional_recommendations", []),
                        "career_development_plan": enhanced_report.get("career_development_plan", []),
                        "confidence_score": enhanced_report.get("confidence_score", 0.9)
                    })
                    
                    # Enrich with experience-aware insights (career insights, gap analysis, resources)
                    try:
                        experience_data = await self._extract_experience_context(assessment_results, state)
                        enhanced_report = await self._enrich_report_content(enhanced_report, assessment_results, experience_data)
                        # Reinforce topic specificity before returning
                        enhanced_report = self._reinforce_topic_specificity(enhanced_report, assessment_results)
                    except Exception:
                        pass
                    return enhanced_report
                    
                except Exception as e:
                    log.warning(f"Failed to parse enhanced report JSON: {e}")
                    # Fallback to base report
                    base = self._generate_generic_test_report(assessment_results, candidate_name)
                    try:
                        experience_data = await self._extract_experience_context(assessment_results, state)
                        base = await self._enrich_report_content(base, assessment_results, experience_data)
                        base = self._reinforce_topic_specificity(base, assessment_results)
                    except Exception:
                        pass
                    return base
            else:
                # Fallback to base report
                base = self._generate_generic_test_report(assessment_results, candidate_name)
                try:
                    experience_data = await self._extract_experience_context(assessment_results, state)
                    base = await self._enrich_report_content(base, assessment_results, experience_data)
                    base = self._reinforce_topic_specificity(base, assessment_results)
                except Exception:
                    pass
                return base
                
        except Exception as e:
            log.error(f"Enhanced report generation failed: {e}")
            # Fallback to base report
            base = self._generate_generic_test_report(assessment_results, candidate_name)
            try:
                experience_data = await self._extract_experience_context(assessment_results, state)
                base = await self._enrich_report_content(base, assessment_results, experience_data)
                base = self._reinforce_topic_specificity(base, assessment_results)
            except Exception:
                pass
            return base

    # ============================================================================
    # SCANNABLE INTELLIGENCE REPORT (DOMAIN-AGNOSTIC)
    # ============================================================================

    def _build_scannable_intelligence_report(self, base_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a compact, action-oriented assessment report.

        - Domain agnostic (no cloud-specific wording)
        - Optimized for skimming and UI rendering
        - Does not rename existing keys; adds new structured view
        """

        def _to_float(value: Any, default: float = 0.0) -> float:
            try:
                if value is None:
                    return default
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    cleaned = value.strip().replace("%", "")
                    return float(cleaned)
                return default
            except Exception:
                return default

        def _short(text: str, max_len: int = 180) -> str:
            if not isinstance(text, str):
                return ""
            text = text.strip()
            if len(text) <= max_len:
                return text
            return text[: max_len - 3].rstrip() + "..."

        assessment_results = base_report.get("assessment_results")
        metadata_sources = [base_report]
        if isinstance(assessment_results, dict):
            metadata_sources.append(assessment_results)

        def _get_metadata_value(*keys: str) -> Any:
            for source in metadata_sources:
                if not isinstance(source, dict):
                    continue
                for key in keys:
                    value = source.get(key)
                    if value not in (None, ""):
                        return value
            return None

        def _derive_performance_label(raw_label: Any, score_percentage: float) -> str:
            if score_percentage >= 75:
                return "Strong"
            if score_percentage >= 45:
                return "Needs Work"
            return "Weak"

        def _derive_competency_label(topic_score: Optional[float], has_strengths: bool, has_gaps: bool) -> str:
            if topic_score is not None:
                if topic_score >= 0.75:
                    return "Strong"
                if topic_score >= 0.45:
                    return "Needs Work"
                return "Weak"
            if refinement_mode and has_gaps:
                return "Needs Work"
            if has_strengths and not has_gaps:
                return "Strong"
            if has_gaps and not has_strengths:
                return "Weak"
            return "Needs Work"

        def _format_question_count(count: Optional[int], singular: str, plural: str) -> str:
            if count is None:
                return plural
            return f"{count} {singular if count == 1 else plural}"

        # ---- Core numeric signals ------------------------------------------------
        total_score = _to_float(base_report.get("total_score"), 0.0)
        max_score = _to_float(base_report.get("max_score"), 0.0)
        percentage = (total_score / max_score * 100.0) if max_score > 0 else 0.0

        weighted_score_percentage = _to_float(_get_metadata_value("weighted_score_percentage"), None)
        display_percentage = weighted_score_percentage if weighted_score_percentage is not None else percentage
        performance_level = _derive_performance_label(_get_metadata_value("performance_level"), display_percentage)

        attempted_questions = _to_optional_int(
            _get_metadata_value("attempted_questions", "answered_questions")
        )
        fully_correct_questions = _to_optional_int(_get_metadata_value("fully_correct_questions"))
        partially_correct_questions = _to_optional_int(_get_metadata_value("partially_correct_questions"))
        incorrect_questions = _to_optional_int(
            _get_metadata_value("incorrect_questions", "incorrect_answers")
        )
        score_explanation = _normalize_sentence(_get_metadata_value("score_explanation"))

        misconceptions = (
            base_report.get("misconceptions")
            or (assessment_results.get("misconceptions") if isinstance(assessment_results, dict) else [])
            or []
        )

        if attempted_questions is None:
            derived_attempted = sum(
                count or 0
                for count in (
                    fully_correct_questions,
                    partially_correct_questions,
                    incorrect_questions,
                )
            )
            if derived_attempted > 0:
                attempted_questions = derived_attempted
        if attempted_questions is None:
            attempted_questions = _to_optional_int(_get_metadata_value("total_questions"))

        if incorrect_questions is None and attempted_questions is not None:
            known_correctness_counts = sum(
                count or 0
                for count in (
                    fully_correct_questions,
                    partially_correct_questions,
                )
            )
            if known_correctness_counts <= attempted_questions:
                incorrect_questions = max(attempted_questions - known_correctness_counts, 0)

        has_partial_questions = (partially_correct_questions or 0) > 0
        has_incorrect_questions = (incorrect_questions or 0) > 0
        perfect_score_mode = (
            attempted_questions is not None
            and attempted_questions > 0
            and fully_correct_questions == attempted_questions
            and not has_partial_questions
            and not has_incorrect_questions
        )
        refinement_mode = has_partial_questions and not has_incorrect_questions
        remediation_mode = has_incorrect_questions or (
            incorrect_questions is None and not refinement_mode and isinstance(misconceptions, list) and bool(misconceptions)
        )

        def _should_include_score_explanation(text: str) -> bool:
            if not text:
                return False
            if refinement_mode and re.search(
                r"\b(error|errors|incorrect|miss|misses|missed|mistake|mistakes|wrong|failure|failures|failed|failing)\b",
                text,
                re.IGNORECASE,
            ):
                return False
            return True

        # ---- Topic normalization -------------------------------------------------
        raw_topics = (
            base_report.get("topics")
            or (assessment_results.get("topics") if isinstance(assessment_results, dict) else [])
            or []
        )
        topics: List[Dict[str, Any]] = []
        for t in raw_topics:
            if not isinstance(t, dict):
                continue
            name = str(t.get("topic") or "").strip() or "Overall performance"
            strengths = [s for s in (t.get("strengths") or []) if isinstance(s, str)]
            gaps = [] if perfect_score_mode else [g for g in (t.get("areas_for_improvement") or []) if isinstance(g, str)]
            topic_score = _to_float(t.get("score"), None) if t.get("score") is not None else None
            topics.append(
                {
                    "name": name,
                    "strengths": strengths,
                    "gaps": gaps,
                    "score": topic_score,
                }
            )

        # Sort topics by explicit score if available, otherwise by gap count
        def _topic_sort_key(t: Dict[str, Any]) -> float:
            if t.get("score") is not None:
                return t["score"]
            # More gaps → lower score
            return -float(len(t.get("gaps") or []))

        sorted_topics = sorted(topics, key=_topic_sort_key, reverse=True) if topics else []
        strongest_topic = sorted_topics[0] if sorted_topics else None
        weakest_topic = None if perfect_score_mode else (sorted_topics[-1] if sorted_topics else None)

        # ---- Summary -------------------------------------------------------------
        # Normalized overall score 0–100
        normalized_score = max(0.0, min(100.0, display_percentage if max_score > 0 else total_score))

        strengths_bullet: List[str] = []
        if attempted_questions and fully_correct_questions is not None:
            strengths_bullet.append(
                _short(f"{fully_correct_questions} of {attempted_questions} questions were fully correct.")
            )
        elif strongest_topic and strongest_topic.get("strengths"):
            strengths_bullet.append(
                _short(
                    f"Performance on {strongest_topic['name']} tasks is more reliable than on other topics."
                )
            )

        gaps_bullets: List[str] = []
        if perfect_score_mode:
            gaps_bullets = []
        elif refinement_mode:
            if weakest_topic:
                gaps_bullets.append(
                    _short(
                        f"{weakest_topic['name']} is the main refinement area, where answers are mostly correct but need tighter precision."
                    )
                )
            elif partially_correct_questions is not None:
                gaps_bullets.append(
                    _short(
                        f"{_format_question_count(partially_correct_questions, 'response was', 'responses were')} partially correct and mainly need tighter precision."
                    )
                )
        elif weakest_topic:
            gaps_bullets.append(
                _short(
                    (
                        f"{weakest_topic['name']} questions account for "
                        f"{_format_question_count(incorrect_questions, 'incorrect response', 'incorrect responses')}"
                        if has_incorrect_questions and incorrect_questions is not None
                        else f"{weakest_topic['name']} is the main area to strengthen next."
                    )
                )
            )
        elif has_incorrect_questions and incorrect_questions is not None:
            gaps_bullets.append(
                _short(
                    f"{_format_question_count(incorrect_questions, 'response was', 'responses were')} incorrect across the assessment."
                )
            )

        if remediation_mode and isinstance(misconceptions, list) and misconceptions:
            first_m = misconceptions[0]
            if isinstance(first_m, dict):
                desc = first_m.get("user_mistake") or first_m.get("description") or first_m.get("summary")
            else:
                desc = str(first_m)
            if desc:
                gaps_bullets.append(_short(desc))

        gaps_bullets = gaps_bullets[:2]

        if perfect_score_mode:
            takeaway = _short(
                f"Overall performance is {performance_level.lower()} with fully correct responses across the assessment."
            )
        elif refinement_mode:
            takeaway = _short(
                f"Overall performance is {performance_level.lower()} and the remaining opportunity is to turn partial-credit answers into fully correct ones."
            )
        elif has_incorrect_questions:
            takeaway = _short(
                f"Overall performance is {performance_level.lower()} with the main risk concentrated in the weakest topic area."
            )
        else:
            takeaway = _short(
                f"Overall performance is {performance_level.lower()} with the main opportunity concentrated in the weakest topic area."
            )

        if _should_include_score_explanation(score_explanation):
            takeaway = _short(f"{takeaway} {score_explanation}")

        summary = {
            "overall_score": round(normalized_score, 1),
            "performance_label": performance_level,
            "strengths": strengths_bullet[:1],
            "gaps": gaps_bullets,
            "takeaway": takeaway,
        }

        # ---- Key insights (exactly 3) -------------------------------------------
        insights: List[Dict[str, Any]] = []

        # Insight 1: Pattern of errors in weakest topic
        if weakest_topic:
            if refinement_mode:
                insights.append(
                    {
                        "title": f"Refinement area in {weakest_topic['name']}",
                        "bullets": [
                            _short("Most answers in this area are close to correct but lose points on precision or completeness."),
                            _short("Small improvements in wording, justification, or final checks can raise these responses to fully correct."),
                        ],
                    }
                )
            else:
                insights.append(
                    {
                        "title": f"Error clustering in {weakest_topic['name']}",
                        "bullets": [
                            _short("Mistakes are concentrated in this topic instead of being evenly distributed."),
                            _short("Question variants in this area tend to fail in similar ways."),
                        ],
                    }
                )

        # Insight 2: Misconceptions, if any
        if perfect_score_mode:
            mastery_bullets: List[str] = []
            if attempted_questions:
                mastery_bullets.append(
                    _short(f"All {attempted_questions} attempted questions were fully correct.")
                )
            if score_explanation:
                mastery_bullets.append(_short(score_explanation))
            insights.append(
                {
                    "title": "Consistent accuracy across the assessment",
                    "bullets": mastery_bullets[:2] or [
                        _short("Responses stayed correct and complete across the full assessment.")
                    ],
                }
            )
        elif refinement_mode:
            refinement_bullets: List[str] = []
            if partially_correct_questions is not None:
                if attempted_questions:
                    refinement_bullets.append(
                        _short(
                            f"{partially_correct_questions} of {attempted_questions} questions earned partial credit rather than being fully incorrect."
                        )
                    )
                else:
                    refinement_bullets.append(
                        _short(
                            f"{_format_question_count(partially_correct_questions, 'question earned', 'questions earned')} partial credit rather than being fully incorrect."
                        )
                    )
            if _should_include_score_explanation(score_explanation):
                refinement_bullets.append(_short(score_explanation))
            if refinement_bullets:
                insights.append(
                    {
                        "title": "Precision is the scoring lever",
                        "bullets": refinement_bullets[:2],
                    }
                )
        elif misconceptions:
            insights.append(
                {
                    "title": "Misconceptions driving repeated errors",
                    "bullets": [
                        _short("Multiple questions reuse the same incorrect assumption rather than different mistakes."),
                        _short("The error pattern stays the same even when the question surface changes."),
                    ],
                }
            )

        # Insight 3: Balance between strengths and gaps
        if strongest_topic and weakest_topic and strongest_topic is not weakest_topic:
            insights.append(
                {
                    "title": "Asymmetric skill profile",
                    "bullets": [
                        _short(f"{strongest_topic['name']} scores sit well above {weakest_topic['name']}."),
                        _short("Cross-topic tasks will be limited by the weakest area rather than the strongest."),
                    ],
                }
            )

        # Ensure exactly 3 insights with concrete, non-duplicated content
        if perfect_score_mode:
            fallback_insights = [
                {
                    "title": "Broad topic coverage",
                    "bullets": [
                        "Performance remained accurate across the different question formats in this assessment.",
                        "No single topic stood out as a drag on the overall result.",
                    ],
                },
                {
                    "title": "Ready for harder variants",
                    "bullets": [
                        "The current result supports moving from fundamentals to more complex problem variants.",
                        "The next gains are more likely to come from difficulty progression than from remediation.",
                    ],
                },
                {
                    "title": "Stable reasoning pattern",
                    "bullets": [
                        "The same reasoning approach held up across multiple prompts rather than only one question style.",
                        "That consistency is a strong signal of reliable understanding.",
                    ],
                },
            ]
        elif refinement_mode:
            fallback_insights = [
                {
                    "title": "Close-to-correct reasoning",
                    "bullets": [
                        "The remaining score gap comes from answers that are nearly complete but not fully precise.",
                        "Tightening the final explanation or verification step should convert more responses to fully correct.",
                    ],
                },
                {
                    "title": "Final-step consistency matters",
                    "bullets": [
                        "A short self-check before submitting can catch missing qualifiers, assumptions, or edge conditions.",
                        "This is a precision problem more than a conceptual understanding problem.",
                    ],
                },
                {
                    "title": "Strong baseline with a narrow gap",
                    "bullets": [
                        "The overall pattern shows solid understanding across the assessment.",
                        "Improvement is most likely to come from sharpening already-correct reasoning rather than relearning fundamentals.",
                    ],
                },
            ]
        else:
            fallback_insights = [
                {
                    "title": "Unclear reasoning steps",
                    "bullets": [
                        "Several answers jump to results without using intermediate checks.",
                        "This makes small mistakes harder to spot before submitting.",
                    ],
                },
                {
                    "title": "Inconsistent handling of edge cases",
                    "bullets": [
                        "Boundary or exception-style questions were answered less accurately than standard ones.",
                        "This matters whenever real-world inputs do not match clean examples.",
                    ],
                },
                {
                    "title": "Gaps between recognition and recall",
                    "bullets": [
                        "Multiple-choice recognition outperformed short-answer recall where you had to generate the idea yourself.",
                        "This gap limits performance when no options are provided as hints.",
                    ],
                },
            ]

        while len(insights) < 3:
            candidate = fallback_insights[len(insights)]
            insights.append(candidate)
        insights = insights[:3]

        # ---- Competency snapshot -------------------------------------------------
        competency_snapshot: List[Dict[str, Any]] = []
        for t in sorted_topics[:5]:
            has_gaps = bool(t.get("gaps"))
            has_strengths = bool(t.get("strengths"))
            label = _derive_competency_label(t.get("score"), has_strengths, has_gaps)
            if perfect_score_mode:
                diagnosis = f"Responses in {t['name']} are consistently aligned with expected outcomes."
            elif refinement_mode:
                if has_gaps and not has_strengths:
                    diagnosis = f"{t['name']} is a refinement area where a few answers need tighter precision or fuller explanation."
                elif has_strengths and has_gaps:
                    diagnosis = f"{t['name']} is mostly strong, with a small precision gap remaining in a few responses."
                else:
                    diagnosis = f"Responses in {t['name']} are consistently aligned with expected outcomes."
            else:
                if has_gaps and not has_strengths:
                    diagnosis = f"Most answers in {t['name']} deviate from the expected pattern in similar ways."
                elif has_strengths and has_gaps:
                    diagnosis = f"{t['name']} shows a mix of clean solutions and recurring slip-ups."
                else:
                    diagnosis = f"Responses in {t['name']} are consistently aligned with expected outcomes."

            competency_snapshot.append(
                {
                    "name": t["name"],
                    "proficiency_label": label,
                    "one_line_diagnosis": _short(diagnosis, 200),
                }
            )

        if not competency_snapshot:
            competency_snapshot.append(
                {
                    "name": "Overall",
                    "proficiency_label": performance_level if performance_level != "Unknown" else "Needs Work",
                    "one_line_diagnosis": _short("Responses show mixed accuracy without a single dominant pattern."),
                }
            )

        # ---- Growth blockers -----------------------------------------------------
        growth_blockers: List[Dict[str, Any]] = []

        if remediation_mode:
            for m in misconceptions[:3]:
                if isinstance(m, dict):
                    concept = m.get("concept_name") or m.get("concept") or m.get("topic") or "Key concept"
                    mistake = m.get("user_mistake") or m.get("description") or m.get("summary") or ""
                else:
                    concept = "Key concept"
                    mistake = str(m)

                if not mistake:
                    continue

                growth_blockers.append(
                    {
                        "concept_name": str(concept),
                        "user_mistake": _short(str(mistake)),
                        "why_it_matters": "The same assumption is producing similar failures across different questions.",
                    }
                )

            if not growth_blockers and weakest_topic:
                growth_blockers.append(
                    {
                        "concept_name": weakest_topic["name"],
                        "user_mistake": "You mix up how to apply this concept when the question format changes.",
                        "why_it_matters": "Tasks that touch this area become inconsistent in outcome.",
                    }
                )

        # ---- Action plan (4–6 verb-first items) ---------------------------------
        checklist: List[str] = []
        weak_name = weakest_topic["name"] if weakest_topic else "your lowest-scoring topic"
        focus_name = strongest_topic["name"] if strongest_topic else "this assessment"

        if perfect_score_mode:
            checklist.append("Document three habits that helped you stay accurate across this assessment.")
            checklist.append(f"Apply the same habits to five advanced {focus_name} questions.")
        else:
            checklist.append(f"List three checks you will always apply before answering {weak_name} questions.")
        if refinement_mode:
            checklist.append(
                f"Rework five {weak_name} responses that were close to correct and note what would make them fully correct."
            )
        elif remediation_mode:
            checklist.append(f"Re-attempt five missed {weak_name} items and write the steps you actually used.")
        elif not perfect_score_mode:
            checklist.append(
                f"Review five {weak_name} questions and note the reasoning step that would make each answer more complete."
            )
        checklist.append("Tag each new question you solve by the pattern of reasoning you applied.")
        if perfect_score_mode:
            checklist.append("Track which checks still hold up when question difficulty or schema complexity increases.")
        elif refinement_mode:
            checklist.append("Log each partially correct response in one line with the missing detail or verification step.")
        elif remediation_mode:
            checklist.append("Log each mistake in one line with the trigger and the wrong decision.")
        else:
            checklist.append("Log each response that felt uncertain and note what additional check would have increased confidence.")

        if strongest_topic and weakest_topic and strongest_topic is not weakest_topic:
            checklist.append(f"Reuse one habit from {strongest_topic['name']} while solving a {weak_name} question.")

        checklist.append("Schedule a short review after two practice blocks to adjust this checklist.")

        # Ensure 4–6 items
        checklist = checklist[:6]
        if len(checklist) < 4:
            checklist.append("Create an error log and capture each new failure with cause and pattern.")

        # ---- Learning resources (optional, tied to blockers) --------------------
        learning_resources: List[Dict[str, Any]] = []
        try:
            resources = (
                base_report.get("resources")
                or (assessment_results.get("resources") if isinstance(assessment_results, dict) else [])
                or []
            )
            if isinstance(resources, list) and growth_blockers:
                for blocker in growth_blockers:
                    concept_name = str(blocker.get("concept_name", "")).lower()
                    matched_link = None
                    for res in resources:
                        if not isinstance(res, dict):
                            continue
                        subtopic = str(res.get("subtopic") or "").lower()
                        links = [l for l in (res.get("links") or []) if isinstance(l, str)]
                        if not links:
                            continue
                        if concept_name and concept_name in subtopic:
                            matched_link = links[0]
                            break
                    if matched_link:
                        learning_resources.append(
                            {
                                "concept_name": blocker.get("concept_name"),
                                "link": matched_link,
                            }
                        )
        except Exception:
            learning_resources = []

        result_payload: Dict[str, Any] = {
            "summary": summary,
            "key_insights": insights,
            "competency_snapshot": competency_snapshot,
            "growth_blockers": growth_blockers,
            "action_plan": checklist,
        }

        if learning_resources:
            result_payload["learning_resources"] = learning_resources

        return result_payload

    @traceable(name="report_generator_process")
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing pipeline following production standards."""
        start_time = time.time()
        tenant_id = state.get("security_context", {}).get("tenant_id") or state.get("tenant_id", "default_tenant_id")

        # Granular timing breakdown for performance monitoring
        timing_breakdown = {
            "cache_check": 0.0,
            "report_generation": 0.0,
            "cache_store": 0.0,
            "total": 0.0
        }

        if not validate_tenant_id(tenant_id):
            return {"ok": False, "analysis_method": "error", "error": "Invalid tenant ID format"}

        # Rate limiting now handled by centralized middleware

        raw_assessment_results = state.get("assessment_results") or state.get("output", {}).get("assessment_results")
        if not raw_assessment_results:
            return {"ok": False, "analysis_method": "error", "error": "Assessment results missing"}

        # Defensive unwrap: if state has full evaluator payload (nested "assessment_results"),
        # use the inner dict so total_score/max_score are present (avoids report showing 0%).
        assessment_results = raw_assessment_results
        if isinstance(raw_assessment_results, dict):
            inner = raw_assessment_results.get("assessment_results")
            if isinstance(inner, dict) and ("total_score" in inner or "max_score" in inner):
                if "total_score" not in raw_assessment_results and "max_score" not in raw_assessment_results:
                    assessment_results = inner
                    log.debug("Report generator: using nested assessment_results for scores")

        if not validate_size(assessment_results, 100 * 1024):  # MAX_INPUT_SIZE
            return {"ok": False, "analysis_method": "error", "error": "Input size exceeded"}

        candidate_name = state.get("candidate_name", "Candidate")

        cache_start = time.time()
        cached = await self.cache.get(tenant_id, state)
        timing_breakdown["cache_check"] = time.time() - cache_start
        if cached:
            self.metrics.cache_hits += 1
            processing_time = time.time() - start_time
            await self.metrics.record_response(processing_time, 1.0, "cached")
            try:
                if isinstance(cached, dict) and "intelligence_report" not in cached:
                    cached["intelligence_report"] = await asyncio.to_thread(self._build_scannable_intelligence_report, cached)
            except Exception:
                # Never fail the request due to reporting transformation
                pass
            return self._wrap_response(cached, "cached", processing_time)

        self.metrics.cache_misses += 1

        # Circuit breaker check now handled by centralized middleware

        # Check if this is a generic test assessment
        test_type = assessment_results.get("test_type")
        evaluation_method = assessment_results.get("evaluation_method")
        constructive_feedback = assessment_results.get("constructive_feedback")
        
        # Generic test indicators: test_type exists AND (evaluation_method indicates generic OR constructive_feedback exists)
        is_generic_test = (
            test_type and (
                evaluation_method in ["behavioral_scoring", "llm_behavioral_analysis", "deterministic_fallback"] or
                constructive_feedback is not None
            )
        )
        
        gen_start = time.time()
        if is_generic_test:
            # Handle generic test assessment with LLM-enhanced reporting
            method = "generic_test_llm"
            result = await self._generate_enhanced_generic_test_report(assessment_results, candidate_name, tenant_id, state)
        else:
            # Gating Logic for traditional assessments
            total_score = assessment_results.get("total_score", 0)
            max_score = assessment_results.get("max_score", 100)
            percentage = (total_score / max_score * 100) if max_score > 0 else 0

            # Always try LLM first for better quality reports
            method = "llm"
            log.info(f"🎯 Report Generator: Score {total_score}/{max_score} ({percentage:.1f}%) - Using enhanced LLM method")
            result = await self.generate_report_with_llm(assessment_results, candidate_name, tenant_id)
            
            # Only fallback to deterministic if LLM completely fails
            if not result.get("topics") or not result.get("summary"):
                log.warning("⚠️ LLM report generation failed, using deterministic fallback")
                method = "deterministic_fallback"
                result = self._deterministic_report(assessment_results)

        # Ensure report score/level match assessment_results (single source of truth for display)
        if isinstance(result, dict) and assessment_results is not None:
            result = self._normalize_report_scores_from_assessment(result, assessment_results, is_generic_test)

        timing_breakdown["report_generation"] = time.time() - gen_start

        # Attach domain-agnostic intelligence report view
        try:
            if isinstance(result, dict):
                result["intelligence_report"] = await asyncio.to_thread(self._build_scannable_intelligence_report, result)
        except Exception:
            log.warning("Failed to build scannable intelligence report", exc_info=True)

        cache_store_start = time.time()
        if method in ["llm", "deterministic", "generic_test", "generic_test_llm"]:
            await self.cache.set(tenant_id, state, result)
        timing_breakdown["cache_store"] = time.time() - cache_store_start

        processing_time = time.time() - start_time
        timing_breakdown["total"] = processing_time
        confidence = 0.9 if method in ["llm", "generic_test", "generic_test_llm"] else 0.7
        await self.metrics.record_response(processing_time, confidence, method)
        
        # Log timing breakdown
        log.info(
            f"⏱️ REPORT_GENERATOR timing breakdown: "
            f"cache_check={timing_breakdown['cache_check']*1000:.1f}ms, "
            f"generation={timing_breakdown['report_generation']*1000:.1f}ms, "
            f"cache_store={timing_breakdown['cache_store']*1000:.1f}ms, "
            f"total={timing_breakdown['total']*1000:.1f}ms"
        )

        # Session Management Integration - MOVED TO BACKGROUND (non-blocking)
        uid = state.get("uid")
        if uid:
            # Fire-and-forget: Don't block response on session updates
            async def _update_session_background():
                try:
                    session_start = time.time()
                    log.info(f"🔄 REPORT_GENERATOR: Starting background session storage for UID={uid}")
                    
                    # Get or reuse existing session (ensures UID always uses same session ID)
                    existing_session = await run_blocking_io(
                        session_manager.get_or_reuse_session,
                        owner_id=uid,
                        kind="candidate_pipeline",
                        owner_type="candidate",
                        initial_step="report_generator",
                        initial_data={"report": result}
                    )
                    
                    # Update session step
                    await run_blocking_io(
                        session_manager.update_step,
                        session_id=existing_session.session_id,
                        step="report_generator",
                        data={"method": method, "confidence": confidence},
                        progress=0.5  # 50% complete after report generation
                    )
                    
                    # Store report generator data in existing session
                    session_data = await run_blocking_io(get_chat_session, existing_session.session_id)
                    if session_data:
                        session_data["report_generator"] = {
                            "report": result,
                            "method": method,
                            "confidence": confidence,
                            "processing_time": processing_time,
                            "timing_breakdown": timing_breakdown,
                            "candidate_name": candidate_name
                        }
                        
                        await run_blocking_io(
                            update_chat_session,
                            session_id=existing_session.session_id,
                            session_data=session_data,
                            metadata={
                                "agent": "report_generator",
                                "uid": uid,
                                "status": "report_generator_complete",
                                "method": method
                            }
                        )
                        session_time = time.time() - session_start
                        log.info(f"✅ REPORT_GENERATOR: Background session update completed in {session_time*1000:.1f}ms for UID={uid}")
                    else:
                        log.warning(f"⚠️ REPORT_GENERATOR: No existing chat session data found for session_id={existing_session.session_id}")
                except Exception as e:
                    log.error(f"❌ REPORT_GENERATOR: Background session update failed for UID={uid}: {e}")
                    import traceback
                    log.debug(f"Report generator session management traceback: {traceback.format_exc()}")
            
            # Fire-and-forget: Start background task without awaiting
            asyncio.create_task(_update_session_background())
            log.info(f"🚀 REPORT_GENERATOR: Started background session update task for UID={uid} (non-blocking)")

            
        # Merge enriched career insights into gap analyzer document for future advisors
        try:
            from chroma import get_gap_doc, upsert_gap_doc
            enriched_fields = {}
            if isinstance(result, dict):
                if result.get("career_insights"):
                    enriched_fields["career_insights"] = result.get("career_insights")
                if result.get("skill_gap_analysis"):
                    enriched_fields["skill_gap_analysis"] = result.get("skill_gap_analysis")
                if result.get("industry_recommendations"):
                    enriched_fields["industry_recommendations"] = result.get("industry_recommendations")
                if result.get("learning_resources"):
                    enriched_fields["learning_resources"] = result.get("learning_resources")

            if enriched_fields:
                existing_gap = await run_blocking_io(get_gap_doc, uid) or {}
                merged_gap = {**existing_gap}
                advisor = merged_gap.get("skill_and_career_advisor") or {}
                raw_gap = advisor.get("raw_skill_gap_analysis_output") or {}
                raw_gap.update(enriched_fields)
                advisor.update({
                    "raw_skill_gap_analysis_output": raw_gap,
                    "last_updated_by": "report_generator",
                    "evaluation_flow": True
                })
                merged_gap["skill_and_career_advisor"] = advisor
                await run_blocking_io(upsert_gap_doc, uid, merged_gap, metadata={
                    "agent": "report_generator",
                    "uid": uid,
                    "status": "report_generator_enriched_gap",
                    "method": method,
                    "evaluation_flow": True
                })
                log.info(f"✅ REPORT_GENERATOR: Enriched gap analyzer doc for UID={uid}")
        except Exception as e:
            log.warning(f"⚠️ REPORT_GENERATOR: Failed to enrich gap analyzer doc for UID={uid}: {e}")

        return self._wrap_response(result, method, processing_time)

    def _wrap_response(self, result: Dict[str, Any], method: str, processing_time: float) -> Dict[str, Any]:
        """Wrap result in standard envelope.

        Note:
            Downstream consumers expect the `report` field to expose only the
            compact, scannable assessment view (summary, key_insights,
            competency_snapshot, growth_blockers, action_plan, ...).

            The richer raw LLM/deterministic payload is still preserved internally
            on `result` (and cached / stored in sessions), but the public
            response surface now standardizes on the compact format.
        """
        def _empty_compact_report() -> Dict[str, Any]:
            return {
                "summary": {
                    "overall_score": 0.0,
                    "performance_label": "Needs Work",
                    "strengths": [],
                    "gaps": [],
                    "takeaway": "Compact report is temporarily unavailable.",
                },
                "key_insights": [],
                "competency_snapshot": [],
                "growth_blockers": [],
                "action_plan": [],
            }

        # Always expose the compact intelligence view as the public `report`.
        # If it was already precomputed, reuse it; otherwise build it on-demand.
        try:
            compact_report = result.get("intelligence_report")
        except AttributeError:
            compact_report = None

        if not isinstance(compact_report, dict):
            try:
                compact_report = self._build_scannable_intelligence_report(result or {})
            except Exception:
                # Keep the public report shape stable even if compact building fails.
                compact_report = _empty_compact_report()

        return {
            "ok": True,
            "analysis_method": method,
            "method_explain": {
                "processing_time": round(processing_time, 3),
                "cache_hit": method == "cached",
                "circuit_open": method == "deterministic_fallback"
            },
            "report": compact_report
        }


# ============================================================================
# 9. MAIN ENTRY POINT
# ============================================================================

_agent = None

@traceable(name="report_generator_agent")
async def report_generator_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry point for report generation."""
    global _agent
    if _agent is None:
        _agent = ReportGeneratorAgent()

    return await _agent.process(state)


