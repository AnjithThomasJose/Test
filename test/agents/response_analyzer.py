"""
Hybrid Response Analyzer for Interview Agent
Combines fast rule-based analysis with LLM fallback for complex cases
"""

import re
import json
import logging
import unicodedata
from typing import Dict, Any, List, Tuple
from datetime import datetime

from models.llm_invoker import invoke_llm

log = logging.getLogger(__name__)

def _sanitize_answer_for_prompt(answer: str) -> str:
    """
    Sanitizes user answer for safe embedding in LLM prompts and JSON parsing.
    Same implementation as in interview_agent.py for consistency.
    
    Args:
        answer: Raw user answer string
        
    Returns:
        Sanitized string safe for embedding in prompts and JSON
    """
    if not isinstance(answer, str):
        return ""
    
    # Step 1: Normalize unicode characters
    normalized = unicodedata.normalize('NFKC', answer)
    
    # Step 2: Escape special characters that break JSON/string embedding
    # Replace newlines and carriage returns with spaces
    sanitized = normalized.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
    
    # Replace tabs with spaces
    sanitized = sanitized.replace('\t', ' ')
    
    # Escape backslashes first (before other escapes)
    sanitized = sanitized.replace('\\', '\\\\')
    
    # Escape double quotes (critical for JSON)
    sanitized = sanitized.replace('"', '\\"')
    
    # Step 3: Remove or replace problematic control characters
    # Keep printable characters and common whitespace
    sanitized = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', sanitized)
    
    # Step 4: Normalize whitespace (multiple spaces to single space)
    sanitized = re.sub(r'\s+', ' ', sanitized)
    sanitized = sanitized.strip()
    
    # Step 5: Limit length to prevent prompt injection and excessive tokens
    max_length = 2000  # Reasonable limit for interview responses
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "... [truncated]"
    
    return sanitized

class HybridResponseAnalyzer:
    """Fast rule-based analysis with LLM fallback for complex responses"""
    
    def __init__(self):
        # Technical keywords by domain
        self.technical_keywords = {
            'ui_ux': ['figma', 'sketch', 'adobe', 'prototype', 'wireframe', 'mockup', 'design system', 'user research', 'usability', 'accessibility', 'aria', 'responsive', 'mobile-first'],
            'frontend': ['react', 'angular', 'vue', 'javascript', 'typescript', 'html', 'css', 'sass', 'scss', 'webpack', 'babel', 'npm', 'yarn', 'component', 'state', 'props', 'hooks'],
            'backend': ['node', 'python', 'java', 'spring', 'django', 'flask', 'api', 'rest', 'graphql', 'database', 'sql', 'mongodb', 'redis', 'microservices'],
            'data': ['python', 'r', 'sql', 'pandas', 'numpy', 'machine learning', 'ai', 'analytics', 'visualization', 'statistics', 'model', 'algorithm'],
            'devops': ['docker', 'kubernetes', 'aws', 'azure', 'gcp', 'ci/cd', 'jenkins', 'gitlab', 'terraform', 'monitoring', 'logging', 'deployment'],
            'mobile': ['ios', 'android', 'swift', 'kotlin', 'react native', 'flutter', 'xcode', 'android studio', 'app store', 'play store']
        }
        
        # Action verbs indicating experience
        self.action_verbs = ['developed', 'built', 'created', 'implemented', 'designed', 'optimized', 'improved', 'managed', 'led', 'architected', 'integrated', 'deployed', 'tested', 'debugged', 'refactored', 'scaled', 'migrated', 'configured', 'maintained']
        
        # Positive sentiment words
        self.positive_words = ['successful', 'effective', 'improved', 'enhanced', 'optimized', 'achieved', 'accomplished', 'exceeded', 'delivered', 'solved', 'resolved', 'innovative', 'creative', 'efficient', 'robust', 'scalable', 'reliable']
        
        # Negative sentiment words
        self.negative_words = ['failed', 'broken', 'issue', 'problem', 'challenge', 'difficult', 'struggled', 'limitation', 'constraint', 'error', 'bug', 'slow', 'inefficient']
        
        # Behavioral indicators
        self.behavioral_indicators = ['team', 'collaborated', 'communicated', 'mentored', 'led', 'managed', 'coordinated', 'presented', 'explained', 'taught', 'learned', 'adapted', 'flexible', 'deadline', 'pressure', 'conflict', 'feedback']
    
    async def analyze_response(self, 
                        answer: str, 
                        conversation_history: List[Dict],
                        candidate_info: Dict[str, Any],
                        job_details: Dict[str, Any],
                        is_first_request: bool = False,
                        uid_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze response using hybrid approach: rule-based first, LLM if needed
        """
        try:
            # Rule-based analysis (fast path)
            rule_based_result = self._rule_based_analysis(answer, candidate_info, job_details, uid_context)
            
            # Determine if LLM analysis is needed
            confidence = self._calculate_confidence(rule_based_result, answer)
            
            if confidence >= 0.7 and not is_first_request:
                # Use rule-based result
                log.info(f"Using rule-based analysis (confidence: {confidence:.2f})")
                return rule_based_result
            else:
                # Use LLM for complex cases or first request
                log.info(f"Using LLM analysis (confidence: {confidence:.2f}, first_request: {is_first_request})")
                return await self._llm_analysis(answer, conversation_history, candidate_info, job_details, is_first_request, uid_context)
                
        except Exception as e:
            log.error(f"Error in hybrid analysis: {e}")
            # Return safe fallback
            return self._create_fallback_analysis(answer)
    
    def _rule_based_analysis(self, answer: str, candidate_info: Dict[str, Any], 
                           job_details: Dict[str, Any], uid_context: Dict[str, Any]) -> Dict[str, Any]:
        """Fast rule-based analysis using keyword matching and heuristics"""
        
        answer_lower = answer.lower()
        words = answer.split()
        word_count = len(words)
        
        # Extract keywords by domain
        detected_domains = []
        technical_keywords = []
        
        for domain, keywords in self.technical_keywords.items():
            domain_matches = [kw for kw in keywords if kw in answer_lower]
            if domain_matches:
                detected_domains.append(domain)
                technical_keywords.extend(domain_matches)
        
        # Count action verbs (experience indicators)
        action_verb_count = sum(1 for verb in self.action_verbs if verb in answer_lower)
        
        # Sentiment analysis
        positive_count = sum(1 for word in self.positive_words if word in answer_lower)
        negative_count = sum(1 for word in self.negative_words if word in answer_lower)
        
        # Behavioral indicators
        behavioral_count = sum(1 for indicator in self.behavioral_indicators if indicator in answer_lower)
        
        # Determine response type
        response_type = "general"
        if technical_keywords:
            response_type = "technical"
        elif behavioral_count > 2:
            response_type = "behavioral"
        elif any(word in answer_lower for word in ['challenge', 'problem', 'solution', 'approach']):
            response_type = "problem_solving"
        
        # Calculate scores
        confidence = min(0.9, 0.3 + (len(technical_keywords) * 0.1) + (action_verb_count * 0.05))
        
        # Engagement score based on length and specificity
        length_score = min(1.0, word_count / 100)  # Normalize to 100 words = 1.0
        engagement_score = min(1.0, 0.3 + length_score + (len(technical_keywords) * 0.1))
        
        # Sentiment score
        sentiment_score = (positive_count - negative_count) / max(1, positive_count + negative_count)
        sentiment = "positive" if sentiment_score > 0.2 else "negative" if sentiment_score < -0.2 else "neutral"
        
        # Context understanding
        context_understanding = self._generate_context_summary(detected_domains, technical_keywords, action_verb_count)
        
        # Follow-up suggestions
        follow_up_suggestions = self._generate_follow_up_suggestions(response_type, detected_domains, word_count)
        
        # Strengths and areas to explore
        strengths_mentioned = technical_keywords[:5]  # Top 5 technical terms
        areas_to_explore = self._suggest_exploration_areas(detected_domains, uid_context)
        
        return {
            "type": response_type,
            "confidence": confidence,
            "keywords": technical_keywords[:10],  # Limit to top 10
            "sentiment": sentiment,
            "length_score": length_score,
            "engagement_score": engagement_score,
            "word_count": word_count,
            "context_understanding": context_understanding,
            "follow_up_suggestions": follow_up_suggestions,
            "strengths_mentioned": strengths_mentioned,
            "areas_to_explore": areas_to_explore,
            "detected_domains": detected_domains,
            "action_verb_count": action_verb_count,
            "behavioral_count": behavioral_count
        }
    
    def _calculate_confidence(self, rule_result: Dict[str, Any], answer: str) -> float:
        """Calculate confidence in rule-based analysis"""
        base_confidence = rule_result.get("confidence", 0.5)
        
        # Boost confidence for longer, more detailed answers
        word_count = len(answer.split())
        if word_count > 50:
            base_confidence += 0.1
        if word_count > 100:
            base_confidence += 0.1
            
        # Boost confidence for technical responses
        if rule_result.get("type") == "technical":
            base_confidence += 0.1
            
        # Reduce confidence for very short answers
        if word_count < 20:
            base_confidence -= 0.2
            
        return min(0.95, max(0.1, base_confidence))
    
    async def _llm_analysis(self, answer: str, conversation_history: List[Dict],
                     candidate_info: Dict[str, Any], job_details: Dict[str, Any],
                     is_first_request: bool, uid_context: Dict[str, Any]) -> Dict[str, Any]:
        """LLM-based analysis for complex cases"""
        
        try:
            # Build context for LLM
            skills = candidate_info.get('skills', 'Various technical skills')
            job_title = job_details.get('title', 'Technical role')
            
            # Sanitize answer for safe embedding in prompt
            sanitized_answer = _sanitize_answer_for_prompt(answer)
            
            prompt = f"""Analyze this interview response and return ONLY a valid JSON object.

RESPONSE: "{sanitized_answer}"

CANDIDATE CONTEXT:
- Skills: {skills}
- Job Title: {job_title}
- First Request: {is_first_request}

IMPORTANT: Return ONLY valid JSON. No explanations, no markdown, no extra text.

Required JSON format:
{{
    "type": "technical",
    "confidence": 0.8,
    "keywords": ["react", "figma"],
    "sentiment": "positive",
    "length_score": 0.7,
    "engagement_score": 0.8,
    "word_count": 45,
    "context_understanding": "Candidate describes technical experience",
    "follow_up_suggestions": ["Ask about specific projects"],
    "strengths_mentioned": ["React expertise"],
    "areas_to_explore": ["Team collaboration"]
}}"""

            response = await invoke_llm(prompt=prompt)
            
            # Clean and validate response
            if not response or not response.strip():
                log.warning("LLM returned empty response")
                return self._create_fallback_analysis(answer)
            
            # Try to extract JSON from response (in case LLM adds extra text)
            response_text = response.strip()
            
            # Look for JSON object in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text
            
            try:
                result = json.loads(json_text)
            except json.JSONDecodeError as e:
                log.warning(f"LLM returned invalid JSON: {e}, response: {response_text[:200]}...")
                return self._create_fallback_analysis(answer)
            
            # Validate required fields
            required_fields = ["type", "confidence", "keywords", "sentiment", "length_score", 
                             "engagement_score", "word_count", "context_understanding", 
                             "follow_up_suggestions", "strengths_mentioned", "areas_to_explore"]
            
            for field in required_fields:
                if field not in result:
                    result[field] = self._get_default_value(field)
            
            return result
            
        except Exception as e:
            log.error(f"LLM analysis failed: {e}")
            return self._create_fallback_analysis(answer)
    
    def _generate_context_summary(self, domains: List[str], keywords: List[str], action_verbs: int) -> str:
        """Generate context understanding summary"""
        if not domains:
            return "General response received"
        
        domain_str = ", ".join(domains[:3])  # Top 3 domains
        keyword_str = ", ".join(keywords[:3])  # Top 3 keywords
        
        if action_verbs > 2:
            return f"Technical response covering {domain_str} with specific experience in {keyword_str}"
        else:
            return f"Response mentions {domain_str} topics including {keyword_str}"
    
    def _generate_follow_up_suggestions(self, response_type: str, domains: List[str], word_count: int) -> List[str]:
        """Generate contextual follow-up suggestions"""
        suggestions = []
        
        if word_count < 30:
            suggestions.append("Ask for more specific details and examples")
        
        if response_type == "technical":
            suggestions.append("Dive deeper into technical implementation")
            suggestions.append("Ask about challenges and solutions")
        
        if "ui_ux" in domains:
            suggestions.append("Explore design process and user research")
        
        if "frontend" in domains:
            suggestions.append("Ask about performance optimization")
        
        if not suggestions:
            suggestions = ["Ask for more details", "Explore related experience"]
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _suggest_exploration_areas(self, detected_domains: List[str], uid_context: Dict[str, Any]) -> List[str]:
        """Suggest areas to explore based on detected domains and UID context"""
        areas = []
        
        # Add areas based on detected domains
        if "ui_ux" in detected_domains:
            areas.extend(["user research", "accessibility", "design systems"])
        
        if "frontend" in detected_domains:
            areas.extend(["performance", "testing", "collaboration"])
        
        if "backend" in detected_domains:
            areas.extend(["scalability", "security", "architecture"])
        
        # Add areas from UID context if available
        if uid_context and "skills_parser" in uid_context:
            skills_data = uid_context["skills_parser"]
            if isinstance(skills_data, dict) and "skills" in skills_data:
                for skill in skills_data["skills"][:3]:  # Top 3 skills
                    if isinstance(skill, dict) and "SkillName" in skill:
                        areas.append(skill["SkillName"].lower())
        
        return list(set(areas))[:5]  # Remove duplicates, limit to 5
    
    def _get_default_value(self, field: str) -> Any:
        """Get default value for missing fields"""
        defaults = {
            "type": "general",
            "confidence": 0.5,
            "keywords": [],
            "sentiment": "neutral",
            "length_score": 0.5,
            "engagement_score": 0.5,
            "word_count": 0,
            "context_understanding": "Response received, basic analysis",
            "follow_up_suggestions": ["Ask for more details"],
            "strengths_mentioned": [],
            "areas_to_explore": []
        }
        return defaults.get(field, "")
    
    def _create_fallback_analysis(self, answer: str) -> Dict[str, Any]:
        """Create safe fallback analysis when all else fails"""
        word_count = len(answer.split())
        
        return {
            "type": "general",
            "confidence": 0.3,
            "keywords": [],
            "sentiment": "neutral",
            "length_score": min(1.0, word_count / 100),
            "engagement_score": 0.4,
            "word_count": word_count,
            "context_understanding": "Fallback analysis due to processing error",
            "follow_up_suggestions": ["Ask for more details"],
            "strengths_mentioned": [],
            "areas_to_explore": []
        }

# Global instance
hybrid_response_analyzer = HybridResponseAnalyzer()