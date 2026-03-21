"""
Question Drift Detector - Identify off-topic or inappropriate interview questions.

This module provides heuristics to detect when generated questions drift away from
the interview topic or contain inappropriate content (PII, personal questions, etc.).
"""

import re
from typing import List


def is_drift(question: str, topic: str) -> bool:
    """
    Return True if question is likely off-topic or inappropriate.
    
    Detection criteria:
    - Low token-overlap with topic tokens AND
    - Contains personal/PII keywords (age, salary, marriage, children) OR
    - Contains 'tell me about your family' style patterns OR
    - Includes direct contact/transaction patterns (email, phone)
    
    Args:
        question: The generated question string
        topic: The interview topic string (e.g., "backend", "Python", "API design")
        
    Returns:
        True if question appears to be off-topic or inappropriate
    """
    if not question or not isinstance(question, str):
        return True  # Empty or invalid question is considered drift
    
    if not topic or not isinstance(topic, str):
        return False  # No topic means we can't detect drift
    
    question_lower = question.lower()
    topic_lower = topic.lower()
    
    # Extract tokens from topic (simple word-based)
    topic_tokens = set(re.findall(r'\b\w+\b', topic_lower))
    question_tokens = set(re.findall(r'\b\w+\b', question_lower))
    
    # Calculate token overlap
    if topic_tokens:
        overlap = len(topic_tokens & question_tokens) / len(topic_tokens)
        low_overlap = overlap < 0.1  # Less than 10% token overlap
    else:
        low_overlap = True
    
    # Personal/PII keywords that indicate inappropriate questions
    personal_keywords = [
        r'\bage\b',
        r'\bsalary\b',
        r'\bpay\b',
        r'\bwage\b',
        r'\bcompensation\b',
        r'\bmarriage\b',
        r'\bmarried\b',
        r'\bchildren\b',
        r'\bkid\b',
        r'\bkid\b',
        r'\bfamily\b',
        r'\bparent\b',
        r'\breligion\b',
        r'\bpolitical\b',
        r'\bethnicity\b',
        r'\brace\b',
        r'\bgender\b',
        r'\bsexual\s+orientation\b',
        r'\bdisability\b',
        r'\bhealth\s+condition\b',
    ]
    
    # Check for personal keywords
    has_personal = any(re.search(pattern, question_lower) for pattern in personal_keywords)
    
    # Family-related patterns
    family_patterns = [
        r'tell\s+me\s+about\s+your\s+family',
        r'what\s+about\s+your\s+family',
        r'do\s+you\s+have\s+(kids|children)',
        r'are\s+you\s+married',
    ]
    has_family_pattern = any(re.search(pattern, question_lower) for pattern in family_patterns)
    
    # Direct contact/transaction patterns
    contact_patterns = [
        r'\bemail\s+(me|us|them)\b',
        r'\bphone\s+(me|us|them)\b',
        r'\bcall\s+(me|us|them)\b',
        r'\bcontact\s+(me|us|them)\b',
        r'\b@\w+\.(com|org|net)',  # Email-like patterns
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}',  # Phone-like patterns
    ]
    has_contact_pattern = any(re.search(pattern, question_lower) for pattern in contact_patterns)
    
    # Drift if: low overlap AND (personal OR family OR contact)
    if low_overlap and (has_personal or has_family_pattern or has_contact_pattern):
        return True
    
    # Also drift if very low overlap and question is very short (likely generic/off-topic)
    if low_overlap and len(question.split()) < 5:
        return True
    
    return False

