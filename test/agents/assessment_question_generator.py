import asyncio
import json
import re
import logging
import html
import hashlib
import os
import uuid
import time
import random
import threading
import socket
import ipaddress
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, replace
from collections import deque
from enum import Enum
from urllib.parse import urlparse
# from pydantic import BaseModel, Field  # Temporarily disabled due to version compatibility

# Generic Test Types and ONETS Framework
class GenericTestType(Enum):
    """Enumeration of supported generic test types"""
    PERSONALITY = "personality"
    PSYCHOMETRIC = "psychometric"
    COGNITIVE = "cognitive"
    APTITUDE = "aptitude"
    BEHAVIORAL = "behavioral"
    LEADERSHIP = "leadership"
    COMMUNICATION = "communication"
    TEAMWORK = "teamwork"
    PROBLEM_SOLVING = "problem_solving"
    CRITICAL_THINKING = "critical_thinking"
    EMOTIONAL_INTELLIGENCE = "emotional_intelligence"
    STRESS_MANAGEMENT = "stress_management"
    TIME_MANAGEMENT = "time_management"
    DECISION_MAKING = "decision_making"
    CREATIVITY = "creativity"
    ADAPTABILITY = "adaptability"
    WORK_STYLE = "work_style"
    MOTIVATION = "motivation"
    VALUES = "values"

class ONETSAssessmentType(Enum):
    """ONETS framework assessment categorization"""
    KNOWLEDGE_BASED = "knowledge_based"  # Factual subjects, theories, fields of knowledge
    SKILL_BASED = "skill_based"          # Practical tasks, skills, applications
    MIXED = "mixed"                      # Combination of both knowledge and skills

class GenericTestManager:
    """Manages generic test identification and question generation"""
    
    def __init__(self):
        self.test_type_keywords = self._initialize_test_type_keywords()
        self.knowledge_keywords = self._initialize_knowledge_keywords()
        self.skill_keywords = self._initialize_skill_keywords()
        self.question_banks = self._initialize_question_banks()
    
    def _initialize_test_type_keywords(self) -> Dict[GenericTestType, List[str]]:
        """Initialize keywords for identifying generic test types"""
        return {
            GenericTestType.PERSONALITY: [
                "personality", "personality test", "personality assessment", "personality questionnaire",
                "big five", "myers-briggs", "mbti", "disc", "enneagram", "personality traits"
            ],
            GenericTestType.PSYCHOMETRIC: [
                "psychometric", "psychometric test", "psychometric assessment", "psychological test",
                "psychological assessment", "psych test", "psych assessment"
            ],
            GenericTestType.COGNITIVE: [
                "cognitive", "cognitive ability", "cognitive test", "cognitive assessment",
                "iq test", "intelligence test", "mental ability", "cognitive ability test"
            ],
            GenericTestType.APTITUDE: [
                "aptitude", "aptitude test", "aptitude assessment", "ability test",
                "logical reasoning", "verbal reasoning", "numerical reasoning", "aptitude test"
            ],
            GenericTestType.BEHAVIORAL: [
                "behavioral", "behavioral test", "behavioral assessment", "behavior assessment",
                "behavioral interview", "behavioral questions"
            ],
            GenericTestType.LEADERSHIP: [
                "leadership", "leadership test", "leadership assessment", "leadership skills",
                "leadership style", "leadership questionnaire"
            ],
            GenericTestType.COMMUNICATION: [
                "communication", "communication test", "communication assessment", "communication skills",
                "communication test", "communication questionnaire"
            ],
            GenericTestType.TEAMWORK: [
                "teamwork", "teamwork test", "teamwork assessment", "team work", "collaboration",
                "team player", "teamwork skills"
            ],
            GenericTestType.PROBLEM_SOLVING: [
                "problem solving", "problem-solving", "problem solving test", "problem solving assessment",
                "analytical thinking", "critical thinking test"
            ],
            GenericTestType.CRITICAL_THINKING: [
                "critical thinking", "critical thinking test", "critical thinking assessment",
                "logical thinking", "analytical reasoning"
            ],
            GenericTestType.EMOTIONAL_INTELLIGENCE: [
                "emotional intelligence", "eq", "emotional quotient", "emotional intelligence test",
                "emotional intelligence assessment", "ei test"
            ],
            GenericTestType.STRESS_MANAGEMENT: [
                "stress management", "stress tolerance", "stress handling", "stress management test",
                "stress tolerance assessment", "stress management assessment"
            ],
            GenericTestType.TIME_MANAGEMENT: [
                "time management", "time planning", "time management test", "time management assessment",
                "time planning assessment", "time management skills"
            ],
            GenericTestType.DECISION_MAKING: [
                "decision making", "decision-making", "decision making test", "decision making assessment",
                "judgment", "decision making skills"
            ],
            GenericTestType.CREATIVITY: [
                "creativity", "creative thinking", "creativity test", "creativity assessment",
                "innovation", "creative problem solving"
            ],
            GenericTestType.ADAPTABILITY: [
                "adaptability", "flexibility", "adaptability test", "adaptability assessment",
                "change management", "adaptability skills"
            ],
            GenericTestType.WORK_STYLE: [
                "work style", "working style", "work style test", "work style assessment",
                "work preferences", "work style questionnaire"
            ],
            GenericTestType.MOTIVATION: [
                "motivation", "motivational", "motivation test", "motivation assessment",
                "motivation questionnaire", "motivation skills"
            ],
            GenericTestType.VALUES: [
                "values", "work values", "values test", "values assessment", "value system",
                "values questionnaire", "work values assessment"
            ]
        }
    
    def _initialize_knowledge_keywords(self) -> List[str]:
        """Initialize keywords for knowledge-based topics"""
        return [
            "theory", "theoretical", "concept", "principles", "fundamentals", "basics",
            "history", "historical", "facts", "factual", "knowledge", "understanding",
            "definition", "definitions", "explanation", "explanations", "analysis",
            "comparison", "compare", "contrast", "evaluation", "assessment", "review",
            "study", "learning", "education", "academic", "scholarly", "research",
            "information", "data", "statistics", "facts", "evidence", "proof",
            "documentation", "literature", "textbook", "curriculum", "syllabus"
        ]
    
    def _initialize_skill_keywords(self) -> List[str]:
        """Initialize keywords for skill-based topics"""
        return [
            "skill", "skills", "ability", "abilities", "competency", "competencies",
            "practice", "practical", "application", "apply", "implement", "implementation",
            "perform", "performance", "execute", "execution", "create", "creation",
            "build", "building", "develop", "development", "design", "designing",
            "solve", "solving", "problem-solving", "troubleshoot", "troubleshooting",
            "debug", "debugging", "code", "coding", "programming", "development",
            "software engineering", "data science", "machine learning", "artificial intelligence",
            "web development", "mobile development", "database", "networking", "security",
            "testing", "quality assurance", "devops", "deployment", "configuration",
            "optimization", "maintenance", "support", "training", "mentoring", "coaching"
        ]
    
    def identify_generic_test_type(self, topic: str) -> Optional[GenericTestType]:
        """Identify if a topic is a generic test type"""
        topic_lower = topic.lower().strip()
        
        # Priority order for checking (more specific first)
        test_type_priority = [
            GenericTestType.PSYCHOMETRIC,
            GenericTestType.PERSONALITY,
            GenericTestType.COGNITIVE,
            GenericTestType.APTITUDE,
            GenericTestType.BEHAVIORAL,
            GenericTestType.LEADERSHIP,
            GenericTestType.COMMUNICATION,
            GenericTestType.TEAMWORK,
            GenericTestType.PROBLEM_SOLVING,
            GenericTestType.CRITICAL_THINKING,
            GenericTestType.EMOTIONAL_INTELLIGENCE,
            GenericTestType.STRESS_MANAGEMENT,
            GenericTestType.TIME_MANAGEMENT,
            GenericTestType.DECISION_MAKING,
            GenericTestType.CREATIVITY,
            GenericTestType.ADAPTABILITY,
            GenericTestType.WORK_STYLE,
            GenericTestType.MOTIVATION,
            GenericTestType.VALUES
        ]
        
        for test_type in test_type_priority:
            keywords = self.test_type_keywords.get(test_type, [])
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # Check for exact match or word boundary match
                if (keyword_lower in topic_lower and 
                    (keyword_lower == topic_lower or 
                     topic_lower.startswith(keyword_lower + " ") or 
                     topic_lower.endswith(" " + keyword_lower) or 
                     " " + keyword_lower + " " in topic_lower)):
                    return test_type
        
        return None
    
    def categorize_onets_assessment_type(self, topic: str) -> ONETSAssessmentType:
        """Categorize topic using ONETS framework"""
        topic_lower = topic.lower()
        
        # Count knowledge and skill indicators
        knowledge_score = sum(1 for keyword in self.knowledge_keywords if keyword in topic_lower)
        skill_score = sum(1 for keyword in self.skill_keywords if keyword in topic_lower)
        
        # Determine assessment type based on scores
        if knowledge_score > skill_score:
            return ONETSAssessmentType.KNOWLEDGE_BASED
        elif skill_score > knowledge_score:
            return ONETSAssessmentType.SKILL_BASED
        elif knowledge_score > 0 and skill_score > 0 and abs(knowledge_score - skill_score) <= 1:
            return ONETSAssessmentType.MIXED
        else:
            # Default to skill-based for technical topics
            return ONETSAssessmentType.SKILL_BASED
    
    def _initialize_question_banks(self) -> Dict[GenericTestType, List[Dict[str, Any]]]:
        """Initialize standardized question banks for generic test types"""
        return {
            GenericTestType.PERSONALITY: [
                {
                    "question_text": "How do you typically respond to new social situations?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "social_behavior",
                    "options": {
                        "A": "I immediately engage with others and introduce myself",
                        "B": "I observe the situation first before participating",
                        "C": "I prefer to stay in the background and let others approach me",
                        "D": "I feel anxious and try to avoid such situations"
                    }
                },
                {
                    "question_text": "When working on a project, you prefer to:",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "work_preference",
                    "options": {
                        "A": "Work independently and present the final result",
                        "B": "Collaborate closely with team members throughout",
                        "C": "Take a leadership role and delegate tasks",
                        "D": "Follow clear instructions and work systematically"
                    }
                },
                {
                    "question_text": "How do you handle criticism or feedback?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "feedback_handling",
                    "options": {
                        "A": "I take it personally and feel defensive",
                        "B": "I listen carefully and ask for clarification",
                        "C": "I ignore it if I disagree with it",
                        "D": "I immediately try to justify my actions"
                    }
                },
                {
                    "question_text": "What motivates you most in your work?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "motivation",
                    "options": {
                        "A": "Recognition and praise from others",
                        "B": "Personal growth and learning opportunities",
                        "C": "Financial rewards and benefits",
                        "D": "Making a positive impact on others"
                    }
                },
                {
                    "question_text": "How do you typically spend your free time?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "leisure_preference",
                    "options": {
                        "A": "Socializing with friends and family",
                        "B": "Pursuing hobbies and personal interests",
                        "C": "Relaxing and recharging alone",
                        "D": "Learning new skills or taking courses"
                    }
                },
                {
                    "question_text": "When making decisions, you tend to:",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "decision_style",
                    "options": {
                        "A": "Rely on logical analysis and data",
                        "B": "Trust your intuition and gut feelings",
                        "C": "Seek input from others before deciding",
                        "D": "Consider all possible outcomes carefully"
                    }
                },
                {
                    "question_text": "How do you prefer to communicate with others?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "communication_style",
                    "options": {
                        "A": "Direct and straightforward",
                        "B": "Diplomatic and considerate",
                        "C": "Detailed and thorough",
                        "D": "Concise and to the point"
                    }
                },
                {
                    "question_text": "What type of work environment suits you best?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "environment_preference",
                    "options": {
                        "A": "Fast-paced and dynamic",
                        "B": "Structured and predictable",
                        "C": "Collaborative and team-oriented",
                        "D": "Quiet and focused"
                    }
                },
                {
                    "question_text": "How do you handle change?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "change_adaptability",
                    "options": {
                        "A": "I embrace change and see it as an opportunity",
                        "B": "I adapt gradually and need time to adjust",
                        "C": "I prefer stability and resist unnecessary changes",
                        "D": "I analyze the change before deciding how to respond"
                    }
                },
                {
                    "question_text": "What energizes you most?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "energy_source",
                    "options": {
                        "A": "Interacting with people and social activities",
                        "B": "Working on challenging problems and projects",
                        "C": "Having quiet time for reflection and planning",
                        "D": "Learning new things and gaining knowledge"
                    }
                },
                {
                    "question_text": "How do you approach risk-taking?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "risk_tolerance",
                    "options": {
                        "A": "I'm comfortable taking calculated risks",
                        "B": "I prefer to avoid risks when possible",
                        "C": "I enjoy taking bold risks for potential rewards",
                        "D": "I carefully evaluate risks before proceeding"
                    }
                },
                {
                    "question_text": "What is your preferred leadership style?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "leadership_style",
                    "options": {
                        "A": "Directive and decisive",
                        "B": "Collaborative and inclusive",
                        "C": "Supportive and encouraging",
                        "D": "Strategic and visionary"
                    }
                },
                {
                    "question_text": "How do you handle pressure and deadlines?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "pressure_response",
                    "options": {
                        "A": "I thrive under pressure and perform better",
                        "B": "I stay calm and focus on what I can control",
                        "C": "I feel stressed but push through to meet deadlines",
                        "D": "I prefer to work ahead to avoid last-minute pressure"
                    }
                },
                {
                    "question_text": "What type of feedback do you prefer?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "feedback_preference",
                    "options": {
                        "A": "Immediate and frequent feedback",
                        "B": "Detailed and constructive feedback",
                        "C": "Positive reinforcement and encouragement",
                        "D": "Honest and direct feedback"
                    }
                },
                {
                    "question_text": "How do you prefer to learn new things?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "learning_preference",
                    "options": {
                        "A": "Through hands-on experience and practice",
                        "B": "By reading and studying theoretical concepts",
                        "C": "Through discussion and collaboration with others",
                        "D": "By observing and following examples"
                    }
                },
                {
                    "question_text": "What drives your career decisions?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "career_motivation",
                    "options": {
                        "A": "Opportunities for advancement and growth",
                        "B": "Work-life balance and job security",
                        "C": "Making a meaningful impact and contribution",
                        "D": "Financial rewards and compensation"
                    }
                }
            ],
            GenericTestType.PSYCHOMETRIC: [
                {
                    "question_text": "Which of the following best describes your approach to problem-solving?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "problem_solving",
                    "options": {
                        "A": "I analyze the problem systematically before taking action",
                        "B": "I prefer to discuss the problem with others first",
                        "C": "I like to try multiple approaches quickly",
                        "D": "I research similar problems and apply proven solutions"
                    }
                },
                {
                    "question_text": "How do you typically handle stress in the workplace?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "stress_management",
                    "options": {
                        "A": "I take breaks and step away from the situation",
                        "B": "I focus on what I can control and ignore the rest",
                        "C": "I talk to colleagues or supervisors about the situation",
                        "D": "I work harder to resolve the source of stress"
                    }
                },
                {
                    "question_text": "When faced with a difficult decision, you typically:",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "decision_making",
                    "options": {
                        "A": "Gather as much information as possible before deciding",
                        "B": "Trust your intuition and make a quick decision",
                        "C": "Seek advice from trusted colleagues or mentors",
                        "D": "Consider the potential consequences of each option"
                    }
                },
                {
                    "question_text": "How do you prefer to receive instructions?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "communication_style",
                    "options": {
                        "A": "Written instructions with clear steps",
                        "B": "Verbal explanations with examples",
                        "C": "Visual demonstrations or diagrams",
                        "D": "Hands-on practice and trial-and-error"
                    }
                },
                {
                    "question_text": "What type of work environment do you find most productive?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "work_environment",
                    "options": {
                        "A": "Quiet, private space with minimal distractions",
                        "B": "Open, collaborative space with team interaction",
                        "C": "Flexible environment that can change as needed",
                        "D": "Structured environment with clear routines"
                    }
                },
                {
                    "question_text": "How do you typically respond to feedback or criticism?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "feedback_reception",
                    "options": {
                        "A": "I take it personally and feel defensive",
                        "B": "I listen carefully and ask for clarification",
                        "C": "I ignore it if I disagree with it",
                        "D": "I immediately try to justify my actions"
                    }
                },
                {
                    "question_text": "When working in a team, you prefer to:",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "teamwork_style",
                    "options": {
                        "A": "Take a leadership role and coordinate the team",
                        "B": "Contribute ideas and collaborate equally",
                        "C": "Focus on your specific tasks and responsibilities",
                        "D": "Support others and help resolve conflicts"
                    }
                },
                {
                    "question_text": "How do you handle tight deadlines?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "time_pressure",
                    "options": {
                        "A": "I work extra hours to meet the deadline",
                        "B": "I prioritize tasks and focus on the most important ones",
                        "C": "I ask for help or delegate some tasks",
                        "D": "I negotiate for more time if possible"
                    }
                },
                {
                    "question_text": "What motivates you most in your work?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "motivation",
                    "options": {
                        "A": "Recognition and praise from others",
                        "B": "Personal growth and learning opportunities",
                        "C": "Financial rewards and benefits",
                        "D": "Making a positive impact on others"
                    }
                },
                {
                    "question_text": "How do you approach learning new skills?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "learning_style",
                    "options": {
                        "A": "I prefer structured courses with clear objectives",
                        "B": "I learn best through hands-on practice and experimentation",
                        "C": "I like to study theory first before applying it",
                        "D": "I learn most effectively by working with others"
                    }
                },
                {
                    "question_text": "When you encounter a problem you can't solve immediately, you:",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "persistence",
                    "options": {
                        "A": "Keep trying different approaches until you find a solution",
                        "B": "Take a break and come back to it later with fresh perspective",
                        "C": "Ask for help from colleagues or experts",
                        "D": "Research similar problems and adapt existing solutions"
                    }
                },
                {
                    "question_text": "How do you prefer to communicate with your team?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "communication_preference",
                    "options": {
                        "A": "Face-to-face meetings and discussions",
                        "B": "Written communication like emails and messages",
                        "C": "Regular team meetings and status updates",
                        "D": "Informal conversations and quick check-ins"
                    }
                },
                {
                    "question_text": "What is your preferred way to organize your work?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "organization_style",
                    "options": {
                        "A": "Detailed to-do lists and schedules",
                        "B": "Flexible planning that adapts to changes",
                        "C": "Project-based organization with clear milestones",
                        "D": "Priority-based organization focusing on urgent tasks"
                    }
                },
                {
                    "question_text": "How do you handle conflicts with colleagues?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "conflict_resolution",
                    "options": {
                        "A": "I avoid confrontation and hope it resolves itself",
                        "B": "I address it directly and try to find a solution",
                        "C": "I seek mediation from a supervisor",
                        "D": "I document the issue and escalate it formally"
                    }
                },
                {
                    "question_text": "What type of projects do you find most engaging?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "project_preference",
                    "options": {
                        "A": "Long-term projects with complex challenges",
                        "B": "Short-term projects with quick results",
                        "C": "Collaborative projects involving multiple teams",
                        "D": "Independent projects where you have full control"
                    }
                }
            ],
            GenericTestType.COMMUNICATION: [
                {
                    "question_text": "How do you typically handle a disagreement with a colleague?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "conflict_resolution",
                    "options": {
                        "A": "I avoid confrontation and hope it resolves itself",
                        "B": "I address it directly and try to find a solution",
                        "C": "I seek mediation from a supervisor",
                        "D": "I document the issue and escalate it formally"
                    }
                },
                {
                    "question_text": "When presenting information to a group, you prefer to:",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "presentation_style",
                    "options": {
                        "A": "Use detailed slides with lots of information",
                        "B": "Focus on key points with visual aids",
                        "C": "Encourage interaction and questions throughout",
                        "D": "Provide handouts and speak briefly"
                    }
                },
                {
                    "question_text": "How do you ensure you understand someone's message correctly?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "listening_skills",
                    "options": {
                        "A": "I repeat back what I heard in my own words",
                        "B": "I ask specific questions about unclear points",
                        "C": "I take notes and review them later",
                        "D": "I observe their body language and tone"
                    }
                },
                {
                    "question_text": "How do you prefer to communicate important information?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "communication_preference",
                    "options": {
                        "A": "Face-to-face meetings for important discussions",
                        "B": "Written emails with detailed explanations",
                        "C": "Phone calls for immediate communication",
                        "D": "Team meetings with all stakeholders present"
                    }
                },
                {
                    "question_text": "When someone is speaking to you, you typically:",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "active_listening",
                    "options": {
                        "A": "Listen attentively and ask clarifying questions",
                        "B": "Take notes while they're speaking",
                        "C": "Make eye contact and nod to show understanding",
                        "D": "Wait for them to finish before responding"
                    }
                },
                {
                    "question_text": "How do you handle difficult conversations?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "difficult_conversations",
                    "options": {
                        "A": "I prepare thoroughly and plan my approach",
                        "B": "I address issues directly and honestly",
                        "C": "I try to find common ground and compromise",
                        "D": "I seek support from others before having the conversation"
                    }
                },
                {
                    "question_text": "What is your preferred way to give feedback?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "feedback_delivery",
                    "options": {
                        "A": "In private, one-on-one conversations",
                        "B": "In writing with specific examples",
                        "C": "In team meetings with everyone present",
                        "D": "Through regular check-ins and informal discussions"
                    }
                },
                {
                    "question_text": "How do you handle interruptions during conversations?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "conversation_management",
                    "options": {
                        "A": "I politely ask them to wait until I finish",
                        "B": "I stop and address their concern immediately",
                        "C": "I acknowledge them and return to my point",
                        "D": "I try to incorporate their input into the discussion"
                    }
                },
                {
                    "question_text": "When communicating with someone from a different culture, you:",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "cross_cultural_communication",
                    "options": {
                        "A": "Adapt your communication style to their preferences",
                        "B": "Use simple, clear language and avoid idioms",
                        "C": "Ask questions to understand their communication style",
                        "D": "Be patient and allow extra time for understanding"
                    }
                },
                {
                    "question_text": "How do you prefer to receive instructions?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "instruction_reception",
                    "options": {
                        "A": "Written instructions with clear steps",
                        "B": "Verbal explanations with examples",
                        "C": "Visual demonstrations or diagrams",
                        "D": "Hands-on practice and trial-and-error"
                    }
                },
                {
                    "question_text": "What is your approach to email communication?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "email_communication",
                    "options": {
                        "A": "Keep emails brief and to the point",
                        "B": "Provide detailed explanations and context",
                        "C": "Use bullet points and clear formatting",
                        "D": "Include all relevant information in one email"
                    }
                },
                {
                    "question_text": "How do you handle miscommunication?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "miscommunication_resolution",
                    "options": {
                        "A": "I clarify immediately to prevent further confusion",
                        "B": "I take responsibility and apologize if needed",
                        "C": "I ask questions to understand what went wrong",
                        "D": "I provide additional context to clarify my message"
                    }
                },
                {
                    "question_text": "When working with a team, you prefer to communicate:",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "team_communication",
                    "options": {
                        "A": "Through regular team meetings and updates",
                        "B": "Via shared documents and collaborative tools",
                        "C": "Through informal conversations and check-ins",
                        "D": "Using project management platforms and notifications"
                    }
                },
                {
                    "question_text": "How do you handle communication in virtual meetings?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "virtual_communication",
                    "options": {
                        "A": "I ensure everyone has a chance to speak",
                        "B": "I use visual aids and screen sharing effectively",
                        "C": "I keep meetings focused and time-bound",
                        "D": "I follow up with written summaries and action items"
                    }
                },
                {
                    "question_text": "What is your preferred way to express disagreement?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "disagreement_expression",
                    "options": {
                        "A": "I present alternative solutions constructively",
                        "B": "I ask questions to understand their perspective",
                        "C": "I provide evidence and data to support my view",
                        "D": "I suggest compromises that address both concerns"
                    }
                },
                {
                    "question_text": "How do you ensure your message is understood?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "message_clarity",
                    "options": {
                        "A": "I ask for confirmation and feedback",
                        "B": "I use simple language and avoid jargon",
                        "C": "I provide examples and illustrations",
                        "D": "I repeat key points in different ways"
                    }
                },
                {
                    "question_text": "When communicating with senior management, you:",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "upward_communication",
                    "options": {
                        "A": "Prepare concise summaries with key points",
                        "B": "Provide detailed reports with supporting data",
                        "C": "Focus on outcomes and business impact",
                        "D": "Schedule regular updates and check-ins"
                    }
                },
                {
                    "question_text": "How do you handle communication during a crisis?",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "category": "crisis_communication",
                    "options": {
                        "A": "I communicate frequently with clear updates",
                        "B": "I focus on facts and avoid speculation",
                        "C": "I coordinate with all stakeholders simultaneously",
                        "D": "I prioritize transparency and honesty"
                    }
                }
            ]
        }
    
    def get_questions_for_generic_test(self, test_type: GenericTestType, num_questions: int, difficulty: str) -> List[Dict[str, Any]]:
        """Get questions from the standardized bank for a generic test type"""
        questions = self.question_banks.get(test_type, [])
        
        if not questions:
            return []
        
        # Filter by difficulty if needed
        filtered_questions = [q for q in questions if q.get("difficulty", "medium").lower() == difficulty.lower()]
        
        # If no questions match the difficulty, use all questions
        if not filtered_questions:
            filtered_questions = questions
        
        # Return requested number of questions (respecting difficulty filter)
        return filtered_questions[:num_questions] if num_questions > 0 else filtered_questions

# Global instance
generic_test_manager = GenericTestManager()
from models.llm_invoker import invoke_llm
from agents.prompt_generator import generate_assessment_questions_prompt
from core.utils import (
    _mask, _sanitize_text_for_llm, _validate_state_inputs, _to_text, _clean_json_text, 
    _scan_balanced_json, _safe_json_loads, _extract_json_from_response, _create_error_response,
    _generate_request_id, _calculate_processing_time
)
from core.config import get_agent_config
from core.security import validate_tenant_id, redact_pii, filter_injection_attempts, sanitize_text_for_llm, PII_PATTERNS, INJECTION_FILTERS
from core.memory import BaseAgentMemory, get_agent_memory
from core.logging_helpers import AgentLogger, create_log_context, log_agent_completion
from langsmith.run_helpers import traceable

# Setup logging - use centralized loggers
log = logging.getLogger('main')
error_logger = logging.getLogger('error')

# Concurrency control - PERFORMANCE OPTIMIZATION: Increased default from 5 to 10 for better throughput
# Can be configured via QG_MAX_CONCURRENCY environment variable
SEM = asyncio.Semaphore(int(os.getenv("QG_MAX_CONCURRENCY", "10")))

# Model routing configuration
MODEL_ROUTING = json.loads(os.getenv("QG_MODEL_ROUTING", '{"coding":"gemini","mcq":"gemini","default":"gemini"}'))

# Get centralized configuration
config = get_agent_config("assessment_question_generator")

# Production Constants
TENANT_ID_REGEX = re.compile(r'^[a-zA-Z0-9_\-]{8,64}$')
USER_ID_REGEX = re.compile(r'^[a-zA-Z0-9_\-]{8,64}$')
MAX_PAYLOAD_SIZE = 100 * 1024  # 100KB
MAX_PROMPT_SIZE = config.max_prompt_chars
MAX_RESPONSE_SIZE = config.max_response_length
CONFIDENCE_THRESHOLD = config.confidence_threshold
QUALITY_THRESHOLD = 0.6
TIMEOUT_SECONDS = config.timeout_seconds
LLM_RETRY_ATTEMPTS = 3
LLM_BASE_BACKOFF = 0.4
CACHE_TTL_MINUTES = 30
MAX_CACHE_ENTRIES = 1000
MAX_TENANT_MEMORY_ENTRIES = 1000

# Use centralized PII patterns and injection filters
PII_PATTERNS = PII_PATTERNS
INJECTION_FILTERS = INJECTION_FILTERS

# Custom memory class for assessment question generator (extends base memory)
class AssessmentQuestionGeneratorMemory(BaseAgentMemory):
    def __init__(self, tenant_id: str = "default"):
        super().__init__(tenant_id, config.adaptation_window)
        # Add any assessment question generator specific fields here if needed

# Use centralized memory management
async def get_assessment_question_generator_memory(tenant_id: str = "default") -> AssessmentQuestionGeneratorMemory:
    """Get or create tenant-scoped assessment question generator memory."""
    return await get_agent_memory("assessment_question_generator", tenant_id, AssessmentQuestionGeneratorMemory)

# Global instances for production patterns
cache = {}             # tenant_id -> cache
metrics = {}           # tenant_id -> PerformanceMetrics

# Constants
ALLOWED_QUESTION_TYPES = {
    "mcq", "multiple_choice", "coding", "short_answer", "long_answer", "essay", "multi"
}

ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}

# Data validation functions (replacing Pydantic models)
def validate_question_type(qtype: str) -> bool:
    """Validate question type"""
    return qtype in ALLOWED_QUESTION_TYPES

def validate_difficulty_level(difficulty: str) -> bool:
    """Validate difficulty level"""
    return difficulty.lower() in ALLOWED_DIFFICULTIES

def validate_analysis_method(method: str) -> bool:
    """Validate analysis method"""
    valid_methods = {"llm", "cached", "rate_limited", "error"}
    return method in valid_methods

def validate_confidence_level(level: str) -> bool:
    """Validate confidence level"""
    valid_levels = {"high", "medium", "low"}
    return level in valid_levels

def validate_question_data(question: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean question data"""
    if not isinstance(question, dict):
        raise ValidationError("Question must be a dictionary")
    
    # Validate required fields
    if "question" not in question or not isinstance(question["question"], str):
        raise ValidationError("Question text is required and must be a string")
    
    if len(question["question"]) < 10 or len(question["question"]) > 1000:
        raise ValidationError("Question text must be between 10 and 1000 characters")
    
    if "type" not in question or not validate_question_type(question["type"]):
        raise ValidationError(f"Invalid question type: {question.get('type')}")
    
    if "difficulty" not in question or not validate_difficulty_level(question["difficulty"]):
        raise ValidationError(f"Invalid difficulty level: {question.get('difficulty')}")
    
    # Normalize difficulty to lowercase
    question["difficulty"] = question["difficulty"].lower()
    
    # Remove correct_answer if present (we don't want it in the output)
    if "correct_answer" in question:
        del question["correct_answer"]
    
    return question

def validate_question_set_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean question set data"""
    if not isinstance(data, dict):
        raise ValidationError("Question set must be a dictionary")
    
    if "questions" not in data or not isinstance(data["questions"], list):
        raise ValidationError("Questions must be a list")
    
    if len(data["questions"]) < 1 or len(data["questions"]) > 50:
        raise ValidationError("Must have between 1 and 50 questions")
    
    # Validate each question
    validated_questions = []
    for i, question in enumerate(data["questions"]):
        try:
            validated_question = validate_question_data(question)
            validated_questions.append(validated_question)
        except ValidationError as e:
            raise ValidationError(f"Question {i+1} validation failed: {str(e)}")
    
    data["questions"] = validated_questions
    
    # Validate topic (optional - LLM response may not include it)
    if "topic" in data:
        if not isinstance(data["topic"], str):
            raise ValidationError("Topic must be a string")
        if len(data["topic"]) < 1 or len(data["topic"]) > 200:
            raise ValidationError("Topic must be between 1 and 200 characters")
    
    # Validate type and difficulty
    if "type" in data and not validate_question_type(data["type"]):
        raise ValidationError(f"Invalid question set type: {data['type']}")
    
    if "difficulty" in data and not validate_difficulty_level(data["difficulty"]):
        raise ValidationError(f"Invalid difficulty level: {data['difficulty']}")
    
    return data

def validate_standard_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Validate standard response format"""
    if not isinstance(response, dict):
        raise ValidationError("Response must be a dictionary")
    
    # Validate required fields
    required_fields = ["ok", "analysis_method", "method_explain", "confidence_score", 
                      "confidence_level", "analysis_context", "questions", "total_questions",
                      "processing_time_seconds", "user_context", "assessment_id"]
    
    for field in required_fields:
        if field not in response:
            raise ValidationError(f"Missing required field: {field}")
    
    # Validate analysis method
    if not validate_analysis_method(response["analysis_method"]):
        raise ValidationError(f"Invalid analysis method: {response['analysis_method']}")
    
    # Validate confidence score
    if not isinstance(response["confidence_score"], (int, float)):
        raise ValidationError("Confidence score must be a number")
    
    if not 0.0 <= response["confidence_score"] <= 1.0:
        raise ValidationError("Confidence score must be between 0.0 and 1.0")
    
    # Validate confidence level
    if not validate_confidence_level(response["confidence_level"]):
        raise ValidationError(f"Invalid confidence level: {response['confidence_level']}")
    
    # Validate total questions
    if not isinstance(response["total_questions"], int) or response["total_questions"] < 0:
        raise ValidationError("Total questions must be a non-negative integer")
    
    # Validate processing time
    if not isinstance(response["processing_time_seconds"], (int, float)) or response["processing_time_seconds"] < 0:
        raise ValidationError("Processing time must be a non-negative number")
    
    return response

# Circuit breaker and rate limiting functionality is now handled by centralized middleware

class PerformanceMetrics:
    def __init__(self, max_entries: int = 5000):
        self.response_times = deque(maxlen=max_entries)
        self.confidence_scores = deque(maxlen=max_entries)
        self.method_counts = {"deterministic": 0, "llm": 0, "cached": 0, "error": 0, "rate_limited": 0}
        self.cache_hits = 0
        self.cache_misses = 0
        self.timeouts = 0
        self.circuit_breaker_events = deque(maxlen=100)
        self._lock = threading.Lock()

    def record_response_time(self, duration: float):
        with self._lock:
            self.response_times.append(duration)

    def record_confidence(self, confidence: float):
        with self._lock:
            self.confidence_scores.append(confidence)

    def record_method(self, method: str):
        with self._lock:
            if method in self.method_counts:
                self.method_counts[method] += 1

    def record_cache_hit(self):
        with self._lock:
            self.cache_hits += 1

    def record_cache_miss(self):
        with self._lock:
            self.cache_misses += 1

    def record_timeout(self):
        with self._lock:
            self.timeouts += 1

    def record_circuit_breaker_event(self, event: str):
        with self._lock:
            self.circuit_breaker_events.append({"event": event, "timestamp": time.time()})

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            response_times = list(self.response_times)
            confidence_scores = list(self.confidence_scores)
            
            return {
                "response_time_p95": sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0,
                "response_time_avg": sum(response_times) / len(response_times) if response_times else 0,
                "confidence_avg": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
                "method_split": dict(self.method_counts),
                "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
                "timeout_rate": self.timeouts / len(response_times) if response_times else 0,
                "circuit_breaker_events": list(self.circuit_breaker_events)
            }

class AnalysisMemory:
    def __init__(self, max_entries: int = MAX_TENANT_MEMORY_ENTRIES):
        self.memories = {}  # tenant_id:user_id -> memory data
        self.max_entries = max_entries
        self._lock = threading.Lock()
    
    def get_memory(self, tenant_id: str, user_id: str) -> Dict[str, Any]:
        key = f"{tenant_id}:{user_id}"
        with self._lock:
            return self.memories.get(key, {
                "analysis_history": [],
                "feedback_scores": [],
                "successful_patterns": [],
                "adaptation_metadata": {"ema_success": 0.5}
            })
    
    def update_memory(self, tenant_id: str, user_id: str, analysis_data: Dict[str, Any]):
        key = f"{tenant_id}:{user_id}"
        with self._lock:
            if key not in self.memories:
                self.memories[key] = {
                    "analysis_history": [],
                    "feedback_scores": [],
                    "successful_patterns": [],
                    "adaptation_metadata": {"ema_success": 0.5}
                }
            
            memory = self.memories[key]
            
            # Add to analysis history (keep last 50)
            memory["analysis_history"].append({
                "timestamp": time.time(),
                "analysis_id": analysis_data.get("assessment_id"),
                "method": analysis_data.get("analysis_method"),
                "confidence": analysis_data.get("confidence_score", 0.0)
            })
            memory["analysis_history"] = memory["analysis_history"][-50:]
            
            # Update EMA for success rate
            alpha = 0.15
            success = 1.0 if analysis_data.get("ok", False) else 0.0
            current_ema = memory["adaptation_metadata"]["ema_success"]
            memory["adaptation_metadata"]["ema_success"] = alpha * success + (1 - alpha) * current_ema
            
            # Enforce memory bounds
            if len(self.memories) > self.max_entries:
                oldest_key = min(self.memories.keys(), 
                               key=lambda k: self.memories[k]["analysis_history"][0]["timestamp"] 
                               if self.memories[k]["analysis_history"] else 0)
                del self.memories[oldest_key]

# Exception classes
class QuestionGenerationError(Exception):
    """Base exception for question generation errors"""
    pass

class ValidationError(QuestionGenerationError):
    """Raised when input validation fails"""
    pass

class GenerationTimeoutError(QuestionGenerationError):
    """Raised when generation times out"""
    pass

class LLMResponseError(QuestionGenerationError):
    """Raised when LLM response is invalid"""
    pass

@dataclass
class QuestionGeneratorConfig:
    """Configuration for question generation"""
    generation_timeout: float = 90.0  # FIX: Increased from 60.0 to 90.0 for better reliability
    default_model: str = "gemini"
    retry_attempts: int = 2
    max_response_length: int = 50000
    enable_caching: bool = True
    confidence_threshold: float = 0.8
    
    @classmethod
    def from_env(cls):
        """Create configuration from environment variables"""
        return cls(
            generation_timeout=float(os.getenv('QG_GENERATION_TIMEOUT', 90.0)),  # FIX: Increased default from 60.0 to 90.0
            default_model=os.getenv('QG_DEFAULT_MODEL', 'gemini'),
            retry_attempts=int(os.getenv('QG_RETRY_ATTEMPTS', 2)),
            max_response_length=int(os.getenv('QG_MAX_RESPONSE_LENGTH', 50000)),
            enable_caching=os.getenv('QG_ENABLE_CACHING', 'true').lower() == 'true',
            confidence_threshold=float(os.getenv('QG_CONFIDENCE_THRESHOLD', 0.8))
        )

# Security and Utility Functions


def validate_user_id(user_id: str) -> bool:
    """Validate user ID format"""
    return bool(USER_ID_REGEX.match(user_id)) if user_id else False



def sanitize_url(url: str) -> str:
    """Sanitize and validate URL"""
    if not url:
        return ""
    
    try:
        parsed = urlparse(url)
        
        # Only allow http/https
        if parsed.scheme not in ['http', 'https']:
            return ""
        
        # Drop query and fragment
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        
        # Validate host length
        if len(parsed.hostname or "") > 253:
            return ""
        
        # Check for dangerous patterns
        for pattern in INJECTION_FILTERS:
            if pattern.search(clean_url):
                return ""
        
        return clean_url
    except Exception:
        return ""

def sanitize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize state for security"""
    sanitized = {}
    
    for key, value in state.items():
        if isinstance(value, str):
            # Redact PII and sanitize URLs
            sanitized_value = redact_pii(value)
            if 'url' in key.lower():
                sanitized_value = sanitize_url(sanitized_value)
            sanitized[key] = sanitized_value
        elif isinstance(value, dict):
            sanitized[key] = sanitize_state(value)
        elif isinstance(value, list):
            sanitized[key] = [sanitize_state(item) if isinstance(item, dict) else redact_pii(str(item)) if isinstance(item, str) else item for item in value]
        else:
            sanitized[key] = value
    
    return sanitized

def validate_payload_size(state: Dict[str, Any]) -> bool:
    """Validate payload size"""
    try:
        serialized = json.dumps(state)
        return len(serialized) <= MAX_PAYLOAD_SIZE
    except Exception:
        return False

# Circuit breaker and rate limiter functions removed - now handled by centralized middleware

def get_or_create_cache(tenant_id: str) -> Dict[str, Any]:
    """Get or create cache for tenant"""
    if tenant_id not in cache:
        cache[tenant_id] = {}
    return cache[tenant_id]

def get_or_create_metrics(tenant_id: str) -> PerformanceMetrics:
    """Get or create metrics for tenant"""
    if tenant_id not in metrics:
        metrics[tenant_id] = PerformanceMetrics()
    return metrics[tenant_id]

def get_or_create_memory() -> AnalysisMemory:
    """Get or create global memory manager"""
    if not hasattr(get_or_create_memory, 'instance'):
        get_or_create_memory.instance = AnalysisMemory()
    return get_or_create_memory.instance

def generate_generic_test_questions(topic: str, difficulty: str, num_questions: int, assessment_type: str) -> Dict[str, Any]:
    """
    Generate questions for generic test types using standardized question banks.
    
    Args:
        topic: The assessment topic
        difficulty: Difficulty level
        num_questions: Number of questions requested
        assessment_type: Type of assessment
        
    Returns:
        Dictionary containing generated questions in the expected format
    """
    # Identify if this is a generic test type
    generic_test_type = generic_test_manager.identify_generic_test_type(topic)
    
    if not generic_test_type:
        return None
    
    log.info(f"Generating questions for generic test type: {generic_test_type.value}")
    
    # Get questions from the standardized bank
    questions = generic_test_manager.get_questions_for_generic_test(
        generic_test_type, 
        num_questions, 
        difficulty
    )
    
    if not questions:
        log.warning(f"No questions available for generic test type: {generic_test_type.value}")
        return None
    
    # Format questions according to the expected structure
    formatted_questions = []
    for question in questions:
        formatted_question = {
            "question_text": question["question_text"],
            "question_type": question["question_type"],
            "difficulty": question.get("difficulty", difficulty),
            "category": question.get("category", "general")
        }
        
        # Add options for MCQ questions
        if question["question_type"] == "mcq" and "options" in question:
            formatted_question["options"] = question["options"]
        
        formatted_questions.append(formatted_question)
    
    # Return in the expected format
    if assessment_type == "multi":
        return {
            "assessment_questions": {
                "multi": formatted_questions
            }
        }
    else:
        return {
            "assessment_questions": {
                assessment_type: formatted_questions
            }
        }

# Deterministic function removed - using LLM-only approach

@traceable(name="assessment_question_generator_agent")
async def assessment_question_generator_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced assessment question generator with LLM-only approach.
    """
    # Use centralized logging
    log_context = create_log_context("assessment_question_generator", state.get("tenant_id", "default"))
    start_time = log_context["start_time"]
    request_id = log_context["request_id"]
    
    log.info(f"\n--- Entering Assessment Question Generator Agent ---")
    
    # Get tenant-scoped memory
    assessment_memory = await get_assessment_question_generator_memory(state.get("tenant_id", "default"))
    
    # Input validation - use assessment_plan directly
    assessment_plan = state.get("assessment_plan", {})
    user_context = state.get("user_context", {})
    
    if not assessment_plan:
        processing_time = _calculate_processing_time(start_time)
        return _create_error_response("Missing required inputs: assessment_plan", processing_time)
    
    # Security validation
    if len(str(assessment_plan)) > MAX_PROMPT_SIZE:
        AgentLogger.log_warning(log_context, f"Assessment plan too large, truncating")
        assessment_plan = str(assessment_plan)[:MAX_PROMPT_SIZE]
    
    try:
        # Extract parameters from assessment_plan for prompt generation
        # Handle both list and dict formats for assessment_plan
        if isinstance(assessment_plan, list) and len(assessment_plan) > 0:
            # If it's a list, take the first item
            assessment_item = assessment_plan[0]
        elif isinstance(assessment_plan, dict):
            # If it's already a dict, use it directly
            assessment_item = assessment_plan
        else:
            # Fallback to empty dict
            assessment_item = {}
        
        topic = assessment_item.get("topic", "Unknown Topic")
        difficulty = assessment_item.get("difficulty", "Medium")
        raw_num_questions = assessment_item.get("num_questions", 5)
        assessment_type = assessment_item.get("type", "multi")
        
        # Handle num_questions format - extract integer value
        if isinstance(raw_num_questions, dict):
            # For multi-type assessments, preserve the dict structure
            if assessment_type == "multi":
                num_questions = raw_num_questions  # Keep the dict for multi-type
                total_questions = sum(raw_num_questions.values()) if raw_num_questions else 5
            else:
                # For single-type assessments, extract the specific type count
                if assessment_type in raw_num_questions:
                    num_questions = raw_num_questions[assessment_type]
                else:
                    num_questions = sum(raw_num_questions.values()) if raw_num_questions else 5
                total_questions = num_questions
        else:
            num_questions = int(raw_num_questions) if raw_num_questions else 5
            total_questions = num_questions
        
        # Ensure total_questions is a positive integer
        total_questions = max(1, min(50, int(total_questions)))
        
        # Check if this is a generic test type first
        generic_questions = generate_generic_test_questions(topic, difficulty, total_questions, assessment_type)
        
        if generic_questions:
            # Use generic questions from standardized bank
            log.info(f"Using generic test questions for topic: {topic}")
            questions = generic_questions
            # Validate generic questions were actually generated
            assessment_questions_dict = questions.get('assessment_questions', {})
            total_generic = sum(len(q_list) if isinstance(q_list, list) else 0 
                              for q_list in assessment_questions_dict.values())
            if total_generic == 0:
                log.warning(f"Generic questions returned empty, falling back to LLM generation")
                generic_questions = None  # Force fallback to LLM
            else:
                log.info(f"✅ Using {total_generic} generic questions from question bank")
        
        if not generic_questions:
            # Fallback to LLM generation
            log.info(f"Generating custom questions via LLM for topic: {topic}")
            # Generate prompt using extracted parameters
            prompt = generate_assessment_questions_prompt(topic, difficulty, num_questions, assessment_type)
            
            # Validate prompt size
            if len(prompt) > MAX_PROMPT_SIZE:
                AgentLogger.log_warning(log_context, f"Prompt too large, truncating from {len(prompt)} to {MAX_PROMPT_SIZE}")
                prompt = prompt[:MAX_PROMPT_SIZE]
            
            log.info("🤖 Calling LLM for assessment question generation...")
            
            # NOTE: Gemini's structured output has a known issue with complex nested structures
            # (Dict[str, List[AssessmentQuestion]]). It consistently returns empty dicts.
            # We skip the structured output attempt and use JSON parsing directly for better performance.
            
            # Generate questions using reliable JSON parsing method
            try:
                # FIX: Use configurable timeout instead of hardcoded 60s
                llm_response = await asyncio.wait_for(
                    invoke_llm(
                        prompt=prompt,
                        task_type="assessment_generation",
                        agent_name="assessment_question_generator"
                    ), 
                    timeout=TIMEOUT_SECONDS
                )
                
                raw_content = (
                    llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                )
                
                if not raw_content.strip():
                    raise LLMResponseError("Empty response from LLM")
                
                # Extract and parse JSON from response (offloaded — regex on large strings)
                extracted_json_str = await asyncio.to_thread(robust_json_extraction, raw_content)
                if not extracted_json_str:
                    raise LLMResponseError("No JSON found in LLM response")
                
                parsed_obj = await asyncio.to_thread(json.loads, extracted_json_str)

                # Normalize into { "assessment_questions": { type: [AssessmentQuestion...] } }
                normalized: Dict[str, Any] = {}
                aq = parsed_obj.get("assessment_questions", parsed_obj.get("questions"))

                # If assessment_questions is a JSON-encoded string, parse it
                if isinstance(aq, str):
                    try:
                        aq = json.loads(aq)
                    except Exception:
                        aq = {}

                if isinstance(aq, list):
                    # If we just got a list of questions, place under inferred assessment_type
                    normalized = {"assessment_questions": {assessment_type: aq}}
                elif isinstance(aq, dict):
                    normalized = {"assessment_questions": aq}
                else:
                    # Try to coerce from alternate shapes
                    normalized = {"assessment_questions": {assessment_type: parsed_obj.get("questions", [])}}

                questions = normalized

            except json.JSONDecodeError as json_err:
                error_msg = f"Failed to parse JSON from LLM response: {json_err}"
                log.error(error_msg)
                raise LLMResponseError(error_msg)
            except Exception as gen_err:
                error_msg = f"Question generation failed: {gen_err}"
                log.error(error_msg)
                raise LLMResponseError(error_msg)
            
            # Validate that questions were actually generated
            assessment_questions = questions.get('assessment_questions', {})
            if not assessment_questions:
                error_msg = "No questions generated. LLM returned empty assessment_questions."
                log.error(error_msg)
                raise LLMResponseError(error_msg)
            
            total_generated = 0
            question_types = []
            for q_type, q_list in assessment_questions.items():
                if isinstance(q_list, list):
                    count = len(q_list)
                    total_generated += count
                    if count > 0:
                        question_types.append(f"{q_type}({count})")
            
            if total_generated == 0:
                error_msg = f"No questions found in assessment_questions. Keys present: {list(assessment_questions.keys())}"
                log.error(error_msg)
                raise LLMResponseError(error_msg)
            
            log.info(f"✅ Generated {total_generated} questions: {', '.join(question_types)}")
        
        # Randomize correct answer positions for MCQ questions to prevent bias
        questions = _randomize_mcq_answer_positions(questions)
        
        processing_time = _calculate_processing_time(start_time)
        
        # Record successful generation
        await assessment_memory.record_attempt(
            'question_generation', 'llm', True, 0.8, processing_time
        )
        
        # Calculate total questions across all types
        assessment_questions_dict = questions.get('assessment_questions', {})
        total_questions_count = 0
        for q_type, q_list in assessment_questions_dict.items():
            if isinstance(q_list, list):
                total_questions_count += len(q_list)
        
        # Log success
        log_agent_completion(log_context, {
            "success": True,
            "questions_count": total_questions_count,
            "question_types": list(assessment_questions_dict.keys())
        }, "llm", processing_time)
        
        # Generate assessment ID for reference
        assessment_id = state.get("assessment_id", str(uuid.uuid4()))
        
        # Update state - use the correct field name for AgentState
        state["generated_questions"] = assessment_questions_dict
        state["total_questions"] = total_questions_count
        state["confidence_level"] = "high"
        state["analysis_context"] = "question_generation"
        state["processing_time_seconds"] = processing_time
        state["assessment_id"] = assessment_id
        state["request_id"] = request_id
        
        
        return state
        
    except Exception as e:
        processing_time = _calculate_processing_time(start_time)
        AgentLogger.log_error(log_context, f"Assessment question generation failed: {str(e)}", processing_time)
        
        # Record failure
        await assessment_memory.record_attempt(
            'question_generation', 'llm', False, 0.0, processing_time
        )
        
        return _create_error_response(f"Assessment question generation failed: {str(e)}", processing_time)

def _randomize_mcq_answer_positions(questions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Randomize the correct answer position for MCQ questions to prevent bias.
    Shuffles options and updates expected_answer accordingly.
    
    Args:
        questions: Dictionary containing assessment_questions with MCQ questions
        
    Returns:
        Dictionary with randomized answer positions
    """
    if not isinstance(questions, dict):
        return questions
    
    assessment_questions = questions.get("assessment_questions", {})
    if not isinstance(assessment_questions, dict):
        return questions
    
    # Process all question lists (multi, mcq, etc.)
    for question_type, question_list in assessment_questions.items():
        if not isinstance(question_list, list):
            continue
        
        for question in question_list:
            if not isinstance(question, dict):
                continue
            
            # Only process MCQ questions
            if question.get("question_type") != "mcq":
                continue
            
            options = question.get("options")
            expected_answer = question.get("expected_answer", "")
            
            # Skip if no options or invalid expected_answer
            if not isinstance(options, dict) or not expected_answer:
                continue
            
            # Validate expected_answer is A, B, C, or D
            expected_answer = str(expected_answer).strip().upper()
            if expected_answer not in ["A", "B", "C", "D"]:
                continue
            
            # Get the correct answer text
            correct_text = options.get(expected_answer)
            if not correct_text:
                continue
            
            # Validate option count before shuffling to prevent data loss
            option_count = len(options)
            expected_count = 4  # Standard MCQ format: A, B, C, D
            
            if option_count != expected_count:
                log.warning(
                    f"MCQ question has {option_count} options, expected {expected_count}. "
                    f"Skipping randomization to prevent data loss. Question: {question.get('question_text', 'Unknown')[:50]}"
                )
                continue
            
            # Create list of option texts (values) for shuffling
            option_texts = list(options.values())
            
            # Validate that we have the expected number of texts
            if len(option_texts) != expected_count:
                log.warning(
                    f"Option texts count ({len(option_texts)}) doesn't match options dict count ({option_count}). "
                    f"Skipping randomization."
                )
                continue
            
            # Shuffle the option texts
            random.shuffle(option_texts)
            
            # Rebuild options dict with shuffled texts but labels in order (A, B, C, D)
            option_labels = ["A", "B", "C", "D"]
            shuffled_options = {label: text for label, text in zip(option_labels, option_texts)}
            
            # Final validation: ensure shuffled options count matches expected
            if len(shuffled_options) != expected_count:
                log.warning(
                    f"Shuffled options count ({len(shuffled_options)}) doesn't match expected count ({expected_count}). "
                    f"Skipping update to prevent data loss."
                )
                continue
            
            # Find the new position of the correct answer
            new_correct_label = None
            for label, text in shuffled_options.items():
                if text == correct_text:
                    new_correct_label = label
                    break
            
            # Update question with shuffled options and new expected_answer
            if new_correct_label:
                question["options"] = shuffled_options
                question["expected_answer"] = new_correct_label
    
    return questions

def robust_json_extraction(response: str) -> str:
    """Robustly extract JSON from LLM response"""
    # Try to find JSON blocks first
    json_blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_blocks:
        return json_blocks[0]
    
    # Try raw JSON
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json_match.group()
    
    # Try balanced braces
    start = response.find('{')
    if start != -1:
        brace_count = 0
        for i, char in enumerate(response[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return response[start:i+1]
    
    return response

def calculate_confidence_level(confidence: float) -> str:
    """Calculate confidence level from score"""
    if confidence >= 0.8:
        return "high"
    elif confidence >= 0.6:
        return "medium"
    else:
        return "low"

def generate_analysis_id() -> str:
    """Generate unique analysis ID"""
    return f"qg_{int(time.time())}_{uuid.uuid4().hex[:8]}"

# Simplified planning functions
def create_simple_plan(assessment_details: Dict[str, Any], user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create simple generation plan"""
    topic = assessment_details.get("topic", "")
    assessment_type = assessment_details.get("type", "multi")
    difficulty = assessment_details.get("difficulty", "medium")
    num_questions = assessment_details.get("num_questions", 5)
    
    # Simple subtopics
    subtopics = [topic, f"{topic} Fundamentals", f"{topic} Applications"]
    
    # Create simple goals
    goals = []
    if assessment_type == "multi":
        # Multi-type assessment: distribute across types
        mcq_ct = max(2, num_questions // 3)
        sa_ct = max(1, num_questions // 4)
        last_ct = max(1, num_questions - mcq_ct - sa_ct)
        
        goals.extend([
            {"type": "mcq", "count": mcq_ct, "subtopics": subtopics[:2], "difficulty": difficulty},
            {"type": "short_answer", "count": sa_ct, "subtopics": subtopics[1:3], "difficulty": difficulty},
            {"type": "coding", "count": last_ct, "subtopics": [f"{topic} Implementation"], "difficulty": difficulty}
        ])
    else:
        # Single-type assessment
        goals.append({"type": assessment_type, "count": num_questions, "subtopics": subtopics, "difficulty": difficulty})
    
    # Simple model routing
    model_routing = {
        "mcq": MODEL_ROUTING.get("mcq", "gemini"),
        "coding": MODEL_ROUTING.get("coding", "gemini"),
        "short_answer": MODEL_ROUTING.get("default", "gemini"),
        "long_answer": MODEL_ROUTING.get("default", "gemini")
    }
    
    return {
        "goals": goals,
        "strategy": "parallel",
        "model_routing": model_routing,
        "topic": topic
    }

# Utility functions
def _validate_difficulty(difficulty: str) -> str:
    """Validate and normalize difficulty level"""
    difficulty = difficulty.lower().strip()
    if difficulty not in ALLOWED_DIFFICULTIES:
        log.warning(f"Invalid difficulty '{difficulty}', defaulting to 'medium'")
        return "medium"
    return difficulty

def _validate_assessment_type(assessment_type: str) -> str:
    """Validate and normalize assessment type"""
    assessment_type = assessment_type.lower().strip()
    if assessment_type not in ALLOWED_QUESTION_TYPES:
        log.warning(f"Invalid assessment type '{assessment_type}', defaulting to 'multi'")
        return "multi"
    return assessment_type

def _validate_num_questions(num_questions: Union[int, Dict[str, int]], config: QuestionGeneratorConfig) -> Dict[str, int]:
    """Validate and normalize number of questions"""
    if isinstance(num_questions, int):
        return {"default": min(max(1, num_questions), 50)}
    elif isinstance(num_questions, dict):
        validated = {}
        for qtype, count in num_questions.items():
            validated[qtype] = min(max(1, count), 50)
        return validated
    else:
        return {"default": 5}

def sanitize_for_llm(text: str, config: QuestionGeneratorConfig) -> str:
    """Sanitize text for LLM input"""
    if not text:
        return "Unknown Topic"
    
    # Basic sanitization
    text = html.escape(text)
    text = text.strip()
    
    # Limit length
    if len(text) > 200:
        text = text[:200] + "..."
    
    return text



def _generate_cache_key(topic: str, difficulty: str, num_questions: Dict[str, int], 
                       assessment_type: str, user_context: Optional[Dict[str, Any]] = None) -> str:
    """Generate cache key for questions"""
    key_data = {
        "topic": topic,
        "difficulty": difficulty,
        "num_questions": num_questions,
        "assessment_type": assessment_type
    }
    
    if user_context:
        key_data["user_id"] = user_context.get("user_id", "anonymous")
    
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_string.encode()).hexdigest()

async def _safe_parse_llm_response(content: str, config: QuestionGeneratorConfig) -> Dict[str, Any]:
    """Safely parse LLM response with error handling"""
    try:
        if content.strip().startswith('{') or content.strip().startswith('['):
            parsed = await asyncio.to_thread(json.loads, content)
            return parsed
        
        json_str = await asyncio.to_thread(robust_json_extraction, content)
        parsed = await asyncio.to_thread(json.loads, json_str)
        return parsed
        
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse JSON response: {e}")
        return {
            "questions": [{"question": content.strip(), "type": "unknown"}],
            "raw_response": True
        }

async def _generate_questions_for_assessment(
    assessment_item: Dict[str, Any], 
    config: QuestionGeneratorConfig
) -> Dict[str, Any]:
    """Generate questions for a single assessment item"""
    details = assessment_item.get("assessment", {})
    
    try:
        # Validate and sanitize inputs
        topic = sanitize_for_llm(str(details.get("topic", "Unknown Topic")), config=config)
        difficulty = _validate_difficulty(str(details.get("difficulty", "Medium")))
        assessment_type = _validate_assessment_type(str(details.get("type", "multi")))
        
        # Validate num_questions
        raw_num_questions = details.get("num_questions", {})
        if isinstance(raw_num_questions, int):
            num_questions = {"default": raw_num_questions}
        else:
            num_questions = _validate_num_questions(raw_num_questions, config)
        
        log.info(f"Generating questions for topic: {topic}, type: {assessment_type}")
        
        # Check cache if enabled
        cache_key = None
        if config.enable_caching:
            cache_key = _generate_cache_key(topic, difficulty, num_questions, assessment_type, details.get("user_context"))
            # Note: Cache is now handled at the main agent level
        
        # Generate prompt for LLM
        # For multi-type assessments, pass the full num_questions dict to preserve all question type counts
        # For single-type assessments, extract the specific count
        if assessment_type == "multi" and isinstance(num_questions, dict):
            # Pass the full dict to preserve coding, mcq, short, long counts
            prompt_num_questions = num_questions
        else:
            # Extract single value for single-type assessments
            prompt_num_questions = num_questions.get("default") or num_questions.get(assessment_type) or 5
            if not isinstance(prompt_num_questions, int):
                prompt_num_questions = next(iter(num_questions.values())) if num_questions else 5
        
        prompt = generate_assessment_questions_prompt(topic, difficulty, prompt_num_questions, assessment_type)
        
        # Call LLM with timeout and retry logic
        # FIX: Use full timeout for each attempt (not divided by retry_attempts)
        # This ensures each retry has enough time to complete
        for attempt in range(config.retry_attempts):
            try:
                llm_response = await asyncio.wait_for(
                    invoke_llm(
                        prompt=prompt,
                        task_type="assessment_generation",
                        agent_name="assessment_question_generator"
                    ),
                    timeout=config.generation_timeout  # Use full timeout for each attempt
                )
                
                # Ensure we always get string content
                content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                
                if not content.strip():
                    raise LLMResponseError("Empty response from LLM")
                
                # Parse and validate response
                generated_questions = await _safe_parse_llm_response(content, config)
                break
                
            except asyncio.TimeoutError:
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, etc.
                log.warning(
                    f"LLM call timeout on attempt {attempt + 1}/{config.retry_attempts}, "
                    f"retrying in {wait_time}s..."
                )
                if attempt < config.retry_attempts - 1:
                    await asyncio.sleep(wait_time)
                else:
                    raise GenerationTimeoutError(
                        f"LLM generation timeout after {config.retry_attempts} attempts "
                        f"(each attempt had {config.generation_timeout}s timeout)"
                    )
            except Exception as e:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                log.error(f"LLM call failed on attempt {attempt + 1}: {e}")
                if attempt < config.retry_attempts - 1:
                    await asyncio.sleep(wait_time)
                else:
                    raise LLMResponseError(f"LLM generation failed after {config.retry_attempts} attempts: {e}")
        
                # Store in cache if enabled
                if config.enable_caching and cache_key:
                    # Note: Cache is now handled at the main agent level
                    pass
        
        # Enforce expected question counts universally (handles LLM under/over generation)
        try:
            expected_total = int(prompt_num_questions)
        except Exception:
            expected_total = 5

        def _extract_q_list(container: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
            # Support multiple possible shapes from different models
            if not isinstance(container, dict):
                return "multi", []
            # Common wrappers
            root = container
            if "assessment_questions" in root and isinstance(root["assessment_questions"], dict):
                root = root["assessment_questions"]
            # Determine primary key
            for key in ("multi", "mcq", "short_answer", "long_answer", "coding"):
                lst = root.get(key)
                if isinstance(lst, list):
                    return key, lst
            # Fallback: first list value
            for k, v in root.items():
                if isinstance(v, list):
                    return str(k), v
            return "multi", []

        def _set_q_list(container: Dict[str, Any], key: str, new_list: List[Dict[str, Any]]) -> Dict[str, Any]:
            # Mirror the discovered structure when writing back
            if not isinstance(container, dict):
                return {"assessment_questions": {key: new_list}}
            if "assessment_questions" in container and isinstance(container["assessment_questions"], dict):
                container["assessment_questions"][key] = new_list
                return container
            container[key] = new_list
            return container

        try:
            primary_key, q_list = _extract_q_list(generated_questions)
            if not isinstance(q_list, list):
                q_list = []

            # Trim excess
            if len(q_list) > expected_total:
                q_list = q_list[:expected_total]

            # If short, attempt one compact fix-up call to synthesize missing items
            missing = expected_total - len(q_list)
            if missing > 0:
                try:
                    # Build a minimal continuation prompt
                    fix_prompt = (
                        "Generate additional assessment questions in JSON only.\n"
                        f"Topic: {topic}\n"
                        f"Difficulty: {difficulty}\n"
                        f"Assessment Type: {assessment_type}\n"
                        f"Need exactly {missing} more questions. Follow the same schema and include expected_answer.\n"
                        "Return as an array under the key 'items'."
                    )
                    # FIX: Use reasonable timeout for fix calls (at least 30s, or half of timeout_seconds)
                    fix_timeout = max(30, int(TIMEOUT_SECONDS * 0.5))
                    fix_resp = await asyncio.wait_for(
                        invoke_llm(
                            prompt=fix_prompt,
                            task_type="assessment_generation",
                            agent_name="assessment_question_generator"
                        ),
                        timeout=fix_timeout
                    )
                    fix_content = fix_resp.content if hasattr(fix_resp, 'content') else str(fix_resp)
                    fix_parsed = await _safe_parse_llm_response(fix_content, config)
                    add_items = []
                    if isinstance(fix_parsed, dict) and isinstance(fix_parsed.get("items"), list):
                        add_items = fix_parsed["items"]
                    elif isinstance(fix_parsed, list):
                        add_items = fix_parsed
                    # Normalize question_type field if missing
                    for it in add_items:
                        if isinstance(it, dict) and not it.get("question_type"):
                            it["question_type"] = primary_key if primary_key != "multi" else (it.get("question_type") or "mcq")
                    # Append up to missing
                    if add_items:
                        q_list.extend(add_items[:missing])
                except Exception:
                    # As a last resort, duplicate with minor tag to meet exact count for downstream systems
                    if q_list:
                        template = q_list[-1]
                        for i in range(missing):
                            clone = dict(template) if isinstance(template, dict) else template
                            if isinstance(clone, dict):
                                qt = clone.get("question_text", "Question")
                                clone["question_text"] = f"{qt} (variant {i+1})"
                            q_list.append(clone)

            # Write back normalized list
            generated_questions = _set_q_list(generated_questions if isinstance(generated_questions, dict) else {}, primary_key, q_list)
        except Exception:
            # Leave as-is on any unexpected error
            pass

        # Add generated questions to assessment
        details["generated_questions"] = generated_questions
        assessment_item["assessment"] = details
        
        return assessment_item
        
    except Exception as e:
        log.error(f"Question generation failed: {e}")
        details["error"] = str(e)
        assessment_item["assessment"] = details
        return assessment_item

async def _execute_simple_generation(plan: Dict[str, Any], config: QuestionGeneratorConfig) -> List[Dict[str, Any]]:
    """Execute simple generation plan with optimized parallel execution"""
    results = []
    
    # PERFORMANCE OPTIMIZATION: Always use parallel execution for better performance
    # The sequential fallback is removed as parallel execution is always faster and safer
    # with proper exception handling via return_exceptions=True
    tasks = []
    
    for goal in plan["goals"]:
        assessment_item = {
            "assessment": {
                "topic": plan["topic"],
                "type": goal["type"],
                "difficulty": goal["difficulty"],
                "num_questions": goal["count"],
                "subtopics": goal["subtopics"]
            }
        }
        
        # Use model routing
        model = plan["model_routing"].get(goal["type"], config.default_model)
        goal_config = replace(config, default_model=model)
        
        tasks.append(_generate_questions_for_assessment(assessment_item, goal_config))
    
    # Execute all goals in parallel for maximum performance
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions gracefully
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            log.error(f"Goal {i} failed: {result}")
            results[i] = {
                "assessment": {
                    "topic": plan["topic"],
                    "type": plan["goals"][i]["type"],
                    "error": str(result)
                }
            }
    
    return results

def _extract_assessment_details(state: Dict[str, Any], config: QuestionGeneratorConfig) -> Dict[str, Any]:
    """Extract assessment details from state"""
    if state.get("assessment_plan"):
        assessment_plan = state["assessment_plan"]
        
        # Handle both single object and array formats
        if isinstance(assessment_plan, list) and len(assessment_plan) > 0:
            # Array format: take the first item
            first_assessment = assessment_plan[0]
        elif isinstance(assessment_plan, dict):
            # Single object format: use it directly
            first_assessment = assessment_plan
        else:
            raise ValidationError(f"Assessment plan must be a dictionary or list, got {type(assessment_plan)}")
        
        # Handle string format
        if isinstance(first_assessment, str):
            try:
                import ast
                first_assessment = ast.literal_eval(first_assessment)
            except (ValueError, SyntaxError) as e:
                raise ValidationError(f"Invalid assessment_plan string format: {str(e)}")
        
        if not isinstance(first_assessment, dict):
            raise ValidationError(f"Assessment plan item must be a dictionary, got {type(first_assessment)}")
        
        # Extract assessment details
        assessment_details = first_assessment.get("assessment", first_assessment)
        
        # Extract topic
        topic = assessment_details.get("topic", "")
        if not topic:
            topic = "Unknown Topic"
        
        # Handle num_questions format
        num_questions = assessment_details.get("num_questions", 5)
        if isinstance(num_questions, dict):
            total_questions = sum(num_questions.values()) if num_questions else 5
        else:
            total_questions = int(num_questions) if num_questions else 5
        
        return {
            "topic": topic,
            "type": assessment_details.get("type", "multi"),
            "difficulty": assessment_details.get("difficulty", "medium"),
            "num_questions": num_questions,  # Keep original structure
            "total_questions": total_questions,  # Add total count
            "timer_per_question": assessment_details.get("timer_per_question", 60),
            "subtopics": assessment_details.get("subtopics", []),
            "skills": assessment_details.get("skills", []),
            "assessment_time_minutes": assessment_details.get("assessment_time_minutes", 45),
            "question_breakdown": num_questions if isinstance(num_questions, dict) else {}
        }
    elif state.get("type"):
        # Direct format
        return {
            "topic": state.get("topic", "Unknown Topic"),
            "type": state.get("type", "multi"),
            "difficulty": state.get("difficulty", "medium"),
            "num_questions": state.get("num_questions", 5),
            "timer_per_question": state.get("timer_per_question", 60)
        }
    else:
        raise ValidationError("No valid assessment requirements found")

def _extract_user_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract user context from state"""
    user_context = {}
    
    # Extract from structured resume if available
    structured_resume = state.get("structured_resume", {})
    if structured_resume:
        user_context["user_profile"] = {
            "skills": structured_resume.get("skills", []),
            "experience": structured_resume.get("work_experience", []),
            "education": structured_resume.get("education", [])
        }
    
    # Extract from direct user context
    if "user_context" in state:
        user_context.update(state["user_context"])
    
    return user_context

def _validate_input_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean input state"""
    validated_state = state.copy()
    
    # Remove any None values
    validated_state = {k: v for k, v in validated_state.items() if v is not None}
    
    # Validate required fields
    if not validated_state.get("assessment_plan") and not validated_state.get("type"):
        raise ValidationError("No assessment plan or type provided")
    
    return validated_state
