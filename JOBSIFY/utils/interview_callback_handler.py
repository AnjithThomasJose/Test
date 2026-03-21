"""
Interview Callback Handler for writing interview events to JSON files
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, List
import logging

log = logging.getLogger(__name__)

class InterviewCallbackHandler:
    """Handles writing interview callback events to JSON files"""
    
    def __init__(self, callback_dir: str = "local_callbacks"):
        self.callback_dir = callback_dir
        self.ensure_callback_dir()
    
    def ensure_callback_dir(self):
        """Ensure the callback directory exists"""
        if not os.path.exists(self.callback_dir):
            os.makedirs(self.callback_dir)
    
    def _generate_filename(self, uid: str, event_type: str, session_id: str = None) -> str:
        """Generate filename for callback event"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Remove last 3 digits of microseconds
        session_part = f"_{session_id}" if session_id else ""
        return f"interview_{event_type}_{uid}{session_part}_{timestamp}.json"
    
    def _write_callback_file(self, filename: str, data: Dict[str, Any]):
        """Write callback data to JSON file"""
        try:
            filepath = os.path.join(self.callback_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            log.info(f"Callback written to: {filepath}")
            
            # Log evaluation results to console for real-time visibility
            self._log_evaluation_result(data)
        except Exception as e:
            log.error(f"Failed to write callback file {filename}: {e}")
    
    def _log_evaluation_result(self, data: Dict[str, Any]):
        """Log evaluation results to console for real-time visibility"""
        event_type = data.get("event_type", "")
        
        if event_type == "skill_detected":
            skill = data.get("skill", "")
            score = data.get("score", 0)
            log.info(f"🎯 SKILL DETECTED: {skill} - Score: {score}/10")
            
        elif event_type == "score_updated":
            criteria = data.get("criteria", "")
            score = data.get("score", 0)
            evidence = data.get("evidence", "")
            log.info(f"📊 SCORE UPDATE: {criteria} = {score}/10 ({evidence})")
            
        elif event_type == "green_flag_triggered":
            flag = data.get("flag", "")
            log.info(f"✅ GREEN FLAG: {flag}")
            
        elif event_type == "red_flag_triggered":
            flag = data.get("flag", "")
            log.warning(f"❌ RED FLAG: {flag}")
            
        elif event_type == "evaluation_completed":
            overall_score = data.get("overall_score", 0)
            status = data.get("status", "")
            log.info(f"🏁 FINAL EVALUATION: Overall Score = {overall_score}/10 | Status = {status}")
            
        elif event_type == "interview_started":
            log.info("🚀 INTERVIEW STARTED")
            
        elif event_type == "interview_ended":
            questions_count = data.get("questions_count", 0)
            responses_count = data.get("responses_count", 0)
            log.info(f"🏁 INTERVIEW ENDED: {questions_count} questions, {responses_count} responses")
            
        elif event_type == "question_asked":
            question = data.get("question", "")
            log.debug(f"❓ QUESTION: {question[:100]}{'...' if len(question) > 100 else ''}")
            
        elif event_type == "response_received":
            response = data.get("response", "")
            log.debug(f"💬 RESPONSE: {response[:100]}{'...' if len(response) > 100 else ''}")
    
    def on_interview_started(self, conversation_length: int):
        """Callback for interview started event"""
        data = {
            "event_type": "interview_started",
            "timestamp": datetime.now().isoformat(),
            "conversation_length": conversation_length
        }
        filename = self._generate_filename("system", "started")
        self._write_callback_file(filename, data)
    
    def on_question_asked(self, question: str):
        """Callback for question asked event"""
        data = {
            "event_type": "question_asked",
            "timestamp": datetime.now().isoformat(),
            "question": question
        }
        filename = self._generate_filename("system", "question")
        self._write_callback_file(filename, data)
    
    def on_response_received(self, response: str, question: str):
        """Callback for response received event"""
        data = {
            "event_type": "response_received",
            "timestamp": datetime.now().isoformat(),
            "response": response,
            "question": question
        }
        filename = self._generate_filename("system", "response")
        self._write_callback_file(filename, data)
    
    def on_skill_detected(self, skill: str, score: float, context: str, old_score: float = None):
        """Callback for skill detected event"""
        data = {
            "event_type": "skill_detected",
            "timestamp": datetime.now().isoformat(),
            "skill": skill,
            "score": score,
            "context": context
        }
        if old_score is not None:
            data["old_score"] = old_score
        filename = self._generate_filename("system", "skill")
        self._write_callback_file(filename, data)
    
    def on_score_updated(self, criteria: str, score: float, evidence: str):
        """Callback for score updated event"""
        data = {
            "event_type": "score_updated",
            "timestamp": datetime.now().isoformat(),
            "criteria": criteria,
            "score": score,
            "evidence": evidence
        }
        filename = self._generate_filename("system", "score")
        self._write_callback_file(filename, data)
    
    def on_red_flag_triggered(self, flag: str, context: str):
        """Callback for red flag triggered event"""
        data = {
            "event_type": "red_flag_triggered",
            "timestamp": datetime.now().isoformat(),
            "flag": flag,
            "context": context
        }
        filename = self._generate_filename("system", "red_flag")
        self._write_callback_file(filename, data)
    
    def on_green_flag_triggered(self, flag: str, context: str):
        """Callback for green flag triggered event"""
        data = {
            "event_type": "green_flag_triggered",
            "timestamp": datetime.now().isoformat(),
            "flag": flag,
            "context": context
        }
        filename = self._generate_filename("system", "green_flag")
        self._write_callback_file(filename, data)
    
    def on_interview_ended(self, questions_count: int, responses_count: int):
        """Callback for interview ended event"""
        data = {
            "event_type": "interview_ended",
            "timestamp": datetime.now().isoformat(),
            "questions_count": questions_count,
            "responses_count": responses_count
        }
        filename = self._generate_filename("system", "ended")
        self._write_callback_file(filename, data)
    
    def on_evaluation_completed(self, overall_score: float, status: str):
        """Callback for evaluation completed event"""
        data = {
            "event_type": "evaluation_completed",
            "timestamp": datetime.now().isoformat(),
            "overall_score": overall_score,
            "status": status
        }
        filename = self._generate_filename("system", "evaluation")
        self._write_callback_file(filename, data)
