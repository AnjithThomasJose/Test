"""
Issue 7.1: Unit tests for validate_resume agent.

Tests the core resume validation logic including:
- Valid resume detection via Groq LLM
- Job description rejection
- Empty/short input handling
- LLM error handling
- RegexValidator section detection
- validate_resume_agent state handling
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio


class TestValidateResumeWithGroq:
    """Tests for validate_resume_with_groq function."""
    
    @pytest.mark.asyncio
    @patch('agents.agents.validate_resume._create_groq_model')
    async def test_valid_resume_detection(self, mock_create_model):
        """Valid resume should pass validation with high confidence."""
        from agents.agents.validate_resume import validate_resume_with_groq
        
        mock_response = MagicMock()
        mock_response.content = '''{
            "is_valid_resume": true,
            "confidence": 0.95,
            "reasons": ["Contains work experience", "Has education section", "Skills listed"],
            "detected_sections": ["experience", "education", "skills", "contact"]
        }'''
        
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        mock_create_model.return_value = mock_model
        
        resume_text = """
        John Doe
        Software Engineer
        john.doe@email.com | (555) 123-4567
        
        Experience:
        Senior Developer at Google (2020-2023)
        - Led team of 5 engineers
        - Built scalable microservices
        
        Education:
        BS Computer Science, MIT, 2018
        
        Skills: Python, Java, AWS, Docker
        """
        
        result = await validate_resume_with_groq(resume_text)
        
        assert result["is_valid"] is True
        assert result["confidence"] >= 0.9
        assert "experience" in result["detected_sections"]
        assert result["method"] == "groq_llm"
    
    @pytest.mark.asyncio
    @patch('agents.agents.validate_resume._create_groq_model')
    async def test_job_description_rejection(self, mock_create_model):
        """Job descriptions should be rejected."""
        from agents.agents.validate_resume import validate_resume_with_groq
        
        mock_response = MagicMock()
        mock_response.content = '''{
            "is_valid_resume": false,
            "confidence": 0.9,
            "reasons": ["Written from hiring perspective", "Contains application instructions"],
            "detected_sections": []
        }'''
        
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        mock_create_model.return_value = mock_model
        
        jd_text = """
        We are seeking a Senior Software Engineer to join our team!
        
        Responsibilities:
        - You will design and implement scalable systems
        - Lead a team of engineers
        
        Requirements:
        - 5+ years of experience
        - BS in Computer Science
        
        Benefits:
        - Competitive salary
        - Health insurance
        
        Apply now at careers@company.com
        """
        
        result = await validate_resume_with_groq(jd_text)
        
        assert result["is_valid"] is False
        assert result["method"] == "groq_llm"
    
    @pytest.mark.asyncio
    async def test_empty_input_rejection(self):
        """Empty or too short input should be rejected immediately."""
        from agents.agents.validate_resume import validate_resume_with_groq
        
        # Empty string
        result = await validate_resume_with_groq("")
        assert result["is_valid"] is False
        assert result["confidence"] == 1.0
        assert "empty" in result["reasons"][0].lower() or "short" in result["reasons"][0].lower()
        
        # Too short
        result = await validate_resume_with_groq("abc")
        assert result["is_valid"] is False
    
    @pytest.mark.asyncio
    async def test_dict_input_handling(self):
        """Dict input should extract text from known keys."""
        from agents.agents.validate_resume import validate_resume_with_groq
        
        # Test with dict input - should extract and validate
        # Even if LLM isn't called, empty/short check happens first
        result = await validate_resume_with_groq({"text": ""})
        assert result["is_valid"] is False
        
        result = await validate_resume_with_groq({"content": "x"})
        assert result["is_valid"] is False
    
    @pytest.mark.asyncio
    @patch('agents.agents.validate_resume._create_groq_model')
    async def test_llm_error_handling(self, mock_create_model):
        """LLM errors should be caught and return error response."""
        from agents.agents.validate_resume import validate_resume_with_groq
        
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(side_effect=Exception("API rate limit exceeded"))
        mock_create_model.return_value = mock_model
        
        resume_text = "John Doe, Software Engineer with 10 years experience at Google and Microsoft."
        
        result = await validate_resume_with_groq(resume_text)
        
        assert result["is_valid"] is False
        assert result["method"] == "groq_llm_error"
        assert "failed" in result["reasons"][0].lower() or "error" in result["reasons"][0].lower()
    
    @pytest.mark.asyncio
    @patch('agents.agents.validate_resume._create_groq_model')
    async def test_malformed_json_response(self, mock_create_model):
        """Malformed JSON from LLM should be handled gracefully."""
        from agents.agents.validate_resume import validate_resume_with_groq
        
        mock_response = MagicMock()
        mock_response.content = "This is not valid JSON at all"
        
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        mock_create_model.return_value = mock_model
        
        resume_text = "John Doe, Software Engineer with extensive experience in Python and AWS."
        
        result = await validate_resume_with_groq(resume_text)
        
        # Should handle gracefully with error method
        assert result["method"] == "groq_llm_error"
        assert result["is_valid"] is False


class TestRegexValidator:
    """Tests for RegexValidator section detection."""
    
    def test_section_detection(self):
        """RegexValidator should detect common resume sections."""
        from agents.agents.validate_resume import RegexValidator
        
        validator = RegexValidator()
        
        resume_text = """
        John Doe
        john.doe@email.com | 555-123-4567
        
        Education:
        Bachelor of Science in Computer Science, MIT, 2018
        GPA: 3.8
        
        Experience:
        Software Engineer at Google (2018-2023)
        - Developed microservices using Python and Go
        - Led team of 3 engineers
        
        Skills:
        Python, Java, AWS, Docker, Kubernetes
        
        Certifications:
        AWS Certified Solutions Architect
        """
        
        is_valid, confidence, reasons = validator.validate(resume_text)
        
        assert is_valid is True
        assert confidence > 0.5
        assert any("passed" in r.lower() or "sections" in r.lower() for r in reasons)
    
    def test_jd_pattern_detection(self):
        """RegexValidator should reject job descriptions."""
        from agents.agents.validate_resume import RegexValidator
        
        validator = RegexValidator()
        
        jd_text = """
        We are seeking a talented Software Engineer to join our team!
        
        About the Role:
        You will be responsible for designing and implementing scalable systems.
        
        Requirements:
        - 5+ years of experience in Python
        - Strong communication skills
        
        What We Offer:
        - Competitive salary and benefits
        - Remote work options
        
        Apply now at careers@company.com
        """
        
        is_valid, confidence, reasons = validator.validate(jd_text)
        
        # Should be rejected as JD
        assert is_valid is False or confidence < 0.5
    
    def test_too_short_text(self):
        """Very short text should be rejected."""
        from agents.agents.validate_resume import RegexValidator
        
        validator = RegexValidator()
        
        is_valid, confidence, reasons = validator.validate("John Doe")
        
        assert is_valid is False
        assert confidence < 0.5


class TestValidateResumeAgent:
    """Tests for validate_resume_agent state handler."""
    
    @pytest.mark.asyncio
    @patch('agents.agents.validate_resume.validate_resume_with_groq')
    async def test_agent_returns_expected_state(self, mock_validate):
        """Agent should return properly structured state dict."""
        from agents.agents.validate_resume import validate_resume_agent
        
        mock_validate.return_value = {
            "is_valid": True,
            "confidence": 0.92,
            "reasons": ["Valid resume structure"],
            "detected_sections": ["education", "experience"],
            "method": "groq_llm",
            "processing_time_ms": 150.0
        }
        
        state = {
            "resume_text": "John Doe, Software Engineer...",
            "user_id": "user123"
        }
        
        result = await validate_resume_agent(state)
        
        assert result["status"] == "completed"
        assert result["node"] == "is_valid_resume"
        assert result["output"] is True
        assert result["is_resume"] is True
        assert result["is_valid_resume"] is True
        assert "validation_metadata" in result
        assert result["validation_metadata"]["method"] == "groq_llm"
    
    @pytest.mark.asyncio
    async def test_agent_missing_resume_text(self):
        """Agent should handle missing resume_text gracefully."""
        from agents.agents.validate_resume import validate_resume_agent
        
        state = {"user_id": "user123"}  # No resume_text
        
        result = await validate_resume_agent(state)
        
        assert result["status"] == "error"
        assert "Missing" in result.get("validation_error", "")


class TestIsValidResumeFunction:
    """Tests for is_valid_resume boolean helper."""
    
    def test_returns_true_for_valid_state(self):
        """Should return True when is_valid_resume is True."""
        from agents.agents.validate_resume import is_valid_resume
        
        state = {"is_valid_resume": True}
        assert is_valid_resume(state) is True
    
    def test_returns_false_for_invalid_state(self):
        """Should return False when is_valid_resume is False."""
        from agents.agents.validate_resume import is_valid_resume
        
        state = {"is_valid_resume": False}
        assert is_valid_resume(state) is False
    
    def test_returns_false_for_non_dict(self):
        """Should return False for non-dict input."""
        from agents.agents.validate_resume import is_valid_resume
        
        assert is_valid_resume(None) is False
        assert is_valid_resume("not a dict") is False
        assert is_valid_resume([]) is False
    
    def test_confidence_threshold(self):
        """Should check confidence threshold when using is_resume format."""
        from agents.agents.validate_resume import is_valid_resume, MIN_CONFIDENCE_THRESHOLD
        
        # Above threshold
        state = {"is_resume": True, "resume_confidence": 0.9}
        assert is_valid_resume(state) is True
        
        # Below threshold
        state = {"is_resume": True, "resume_confidence": 0.3}
        assert is_valid_resume(state) is False
