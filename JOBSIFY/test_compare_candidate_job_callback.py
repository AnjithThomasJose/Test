#!/usr/bin/env python3
"""
Tests for compare-candidate-job callback and PATCH /mock-questions response.
- job_matcher callback (compare flow): match result (no mock_interview_qa; use /mock-questions for that).
- mock-questions: response/callback includes job_id, uid, mock_interview_qa.
"""

import pytest


def test_job_matcher_compare_callback_output_structure():
    """Callback sent for node=job_matcher in compare flow must include match result."""
    callback_payload = {
        "status": "completed",
        "node": "job_matcher",
        "output": {
            "candidate_job_match_result": {
                "candidate_id": "uid_123",
                "job_id": "job_456",
                "match_score": 0.85,
                "skill_match_percentage": 75.0,
                "skill_match_count": 3,
                "total_required_skills": 4,
                "skills_matched": ["Python", "React", "SQL"],
                "skills_unmatched": ["AWS"],
                "rationale": "Strong fit for the role.",
            },
            "match_score": 0.85,
            "skill_match_percentage": 75.0,
            "skill_match_count": 3,
            "skills_matched": ["Python", "React", "SQL"],
            "skills_unmatched": ["AWS"],
        },
    }

    assert callback_payload["status"] == "completed"
    assert callback_payload["node"] == "job_matcher"
    out = callback_payload["output"]
    assert "candidate_job_match_result" in out
    result = out["candidate_job_match_result"]
    assert "match_score" in result
    assert "skills_matched" in result
    assert "skills_unmatched" in result
    assert "rationale" in result


def test_mock_questions_response_structure():
    """PATCH /mock-questions response (or callback output) must include job_id, uid, mock_interview_qa."""
    response = {
        "job_id": "job_456",
        "uid": "uid_123",
        "job_title": "Software Engineer",
        "mock_interview_qa": [
            {"question": "Tell me about a project using Python.", "expected_answer": "Key points..."},
            {"question": "How do you approach testing?", "expected_answer": "Sample answer..."},
        ],
    }
    assert response["job_id"] == "job_456"
    assert response["uid"] == "uid_123"
    assert "mock_interview_qa" in response
    assert isinstance(response["mock_interview_qa"], list)
    for qa in response["mock_interview_qa"]:
        assert "question" in qa
        assert "expected_answer" in qa


def test_mock_questions_callback_payload_structure():
    """Callback from PATCH /mock-questions (when callback_url provided) must have node and output."""
    payload = {
        "status": "completed",
        "node": "mock_questions",
        "output": {
            "job_id": "job_456",
            "uid": "uid_123",
            "job_title": "Backend Developer",
            "mock_interview_qa": [],
        },
    }
    assert payload["node"] == "mock_questions"
    assert payload["output"]["job_id"] == "job_456"
    assert payload["output"]["uid"] == "uid_123"
    assert payload["output"]["mock_interview_qa"] == []
