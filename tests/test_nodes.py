from unittest.mock import MagicMock, patch
from langgraph.types import Command

from hiregraph.models import ResumeInfo
from hiregraph.nodes import (parse_resume, compensate, external_research_service,)


def test_parse_resume_with_mocked_llm():

    fake_result = ResumeInfo(
        candidate_name="Test Candidate",
        skills=[
            "Python",
            "Docker",
            "AWS"
        ],
        education="BS Computer Science"
    )

    mock_parser = MagicMock()
    mock_parser.invoke.return_value = fake_result

    with patch(
        "hiregraph.nodes.resume_parser",
        mock_parser
    ):

        state = {
            "resume": """
            Test Candidate

            Software Engineer

            Skills:
            Python, Docker, AWS

            Education:
            BS Computer Science
            """
        }

        result = parse_resume(state)

    assert result["candidate_name"] == "Test Candidate"

    assert result["skills"] == [
        "Python",
        "Docker",
        "AWS"
    ]

    assert result["education"] == "BS Computer Science"

    assert len(result["audit_trail"]) == 1

    assert (
        result["audit_trail"][0]["node"]
        == "parse_resume"
    )

def test_compensation_node():

    state = {
        "failed_step": "research",
        "error": "External service unavailable"
    }

    result = compensate(state)

    assert result["terminal_status"] == "compensated"

    assert len(result["audit_trail"]) == 1

    assert (
        result["audit_trail"][0]["node"]
        == "compensate"
    )

    assert (
        result["audit_trail"][0]["status"]
        == "completed"
    )

def test_external_failure_reaches_compensation():

    state = {
        "candidate_name": "Test Candidate",
        "external_retry_attempts": 2,
        "audit_trail": [],
    }

    with patch(
        "hiregraph.nodes.check_external_research_service",
        side_effect=TimeoutError(
            "Research service timed out"
        ),
    ):
        command = external_research_service(
            state
        )

    assert command.goto == "compensate"

    assert (
        command.update["external_retry_attempts"]
        == 3
    )

    assert (
        command.update["failed_step"]
        == "external_research_service"
    )

    assert (
        "timed out"
        in command.update["error"]
    )