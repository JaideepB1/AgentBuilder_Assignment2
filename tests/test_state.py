from hiregraph.state import HireState


def test_hire_state_can_hold_candidate_data():

    state: HireState = {
        "resume": "Example resume",
        "job_description": "Example job description",
        "candidate_name": "John Smith",
        "skills": ["Python", "Docker"],
        "years_experience": 8.0,
        "education": "BS Computer Science",
        "required_skills": ["Python", "AWS"],
        "scores": [],
        "skill_results": [],
        "audit_trail": [],
        "messages": [],
        "iterations": 0,
        "recovery_attempts": 0,
    }

    assert state["candidate_name"] == "John Smith"
    assert state["years_experience"] == 8.0
    assert "Python" in state["skills"]
    assert state["scores"] == []