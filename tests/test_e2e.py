from pathlib import Path
from uuid import uuid4

from hiregraph.graph import graph


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "sample_data"


def load_text(filename: str) -> str:

    return (
        DATA_DIR / filename
    ).read_text(
        encoding="utf-8"
    )


def test_weak_candidate_end_to_end():

    resume = load_text(
        "weak_resume.txt"
    )

    job_description = load_text(
        "backend_jd.txt"
    )

    config = {
        "configurable": {
            "thread_id":
                f"test-weak-{uuid4()}"
        }
    }

    initial_state = {
        "resume": resume,
        "job_description": job_description,
        "scores": [],
        "skill_results": [],
        "audit_trail": [],
        "messages": [],
        "iterations": 0,
        "recovery_attempts": 0,
    }

    result = graph.invoke(
        initial_state,
        config=config
    )

    assert (
        result["terminal_status"]
        == "completed"
    )

    assert (
        result["candidate_name"]
        == "Alex Johnson"
    )

    assert (
        result["recommendation"]
        == "reject"
    )

    assert (
        result["total_score"]
        < 60
    )

    assert (
        len(result["audit_trail"])
        > 0
    )