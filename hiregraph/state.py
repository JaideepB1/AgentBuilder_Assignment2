from typing import TypedDict, Annotated, Literal
import operator

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class HireState(TypedDict, total=False):
    resume: str
    job_description: str
    working_job_description: str
    candidate_name: str
    
    skills: list[str]
    years_experience: float
    education: str

    required_skills: list[str]

    seniority: Literal[
        "junior",
        "mid",
        "senior",
        "executive"
    ]

    scores: Annotated[list[dict], operator.add]
    skill_results: Annotated[list[dict], operator.add]

    total_score: float
    scorecard: dict

    recommendation: Literal[
        "advance",
        "reject",
        "borderline"
    ]

    messages: Annotated[list[BaseMessage], add_messages]
    research_summary: str

    email: str
    email_feedback: str
    email_rating: int
    iterations: int

    error: str | None
    failed_step: str | None
    recovery_attempts: int
    external_retry_attempts: int
    terminal_status: str | None

    audit_trail: Annotated[list[dict], operator.add]

class SkillWorkerState(TypedDict):
    skill: str
    resume: str

    skill_results: Annotated[
        list[dict],
        operator.add
    ]