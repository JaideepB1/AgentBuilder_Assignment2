from typing import Literal

from pydantic import BaseModel, Field


class ResumeInfo(BaseModel):
    candidate_name: str
    skills: list[str]
    education: str

class NormalizedSkills(BaseModel):
    skills: list[str]

class ExperienceInfo(BaseModel):
    years_experience: float = Field(ge=0)


class JobInfo(BaseModel):
    required_skills: list[str]


class SeniorityResult(BaseModel):
    seniority: Literal[
        "junior",
        "mid",
        "senior",
        "executive"
    ]
    reasoning: str


class ScoreResult(BaseModel):
    score: int = Field(
        ge=0,
        le=100
    )
    reasoning: str


class SkillResult(BaseModel):
    score: int = Field(
        ge=0,
        le=100
    )
    evidence: str


class EmailDraft(BaseModel):
    subject: str
    body: str

class EmailReview(BaseModel):
    rating: int = Field(
        ge=1,
        le=10
    )
    feedback: str

class RecoveryResult(BaseModel):
    corrected_text: str
    explanation: str