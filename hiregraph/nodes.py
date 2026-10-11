import time
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.types import Command, Send
from langchain_core.messages import HumanMessage

from .state import (HireState, SkillWorkerState,)
from .models import (
    ResumeInfo,
    NormalizedSkills,
    ExperienceInfo,
    JobInfo,
    SeniorityResult,
    ScoreResult,
    SkillResult,
    EmailDraft,
    EmailReview,
    RecoveryResult,
)

from .tools import TOOLS

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

resume_parser = llm.with_structured_output(
    ResumeInfo
)

skill_normalizer = llm.with_structured_output(
    NormalizedSkills
)

experience_parser = llm.with_structured_output(
    ExperienceInfo
)

job_parser = llm.with_structured_output(
    JobInfo
)

seniority_classifier = llm.with_structured_output(
    SeniorityResult
)

score_model = llm.with_structured_output(
    ScoreResult
)

skill_evaluator = llm.with_structured_output(
    SkillResult
)

research_llm = llm.bind_tools(TOOLS)

email_generator = llm.with_structured_output(
    EmailDraft
)

email_critic = llm.with_structured_output(
    EmailReview
)

recovery_model = llm.with_structured_output(
    RecoveryResult
)

public_signal_scorer = llm.with_structured_output(
    ScoreResult
)


def make_audit_entry(
    node: str,
    status: str,
    detail: str = "",
    duration_ms: float | None = None
):
    return {
        "node": node,
        "status": status,
        "detail": detail,
        "timestamp": time.time(),
        "duration_ms": duration_ms
    }



def parse_resume(state: HireState):
    start = time.perf_counter()

    result = resume_parser.invoke(
        f"""
        Extract the candidate's name, skills,
        and education from this resume.

        Resume:
        {state["resume"]}
        """
    )

    duration_ms = (
        time.perf_counter() - start
    ) * 1000

    return {
        "candidate_name": result.candidate_name,
        "skills": result.skills,
        "education": result.education,
        "audit_trail": [
            make_audit_entry(
                "parse_resume",
                "success",
                "Parsed candidate resume",
                round(duration_ms, 2)
            )
        ]
    }


def normalize_skills(state: HireState):
    result = skill_normalizer.invoke(
        f"""
        Normalize the following candidate skills
        into standard industry names.

        Examples:

        py -> Python
        postgres -> PostgreSQL
        js -> JavaScript
        aws cloud -> AWS

        Do not invent skills that are not present.

        Candidate skills:
        {state["skills"]}
        """
    )

    return {
        "skills": result.skills,

        "audit_trail": [
            make_audit_entry(
                "normalize_skills",
                "success",
                "Normalized candidate skills"
            )
        ]
    }


def extract_experience(state: HireState):
    result = experience_parser.invoke(
        f"""
        Determine the candidate's total relevant
        professional experience in years.

        Do not count overlapping jobs twice.

        Resume:
        {state["resume"]}
        """
    )

    return {
        "years_experience":
            result.years_experience,

        "audit_trail": [
            make_audit_entry(
                "extract_experience",
                "success",
                (
                    "Extracted "
                    f"{result.years_experience} "
                    "years of experience"
                )
            )
        ]
    }


def parse_jd(
    state: HireState
) -> Command[
    Literal[
        "classify_seniority",
        "repair_jd",
        "compensate"
    ]
]:
    try:
        result = job_parser.invoke(
            f"""
            Extract the technical and professional
            requirements from this job description.

            Return them as required_skills.

            Job Description:
            {state.get("working_job_description", state["job_description"])}
            """
        )

        if not result.required_skills:
            raise ValueError(
                "No job requirements were extracted."
            )

        return Command(
            update={
                "required_skills":
                    result.required_skills,

                "error": None,
                "failed_step": None,

                "audit_trail": [
                    make_audit_entry(
                        "parse_jd",
                        "success",
                        (
                            "Extracted "
                            f"{len(result.required_skills)} "
                            "job requirements"
                        )
                    )
                ]
            },

            goto="classify_seniority"
        )

    except (
        ValueError,
        TypeError,
        KeyError
    ) as exc:

        attempts = state.get(
            "recovery_attempts",
            0
        )

        if attempts >= 2:
            return Command(
                update={
                    "error": str(exc),
                    "failed_step": "parse_jd",

                    "audit_trail": [
                        make_audit_entry(
                            "parse_jd",
                            "failed",
                            (
                                "Parser recovery "
                                "limit reached"
                            )
                        )
                    ]
                },

                goto="compensate"
            )

        return Command(
            update={
                "error": str(exc),
                "failed_step": "parse_jd",

                "audit_trail": [
                    make_audit_entry(
                        "parse_jd",
                        "recoverable_error",
                        str(exc)
                    )
                ]
            },

            goto="repair_jd"
        )

def classify_seniority(
    state: HireState
) -> Command[
    Literal[
        "junior_path",
        "mid_path",
        "senior_path",
        "executive_path"
    ]
]:
    result = seniority_classifier.invoke(
        f"""
        Classify this candidate into exactly one
        seniority level:

        - junior
        - mid
        - senior
        - executive

        Consider:
        - years of relevant experience
        - technical skills
        - education
        - responsibilities shown in the resume

        Candidate name:
        {state["candidate_name"]}

        Years of experience:
        {state["years_experience"]}

        Skills:
        {state["skills"]}

        Education:
        {state["education"]}

        Resume:
        {state["resume"]}
        """
    )

    routes = {
        "junior": "junior_path",
        "mid": "mid_path",
        "senior": "senior_path",
        "executive": "executive_path"
    }

    return Command(
        update={
            "seniority": result.seniority,

            "audit_trail": [
                make_audit_entry(
                    "classify_seniority",
                    "success",
                    (
                        f"Classified candidate as "
                        f"{result.seniority}. "
                        f"{result.reasoning}"
                    )
                )
            ]
        },

        goto=routes[result.seniority]
    )

def junior_path(state: HireState):
    return {
        "audit_trail": [
            make_audit_entry(
                "junior_path",
                "success",
                "Using junior candidate scoring path"
            )
        ]
    }


def mid_path(state: HireState):
    return {
        "audit_trail": [
            make_audit_entry(
                "mid_path",
                "success",
                "Using mid-level candidate scoring path"
            )
        ]
    }


def senior_path(state: HireState):
    return {
        "audit_trail": [
            make_audit_entry(
                "senior_path",
                "success",
                "Using senior candidate scoring path"
            )
        ]
    }


def executive_path(state: HireState):
    return {
        "audit_trail": [
            make_audit_entry(
                "executive_path",
                "success",
                "Using executive candidate scoring path"
            )
        ]
    }

def score_skills(state: HireState):
    result = score_model.invoke(
        f"""
        Evaluate how well the candidate's skills
        match the skills required by the job.

        Candidate skills:
        {state["skills"]}

        Required job skills:
        {state["required_skills"]}

        Give a score from 0 to 100.

        Consider:
        - direct matches
        - closely related skills
        - important missing required skills

        A higher score means the candidate's skills
        match the job requirements more closely.
        """
    )

    return {
        "scores": [
            {
                "dimension": "skills",
                "score": result.score,
                "reasoning": result.reasoning
            }
        ],

        "audit_trail": [
            make_audit_entry(
                "score_skills",
                "success",
                f"Skills score: {result.score}"
            )
        ]
    }

def score_experience(state: HireState):
    result = score_model.invoke(
        f"""
        Evaluate the candidate's professional
        experience for this job.

        Candidate seniority:
        {state["seniority"]}

        Candidate years of experience:
        {state["years_experience"]}

        Resume:
        {state["resume"]}

        Job description:
        {state["job_description"]}

        Give a score from 0 to 100.

        Consider:
        - amount of relevant experience
        - relevance of previous responsibilities
        - leadership or ownership when appropriate
        - whether the experience matches the
          seniority expected by the job

        A higher score means stronger experience
        alignment with the job.
        """
    )

    return {
        "scores": [
            {
                "dimension": "experience",
                "score": result.score,
                "reasoning": result.reasoning
            }
        ],

        "audit_trail": [
            make_audit_entry(
                "score_experience",
                "success",
                f"Experience score: {result.score}"
            )
        ]
    }

def score_education(state: HireState):
    result = score_model.invoke(
        f"""
        Evaluate the candidate's education
        relative to this job.

        Candidate education:
        {state["education"]}

        Job description:
        {state["job_description"]}

        Give a score from 0 to 100.

        Consider:
        - relevant degree or training
        - whether the job explicitly requires
          a degree
        - relevant alternative qualifications

        Do not penalize the candidate heavily
        for education when the job description
        does not emphasize education.

        A higher score means stronger education
        alignment with the job.
        """
    )

    return {
        "scores": [
            {
                "dimension": "education",
                "score": result.score,
                "reasoning": result.reasoning
            }
        ],

        "audit_trail": [
            make_audit_entry(
                "score_education",
                "success",
                f"Education score: {result.score}"
            )
        ]
    }

def aggregate_scores(state: HireState):
    scores = state["scores"]

    scorecard = {}

    for result in scores:
        scorecard[result["dimension"]] = {
            "score": result["score"],
            "reasoning": result["reasoning"]
        }

    total_score = sum(
        result["score"]
        for result in scores
    ) / len(scores)

    return {
        "scorecard": scorecard,
        "total_score": round(total_score, 2),

        "audit_trail": [
            make_audit_entry(
                "aggregate_scores",
                "success",
                f"Overall score: {round(total_score, 2)}"
            )
        ]
    }

def assign_skill_workers(state: HireState):
    return [
        Send(
            "skill_worker",
            {
                "skill": skill,
                "resume": state["resume"],
                "skill_results": []
            }
        )
        for skill in state["required_skills"]
    ]

def skill_worker(
    state: SkillWorkerState
):
    result = skill_evaluator.invoke(
        f"""
        Evaluate the candidate's evidence for
        the following required skill.

        Required skill:
        {state["skill"]}

        Candidate resume:
        {state["resume"]}

        Give a score from 0 to 100.

        Scoring guidance:

        90-100:
        Strong direct evidence of professional
        experience using the skill.

        70-89:
        Good evidence of using the skill.

        40-69:
        Partial, indirect, or limited evidence.

        1-39:
        Very weak evidence.

        0:
        No evidence of the skill in the resume.

        Do not invent experience.

        In the evidence field, briefly explain
        what evidence from the resume supports
        the score.
        """
    )

    return {
        "skill_results": [
            {
                "skill": state["skill"],
                "score": result.score,
                "evidence": result.evidence
            }
        ]
    }

def collect_skill_results(
    state: HireState
):
    results = state.get(
        "skill_results",
        []
    )

    if not results:
        average_skill_score = 0
    else:
        average_skill_score = sum(
            result["score"]
            for result in results
        ) / len(results)

    return {
        "audit_trail": [
            make_audit_entry(
                "collect_skill_results",
                "success",
                (
                    f"Evaluated {len(results)} "
                    f"required skills. "
                    f"Average worker score: "
                    f"{average_skill_score:.2f}"
                )
            )
        ]
    }

def prepare_research(
    state: HireState
) -> Command[
    Literal[
        "research_agent",
        "compensate"
    ]
]:
    try:
        worker_summary = "\n".join(
            (
                f"- {item['skill']}: "
                f"{item['score']}/100. "
                f"{item['evidence']}"
            )
            for item in state.get(
                "skill_results",
                []
            )
        )

        message = HumanMessage(
            content=f"""
            You are the public-signal research
            agent for a hiring evaluation.

            Candidate:
            {state["candidate_name"]}

            Resume:
            {state["resume"]}

            Job description:
            {state["job_description"]}

            Skill worker results:
            {worker_summary}

            You must call at least one available
            research tool before producing your
            final research summary.

            Do not invent public information.
            """
        )

        return Command(
            update={
                "messages": [message],

                "audit_trail": [
                    make_audit_entry(
                        "prepare_research",
                        "success",
                        "Prepared research request"
                    )
                ]
            },

            goto="research_agent"
        )

    except (
        KeyError,
        TypeError,
        ValueError
    ) as exc:

        return Command(
            update={
                "error": str(exc),
                "failed_step":
                    "prepare_research",

                "audit_trail": [
                    make_audit_entry(
                        "prepare_research",
                        "failed",
                        str(exc)
                    )
                ]
            },

            goto="compensate"
        )

def research_agent(
    state: HireState
):
    response = research_llm.invoke(
        state["messages"]
    )

    return {
        "messages": [response],

        "audit_trail": [
            make_audit_entry(
                "research_agent",
                "success",
                (
                    "Research agent responded "
                    "or requested a tool"
                )
            )
        ]
    }

def route_research(
    state: HireState
) -> Literal[
    "research_tools",
    "finish_research"
]:
    last_message = state["messages"][-1]

    if getattr(
        last_message,
        "tool_calls",
        None
    ):
        return "research_tools"

    return "finish_research"

def finish_research(
    state: HireState
):
    last_message = state["messages"][-1]

    content = last_message.content

    if isinstance(content, str):
        summary = content
    else:
        summary = str(content)

    return {
        "research_summary": summary,

        "audit_trail": [
            make_audit_entry(
                "finish_research",
                "success",
                "Research completed"
            )
        ]
    }

def make_decision(
    state: HireState
) -> Command[
    Literal[
        "draft_email",
        "draft_email_borderline",
        "draft_email_reject"
    ]
]:
    score = state["total_score"]

    if score >= 75:
        recommendation = "advance"
        next_node = "draft_email"

    elif score >= 60:
        recommendation = "borderline"
        next_node = "draft_email_borderline"

    else:
        recommendation = "reject"
        next_node = "draft_email_reject"

    return Command(
        update={
            "recommendation": recommendation,

            "audit_trail": [
                make_audit_entry(
                    "make_decision",
                    "success",
                    (
                        f"Recommendation: "
                        f"{recommendation}. "
                        f"Overall score: {score}"
                    )
                )
            ]
        },

        goto=next_node
    )

def draft_email(
    state: HireState
):
    return generate_candidate_email(
        state,
        "advance"
    )


def draft_email_borderline(
    state: HireState
):
    return generate_candidate_email(
        state,
        "borderline"
    )


def draft_email_reject(
    state: HireState
):
    return generate_candidate_email(
        state,
        "reject"
    )

def generate_candidate_email(
    state: HireState,
    recommendation: str
):
    result = email_generator.invoke(
        f"""
        Write a professional candidate-facing
        recruiting email.

        Candidate:
        {state["candidate_name"]}

        Hiring recommendation:
        {recommendation}

        Candidate seniority:
        {state["seniority"]}

        Job description:
        {state["job_description"]}

        Rules:

        1. Be concise and professional.

        2. Do not reveal internal numeric scores.

        3. Do not mention the AI system,
           LangGraph, internal scoring,
           research tools, or audit logs.

        4. Do not invent company information.

        5. If the recommendation is "advance",
           tell the candidate that the hiring
           team would like to move forward.

        6. If the recommendation is "borderline",
           write a neutral follow-up email
           explaining that the application
           remains under consideration.

        7. If the recommendation is "reject",
           politely explain that the team will
           not be moving forward.

        Do not include placeholder fields such as
        [Your Name], [Your Position], or [Company Name].

        End the email with:

        Best regards,
        Hiring Team
        
        Return a subject and body.
        """
    )

    email = (
        f"Subject: {result.subject}\n\n"
        f"{result.body}"
    )

    return {
        "email": email,
        "iterations": 0,

        "audit_trail": [
            make_audit_entry(
                "draft_email",
                "success",
                (
                    "Generated initial "
                    f"{recommendation} email"
                )
            )
        ]
    }

def critique_email(
    state: HireState
) -> Command[
    Literal[
        "revise_email",
        "finalize"
    ]
]:
    result = email_critic.invoke(
        f"""
        Evaluate this recruiting email.

        Hiring recommendation:
        {state["recommendation"]}

        Email:
        {state["email"]}

        Grade it from 1 to 10.

        Evaluate:

        - professionalism
        - clarity
        - tone
        - consistency with the hiring decision
        - whether it avoids exposing internal
          scores or reasoning
        - whether it avoids invented information

        A rating of 8 or higher means the email
        is acceptable.

        Provide specific feedback explaining
        any problems that should be corrected.

        Do not include placeholder fields such as
        [Your Name], [Your Position], or [Company Name].

        End the email with:

        Best regards,
        Hiring Team
        """
    )

    current_iterations = state.get(
        "iterations",
        0
    )

    new_iterations = current_iterations + 1

    if (
        result.rating >= 8
        or new_iterations >= 3
    ):
        next_node = "finalize"
    else:
        next_node = "revise_email"

    return Command(
        update={
            "email_rating": result.rating,
            "email_feedback": result.feedback,
            "iterations": new_iterations,

            "audit_trail": [
                make_audit_entry(
                    "critique_email",
                    "success",
                    (
                        f"Email rating: "
                        f"{result.rating}/10. "
                        f"Critique iteration: "
                        f"{new_iterations}"
                    )
                )
            ]
        },

        goto=next_node
    )

def revise_email(
    state: HireState
):
    result = email_generator.invoke(
        f"""
        Revise the recruiting email using
        the critic's feedback.

        Hiring recommendation:
        {state["recommendation"]}

        Current email:
        {state["email"]}

        Critic feedback:
        {state["email_feedback"]}

        Requirements:

        - preserve the hiring recommendation
        - improve professionalism and clarity
        - address the critic's feedback
        - do not expose numeric scores
        - do not expose internal reasoning
        - do not mention AI or LangGraph
        - do not invent company information

        Return the improved subject and body.
        """
    )

    email = (
        f"Subject: {result.subject}\n\n"
        f"{result.body}"
    )

    return {
        "email": email,

        "audit_trail": [
            make_audit_entry(
                "revise_email",
                "success",
                "Revised email using critic feedback"
            )
        ]
    }

def finalize(
    state: HireState
):
    return {
        "terminal_status": "completed",

        "audit_trail": [
            make_audit_entry(
                "finalize",
                "success",
                (
                    f"Final recommendation: "
                    f"{state['recommendation']}. "
                    f"Final email rating: "
                    f"{state.get('email_rating', 'N/A')}"
                )
            )
        ]
    }

def repair_jd(
    state: HireState
):
    result = recovery_model.invoke(
        f"""
        A job-description parser failed.

        Original job description:

        {state["job_description"]}

        Parser error:

        {state.get("error", "Unknown error")}

        Rewrite the job description into a
        clearer form that preserves the original
        requirements.

        Do not invent requirements.

        Your corrected_text should contain only
        information supported by the original
        job description.
        """
    )

    attempts = (
        state.get(
            "recovery_attempts",
            0
        )
        + 1
    )

    return {
        "working_job_description": result.corrected_text,
        "recovery_attempts": attempts,
        "audit_trail": [
            make_audit_entry(
                "repair_jd",
                "success",
                f"Repaired JD for parser retry {attempts}"
            )
        ]
    }

def compensate(
    state: HireState
):
    failed_step = state.get(
        "failed_step",
        "unknown"
    )

    error = state.get(
        "error",
        "Unknown error"
    )

    return {
        "terminal_status": "compensated",

        "audit_trail": [
            make_audit_entry(
                "compensate",
                "completed",
                (
                    f"Workflow safely terminated "
                    f"after failure in "
                    f"{failed_step}. "
                    f"Error: {error}"
                )
            )
        ]
    }

def start_research(
    state: HireState
) -> Command[
    Literal[
        "prepare_research",
        "compensate"
    ]
]:
    if not state.get("candidate_name"):
        return Command(
            update={
                "error":
                    "Candidate name is missing.",
                "failed_step":
                    "research",

                "audit_trail": [
                    make_audit_entry(
                        "start_research",
                        "failed",
                        "Missing candidate name"
                    )
                ]
            },
            goto="compensate"
        )

    return Command(
        update={
            "error": None,
            "failed_step": None,

            "audit_trail": [
                make_audit_entry(
                    "start_research",
                    "success",
                    "Research stage started"
                )
            ]
        },
        goto="prepare_research"
    )

def score_public_signals(state: HireState):
    start = time.perf_counter()

    result = public_signal_scorer.invoke(
        f"""
        Score the candidate's public professional
        signals from 0 to 100.

        Use ONLY the research summary below.

        Research summary:
        {state.get("research_summary", "")}

        Rules:
        - Do not invent public information.
        - If no verified public information is available,
          use a neutral score of 50.
        - Strong verified professional evidence may
          increase the score.
        - Verified negative or clearly weak evidence
          may decrease the score.
        - Lack of a public profile alone should not
          punish the candidate.

        Return a score and concise reasoning.
        """
    )

    duration_ms = (
        time.perf_counter() - start
    ) * 1000

    return {
        "audit_trail": [
            make_audit_entry(
                "score_public_signals",
                "success",
                f"Public signal score: {result.score}",
                round(duration_ms, 2)
            )
        ],
        "scorecard": {
            **state.get("scorecard", {}),
            "public_signals": {
                "score": result.score,
                "reasoning": result.reasoning
            }
        }
    }

def calculate_final_score(state: HireState):
    scorecard = state.get("scorecard", {})

    dimension_scores = []

    for dimension in [
        "skills",
        "experience",
        "education",
        "public_signals",
    ]:
        if dimension in scorecard:
            dimension_scores.append(
                scorecard[dimension]["score"]
            )

    if not dimension_scores:
        final_score = 0.0
    else:
        final_score = (
            sum(dimension_scores)
            / len(dimension_scores)
        )

    return {
        "total_score": round(final_score, 2),
        "audit_trail": [
            make_audit_entry(
                "calculate_final_score",
                "success",
                (
                    "Calculated final score using "
                    f"{len(dimension_scores)} dimensions"
                )
            )
        ]
    }

def run_external_research(state: HireState):
    """
    Represents an external research-service operation.

    Exceptions such as TimeoutError and ConnectionError
    are intentionally allowed to escape so LangGraph's
    RetryPolicy can retry this node.
    """

    candidate_name = state["candidate_name"]

    return {
        "audit_trail": [
            make_audit_entry(
                "run_external_research",
                "success",
                f"External research service available for {candidate_name}"
            )
        ]
    }

def check_external_research_service(
    candidate_name: str
):
    """
    Simulates contacting an external research service.

    This function is separated so tests can mock
    external service failures deterministically.
    """
    return (
        f"External research service available "
        f"for {candidate_name}"
    )

def external_research_service(
    state: HireState
) -> Command[
    Literal[
        "start_research",
        "external_research_service",
        "compensate"
    ]
]:
    attempts = state.get(
        "external_retry_attempts",
        0
    )

    try:
        candidate_name = state["candidate_name"]

        check_external_research_service(
            candidate_name
        )

        return Command(
            update={
                "external_retry_attempts": attempts,
                "error": None,
                "failed_step": None,
                "audit_trail": [
                    make_audit_entry(
                        "external_research_service",
                        "success",
                        (
                            "External research service "
                            f"available for {candidate_name}"
                        )
                    )
                ]
            },
            goto="start_research"
        )

    except (TimeoutError, ConnectionError) as exc:

        new_attempts = attempts + 1

        if new_attempts >= 3:
            return Command(
                update={
                    "external_retry_attempts":
                        new_attempts,
                    "error": str(exc),
                    "failed_step":
                        "external_research_service",
                    "audit_trail": [
                        make_audit_entry(
                            "external_research_service",
                            "failed",
                            (
                                "External service failed "
                                "after retry attempts"
                            )
                        )
                    ]
                },
                goto="compensate"
            )

        return Command(
            update={
                "external_retry_attempts":
                    new_attempts,
                "error": str(exc),
                "failed_step":
                    "external_research_service",
                "audit_trail": [
                    make_audit_entry(
                        "external_research_service",
                        "retrying",
                        (
                            "External service failure; "
                            f"retry {new_attempts}"
                        )
                    )
                ]
            },
            goto="external_research_service"
        )