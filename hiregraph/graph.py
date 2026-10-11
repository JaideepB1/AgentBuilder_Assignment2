from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy
from langgraph.checkpoint.memory import InMemorySaver

from .state import HireState
from .nodes import (
    parse_resume,
    normalize_skills,
    extract_experience,
    parse_jd,
    classify_seniority,
    junior_path,
    mid_path,
    senior_path,
    executive_path,
    score_skills,
    score_experience,
    score_education,
    aggregate_scores,
    skill_worker,
    collect_skill_results,
    assign_skill_workers,
    prepare_research,
    research_agent,
    route_research,
    finish_research,
    make_decision,
    draft_email,
    draft_email_borderline,
    draft_email_reject,
    critique_email,
    revise_email,
    finalize,
    repair_jd,
    start_research,
    compensate,
    score_public_signals,
    calculate_final_score,
    external_research_service,
)

from .tools import TOOLS

external_retry = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,
    backoff_factor=2.0,
    retry_on=(TimeoutError, ConnectionError)
)

builder = StateGraph(HireState)

memory = InMemorySaver()

research_tool_node = ToolNode(
    TOOLS
)


builder.add_node(
    "parse_resume",
    parse_resume
)

builder.add_node(
    "normalize_skills",
    normalize_skills
)

builder.add_node(
    "extract_experience",
    extract_experience
)

builder.add_node(
    "parse_jd",
    parse_jd
)

builder.add_node(
    "classify_seniority",
    classify_seniority
)

builder.add_node(
    "junior_path",
    junior_path
)

builder.add_node(
    "mid_path",
    mid_path
)

builder.add_node(
    "senior_path",
    senior_path
)

builder.add_node(
    "executive_path",
    executive_path
)

builder.add_node(
    "score_skills",
    score_skills
)

builder.add_node(
    "score_experience",
    score_experience
)

builder.add_node(
    "score_education",
    score_education
)

builder.add_node(
    "aggregate_scores",
    aggregate_scores
)

builder.add_node(
    "skill_worker",
    skill_worker
)

builder.add_node(
    "prepare_research",
    prepare_research
)

builder.add_node(
    "research_agent",
    research_agent,
    retry_policy=external_retry
)

builder.add_node(
    "research_tools",
    research_tool_node,
    retry_policy=external_retry
)

builder.add_node(
    "finish_research",
    finish_research
)

builder.add_node(
    "make_decision",
    make_decision
)

builder.add_node(
    "draft_email",
    draft_email
)

builder.add_node(
    "draft_email_borderline",
    draft_email_borderline
)

builder.add_node(
    "draft_email_reject",
    draft_email_reject
)

builder.add_node(
    "critique_email",
    critique_email
)

builder.add_node(
    "revise_email",
    revise_email
)

builder.add_node(
    "finalize",
    finalize
)

builder.add_node(
    "repair_jd",
    repair_jd
)

builder.add_node(
    "start_research",
    start_research
)

builder.add_node(
    "compensate",
    compensate
)

builder.add_node(
    "score_public_signals",
    score_public_signals
)

builder.add_node(
    "calculate_final_score",
    calculate_final_score
)

builder.add_node(
    "collect_skill_results",
    collect_skill_results
)

builder.add_node(
    "external_research_service",
    external_research_service
)

builder.add_edge(
    START,
    "parse_resume"
)

builder.add_edge(
    "parse_resume",
    "normalize_skills"
)

builder.add_edge(
    "normalize_skills",
    "extract_experience"
)

builder.add_edge(
    "extract_experience",
    "parse_jd"
)

builder.add_edge(
    "repair_jd",
    "parse_jd"
)

for path in [
    "junior_path",
    "mid_path",
    "senior_path",
    "executive_path"
]:
    builder.add_edge(
        path,
        "score_skills"
    )

    builder.add_edge(
        path,
        "score_experience"
    )

    builder.add_edge(
        path,
        "score_education"
    )

builder.add_edge(
    "score_skills",
    "aggregate_scores"
)

builder.add_edge(
    "score_experience",
    "aggregate_scores"
)

builder.add_edge(
    "score_education",
    "aggregate_scores"
)

builder.add_conditional_edges(
    "aggregate_scores",
    assign_skill_workers,
    ["skill_worker"]
)

builder.add_edge(
    "skill_worker",
    "collect_skill_results"
)

builder.add_edge(
    "collect_skill_results",
    "external_research_service"
)

builder.add_conditional_edges(
    "research_agent",
    route_research,
    {
        "research_tools":
            "research_tools",

        "finish_research":
            "finish_research",
    }
)

builder.add_edge(
    "research_tools",
    "research_agent"
)

builder.add_edge(
    "finish_research",
    "score_public_signals"
)

builder.add_edge(
    "score_public_signals",
    "calculate_final_score"
)

builder.add_edge(
    "calculate_final_score",
    "make_decision"
)

builder.add_edge(
    "draft_email",
    "critique_email"
)

builder.add_edge(
    "draft_email_borderline",
    "critique_email"
)

builder.add_edge(
    "draft_email_reject",
    "critique_email"
)

builder.add_edge(
    "revise_email",
    "critique_email"
)

builder.add_edge(
    "finalize",
    END
)

builder.add_edge(
    "compensate",
    END
)


graph = builder.compile(
    checkpointer = memory
)

def save_graph_png(
    output_path: str = "output/graph.png"
):
    import os

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    png_data = graph.get_graph().draw_mermaid_png()

    with open(output_path, "wb") as file:
        file.write(png_data)

    return output_path