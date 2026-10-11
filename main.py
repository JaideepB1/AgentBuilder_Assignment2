from pathlib import Path

from hiregraph.graph import graph, save_graph_png


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "sample_data"


def load_text(filename: str) -> str:
    path = DATA_DIR / filename
    return path.read_text(encoding="utf-8")

def run_candidate(
    label: str,
    resume_file: str,
    job_description: str
):
    resume = load_text(resume_file)

    config = {
        "configurable": {
            "thread_id": label
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
        "external_retry_attempts": 0,
    }

    result = graph.invoke(
        initial_state,
        config=config
    )

    return {
        "scenario": label,
        "candidate": result.get(
            "candidate_name",
            "Unknown"
        ),
        "seniority": result.get(
            "seniority",
            "Unknown"
        ),
        "score": result.get(
            "total_score",
            0
        ),
        "recommendation": result.get(
            "recommendation",
            "N/A"
        ),
        "status": result.get(
            "terminal_status",
            "Unknown"
        ),
        "email": result.get(
            "email",
            ""
        ),
        "audit_trail": result.get(
            "audit_trail",
            []
        ),
    }


def print_scoreboard(results):
    print("\n")
    print("=" * 78)
    print("HIREGRAPH SCOREBOARD")
    print("=" * 78)

    print(
        f"{'Scenario':<15}"
        f"{'Candidate':<20}"
        f"{'Level':<12}"
        f"{'Score':<10}"
        f"{'Decision':<15}"
    )

    print("-" * 78)

    for result in results:
        print(
            f"{result['scenario']:<15}"
            f"{result['candidate']:<20}"
            f"{result['seniority']:<12}"
            f"{result['score']:<10}"
            f"{result['recommendation']:<15}"
        )

    print("=" * 78)


def main():

    graph_path = save_graph_png()

    print(
        f"Graph visualization saved to: "
        f"{graph_path}"
    )

    job_description = load_text(
        "backend_jd.txt"
    )


    scenarios = [
        (
            "Strong",
            "strong_resume.txt"
        ),
        (
            "Borderline",
            "borderline_resume.txt"
        ),
        (
            "Weak",
            "weak_resume.txt"
        ),
    ]

    results = []

    for label, resume_file in scenarios:

        print()
        print("=" * 50)
        print(f"Running {label} candidate")
        print("=" * 50)

        result = run_candidate(
            label,
            resume_file,
            job_description
        )

        results.append(result)

        print(
            "Candidate:",
            result["candidate"]
        )

        print(
            "Seniority:",
            result["seniority"]
        )

        print(
            "Score:",
            result["score"]
        )

        print(
            "Recommendation:",
            result["recommendation"]
        )

        print(
            "Status:",
            result["status"]
        )

    print_scoreboard(results)


if __name__ == "__main__":
    main()