import os

from langchain_core.tools import tool


USE_MOCKS = os.getenv(
    "USE_MOCKS",
    "true"
).lower() == "true"


@tool
def github_lookup(candidate_name: str) -> str:
    """
    Look for public GitHub information about a candidate.

    Use this when public GitHub information could provide
    additional evidence about a candidate's technical work.
    """

    if USE_MOCKS:
        return (
            f"Mock GitHub lookup for {candidate_name}: "
            "No verified GitHub profile was supplied, so no "
            "additional GitHub evidence is available."
        )

    # Real GitHub integration can be added here later.
    return (
        f"No real GitHub integration is configured for "
        f"{candidate_name}."
    )


@tool
def public_profile_search(
    candidate_name: str
) -> str:
    """
    Search for public professional information about a candidate.

    Use this when public professional information could provide
    additional evidence relevant to the hiring evaluation.
    """

    if USE_MOCKS:
        return (
            f"Mock public profile search for {candidate_name}: "
            "No verified external professional information "
            "was found."
        )

    # Real search integration can be added here later.
    return (
        f"No real public-profile search integration is "
        f"configured for {candidate_name}."
    )


TOOLS = [
    github_lookup,
    public_profile_search,
]