"""ST Research Agent v2 foundation package."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from st_agent.api import ResearchCopilot

__all__ = ["ResearchCopilot", "__version__"]
__version__ = "0.10.0"


def __getattr__(name: str):
    # Spawn workers need contracts and execution only, not the complete project API.
    if name == "ResearchCopilot":
        from st_agent.api import ResearchCopilot

        globals()[name] = ResearchCopilot
        return ResearchCopilot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
