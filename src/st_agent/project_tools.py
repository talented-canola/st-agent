"""Read-only project-bound tools exposed to the live workbench.

Transforming tools deliberately do not pass through this adapter.  The old
adapter created revisions directly from a Streamlit request and bypassed the
approval, worker, ToolResult and provenance runtime.  They remain unavailable
to the live model until the plan-bound executor is connected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from st_agent.scientific_tools import execute_scientific_tool


class ProjectToolExecutor:
    """Execute a verified tool against one registered dataset."""

    # Plot kernels need a bounded researcher-facing artifact path.  Transforming
    # tools need a plan-bound approval and execution runtime.  Neither belongs
    # in the direct UI adapter.
    supported_tools = (
        "dataset.inspect",
        "qc.compute_metrics",
    )

    def __init__(self, copilot, dataset_id: str) -> None:
        self.copilot = copilot
        self.dataset_id = dataset_id

    def __call__(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self.supported_tools:
            raise ValueError(
                f"{name} requires a reviewed execution plan and is unavailable in live chat"
            )
        dataset = self.copilot.get_dataset(self.dataset_id)
        if dataset.active_revision_id is None:
            raise ValueError("Dataset has no active revision")
        input_revision = self.copilot.revisions.get(dataset.active_revision_id)
        source = self._revision_path(input_revision.relative_path)
        return {
            **execute_scientific_tool(name, source, arguments=arguments),
            "input_revision_id": input_revision.id,
            "output_revision_id": None,
        }

    def _revision_path(self, relative_path: str | None) -> Path:
        if not relative_path:
            raise ValueError("Active revision has no data file")
        root = self.copilot.layout.root.resolve()
        source = (root / relative_path).resolve()
        if root not in source.parents or not source.is_file():
            raise ValueError("Active revision path is missing or outside the project")
        return source
