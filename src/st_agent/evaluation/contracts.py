"""Versionable scenario contracts independent of production routing."""

from pydantic import Field, model_validator

from st_agent.contracts import StrictRecord
from st_agent.planning import TaskOperation


class EvalTurn(StrictRecord):
    semantic_id: str
    utterance_family: str
    utterance: str = Field(min_length=1)


class EvalCase(StrictRecord):
    id: str
    fixture: str
    turns: tuple[EvalTurn, ...] = Field(min_length=1)
    expected_operation: TaskOperation
    expected_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    invariants: tuple[str, ...] = ()

    @model_validator(mode="after")
    def check_tools(self):
        if set(self.expected_tools) & set(self.forbidden_tools):
            raise ValueError("Expected and forbidden tools overlap")
        return self
