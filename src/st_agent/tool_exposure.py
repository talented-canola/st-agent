"""Strict provider schemas for admitted scientific tools."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from st_agent.tool_registry import SCIENTIFIC_TOOLS

EXPOSED_TOOLS = SCIENTIFIC_TOOLS
INTERNAL_TO_PROVIDER = {name: name.replace(".", "_") for name in EXPOSED_TOOLS}
PROVIDER_TO_INTERNAL = {value: key for key, value in INTERNAL_TO_PROVIDER.items()}


def provider_tools(allowed: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    result = []
    for internal_name, tool in EXPOSED_TOOLS.items():
        if allowed is not None and internal_name not in allowed:
            continue
        schema = tool.arguments.model_json_schema()
        schema["additionalProperties"] = False
        result.append(
            {
                "type": "function",
                "function": {
                    "name": INTERNAL_TO_PROVIDER[internal_name],
                    "description": (
                        "Coordinate-only KMeans partition. Does not use expression or identify biological domains."
                        if internal_name == "spatial.domains" else
                        "Expression-weighted section-local spatial graph Leiden; requires provenance-recorded PCA."
                        if internal_name == "spatial.expression_domains" else
                        f"Execute verified scientific tool {internal_name}."),
                    "parameters": schema,
                    "strict": True,
                },
            }
        )
    return result


def validate_provider_call(
    name: str, arguments: str, *, allowed: tuple[str, ...] | None = None
) -> tuple[str, dict[str, Any]]:
    if name not in PROVIDER_TO_INTERNAL:
        raise ValueError(f"Unknown or unauthorized tool: {name}")
    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError as error:
        raise ValueError("Tool arguments are not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("Tool arguments must be an object")
    internal = PROVIDER_TO_INTERNAL[name]
    if allowed is not None and internal not in allowed:
        raise ValueError(f"Tool is not executable in this context: {internal}")
    try:
        validated = EXPOSED_TOOLS[internal].arguments.model_validate_json(arguments)
    except ValidationError as error:
        raise ValueError("Tool arguments failed strict validation") from error
    return internal, validated.model_dump(mode="json")
