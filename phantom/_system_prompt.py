"""System prompt generation for LLM integration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ._registry import get_operation_signature_from_func


def _format_operation_summary(
    operations: dict[str, Callable[..., Any]],
) -> str:
    """Build a concise summary of available operations for the system prompt."""
    if not operations:
        return "No operations registered."

    lines = []
    for op_name, op_func in operations.items():
        sig = get_operation_signature_from_func(op_name, op_func)
        doc = sig["doc"] or "No description."

        params = []
        for param_name, param_info in sig["params"].items():
            if param_info.get("is_ref"):
                params.append(f"{param_name}: ref")
            else:
                type_name = param_info.get("type", "str")
                if "default" in param_info:
                    default = param_info["default"]
                    params.append(
                        f"{param_name}: {type_name} = {default!r}"
                    )
                else:
                    params.append(f"{param_name}: {type_name}")

        lines.append(f"- {op_name}({', '.join(params)}): {doc.strip()}")

    return "\n".join(lines)


def build_system_prompt(
    operations: dict[str, Callable[..., Any]],
    developer_prompt: str = "",
) -> str:
    """
    Build the Phantom system prompt for LLM integration.

    Generates a prompt explaining how refs and tool chaining work,
    lists available operations, and appends any developer-provided context.

    Args:
        operations: The session's registered operations dict.
        developer_prompt: Additional context from the developer.

    Returns:
        The complete system prompt string.
    """
    ops_summary = _format_operation_summary(operations)

    parts = [
        _PHANTOM_PROMPT.format(operations=ops_summary),
    ]

    if developer_prompt:
        parts.append(developer_prompt)

    return "\n\n".join(parts)


_PHANTOM_PROMPT = """\
## Tool System

You have access to tools that build a lazy computation graph. \
Each tool call returns a **ref** — an opaque handle to a result. \
Use refs to chain operations together.

### How It Works
1. Call a tool (e.g., `load_csv(path="data.csv")`) → returns `{{"ref": "@a3f2", ...}}`
2. Pass that ref ID to the next tool: `filter(data="@a3f2", column="age", value="30")`
3. Chain as many operations as needed. \
Data stays in the application — you only see ref IDs.

### Inspecting Data
Use `peek(ref="@a3f2")` to inspect any ref's contents: \
type, shape, columns, and sample rows. \
Always peek before transforming unfamiliar data.

### Rules
- Use the exact ref ID string (e.g., `"@a3f2"`) returned by tool calls. \
Never fabricate ref IDs.
- Present results and insights to the user in natural language, not raw ref IDs.
- Chain operations step by step. Each tool call does one thing.

### Security
- Only access files the user has explicitly mentioned or that are \
in the working directory.
- Use the dedicated read operations (read_csv, read_parquet, etc.) for file access — \
never use raw SQL to read from file paths.
- Do not attempt to access system files, environment variables, credentials, \
or private keys.
- Do not construct SQL containing COPY, INSTALL, LOAD, ATTACH, or other \
administrative statements.

### Available Operations
{operations}"""
