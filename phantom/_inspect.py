"""Inspect - Pluggable data inspection for LLM feedback."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

InspectorFunc = Callable[[Any], dict[str, Any]]

_default_inspectors: dict[type, InspectorFunc] = {}


def _inspect_value(
    value: Any,
    session_inspectors: dict[type, InspectorFunc] | None = None,
) -> dict[str, Any]:
    """
    Run the appropriate inspector for a value.

    Args:
        value: The value to inspect
        session_inspectors: Optional session-specific inspectors (checked first)

    Returns:
        Dict describing the value
    """

    if session_inspectors:
        for dtype, inspector_fn in session_inspectors.items():
            if isinstance(value, dtype):
                return inspector_fn(value)

    for dtype, inspector_fn in _default_inspectors.items():
        if isinstance(value, dtype):
            return inspector_fn(value)

    return {"type": type(value).__name__, "value": repr(value)[:200]}


def _inspect_list(data: list[Any]) -> dict[str, Any]:
    """Default inspector for lists."""
    if not data:
        return {"type": "list", "length": 0}

    sample = data[:5]
    result: dict[str, Any] = {"type": "list", "length": len(data), "sample": sample}

    if isinstance(data[0], dict):
        keys: set[str] = set()
        for item in data[:100]:
            if isinstance(item, dict):
                keys.update(item.keys())
        result["keys"] = sorted(keys)

    return result


def _inspect_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Default inspector for dicts."""
    return {
        "type": "dict",
        "keys": list(data.keys()),
        "sample": {k: data[k] for k in list(data.keys())[:5]},
    }


_default_inspectors[list] = _inspect_list
_default_inspectors[dict] = _inspect_dict
