"""Inspect - Pluggable data inspection for LLM feedback."""

from __future__ import annotations

from typing import Any, Callable

from ._ref import Ref
from ._registry import get_ref
from ._resolve import resolve

_inspectors: dict[type, Callable[[Any], dict[str, Any]]] = {}


def inspector(data_type: type) -> Callable[[Callable[[Any], dict[str, Any]]], Callable[[Any], dict[str, Any]]]:
    """
    Decorator to register an inspector for a data type.

    Inspectors are functions that take a value and return a dict
    describing it. The dict format is entirely up to the inspector -
    different data types need different representations.

    Args:
        data_type: The type this inspector handles

    Example:
        @inspector(pd.DataFrame)
        def inspect_dataframe(df: pd.DataFrame) -> dict[str, Any]:
            return {
                "type": "dataframe",
                "shape": list(df.shape),
                "columns": list(df.columns),
            }
    """
    def decorator(fn: Callable[[Any], dict[str, Any]]) -> Callable[[Any], dict[str, Any]]:
        _inspectors[data_type] = fn
        return fn
    return decorator


def peek(ref: Ref[Any] | str) -> dict[str, Any]:
    """
    Peek at a ref's resolved value and return info about it.

    Forces resolution, then runs the appropriate inspector.
    Returns a dict with ref metadata plus inspector output.

    Args:
        ref: The ref to peek at (or ref ID string)

    Returns:
        Dict containing ref info and inspector output

    Example:
        data_ref = phantom.ref("load_data", source="users.json")
        info = phantom.peek(data_ref)
        # {"ref": "@a3f2", "op": "load_data", "type": "list", ...}
    """
    if isinstance(ref, str):
        ref = get_ref(ref)

    value = resolve(ref)
    info = _inspect_value(value)

    return {
        "ref": ref.id,
        "op": ref.op,
        "parents": [p.id for p in ref.parents],
        **info,
    }


def _inspect_value(value: Any) -> dict[str, Any]:
    """Run the appropriate inspector for a value."""
    for dtype, inspector_fn in _inspectors.items():
        if isinstance(value, dtype):
            return inspector_fn(value)
    return {"type": type(value).__name__, "value": repr(value)[:200]}



@inspector(list)
def _inspect_list(data: list[Any]) -> dict[str, Any]:
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


@inspector(dict)
def _inspect_dict(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "dict",
        "keys": list(data.keys()),
        "sample": {k: data[k] for k in list(data.keys())[:5]},
    }


try:
    import pandas as pd

    @inspector(pd.DataFrame)
    def _inspect_dataframe(df: pd.DataFrame) -> dict[str, Any]:
        return {
            "type": "dataframe",
            "shape": list(df.shape),
            "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample": df.head(5).to_dict("records"),
        }
except ImportError:
    pass
