"""Resolve - DAG resolution engine."""

from __future__ import annotations

from typing import Any

from ._ref import Ref
from ._registry import get_operation, get_ref


def resolve(ref: Ref[Any] | str, *, _cache: dict[str, Any] | None = None) -> Any:
    """
    Resolve a ref to its concrete value.

    Walks the computation DAG, resolving dependencies first,
    then executes the operation with resolved arguments.

    Args:
        ref: The ref to resolve (or ref ID string)
        _cache: Internal cache for resolved values (avoid re-computation)

    Returns:
        The concrete value produced by the operation

    Example:
        @phantom.op
        def double(x: int) -> int:
            return x * 2

        ref = phantom.ref("double", x=5)
        result = phantom.resolve(ref)  # Returns 10
    """
    if _cache is None:
        _cache = {}

    if isinstance(ref, str):
        ref = get_ref(ref)

    if ref.id in _cache:
        return _cache[ref.id]

    op_func = get_operation(ref.op)

    resolved_args: dict[str, Any] = {}
    for key, value in ref.args.items():
        if isinstance(value, Ref):
            resolved_args[key] = resolve(value, _cache=_cache)
        else:
            resolved_args[key] = value

    result = op_func(**resolved_args)

    _cache[ref.id] = result
    return result
