"""Registry - Operation registration and ref storage."""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Callable, TypeVar, get_type_hints

from ._ref import Ref

T = TypeVar("T")

_operations: dict[str, Callable[..., Any]] = {}
_refs: dict[str, Ref[Any]] = {}


def op(func: Callable[..., T]) -> Callable[..., T]:
    """
    Register a function as a Phantom operation.

    The decorated function defines concrete behavior that executes
    when a ref is resolved. The function name becomes the operation name.

    Example:
        @phantom.op
        def load(source: str) -> pd.DataFrame:
            return pd.read_parquet(source)

        # Creates a ref, doesn't execute yet
        data = phantom.ref("load", source="data.parquet")

        # Now executes
        df = phantom.resolve(data)
    """
    name = func.__name__
    _operations[name] = func

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return func(*args, **kwargs)

    return wrapper


def get_operation(name: str) -> Callable[..., Any]:
    """Get a registered operation by name."""
    if name not in _operations:
        raise KeyError(f"Unknown operation: {name}")
    return _operations[name]


def list_operations() -> list[str]:
    """List all registered operation names."""
    return list(_operations.keys())


def get_operation_signature(name: str) -> dict[str, Any]:
    """Get operation signature info (for tool generation)."""
    func = get_operation(name)
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

    params = {}
    for param_name, param in sig.parameters.items():
        param_info: dict[str, Any] = {}
        if param_name in hints:
            param_info["type"] = hints[param_name].__name__ if hasattr(hints[param_name], "__name__") else str(hints[param_name])
        if param.default is not inspect.Parameter.empty:
            param_info["default"] = param.default
        params[param_name] = param_info

    return {
        "name": name,
        "doc": func.__doc__,
        "params": params,
        "return_type": hints.get("return", Any).__name__ if hasattr(hints.get("return", Any), "__name__") else str(hints.get("return", Any)),
    }


def register_ref(ref: Ref[T]) -> Ref[T]:
    """Store a ref in the global registry."""
    _refs[ref.id] = ref
    return ref


def get_ref(ref_id: str) -> Ref[Any]:
    """Retrieve a ref by ID."""
    if ref_id not in _refs:
        raise KeyError(f"Unknown ref: {ref_id}")
    return _refs[ref_id]


def list_refs() -> list[Ref[Any]]:
    """List all registered refs."""
    return list(_refs.values())


def clear() -> None:
    """Clear all refs (useful for testing)."""
    _refs.clear()
