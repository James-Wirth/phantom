"""Registry - Operation registration and ref storage."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, get_origin, get_type_hints

from ._ref import Ref

T = TypeVar("T")


def _is_ref_type(type_hint: Any) -> bool:
    """Check if a type hint is Ref or Ref[T]."""
    if type_hint is Ref:
        return True
    origin = get_origin(type_hint)
    return origin is Ref

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
            type_hint = hints[param_name]
            param_info["type_hint"] = type_hint
            if hasattr(type_hint, "__name__"):
                param_info["type"] = type_hint.__name__
            else:
                param_info["type"] = str(type_hint)
            param_info["is_ref"] = _is_ref_type(type_hint)
        if param.default is not inspect.Parameter.empty:
            param_info["default"] = param.default
        params[param_name] = param_info

    return_hint = hints.get("return", Any)
    if hasattr(return_hint, "__name__"):
        return_type = return_hint.__name__
    else:
        return_type = str(return_hint)

    return {
        "name": name,
        "doc": func.__doc__,
        "params": params,
        "return_type": return_type,
    }


def validate_args(op_name: str, args: dict[str, Any]) -> list[str]:
    """
    Validate arguments against an operation's signature.

    Returns a list of warnings (not errors) for issues like:
    - Missing required parameters
    - Unknown parameters
    - Basic type mismatches

    This is advisory - it doesn't block ref creation, just warns.

    Args:
        op_name: Name of the registered operation
        args: Arguments dict to validate

    Returns:
        List of warning messages (empty if all OK)
    """
    warnings = []
    sig = get_operation_signature(op_name)
    params = sig["params"]

    for param_name, param_info in params.items():
        if "default" not in param_info and param_name not in args:
            warnings.append(f"Missing required parameter: '{param_name}'")

    for arg_name in args:
        if arg_name not in params:
            warnings.append(f"Unknown parameter: '{arg_name}'")

    for arg_name, arg_value in args.items():
        if arg_name not in params:
            continue
        param_info = params[arg_name]
        if param_info.get("is_ref"):
            if not isinstance(arg_value, Ref):
                actual = type(arg_value).__name__
                warnings.append(
                    f"Parameter '{arg_name}' expects a Ref, got {actual}"
                )
        else:
            expected_type = param_info.get("type", "")
            actual_type = type(arg_value).__name__
            primitive_types = ("str", "int", "float", "bool")
            if expected_type in primitive_types and actual_type != expected_type:
                if actual_type != "str":
                    warnings.append(
                        f"Parameter '{arg_name}' expects {expected_type}, "
                        f"got {actual_type}"
                    )

    return warnings


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


def _python_type_to_json_schema(type_name: str) -> str:
    """Map Python type names to JSON Schema types."""
    mapping = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
        "None": "null",
        "NoneType": "null",
    }
    return mapping.get(type_name, "string")


def get_tools(
    format: str = "openai", include_peek: bool = True
) -> list[dict[str, Any]]:
    """
    Generate tool definitions for all registered operations.

    Args:
        format: The schema format to use. Options: "openai", "anthropic"
        include_peek: Whether to include the peek tool (default True)

    Returns:
        A list of tool definitions in the specified format.

    Example:
        @phantom.op
        def search(query: str, limit: int = 10) -> list[dict]:
            '''Search for items matching query.'''
            ...

        # OpenAI format (default)
        tools = phantom.get_tools()
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools,
        )

        # Anthropic format
        tools = phantom.get_tools(format="anthropic")
        response = anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            messages=messages,
            tools=tools,
        )
    """
    if format not in ("openai", "anthropic"):
        raise ValueError(f"Unknown format: {format}. Use 'openai' or 'anthropic'.")

    tools = []
    for op_name in list_operations():
        sig = get_operation_signature(op_name)

        properties = {}
        required = []

        for param_name, param_info in sig["params"].items():
            is_ref = param_info.get("is_ref", False)
            if is_ref:
                prop: dict[str, Any] = {
                    "type": "string",
                    "description": "A ref ID (e.g., '@abc123') from a prior op",
                    "pattern": "^@[a-f0-9]+$",
                }
            else:
                prop = {
                    "type": _python_type_to_json_schema(param_info.get("type", "str")),
                    "description": f"The {param_name} parameter",
                }
            properties[param_name] = prop

            if "default" not in param_info:
                required.append(param_name)

        name = sig["name"]
        description = sig["doc"] or f"Execute the {name} operation"
        parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        if format == "openai":
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            })
        else:  # anthropic
            tools.append({
                "name": name,
                "description": description,
                "input_schema": parameters,
            })

    if include_peek:
        peek_params = {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "The ref ID to inspect (e.g., '@abc123')",
                    "pattern": "^@[a-f0-9]+$",
                }
            },
            "required": ["ref"],
        }
        peek_description = (
            "Inspect a ref to see its type, shape, columns, and sample "
            "data. Use this to understand structure before transforming."
        )

        if format == "openai":
            tools.append({
                "type": "function",
                "function": {
                    "name": "peek",
                    "description": peek_description,
                    "parameters": peek_params,
                },
            })
        else:  # anthropic
            tools.append({
                "name": "peek",
                "description": peek_description,
                "input_schema": peek_params,
            })

    return tools
