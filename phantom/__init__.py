"""
Phantom - The semantic-concrete bridge for LLM data pipelines.

Phantom separates concerns:
- LLM works only with refs (semantic handles)
- Developer implements only concrete operations
- Phantom bridges them via a lazy computation graph

Example:
    import phantom

    # Developer defines operations
    @phantom.op
    def load(source: str) -> pd.DataFrame:
        return pd.read_parquet(source)

    @phantom.op
    def filter(data: pd.DataFrame, condition: str) -> pd.DataFrame:
        return data.query(condition)

    # Create refs (lazy - nothing executes)
    sales = phantom.ref("load", source="sales.parquet")
    filtered = phantom.ref("filter", data=sales, condition="amount > 100")

    # Resolve when needed
    df = phantom.resolve(filtered)
"""

from typing import Any, TypeVar

from ._errors import CycleError, ResolutionError
from ._inspect import apeek, inspector, peek
from ._ref import Ref
from ._registry import (
    clear,
    get_operation,
    get_operation_signature,
    get_ref,
    get_tools,
    list_operations,
    list_refs,
    op,
    register_ref,
    validate_args,
)
from ._resolve import aresolve, resolve
from ._result import ToolResult
from ._serialize import deserialize_graph, load_graph, save_graph, serialize_graph
from ._session import Session

__all__ = [
    # Core types
    "Ref",
    "ToolResult",
    "Session",
    # Errors
    "ResolutionError",
    "CycleError",
    # Decorators
    "op",
    "inspector",
    # Functions
    "ref",
    "ref_from_tool_call",
    "handle_tool_call",
    "resolve",
    "aresolve",
    "peek",
    "apeek",
    "get",
    # Registry
    "list_operations",
    "list_refs",
    "get_operation_signature",
    "get_tools",
    "validate_args",
    "clear",
    # Serialization
    "serialize_graph",
    "deserialize_graph",
    "save_graph",
    "load_graph",
]

T = TypeVar("T")


def ref(op_name: str, **kwargs: Any) -> Ref[Any]:
    """
    Create a ref for an operation.

    This builds a node in the computation graph. Nothing executes
    until resolve() is called.

    Args:
        op_name: Name of the registered operation
        **kwargs: Arguments to pass to the operation

    Returns:
        A Ref that can be resolved later

    Example:
        sales = phantom.ref("load", source="sales.parquet")
        filtered = phantom.ref("filter", data=sales, condition="x > 0")
    """

    get_operation(op_name)

    new_ref: Ref[Any] = Ref(op=op_name, args=kwargs)
    return register_ref(new_ref)


def get(ref_id: str) -> Ref[Any]:
    """
    Get a ref by its ID.

    Args:
        ref_id: The ref ID (e.g., "@a3f2")

    Returns:
        The Ref object

    Example:
        ref = phantom.get("@a3f2")
    """
    return get_ref(ref_id)


def ref_from_tool_call(op_name: str, arguments: dict[str, Any]) -> Ref[Any]:
    """
    Create a ref from an LLM tool call, auto-resolving ref ID strings.

    This is the recommended way to handle tool calls from LLMs. It automatically
    converts any string argument starting with "@" into the corresponding Ref.

    Args:
        op_name: Name of the operation (from tool call)
        arguments: Arguments dict (from tool call), may contain "@ref_id" strings

    Returns:
        A new Ref for this operation

    Example:
        # LLM returns: {"name": "filter", "arguments": {"data": "@a3f2", "col": "x"}}
        ref = phantom.ref_from_tool_call("filter", {"data": "@a3f2", "col": "x"})
        # "@a3f2" is automatically resolved to the actual Ref
    """
    resolved_args: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, str) and value.startswith("@"):
            resolved_args[key] = get_ref(value)
        else:
            resolved_args[key] = value

    return ref(op_name, **resolved_args)


def handle_tool_call(name: str, arguments: dict[str, Any]) -> ToolResult:
    """
    Handle any LLM tool call, including peek.

    This is the recommended single entry point for processing tool calls.
    It handles both lazy operations (which create refs) and eager operations
    like peek (which resolve and inspect immediately).

    Args:
        name: Tool/operation name from LLM
        arguments: Arguments dict from LLM (may contain "@ref_id" strings)

    Returns:
        ToolResult that can be serialized back to the LLM

    Example:
        for tool_call in response.tool_calls:
            result = phantom.handle_tool_call(
                tool_call.function.name,
                json.loads(tool_call.function.arguments)
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result.to_dict())
            })
    """
    if name == "peek":
        ref_id = arguments.get("ref") or arguments.get("ref_id")
        if ref_id is None:
            raise ValueError("peek requires a 'ref' or 'ref_id' argument")
        return ToolResult.from_peek(peek(get_ref(ref_id)))
    else:
        return ToolResult.from_ref(ref_from_tool_call(name, arguments))
