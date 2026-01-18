"""Session - Isolated context for concurrent LLM conversations."""

from __future__ import annotations

import uuid
from typing import Any, TypeVar

from ._ref import Ref
from ._registry import get_operation
from ._resolve import aresolve as _aresolve
from ._resolve import resolve as _resolve
from ._result import ToolResult

T = TypeVar("T")


class Session:
    """
    Isolated context for a single LLM conversation.

    Each session maintains its own ref registry, preventing conflicts
    between concurrent LLM sessions. Operations remain global (they are
    stateless and registered at import time).

    Example:
        session = phantom.Session()

        # Create refs in this session
        data = session.ref("load", source="data.csv")
        filtered = session.ref("filter", data=data, condition="x > 0")

        # Resolve within this session
        result = session.resolve(filtered)

        # Sessions are isolated
        other_session = phantom.Session()
        # other_session.get(data.id) would raise KeyError
    """

    def __init__(self, session_id: str | None = None):
        """
        Create a new isolated session.

        Args:
            session_id: Optional custom ID. Auto-generated if not provided.
        """
        self.id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        self._refs: dict[str, Ref[Any]] = {}

    def ref(self, op_name: str, **kwargs: Any) -> Ref[Any]:
        """
        Create a ref for an operation in this session.

        Args:
            op_name: Name of the registered operation
            **kwargs: Arguments to pass to the operation

        Returns:
            A Ref registered in this session
        """
        get_operation(op_name)

        new_ref: Ref[Any] = Ref(op=op_name, args=kwargs)
        self._refs[new_ref.id] = new_ref
        return new_ref

    def get(self, ref_id: str) -> Ref[Any]:
        """
        Get a ref by its ID from this session.

        Args:
            ref_id: The ref ID (e.g., "@a3f2")

        Returns:
            The Ref object

        Raises:
            KeyError: If ref not found in this session
        """
        if ref_id not in self._refs:
            raise KeyError(f"Unknown ref in session {self.id}: {ref_id}")
        return self._refs[ref_id]

    def resolve(self, ref: Ref[Any] | str) -> Any:
        """
        Resolve a ref to its concrete value.

        Args:
            ref: The ref to resolve (or ref ID string)

        Returns:
            The concrete value produced by the operation
        """
        if isinstance(ref, str):
            ref = self.get(ref)
        return _resolve(ref)

    async def aresolve(self, ref: Ref[Any] | str, *, parallel: bool = True) -> Any:
        """
        Resolve a ref asynchronously with optional parallel execution.

        When parallel=True, independent branches of the DAG execute concurrently.
        This can significantly speed up I/O-bound workflows.

        Args:
            ref: The ref to resolve (or ref ID string)
            parallel: If True, execute independent branches concurrently

        Returns:
            The concrete value produced by the operation
        """
        if isinstance(ref, str):
            ref = self.get(ref)
        return await _aresolve(ref, parallel=parallel)

    def list_refs(self) -> list[Ref[Any]]:
        """List all refs in this session."""
        return list(self._refs.values())

    def clear(self) -> None:
        """Clear all refs from this session."""
        self._refs.clear()

    def ref_from_tool_call(self, op_name: str, arguments: dict[str, Any]) -> Ref[Any]:
        """
        Create a ref from an LLM tool call, auto-resolving ref ID strings.

        Args:
            op_name: Name of the operation (from tool call)
            arguments: Arguments dict, may contain "@ref_id" strings

        Returns:
            A new Ref for this operation
        """
        resolved_args: dict[str, Any] = {}
        for key, value in arguments.items():
            if isinstance(value, str) and value.startswith("@"):
                resolved_args[key] = self.get(value)
            else:
                resolved_args[key] = value

        return self.ref(op_name, **resolved_args)

    def handle_tool_call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """
        Handle any LLM tool call within this session.

        This is the recommended single entry point for processing tool calls.
        It handles both lazy operations (which create refs) and eager operations
        like peek (which resolve and inspect immediately).

        Args:
            name: Tool/operation name from LLM
            arguments: Arguments dict (may contain "@ref_id" strings)

        Returns:
            ToolResult that can be serialized back to the LLM

        Example:
            for tool_call in response.tool_calls:
                result = session.handle_tool_call(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result.to_dict())
                })
        """
        from ._inspect import peek

        if name == "peek":
            ref_id = arguments.get("ref") or arguments.get("ref_id")
            return ToolResult.from_peek(peek(self.get(ref_id)))
        else:
            return ToolResult.from_ref(self.ref_from_tool_call(name, arguments))

    def __repr__(self) -> str:
        return f"Session({self.id!r}, refs={len(self._refs)})"
