"""Session - Isolated context for concurrent LLM conversations."""

from __future__ import annotations

import uuid
from typing import Any, TypeVar

from ._errors import ResolutionError
from ._ref import Ref
from ._registry import get_operation, get_operation_signature
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
            valid = list(self._refs.keys())[:5]
            if valid:
                raise KeyError(
                    f"Unknown ref '{ref_id}' in session. Valid refs: {valid}"
                )
            else:
                raise KeyError(f"Unknown ref '{ref_id}' in session (no refs exist)")
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

    def ref_from_tool_call(
        self, op_name: str, arguments: dict[str, Any], *, coerce_types: bool = True
    ) -> Ref[Any]:
        """
        Create a ref from an LLM tool call, auto-resolving ref ID strings.

        Args:
            op_name: Name of the operation (from tool call)
            arguments: Arguments dict, may contain "@ref_id" strings
            coerce_types: If True, coerce string args to expected types

        Returns:
            A new Ref for this operation
        """
        resolved_args: dict[str, Any] = {}

        sig = get_operation_signature(op_name) if coerce_types else None
        params = sig["params"] if sig else {}

        for key, value in arguments.items():
            if isinstance(value, str) and value.startswith("@"):
                resolved_args[key] = self.get(value)
            elif coerce_types and isinstance(value, str) and key in params:
                resolved_args[key] = self._coerce_type(value, params[key])
            else:
                resolved_args[key] = value

        return self.ref(op_name, **resolved_args)

    def _coerce_type(self, value: str, param_info: dict[str, Any]) -> Any:
        """Coerce a string value to the expected type if possible."""
        expected_type = param_info.get("type", "")

        if expected_type == "int":
            try:
                return int(value)
            except ValueError:
                return value
        elif expected_type == "float":
            try:
                return float(value)
            except ValueError:
                return value
        elif expected_type == "bool":
            if value.lower() in ("true", "1", "yes"):
                return True
            elif value.lower() in ("false", "0", "no"):
                return False
            return value

        return value

    def handle_tool_call(
        self, name: str, arguments: dict[str, Any], *, catch_errors: bool = False
    ) -> ToolResult:
        """
        Handle any LLM tool call within this session.

        This is the recommended single entry point for processing tool calls.
        It handles both lazy operations (which create refs) and eager operations
        like peek (which resolve and inspect immediately).

        Args:
            name: Tool/operation name from LLM
            arguments: Arguments dict (may contain "@ref_id" strings)
            catch_errors: If True, catch ResolutionError and return structured
                error result instead of raising

        Returns:
            ToolResult that can be serialized back to the LLM

        Example:
            for tool_call in response.tool_calls:
                result = session.handle_tool_call(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments),
                    catch_errors=True,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result.to_dict())
                })
        """
        from ._inspect import peek

        try:
            if name == "peek":
                ref_id = arguments.get("ref") or arguments.get("ref_id")
                return ToolResult.from_peek(peek(self.get(ref_id)))
            else:
                return ToolResult.from_ref(self.ref_from_tool_call(name, arguments))
        except ResolutionError as e:
            if catch_errors:
                return ToolResult.from_error(e)
            raise

    def __repr__(self) -> str:
        return f"Session({self.id!r}, refs={len(self._refs)})"
