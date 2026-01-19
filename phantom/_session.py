"""Session - Isolated context for concurrent LLM conversations."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from concurrent.futures import Executor
from typing import Any, TypeVar

from ._errors import ResolutionError
from ._graph import group_by_level, topological_order
from ._ref import Ref
from ._registry import get_operation, get_operation_signature
from ._result import ToolResult

T = TypeVar("T")
logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        session_id: str | None = None,
        allowed_ops: set[str] | None = None,
        operations: dict[str, Callable[..., Any]] | None = None,
        strict_hooks: bool = False,
    ):
        """
        Create a new isolated session.

        Args:
            session_id: Optional custom ID. Auto-generated if not provided.
            allowed_ops: If provided, only these operations can be used.
                        None means all operations are allowed.
            operations: Optional custom operations dict. If provided, only these
                       operations are available (ignores global registry).
                       None means use the global registry.
            strict_hooks: If True, hook errors are re-raised instead of logged.
        """
        self.id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        self._refs: dict[str, Ref[Any]] = {}
        self._value_cache: dict[str, Any] = {}
        self._allowed_ops = allowed_ops
        self._operations = operations
        self._strict_hooks = strict_hooks
        self._reverse_deps: dict[str, set[str]] = {}

        self._hooks: dict[str, list[Callable[..., None]]] = {
            "before_resolve": [],
            "after_resolve": [],
            "on_error": [],
            "on_progress": [],
        }

    def _get_operation(self, name: str) -> Callable[..., Any]:
        """Get operation from session-local or global registry."""
        if self._operations is not None:
            if name not in self._operations:
                raise KeyError(f"Unknown operation in session: {name}")
            return self._operations[name]
        return get_operation(name)

    def ref(self, op_name: str, **kwargs: Any) -> Ref[Any]:
        """
        Create a ref for an operation in this session.

        Args:
            op_name: Name of the registered operation
            **kwargs: Arguments to pass to the operation

        Returns:
            A Ref registered in this session

        Raises:
            PermissionError: If op_name is not in allowed_ops (when set)
        """
        if self._allowed_ops is not None and op_name not in self._allowed_ops:
            raise PermissionError(
                f"Operation '{op_name}' is not allowed in this session. "
                f"Allowed operations: {sorted(self._allowed_ops)}"
            )

        self._get_operation(op_name)

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

        Uses the session's persistent cache to avoid re-computing values.
        If a ref has already been resolved in this session, returns the
        cached value immediately.

        Args:
            ref: The ref to resolve (or ref ID string)

        Returns:
            The concrete value produced by the operation
        """
        if isinstance(ref, str):
            ref = self.get(ref)
        return self._resolve_with_cache(ref)

    def _resolve_with_cache(self, ref: Ref[Any]) -> Any:
        """Resolve using session's persistent cache."""
        order = topological_order(ref)
        self._build_reverse_deps(order)
        total = len(order)
        completed = 0

        for node in order:
            if node.id in self._value_cache:
                completed += 1
                continue

            self._emit("on_progress", completed=completed, total=total, current=node)

            resolved_args: dict[str, Any] = {}
            for key, value in node.args.items():
                if isinstance(value, Ref):
                    resolved_args[key] = self._value_cache[value.id]
                else:
                    resolved_args[key] = value

            try:
                op_func = self._get_operation(node.op)
            except KeyError as e:
                chain = [r.id for r in order[: order.index(node) + 1]]
                raise ResolutionError(
                    f"Unknown operation: {node.op}",
                    node,
                    chain,
                    e,
                ) from e

            self._emit("before_resolve", ref=node, args=resolved_args)

            try:
                result = op_func(**resolved_args)
            except Exception as e:
                self._emit("on_error", ref=node, error=e)
                chain = [r.id for r in order[: order.index(node) + 1]]
                raise ResolutionError(
                    f"Operation '{node.op}' failed: {e}",
                    node,
                    chain,
                    e,
                ) from e

            self._emit("after_resolve", ref=node, result=result)
            self._value_cache[node.id] = result
            completed += 1

        self._emit("on_progress", completed=total, total=total, current=None)
        return self._value_cache[ref.id]

    def _build_reverse_deps(self, order: list[Ref[Any]]) -> None:
        """Build reverse dependency map for cache invalidation."""
        for node in order:
            for parent in node.parents:
                self._reverse_deps.setdefault(parent.id, set()).add(node.id)

    async def aresolve(
        self,
        ref: Ref[Any] | str,
        *,
        parallel: bool = True,
        timeout: float | None = None,
        executor: Executor | None = None,
    ) -> Any:
        """
        Resolve a ref asynchronously with optional parallel execution.

        Uses the session's persistent cache to avoid re-computing values.
        When parallel=True, independent branches of the DAG execute concurrently.

        Args:
            ref: The ref to resolve (or ref ID string)
            parallel: If True, execute independent branches concurrently
            timeout: Optional timeout in seconds. Raises TimeoutError if exceeded.
            executor: Optional executor for sync operations. Use ProcessPoolExecutor
                     for true parallelism of CPU-bound operations. Default uses
                     ThreadPoolExecutor which only parallelizes I/O-bound work.

        Returns:
            The concrete value produced by the operation
        """
        if isinstance(ref, str):
            ref = self.get(ref)

        if timeout is not None:
            return await asyncio.wait_for(
                self._aresolve_with_cache(
                    ref, parallel=parallel, executor=executor
                ),
                timeout=timeout,
            )
        return await self._aresolve_with_cache(
            ref, parallel=parallel, executor=executor
        )

    async def _aresolve_with_cache(
        self,
        ref: Ref[Any],
        *,
        parallel: bool = True,
        executor: Executor | None = None,
    ) -> Any:
        """Async resolve using session's persistent cache."""
        order = topological_order(ref)
        self._build_reverse_deps(order)
        total = len(order)

        if parallel:
            levels = group_by_level(order)
            completed = 0
            for level in levels:
                to_execute = [n for n in level if n.id not in self._value_cache]
                cached_count = len(level) - len(to_execute)
                completed += cached_count

                if not to_execute:
                    continue

                self._emit(
                    "on_progress", completed=completed, total=total, current=None
                )

                if len(to_execute) == 1:
                    await self._execute_one_async(to_execute[0], order, executor)
                else:
                    await asyncio.gather(*[
                        self._execute_one_async(node, order, executor)
                        for node in to_execute
                    ])
                completed += len(to_execute)

            self._emit("on_progress", completed=total, total=total, current=None)
        else:
            completed = 0
            for node in order:
                if node.id not in self._value_cache:
                    self._emit(
                        "on_progress", completed=completed, total=total, current=node
                    )
                    await self._execute_one_async(node, order, executor)
                completed += 1
            self._emit("on_progress", completed=total, total=total, current=None)

        return self._value_cache[ref.id]

    async def _execute_one_async(
        self,
        node: Ref[Any],
        order: list[Ref[Any]],
        executor: Executor | None = None,
    ) -> None:
        """Execute a single operation asynchronously."""
        resolved_args: dict[str, Any] = {}
        for key, value in node.args.items():
            if isinstance(value, Ref):
                resolved_args[key] = self._value_cache[value.id]
            else:
                resolved_args[key] = value

        try:
            op_func = self._get_operation(node.op)
        except KeyError as e:
            chain = [r.id for r in order[: order.index(node) + 1]]
            raise ResolutionError(
                f"Unknown operation: {node.op}",
                node,
                chain,
                e,
            ) from e

        self._emit("before_resolve", ref=node, args=resolved_args)

        try:
            if asyncio.iscoroutinefunction(op_func):
                result = await op_func(**resolved_args)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    executor, lambda: op_func(**resolved_args)
                )
        except ResolutionError:
            raise
        except Exception as e:
            self._emit("on_error", ref=node, error=e)
            chain = [r.id for r in order[: order.index(node) + 1]]
            raise ResolutionError(
                f"Operation '{node.op}' failed: {e}",
                node,
                chain,
                e,
            ) from e

        self._emit("after_resolve", ref=node, result=result)
        self._value_cache[node.id] = result

    def list_refs(self) -> list[Ref[Any]]:
        """List all refs in this session."""
        return list(self._refs.values())

    def get_tools(
        self, format: str = "openai", include_peek: bool = True
    ) -> list[dict[str, Any]]:
        """
        Get tool definitions, filtered to allowed operations.

        If allowed_ops was set at session creation, only those operations
        are included. Otherwise, all registered operations are included.

        Args:
            format: The schema format to use. Options: "openai", "anthropic"
            include_peek: Whether to include the peek tool (default True)

        Returns:
            A list of tool definitions in the specified format.
        """
        from ._registry import get_tools as _get_tools

        all_tools = _get_tools(format=format, include_peek=include_peek)

        if self._allowed_ops is None:
            return all_tools

        filtered = []
        for tool in all_tools:
            if format == "openai":
                name = tool.get("function", {}).get("name", "")
            else:
                name = tool.get("name", "")

            if name == "peek" and include_peek:
                filtered.append(tool)
            elif name in self._allowed_ops:
                filtered.append(tool)

        return filtered

    def clear(self) -> None:
        """Clear all refs and cached values from this session."""
        self._refs.clear()
        self._value_cache.clear()
        self._reverse_deps.clear()

    def on(self, event: str) -> Callable[[Callable[..., None]], Callable[..., None]]:
        """
        Register a session-scoped hook for resolution events.

        Available events:
            - "before_resolve": Called before each operation executes
              Signature: (ref: Ref, args: dict[str, Any]) -> None

            - "after_resolve": Called after each operation completes successfully
              Signature: (ref: Ref, result: Any) -> None

            - "on_error": Called when an operation fails
              Signature: (ref: Ref, error: Exception) -> None

            - "on_progress": Called during resolution to report progress
              Signature: (completed: int, total: int, current: Ref | None) -> None

        Example:
            session = phantom.Session()

            @session.on("before_resolve")
            def log_start(ref, args):
                print(f"Starting {ref.op}...")

            @session.on("after_resolve")
            def log_done(ref, result):
                print(f"Finished {ref.op}")

            @session.on("on_progress")
            def show_progress(completed, total, current):
                print(f"Progress: {completed}/{total}")
        """
        if event not in self._hooks:
            raise ValueError(
                f"Unknown event: {event}. Valid events: {list(self._hooks.keys())}"
            )

        def decorator(fn: Callable[..., None]) -> Callable[..., None]:
            self._hooks[event].append(fn)
            return fn

        return decorator

    def _emit(self, event: str, **kwargs: Any) -> None:
        """Emit event to session's hooks."""
        for hook in self._hooks.get(event, []):
            try:
                hook(**kwargs)
            except Exception as e:
                logger.warning(f"Hook error in '{event}': {e}")
                if self._strict_hooks:
                    raise

    def clear_hooks(self, event: str | None = None) -> None:
        """
        Clear session's hooks.

        Args:
            event: If provided, clear only this event's hooks.
                   If None, clear all hooks.
        """
        if event is not None:
            if event in self._hooks:
                self._hooks[event].clear()
        else:
            for handlers in self._hooks.values():
                handlers.clear()

    def list_hooks(self) -> dict[str, int]:
        """
        List the number of registered hooks per event in this session.

        Returns:
            Dict mapping event names to hook counts
        """
        return {event: len(handlers) for event, handlers in self._hooks.items()}

    def invalidate(self, ref_id: str | None = None, *, cascade: bool = True) -> int:
        """
        Invalidate cached values.

        Args:
            ref_id: If provided, only invalidate this ref's cached value.
                   If None, clear all cached values.
            cascade: If True (default), also invalidate all refs that depend
                    on this ref. Only applies when ref_id is provided.

        Returns:
            Number of refs invalidated.
        """
        if ref_id is None:
            count = len(self._value_cache)
            self._value_cache.clear()
            return count

        invalidated: set[str] = set()
        self._cascade_invalidate(ref_id, invalidated, cascade)
        return len(invalidated)

    def _cascade_invalidate(
        self, ref_id: str, invalidated: set[str], cascade: bool
    ) -> None:
        """Recursively invalidate a ref and optionally its dependents."""
        if ref_id in invalidated:
            return
        if ref_id in self._value_cache:
            del self._value_cache[ref_id]
            invalidated.add(ref_id)
        if cascade:
            for dep_id in self._reverse_deps.get(ref_id, []):
                self._cascade_invalidate(dep_id, invalidated, cascade)

    def peek(self, ref: Ref[Any] | str) -> dict[str, Any]:
        """
        Peek at a ref's resolved value and return info about it.

        Uses the session's cache, so repeated peeks are instant.
        This is the recommended way to inspect refs in a session.

        Args:
            ref: The ref to peek at (or ref ID string)

        Returns:
            Dict containing ref info and inspector output
        """
        from ._inspect import _inspect_value

        if isinstance(ref, str):
            ref = self.get(ref)

        value = self.resolve(ref)
        info = _inspect_value(value)

        return {
            "ref": ref.id,
            "op": ref.op,
            "parents": [p.id for p in ref.parents],
            **info,
        }

    async def apeek(self, ref: Ref[Any] | str) -> dict[str, Any]:
        """
        Async version of peek.

        Uses the session's cache, so repeated peeks are instant.

        Args:
            ref: The ref to peek at (or ref ID string)

        Returns:
            Dict containing ref info and inspector output
        """
        from ._inspect import _inspect_value

        if isinstance(ref, str):
            ref = self.get(ref)

        value = await self.aresolve(ref)
        info = _inspect_value(value)

        return {
            "ref": ref.id,
            "op": ref.op,
            "parents": [p.id for p in ref.parents],
            **info,
        }

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

        Uses the session's cache, so repeated peeks are instant.

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
        try:
            if name == "peek":
                ref_id = arguments.get("ref") or arguments.get("ref_id")
                if ref_id is None:
                    raise ValueError("peek requires a 'ref' or 'ref_id' argument")
                return ToolResult.from_peek(self.peek(ref_id))
            else:
                return ToolResult.from_ref(self.ref_from_tool_call(name, arguments))
        except ResolutionError as e:
            if catch_errors:
                return ToolResult.from_error(e)
            raise

    def __repr__(self) -> str:
        return f"Session({self.id!r}, refs={len(self._refs)})"
