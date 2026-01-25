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
from ._operation_set import OperationSet
from ._ref import Ref
from ._result import ToolResult

T = TypeVar("T")
logger = logging.getLogger(__name__)


class Session:
    """
    Isolated context for LLM conversations with session-scoped operations.

    Each session maintains its own operation registry and ref storage,
    providing complete isolation between concurrent LLM sessions.
    Operations are registered using the @session.op decorator.

    Example:
        session = phantom.Session()

        @session.op
        def load(source: str) -> pd.DataFrame:
            return pd.read_parquet(source)

        @session.op
        def filter(data: pd.DataFrame, condition: str) -> pd.DataFrame:
            return data.query(condition)

        # Create refs (lazy - nothing executes yet)
        data = session.ref("load", source="data.csv")
        filtered = session.ref("filter", data=data, condition="x > 0")

        # Resolve when needed
        result = session.resolve(filtered)
    """

    def __init__(
        self,
        session_id: str | None = None,
        strict_hooks: bool = False,
    ):
        """
        Create a new isolated session.

        Args:
            session_id: Optional custom ID. Auto-generated if not provided.
            strict_hooks: If True, hook errors are re-raised instead of logged.
        """
        self.id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        self._refs: dict[str, Ref[Any]] = {}
        self._value_cache: dict[str, Any] = {}
        self._operations: dict[str, Callable[..., Any]] = {}
        self._inspectors: dict[type, Callable[[Any], dict[str, Any]]] = {}
        self._strict_hooks = strict_hooks
        self._reverse_deps: dict[str, set[str]] = {}

        self._hooks: dict[str, list[Callable[..., None]]] = {
            "before_resolve": [],
            "after_resolve": [],
            "on_error": [],
            "on_progress": [],
        }

    def op(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Register a function as an operation in this session.

        The decorated function can be called directly or used to create refs.
        The function name becomes the operation name.

        Example:
            session = phantom.Session()

            @session.op
            def add(x: int, y: int) -> int:
                return x + y

            # Create a ref (lazy)
            ref = session.ref("add", x=1, y=2)

            # Or call directly
            result = add(1, 2)
        """
        name = func.__name__
        self._operations[name] = func
        return func

    def register(
        self,
        *items: Callable[..., Any] | OperationSet,
        name: str | None = None,
    ) -> Session:
        """
        Register operations explicitly.

        This is an alternative to the @session.op decorator for cases where
        you define operations in separate modules or want explicit control.

        Args:
            *items: Functions or OperationSets to register
            name: Custom operation name (only valid with single function)

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If name provided with multiple items, or no items given
            TypeError: If item is not callable or OperationSet

        Example:
            # Register individual functions
            session.register(load_csv, merge, filter)

            # Register with custom name
            session.register(load_csv, name="load")

            # Register an OperationSet
            from ops.data import data_ops
            session.register(data_ops)

            # Chain registrations
            session.register(data_ops).register(utils_ops)

            # Mix functions and sets
            session.register(data_ops, standalone_func)
        """
        if not items:
            raise ValueError("register() requires at least one item")

        if name is not None and len(items) != 1:
            raise ValueError("'name' can only be used with a single function")

        for item in items:
            if isinstance(item, OperationSet):
                for op_name, func in item:
                    self._operations[op_name] = func
            elif callable(item):
                op_name = name if name is not None else item.__name__
                self._operations[op_name] = item
            else:
                raise TypeError(
                    f"Expected callable or OperationSet, got {type(item).__name__}"
                )

        return self

    def list_operations(self) -> list[str]:
        """List all operations registered in this session."""
        return list(self._operations.keys())

    def inspector(
        self, data_type: type
    ) -> Callable[[Callable[[Any], dict[str, Any]]], Callable[[Any], dict[str, Any]]]:
        """
        Register a custom inspector for a data type in this session.

        Inspectors define how `peek` summarizes data for the LLM.
        Session inspectors take precedence over default inspectors.

        Example:
            session = phantom.Session()

            @session.inspector(pd.DataFrame)
            def inspect_df(df: pd.DataFrame) -> dict[str, Any]:
                return {
                    "type": "dataframe",
                    "shape": list(df.shape),
                    "columns": list(df.columns),
                }
        """
        def decorator(
            fn: Callable[[Any], dict[str, Any]]
        ) -> Callable[[Any], dict[str, Any]]:
            self._inspectors[data_type] = fn
            return fn
        return decorator

    def _get_operation(self, name: str) -> Callable[..., Any]:
        """Get operation from this session's registry."""
        if name not in self._operations:
            raise KeyError(
                f"Unknown operation: '{name}'. "
                f"Register it with @session.op first."
            )
        return self._operations[name]

    def _get_operation_signature(self, name: str) -> dict[str, Any]:
        """Get operation signature for tool generation and type coercion."""
        from ._registry import get_operation_signature_from_func

        func = self._get_operation(name)
        return get_operation_signature_from_func(name, func)

    def ref(self, op_name: str, **kwargs: Any) -> Ref[Any]:
        """
        Create a ref for an operation in this session.

        Args:
            op_name: Name of the registered operation
            **kwargs: Arguments to pass to the operation

        Returns:
            A Ref registered in this session

        Raises:
            KeyError: If op_name is not registered in this session
        """

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
        Get tool definitions for this session's operations.

        Generates tool schemas from all operations registered with @session.op.

        Args:
            format: The schema format to use. Options: "openai", "anthropic"
            include_peek: Whether to include the peek tool (default True)

        Returns:
            A list of tool definitions in the specified format.
        """
        from ._registry import get_tools as _get_tools

        return _get_tools(
            self._operations,
            format=format,
            include_peek=include_peek,
        )

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
        Uses session-scoped inspectors if registered, otherwise defaults.

        Args:
            ref: The ref to peek at (or ref ID string)

        Returns:
            Dict containing ref info and inspector output
        """
        from ._inspect import _inspect_value

        if isinstance(ref, str):
            ref = self.get(ref)

        value = self.resolve(ref)
        info = _inspect_value(value, self._inspectors)

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
        Uses session-scoped inspectors if registered, otherwise defaults.

        Args:
            ref: The ref to peek at (or ref ID string)

        Returns:
            Dict containing ref info and inspector output
        """
        from ._inspect import _inspect_value

        if isinstance(ref, str):
            ref = self.get(ref)

        value = await self.aresolve(ref)
        info = _inspect_value(value, self._inspectors)

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

        sig = self._get_operation_signature(op_name) if coerce_types else None
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
        self, name: str, arguments: dict[str, Any] | str, *, catch_errors: bool = False
    ) -> ToolResult:
        """
        Handle any LLM tool call within this session.

        This is the recommended single entry point for processing tool calls.
        It handles both lazy operations (which create refs) and eager operations
        like peek (which resolve and inspect immediately).

        Uses the session's cache, so repeated peeks are instant.

        Args:
            name: Tool/operation name from LLM
            arguments: Arguments dict or JSON string (may contain "@ref_id" strings)
            catch_errors: If True, catch ResolutionError and return structured
                error result instead of raising

        Returns:
            ToolResult that can be serialized back to the LLM

        Example:
            for tool_call in response.tool_calls:
                result = session.handle_tool_call(
                    tool_call.function.name,
                    tool_call.function.arguments,  # JSON string or dict
                    catch_errors=True,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result.to_json(),
                })
        """
        import json

        if isinstance(arguments, str):
            arguments = json.loads(arguments)

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

    def save_graph(self, ref: Ref[Any] | str, path: str) -> None:
        """
        Save a computation graph from this session to a JSON file.

        Args:
            ref: The root ref of the graph to save (or ref ID string)
            path: File path to write to

        Example:
            session.save_graph(result_ref, "analysis.json")
        """
        import json
        from pathlib import Path

        from ._serialize import serialize_graph

        if isinstance(ref, str):
            ref = self.get(ref)

        data = serialize_graph(ref)
        Path(path).write_text(json.dumps(data, indent=2))

    def load_graph(self, path: str) -> Ref[Any]:
        """
        Load a computation graph from a JSON file into this session.

        The loaded refs are automatically registered in this session.

        Args:
            path: File path to read from

        Returns:
            The root Ref with all dependencies registered in this session

        Example:
            ref = session.load_graph("analysis.json")
            result = session.resolve(ref)
        """
        import json
        from pathlib import Path

        from ._serialize import deserialize_graph

        data = json.loads(Path(path).read_text())
        root = deserialize_graph(data)

        def register_refs(r: Ref[Any]) -> None:
            if r.id in self._refs:
                return
            for parent in r.parents:
                register_refs(parent)
            self._refs[r.id] = r

        register_refs(root)
        return root

    def __repr__(self) -> str:
        return f"Session({self.id!r}, refs={len(self._refs)})"
