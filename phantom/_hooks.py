"""Hooks - Event system for resolution lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_hooks: dict[str, list[Callable[..., None]]] = {
    "before_resolve": [],
    "after_resolve": [],
    "on_error": [],
}


def on(event: str) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """
    Decorator to register a hook for resolution events.

    Available events:
        - "before_resolve": Called before each operation executes
          Signature: (ref: Ref, args: dict[str, Any]) -> None

        - "after_resolve": Called after each operation completes successfully
          Signature: (ref: Ref, result: Any) -> None

        - "on_error": Called when an operation fails
          Signature: (ref: Ref, error: Exception) -> None

    Example:
        @phantom.on("before_resolve")
        def log_start(ref, args):
            print(f"Starting {ref.op}...")

        @phantom.on("after_resolve")
        def log_done(ref, result):
            print(f"Finished {ref.op}")

        @phantom.on("on_error")
        def log_error(ref, error):
            print(f"Error in {ref.op}: {error}")
    """
    if event not in _hooks:
        raise ValueError(f"Unknown event: {event}. Valid events: {list(_hooks.keys())}")

    def decorator(fn: Callable[..., None]) -> Callable[..., None]:
        _hooks[event].append(fn)
        return fn

    return decorator


def emit(event: str, **kwargs: Any) -> None:
    """
    Emit an event to all registered hooks.

    This is called internally by the resolution engine.
    Users typically don't need to call this directly.

    Args:
        event: The event name
        **kwargs: Event-specific data passed to handlers
    """
    if event not in _hooks:
        return

    for hook in _hooks[event]:
        try:
            hook(**kwargs)
        except Exception:
            pass


def clear_hooks(event: str | None = None) -> None:
    """
    Clear registered hooks.

    Args:
        event: If provided, clear only this event's hooks.
               If None, clear all hooks.
    """
    if event is not None:
        if event in _hooks:
            _hooks[event].clear()
    else:
        for handlers in _hooks.values():
            handlers.clear()


def list_hooks() -> dict[str, int]:
    """
    List the number of registered hooks per event.

    Returns:
        Dict mapping event names to hook counts
    """
    return {event: len(handlers) for event, handlers in _hooks.items()}
