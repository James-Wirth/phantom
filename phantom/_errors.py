"""Errors - Custom exceptions with rich context for debugging."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._ref import Ref


class ResolutionError(Exception):
    """
    Error during ref resolution with full lineage context.

    Provides detailed information about which ref failed and the
    chain of refs that led to the failure.

    Attributes:
        ref: The ref that failed to resolve
        chain: List of ref IDs from root to the failing ref
        cause: The underlying exception that caused the failure
    """

    def __init__(
        self,
        message: str,
        ref: Ref,
        chain: list[str],
        cause: Exception | None = None,
    ):
        self.ref = ref
        self.chain = chain
        self.cause = cause
        super().__init__(self._format_message(message))

    def _format_message(self, message: str) -> str:
        chain_str = " -> ".join(self.chain)
        return (
            f"{message}\n"
            f"  Ref: {self.ref.id} (op={self.ref.op})\n"
            f"  Chain: {chain_str}"
        )


class TypeValidationError(ResolutionError):
    """
    Type mismatch during resolution.

    Raised when a resolved value's type doesn't match the expected
    Ref[T] parameter type declared in the operation signature.

    Attributes:
        ref: The ref that produced the mismatched value
        chain: List of ref IDs from root to the failing ref
        expected_type: The type declared in Ref[T]
        actual_type: The actual type of the resolved value
    """

    def __init__(
        self,
        message: str,
        ref: Ref,
        chain: list[str],
        expected_type: type,
        actual_type: type,
    ):
        self.expected_type = expected_type
        self.actual_type = actual_type
        super().__init__(message, ref, chain)

    def _format_message(self, message: str) -> str:
        base = super()._format_message(message)
        return (
            f"{base}\n"
            f"  Expected: {self.expected_type}\n"
            f"  Actual: {self.actual_type}"
        )


class CycleError(Exception):
    """
    Cycle detected in the computation graph.

    Raised when resolution encounters a ref that is already
    being resolved (indicating a circular dependency).
    """

    def __init__(self, ref_id: str, chain: list[str]):
        self.ref_id = ref_id
        self.chain = chain
        chain_str = " -> ".join(chain + [ref_id])
        super().__init__(f"Cycle detected: {chain_str}")
