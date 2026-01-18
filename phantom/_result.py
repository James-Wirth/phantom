"""Result - Tool call result types for LLM integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._ref import Ref

if TYPE_CHECKING:
    from ._errors import ResolutionError


@dataclass
class ToolResult:
    """
    Result from handling an LLM tool call.

    Encapsulates the response that should be sent back to the LLM,
    with metadata about what kind of operation was performed.

    Attributes:
        kind: Type of result - "ref" for lazy operations, "peek" for inspection
        data: The dict to serialize back to the LLM
        ref: The Ref object (only for kind="ref")
    """

    kind: str
    data: dict[str, Any]
    ref: Ref[Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for LLM consumption."""
        return self.data

    @classmethod
    def from_ref(cls, ref: Ref[Any]) -> ToolResult:
        """Create result for a lazy operation that created a ref."""
        serialized_args = {}
        for k, v in ref.args.items():
            serialized_args[k] = v.id if isinstance(v, Ref) else v

        return cls(
            kind="ref",
            data={"ref": ref.id, "op": ref.op, "args": serialized_args},
            ref=ref,
        )

    @classmethod
    def from_peek(cls, peek_data: dict[str, Any]) -> ToolResult:
        """Create result for a peek inspection."""
        return cls(
            kind="peek",
            data=peek_data,
            ref=None,
        )

    @classmethod
    def from_error(cls, error: ResolutionError) -> ToolResult:
        """Create result for a resolution error."""
        return cls(
            kind="error",
            data={
                "error": True,
                "type": (
                    type(error.cause).__name__ if error.cause else "ResolutionError"
                ),
                "message": str(error.cause) if error.cause else str(error),
                "ref": error.ref.id,
                "op": error.ref.op,
                "chain": error.chain,
            },
            ref=error.ref,
        )
