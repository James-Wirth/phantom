"""Result - Tool call result types for LLM integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._ref import Ref


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
        return cls(
            kind="ref",
            data={"ref": ref.id, "op": ref.op},
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
