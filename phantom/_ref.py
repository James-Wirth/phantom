"""Ref - Semantic reference for LLM data pipelines."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Ref(Generic[T]):
    """
    A semantic reference - a node in a lazy computation graph.

    Refs bridge semantic space (where LLMs reason) and concrete space
    (where code executes). The LLM works with refs as opaque handles;
    resolution executes the actual computation.

    Attributes:
        id: Unique identifier (e.g., "@a3f2")
        op: Operation name that creates this ref
        args: Arguments to the operation (may include other Refs)
        meta: Metadata for LLM context (type info, descriptions, etc.)
    """

    op: str
    args: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"@{uuid.uuid4().hex[:6]}")

    @property
    def parents(self) -> tuple[Ref[Any], ...]:
        """Extract parent refs from args (for lineage tracking)."""
        return tuple(v for v in self.args.values() if isinstance(v, Ref))

    def to_dict(self) -> dict[str, Any]:
        """Serialize for LLM consumption."""
        return {
            "ref": self.id,
            "op": self.op,
            "parents": [p.id for p in self.parents],
            "meta": self.meta,
        }

    def __repr__(self) -> str:
        return self.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Ref):
            return self.id == other.id
        return False
