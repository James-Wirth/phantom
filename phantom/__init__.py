"""
Phantom - The semantic-concrete bridge for LLM data pipelines.

Phantom uses a session-first architecture for isolation and concurrency safety.

Example:
    import phantom

    # Developer defines operations (global, stateless)
    @phantom.op
    def load(source: str) -> pd.DataFrame:
        return pd.read_parquet(source)

    @phantom.op
    def filter(data: pd.DataFrame, condition: str) -> pd.DataFrame:
        return data.query(condition)

    # Create a session for ref management
    session = phantom.Session()

    # Create refs (lazy - nothing executes)
    sales = session.ref("load", source="sales.parquet")
    filtered = session.ref("filter", data=sales, condition="amount > 100")

    # Resolve when needed
    df = session.resolve(filtered)
"""

from ._errors import CycleError, ResolutionError
from ._inspect import inspector
from ._ref import Ref
from ._registry import (
    get_operation_signature,
    get_tools,
    list_operations,
    op,
    validate_args,
)
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
    # Global utilities
    "list_operations",
    "get_operation_signature",
    "get_tools",
    "validate_args",
    # Serialization
    "serialize_graph",
    "deserialize_graph",
    "save_graph",
    "load_graph",
]
