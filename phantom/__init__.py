"""
Phantom - The semantic-concrete bridge for LLM data pipelines.

Phantom uses session-scoped operations for isolation and concurrency safety.

Example:
    import phantom

    # Create a session
    session = phantom.Session()

    # Register operations with @session.op
    @session.op
    def load(source: str) -> pd.DataFrame:
        return pd.read_parquet(source)

    @session.op
    def filter(data: pd.DataFrame, condition: str) -> pd.DataFrame:
        return data.query(condition)

    # Register custom inspectors with @session.inspector
    @session.inspector(pd.DataFrame)
    def inspect_df(df):
        return {"shape": list(df.shape), "columns": list(df.columns)}

    # Create refs (lazy - nothing executes yet)
    sales = session.ref("load", source="sales.parquet")
    filtered = session.ref("filter", data=sales, condition="amount > 100")

    # Resolve when needed
    df = session.resolve(filtered)

    # Get tools for LLM integration
    tools = session.get_tools()

    # Save and load graphs
    session.save_graph(filtered, "pipeline.json")
    loaded = session.load_graph("pipeline.json")
"""

from ._errors import CycleError, ResolutionError, TypeValidationError
from ._operation_set import OperationSet
from ._ref import Ref
from ._result import ToolResult
from ._session import Session

__all__ = [
    # Core types
    "Ref",
    "ToolResult",
    "Session",
    "OperationSet",
    # Errors
    "ResolutionError",
    "TypeValidationError",
    "CycleError",
]
