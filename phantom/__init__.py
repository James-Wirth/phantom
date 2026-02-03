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

from ._chat import Chat, ChatResponse
from ._errors import CycleError, MaxTurnsError, ResolutionError, TypeValidationError
from ._operation_set import OperationSet
from ._providers import (
    AnthropicProvider,
    CallOptions,
    GoogleProvider,
    LLMProvider,
    OpenAIProvider,
    ProviderResponse,
    ProviderToolCall,
    Usage,
    get_provider,
    register_provider,
)
from ._ref import Ref
from ._result import ToolResult
from ._security import (
    DEFAULT_DENY_PATTERNS,
    FileSizeGuard,
    Guard,
    PathGuard,
    SecurityError,
    SecurityPolicy,
)
from ._session import Session

__all__ = [
    # Core types
    "Ref",
    "ToolResult",
    "Session",
    "OperationSet",
    # LLM interface
    "Chat",
    "ChatResponse",
    # Provider interface
    "LLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "GoogleProvider",
    "CallOptions",
    "Usage",
    "ProviderResponse",
    "ProviderToolCall",
    "get_provider",
    "register_provider",
    # Security
    "DEFAULT_DENY_PATTERNS",
    "SecurityError",
    "SecurityPolicy",
    "Guard",
    "PathGuard",
    "FileSizeGuard",
    # Errors
    "ResolutionError",
    "TypeValidationError",
    "CycleError",
    "MaxTurnsError",
]
