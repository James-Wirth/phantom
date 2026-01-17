"""
Phantom - The semantic-concrete bridge for LLM data pipelines.

Phantom separates concerns:
- LLM works only with refs (semantic handles)
- Developer implements only concrete operations
- Phantom bridges them via a lazy computation graph

Example:
    import phantom

    # Developer defines operations
    @phantom.op
    def load(source: str) -> pd.DataFrame:
        return pd.read_parquet(source)

    @phantom.op
    def filter(data: pd.DataFrame, condition: str) -> pd.DataFrame:
        return data.query(condition)

    # Create refs (lazy - nothing executes)
    sales = phantom.ref("load", source="sales.parquet")
    filtered = phantom.ref("filter", data=sales, condition="amount > 100")

    # Resolve when needed
    df = phantom.resolve(filtered)
"""

from typing import Any, TypeVar

from ._ref import Ref
from ._registry import (
    clear,
    get_operation,
    get_operation_signature,
    get_ref,
    list_operations,
    list_refs,
    op,
    register_ref,
)
from ._resolve import resolve

__all__ = [
    # Core types
    "Ref",
    # Decorators
    "op",
    # Functions
    "ref",
    "resolve",
    "get",
    # Registry
    "list_operations",
    "list_refs",
    "get_operation_signature",
    "clear",
]

T = TypeVar("T")


def ref(op_name: str, **kwargs: Any) -> Ref[Any]:
    """
    Create a ref for an operation.

    This builds a node in the computation graph. Nothing executes
    until resolve() is called.

    Args:
        op_name: Name of the registered operation
        **kwargs: Arguments to pass to the operation

    Returns:
        A Ref that can be resolved later

    Example:
        sales = phantom.ref("load", source="sales.parquet")
        filtered = phantom.ref("filter", data=sales, condition="x > 0")
    """

    get_operation(op_name)

    new_ref: Ref[Any] = Ref(op=op_name, args=kwargs)
    return register_ref(new_ref)


def get(ref_id: str) -> Ref[Any]:
    """
    Get a ref by its ID.

    Args:
        ref_id: The ref ID (e.g., "@a3f2")

    Returns:
        The Ref object

    Example:
        ref = phantom.get("@a3f2")
    """
    return get_ref(ref_id)
