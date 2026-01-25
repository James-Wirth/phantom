"""Serialize - Internal utilities for graph serialization."""

from __future__ import annotations

from typing import Any

from ._ref import Ref


def serialize_graph(root: Ref[Any]) -> dict[str, Any]:
    """
    Serialize a ref and all its dependencies to a JSON-compatible dict.

    This is an internal utility used by Session.save_graph().

    Args:
        root: The root ref of the graph to serialize

    Returns:
        A dict with 'root', 'refs', and 'version' keys
    """
    refs: dict[str, dict[str, Any]] = {}

    def collect(ref: Ref[Any]) -> None:
        if ref.id in refs:
            return

        serialized_args: dict[str, Any] = {}
        for key, value in ref.args.items():
            if isinstance(value, Ref):
                serialized_args[key] = value.id
                collect(value)
            else:
                serialized_args[key] = value

        refs[ref.id] = {
            "id": ref.id,
            "op": ref.op,
            "args": serialized_args,
            "meta": ref.meta,
        }

    collect(root)

    return {
        "version": "1.0",
        "root": root.id,
        "refs": refs,
    }


def deserialize_graph(data: dict[str, Any]) -> Ref[Any]:
    """
    Reconstruct a ref graph from serialized form.

    This is an internal utility used by Session.load_graph().

    Args:
        data: The serialized graph dict (from serialize_graph)

    Returns:
        The root Ref with all dependencies reconstructed

    Raises:
        ValueError: If the data format is invalid
    """
    if "version" not in data or "root" not in data or "refs" not in data:
        raise ValueError("Invalid serialized graph: missing required keys")

    refs_data = data["refs"]
    root_id = data["root"]

    built_refs: dict[str, Ref[Any]] = {}

    def build(ref_id: str) -> Ref[Any]:
        if ref_id in built_refs:
            return built_refs[ref_id]

        if ref_id not in refs_data:
            raise ValueError(f"Invalid serialized graph: missing ref {ref_id}")

        ref_data = refs_data[ref_id]

        resolved_args: dict[str, Any] = {}
        for key, value in ref_data["args"].items():
            if isinstance(value, str) and value.startswith("@"):
                resolved_args[key] = build(value)
            else:
                resolved_args[key] = value

        ref = Ref(
            op=ref_data["op"],
            args=resolved_args,
            meta=ref_data.get("meta", {}),
            id=ref_data["id"],
        )
        built_refs[ref_id] = ref
        return ref

    return build(root_id)
