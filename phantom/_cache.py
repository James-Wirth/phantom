"""LRU cache with dependency-aware eviction for Phantom."""

from __future__ import annotations

import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheEntry:
    """A single cached value with metadata."""

    value: Any
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0


class LRUCache:
    """
    LRU cache with dependency-aware eviction.

    When an entry is evicted due to size limits, all entries that depend on it
    (tracked via reverse_deps passed to set()) are also evicted to maintain
    cache consistency.

    Example:
        cache = LRUCache(max_size=100)

        # When setting values, pass reverse_deps for cascade eviction
        cache.set("@abc", value, reverse_deps={"@abc": {"@def", "@ghi"}})

        # Getting a value updates its access time (LRU tracking)
        value = cache.get("@abc")

        # Delete with cascade
        cache.delete("@abc", reverse_deps, cascade=True)
    """

    def __init__(
        self,
        max_size: int | None = None,
        max_bytes: int | None = None,
    ):
        """
        Create an LRU cache.

        Args:
            max_size: Maximum number of entries. None = unlimited.
            max_bytes: Maximum total size in bytes. None = unlimited.
        """
        self._max_size = max_size
        self._max_bytes = max_bytes
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._current_bytes = 0

    def get(self, key: str) -> Any:
        """
        Get a value, updating its access time (moves to end for LRU).

        Args:
            key: The cache key

        Returns:
            The cached value

        Raises:
            KeyError: If key not in cache
        """
        if key not in self._entries:
            raise KeyError(key)
        self._entries.move_to_end(key)
        entry = self._entries[key]
        entry.last_access = time.time()
        return entry.value

    def __contains__(self, key: str) -> bool:
        """Check if key exists (does NOT update access time)."""
        return key in self._entries

    def set(
        self,
        key: str,
        value: Any,
        reverse_deps: dict[str, set[str]],
    ) -> set[str]:
        """
        Set a value, potentially evicting LRU entries.

        Args:
            key: The cache key
            value: The value to cache
            reverse_deps: Dependency map for cascade eviction.
                         Maps ref_id -> set of dependent ref_ids.

        Returns:
            Set of evicted keys (for logging/notification)
        """
        evicted: set[str] = set()

        size_bytes = self._estimate_size(value) if self._max_bytes else 0

        if key in self._entries:
            old_entry = self._entries[key]
            self._current_bytes -= old_entry.size_bytes

        while self._needs_eviction(size_bytes, exclude_key=key):
            evicted_keys = self._evict_lru(reverse_deps, exclude_key=key)
            if evicted_keys:
                evicted.update(evicted_keys)
            else:
                break

        self._entries[key] = CacheEntry(value=value, size_bytes=size_bytes)
        self._entries.move_to_end(key)
        self._current_bytes += size_bytes
        return evicted

    def _needs_eviction(
            self,
            incoming_size: int,
            exclude_key: str | None = None
        ) -> bool:
        """Check if we need to evict before adding new entry."""
        entry_count = len(self._entries)
        if exclude_key and exclude_key in self._entries:
            entry_count -= 1

        if self._max_size is not None and entry_count >= self._max_size:
            return True
        if self._max_bytes is not None:
            projected_bytes = self._current_bytes + incoming_size
            if projected_bytes > self._max_bytes:
                return True
        return False

    def _evict_lru(
        self,
        reverse_deps: dict[str, set[str]],
        exclude_key: str | None = None,
    ) -> set[str]:
        """
        Evict the least recently used entry and its dependents.

        Args:
            reverse_deps: Dependency map for cascade eviction
            exclude_key: Key to exclude from eviction (the one being set)

        Returns:
            Set of evicted keys
        """
        if not self._entries:
            return set()

        lru_key = None
        for key in self._entries:
            if key != exclude_key:
                lru_key = key
                break

        if lru_key is None:
            return set()

        evicted: set[str] = set()
        self._cascade_evict(lru_key, reverse_deps, evicted)
        return evicted

    def _cascade_evict(
        self,
        key: str,
        reverse_deps: dict[str, set[str]],
        evicted: set[str],
    ) -> None:
        """Evict a key and all its dependents recursively."""
        if key not in self._entries or key in evicted:
            return

        for dep_key in reverse_deps.get(key, set()):
            self._cascade_evict(dep_key, reverse_deps, evicted)

        if key in self._entries:
            entry = self._entries.pop(key)
            self._current_bytes -= entry.size_bytes
            evicted.add(key)

    def delete(
        self,
        key: str,
        reverse_deps: dict[str, set[str]],
        cascade: bool = True,
    ) -> int:
        """
        Delete an entry, optionally cascading to dependents.

        Args:
            key: The cache key to delete
            reverse_deps: Dependency map for cascade deletion
            cascade: If True, also delete all dependent entries

        Returns:
            Number of entries deleted
        """
        if key not in self._entries:
            return 0

        deleted: set[str] = set()
        if cascade:
            self._cascade_evict(key, reverse_deps, deleted)
        else:
            entry = self._entries.pop(key)
            self._current_bytes -= entry.size_bytes
            deleted.add(key)

        return len(deleted)

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()
        self._current_bytes = 0

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._entries)

    def keys(self) -> list[str]:
        """Return list of cached keys."""
        return list(self._entries.keys())

    def _estimate_size(self, value: Any) -> int:
        """
        Estimate memory size of a value.

        Uses specialized methods for common data types (DataFrames, arrays),
        falls back to sys.getsizeof for other types.
        """
        try:
            # Pandas
            if hasattr(value, "memory_usage"):
                usage = value.memory_usage(deep=True)
                if hasattr(usage, "sum"):
                    return int(usage.sum())
                return int(usage)

            # Polars
            if hasattr(value, "estimated_size"):
                return value.estimated_size()

            # Numpy
            if hasattr(value, "nbytes"):
                return value.nbytes

            return sys.getsizeof(value)
        except (TypeError, AttributeError, ValueError, OSError):
            return 0
