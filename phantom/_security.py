"""Security policy infrastructure for Phantom operations.

Provides composable guards that validate operation arguments before execution,
preventing path traversal, SSRF, ReDoS, and resource exhaustion attacks.

Usage:
    from phantom import Session, SecurityPolicy, PathGuard, URLGuard

    policy = SecurityPolicy()
    policy.bind(PathGuard(allowed_dirs=["/data"]), ops=["read_text"], args=["path"])

    session = Session(policy=policy)
"""

from __future__ import annotations

import fnmatch
import ipaddress
import os
import socket
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_DENY_PATTERNS: list[str] = [
    "*.env",
    ".git",
    ".ssh",
    "*.pem",
    "*.key",
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    ".aws",
    "credentials*",
    ".netrc",
    ".npmrc",
    "*.secret",
    "*.secrets",
]


class SecurityError(ValueError):
    """Raised when a security guard blocks an operation.

    Attributes:
        op_name: The operation that was blocked.
        arg_name: The argument that failed validation.
        guard_name: The guard class that triggered the block.
    """

    def __init__(
        self,
        message: str,
        *,
        op_name: str,
        arg_name: str,
        guard_name: str,
    ) -> None:
        self.op_name = op_name
        self.arg_name = arg_name
        self.guard_name = guard_name
        super().__init__(
            f"[{guard_name}] Blocked {op_name}(…{arg_name}=…): {message}"
        )


# =============================================================================
# Guard base class
# =============================================================================


class Guard(ABC):
    """Base class for security guards.

    Subclasses implement ``check()`` to validate a single argument value.
    Raise ``SecurityError`` to block the operation.
    """

    @abstractmethod
    def check(self, value: Any, *, op_name: str, arg_name: str) -> None:
        """Validate *value*. Raise ``SecurityError`` if not allowed."""


# =============================================================================
# Built-in guards
# =============================================================================


class PathGuard(Guard):
    """Restrict file paths to allowed directories.

    Resolves paths to absolute form (following symlinks) and verifies they
    fall within at least one of the *allowed_dirs*. Optionally rejects paths
    matching *deny_patterns* (fnmatch globs checked against all path
    components, not just the filename).

    Args:
        allowed_dirs: Directories the operation may access. When ``None``,
            no directory restriction is applied (deny patterns and other
            guards still run).
        deny_patterns: Glob patterns checked against every path component
            (e.g. ``["*.env", ".git"]`` blocks both ``prod.env`` and
            ``.git/config``).
    """

    def __init__(
        self,
        allowed_dirs: list[str | Path] | None = None,
        deny_patterns: list[str] | None = None,
    ) -> None:
        self._allowed = (
            [Path(d).resolve() for d in allowed_dirs] if allowed_dirs else []
        )
        self._deny = deny_patterns or []

    def check(self, value: Any, *, op_name: str, arg_name: str) -> None:
        try:
            resolved = Path(value).resolve()
        except (TypeError, OSError) as exc:
            raise SecurityError(
                f"Invalid path: {exc}",
                op_name=op_name,
                arg_name=arg_name,
                guard_name="PathGuard",
            ) from exc

        for pattern in self._deny:
            for component in resolved.parts:
                if fnmatch.fnmatch(component, pattern):
                    raise SecurityError(
                        f"Path matches deny pattern '{pattern}': {value}",
                        op_name=op_name,
                        arg_name=arg_name,
                        guard_name="PathGuard",
                    )

        if not self._allowed:
            return

        for allowed in self._allowed:
            try:
                resolved.relative_to(allowed)
                return
            except ValueError:
                continue

        dirs = ", ".join(str(d) for d in self._allowed)
        raise SecurityError(
            f"Path '{value}' is outside allowed directories: [{dirs}]",
            op_name=op_name,
            arg_name=arg_name,
            guard_name="PathGuard",
        )


class URLGuard(Guard):
    """Prevent SSRF by restricting URLs.

    Validates URL scheme, hostname, and optionally resolves the host to
    check for private/loopback IP addresses.

    Relative paths (no scheme and no host, e.g. ``"/users"``) are allowed
    through without validation — they are used by client-based operations
    where the base URL was already validated at client creation time.

    Args:
        allowed_hosts: If set, only these hosts are permitted.
        blocked_hosts: Hosts that are always blocked.
        allowed_schemes: Permitted URL schemes (default: ``["https", "http"]``).
        block_private: Block private/loopback IPs (default: ``True``).
        resolve_dns: When ``True`` (the default), resolve hostnames via DNS
            and check the resolved IPs against private/loopback ranges.
            This closes the DNS-rebinding SSRF bypass where a hostname
            resolves to an internal IP.
    """

    _METADATA_IPS = frozenset({"169.254.169.254", "fd00:ec2::254"})

    def __init__(
        self,
        allowed_hosts: list[str] | None = None,
        blocked_hosts: list[str] | None = None,
        allowed_schemes: list[str] | None = None,
        block_private: bool = True,
        resolve_dns: bool = True,
    ) -> None:
        self._allowed_hosts = (
            {h.lower() for h in allowed_hosts} if allowed_hosts else None
        )
        self._blocked_hosts = (
            {h.lower() for h in blocked_hosts} if blocked_hosts else set()
        )
        self._allowed_schemes = set(allowed_schemes or ["https", "http"])
        self._block_private = block_private
        self._resolve_dns = resolve_dns

    def check(self, value: Any, *, op_name: str, arg_name: str) -> None:
        try:
            parsed = urlparse(str(value))
        except Exception as exc:
            raise SecurityError(
                f"Invalid URL: {exc}",
                op_name=op_name,
                arg_name=arg_name,
                guard_name="URLGuard",
            ) from exc

        if not parsed.scheme and not parsed.netloc:
            return

        if parsed.scheme not in self._allowed_schemes:
            raise SecurityError(
                f"URL scheme '{parsed.scheme}' is not allowed "
                f"(allowed: {sorted(self._allowed_schemes)})",
                op_name=op_name,
                arg_name=arg_name,
                guard_name="URLGuard",
            )

        hostname = (parsed.hostname or "").lower()

        if hostname in self._blocked_hosts:
            raise SecurityError(
                f"Host '{hostname}' is blocked",
                op_name=op_name,
                arg_name=arg_name,
                guard_name="URLGuard",
            )

        if self._allowed_hosts is not None and hostname not in self._allowed_hosts:
            raise SecurityError(
                f"Host '{hostname}' is not in the allowed hosts list",
                op_name=op_name,
                arg_name=arg_name,
                guard_name="URLGuard",
            )

        if self._block_private:
            self._check_private(
                hostname, op_name=op_name, arg_name=arg_name
            )

    def _check_private(
        self, hostname: str, *, op_name: str, arg_name: str
    ) -> None:
        """Block private, loopback, and metadata IPs."""
        if hostname in self._METADATA_IPS:
            raise SecurityError(
                f"Host '{hostname}' is a cloud metadata endpoint",
                op_name=op_name,
                arg_name=arg_name,
                guard_name="URLGuard",
            )

        try:
            addr = ipaddress.ip_address(hostname)
        except ValueError:
            if self._resolve_dns:
                self._check_dns(
                    hostname, op_name=op_name, arg_name=arg_name
                )
            return

        if addr.is_loopback or addr.is_private or addr.is_reserved:
            raise SecurityError(
                f"Host '{hostname}' resolves to a private/loopback "
                "address",
                op_name=op_name,
                arg_name=arg_name,
                guard_name="URLGuard",
            )

    def _check_dns(
        self, hostname: str, *, op_name: str, arg_name: str
    ) -> None:
        """Resolve *hostname* via DNS and block private/metadata IPs."""
        try:
            infos = socket.getaddrinfo(
                hostname, None, proto=socket.IPPROTO_TCP
            )
        except socket.gaierror:
            return

        for info in infos:
            resolved_ip = info[4][0]
            if resolved_ip in self._METADATA_IPS:
                raise SecurityError(
                    f"Host '{hostname}' resolves to a cloud metadata "
                    f"endpoint ({resolved_ip})",
                    op_name=op_name,
                    arg_name=arg_name,
                    guard_name="URLGuard",
                )
            try:
                addr = ipaddress.ip_address(resolved_ip)
            except ValueError:
                continue
            if addr.is_loopback or addr.is_private or addr.is_reserved:
                raise SecurityError(
                    f"Host '{hostname}' resolves to a private/loopback "
                    f"address ({resolved_ip})",
                    op_name=op_name,
                    arg_name=arg_name,
                    guard_name="URLGuard",
                )


class FileSizeGuard(Guard):
    """Prevent reading oversized files.

    Checks the file size (via ``os.path.getsize``) before the read operation
    executes.

    Args:
        max_bytes: Maximum file size in bytes (default: 50 MB).
    """

    def __init__(self, max_bytes: int = 50_000_000) -> None:
        self._max_bytes = max_bytes

    def check(self, value: Any, *, op_name: str, arg_name: str) -> None:
        path = str(value)
        try:
            size = os.path.getsize(path)
        except OSError:
            return

        if size > self._max_bytes:
            mb = self._max_bytes / (1024 * 1024)
            actual_mb = size / (1024 * 1024)
            raise SecurityError(
                f"File size ({actual_mb:.1f} MB) exceeds limit ({mb:.1f} MB)",
                op_name=op_name,
                arg_name=arg_name,
                guard_name="FileSizeGuard",
            )


# =============================================================================
# SecurityPolicy
# =============================================================================


class SecurityPolicy:
    """Composable security policy that binds guards to operations.

    Guards are checked at resolution time, before the operation executes.
    If any guard raises ``SecurityError``, the operation is blocked.

    Example:
        policy = SecurityPolicy()
        policy.bind(PathGuard(["/data"]), ops=["read_text"], args=["path"])
        policy.bind(RegexGuard(), ops=["search_text"], args=["pattern"])

        session = Session(policy=policy)
    """

    def __init__(self) -> None:
        self._bindings: list[tuple[Guard, frozenset[str], frozenset[str]]] = []

    def bind(
        self,
        guard: Guard,
        *,
        ops: list[str],
        args: list[str],
    ) -> SecurityPolicy:
        """Bind a guard to specific operations and argument names.

        Args:
            guard: The guard instance to apply.
            ops: Operation names this guard applies to.
            args: Argument names this guard validates.

        Returns:
            self (for method chaining).
        """
        self._bindings.append((guard, frozenset(ops), frozenset(args)))
        return self

    def check(self, op_name: str, arguments: dict[str, Any]) -> None:
        """Run all applicable guards against the given arguments.

        Raises:
            SecurityError: If any guard blocks the operation.
        """
        for guard, ops, arg_names in self._bindings:
            if op_name not in ops:
                continue
            for arg_name in arg_names:
                if arg_name in arguments:
                    guard.check(
                        arguments[arg_name],
                        op_name=op_name,
                        arg_name=arg_name,
                    )

    def __or__(self, other: SecurityPolicy) -> SecurityPolicy:
        """Merge two policies: ``combined = policy_a | policy_b``."""
        merged = SecurityPolicy()
        merged._bindings = list(self._bindings) + list(other._bindings)
        return merged

    def __repr__(self) -> str:
        guards = [type(g).__name__ for g, _, _ in self._bindings]
        return f"SecurityPolicy(guards={guards})"


__all__ = [
    "DEFAULT_DENY_PATTERNS",
    "SecurityError",
    "Guard",
    "PathGuard",
    "URLGuard",
    "FileSizeGuard",
    "SecurityPolicy",
]
