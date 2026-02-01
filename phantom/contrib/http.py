"""
HTTP / API orchestration operations for Phantom.

Provides HTTP request operations for LLM-driven API workflows.
The LLM chains API calls without seeing verbose JSON responses â
caching prevents redundant requests within a session.

Installation:
    pip install phantom[http]

Usage:
    from phantom import Session
    from phantom.contrib.http import http_ops

    session = Session()
    session.register(http_ops)  # registers ops AND inspector

    # One-shot requests
    response = session.ref("http_get", url="https://api.example.com/data")
    items = session.ref("extract_json", response=response, path="body.data.items")

    # Reusable client with shared auth
    client = session.ref("create_client",
                         base_url="https://api.example.com",
                         headers={"Authorization": "Bearer token123"})
    users = session.ref("client_get", client=client, path="/users")
"""

# mypy: ignore-errors
# Rationale: This module uses runtime-imported httpx types which mypy cannot
# properly analyze. The dynamic import via require_dependency() is necessary
# for the optional dependency pattern (pip install phantom[http]).

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from phantom import OperationSet, Ref
from phantom._security import SecurityPolicy, URLGuard

from ._base import require_dependency

httpx = require_dependency("http", "httpx", "http")
Client = httpx.Client

_HTTP_URL_OPS = ["http_get", "http_post", "http_put", "http_delete"]
_HTTP_CLIENT_OPS = ["client_get", "client_post"]


def _default_http_policy() -> SecurityPolicy:
    """Build the default security policy for HTTP operations."""
    guard = URLGuard(block_private=True, resolve_dns=True)
    policy = SecurityPolicy()
    policy.bind(guard, ops=_HTTP_URL_OPS, args=["url"])
    policy.bind(guard, ops=["create_client"], args=["base_url"])
    policy.bind(guard, ops=_HTTP_CLIENT_OPS, args=["path"])
    return policy


http_ops = OperationSet(default_policy=_default_http_policy())


# =============================================================================
# Internal Helpers
# =============================================================================


def _reject_absolute_url(path: str) -> None:
    """Reject absolute URLs in client operation paths."""
    parsed = urlparse(path)
    if parsed.scheme and parsed.netloc:
        raise ValueError(
            f"Absolute URLs are not allowed in client operations "
            f"(got '{path}'). Use http_get/http_post for absolute "
            f"URLs, or use a relative path with a client."
        )


# =============================================================================
# Response Helpers
# =============================================================================


def _build_response(response: Any) -> dict[str, Any]:
    """Convert an httpx Response to a serializable dict.

    Response format:
        {"status": int, "headers": dict, "body": <parsed JSON or text>, "url": str}
    """
    headers = dict(response.headers)
    content_type = response.headers.get("content-type", "")

    body: Any
    if "application/json" in content_type:
        try:
            body = response.json()
        except Exception:
            body = response.text
    else:
        body = response.text

    return {
        "status": response.status_code,
        "headers": headers,
        "body": body,
        "url": str(response.url),
    }


# =============================================================================
# Stateless HTTP Operations
# =============================================================================


@http_ops.op
def http_get(
    url: str,
    headers: dict | None = None,
    params: dict | None = None,
) -> dict:
    """Send an HTTP GET request.

    Args:
        url: Request URL.
        headers: Optional request headers.
        params: Optional query parameters.

    Returns:
        Response dict with keys: status, headers, body, url.
    """
    response = httpx.get(url, headers=headers, params=params)
    return _build_response(response)


@http_ops.op
def http_post(
    url: str,
    body: dict | None = None,
    headers: dict | None = None,
) -> dict:
    """Send an HTTP POST request with optional JSON body.

    Args:
        url: Request URL.
        body: Optional JSON body.
        headers: Optional request headers.

    Returns:
        Response dict with keys: status, headers, body, url.
    """
    response = httpx.post(url, json=body, headers=headers)
    return _build_response(response)


@http_ops.op
def http_put(
    url: str,
    body: dict | None = None,
    headers: dict | None = None,
) -> dict:
    """Send an HTTP PUT request with optional JSON body.

    Args:
        url: Request URL.
        body: Optional JSON body.
        headers: Optional request headers.

    Returns:
        Response dict with keys: status, headers, body, url.
    """
    response = httpx.put(url, json=body, headers=headers)
    return _build_response(response)


@http_ops.op
def http_delete(
    url: str,
    headers: dict | None = None,
) -> dict:
    """Send an HTTP DELETE request.

    Args:
        url: Request URL.
        headers: Optional request headers.

    Returns:
        Response dict with keys: status, headers, body, url.
    """
    response = httpx.delete(url, headers=headers)
    return _build_response(response)


# =============================================================================
# Response Extraction
# =============================================================================


@http_ops.op
def extract_json(response: Ref[dict], path: str) -> Any:
    """Extract a value from a response dict using a dot-separated path.

    Navigates nested dicts and lists. Numeric path segments index into lists.

    Examples:
        extract_json(response, "body.data") -> response["body"]["data"]
        extract_json(response, "body.items.0.name")
            -> response["body"]["items"][0]["name"]
        extract_json(response, "status") -> response["status"]

    Args:
        response: Response dict (from an HTTP operation).
        path: Dot-separated path (e.g. "body.data.items").

    Returns:
        The extracted value.
    """
    value: Any = response
    for key in path.split("."):
        if isinstance(value, dict):
            value = value[key]
        elif isinstance(value, list | tuple):
            value = value[int(key)]
        else:
            raise TypeError(
                f"Cannot index into {type(value).__name__} with key '{key}'"
            )
    return value


@http_ops.op
def extract_headers(response: Ref[dict]) -> dict:
    """Extract headers from a response dict.

    Args:
        response: Response dict (from an HTTP operation).

    Returns:
        Dict of response headers.
    """
    return response["headers"]


# =============================================================================
# Client-Based Operations (Stateful Resource)
# =============================================================================


@http_ops.op
def create_client(
    base_url: str,
    headers: dict | None = None,
    timeout: float = 30.0,
) -> Client:
    """Create a reusable HTTP client with shared configuration.

    Useful for APIs requiring authentication or a common base URL.
    The client maintains connection pooling for performance.

    Args:
        base_url: Base URL for all requests (e.g. "https://api.example.com").
        headers: Default headers for all requests (e.g. auth tokens).
        timeout: Request timeout in seconds (default: 30).

    Returns:
        An httpx.Client instance.
    """
    return httpx.Client(
        base_url=base_url,
        headers=headers or {},
        timeout=timeout,
    )


@http_ops.op
def client_get(
    client: Ref[Client],
    path: str,
    params: dict | None = None,
) -> dict:
    """Send a GET request using a client.

    Args:
        client: HTTP client (from create_client).
        path: Request path relative to the client's base_url.
        params: Optional query parameters.

    Returns:
        Response dict with keys: status, headers, body, url.
    """
    _reject_absolute_url(path)
    response = client.get(path, params=params)
    return _build_response(response)


@http_ops.op
def client_post(
    client: Ref[Client],
    path: str,
    body: dict | None = None,
) -> dict:
    """Send a POST request using a client.

    Args:
        client: HTTP client (from create_client).
        path: Request path relative to the client's base_url.
        body: Optional JSON body.

    Returns:
        Response dict with keys: status, headers, body, url.
    """
    _reject_absolute_url(path)
    response = client.post(path, json=body)
    return _build_response(response)


# =============================================================================
# Inspector
# =============================================================================


@http_ops.inspector(Client)
def _inspect_client(client: Client) -> dict[str, Any]:
    """HTTP client inspector for LLM context."""
    timeout = client.timeout
    timeout_seconds = timeout.read if timeout is not None else None

    return {
        "type": "httpx.Client",
        "base_url": str(client.base_url),
        "headers": dict(client.headers),
        "timeout": timeout_seconds,
    }


# =============================================================================
# Public API
# =============================================================================

__all__ = ["http_ops"]
