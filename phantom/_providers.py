"""LLM provider abstractions for Chat integration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._session import Session


@dataclass
class Usage:
    """Token usage from an LLM API call."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CallOptions:
    """Options for ``LLMProvider.call()``.

    Providers read the fields they support and ignore the rest.
    Provider-specific kwargs can be passed via ``extra``.
    """

    client: Any
    model: str
    system: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    max_tokens: int = 4096
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderToolCall:
    """Normalized tool call from any LLM provider."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ProviderResponse:
    """Normalized response from any LLM provider."""

    text: str | None
    tool_calls: list[ProviderToolCall]
    stop_reason: str | None
    usage: Usage
    raw: Any

    @property
    def is_done(self) -> bool:
        """Whether the model finished its turn (no more tool calls)."""
        return self.stop_reason in ("end_turn", "stop")


@runtime_checkable
class LLMProvider(Protocol):
    """Unified interface for LLM provider adapters.

    Implement this protocol to add support for a new LLM provider.
    Each method handles a specific aspect of the provider API:

    - ``create_client``: SDK client instantiation.
    - ``default_model``: Sensible default model selection.
    - ``get_tools``: Tool-schema formatting for this provider.
    - ``call``: API request execution and response normalization.
    - ``format_tool_result_block``: Single tool-result formatting.
    - ``format_tool_results``: Batch tool-result message assembly.
    - ``format_assistant_message``: Response formatting for history.

    Example::

        class MyProvider:
            def create_client(self) -> Any:
                return MySDK()

            def default_model(self) -> str:
                return "my-model-v1"

            def get_tools(self, session):
                return session.get_tools(format="openai")

            def call(self, options: CallOptions) -> ProviderResponse:
                ...

            # ... remaining methods ...

        register_provider("my_provider", MyProvider)
    """

    def create_client(self) -> Any:
        """Create and return an SDK client instance."""
        ...

    def default_model(self) -> str:
        """Return the default model identifier for this provider."""
        ...

    def get_tools(
        self, session: Session
    ) -> list[dict[str, Any]]:
        """Generate tool definitions in this provider's format."""
        ...

    def call(self, options: CallOptions) -> ProviderResponse:
        """Execute an LLM API call and return a normalized response."""
        ...

    def format_tool_result_block(
        self,
        tool_call_id: str,
        result_json: str,
        is_error: bool,
    ) -> dict[str, Any]:
        """Format a single tool result for the provider's API."""
        ...

    def format_tool_results(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Assemble result blocks into provider-specific messages."""
        ...

    def format_assistant_message(
        self,
        response: ProviderResponse,
    ) -> dict[str, Any]:
        """Format the assistant's response for conversation history."""
        ...


class AnthropicProvider:
    """Adapter for the Anthropic Messages API."""

    def create_client(self) -> Any:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic is required for provider='anthropic'. "
                "Install with: pip install anthropic"
            ) from None
        return anthropic.Anthropic()

    def get_tools(
        self, session: Session
    ) -> list[dict[str, Any]]:
        return session.get_tools(format="anthropic")

    def default_model(self) -> str:
        return "claude-sonnet-4-20250514"

    def call(self, options: CallOptions) -> ProviderResponse:
        kwargs: dict[str, Any] = {
            "model": options.model,
            "max_tokens": options.max_tokens,
            "system": options.system,
            "tools": options.tools,
            "messages": options.messages,
        }
        if options.temperature is not None:
            kwargs["temperature"] = options.temperature
        if options.top_p is not None:
            kwargs["top_p"] = options.top_p
        if options.stop_sequences is not None:
            kwargs["stop_sequences"] = options.stop_sequences
        kwargs.update(options.extra)

        response = options.client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ProviderToolCall] = []
        for block in response.content:
            if block.type == "text" and block.text.strip():
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ProviderToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        return ProviderResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            raw=response,
        )

    def format_tool_results(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Single user message containing all tool_result blocks."""
        return [{"role": "user", "content": results}]

    def format_tool_result_block(
        self,
        tool_call_id: str,
        result_json: str,
        is_error: bool,
    ) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": result_json,
            "is_error": is_error,
        }

    def format_assistant_message(
        self, response: ProviderResponse
    ) -> dict[str, Any]:
        return {"role": "assistant", "content": response.raw.content}


class OpenAIProvider:
    """Adapter for the OpenAI Chat Completions API."""

    def create_client(self) -> Any:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai is required for provider='openai'. "
                "Install with: pip install openai"
            ) from None
        return openai.OpenAI()

    def get_tools(
        self, session: Session
    ) -> list[dict[str, Any]]:
        return session.get_tools(format="openai")

    def default_model(self) -> str:
        return "gpt-4o"

    def call(self, options: CallOptions) -> ProviderResponse:
        full_messages = (
            [{"role": "system", "content": options.system}]
            + options.messages
        )

        kwargs: dict[str, Any] = {
            "model": options.model,
            "max_tokens": options.max_tokens,
            "tools": options.tools,
            "messages": full_messages,
        }
        if options.temperature is not None:
            kwargs["temperature"] = options.temperature
        if options.top_p is not None:
            kwargs["top_p"] = options.top_p
        if options.stop_sequences is not None:
            kwargs["stop"] = options.stop_sequences
        kwargs.update(options.extra)

        response = options.client.chat.completions.create(**kwargs)

        message = response.choices[0].message
        tool_calls: list[ProviderToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ProviderToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(
                            tc.function.arguments
                        ),
                    )
                )

        raw_usage = response.usage
        usage = Usage(
            input_tokens=(
                raw_usage.prompt_tokens if raw_usage else 0
            ),
            output_tokens=(
                raw_usage.completion_tokens
                if raw_usage
                else 0
            ),
        )

        return ProviderResponse(
            text=message.content if message.content else None,
            tool_calls=tool_calls,
            stop_reason=response.choices[0].finish_reason,
            usage=usage,
            raw=response,
        )

    def format_tool_results(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Each tool result is a separate message."""
        return results

    def format_tool_result_block(
        self,
        tool_call_id: str,
        result_json: str,
        is_error: bool,
    ) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result_json,
        }

    def format_assistant_message(
        self, response: ProviderResponse
    ) -> dict[str, Any]:
        message = response.raw.choices[0].message
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": message.content,
        }
        if message.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        return msg


# --- Provider registry ---

_PROVIDERS: dict[str, Callable[[], LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}


def register_provider(
    name: str, cls: Callable[[], LLMProvider]
) -> None:
    """Register a custom LLM provider.

    Args:
        name: Provider name for use in ``Chat(provider=name)``.
        cls: A class (or factory) producing an ``LLMProvider``.
    """
    _PROVIDERS[name] = cls


def get_provider(name: str) -> LLMProvider:
    """Get a provider instance by name.

    Args:
        name: One of the registered provider names.

    Raises:
        ValueError: If the provider name is not registered.
    """
    if name not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider: {name!r}. "
            f"Available: {list(_PROVIDERS.keys())}"
        )
    return _PROVIDERS[name]()
