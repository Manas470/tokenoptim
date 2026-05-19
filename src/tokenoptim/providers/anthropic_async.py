"""Async Anthropic Claude provider with streaming support."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from tokenoptim.providers.base import BaseProvider


class AsyncAnthropicProvider(BaseProvider):
    """
    Async provider for Anthropic Claude models.

    Supports both full responses and token-by-token streaming.

    Example
    -------
    >>> provider = AsyncAnthropicProvider(model="claude-haiku-4-5-20251001")
    >>>
    >>> # Non-streaming
    >>> resp = await provider.chat([{"role": "user", "content": "Hello"}])
    >>>
    >>> # Streaming
    >>> async for chunk in provider.stream([{"role": "user", "content": "Hello"}]):
    ...     print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        enable_caching: bool = True,
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as e:
            raise ImportError("Run: pip install anthropic") from e

        self._anthropic = _anthropic
        self.model = model
        self.enable_caching = enable_caching
        self._client = _anthropic.AsyncAnthropic(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic-async"

    def chat(self, messages, system=None, max_tokens=1024, **kwargs):
        raise RuntimeError("Use `await async_chat(...)` for async providers.")

    async def async_chat(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> dict:
        """Send a chat request and return a normalised response dict."""
        system_param = self._build_system(system)
        api_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            **kwargs,
        }
        if system_param is not None:
            api_kwargs["system"] = system_param

        response = await self._client.messages.create(**api_kwargs)
        usage = response.usage
        return {
            "content": response.content[0].text if response.content else "",
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0),
            "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", 0),
            "model": response.model,
            "raw": response,
        }

    async def stream(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Yield response text token-by-token as an async generator.

        Usage
        -----
        >>> async for chunk in provider.stream(messages):
        ...     print(chunk, end="", flush=True)
        """
        system_param = self._build_system(system)
        api_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            **kwargs,
        }
        if system_param is not None:
            api_kwargs["system"] = system_param

        async with self._client.messages.stream(**api_kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    def count_tokens(self, messages, system=None) -> int:
        text = " ".join(str(m.get("content", "")) for m in messages)
        if system:
            text += " " + system
        return len(text) // 4

    def _build_system(self, system: str | None):
        if not system:
            return None
        if self.enable_caching:
            return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        return system
