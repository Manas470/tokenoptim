"""Async OpenAI / OpenAI-compatible provider with streaming support."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from tokenoptim.providers.base import BaseProvider


class AsyncOpenAIProvider(BaseProvider):
    """
    Async provider for OpenAI and OpenAI-compatible APIs.

    Example
    -------
    >>> provider = AsyncOpenAIProvider(model="gpt-4o-mini")
    >>> resp = await provider.async_chat([{"role": "user", "content": "Hello"}])
    >>>
    >>> async for chunk in provider.stream(messages):
    ...     print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("Run: pip install openai") from e

        self.model = model
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)

    @property
    def provider_name(self) -> str:
        return "openai-async"

    def chat(self, messages, system=None, max_tokens=1024, **kwargs):
        raise RuntimeError("Use `await async_chat(...)` for async providers.")

    async def async_chat(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> dict:
        all_messages = self._inject_system(messages, system)
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            max_tokens=max_tokens,
            **kwargs,
        )
        usage = response.usage
        return {
            "content": response.choices[0].message.content or "",
            "input_tokens": getattr(usage, "prompt_tokens", 0),
            "output_tokens": getattr(usage, "completion_tokens", 0),
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
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
        """Yield response text token-by-token."""
        all_messages = self._inject_system(messages, system)
        async with await self._client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    def count_tokens(self, messages, system=None) -> int:
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(self.model)
            text = (system or "") + " " + " ".join(str(m.get("content", "")) for m in messages)
            return len(enc.encode(text))
        except Exception:
            text = " ".join(str(m.get("content", "")) for m in messages)
            return len(text) // 4

    def _inject_system(self, messages: list[dict], system: str | None) -> list[dict]:
        all_messages = list(messages)
        if system and (not all_messages or all_messages[0].get("role") != "system"):
            all_messages.insert(0, {"role": "system", "content": system})
        return all_messages
