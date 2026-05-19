"""OpenAI / OpenAI-compatible provider."""

from __future__ import annotations

from typing import Any

from tokenoptim.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """
    Provider for OpenAI and OpenAI-compatible APIs (Together, Groq, etc.).

    Parameters
    ----------
    api_key : str | None
        Defaults to OPENAI_API_KEY env var.
    model : str
        Defaults to "gpt-4o-mini".
    base_url : str | None
        Override for OpenAI-compatible endpoints (e.g., Groq, Together).

    Example
    -------
    >>> # Groq example
    >>> provider = OpenAIProvider(
    ...     api_key="gsk_...",
    ...     model="llama-3.1-8b-instant",
    ...     base_url="https://api.groq.com/openai/v1",
    ... )
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            ) from e

        self.model = model
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    @property
    def provider_name(self) -> str:
        return "openai"

    def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> dict:
        # Inject system message at the front if provided and not already there
        all_messages = list(messages)
        if system and (not all_messages or all_messages[0].get("role") != "system"):
            all_messages.insert(0, {"role": "system", "content": system})

        response = self._client.chat.completions.create(
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

    def count_tokens(self, messages: list[dict], system: str | None = None) -> int:
        """Approximate token count (tiktoken if available, else char-based)."""
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(self.model)
            text = ""
            if system:
                text += system + " "
            for m in messages:
                text += str(m.get("content", "")) + " "
            return len(enc.encode(text))
        except Exception:
            text = " ".join(str(m.get("content", "")) for m in messages)
            if system:
                text += " " + system
            return len(text) // 4
