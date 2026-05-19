"""Anthropic Claude provider."""

from __future__ import annotations

from typing import Any, Optional

from tokenoptim.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    """
    Provider for Anthropic Claude models.

    Parameters
    ----------
    api_key : str | None
        Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
    model : str
        Model to use. Defaults to "claude-sonnet-4-6".
    enable_caching : bool
        Add cache_control headers to system prompt (prompt caching).
        Can reduce input costs by 90% on repeated calls.

    Example
    -------
    >>> provider = AnthropicProvider(model="claude-haiku-4-5-20251001")
    >>> resp = provider.chat([{"role": "user", "content": "Hello"}])
    >>> print(resp["content"])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        enable_caching: bool = True,
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from e

        self._anthropic = _anthropic
        self.model = model
        self.enable_caching = enable_caching
        self._client = _anthropic.Anthropic(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> dict:
        # Build system param with optional cache_control
        system_param = None
        if system:
            if self.enable_caching:
                system_param = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                system_param = system

        api_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            **kwargs,
        }
        if system_param is not None:
            api_kwargs["system"] = system_param

        response = self._client.messages.create(**api_kwargs)

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

    def count_tokens(self, messages: list[dict], system: Optional[str] = None) -> int:
        """Use Anthropic's token counting API."""
        try:
            kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
            if system:
                kwargs["system"] = system
            result = self._client.messages.count_tokens(**kwargs)
            return result.input_tokens
        except Exception:
            # Fallback: approx count
            text = " ".join(str(m.get("content", "")) for m in messages)
            if system:
                text += " " + system
            return len(text) // 4
