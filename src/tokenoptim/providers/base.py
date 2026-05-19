"""Abstract base class for LLM providers."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseProvider(ABC):
    """
    Common interface all providers must implement.

    Providers wrap underlying SDK clients and return a normalised response dict:
    {
        "content": str,           # The text response
        "input_tokens": int,
        "output_tokens": int,
        "cache_read_tokens": int,
        "cache_write_tokens": int,
        "model": str,
        "raw": Any,               # Original SDK response object
    }
    """

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> dict:
        """Send a chat request and return a normalised response dict."""
        ...

    @abstractmethod
    def count_tokens(self, messages: list[dict], system: Optional[str] = None) -> int:
        """Return approximate token count for the given messages."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...
