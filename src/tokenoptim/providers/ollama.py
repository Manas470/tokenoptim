"""Ollama provider for locally-hosted models."""

from __future__ import annotations

from typing import Any, Optional

from tokenoptim.providers.base import BaseProvider


class OllamaProvider(BaseProvider):
    """
    Provider for Ollama (locally-hosted open-source models).

    Requires Ollama running locally: https://ollama.com

    Parameters
    ----------
    model : str
        Model name, e.g. "llama3.2", "mistral", "phi4".
    host : str
        Ollama host. Defaults to "http://localhost:11434".

    Example
    -------
    >>> provider = OllamaProvider(model="llama3.2")
    >>> resp = provider.chat([{"role": "user", "content": "Explain tokenization"}])
    """

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")

    @property
    def provider_name(self) -> str:
        return "ollama"

    def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> dict:
        import urllib.request
        import json

        all_messages = list(messages)
        if system and (not all_messages or all_messages[0].get("role") != "system"):
            all_messages.insert(0, {"role": "system", "content": system})

        payload = json.dumps({
            "model": self.model,
            "messages": all_messages,
            "stream": False,
            "options": {"num_predict": max_tokens},
            **kwargs,
        }).encode()

        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())

        content = data.get("message", {}).get("content", "")
        prompt_eval = data.get("prompt_eval_count", 0)
        eval_count = data.get("eval_count", 0)

        return {
            "content": content,
            "input_tokens": prompt_eval,
            "output_tokens": eval_count,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "model": self.model,
            "raw": data,
        }

    def count_tokens(self, messages: list[dict], system: Optional[str] = None) -> int:
        text = " ".join(str(m.get("content", "")) for m in messages)
        if system:
            text += " " + system
        return len(text) // 4
