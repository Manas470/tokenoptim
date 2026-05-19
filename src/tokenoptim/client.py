"""
OptimizedClient — the one-stop interface for token-optimized LLM calls.

Combines PromptCompressor + OutputCompressor + MemoryManager + TokenCounter
into a single object that wraps any provider.

Quick start
-----------
>>> from tokenoptim import OptimizedClient
>>> from tokenoptim.providers import AnthropicProvider
>>>
>>> client = OptimizedClient(
...     provider=AnthropicProvider(),
...     compress_prompts=True,
...     output_level="full",
...     memory_enabled=True,
...     memory_max_turns=10,
... )
>>>
>>> resp = client.chat("Explain PySpark in detail")
>>> print(resp["content"])
>>> print(client.counter.report())
"""

from __future__ import annotations

import logging
from typing import Any

from tokenoptim.core.cache import ResponseCache
from tokenoptim.core.compressor import PromptCompressor
from tokenoptim.core.counter import TokenCounter
from tokenoptim.core.memory import MemoryManager
from tokenoptim.core.output_style import CompressionLevel, OutputCompressor
from tokenoptim.core.retry import RetryConfig, with_retry
from tokenoptim.providers.base import BaseProvider

logger = logging.getLogger("tokenoptim")


class OptimizedClient:
    """
    Provider-agnostic LLM client with built-in token optimization.

    Parameters
    ----------
    provider : BaseProvider
        One of AnthropicProvider, OpenAIProvider, OllamaProvider.
    compress_prompts : bool
        Apply PromptCompressor to user messages before sending.
    prompt_compression_level : str
        "light" | "medium" | "full" (default "medium")
    output_level : str | CompressionLevel
        Output compression level (default "standard" ≈ 40% reduction).
    memory_enabled : bool
        Enable conversation memory (default True).
    memory_max_turns : int
        How many turns to keep in the active memory window.
    token_budget : int | None
        Optional token budget for the session.
    system : str | None
        Base system prompt (compression directive is prepended automatically).
    retry : RetryConfig | None
        Retry policy for transient failures. Defaults to 3 retries w/ backoff.
    cache : ResponseCache | None
        Response cache. Pass a ResponseCache instance to avoid duplicate calls.
    """

    def __init__(
        self,
        provider: BaseProvider,
        compress_prompts: bool = True,
        prompt_compression_level: str = "medium",
        output_level: str | CompressionLevel = CompressionLevel.STANDARD,
        memory_enabled: bool = True,
        memory_max_turns: int = 20,
        token_budget: int | None = None,
        system: str | None = None,
        retry: RetryConfig | None = None,
        cache: ResponseCache | None = None,
    ) -> None:
        self.provider = provider
        self._base_system = system or ""
        self._retry = retry or RetryConfig()
        self.cache = cache

        self.prompt_compressor = PromptCompressor(
            level=prompt_compression_level,
        ) if compress_prompts else None

        self.output_compressor = OutputCompressor(level=output_level)

        self.memory = MemoryManager(
            enabled=memory_enabled,
            max_turns=memory_max_turns,
        )

        self.counter = TokenCounter(budget=token_budget)
        self._compress_prompts = compress_prompts

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def chat(self, user_message: str, max_tokens: int = 1024, **kwargs: Any) -> dict:
        """
        Send a user message and return the provider's response dict.

        Steps:
          1. Compress user message (if enabled)
          2. Add to memory
          3. Build messages from memory
          4. Inject output compression system prompt
          5. Call provider
          6. Record token usage
          7. Add assistant response to memory

        Returns the normalised response dict:
        {
            "content": str,
            "input_tokens": int,
            "output_tokens": int,
            "cache_read_tokens": int,
            "cache_write_tokens": int,
            "model": str,
            "raw": Any,
            "_prompt_stats": CompressionStats | None,
        }
        """
        # Step 1: Compress user message
        prompt_stats = None
        if self.prompt_compressor:
            user_message, prompt_stats = self.prompt_compressor.compress(user_message)

        # Step 2: Add to memory
        self.memory.add_user(user_message)

        # Step 3: Build messages (memory handles windowing)
        messages = self.memory.to_messages(include_system=False)

        # Step 4: Build system prompt with output compression directive
        system = self.output_compressor.build_system_prompt(self._base_system)

        # Step 5a: Check response cache (avoid duplicate API calls)
        cache_key = None
        if self.cache:
            model = getattr(self.provider, "model", "")
            cache_key = self.cache.make_key(messages, system=system or None, model=model)
            if cached := self.cache.get(cache_key):
                logger.debug("cache hit — skipping API call")
                cached["_prompt_stats"] = prompt_stats
                cached["_from_cache"] = True
                self.memory.add_assistant(cached["content"])
                return cached

        # Step 5b: Call provider with retry on transient failures
        def _call():
            return self.provider.chat(
                messages=messages,
                system=system or None,
                max_tokens=max_tokens,
                **kwargs,
            )

        response = with_retry(_call, self._retry)

        # Step 5c: Populate cache
        if self.cache and cache_key:
            self.cache.set(cache_key, response)

        # Step 6: Record usage
        self.counter.record(
            input_tokens=response.get("input_tokens", 0),
            output_tokens=response.get("output_tokens", 0),
            cache_read_tokens=response.get("cache_read_tokens", 0),
            cache_write_tokens=response.get("cache_write_tokens", 0),
        )
        if self.counter.is_over_budget():
            logger.warning(
                "Token budget exceeded: %d / %d",
                self.counter.session_total.total_tokens,
                self.counter.budget,
            )

        # Step 7: Add assistant response to memory
        self.memory.add_assistant(response["content"])

        response["_prompt_stats"] = prompt_stats
        return response

    # ------------------------------------------------------------------
    # Memory controls
    # ------------------------------------------------------------------

    def toggle_memory(self) -> bool:
        """Toggle memory on/off. Returns new state."""
        state = self.memory.toggle()
        print(f"[tokenoptim] Memory {'ON ✅' if state else 'OFF ❌'}")
        return state

    def clear_memory(self) -> None:
        """Wipe conversation history."""
        self.memory.clear()
        print("[tokenoptim] Memory cleared.")

    def memory_stats(self) -> dict:
        return self.memory.stats()

    # ------------------------------------------------------------------
    # Output compression controls
    # ------------------------------------------------------------------

    def set_output_level(self, level: str | CompressionLevel) -> None:
        """Change output compression level mid-conversation."""
        self.output_compressor = OutputCompressor(
            level=level,
            max_output_tokens=self.output_compressor.max_output_tokens,
        )
        savings = OutputCompressor.estimate_savings(level)
        print(f"[tokenoptim] Output compression → {level} (est. {savings} savings)")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> None:
        """Print a full status report."""
        print("\n━━━━ tokenoptim status ━━━━")
        print(f"Provider   : {self.provider.provider_name}")
        if self.prompt_compressor:
            prompt_info = f'compressed ({self.prompt_compressor.level})'
        else:
            prompt_info = 'off'
        print(f"Prompts    : {prompt_info}")
        print(f"Output     : {self.output_compressor.level.value} "
              f"(est. {OutputCompressor.estimate_savings(self.output_compressor.level)} savings)")
        mem = self.memory.stats()
        print(f"Memory     : {'ON' if mem['enabled'] else 'OFF'} | "
              f"{mem['active_window_turns']} turns | "
              f"~{mem['estimated_window_tokens']} tokens")
        if self.counter.budget:
            used = self.counter.session_total.total_tokens
            print(f"Budget     : {used} / {self.counter.budget} tokens used")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
