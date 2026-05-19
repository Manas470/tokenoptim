"""
AsyncOptimizedClient — production async LLM client with token optimization.

Supports:
  - Full async/await
  - Token-by-token streaming via async generators
  - All optimizations: prompt compression, output compression, memory, token counting
  - Automatic retry with exponential backoff

Quick start
-----------
>>> import asyncio
>>> from tokenoptim import AsyncOptimizedClient
>>> from tokenoptim.providers import AsyncAnthropicProvider
>>>
>>> async def main():
...     client = AsyncOptimizedClient(
...         provider=AsyncAnthropicProvider(),
...         compress_prompts=True,
...         output_level="full",
...         memory_enabled=True,
...     )
...     # Standard async call
...     resp = await client.chat("Explain distributed systems")
...     print(resp["content"])
...
...     # Streaming call — tokens arrive as they're generated
...     async for chunk in client.stream("Write a function to compress text"):
...         print(chunk, end="", flush=True)
...
>>> asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Optional

from tokenoptim.core.compressor import PromptCompressor
from tokenoptim.core.output_style import OutputCompressor, CompressionLevel
from tokenoptim.core.memory import MemoryManager
from tokenoptim.core.counter import TokenCounter
from tokenoptim.core.retry import RetryConfig, with_retry_async

logger = logging.getLogger("tokenoptim")


class AsyncOptimizedClient:
    """
    Production-grade async LLM client with token optimization.

    Parameters
    ----------
    provider : AsyncAnthropicProvider | AsyncOpenAIProvider
        An async-capable provider instance.
    compress_prompts : bool
        Apply PromptCompressor to user messages.
    prompt_compression_level : str
        "light" | "medium" | "full"
    output_level : str | CompressionLevel
        Output compression level (default "standard").
    memory_enabled : bool
        Enable conversation memory.
    memory_max_turns : int
        Active memory window size.
    token_budget : int | None
        Optional session token budget.
    system : str | None
        Base system prompt.
    retry : RetryConfig | None
        Retry policy for transient failures (429, 5xx). Defaults to 3 retries.
    """

    def __init__(
        self,
        provider,
        compress_prompts: bool = True,
        prompt_compression_level: str = "medium",
        output_level: str | CompressionLevel = CompressionLevel.STANDARD,
        memory_enabled: bool = True,
        memory_max_turns: int = 20,
        token_budget: Optional[int] = None,
        system: Optional[str] = None,
        retry: Optional[RetryConfig] = None,
    ) -> None:
        self.provider = provider
        self._base_system = system or ""
        self._retry = retry or RetryConfig()

        self.prompt_compressor = (
            PromptCompressor(level=prompt_compression_level) if compress_prompts else None
        )
        self.output_compressor = OutputCompressor(level=output_level)
        self.memory = MemoryManager(enabled=memory_enabled, max_turns=memory_max_turns)
        self.counter = TokenCounter(budget=token_budget)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    async def chat(
        self,
        user_message: str,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> dict:
        """
        Send a message and return the full response dict asynchronously.

        Applies prompt compression → memory management → output compression → provider call.
        Retries automatically on transient errors.
        """
        prompt_stats = None
        if self.prompt_compressor:
            user_message, prompt_stats = self.prompt_compressor.compress(user_message)

        self.memory.add_user(user_message)
        messages = self.memory.to_messages(include_system=False)
        system = self.output_compressor.build_system_prompt(self._base_system)

        logger.debug(
            "chat: provider=%s tokens_est=%d memory_turns=%d",
            self.provider.provider_name,
            self.counter.session_total.total_tokens,
            self.memory.stats()["active_window_turns"],
        )

        async def _call():
            return await self.provider.async_chat(
                messages=messages,
                system=system or None,
                max_tokens=max_tokens,
                **kwargs,
            )

        response = await with_retry_async(_call, self._retry)

        self.counter.record(
            input_tokens=response.get("input_tokens", 0),
            output_tokens=response.get("output_tokens", 0),
            cache_read_tokens=response.get("cache_read_tokens", 0),
            cache_write_tokens=response.get("cache_write_tokens", 0),
        )

        self.memory.add_assistant(response["content"])
        response["_prompt_stats"] = prompt_stats

        if self.counter.is_over_budget():
            logger.warning("Token budget exceeded: %d / %d", self.counter.session_total.total_tokens, self.counter.budget)

        return response

    async def stream(
        self,
        user_message: str,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream the response token-by-token as an async generator.

        Collects the full response in memory after streaming completes.

        Usage
        -----
        >>> async for chunk in client.stream("Explain tokenization"):
        ...     print(chunk, end="", flush=True)
        """
        if not hasattr(self.provider, "stream"):
            raise NotImplementedError(
                f"Provider {self.provider.provider_name} does not support streaming. "
                "Use AsyncAnthropicProvider or AsyncOpenAIProvider."
            )

        prompt_stats = None
        if self.prompt_compressor:
            user_message, prompt_stats = self.prompt_compressor.compress(user_message)

        self.memory.add_user(user_message)
        messages = self.memory.to_messages(include_system=False)
        system = self.output_compressor.build_system_prompt(self._base_system)

        full_response = []
        async for chunk in self.provider.stream(
            messages=messages,
            system=system or None,
            max_tokens=max_tokens,
            **kwargs,
        ):
            full_response.append(chunk)
            yield chunk

        # Store complete response in memory after streaming finishes
        complete = "".join(full_response)
        self.memory.add_assistant(complete)
        # Approximate token count for streaming (exact count not available without response obj)
        approx_out = len(complete) // 4
        approx_in = sum(len(str(m.get("content", ""))) for m in messages) // 4
        self.counter.record(input_tokens=approx_in, output_tokens=approx_out)

    # ------------------------------------------------------------------
    # Concurrent batch calls
    # ------------------------------------------------------------------

    async def batch_chat(
        self,
        messages: list[str],
        max_tokens: int = 1024,
        max_concurrency: int = 5,
        **kwargs: Any,
    ) -> list[dict]:
        """
        Send multiple independent messages concurrently.

        Parameters
        ----------
        messages : list[str]
            List of user messages to send in parallel.
        max_concurrency : int
            Max simultaneous API calls (respect rate limits).

        Returns
        -------
        list[dict] in the same order as input messages.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _limited_chat(msg: str) -> dict:
            async with semaphore:
                # Each batch call uses its own compressor + output style
                # but shares the counter
                prompt_stats = None
                user_msg = msg
                if self.prompt_compressor:
                    user_msg, prompt_stats = self.prompt_compressor.compress(msg)
                system = self.output_compressor.build_system_prompt(self._base_system)

                async def _call():
                    return await self.provider.async_chat(
                        messages=[{"role": "user", "content": user_msg}],
                        system=system or None,
                        max_tokens=max_tokens,
                        **kwargs,
                    )

                resp = await with_retry_async(_call, self._retry)
                self.counter.record(
                    input_tokens=resp.get("input_tokens", 0),
                    output_tokens=resp.get("output_tokens", 0),
                )
                resp["_prompt_stats"] = prompt_stats
                return resp

        return await asyncio.gather(*[_limited_chat(m) for m in messages])

    # ------------------------------------------------------------------
    # Memory controls
    # ------------------------------------------------------------------

    def toggle_memory(self) -> bool:
        state = self.memory.toggle()
        logger.info("Memory %s", "ON" if state else "OFF")
        return state

    def clear_memory(self) -> None:
        self.memory.clear()

    def memory_stats(self) -> dict:
        return self.memory.stats()

    def set_output_level(self, level: str | CompressionLevel) -> None:
        self.output_compressor = OutputCompressor(level=level)

    def status(self) -> None:
        print("\n━━━━ tokenoptim AsyncClient ━━━━")
        print(f"Provider   : {self.provider.provider_name}")
        print(f"Prompts    : {'compressed' if self.prompt_compressor else 'off'}")
        print(f"Output     : {self.output_compressor.level.value}")
        mem = self.memory.stats()
        print(f"Memory     : {'ON' if mem['enabled'] else 'OFF'} | {mem['active_window_turns']} turns")
        print(f"Tokens used: {self.counter.session_total.total_tokens}")
        print(f"API calls  : {self.counter.call_count}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
