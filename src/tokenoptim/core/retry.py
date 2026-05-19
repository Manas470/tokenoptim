"""
Retry logic with exponential backoff for transient LLM API failures.

Handles:
  - 429 Rate limit errors
  - 500/502/503/504 Server errors
  - Network timeouts
  - Anthropic and OpenAI exception types

Zero dependencies — uses only stdlib.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

logger = logging.getLogger("tokenoptim.retry")

T = TypeVar("T")

# Exception class names that indicate a retriable error
_RETRIABLE_NAMES = {
    # Anthropic
    "RateLimitError",
    "InternalServerError",
    "APIConnectionError",
    "APITimeoutError",
    # OpenAI
    "RateLimitError",
    "InternalServerError",
    "APIConnectionError",
    "APITimeoutError",
    # Generic
    "ConnectionError",
    "TimeoutError",
}

_RETRIABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retriable(exc: Exception) -> bool:
    """Return True if the exception is a transient, retriable error."""
    name = type(exc).__name__
    if name in _RETRIABLE_NAMES:
        return True
    # Check for status_code attribute (Anthropic/OpenAI SDK pattern)
    status = getattr(exc, "status_code", None)
    if status in _RETRIABLE_STATUS_CODES:
        return True
    return False


def _get_retry_after(exc: Exception) -> float | None:
    """Extract Retry-After header value if present (seconds)."""
    headers = getattr(exc, "response", None)
    if headers is None:
        return None
    retry_after = None
    if hasattr(headers, "headers"):
        retry_after = headers.headers.get("retry-after") or headers.headers.get("x-ratelimit-reset-requests")
    if retry_after:
        try:
            return float(retry_after)
        except (ValueError, TypeError):
            pass
    return None


@dataclass
class RetryConfig:
    """
    Configuration for the retry policy.

    Parameters
    ----------
    max_attempts : int
        Total attempts (including the first). Default 4 (= 3 retries).
    base_delay : float
        Initial backoff delay in seconds. Default 1.0.
    max_delay : float
        Cap on backoff delay. Default 60.0.
    jitter : bool
        Add ±25% random jitter to prevent thundering herd. Default True.
    backoff_factor : float
        Multiplier per retry. Default 2.0 (exponential).
    """

    max_attempts: int = 4
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True
    backoff_factor: float = 2.0

    def delay_for(self, attempt: int) -> float:
        """Return the wait time (seconds) for the given attempt number (0-indexed)."""
        delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)
        if self.jitter:
            delay *= 0.75 + random.random() * 0.5  # ±25%
        return delay


def with_retry(fn: Callable[[], T], config: RetryConfig) -> T:
    """
    Call fn() with synchronous retry logic.

    Parameters
    ----------
    fn : Callable
        Zero-argument callable to retry.
    config : RetryConfig
        Retry policy.

    Returns
    -------
    Result of fn() on success.

    Raises
    ------
    Last exception if all attempts exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(config.max_attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if not _is_retriable(exc) or attempt == config.max_attempts - 1:
                raise

            # Respect Retry-After if present
            wait = _get_retry_after(exc) or config.delay_for(attempt)
            logger.warning(
                "Attempt %d/%d failed (%s: %s). Retrying in %.1fs...",
                attempt + 1,
                config.max_attempts,
                type(exc).__name__,
                str(exc)[:120],
                wait,
            )
            time.sleep(wait)

    raise last_exc  # type: ignore[misc]


async def with_retry_async(fn: Callable[[], "asyncio.Future[T]"], config: RetryConfig) -> T:
    """
    Call async fn() with retry logic.

    Parameters
    ----------
    fn : async Callable
        Zero-argument async callable (coroutine factory) to retry.
    config : RetryConfig
        Retry policy.
    """
    last_exc: Exception | None = None
    for attempt in range(config.max_attempts):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            if not _is_retriable(exc) or attempt == config.max_attempts - 1:
                raise

            wait = _get_retry_after(exc) or config.delay_for(attempt)
            logger.warning(
                "Async attempt %d/%d failed (%s). Retrying in %.1fs...",
                attempt + 1,
                config.max_attempts,
                type(exc).__name__,
                wait,
            )
            await asyncio.sleep(wait)

    raise last_exc  # type: ignore[misc]
