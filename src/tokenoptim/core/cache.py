"""
ResponseCache — disk-based and in-memory LLM response caching.

Avoids duplicate API calls for identical (messages, system, model) combinations.
Saves the full response dict to disk as JSON so the cache survives restarts.

Zero dependencies — uses only stdlib (hashlib, json, pathlib).

Usage
-----
>>> from tokenoptim.core.cache import ResponseCache
>>>
>>> cache = ResponseCache(directory="~/.cache/tokenoptim")
>>>
>>> key = cache.make_key(messages, system, model)
>>> if cached := cache.get(key):
...     return cached  # free — no API call
>>>
>>> response = call_llm(...)
>>> cache.set(key, response)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("tokenoptim.cache")


def _hash_key(messages: list[dict], system: Optional[str], model: str) -> str:
    """Deterministic SHA-256 key for a (messages, system, model) triple."""
    payload = json.dumps(
        {"messages": messages, "system": system or "", "model": model},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class ResponseCache:
    """
    Two-level cache: in-memory (fast) backed by disk (persistent).

    Parameters
    ----------
    directory : str | Path | None
        Directory for disk cache. None = memory-only (lost on restart).
    max_memory_entries : int
        Maximum entries kept in the hot in-memory cache. LRU eviction.
    ttl_seconds : int | None
        Time-to-live for disk cache entries in seconds. None = forever.
    enabled : bool
        Master on/off switch.

    Example
    -------
    >>> cache = ResponseCache(directory="~/.cache/tokenoptim", ttl_seconds=3600)
    >>> key = cache.make_key(messages, system="...", model="claude-haiku-4-5-20251001")
    >>> if hit := cache.get(key):
    ...     return hit
    >>> resp = call_llm(...)
    >>> cache.set(key, resp)
    """

    def __init__(
        self,
        directory: Optional[str | Path] = None,
        max_memory_entries: int = 256,
        ttl_seconds: Optional[int] = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds
        self._max_mem = max_memory_entries
        self._memory: dict[str, dict] = {}   # key → response dict
        self._timestamps: dict[str, float] = {}  # key → set time

        self._disk_dir: Optional[Path] = None
        if directory:
            self._disk_dir = Path(directory).expanduser().resolve()
            self._disk_dir.mkdir(parents=True, exist_ok=True)
            logger.info("ResponseCache disk directory: %s", self._disk_dir)

        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def make_key(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        model: str = "",
    ) -> str:
        return _hash_key(messages, system, model)

    def get(self, key: str) -> Optional[dict]:
        """Return cached response or None if not found / expired."""
        if not self.enabled:
            return None

        # 1. Memory cache
        if key in self._memory:
            if not self._is_expired(key):
                self._hits += 1
                logger.debug("Cache HIT (memory): %s…", key[:12])
                return self._memory[key]
            else:
                self._evict_memory(key)

        # 2. Disk cache
        if self._disk_dir:
            path = self._disk_dir / f"{key}.json"
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    cached_at = data.get("_cached_at", 0)
                    if self.ttl_seconds and (time.time() - cached_at) > self.ttl_seconds:
                        path.unlink(missing_ok=True)
                        self._misses += 1
                        return None
                    response = {k: v for k, v in data.items() if k != "_cached_at"}
                    self._set_memory(key, response)
                    self._hits += 1
                    logger.debug("Cache HIT (disk): %s…", key[:12])
                    return response
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Cache read error for %s: %s", key[:12], e)

        self._misses += 1
        return None

    def set(self, key: str, response: dict) -> None:
        """Store a response in both memory and disk cache."""
        if not self.enabled:
            return

        # Strip non-serialisable fields (raw SDK object)
        serialisable = {k: v for k, v in response.items() if k != "raw"}

        self._set_memory(key, serialisable)

        if self._disk_dir:
            path = self._disk_dir / f"{key}.json"
            try:
                payload = {**serialisable, "_cached_at": time.time()}
                path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            except OSError as e:
                logger.warning("Cache write error for %s: %s", key[:12], e)

    def invalidate(self, key: str) -> None:
        """Remove a specific entry from memory and disk."""
        self._evict_memory(key)
        if self._disk_dir:
            (self._disk_dir / f"{key}.json").unlink(missing_ok=True)

    def clear(self) -> None:
        """Wipe all cached entries."""
        self._memory.clear()
        self._timestamps.clear()
        if self._disk_dir:
            for f in self._disk_dir.glob("*.json"):
                f.unlink(missing_ok=True)
        logger.info("Cache cleared.")

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(self._hits / total * 100, 1) if total else 0.0,
            "memory_entries": len(self._memory),
            "disk_dir": str(self._disk_dir) if self._disk_dir else None,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _set_memory(self, key: str, value: dict) -> None:
        if len(self._memory) >= self._max_mem:
            # Evict oldest entry (simple FIFO — good enough for LRU approximation)
            oldest = min(self._timestamps, key=self._timestamps.get)  # type: ignore[arg-type]
            self._evict_memory(oldest)
        self._memory[key] = value
        self._timestamps[key] = time.time()

    def _evict_memory(self, key: str) -> None:
        self._memory.pop(key, None)
        self._timestamps.pop(key, None)

    def _is_expired(self, key: str) -> bool:
        if not self.ttl_seconds:
            return False
        return (time.time() - self._timestamps.get(key, 0)) > self.ttl_seconds

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"ResponseCache(hits={s['hits']}, misses={s['misses']}, "
            f"hit_rate={s['hit_rate_pct']}%, mem={s['memory_entries']})"
        )
