"""Tests for ResponseCache."""

import time

from tokenoptim.core.cache import ResponseCache

SAMPLE_RESPONSE = {
    "content": "Tokenization splits text into tokens.",
    "input_tokens": 10,
    "output_tokens": 8,
    "model": "claude-haiku-4-5-20251001",
}
MESSAGES = [{"role": "user", "content": "What is tokenization?"}]


def test_cache_miss_returns_none():
    cache = ResponseCache()
    key = cache.make_key(MESSAGES, model="test")
    assert cache.get(key) is None


def test_cache_set_and_get():
    cache = ResponseCache()
    key = cache.make_key(MESSAGES, model="test")
    cache.set(key, SAMPLE_RESPONSE)
    hit = cache.get(key)
    assert hit is not None
    assert hit["content"] == SAMPLE_RESPONSE["content"]
    assert hit["input_tokens"] == SAMPLE_RESPONSE["input_tokens"]


def test_cache_strips_raw_field():
    cache = ResponseCache()
    key = cache.make_key(MESSAGES, model="test")
    resp_with_raw = {**SAMPLE_RESPONSE, "raw": object()}
    cache.set(key, resp_with_raw)
    hit = cache.get(key)
    assert "raw" not in hit


def test_cache_hit_rate():
    cache = ResponseCache()
    key = cache.make_key(MESSAGES, model="test")
    cache.get(key)       # miss
    cache.set(key, SAMPLE_RESPONSE)
    cache.get(key)       # hit
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate_pct"] == 50.0


def test_cache_ttl_expiry():
    cache = ResponseCache(ttl_seconds=1)
    key = cache.make_key(MESSAGES, model="test")
    cache.set(key, SAMPLE_RESPONSE)
    assert cache.get(key) is not None
    time.sleep(1.1)
    assert cache.get(key) is None


def test_cache_different_messages_different_keys():
    cache = ResponseCache()
    msgs_a = [{"role": "user", "content": "Question A"}]
    msgs_b = [{"role": "user", "content": "Question B"}]
    key_a = cache.make_key(msgs_a, model="test")
    key_b = cache.make_key(msgs_b, model="test")
    assert key_a != key_b


def test_cache_same_content_same_key():
    cache = ResponseCache()
    key1 = cache.make_key(MESSAGES, system="sys", model="m")
    key2 = cache.make_key(MESSAGES, system="sys", model="m")
    assert key1 == key2


def test_cache_invalidate():
    cache = ResponseCache()
    key = cache.make_key(MESSAGES, model="test")
    cache.set(key, SAMPLE_RESPONSE)
    cache.invalidate(key)
    assert cache.get(key) is None


def test_cache_clear():
    cache = ResponseCache()
    for i in range(5):
        msgs = [{"role": "user", "content": f"Q{i}"}]
        cache.set(cache.make_key(msgs, model="t"), SAMPLE_RESPONSE)
    cache.clear()
    assert cache.stats()["memory_entries"] == 0


def test_cache_disabled():
    cache = ResponseCache(enabled=False)
    key = cache.make_key(MESSAGES, model="test")
    cache.set(key, SAMPLE_RESPONSE)
    assert cache.get(key) is None  # disabled — always miss


def test_cache_max_memory_eviction():
    cache = ResponseCache(max_memory_entries=3)
    for i in range(5):
        msgs = [{"role": "user", "content": f"Q{i}"}]
        cache.set(cache.make_key(msgs, model="t"), SAMPLE_RESPONSE)
    assert cache.stats()["memory_entries"] <= 3


def test_disk_cache_persists(tmp_path):
    cache1 = ResponseCache(directory=tmp_path)
    key = cache1.make_key(MESSAGES, model="test")
    cache1.set(key, SAMPLE_RESPONSE)

    # New cache instance pointing at same directory
    cache2 = ResponseCache(directory=tmp_path)
    hit = cache2.get(key)
    assert hit is not None
    assert hit["content"] == SAMPLE_RESPONSE["content"]
