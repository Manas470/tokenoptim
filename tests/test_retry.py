"""Tests for RetryConfig and with_retry."""

import pytest
from tokenoptim.core.retry import RetryConfig, with_retry, _is_retriable


class _FakeRateLimitError(Exception):
    """Simulates an Anthropic/OpenAI RateLimitError."""
    status_code = 429


class _FakeServerError(Exception):
    status_code = 503


class _FakeFatalError(Exception):
    status_code = 400  # not retriable


def test_retry_succeeds_first_attempt():
    cfg = RetryConfig(max_attempts=3, base_delay=0)
    calls = []
    def fn():
        calls.append(1)
        return "ok"
    result = with_retry(fn, cfg)
    assert result == "ok"
    assert len(calls) == 1


def test_retry_succeeds_after_transient_failures():
    cfg = RetryConfig(max_attempts=4, base_delay=0, jitter=False)
    calls = []
    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise _FakeRateLimitError("rate limit")
        return "recovered"
    result = with_retry(fn, cfg)
    assert result == "recovered"
    assert len(calls) == 3


def test_retry_raises_after_max_attempts():
    cfg = RetryConfig(max_attempts=3, base_delay=0, jitter=False)
    def fn():
        raise _FakeRateLimitError("always fails")
    with pytest.raises(_FakeRateLimitError):
        with_retry(fn, cfg)


def test_retry_does_not_retry_fatal_errors():
    cfg = RetryConfig(max_attempts=5, base_delay=0)
    calls = []
    def fn():
        calls.append(1)
        raise _FakeFatalError("bad request")
    with pytest.raises(_FakeFatalError):
        with_retry(fn, cfg)
    assert len(calls) == 1  # should not retry 400s


def test_is_retriable_by_status_code():
    for code in [429, 500, 502, 503, 504]:
        exc = Exception()
        exc.status_code = code  # type: ignore[attr-defined]
        assert _is_retriable(exc), f"status {code} should be retriable"


def test_is_retriable_by_class_name():
    class RateLimitError(Exception): pass
    class APIConnectionError(Exception): pass
    assert _is_retriable(RateLimitError())
    assert _is_retriable(APIConnectionError())


def test_retry_config_delay_grows():
    cfg = RetryConfig(base_delay=1.0, backoff_factor=2.0, max_delay=60.0, jitter=False)
    assert cfg.delay_for(0) == 1.0
    assert cfg.delay_for(1) == 2.0
    assert cfg.delay_for(2) == 4.0


def test_retry_config_delay_capped():
    cfg = RetryConfig(base_delay=10.0, backoff_factor=10.0, max_delay=50.0, jitter=False)
    assert cfg.delay_for(3) == 50.0


@pytest.mark.asyncio
async def test_async_retry_recovers():
    pytest.importorskip("asyncio")
    from tokenoptim.core.retry import with_retry_async
    cfg = RetryConfig(max_attempts=3, base_delay=0, jitter=False)
    calls = []
    async def fn():
        calls.append(1)
        if len(calls) < 2:
            raise _FakeRateLimitError("rate limit")
        return "async_ok"
    result = await with_retry_async(fn, cfg)
    assert result == "async_ok"
    assert len(calls) == 2
