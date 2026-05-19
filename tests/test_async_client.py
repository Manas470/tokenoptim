"""
Tests for AsyncOptimizedClient — uses a mock async provider, no real API calls.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from tokenoptim.async_client import AsyncOptimizedClient


def _make_response(content="Hello from mock", input_tokens=10, output_tokens=5):
    return {
        "content": content,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "model": "mock-model",
        "raw": None,
    }


def _make_provider(content="Hello from mock"):
    provider = MagicMock()
    provider.provider_name = "mock-async"
    provider.async_chat = AsyncMock(return_value=_make_response(content))
    return provider


@pytest.mark.asyncio
async def test_async_chat_returns_content():
    provider = _make_provider("Tokenization explained.")
    client = AsyncOptimizedClient(provider=provider, compress_prompts=False, memory_enabled=False)
    resp = await client.chat("What is tokenization?")
    assert resp["content"] == "Tokenization explained."


@pytest.mark.asyncio
async def test_async_chat_records_tokens():
    provider = _make_provider()
    client = AsyncOptimizedClient(provider=provider, compress_prompts=False)
    await client.chat("Hello")
    assert client.counter.session_total.input_tokens == 10
    assert client.counter.session_total.output_tokens == 5


@pytest.mark.asyncio
async def test_async_chat_compresses_prompt():
    provider = _make_provider()
    client = AsyncOptimizedClient(
        provider=provider,
        compress_prompts=True,
        prompt_compression_level="medium",
    )
    verbose = "Could you please help me understand what tokenization means? Hope this helps!"
    await client.chat(verbose)
    # Check that what was sent to provider is shorter than original
    call_args = provider.async_chat.call_args
    sent_messages = call_args[1]["messages"] if call_args[1] else call_args[0][0]
    sent_content = sent_messages[-1]["content"]
    assert len(sent_content) <= len(verbose)


@pytest.mark.asyncio
async def test_async_memory_accumulates():
    provider = _make_provider()
    client = AsyncOptimizedClient(provider=provider, compress_prompts=False, memory_enabled=True)
    await client.chat("First message")
    await client.chat("Second message")
    stats = client.memory_stats()
    assert stats["active_window_turns"] >= 2


@pytest.mark.asyncio
async def test_async_toggle_memory():
    provider = _make_provider()
    client = AsyncOptimizedClient(provider=provider, compress_prompts=False, memory_enabled=True)
    assert client.memory.is_enabled
    client.toggle_memory()
    assert not client.memory.is_enabled
    client.toggle_memory()
    assert client.memory.is_enabled


@pytest.mark.asyncio
async def test_async_batch_chat_returns_all():
    provider = MagicMock()
    provider.provider_name = "mock-async"
    provider.async_chat = AsyncMock(side_effect=[
        _make_response("A"), _make_response("B"), _make_response("C")
    ])
    client = AsyncOptimizedClient(provider=provider, compress_prompts=False)
    results = await client.batch_chat(["Q1", "Q2", "Q3"], max_concurrency=3)
    assert len(results) == 3
    contents = {r["content"] for r in results}
    assert contents == {"A", "B", "C"}


@pytest.mark.asyncio
async def test_async_budget_warning(caplog):
    import logging
    provider = _make_provider()
    client = AsyncOptimizedClient(
        provider=provider,
        compress_prompts=False,
        token_budget=5,  # very low — will be exceeded immediately
    )
    with caplog.at_level(logging.WARNING, logger="tokenoptim"):
        await client.chat("Hello")
    assert any("budget" in r.message.lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_async_stream_yields_chunks():
    async def _fake_stream(*args, **kwargs):
        for chunk in ["Hello", " world", "!"]:
            yield chunk

    provider = MagicMock()
    provider.provider_name = "mock-async"
    provider.stream = _fake_stream

    client = AsyncOptimizedClient(provider=provider, compress_prompts=False)
    chunks = []
    async for chunk in client.stream("Say hello"):
        chunks.append(chunk)

    assert "".join(chunks) == "Hello world!"
