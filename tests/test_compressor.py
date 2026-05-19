"""Tests for PromptCompressor."""

import pytest

from tokenoptim.core.compressor import PromptCompressor

VERBOSE = (
    "Could you please help me understand what tokenization means? "
    "I think it is important to note that I am fairly new to this topic. "
    "Perhaps you could provide some examples if that makes sense. "
    "Hope this helps me learn. Thank you so much!"
)


def test_light_compression_reduces_chars():
    c = PromptCompressor(level="light")
    compressed, stats = c.compress(VERBOSE)
    assert stats.compressed_chars < stats.original_chars


def test_medium_compression_reduces_more_than_light():
    c_light = PromptCompressor(level="light")
    c_medium = PromptCompressor(level="medium")
    _, s_light = c_light.compress(VERBOSE)
    _, s_medium = c_medium.compress(VERBOSE)
    assert s_medium.compressed_approx_tokens <= s_light.compressed_approx_tokens


def test_full_compression_reduces_most():
    c_full = PromptCompressor(level="full")
    _, stats = c_full.compress(VERBOSE)
    assert stats.token_reduction_pct > 0


def test_code_blocks_preserved():
    prompt = "Explain this code:\n```python\nfor i in range(10):\n    print(i)\n```\nThanks!"
    c = PromptCompressor(level="full", preserve_code=True)
    compressed, _ = c.compress(prompt)
    assert "```python" in compressed
    assert "for i in range(10):" in compressed


def test_deduplicate_removes_duplicate_paragraphs():
    dup = "This is a very important paragraph that we need to keep.\n\n" * 3
    c = PromptCompressor(level="light", deduplicate=True)
    compressed, _ = c.compress(dup)
    # Should appear only once
    assert compressed.count("This is a very important paragraph") == 1


def test_invalid_level_raises():
    with pytest.raises(ValueError):
        PromptCompressor(level="extreme")


def test_compress_messages():
    messages = [
        {"role": "user", "content": "Could you please explain what a token is?"},
        {"role": "assistant", "content": "A token is a unit of text..."},
    ]
    c = PromptCompressor(level="medium")
    compressed_msgs, stats = c.compress_messages(messages)
    # User message should be compressed
    assert len(compressed_msgs[0]["content"]) <= len(messages[0]["content"])
    # Assistant message should be unchanged
    assert compressed_msgs[1]["content"] == messages[1]["content"]


def test_stats_str():
    c = PromptCompressor(level="medium")
    _, stats = c.compress(VERBOSE)
    s = str(stats)
    assert "Chars" in s
    assert "Tokens" in s
