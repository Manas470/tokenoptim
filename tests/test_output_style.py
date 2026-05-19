"""Tests for OutputCompressor."""

import pytest
from tokenoptim.core.output_style import OutputCompressor, CompressionLevel


def test_off_level_returns_base_system():
    oc = OutputCompressor(level="off")
    result = oc.build_system_prompt("You are helpful.")
    assert result == "You are helpful."


def test_standard_prepends_directive():
    oc = OutputCompressor(level="standard")
    result = oc.build_system_prompt("You are helpful.")
    assert "COMPRESSION MODE" in result
    assert "You are helpful." in result


def test_full_level_has_caveman_language():
    oc = OutputCompressor(level="full")
    result = oc.build_system_prompt()
    assert "caveman" in result.lower() or "drop all articles" in result.lower()


def test_max_output_tokens_appended():
    oc = OutputCompressor(level="standard", max_output_tokens=200)
    result = oc.build_system_prompt()
    assert "200" in result
    assert "BUDGET" in result


def test_patch_messages_injects_system():
    oc = OutputCompressor(level="full")
    messages = [{"role": "user", "content": "Hello"}]
    patched = oc.patch_messages(messages)
    assert patched[0]["role"] == "system"
    assert len(patched) == 2


def test_patch_messages_merges_existing_system():
    oc = OutputCompressor(level="standard")
    messages = [
        {"role": "system", "content": "Existing system prompt."},
        {"role": "user", "content": "Hello"},
    ]
    patched = oc.patch_messages(messages)
    assert patched[0]["role"] == "system"
    assert "Existing system prompt." in patched[0]["content"]
    assert "COMPRESSION" in patched[0]["content"]


def test_estimate_savings_all_levels():
    for level in CompressionLevel:
        est = OutputCompressor.estimate_savings(level)
        assert "%" in est


def test_custom_exceptions_appended():
    oc = OutputCompressor(level="full", custom_exceptions=["legal disclaimers", "pricing"])
    result = oc.build_system_prompt()
    assert "legal disclaimers" in result
    assert "pricing" in result
