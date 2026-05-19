"""Tests for MemoryManager."""

from tokenoptim.core.memory import MemoryManager


def test_memory_adds_turns():
    mem = MemoryManager(enabled=True)
    mem.add_user("Hello")
    mem.add_assistant("Hi there!")
    messages = mem.to_messages(include_system=False)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_memory_disabled_returns_only_last_user():
    mem = MemoryManager(enabled=False)
    mem.add_user("First message")
    mem.add_assistant("First response")
    mem.add_user("Second message")
    messages = mem.to_messages(include_system=False)
    assert len(messages) == 1
    assert messages[0]["content"] == "Second message"


def test_toggle_switches_state():
    mem = MemoryManager(enabled=True)
    assert mem.is_enabled is True
    state = mem.toggle()
    assert state is False
    assert mem.is_enabled is False
    mem.toggle()
    assert mem.is_enabled is True


def test_system_message_included():
    mem = MemoryManager(enabled=True)
    mem.set_system("You are a helpful assistant.")
    mem.add_user("Hi")
    messages = mem.to_messages(include_system=True)
    assert messages[0]["role"] == "system"
    assert "helpful assistant" in messages[0]["content"]


def test_window_trimming():
    mem = MemoryManager(enabled=True, max_turns=2)
    for i in range(10):
        mem.add_user(f"Message {i}")
        mem.add_assistant(f"Response {i}")
    messages = mem.to_messages(include_system=False)
    # max_turns=2 means 2 pairs = 4 turns max, plus possibly a compaction summary system msg
    assert len(messages) <= 5


def test_clear_resets_history():
    mem = MemoryManager(enabled=True)
    mem.add_user("Hello")
    mem.add_assistant("World")
    mem.clear()
    messages = mem.to_messages(include_system=False)
    assert messages == []


def test_stats_returns_dict():
    mem = MemoryManager(enabled=True)
    mem.add_user("Test message")
    stats = mem.stats()
    assert "enabled" in stats
    assert "active_window_turns" in stats
    assert "estimated_window_tokens" in stats
    assert stats["enabled"] is True
