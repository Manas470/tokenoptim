"""
MemoryManager — toggleable conversation context with auto-compaction.

Problems solved:
  1. Users forget to truncate history → ballooning input tokens each turn
  2. Full history is rarely needed → 80% of tokens are ancient context
  3. No easy on/off switch → hard to A/B test with/without memory

Features:
  - Toggle memory ON/OFF mid-conversation
  - Configurable window size (last N messages)
  - Auto-summarize older turns when window fills (compaction)
  - Token budget awareness — compacts when approaching limit
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    token_estimate: int = 0

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    @staticmethod
    def _approx_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def __post_init__(self):
        if not self.token_estimate:
            self.token_estimate = self._approx_tokens(self.content)


class MemoryManager:
    """
    Manage conversation history with a toggle switch and auto-compaction.

    Parameters
    ----------
    enabled : bool
        Whether memory is active. When False, only the latest user message
        is passed to the LLM (stateless mode).
    max_turns : int
        Maximum number of message pairs to keep in the active window.
    max_tokens : int | None
        If set, auto-compact history when estimated token count exceeds this.
    compaction_summary_prompt : str
        The prompt injected when asking the model to summarize older history.
        You can customise this.
    keep_system_message : bool
        Always preserve the original system message regardless of window trimming.

    Example
    -------
    >>> mem = MemoryManager(enabled=True, max_turns=10)
    >>> mem.add("user", "What is PySpark?")
    >>> mem.add("assistant", "PySpark is the Python API for Apache Spark...")
    >>> messages = mem.to_messages()  # ready for LLM call

    Toggle off mid-conversation:
    >>> mem.enabled = False
    >>> messages = mem.to_messages()  # returns only last user turn
    """

    DEFAULT_COMPACTION_PROMPT = (
        "Summarize the conversation so far in ≤150 words. "
        "Preserve: key decisions, facts established, user's goal, open questions. "
        "Omit: greetings, filler, repeated content."
    )

    def __init__(
        self,
        enabled: bool = True,
        max_turns: int = 20,
        max_tokens: Optional[int] = None,
        compaction_summary_prompt: Optional[str] = None,
        keep_system_message: bool = True,
    ) -> None:
        self.enabled = enabled
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.compaction_summary_prompt = (
            compaction_summary_prompt or self.DEFAULT_COMPACTION_PROMPT
        )
        self.keep_system_message = keep_system_message

        self._history: list[Turn] = []
        self._system_message: Optional[str] = None
        self._compaction_summary: Optional[str] = None
        self._total_turns_added: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        return self.enabled

    def toggle(self) -> bool:
        """Flip memory on/off. Returns new state."""
        self.enabled = not self.enabled
        return self.enabled

    def set_system(self, content: str) -> None:
        """Set the persistent system message."""
        self._system_message = content

    def add(self, role: str, content: str) -> None:
        """Append a turn to history."""
        self._history.append(Turn(role=role, content=content))
        self._total_turns_added += 1
        self._maybe_compact()

    def add_user(self, content: str) -> None:
        self.add("user", content)

    def add_assistant(self, content: str) -> None:
        self.add("assistant", content)

    def to_messages(self, include_system: bool = True) -> list[dict]:
        """
        Build the messages list to pass to an LLM.

        When memory is OFF: returns system message + last user turn only.
        When memory is ON:  returns system + compaction summary (if any) + windowed history.
        """
        messages: list[dict] = []

        if include_system and self._system_message:
            messages.append({"role": "system", "content": self._system_message})

        if not self.enabled:
            # Stateless: only the most recent user message
            last_user = self._last_user_turn()
            if last_user:
                messages.append(last_user.to_dict())
            return messages

        # Memory ON: compaction summary + windowed history
        window = self._windowed_history()
        if self._compaction_summary:
            messages.append({
                "role": "system",
                "content": f"[Conversation summary so far]: {self._compaction_summary}",
            })

        messages.extend(t.to_dict() for t in window)
        return messages

    def clear(self) -> None:
        """Wipe history and compaction summary."""
        self._history.clear()
        self._compaction_summary = None
        self._total_turns_added = 0

    def stats(self) -> dict:
        """Return memory usage statistics."""
        window = self._windowed_history()
        total_tokens = sum(t.token_estimate for t in window)
        return {
            "enabled": self.enabled,
            "total_turns_added": self._total_turns_added,
            "active_window_turns": len(window),
            "estimated_window_tokens": total_tokens,
            "has_compaction_summary": self._compaction_summary is not None,
            "compaction_summary_tokens": (
                len(self._compaction_summary) // 4 if self._compaction_summary else 0
            ),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _windowed_history(self) -> list[Turn]:
        """Return the last max_turns turns."""
        return self._history[-self.max_turns * 2:]  # *2 because each pair = 2 turns

    def _last_user_turn(self) -> Optional[Turn]:
        for turn in reversed(self._history):
            if turn.role == "user":
                return turn
        return None

    def _maybe_compact(self) -> None:
        """
        If we've exceeded the window, drop the oldest turns and store a
        compaction marker. In a real integration the caller would generate
        the summary via the LLM; here we produce a token-count placeholder.
        """
        if len(self._history) > self.max_turns * 2:
            cutoff = len(self._history) - self.max_turns * 2
            dropped = self._history[:cutoff]
            self._history = self._history[cutoff:]

            # Build a lightweight local summary (caller can replace this with an LLM call)
            dropped_roles = [t.role for t in dropped]
            token_count = sum(t.token_estimate for t in dropped)
            self._compaction_summary = (
                f"[{len(dropped)} earlier turns compacted, ~{token_count} tokens saved. "
                f"Roles: {', '.join(set(dropped_roles))}. "
                "Call mem.get_compaction_prompt() + LLM to generate a rich summary.]"
            )

        if self.max_tokens:
            window_tokens = sum(t.token_estimate for t in self._windowed_history())
            if window_tokens > self.max_tokens:
                # Hard trim to keep under budget
                while (
                    self._history
                    and sum(t.token_estimate for t in self._windowed_history()) > self.max_tokens
                ):
                    self._history.pop(0)

    def get_compaction_prompt(self) -> list[dict]:
        """
        Return messages you should send to an LLM to generate a rich summary
        of the currently compacted turns. Feed the result back via
        set_compaction_summary().
        """
        history_text = "\n".join(
            f"{t.role.upper()}: {t.content}" for t in self._history
        )
        return [
            {"role": "user", "content": f"{self.compaction_summary_prompt}\n\n{history_text}"}
        ]

    def set_compaction_summary(self, summary: str) -> None:
        """Store a model-generated compaction summary."""
        self._compaction_summary = summary

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"MemoryManager(enabled={self.enabled}, "
            f"turns={s['active_window_turns']}, "
            f"~tokens={s['estimated_window_tokens']})"
        )
