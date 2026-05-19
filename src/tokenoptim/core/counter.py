"""
TokenCounter — track token usage across providers in real time.

Aggregates per-call, per-session, and lifetime token budgets.
Supports Anthropic, OpenAI, and approximate counting for others.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def effective_tokens(self) -> int:
        """Tokens that will be billed (cache reads are cheap)."""
        return self.input_tokens + self.output_tokens + self.cache_write_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )

    def __str__(self) -> str:
        parts = [f"in={self.input_tokens}", f"out={self.output_tokens}"]
        if self.cache_read_tokens:
            parts.append(f"cache_read={self.cache_read_tokens}")
        if self.cache_write_tokens:
            parts.append(f"cache_write={self.cache_write_tokens}")
        parts.append(f"total={self.total_tokens}")
        return "TokenUsage(" + ", ".join(parts) + ")"


class TokenCounter:
    """
    Aggregate token usage across multiple LLM calls.

    Parameters
    ----------
    budget : int | None
        Optional token budget. Call .is_over_budget() to check.

    Example
    -------
    >>> counter = TokenCounter(budget=100_000)
    >>> counter.record(input_tokens=500, output_tokens=120)
    >>> print(counter.session_total)
    >>> counter.report()
    """

    def __init__(self, budget: int | None = None) -> None:
        self.budget = budget
        self._calls: list[TokenUsage] = []

    def record(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> TokenUsage:
        """Record a single LLM call's token usage. Returns the TokenUsage for this call."""
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        self._calls.append(usage)
        return usage

    def record_from_anthropic(self, response) -> TokenUsage:
        """Extract and record usage from an Anthropic response object."""
        u = response.usage
        return self.record(
            input_tokens=getattr(u, "input_tokens", 0),
            output_tokens=getattr(u, "output_tokens", 0),
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0),
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0),
        )

    def record_from_openai(self, response) -> TokenUsage:
        """Extract and record usage from an OpenAI response object."""
        u = response.usage
        return self.record(
            input_tokens=getattr(u, "prompt_tokens", 0),
            output_tokens=getattr(u, "completion_tokens", 0),
        )

    @property
    def session_total(self) -> TokenUsage:
        """Aggregate usage across all recorded calls this session."""
        total = TokenUsage()
        for call in self._calls:
            total = total + call
        return total

    @property
    def call_count(self) -> int:
        return len(self._calls)

    def is_over_budget(self) -> bool:
        if self.budget is None:
            return False
        return self.session_total.effective_tokens > self.budget

    def remaining_budget(self) -> int | None:
        if self.budget is None:
            return None
        return max(0, self.budget - self.session_total.effective_tokens)

    def reset(self) -> None:
        self._calls.clear()

    def report(self) -> str:
        """Print a formatted usage report."""
        total = self.session_total
        lines = [
            "╔══════════════════════════════════════╗",
            "║       tokenoptim — Token Report       ║",
            "╠══════════════════════════════════════╣",
            f"║  API calls made:    {self.call_count:<18}║",
            f"║  Input tokens:      {total.input_tokens:<18}║",
            f"║  Output tokens:     {total.output_tokens:<18}║",
        ]
        if total.cache_read_tokens:
            lines.append(f"║  Cache reads:       {total.cache_read_tokens:<18}║")
        if total.cache_write_tokens:
            lines.append(f"║  Cache writes:      {total.cache_write_tokens:<18}║")
        lines.append(f"║  Total tokens:      {total.total_tokens:<18}║")
        if self.budget:
            remaining = self.remaining_budget()
            pct = round((total.effective_tokens / self.budget) * 100, 1)
            lines.append(f"║  Budget used:       {pct}%{' ':<14}║")
            lines.append(f"║  Budget remaining:  {remaining:<18}║")
        lines.append("╚══════════════════════════════════════╝")
        report = "\n".join(lines)
        print(report)
        return report

    def __repr__(self) -> str:
        t = self.session_total
        return f"TokenCounter(calls={self.call_count}, total_tokens={t.total_tokens})"
