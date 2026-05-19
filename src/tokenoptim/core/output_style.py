"""
OutputCompressor — system-prompt injection to compress LLM outputs.

Inspired by caveman (https://github.com/juliusbrussee/caveman) but extended with:
  - 6 compression levels
  - Auto-clarity fallback for safety-critical messages
  - Multi-provider system prompt injection
  - Measurable token budget enforcement via max_output_tokens hints

Typical output token reduction: 40-75% depending on level.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class CompressionLevel(str, Enum):
    """Output compression intensity levels."""

    OFF = "off"           # No compression — passthrough
    LITE = "lite"         # ~20% reduction: strip pleasantries
    STANDARD = "standard" # ~40% reduction: caveman-lite mode (DEFAULT)
    FULL = "full"         # ~60% reduction: caveman-full mode
    ULTRA = "ultra"       # ~70% reduction: extreme compression
    ANCIENT = "ancient"   # ~75% reduction: single-syllable preference, abbreviations everywhere


# ---------------------------------------------------------------------------
# System prompt fragments for each level
# ---------------------------------------------------------------------------

_BASE_SAFETY = """
EXCEPTION — always use full clear prose for:
- Security warnings or irreversible actions
- Multi-step sequences where ambiguity risks harm
- When user repeats a question (they may be confused)
Resume compression after the critical section.
""".strip()

_PROMPTS: dict[CompressionLevel, str] = {
    CompressionLevel.OFF: "",

    CompressionLevel.LITE: f"""
Respond concisely. Skip greetings, sign-offs, and filler phrases like "Great question!" or
"Certainly!" or "Hope this helps". Get straight to the answer. Keep all technical accuracy.
{_BASE_SAFETY}
""".strip(),

    CompressionLevel.STANDARD: f"""
COMPRESSION MODE: STANDARD
Speak like a smart, terse engineer. Rules:
- No pleasantries, greetings, or sign-offs
- Drop hedging ("I think", "perhaps", "it seems", "maybe")
- Drop filler ("in order to" → "to", "it is important to note that" → note:)
- Use fragments where meaning is clear ("Checking config" not "I am now checking the config")
- Prefer active voice, short sentences
- Lists > paragraphs when listing things
- All code, commands, paths, errors: exact and complete
{_BASE_SAFETY}
""".strip(),

    CompressionLevel.FULL: f"""
COMPRESSION MODE: FULL (caveman)
Talk like caveman. Smart caveman — knows code, knows systems.
Rules:
- Drop all articles (a, an, the) unless ambiguous
- Drop subject pronouns when obvious ("Run this" not "You should run this")
- No filler. No hedge. No pleasantry.
- Fragment sentences ok. Meaning > grammar.
- Abbreviate freely: config, impl, fn, var, msg, err, auth, req, res, db
- Bullets > prose always
- Code blocks: exact, never compressed
- Numbers: use digits (3 not three)
{_BASE_SAFETY}
""".strip(),

    CompressionLevel.ULTRA: f"""
COMPRESSION MODE: ULTRA
Max compression. Every word earns its place.
- Articles: gone
- Pronouns: gone when clear
- Be/have/do auxiliaries: drop if meaning survives
- Prepositions: omit if meaning clear from structure
- Use symbols: → for "leads to / results in", & for "and", w/ for "with", w/o for "without"
- Use colons and dashes aggressively to replace connective tissue
- Code: always exact
- Emit facts only. No meta-commentary on what you're doing.
{_BASE_SAFETY}
""".strip(),

    CompressionLevel.ANCIENT: f"""
COMPRESSION MODE: ANCIENT
Extreme compression. Stone tablet space is precious.
- 1-syllable words preferred. Longer word only if no short synonym.
- No articles. No aux verbs. No pronouns.
- Abbreviate all common terms: fn, var, cfg, db, auth, svc, req, res, err, msg, tok
- Symbols everywhere: → & w/ w/o != == ≤ ≥ + - * /
- Numbers only digits. Units abbreviated: s, ms, kb, mb, gb
- Bullets only. No prose paragraphs.
- Code: exact, never touched.
- Skip transition words entirely.
{_BASE_SAFETY}
""".strip(),
}


class OutputCompressor:
    """
    Inject a compression system prompt to reduce LLM output tokens.

    Parameters
    ----------
    level : CompressionLevel | str
        How aggressively to compress output. Defaults to STANDARD (~40% reduction).
    max_output_tokens : int | None
        If set, appends a token budget hint to the system prompt.
    custom_exceptions : list[str] | None
        Extra topics where the model should revert to full prose.

    Example
    -------
    >>> oc = OutputCompressor(level="full")
    >>> system_prompt = oc.build_system_prompt("You are a helpful assistant.")
    >>> # Pass system_prompt to your LLM client
    """

    def __init__(
        self,
        level: CompressionLevel | str = CompressionLevel.STANDARD,
        max_output_tokens: Optional[int] = None,
        custom_exceptions: Optional[list[str]] = None,
    ) -> None:
        self.level = CompressionLevel(level)
        self.max_output_tokens = max_output_tokens
        self.custom_exceptions = custom_exceptions or []

    def build_system_prompt(self, base_system: str = "") -> str:
        """Return a system prompt with the compression directive prepended."""
        compression_directive = _PROMPTS[self.level]
        if not compression_directive:
            return base_system

        parts = [compression_directive]

        if self.max_output_tokens:
            parts.append(
                f"\nBUDGET: Target ≤{self.max_output_tokens} tokens in your response. "
                "Be ruthlessly concise but complete."
            )

        if self.custom_exceptions:
            exceptions_str = "; ".join(self.custom_exceptions)
            parts.append(
                f"\nALWAYS use full prose for: {exceptions_str}"
            )

        directive = "\n".join(parts)

        if base_system:
            return f"{directive}\n\n---\n\n{base_system}"
        return directive

    def patch_messages(self, messages: list[dict], base_system: str = "") -> list[dict]:
        """
        Inject the compression directive into a messages list.
        Handles both 'system' role messages and Anthropic-style system param.

        Returns a new messages list with the system prompt injected/prepended.
        """
        if self.level == CompressionLevel.OFF:
            return messages

        system_prompt = self.build_system_prompt(base_system)
        result = list(messages)

        # If there's already a system message at index 0, merge
        if result and result[0].get("role") == "system":
            existing = result[0].get("content", "")
            result[0] = {
                **result[0],
                "content": f"{system_prompt}\n\n{existing}".strip(),
            }
        else:
            result.insert(0, {"role": "system", "content": system_prompt})

        return result

    @staticmethod
    def estimate_savings(level: CompressionLevel | str) -> str:
        """Return a human-readable estimate of token savings for the given level."""
        level = CompressionLevel(level)
        estimates = {
            CompressionLevel.OFF: "0%",
            CompressionLevel.LITE: "~20%",
            CompressionLevel.STANDARD: "~40%",
            CompressionLevel.FULL: "~60%",
            CompressionLevel.ULTRA: "~70%",
            CompressionLevel.ANCIENT: "~75%",
        }
        return estimates[level]

    def __repr__(self) -> str:
        return (
            f"OutputCompressor(level={self.level.value!r}, "
            f"max_output_tokens={self.max_output_tokens})"
        )
