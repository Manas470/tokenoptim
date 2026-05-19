"""
MLPromptCompressor — high-quality prompt compression using LLMLingua.

LLMLingua (Microsoft Research) uses a small language model (phi-2 or similar)
to compute token-level perplexity and selectively drop low-importance tokens,
achieving 4–20× compression with minimal quality loss.

This module wraps llmlingua as an optional dependency. If not installed,
it falls back gracefully to the regex-based PromptCompressor.

Install
-------
    pip install "tokenoptim[ml]"
    # or: pip install llmlingua

Usage
-----
>>> from tokenoptim.core.ml_compressor import MLPromptCompressor
>>>
>>> c = MLPromptCompressor(target_token_rate=0.5)  # keep 50% of tokens
>>> compressed, stats = c.compress(long_prompt)
>>> print(stats)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from tokenoptim.core.compressor import PromptCompressor, CompressionStats

logger = logging.getLogger("tokenoptim.ml_compressor")


@dataclass
class MLCompressionStats:
    original_tokens: int = 0
    compressed_tokens: int = 0
    method: str = "ml"

    @property
    def reduction_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return round((1 - self.compressed_tokens / self.original_tokens) * 100, 1)

    def __str__(self) -> str:
        return (
            f"[{self.method}] ~Tokens: {self.original_tokens} → {self.compressed_tokens} "
            f"({self.reduction_pct}% ↓)"
        )


class MLPromptCompressor:
    """
    ML-based prompt compressor using LLMLingua.

    Parameters
    ----------
    target_token_rate : float
        Fraction of tokens to keep (0.3 = keep 30% → 70% reduction).
        Range: 0.1–0.9. Default 0.5.
    model_name : str
        Small LM used to compute perplexity for token selection.
        Default: "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
        (fast, good quality). Also supports "microsoft/phi-2".
    use_sentence_level : bool
        If True, drop entire sentences instead of individual tokens.
        Less distortion for long documents.
    fallback_level : str
        If llmlingua is not installed, fall back to this regex level.
        "light" | "medium" | "full". Default "full".

    Example
    -------
    >>> c = MLPromptCompressor(target_token_rate=0.4)
    >>> compressed, stats = c.compress(
    ...     "Could you please help me understand..." * 50
    ... )
    >>> print(stats)
    # [ml] ~Tokens: 1200 → 480 (60.0% ↓)
    """

    def __init__(
        self,
        target_token_rate: float = 0.5,
        model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_sentence_level: bool = False,
        fallback_level: str = "full",
        device: Optional[str] = None,
    ) -> None:
        if not 0.1 <= target_token_rate <= 0.9:
            raise ValueError("target_token_rate must be between 0.1 and 0.9")

        self.target_token_rate = target_token_rate
        self.model_name = model_name
        self.use_sentence_level = use_sentence_level
        self._fallback = PromptCompressor(level=fallback_level)
        self._compressor = None
        self._backend = "regex"  # will update on successful init

        try:
            from llmlingua import PromptCompressor as LLMLinguaCompressor
            self._compressor = LLMLinguaCompressor(
                model_name=model_name,
                use_llmlingua2=True,
                device_map=device or "cpu",
            )
            self._backend = "llmlingua"
            logger.info("MLPromptCompressor: using LLMLingua backend (%s)", model_name)
        except ImportError:
            logger.warning(
                "llmlingua not installed — falling back to regex compression (level=%s). "
                "Install with: pip install llmlingua",
                fallback_level,
            )
        except Exception as e:
            logger.warning(
                "LLMLingua failed to load (%s) — falling back to regex. Error: %s",
                model_name,
                e,
            )

    @property
    def backend(self) -> str:
        """Returns 'llmlingua' or 'regex' depending on what's available."""
        return self._backend

    def compress(self, text: str) -> tuple[str, MLCompressionStats]:
        """
        Compress a prompt string.

        Returns (compressed_text, MLCompressionStats).
        """
        orig_tokens = max(1, len(text) // 4)

        if self._compressor is not None:
            return self._compress_llmlingua(text, orig_tokens)
        else:
            return self._compress_fallback(text, orig_tokens)

    def compress_messages(
        self, messages: list[dict]
    ) -> tuple[list[dict], MLCompressionStats]:
        """
        Compress a messages list. System and user turns are compressed;
        assistant turns are left intact.
        """
        total_orig = 0
        total_comp = 0
        result = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("system", "user") and isinstance(content, str):
                compressed_text, stats = self.compress(content)
                total_orig += stats.original_tokens
                total_comp += stats.compressed_tokens
                result.append({**msg, "content": compressed_text})
            else:
                tokens = max(1, len(str(content)) // 4)
                total_orig += tokens
                total_comp += tokens
                result.append(msg)

        return result, MLCompressionStats(
            original_tokens=total_orig,
            compressed_tokens=total_comp,
            method=self._backend,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compress_llmlingua(self, text: str, orig_tokens: int) -> tuple[str, MLCompressionStats]:
        try:
            result = self._compressor.compress_prompt(  # type: ignore[union-attr]
                text,
                rate=self.target_token_rate,
                force_tokens=["\n", ".", "!", "?"],  # always keep structure markers
                drop_consecutive=True,
            )
            compressed = result.get("compressed_prompt", text)
            comp_tokens = result.get("origin_tokens", orig_tokens) - result.get("savings", 0)
            if comp_tokens <= 0:
                comp_tokens = max(1, len(compressed) // 4)
            return compressed, MLCompressionStats(
                original_tokens=orig_tokens,
                compressed_tokens=comp_tokens,
                method="llmlingua",
            )
        except Exception as e:
            logger.warning("LLMLingua compression failed: %s — using fallback", e)
            return self._compress_fallback(text, orig_tokens)

    def _compress_fallback(self, text: str, orig_tokens: int) -> tuple[str, MLCompressionStats]:
        compressed, stats = self._fallback.compress(text)
        return compressed, MLCompressionStats(
            original_tokens=stats.original_approx_tokens,
            compressed_tokens=stats.compressed_approx_tokens,
            method="regex-fallback",
        )
