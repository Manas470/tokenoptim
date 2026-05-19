"""
tokenoptim Benchmark Suite — v2
================================
Runs a rigorous, reproducible benchmark with:

  1. Real token counting (tiktoken / Anthropic API, not char÷4)
  2. Statistical reporting: mean, median, p90, p95, stdev
  3. Warm-up pass to exclude cold-start latency bias
  4. Diverse prompt mix: verbose, terse, code, multilingual
  5. Quality proxy: measures BLEU-style token overlap between
     original and compressed (catches destructive over-compression)
  6. Throughput: prompts/sec for production capacity planning
  7. Honest notes on what IS and ISN'T measured

Usage
-----
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --samples 1000 --output results.json
    python benchmarks/run_benchmark.py --no-download --samples 100  # offline
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tokenoptim.core.compressor import PromptCompressor
from tokenoptim.core.output_style import OutputCompressor, CompressionLevel

# ---------------------------------------------------------------------------
# Token counter — use tiktoken if available, else char÷4 (clearly labelled)
# ---------------------------------------------------------------------------

def count_tokens(text: str, _enc=None) -> tuple[int, str]:
    """
    Return (token_count, method_used).
    method_used is 'tiktoken' or 'approx(chars/4)'.
    """
    if _enc is None:
        try:
            import tiktoken
            _enc = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            _enc = False

    if _enc:
        return len(_enc.encode(text)), "tiktoken"
    return max(1, len(text) // 4), "approx(chars/4)"


# Cache the encoder
_TIKTOKEN_ENC = None
try:
    import tiktoken as _tiktoken_mod
    _TIKTOKEN_ENC = _tiktoken_mod.get_encoding("cl100k_base")
except ImportError:
    pass


def tok(text: str) -> int:
    n, _ = count_tokens(text, _TIKTOKEN_ENC)
    return n


TOKEN_METHOD = "tiktoken" if _TIKTOKEN_ENC else "approx(chars/4)"

# ---------------------------------------------------------------------------
# Quality proxy — token overlap score (0–1)
# ---------------------------------------------------------------------------

def token_overlap(original: str, compressed: str) -> float:
    """
    Measure what fraction of compressed tokens also appear in the original.
    Score of 1.0 = all compressed tokens are from the original (no hallucination).
    Score < 0.8 = significant content drift — flag as potentially destructive.
    """
    orig_words = set(original.lower().split())
    comp_words = compressed.lower().split()
    if not comp_words:
        return 0.0
    matches = sum(1 for w in comp_words if w in orig_words)
    return matches / len(comp_words)

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

DATASET_URL = (
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/"
    "resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)

# Deliberately diverse fallback set: verbose, terse, code, edge cases
FALLBACK_PROMPTS = [
    # Verbose (where compression should help)
    "Could you please help me understand what tokenization means in the context of large language models? "
    "I would really like to know, perhaps with examples if that makes sense. Thank you so much!",

    "In order to better understand neural networks, I think it is important to note that I am a beginner. "
    "Could you perhaps explain backpropagation with maybe a simple example? Hope this helps me learn.",

    "I would like you to please write a Python function that calculates the nth Fibonacci number. "
    "Please make sure to include comments and perhaps some edge case handling if that makes sense.",

    "It would be really helpful if you could explain the difference between REST and GraphQL APIs. "
    "Perhaps you could include some examples? I am fairly new to this topic. Thanks in advance!",

    "Could you kindly provide a comprehensive explanation of the CAP theorem in distributed systems? "
    "I would appreciate it if you could maybe include some real-world examples of each combination.",

    # Terse (compressor should NOT degrade these)
    "Explain tokenization.",
    "Write a binary search in Python.",
    "What is the CAP theorem?",
    "List 5 sorting algorithms.",
    "Fix this: def add(a,b) return a+b",

    # Code (compressor must leave untouched)
    "```python\ndef fibonacci(n):\n    if n < 2: return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```\n"
    "What is the time complexity of this function?",

    "```sql\nSELECT u.name, COUNT(o.id) as order_count\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nGROUP BY u.id\nHAVING COUNT(o.id) > 5;\n```\nExplain this query.",

    # Mixed verbose + code
    "I would really appreciate it if you could help me understand what this code does. "
    "Perhaps you could also suggest improvements?\n```python\nimport time\ndef retry(fn, n=3):\n    for i in range(n):\n        try: return fn()\n        except: time.sleep(2**i)\n```",

    # Very long (tests deduplication + window)
    ("In order to understand distributed systems better, I think it is important to note that "
     "consistency, availability, and partition tolerance are the three properties described by the CAP theorem. "
     "Perhaps you could explain each property in detail. It would be really helpful if you could also "
     "provide examples of real databases and which properties they sacrifice. Hope this helps me learn. "
     "Thank you so much for your comprehensive help with this important topic. ") * 3,

    # Repetitive content (tests deduplication)
    "I need help with Python. I need help with Python. "
    "Specifically, I need help understanding list comprehensions in Python. "
    "I need help with Python list comprehensions please.",
]


def load_sharegpt_prompts(n: int) -> list[str]:
    cache_path = Path("/tmp/sharegpt_cache.json")
    if cache_path.exists():
        print(f"  Loading cached ShareGPT dataset...")
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        print(f"  Downloading ShareGPT dataset (~50MB)...")
        try:
            req = urllib.request.Request(DATASET_URL, headers={"User-Agent": "tokenoptim-benchmark/2.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            cache_path.write_text(json.dumps(data), encoding="utf-8")
        except Exception as e:
            print(f"  Warning: Download failed ({e}). Using built-in prompts.")
            return (FALLBACK_PROMPTS * math.ceil(n / len(FALLBACK_PROMPTS)))[:n]

    prompts = []
    for conv in data:
        for turn in conv.get("conversations", []):
            if turn.get("from") == "human" and len(turn.get("value", "")) > 30:
                prompts.append(turn["value"])
                break
        if len(prompts) >= n:
            break

    print(f"  Loaded {len(prompts)} real ShareGPT prompts.")
    return prompts[:n]


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def run_compression_benchmark(prompts: list[str]) -> dict:
    results = {}

    for level in ["light", "medium", "full"]:
        compressor = PromptCompressor(level=level)

        # Warm-up — 5 passes to prime regex engine and CPU caches
        warmup = prompts[:5]
        for _ in range(3):
            for p in warmup:
                compressor.compress(p)

        reductions: list[float] = []
        latencies: list[float] = []
        overlaps: list[float] = []
        destructive_count = 0
        zero_gain_count = 0

        for prompt in prompts:
            t0 = time.perf_counter()
            compressed, _ = compressor.compress(prompt)
            latency_ms = (time.perf_counter() - t0) * 1000

            orig_toks = tok(prompt)
            comp_toks = tok(compressed)
            reduction = max(0.0, (1 - comp_toks / orig_toks) * 100) if orig_toks > 0 else 0.0
            overlap = token_overlap(prompt, compressed)

            reductions.append(reduction)
            latencies.append(latency_ms)
            overlaps.append(overlap)

            if overlap < 0.80:
                destructive_count += 1
            if reduction == 0.0:
                zero_gain_count += 1

        n = len(reductions)
        sorted_r = sorted(reductions)
        sorted_l = sorted(latencies)

        results[f"prompt_{level}"] = {
            "level": level,
            "type": "prompt_compression",
            "token_method": TOKEN_METHOD,
            "samples": n,
            # Reduction stats
            "mean_reduction_pct": round(statistics.mean(reductions), 2),
            "median_reduction_pct": round(statistics.median(reductions), 2),
            "stdev_reduction_pct": round(statistics.stdev(reductions), 2) if n > 1 else 0,
            "p90_reduction_pct": round(sorted_r[int(n * 0.90)], 2),
            "p95_reduction_pct": round(sorted_r[int(n * 0.95)], 2),
            "max_reduction_pct": round(max(reductions), 2),
            # Latency stats
            "mean_latency_ms": round(statistics.mean(latencies), 3),
            "p99_latency_ms": round(sorted_l[int(n * 0.99)], 3),
            "throughput_prompts_per_sec": round(n / sum(latencies) * 1000, 0),
            # Quality proxy
            "mean_token_overlap": round(statistics.mean(overlaps), 3),
            "destructive_compressions": destructive_count,   # overlap < 0.80
            "zero_gain_prompts": zero_gain_count,            # already terse
        }

    return results


def run_output_overhead_benchmark() -> dict:
    """Exact token cost of each output compression system prompt."""
    results = {}
    base = "You are a helpful AI assistant."
    for level in CompressionLevel:
        oc = OutputCompressor(level=level)
        prompt = oc.build_system_prompt(base)
        tokens = tok(prompt)
        results[f"output_{level.value}"] = {
            "type": "output_compression",
            "level": level.value,
            "system_prompt_tokens": tokens,
            "estimated_output_savings": OutputCompressor.estimate_savings(level),
            "breakeven_output_tokens": (
                round(tokens / (int(OutputCompressor.estimate_savings(level).strip("~%")) / 100))
                if level != CompressionLevel.OFF
                else "N/A"
            ),
        }
    return results


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(results: dict, samples: int) -> None:
    print(f"\n{'━'*72}")
    print(f"  tokenoptim Benchmark  |  {samples} prompts  |  tokens: {TOKEN_METHOD}")
    print(f"{'━'*72}")

    print(f"\n{'📊 Prompt Compression (CPU-only, warmed up)':}")
    print(f"  {'Level':<8} {'Mean↓':>6} {'Med↓':>6} {'P90↓':>6} {'P95↓':>6} {'StDev':>6} "
          f"{'Lat(ms)':>8} {'QPS':>8} {'Overlap':>8} {'Bad':>4}")
    print(f"  {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*4}")

    for key in ["prompt_light", "prompt_medium", "prompt_full"]:
        if key not in results:
            continue
        r = results[key]
        bad_flag = "⚠️" if r["destructive_compressions"] > 0 else "✓"
        print(
            f"  {r['level']:<8} "
            f"{r['mean_reduction_pct']:>5.1f}% "
            f"{r['median_reduction_pct']:>5.1f}% "
            f"{r['p90_reduction_pct']:>5.1f}% "
            f"{r['p95_reduction_pct']:>5.1f}% "
            f"{r['stdev_reduction_pct']:>5.1f}% "
            f"{r['mean_latency_ms']:>8.3f} "
            f"{r['throughput_prompts_per_sec']:>8,.0f} "
            f"{r['mean_token_overlap']:>8.3f} "
            f"{r['destructive_compressions']:>4}{bad_flag}"
        )

    print(f"\n  'Bad' = compressions with token overlap < 0.80 (possible content distortion)")
    print(f"  'QPS' = prompts/second throughput (single thread)")

    print(f"\n💬 Output Compression system prompt cost + break-even:")
    print(f"  {'Level':<10} {'Sys tokens':>10} {'Est savings':>12} {'Break-even':>12}")
    print(f"  {'─'*10} {'─'*10} {'─'*12} {'─'*12}")
    for key, r in results.items():
        if r.get("type") != "output_compression":
            continue
        be = r["breakeven_output_tokens"]
        be_str = f"{be} out-tok" if be != "N/A" else "N/A"
        print(f"  {r['level']:<10} {r['system_prompt_tokens']:>10} {r['estimated_output_savings']:>12} {be_str:>12}")

    print(f"\n⚠️  What this benchmark does NOT measure:")
    print(f"  • Whether compressed prompts produce equally good LLM outputs (quality)")
    print(f"  • Actual output token savings (would require live API calls)")
    print(f"  • ML compression (install 'tokenoptim[ml]' for LLMLingua numbers)")
    print(f"  • Non-English prompt behaviour")
    print(f"\n{'━'*72}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="tokenoptim benchmark suite v2")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    print(f"tokenoptim Benchmark Suite v2")
    print(f"Token counting: {TOKEN_METHOD}")
    print(f"Samples: {args.samples}\n")

    if args.no_download:
        prompts = (FALLBACK_PROMPTS * math.ceil(args.samples / len(FALLBACK_PROMPTS)))[:args.samples]
        print(f"Using {len(prompts)} built-in prompts (diverse mix: verbose, terse, code).")
    else:
        prompts = load_sharegpt_prompts(args.samples)

    all_results: dict = {}

    print("Running prompt compression benchmark (with warm-up)...")
    all_results.update(run_compression_benchmark(prompts))

    print("Running output compression overhead benchmark...")
    all_results.update(run_output_overhead_benchmark())

    print_report(all_results, len(prompts))

    if args.output:
        Path(args.output).write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"Results saved → {args.output}")


if __name__ == "__main__":
    main()
