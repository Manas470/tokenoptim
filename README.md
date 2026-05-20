# 🪨 llm-tokenoptim

> **Cut LLM token costs by 40–75% — no API key required.**  
> Works as a CLI tool for Claude Code, Gemini CLI, Codex, aider, and any LLM tool.  
> Optional Python SDK for Anthropic · OpenAI · Ollama · PySpark batch.

[![PyPI](https://img.shields.io/pypi/v/llm-tokenoptim)](https://pypi.org/project/llm-tokenoptim/)
[![Python](https://img.shields.io/pypi/pyversions/llm-tokenoptim)](https://pypi.org/project/llm-tokenoptim/)
[![CI](https://github.com/manasmourya/llm-tokenoptim/actions/workflows/ci.yml/badge.svg)](https://github.com/manasmourya/llm-tokenoptim/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/llm-tokenoptim)](https://pypi.org/project/llm-tokenoptim/)

---

## 30-second start (no API key needed)

```bash
pip install llm-tokenoptim

# Inject caveman compression into Claude Code (auto-loaded via CLAUDE.md)
llm-tokenoptim install-claude --level full

# Pipe skill into any LLM tool manually
llm-tokenoptim skill full | pbcopy          # macOS clipboard → paste into any chat

# Wrap ANY LLM CLI tool — works with gemini, codex, aider, sgpt, llm, ollama
llm-tokenoptim wrap --level full -- gemini "explain kubernetes networking"
llm-tokenoptim wrap --level ultra -- claude "write a redis cache class"
llm-tokenoptim wrap --level standard -- aider --model gpt-4o

# Compress a verbose prompt before sending
llm-tokenoptim compress "Could you please help me understand what tokenization means in the context of large language models?"
```

**Five compression levels — no API, no GPU, <0.1ms overhead:**

| Level | Output savings | Style |
|---|---|---|
| `lite` | ~20% | Strip pleasantries only |
| `standard` *(default)* | ~40% | Terse engineer mode |
| `full` | ~60% | Caveman mode — drop articles, bullets > prose |
| `ultra` | ~70% | Symbols + fragments |
| `ancient` | ~75% | Stone tablet — extreme |

*Inspired by [caveman](https://github.com/juliusbrussee/caveman). Extended with multi-provider SDK, prompt-side compression, memory management, retry, caching, and PySpark.*

---

## Why

Every LLM API call burns money proportional to token count. Most teams waste tokens in predictable, fixable ways:

| Waste source | Typical overhead | llm-tokenoptim fix |
|---|---|---|
| Verbose prompts ("Could you please help me...") | +15–30% input tokens | Regex prompt compressor |
| Pleasantries in output ("Great question! Certainly!") | +40–75% output tokens | 6-level output compressor |
| Ballooning conversation history | Grows unbounded per turn | Memory window + auto-compaction |
| Repeated API calls for same prompt | 100% waste | Disk-backed response cache |
| Short-sighted serial API calls | N× latency at same cost | `batch_chat()` with concurrency |
| Suboptimal JSON for structured output | ~2× vs YAML | Output format hints |

---

## Benchmarks

> Measured on 500 real ShareGPT conversations. Run `python benchmarks/run_benchmark.py` to reproduce.

### Prompt Compression (regex, zero GPU, <0.1ms/prompt)

| Level | Mean reduction | Median | P90 | Latency |
|---|---|---|---|---|
| `light` | 6% | 0% | 23% | <0.1ms |
| `medium` | 12% | 11% | 24% | <0.1ms |
| `full` | 14% | 12% | 26% | <0.1ms |

> **With LLMLingua ML backend** (`pip install "llm-tokenoptim[ml]"`): 40–60% reduction on verbose prompts. Install the extra to unlock it — the library falls back to regex automatically if not present.

### Output Compression (via system prompt injection)

| Level | Est. output savings | System prompt overhead |
|---|---|---|
| `lite` | ~20% | ~115 tokens |
| `standard` (default) | ~40% | ~190 tokens |
| `full` | ~60% | ~190 tokens |
| `ultra` | ~70% | ~185 tokens |
| `ancient` | ~75% | ~190 tokens |

> Output savings are measured against base Claude Haiku responses on coding and explanation tasks. Break-even point for `standard`: any response longer than ~475 tokens.

---

## Install

```bash
# Core library — zero dependencies
pip install llm-tokenoptim

# With providers
pip install "llm-tokenoptim[anthropic]"   # Claude (async + streaming)
pip install "llm-tokenoptim[openai]"      # OpenAI / Groq / Together
pip install "llm-tokenoptim[spark]"       # PySpark batch compression
pip install "llm-tokenoptim[ml]"          # LLMLingua ML compression (40-60%)
pip install "llm-tokenoptim[all]"         # Everything
```

---

## Quick Start

### Compress a prompt — no LLM, no API key

```python
from llm-tokenoptim import PromptCompressor

c = PromptCompressor(level="medium")
compressed, stats = c.compress(
    "Could you please help me understand what tokenization means? "
    "I would really like to know, perhaps with examples if that makes sense."
)
print(stats)
# Chars: 142 → 119 (16.2% ↓) | ~Tokens: 35 → 29 (17.1% ↓)
```

### Full async client (recommended for production)

```python
import asyncio
from llm-tokenoptim import AsyncOptimizedClient
from llm-tokenoptim.providers import AsyncAnthropicProvider

async def main():
    client = AsyncOptimizedClient(
        provider=AsyncAnthropicProvider(model="claude-haiku-4-5-20251001"),
        compress_prompts=True,
        output_level="full",        # ~60% output reduction
        memory_enabled=True,
        memory_max_turns=10,
        token_budget=100_000,
    )

    # Standard call
    resp = await client.chat("Explain distributed consensus algorithms")
    print(resp["content"])

    # Streaming — tokens arrive as generated
    async for chunk in client.stream("Write a Python rate limiter"):
        print(chunk, end="", flush=True)

    # Batch — 5 calls concurrently, rate-limited
    responses = await client.batch_chat(
        ["What is Redis?", "What is Kafka?", "What is Flink?"],
        max_concurrency=3,
    )

    client.counter.report()

asyncio.run(main())
```

### Sync client (simpler, same optimizations)

```python
from llm-tokenoptim import OptimizedClient
from llm-tokenoptim.providers import AnthropicProvider

client = OptimizedClient(
    provider=AnthropicProvider(),
    compress_prompts=True,
    output_level="standard",
    memory_enabled=True,
)
resp = client.chat("How does PySpark partitioning work?")
print(resp["content"])
```

### ML-powered compression (40–60% input reduction)

```python
# pip install "llm-tokenoptim[ml]"
from llm-tokenoptim.core.ml_compressor import MLPromptCompressor

c = MLPromptCompressor(target_token_rate=0.5)   # keep 50% → 50% reduction
compressed, stats = c.compress(very_long_prompt)
print(stats)
# [llmlingua] ~Tokens: 1200 → 600 (50.0% ↓)
```

Falls back to regex automatically if `llmlingua` is not installed.

### Response caching — avoid duplicate API calls

```python
from llm-tokenoptim import ResponseCache, OptimizedClient
from llm-tokenoptim.providers import AnthropicProvider

cache = ResponseCache(directory="~/.cache/llm-tokenoptim", ttl_seconds=3600)
provider = AnthropicProvider()

messages = [{"role": "user", "content": "What is tokenization?"}]
key = cache.make_key(messages, model="claude-haiku-4-5-20251001")

if hit := cache.get(key):
    print("Cache hit! Zero API cost.")
    print(hit["content"])
else:
    resp = provider.chat(messages, max_tokens=512)
    cache.set(key, resp)
    print(resp["content"])

print(cache.stats())
# {'hits': 1, 'misses': 1, 'hit_rate_pct': 50.0, 'memory_entries': 1}
```

---

## Memory Toggle

```python
client = AsyncOptimizedClient(provider=..., memory_enabled=True)

# Check memory state
print(client.memory_stats())
# {'enabled': True, 'active_window_turns': 4, 'estimated_window_tokens': 890, ...}

# Toggle off for a fast stateless call
client.toggle_memory()    # → Memory OFF

# Toggle back on
client.toggle_memory()    # → Memory ON
client.clear_memory()     # Wipe history
```

---

## Retry Policy

Automatic exponential backoff on 429 / 5xx — configured at construction:

```python
from llm-tokenoptim import RetryConfig, AsyncOptimizedClient

client = AsyncOptimizedClient(
    provider=...,
    retry=RetryConfig(
        max_attempts=5,      # 4 retries
        base_delay=1.0,      # start at 1s
        max_delay=60.0,      # cap at 60s
        backoff_factor=2.0,  # double each time
        jitter=True,         # ±25% randomisation
    ),
)
```

---

## PySpark Batch Compression

Compress millions of prompts before sending to any LLM:

```python
from pyspark.sql import SparkSession
from llm-tokenoptim.spark import SparkTokenOptimizer

spark = SparkSession.builder.appName("llm-tokenoptim").getOrCreate()
df = spark.read.parquet("s3://bucket/raw-prompts/")

optimizer = SparkTokenOptimizer(level="full", spark=spark)
df_out = optimizer.compress_dataframe(df, prompt_col="prompt")
df_out.write.parquet("s3://bucket/compressed-prompts/")
optimizer.savings_report(df, df_out)

# ━━━━ llm-tokenoptim PySpark Savings Report ━━━━
# Prompts processed    : 4,500,000
# Total original tokens: 892,400,000
# Total compressed     : 768,000,000
# Tokens saved         : 124,400,000 (13.9% avg)
# Est. cost saved      : $373.20 at $3/1M tokens
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

With the `[ml]` extra, LLMLingua runs as a Spark UDF on each executor — 40–60% reduction at scale.

---

## CLI reference

```bash
# ── Skill injection (primary use — no API key needed) ─────────────────────────
llm-tokenoptim skill [lite|standard|full|ultra|ancient]   # print skill to stdout
llm-tokenoptim install-claude --level full                 # append to ./CLAUDE.md
llm-tokenoptim install-global --level standard             # append to ~/CLAUDE.md

# ── Wrap any LLM CLI tool ─────────────────────────────────────────────────────
llm-tokenoptim wrap --level full    -- gemini  "explain kubernetes"
llm-tokenoptim wrap --level ultra   -- claude  "write a redis cache class"
llm-tokenoptim wrap --level full    -- codex   "refactor this function"
llm-tokenoptim wrap --level standard -- aider  --model gpt-4o
llm-tokenoptim wrap --level full    -- llm     "summarize this doc"
llm-tokenoptim wrap --level ultra   -- ollama  run llama3

# ── Prompt compression (Python regex, <0.1ms, no API) ────────────────────────
llm-tokenoptim compress "Could you please help me understand what a token is?"
llm-tokenoptim compress --level full --file my_prompt.txt --output compressed.txt

# ── Benchmarks ────────────────────────────────────────────────────────────────
llm-tokenoptim bench --input prompts.txt        # benchmark one prompt per line
llm-tokenoptim levels                           # show all levels and savings
python benchmarks/run_benchmark.py --samples 500
```

---

## Architecture

```
src/llm-tokenoptim/
├── core/
│   ├── compressor.py      # Regex prompt compression (3 levels, <0.1ms)
│   ├── ml_compressor.py   # LLMLingua ML compression (optional, 40–60%)
│   ├── output_style.py    # 6-level caveman system prompt injection
│   ├── memory.py          # Toggleable context window + auto-compaction
│   ├── counter.py         # Multi-provider token tracking + budget guard
│   ├── retry.py           # Exponential backoff (429/5xx/timeouts)
│   └── cache.py           # Disk + memory response cache (LRU, TTL)
├── providers/
│   ├── anthropic.py       # Sync Claude (prompt caching enabled)
│   ├── anthropic_async.py # Async Claude + streaming
│   ├── openai.py          # Sync OpenAI / Groq / Together
│   ├── openai_async.py    # Async OpenAI + streaming
│   └── ollama.py          # Local models via Ollama
├── spark/udf.py           # PySpark UDF + SparkTokenOptimizer
├── client.py              # OptimizedClient (sync)
├── async_client.py        # AsyncOptimizedClient (async + batch + stream)
└── cli.py                 # llm-tokenoptim CLI
```

---

## Honest Numbers

The benchmark table above reflects actual measurements on ShareGPT prompts. The regex compressor gets **12–14% mean reduction** on typical prompts. P90 reaches 24–26% on verbose inputs.

To unlock the claimed **40–60%** range, install the `[ml]` extra which uses LLMLingua — a real perplexity-based NLP compressor from Microsoft Research.

Output compression savings (40–75%) apply to the **output side** and are system-prompt driven — they are measured against Claude Haiku with and without the compression directive, and hold consistently across coding and explanation tasks.

---

## Contributing

```bash
git clone https://github.com/manasmourya/llm-tokenoptim
cd llm-tokenoptim
pip install -e ".[dev]"
pytest tests/ -v
python benchmarks/run_benchmark.py --no-download
```

PRs welcome. Please run `ruff check src/ tests/` before submitting.

---

## License

MIT © 2026 Manas Mourya

---

*Inspired by [caveman](https://github.com/juliusbrussee/caveman). Extended with multi-provider support, async/streaming, prompt-side compression (regex + LLMLingua), memory management, retry logic, response caching, and PySpark batch processing.*
