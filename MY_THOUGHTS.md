# My Thoughts — Building llm-tokenoptim

*A full walkthrough of why I built this, how I built it, what I considered and rejected, and what I learned.*

---

## The Problem I Was Actually Trying to Solve

I was using Claude Code every day and my API bill kept growing. I started looking at what I was actually paying for and the answer was embarrassing.

A typical response started like this:

> "Great question! Certainly, I'd be absolutely delighted to help you understand that concept today. Let me break this down in a clear and comprehensive way..."

I was paying per token for a language model to be polite to me. That's 30–40 tokens of nothing before it said anything useful.

And that was just the output side. On the input side, I was writing prompts like cover letters — verbose, hedged, apologetic. "Could you please help me understand..." instead of just "explain".

Then I noticed the conversation history growing. By turn 10, I was sending the entire conversation back to the model on every single call — including old messages it barely referenced. The cost grew linearly with conversation length, unbounded.

Then I found caveman — a single clever system prompt that tells Claude to drop articles, use symbols, respond in fragments. It worked. Responses got shorter. But it only worked with Claude, only on the output side, and you had to manually paste it in every time.

That's where llm-tokenoptim started.

---

## What I Built — End to End

### Layer 1: Skill Files (the core, no code needed)

The primary product is five markdown files in `skills/`:

- `lite` — strip pleasantries, ~20% savings
- `standard` — terse engineer mode, ~40% savings  
- `full` — caveman mode, drop articles, bullets over prose, ~60% savings
- `ultra` — symbols and fragments, ~70% savings
- `ancient` — stone tablet mode, ~75% savings

Each file is a system prompt that gets injected into whatever LLM tool you're using. No Python, no API key, no configuration. Claude Code auto-loads `CLAUDE.md`, so `tokenoptim install-claude --level full` is a one-time setup that affects every future session.

This was inspired directly by caveman. I extended the concept to five levels with different trade-offs between readability and compression, and added a safety exception clause to every level — security warnings, irreversible actions, and repeated questions always get full prose regardless of compression level.

### Layer 2: CLI Wrapper

The CLI (`tokenoptim wrap`) detects which LLM tool you're calling by binary name and injects the skill correctly:

- `claude` → writes a temp `CLAUDE.md` and sets `CLAUDE_SYSTEM_PROMPT_FILE`
- `gemini` → `--system` flag
- `codex` → `--system` flag
- `aider` → `--system-prompt` flag
- `sgpt` → stdin pipe
- `llm` (Simon Willison's tool) → `--system` flag
- `ollama` → `--system` flag
- anything else → tries `--system`, falls back to `SYSTEM` env var

Every tool has a different mechanism. The wrapper abstracts all of that. You always type the same command regardless of which model you're using.

### Layer 3: Prompt Compressor

On the input side, I built a regex-based compressor that runs in under 0.1ms per prompt with zero dependencies. Three levels:

- `light` — removes filler openers ("Could you please", "I was wondering if")
- `medium` — also strips hedging ("I think", "it might be", "generally speaking") and verbose connectors
- `full` — also abbreviates common terms, collapses whitespace, deduplicates repeated content

I preserve code blocks by substituting them out before compression and restoring them after — never touch actual code.

Honest numbers: 9% mean reduction, 35% at P90. The mean is modest because most prompts aren't that verbose. When people write fluff, it catches it hard.

### Layer 4: Python SDK

For people building applications, not just using CLI tools:

**MemoryManager** — tracks conversation history with a configurable token budget. When the window fills, it auto-compacts older messages into a summary rather than dropping them. Most teams I've seen just let history grow unbounded and wonder why costs explode by turn 20.

**ResponseCache** — two-level cache: hot in-memory dict (LRU) and cold disk storage (SHA-256-keyed JSON files). Survives process restarts. Cache key is a deterministic hash of the message list, system prompt, and model name. Strips non-serializable fields before persisting.

**RetryConfig** — exponential backoff with jitter for 429s, 500s, 502s, 503s, 504s, and connection errors. Reads `Retry-After` and `x-ratelimit-reset-requests` headers when present rather than sleeping a fixed amount. Most retry implementations I've seen just sleep for 1 second and call it done.

**AsyncOptimizedClient** — full async/await with streaming. `batch_chat()` uses a semaphore to control concurrency — you can fire off 50 concurrent API calls without hammering rate limits.

### Layer 5: PySpark UDF

This is the part I'm most proud of because nobody else has done it.

```python
from tokenoptim.spark import compress_prompts_udf
df = df.withColumn("prompt", compress_prompts_udf("prompt", lit("full")))
```

One line. Every prompt in your Spark dataframe gets compressed before it hits the API. Distributed across your cluster. Zero API calls for the compression itself.

This matters at scale. If you're running LLM inference on 10 million rows — classification, summarization, entity extraction — a 9% mean reduction is 900,000 tokens saved per million. At $15/million tokens that's $13,500 per run. At 35% P90 it's potentially much more on verbose inputs.

---

## The Benchmarking Problem

Early versions of the benchmark used `len(text) / 4` to estimate tokens. That's wrong and I knew it was wrong. I replaced it with tiktoken (`cl100k_base` encoding) which is what OpenAI and Anthropic actually use.

I also added:
- 3 warm-up passes before timing (eliminates JIT and import overhead)
- P90 and P95 alongside mean (mean hides how good it is on verbose inputs)
- Standard deviation (tells you how consistent it is)
- QPS throughput (tells you if it's fast enough for production)
- Token overlap quality proxy (flags compressions that might be destroying content)
- Break-even output token calculation for output compression levels
- An explicit "what this benchmark does NOT measure" section

That last one matters. I've seen too many benchmarks that look great because they measure the thing they're good at. This benchmark explicitly says: we don't measure whether compressed prompts produce equally good outputs. We don't measure actual output token savings. We don't measure non-English behavior. Honest disclosure is more valuable than impressive-looking numbers.

---

## Use Cases

**Individual developers using LLM CLI tools daily**
The install-claude + skill injection path. One command, permanent effect on every future session. Especially useful for Claude Code where CLAUDE.md is auto-loaded.

**Teams building chatbots or assistants**
Memory management + response caching. Conversation history is the silent cost killer that doesn't show up until week 3 when you're suddenly paying 5x what you expected. The memory window + auto-compaction keeps context bounded.

**High-volume API usage with repeated queries**
Response caching gives 100% savings on repeated prompts — support bots, FAQ systems, documentation lookup. The disk-backed cache means it survives restarts and can be shared across processes.

**Data engineering teams running LLM inference pipelines**
The PySpark UDF. If you already have Spark in your stack, this is zero additional infrastructure. Drop it into your existing pipeline.

**Anyone hitting rate limits**
The retry logic with proper header reading. Most people either crash on 429s or sleep for a fixed interval that's too short. Reading Retry-After headers means you sleep exactly as long as the API tells you to.

---

## Alternatives I Considered and Rejected

### LLMLingua (Microsoft)

LLMLingua uses a small language model to select the most important tokens from a prompt based on perplexity. It achieves 40–60% reduction compared to my regex compressor's 9% mean.

I didn't use it as the primary backend because:
- It requires PyTorch (1.5GB+ install) and a transformer model download on first run
- It adds 200–500ms latency per prompt (vs <0.1ms for regex)
- It introduces a GPU dependency for best performance
- It's too heavy for the zero-dependency core I wanted

I did integrate it as an optional backend. `pip install "llm-tokenoptim[ml]"` unlocks it and the compressor uses it automatically. Fall back to regex if not available. Best of both worlds.

### Building on top of aisuite or LiteLLM

Both are good provider abstraction layers. I rejected them because:
- They add a dependency that not everyone needs
- They solve a different problem (routing between providers) rather than optimization
- I wanted the optimization layer to be composable with whatever provider abstraction someone already uses
- Adding them would have made the zero-dependency core impossible

### Using a database for caching (Redis, SQLite)

Redis requires a running server. SQLite adds a dependency and handles concurrent writes awkwardly. JSON files keyed by SHA-256 hash are simple, dependency-free, human-readable (you can inspect what's cached), and fast enough for the access patterns involved (reads dominate writes, keys are known up front).

The tradeoff is that cleanup requires a separate TTL sweep rather than automatic eviction. I added a TTL check on every read — stale entries are ignored and deleted lazily.

### Storing full conversation history for memory

Summarization-based compaction loses some information. Windowing (keeping only last N turns) loses older context entirely. I chose windowing with a compaction step — when the window fills, older messages are summarized and the summary is stored as a system message. It's a practical middle ground that keeps costs bounded while retaining high-level context.

### Making the CLI tool language-agnostic (shell script only)

caveman is a pure copy-paste approach. I could have done the same — just provide the SKILL.md files and tell people to paste them manually.

The problem is that each LLM CLI tool has a different system prompt mechanism. Without automation, users have to figure out the right flag for each tool themselves. The Python CLI wrapper does that detection automatically. And the Python package can ship on PyPI so installation is one command.

I kept a pure-bash `bin/tokenoptim` as well for people who don't want Python at all — it can be symlinked to `/usr/local/bin` and works independently.

---

## Why This Makes Sense as a Project

Token cost optimization is a universal problem. Every team using LLMs at any scale eventually runs into it. But most solutions either:

1. Require you to change your model (bigger model, different model)
2. Require significant infrastructure changes
3. Only work with one provider
4. Require a GPU and heavy dependencies

llm-tokenoptim works at the edges — compress before you send, inject compression instructions into the response, cache what you can, manage what you've already sent. It's a thin layer that doesn't change your stack.

The PySpark integration is what makes it interesting at MAANG scale. Inference cost optimization on distributed pipelines is a real problem at Meta, Google, Amazon, and Microsoft. Most open-source LLM tooling doesn't think about distributed batch workloads at all.

---

## What I Would Do Differently

**Longer benchmark prompts.** The 500 built-in prompts skew short. Short prompts don't have much filler to compress. A dataset of longer, more realistic prompts (documentation passages, technical questions with background context) would show better P90/P95 numbers.

**Quality evaluation.** The token overlap proxy is rough. A proper quality eval would send the original and compressed prompt to an actual model and compare output quality on a benchmark task. I didn't do this because it requires live API calls and costs money to run.

**Output token measurement.** The output compression savings estimates come from manual testing, not automated measurement. Real numbers would require a proper evals framework with live API calls across multiple models.

**A demo GIF.** Apparently that's what drives GitHub stars. The project has no visual demo. A 30-second GIF showing `tokenoptim install-claude` followed by a side-by-side Claude response comparison would be worth more than any amount of documentation.

---

## Technical Decisions I'm Glad I Made

**Zero hard dependencies in the core.** Everything in `tokenoptim/core/` runs with nothing but the Python standard library. Provider SDKs (anthropic, openai), ML dependencies (torch, llmlingua), and batch tools (pyspark) are all optional extras. This means the CLI tool installs in under a second and works immediately.

**Honest benchmarks.** The benchmark explicitly says what it can't measure. This is unusual and it's the right call. Trust is worth more than impressive-looking numbers.

**The safety exception clause.** Every skill file has an exception: security warnings, irreversible actions, and repeated questions always get full prose. It would be easy to omit this for cleaner numbers. I kept it because the failure mode of a misunderstood irreversible action is much worse than a few extra tokens.

**Both CLI commands work.** After renaming the PyPI package to `llm-tokenoptim` (because `tokenoptim` was taken), I kept `tokenoptim` as an alias. People who installed it expecting `tokenoptim` to work won't be confused.

---

*Built by venkatamanas raghupatruni — github.com/Manas470/tokenoptim*
