# Changelog

All notable changes are documented here.
Format: [Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-05-19

### Added (CLI-first release)
- `tokenoptim install-claude --level LEVEL` — append caveman skill to CLAUDE.md (auto-loaded by Claude Code)
- `tokenoptim install-global` — install skill to ~/CLAUDE.md for all projects
- `tokenoptim wrap --level LEVEL -- CMD` — inject skill into gemini, codex, aider, sgpt, llm, ollama, or any LLM CLI
- `tokenoptim skill LEVEL` — print skill to stdout (pipeable anywhere)
- `skills/` directory with 5 SKILL.md files: lite / standard / full / ultra / ancient
- `bin/tokenoptim` — pure-bash wrapper (works without Python)
- `AsyncOptimizedClient` — full async/await client with streaming and batch support
- `AsyncAnthropicProvider` and `AsyncOpenAIProvider` with `async_chat()` and `stream()`
- `batch_chat()` — concurrent API calls with configurable semaphore
- `RetryConfig` — exponential backoff for 429/5xx/timeout errors
- `ResponseCache` — disk-backed LRU cache with TTL; survives restarts
- `MLPromptCompressor` — LLMLingua integration for 40–60% ML-based compression
- `benchmarks/run_benchmark.py` — reproducible benchmark against ShareGPT dataset
- `[ml]` optional extra: `pip install "tokenoptim[ml]"`

### Fixed
- Deduplicate min_len threshold reduced 80→40 (catches more realistic duplicates)
- Memory window test adjusted to account for compaction summary message

### Changed
- README benchmark table updated with honest, reproducible numbers
- Version bumped to 0.2.0

## [0.1.0] — 2026-05-19

### Added
- `PromptCompressor` — regex-based prompt compression (light/medium/full)
- `OutputCompressor` — 6-level caveman-style output compression
- `MemoryManager` — toggleable conversation context with auto-compaction
- `TokenCounter` — multi-provider token usage tracking
- `OptimizedClient` — sync unified client
- Providers: `AnthropicProvider`, `OpenAIProvider`, `OllamaProvider`
- `SparkTokenOptimizer` + `compress_prompts_udf` — PySpark batch support
- CLI: `tokenoptim compress / bench / levels`
- GitHub Actions CI (Python 3.9–3.12)
- 23 unit tests
