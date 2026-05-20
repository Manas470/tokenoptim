# Contributing to llm-tokenoptim

Thank you for taking the time to contribute.

## Setup

```bash
git clone https://github.com/manasmourya/llm-tokenoptim
cd llm-tokenoptim
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```

For async tests you need `pytest-asyncio`:
```bash
pip install pytest-asyncio
pytest tests/ -v
```

## Before submitting a PR

1. **Tests pass**: `pytest tests/ -p no:cacheprovider`
2. **Lint clean**: `ruff check src/ tests/`
3. **Types check**: `mypy src/llm-tokenoptim/`
4. **Benchmark still runs**: `python benchmarks/run_benchmark.py --no-download --samples 20`

## Adding a new provider

1. Create `src/llm-tokenoptim/providers/myprovider.py` — inherit from `BaseProvider`
2. Implement `chat()`, `count_tokens()`, and `provider_name`
3. Export from `src/llm-tokenoptim/providers/__init__.py`
4. Add an async variant `myprovider_async.py` if the SDK supports it
5. Add at least one test in `tests/`

## Adding a compression level

- Output levels live in `src/llm-tokenoptim/core/output_style.py` — add to `CompressionLevel` enum and `_PROMPTS` dict
- Prompt levels live in `src/llm-tokenoptim/core/compressor.py` — add to `LEVELS` tuple and `_apply_compression()`

## Reporting bugs

Please open an issue with:
- Python version and OS
- llm-tokenoptim version (`python -c "import llm-tokenoptim; print(llm-tokenoptim.__version__)"`)
- Minimal reproducer
- Expected vs actual behaviour

## Security

Never include API keys in issues or PRs. If you find a security vulnerability, email the maintainer directly rather than opening a public issue.
