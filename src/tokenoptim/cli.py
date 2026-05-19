"""
tokenoptim CLI

Works without an API key. Primary use: inject compression skills into
Claude Code, Gemini CLI, Codex, aider, or any LLM CLI tool.

Commands
--------
  tokenoptim skill [LEVEL]          Print skill system prompt to stdout
  tokenoptim levels                 Show all levels + savings estimates
  tokenoptim compress TEXT          Compress a prompt (Python NLP, no API)
  tokenoptim bench --input FILE     Benchmark compression on a prompt file
  tokenoptim install-claude         Add skill to ./CLAUDE.md (Claude Code)
  tokenoptim install-global         Add skill to ~/CLAUDE.md
  tokenoptim wrap --level L CMD     Inject skill into any CLI tool call

Examples
--------
  # Pipe skill into clipboard and paste into CLAUDE.md manually
  tokenoptim skill full | pbcopy

  # Install directly into project CLAUDE.md — Claude Code picks it up
  tokenoptim install-claude --level full

  # Wrap any LLM CLI tool
  tokenoptim wrap --level full -- gemini "explain kubernetes networking"
  tokenoptim wrap --level ultra -- claude "write a redis cache class"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Locate the skills directory relative to this file's installed location
_HERE = Path(__file__).parent
_PACKAGE_ROOT = _HERE.parent.parent.parent          # src/tokenoptim/../../..
_SKILLS_DIR   = _PACKAGE_ROOT / "skills"            # repo: skills/
if not _SKILLS_DIR.exists():
    # Installed via pip — skills are in package data next to src/
    _SKILLS_DIR = _HERE / "skills"

LEVELS = ["lite", "standard", "full", "ultra", "ancient"]
DEFAULT_LEVEL = "standard"
SAVINGS = {"lite": "~20%", "standard": "~40%", "full": "~60%", "ultra": "~70%", "ancient": "~75%"}


# ---------------------------------------------------------------------------
# skill content helpers
# ---------------------------------------------------------------------------

def _skill_path(level: str) -> Path:
    p = _SKILLS_DIR / level / "SKILL.md"
    if not p.exists():
        print(f"Error: unknown level '{level}'. Valid: {' | '.join(LEVELS)}", file=sys.stderr)
        sys.exit(1)
    return p


def _skill_text(level: str) -> str:
    return _skill_path(level).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_skill(args) -> None:
    """Print the skill system prompt to stdout."""
    print(_skill_text(args.level), end="")


def cmd_levels(args) -> None:
    print("\n  tokenoptim compression levels\n")
    print(f"  {'LEVEL':<10} {'OUTPUT SAVINGS':<18} DESCRIPTION")
    print(f"  {'─'*10} {'─'*18} {'─'*30}")
    descs = {
        "lite":     "strip pleasantries only",
        "standard": "terse engineer mode (default)",
        "full":     "caveman mode",
        "ultra":    "symbols + fragments",
        "ancient":  "stone tablet — extreme",
    }
    for lvl in LEVELS:
        print(f"  {lvl:<10} {SAVINGS[lvl]:<18} {descs[lvl]}")
    print()
    print("  Prompt compression levels (no API, <0.1ms/prompt):")
    print("  light   ~15% | medium  ~30% | full  ~40%")
    print()


def cmd_compress(args) -> None:
    """Compress a prompt with the regex NLP compressor."""
    from tokenoptim.core.compressor import PromptCompressor

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        print("Reading from stdin… (Ctrl+D to finish)", file=sys.stderr)
        text = sys.stdin.read()

    if args.level:
        compressor = PromptCompressor(level=args.level)
        compressed, stats = compressor.compress(text)
        if not args.quiet:
            print(f"\n{stats}\n", file=sys.stderr)
        if args.output:
            Path(args.output).write_text(compressed, encoding="utf-8")
            print(f"Saved → {args.output}", file=sys.stderr)
        else:
            print(compressed)
    else:
        for level in ["light", "medium", "full"]:
            compressor = PromptCompressor(level=level)
            compressed, stats = compressor.compress(text)
            print(f"[{level:6}] {stats.token_reduction_pct:4.1f}% | {compressed[:100]}")


def cmd_bench(args) -> None:
    """Benchmark compression on a file (one prompt per line)."""
    import statistics
    from tokenoptim.core.compressor import PromptCompressor

    prompts = [p for p in Path(args.input).read_text(encoding="utf-8").splitlines() if p.strip()]
    print(f"\nBenchmark: {len(prompts)} prompts\n")
    print(f"  {'Level':<8} {'Mean↓':>7} {'P90↓':>7} {'QPS':>8}")
    print(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*8}")
    import time
    for level in ["light", "medium", "full"]:
        c = PromptCompressor(level=level)
        reductions, times = [], []
        for p in prompts:
            t0 = time.perf_counter()
            _, s = c.compress(p)
            times.append(time.perf_counter() - t0)
            reductions.append(s.token_reduction_pct)
        mean_r  = statistics.mean(reductions)
        p90_r   = sorted(reductions)[int(len(reductions) * 0.9)]
        qps     = len(prompts) / sum(times) if sum(times) > 0 else 0
        print(f"  {level:<8} {mean_r:>6.1f}% {p90_r:>6.1f}% {qps:>8,.0f}")
    print()


def cmd_install_claude(args) -> None:
    """Append the tokenoptim skill into CLAUDE.md."""
    level  = args.level
    target = Path.home() / "CLAUDE.md" if args.global_ else Path("./CLAUDE.md")
    marker = "<!-- tokenoptim -->"

    if target.exists() and marker in target.read_text(encoding="utf-8"):
        print(f"tokenoptim already in {target}. Remove the <!-- tokenoptim --> block to re-install.")
        return

    skill = _skill_text(level)
    with target.open("a", encoding="utf-8") as f:
        f.write(f"\n{marker}\n{skill}\n<!-- /tokenoptim -->\n")

    print(f"✅ tokenoptim/{level} → {target}")
    print("   Claude Code picks up CLAUDE.md automatically.")


def cmd_wrap(args) -> None:
    """
    Inject the tokenoptim skill into any LLM CLI call.

    Detects the tool by name and passes the system prompt the right way.
    Supported: claude, gemini, codex, aider, sgpt, llm, ollama, and generic.
    """
    if not args.command:
        print("Usage: tokenoptim wrap [--level L] -- COMMAND [ARGS...]", file=sys.stderr)
        sys.exit(1)

    skill = _skill_text(args.level)
    cmd   = args.command[0]
    rest  = args.command[1:]
    name  = Path(cmd).name

    # Build the subprocess call depending on which tool it is
    if name == "claude":
        # Claude Code: write a temp CLAUDE.md and point to it
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", prefix="tokenoptim-claude-",
            delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(skill)
            tmp_path = tmp.name
        env = {**os.environ, "CLAUDE_SYSTEM_PROMPT_FILE": tmp_path}
        try:
            subprocess.run([cmd, *rest], env=env, check=False)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    elif name == "gemini":
        subprocess.run([cmd, "--system", skill, *rest], check=False)

    elif name == "codex":
        subprocess.run([cmd, "--system", skill, *rest], check=False)

    elif name == "aider":
        subprocess.run([cmd, "--system-prompt", skill, *rest], check=False)

    elif name in ("sgpt", "shell-gpt"):
        # shell-gpt accepts piped system content
        proc = subprocess.Popen([cmd, *rest], stdin=subprocess.PIPE)
        proc.communicate(input=skill.encode())

    elif name == "llm":
        # simonw/llm CLI
        subprocess.run([cmd, "--system", skill, *rest], check=False)

    elif name == "ollama":
        subprocess.run([cmd, "run", *rest, "--system", skill], check=False)

    else:
        # Generic: try --system flag; fall back to SYSTEM env var
        env = {**os.environ, "SYSTEM": skill}
        result = subprocess.run([cmd, "--system", skill, *rest], check=False)
        if result.returncode != 0:
            subprocess.run([cmd, *rest], env=env, check=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tokenoptim",
        description="🪨 tokenoptim — cut LLM token costs by 40-75% with any CLI tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # skill
    p = sub.add_parser("skill", help="Print skill system prompt")
    p.add_argument("level", nargs="?", default=DEFAULT_LEVEL, choices=LEVELS)

    # levels
    sub.add_parser("levels", help="Show all levels and savings")

    # compress
    p = sub.add_parser("compress", help="Compress a prompt (no API needed)")
    p.add_argument("text", nargs="?", help="Text to compress")
    p.add_argument("--level", "-l", choices=["light", "medium", "full"])
    p.add_argument("--file", "-f")
    p.add_argument("--output", "-o")
    p.add_argument("--quiet", "-q", action="store_true")

    # bench
    p = sub.add_parser("bench", help="Benchmark compression on a file")
    p.add_argument("--input", "-i", required=True)

    # install-claude
    p = sub.add_parser("install-claude", help="Add skill to ./CLAUDE.md")
    p.add_argument("--level", "-l", default=DEFAULT_LEVEL, choices=LEVELS)
    p.add_argument("--global", dest="global_", action="store_true",
                   help="Install to ~/CLAUDE.md instead")

    # install-global (alias)
    p = sub.add_parser("install-global", help="Add skill to ~/CLAUDE.md")
    p.add_argument("--level", "-l", default=DEFAULT_LEVEL, choices=LEVELS)
    p.set_defaults(global_=True)

    # wrap
    p = sub.add_parser("wrap", help="Inject skill into any LLM CLI call")
    p.add_argument("--level", "-l", default=DEFAULT_LEVEL, choices=LEVELS)
    p.add_argument("command", nargs=argparse.REMAINDER,
                   help="The LLM CLI command and its arguments")

    args = parser.parse_args()

    dispatch = {
        "skill":          cmd_skill,
        "levels":         cmd_levels,
        "compress":       cmd_compress,
        "bench":          cmd_bench,
        "install-claude": cmd_install_claude,
        "install-global": lambda a: cmd_install_claude(
            argparse.Namespace(level=a.level, global_=True)
        ),
        "wrap":           cmd_wrap,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
