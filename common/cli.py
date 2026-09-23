"""Shared command-line runner used by each example's ``run.py``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from .agent import Agent, AgentResult, ToolRegistry
from .providers import PROVIDERS, ProviderError, Step, get_provider

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path | None = None) -> None:
    """Minimal ``.env`` loader (KEY=VALUE lines). Existing env vars win."""
    path = path or REPO_ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def build_parser(description: str, default_task: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--provider", default=os.environ.get("AGENT_PROVIDER", "mock"),
                   choices=PROVIDERS, help="model provider (default: mock, runs offline)")
    p.add_argument("--model", help="model / deployment name (overrides the *_MODEL env var)")
    p.add_argument("--task", default=default_task, help="what to ask the agent")
    p.add_argument("--data-dir", type=Path, help="directory with the example's data files")
    p.add_argument("--max-steps", type=int, default=int(os.environ.get("AGENT_MAX_STEPS", 8)))
    p.add_argument("--trace-file", type=Path, help="write a JSONL trace of the run here")
    p.add_argument("-v", "--verbose", action="store_true", help="log every model and tool call")
    return p


def main(
    *,
    description: str,
    system_prompt: str,
    default_task: str,
    build_registry: Callable[[Path | None], ToolRegistry],
    mock_script: Callable[[], Sequence[Step]],
    argv: Sequence[str] | None = None,
) -> AgentResult | None:
    load_dotenv()
    args = build_parser(description, default_task).parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    try:
        provider = get_provider(args.provider, model=args.model,
                                script=mock_script() if args.provider == "mock" else None)
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    if args.provider == "mock" and args.task != default_task:
        print("note: the mock provider replays a fixed script and ignores --task.\n",
              file=sys.stderr)

    agent = Agent(provider, build_registry(args.data_dir), system_prompt=system_prompt,
                  max_steps=args.max_steps)
    try:
        result = agent.run(args.task)
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(result.output)
    print(f"\n[{result.stop_reason} after {result.steps} step(s); tools used: "
          f"{', '.join(result.trace.tool_calls) or 'none'}]", file=sys.stderr)
    if args.trace_file:
        result.trace.to_jsonl(args.trace_file)
        print(f"[trace written to {args.trace_file}]", file=sys.stderr)
    return result
