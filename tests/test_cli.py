"""Smoke-test every example's run.py end to end with the offline mock provider."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "retail_inventory": "Purchase plan",
    "real_estate_listings": "Verdict",
    "manufacturing_maintenance": "At-risk machines",
}


@pytest.mark.parametrize("example, expected", EXAMPLES.items())
def test_run_py_mock(example, expected, tmp_path):
    trace = tmp_path / "trace.jsonl"
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env["AGENT_PROVIDER"] = "mock"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "examples" / example / "run.py"), "--provider", "mock",
         "--trace-file", str(trace)],
        capture_output=True, text=True, cwd=tmp_path, env=env, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert expected in proc.stdout
    assert "final after" in proc.stderr
    assert trace.read_text().count("\n") >= 3


def test_missing_key_gives_clear_error(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    proc = subprocess.run(
        [sys.executable, "-m", "examples.retail_inventory.run", "--provider", "openai"],
        capture_output=True, text=True, cwd=ROOT, env={**env, "OPENAI_API_KEY": ""}, timeout=60,
    )
    assert proc.returncode == 2
    assert "OPENAI_API_KEY is not set" in proc.stderr
