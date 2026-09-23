"""Run the manufacturing maintenance-log analyzer agent.

    python -m examples.manufacturing_maintenance.run --provider mock
    python examples/manufacturing_maintenance/run.py --provider ollama --task "..."
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python examples/manufacturing_maintenance/run.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.cli import main  # noqa: E402
from examples.manufacturing_maintenance.prompts import (  # noqa: E402
    DEFAULT_TASK,
    SYSTEM_PROMPT,
    mock_script,
)
from examples.manufacturing_maintenance.tools import build_registry  # noqa: E402

if __name__ == "__main__":
    main(
        description="Maintenance-log analyzer that flags recurring failures and at-risk machines.",
        system_prompt=SYSTEM_PROMPT,
        default_task=DEFAULT_TASK,
        build_registry=build_registry,
        mock_script=mock_script,
    )
