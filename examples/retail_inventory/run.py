"""Run the retail inventory agent.

    python -m examples.retail_inventory.run --provider mock
    python examples/retail_inventory/run.py --provider openai --task "..."
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python examples/retail_inventory/run.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.cli import main  # noqa: E402
from examples.retail_inventory.prompts import DEFAULT_TASK, SYSTEM_PROMPT, mock_script  # noqa: E402
from examples.retail_inventory.tools import build_registry  # noqa: E402

if __name__ == "__main__":
    main(
        description="Retail inventory forecasting and reorder agent.",
        system_prompt=SYSTEM_PROMPT,
        default_task=DEFAULT_TASK,
        build_registry=build_registry,
        mock_script=mock_script,
    )
