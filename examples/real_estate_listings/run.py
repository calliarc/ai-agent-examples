"""Run the real estate listing summarizer / comparable finder agent.

    python -m examples.real_estate_listings.run --provider mock
    python examples/real_estate_listings/run.py --provider anthropic --task "..."
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python examples/real_estate_listings/run.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.cli import main  # noqa: E402
from examples.real_estate_listings.prompts import (  # noqa: E402
    DEFAULT_TASK,
    SYSTEM_PROMPT,
    mock_script,
)
from examples.real_estate_listings.tools import build_registry  # noqa: E402

if __name__ == "__main__":
    main(
        description="Real estate listing summarizer and comparable-property finder.",
        system_prompt=SYSTEM_PROMPT,
        default_task=DEFAULT_TASK,
        build_registry=build_registry,
        mock_script=mock_script,
    )
