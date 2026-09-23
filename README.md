# AI Agent Examples

Practical AI agents for retail, real estate and manufacturing.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/calliarc/ai-agent-examples/actions/workflows/ci.yml/badge.svg)](https://github.com/calliarc/ai-agent-examples/actions/workflows/ci.yml)
![Version: 0.1.0](https://img.shields.io/badge/version-0.1.0-blue)

> **Status:** v0.1.0, first working release. All three examples run offline with a mock model and against real LLM providers.

## Features

- Retail: inventory forecasting and reorder recommendation agent
- Real estate: listing summarizer and comparable-property finder
- Manufacturing: maintenance-log analyzer that flags recurring failures
- Each example is self-contained with sample data, prompts and tests
- Provider-agnostic: swap between hosted and open-source models

## Tech stack

- Python 3.10+
- Tool-calling LLM APIs: OpenAI-compatible Chat Completions (OpenAI, Azure OpenAI, Ollama, vLLM, ...) and the Anthropic Messages API, called with `httpx`
- Pydantic (tool argument schemas and validation)
- Synthetic sample datasets (CSV, JSON, plain-text logs)
- pytest, Ruff, GitHub Actions

## Getting started

```bash
git clone https://github.com/calliarc/ai-agent-examples.git
cd ai-agent-examples
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Run an example offline with the deterministic mock provider (no API key needed)
python -m examples.retail_inventory.run

# Run the tests (offline)
pytest
```

To use a real model, copy `.env.example` to `.env`, fill in the provider you want, and pass `--provider`:

```bash
cp .env.example .env
python -m examples.retail_inventory.run --provider openai -v
python -m examples.real_estate_listings.run --provider anthropic --task "Summarize L-2033 for a first-time buyer."
python -m examples.manufacturing_maintenance.run --provider ollama --model llama3.1
```

Every `run.py` accepts `--provider`, `--model`, `--task`, `--data-dir`, `--max-steps`, `--trace-file trace.jsonl` and `-v` (log each model and tool call).

## Examples

| Example | What the agent does | Tools | Sample data |
|---|---|---|---|
| [`retail_inventory`](examples/retail_inventory) | Forecasts demand per SKU and produces a purchase plan with reorder quantities | `list_products`, `get_sales_history`, `forecast_demand`, `compute_reorder`, `reorder_report` | 120 days of sales, stock levels and supplier lead times for 8 SKUs |
| [`real_estate_listings`](examples/real_estate_listings) | Summarizes a listing, finds comparables and gives a comps-based price check | `search_listings`, `get_listing`, `summarize_listing`, `find_comparables` | 36 listings in a fictional city |
| [`manufacturing_maintenance`](examples/manufacturing_maintenance) | Parses maintenance logs, groups recurring failures, computes MTBF and flags at-risk machines | `parse_logs`, `list_events`, `find_recurring_failures`, `compute_mtbf`, `flag_at_risk_machines` | ~220 log events for 6 machines |

All sample data is synthetic and marked as such; each example has a seeded `data/generate_data.py` to regenerate it.

## Providers

| `--provider` | API | Configuration (env vars) |
|---|---|---|
| `mock` (default) | Scripted, deterministic, offline | none |
| `openai` | Chat Completions | `OPENAI_API_KEY`, `OPENAI_MODEL`, optional `OPENAI_BASE_URL` |
| `azure` | Azure OpenAI (v1 endpoint, or classic with `AZURE_OPENAI_API_VERSION`) | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT` |
| `anthropic` | Messages API | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` |
| `ollama` | Ollama's OpenAI-compatible endpoint | `OLLAMA_MODEL`, optional `OLLAMA_BASE_URL` |
| `vllm` | vLLM OpenAI-compatible server | `VLLM_MODEL`, optional `VLLM_BASE_URL`, `VLLM_API_KEY` |
| `openai-compatible` | Any other compatible server (LM Studio, LiteLLM, ...) | `LLM_BASE_URL`, `LLM_MODEL`, optional `LLM_API_KEY` |

Default model names are in `.env.example`; override with `--model` or the `*_MODEL` variables. Local models need tool-calling support (for vLLM, start the server with `--enable-auto-tool-choice` and a `--tool-call-parser`). The mock provider replays a fixed script per example and ignores `--task`.

## How it works

The shared loop in [`common/`](common) is intentionally small:

- `common/agent.py`: `ToolRegistry` turns typed Python functions into tools. It builds a JSON schema from the type hints with Pydantic, validates the model's arguments, and returns errors to the model instead of crashing. `Agent.run()` calls the model, runs the requested tools, feeds the results back, and stops at a final answer or `max_steps`. `Trace` records every model and tool call (logged, and exportable as JSONL).
- `common/providers.py`: adapters that translate one neutral message format to and from OpenAI-compatible Chat Completions and the Anthropic Messages API, plus `ScriptedProvider` for tests and offline demos.
- `common/cli.py`: the shared command-line runner and a minimal `.env` loader.

```python
from typing import Annotated
from pydantic import Field
from common import Agent, ToolRegistry, get_provider

tools = ToolRegistry()

@tools.tool
def stock_level(sku: Annotated[str, Field(description="product SKU")]) -> dict:
    """Current stock for a SKU."""
    return {"sku": sku, "on_hand": 42}

agent = Agent(get_provider("openai"), tools, system_prompt="You are an inventory assistant.")
print(agent.run("How many units of SKU-1 do we have?").output)
```

## Roadmap

- [x] Initial release
- [x] Documentation and examples
- [x] CI and automated tests
- [ ] Streaming responses and parallel tool execution
- [ ] Optional retries and cost/token accounting in traces
- [ ] More examples

Have an idea? [Open an issue](https://github.com/calliarc/ai-agent-examples/issues).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE) © 2026 CalliArc

---

Built and maintained by [CalliArc](https://www.calliarc.com/). Need help with AI development? [Talk to our team](https://www.calliarc.com/services/artificial-intelligence-development/).
