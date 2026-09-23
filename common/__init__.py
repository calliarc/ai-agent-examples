"""Shared, provider-agnostic agent loop used by every example."""

from .agent import (
    Agent,
    AgentResult,
    Message,
    Provider,
    Tool,
    ToolCall,
    ToolRegistry,
    ToolSpec,
    Trace,
)
from .providers import (
    AnthropicProvider,
    OpenAICompatibleProvider,
    ProviderError,
    ScriptedProvider,
    final_step,
    get_provider,
    tool_results,
    tool_step,
)

__version__ = "0.1.0"

__all__ = [
    "Agent", "AgentResult", "Message", "Provider", "Tool", "ToolCall", "ToolRegistry",
    "ToolSpec", "Trace", "AnthropicProvider", "OpenAICompatibleProvider", "ProviderError",
    "ScriptedProvider", "final_step", "get_provider", "tool_results", "tool_step",
    "__version__",
]
