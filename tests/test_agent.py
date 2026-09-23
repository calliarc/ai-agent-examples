"""Tests for the shared agent loop, tool registry and provider adapters (all offline)."""

from __future__ import annotations

import json
from typing import Annotated, Literal

import httpx
import pytest
from pydantic import BaseModel, Field

from common.agent import Agent, Message, ToolCall, ToolRegistry
from common.providers import (
    AnthropicProvider,
    OpenAICompatibleProvider,
    ProviderError,
    ScriptedProvider,
    final_step,
    get_provider,
    tool_results,
    tool_step,
)


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool
    def add(a: int, b: Annotated[int, Field(description="second number")] = 1) -> int:
        """Add two integers."""
        return a + b

    class Point(BaseModel):
        x: float
        y: float

    @reg.tool(name="make_point", description="Build a point.")
    def point(x: float, y: float, unit: Literal["m", "km"] = "m") -> Point:
        return Point(x=x, y=y)

    @reg.tool
    def boom() -> str:
        """Always fails."""
        raise RuntimeError("kaboom")

    return reg


def test_schema_generated_from_type_hints(registry):
    spec = {s.name: s for s in registry.specs}
    add = spec["add"].parameters
    assert add["type"] == "object"
    assert add["required"] == ["a"]
    assert add["properties"]["a"]["type"] == "integer"
    assert add["properties"]["b"]["description"] == "second number"
    assert add["properties"]["b"]["default"] == 1
    assert spec["add"].description == "Add two integers."
    assert spec["make_point"].parameters["properties"]["unit"]["enum"] == ["m", "km"]
    assert spec["make_point"].description == "Build a point."
    assert "title" not in json.dumps(add)
    assert spec["boom"].parameters["properties"] == {}


def test_execute_validates_and_serializes(registry):
    assert registry.execute(ToolCall("1", "add", {"a": 2, "b": 3})) == ("5", False)
    assert registry.execute(ToolCall("1", "add", {"a": "4"})) == ("5", False)  # coerced
    out, err = registry.execute(ToolCall("1", "make_point", {"x": 1, "y": 2.5}))
    assert not err and json.loads(out) == {"x": 1.0, "y": 2.5}


@pytest.mark.parametrize("call, fragment", [
    (ToolCall("1", "nope", {}), "unknown tool"),
    (ToolCall("1", "add", {}), "a: Field required"),
    (ToolCall("1", "add", {"a": 1, "c": 2}), "Extra inputs"),
    (ToolCall("1", "make_point", {"x": 1, "y": 1, "unit": "mi"}), "invalid arguments"),
    (ToolCall("1", "boom", {}), "RuntimeError: kaboom"),
])
def test_execute_errors_are_returned_not_raised(registry, call, fragment):
    out, err = registry.execute(call)
    assert err and fragment in out


def test_untyped_tool_rejected():
    reg = ToolRegistry()
    with pytest.raises(TypeError):
        reg.tool(lambda x: x)
    reg.tool(lambda: 1, name="one")
    with pytest.raises(ValueError):
        reg.tool(lambda: 2, name="one")


def test_agent_loop_runs_tools_then_answers(registry):
    provider = ScriptedProvider([
        tool_step(("add", {"a": 2, "b": 40}), ("boom", {})),
        final_step(lambda msgs: f"answer={tool_results(msgs, 'add')[0]}"),
    ])
    result = Agent(provider, registry, system_prompt="sys").run("what is 2+40?")
    assert result.output == "answer=42"
    assert result.stop_reason == "final" and result.steps == 2
    assert result.trace.tool_calls == ["add", "boom"]
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert [m.is_error for m in tool_msgs] == [False, True]
    # the provider saw the system prompt, the tools and the tool results
    assert provider.requests[0]["system"] == "sys"
    assert set(provider.requests[0]["tools"]) == {"add", "make_point", "boom"}
    assert provider.requests[1]["messages"][-1].name == "boom"


def test_agent_stops_at_max_steps(registry, tmp_path):
    provider = ScriptedProvider([tool_step(("add", {"a": 1}))] * 10)
    result = Agent(provider, registry, max_steps=3).run("loop forever")
    assert result.stop_reason == "max_steps" and result.steps == 3
    assert len(provider.requests) == 3
    path = tmp_path / "trace.jsonl"
    result.trace.to_jsonl(path)
    kinds = [json.loads(line)["kind"] for line in path.read_text().splitlines()]
    assert kinds[-1] == "max_steps" and kinds.count("tool") == 3


def test_scripted_provider_exhausts_gracefully(registry):
    result = Agent(ScriptedProvider([]), registry).run("hi")
    assert "script exhausted" in result.output


# --------------------------------------------------------------------------- #
# Provider wire formats (no network: httpx.MockTransport)
# --------------------------------------------------------------------------- #
CONVERSATION = [
    Message(role="user", content="add 2 and 3"),
    Message(role="assistant", content="", tool_calls=[ToolCall("c1", "add", {"a": 2, "b": 3}),
                                                      ToolCall("c2", "boom", {})]),
    Message(role="tool", content="5", tool_call_id="c1", name="add"),
    Message(role="tool", content="Error: x", tool_call_id="c2", name="boom", is_error=True),
]


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_compatible_round_trip(registry):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "call_9", "type": "function",
                            "function": {"name": "add", "arguments": "{\"a\": 1, \"b\": 2}"}}],
        }}]})

    p = OpenAICompatibleProvider(model="m", base_url="https://example.test/v1/", api_key="k",
                                 client=_client(handler))
    reply = p.complete("sys", CONVERSATION, registry.specs)
    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    body = seen["body"]
    assert body["model"] == "m"
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][2]["tool_calls"][0]["function"] == {
        "name": "add", "arguments": json.dumps({"a": 2, "b": 3})}
    assert body["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": "5"}
    assert body["tools"][0]["type"] == "function"
    assert body["tools"][0]["function"]["name"] == "add"
    assert "temperature" not in body and "max_tokens" not in body
    assert reply.tool_calls == [ToolCall("call_9", "add", {"a": 1, "b": 2})]


def test_openai_bad_json_arguments_surface_as_tool_error(registry):
    msg = OpenAICompatibleProvider.parse_response({"choices": [{"message": {
        "content": "", "tool_calls": [{"id": "x", "function": {"name": "add",
                                                               "arguments": "{oops"}}]}}]})
    out, err = registry.execute(msg.tool_calls[0])
    assert err and "invalid arguments" in out


def test_anthropic_round_trip(registry):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "type": "message", "role": "assistant", "stop_reason": "tool_use",
            "content": [{"type": "text", "text": "Let me add."},
                        {"type": "tool_use", "id": "toolu_1", "name": "add",
                         "input": {"a": 1}}],
        })

    p = AnthropicProvider(model="claude-x", api_key="k", client=_client(handler))
    reply = p.complete("sys", CONVERSATION, registry.specs)
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "k"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    body = seen["body"]
    assert body["system"] == "sys" and body["max_tokens"] > 0
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert body["messages"][1]["content"][0] == {
        "type": "tool_use", "id": "c1", "name": "add", "input": {"a": 2, "b": 3}}
    results = body["messages"][2]["content"]  # both results merged into one user turn
    assert [r["tool_use_id"] for r in results] == ["c1", "c2"]
    assert results[1]["is_error"] is True
    assert body["tools"][0]["input_schema"]["type"] == "object"
    assert reply.content == "Let me add."
    assert reply.tool_calls == [ToolCall("toolu_1", "add", {"a": 1})]


def test_provider_http_error_raises():
    p = AnthropicProvider(model="m", api_key="bad", client=_client(
        lambda r: httpx.Response(401, json={"type": "error", "error": {"message": "nope"}})))
    with pytest.raises(ProviderError, match="401"):
        p.complete("", [Message(role="user", content="hi")], [])


def test_get_provider_from_env(monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_VERSION"):
        monkeypatch.delenv(var, raising=False)
    assert get_provider("mock").name == "mock"
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        get_provider("openai")
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        get_provider("anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    assert get_provider("openai", model="x").model == "x"

    ollama = get_provider("ollama")
    assert ollama.base_url == "http://localhost:11434/v1"
    assert "Authorization" not in ollama.headers

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "dep")
    az = get_provider("azure")
    assert az.base_url == "https://res.openai.azure.com/openai/v1"
    assert az.headers["api-key"] == "az" and "Authorization" not in az.headers
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    az = get_provider("azure")
    assert az.base_url.endswith("/openai/deployments/dep")
    assert az.params == {"api-version": "2024-10-21"}

    with pytest.raises(ValueError):
        get_provider("nope")
