"""Model provider adapters.

All adapters implement ``complete(system, messages, tools) -> Message`` using
the neutral types from :mod:`common.agent`.

* ``OpenAICompatibleProvider`` - the Chat Completions API
  (``POST {base_url}/chat/completions``). Works with OpenAI, Azure OpenAI
  (v1 endpoint), Ollama, vLLM, LM Studio and other compatible servers.
* ``AnthropicProvider`` - the Anthropic Messages API (``POST /v1/messages``).
* ``ScriptedProvider`` - deterministic, offline "mock" model that replays a
  script. Used by the tests and by ``--provider mock`` demos.

HTTP is done with ``httpx`` directly so there is no SDK version coupling;
pass your own ``httpx.Client`` to customise proxies, retries or timeouts.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from .agent import Message, ToolCall, ToolSpec

DEFAULT_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class ProviderError(RuntimeError):
    """Raised when a provider returns an error or an unexpected payload."""


def _raise_for_status(resp: httpx.Response, provider: str) -> None:
    if resp.status_code >= 400:
        body = resp.text[:500]
        raise ProviderError(f"{provider} API error {resp.status_code}: {body}")


# --------------------------------------------------------------------------- #
# OpenAI-compatible Chat Completions
# --------------------------------------------------------------------------- #
class OpenAICompatibleProvider:
    """Chat Completions with function tools.

    Request:  ``tools=[{"type": "function", "function": {name, description, parameters}}]``
    Response: ``choices[0].message.tool_calls[i].function.{name, arguments(JSON string)}``
    Tool results are sent back as ``{"role": "tool", "tool_call_id", "content"}``.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        name: str = "openai",
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.params = params or {}
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        if api_key and not any(h.lower() in ("authorization", "api-key") for h in self.headers):
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.client = client or httpx.Client(timeout=DEFAULT_TIMEOUT)

    def build_payload(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> dict:
        wire: list[dict[str, Any]] = []
        if system:
            wire.append({"role": "system", "content": system})
        for m in messages:
            if m.role == "tool":
                wire.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            elif m.role == "assistant" and m.tool_calls:
                wire.append({
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {"id": c.id, "type": "function",
                         "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                        for c in m.tool_calls
                    ],
                })
            else:
                wire.append({"role": m.role, "content": m.content})
        payload: dict[str, Any] = {"model": self.model, "messages": wire}
        if tools:
            payload["tools"] = [
                {"type": "function",
                 "function": {"name": t.name, "description": t.description,
                              "parameters": t.parameters}}
                for t in tools
            ]
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload

    @staticmethod
    def parse_response(data: dict) -> Message:
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected chat completions payload: {str(data)[:300]}") from exc
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError:
                args = {"__raw_arguments__": raw}  # will fail validation and be reported back
            calls.append(ToolCall(id=tc.get("id") or f"call_{i}", name=fn.get("name", ""),
                                  arguments=args))
        return Message(role="assistant", content=msg.get("content") or "", tool_calls=calls)

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Message:
        resp = self.client.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            params=self.params or None,
            json=self.build_payload(system, messages, tools),
        )
        _raise_for_status(resp, self.name)
        return self.parse_response(resp.json())


# --------------------------------------------------------------------------- #
# Anthropic Messages API
# --------------------------------------------------------------------------- #
class AnthropicProvider:
    """Anthropic Messages API with client tools.

    Request:  ``tools=[{name, description, input_schema}]``, top-level ``system``.
    Response: ``content`` blocks of type ``text`` and ``tool_use`` ({id, name, input}).
    Tool results go back in a *user* turn as ``tool_result`` blocks.
    """

    API_VERSION = "2023-06-01"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        max_tokens: int = 2048,
        temperature: float | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.name = "anthropic"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.headers = {
            "content-type": "application/json",
            "anthropic-version": self.API_VERSION,
        }
        if api_key:
            self.headers["x-api-key"] = api_key
        self.client = client or httpx.Client(timeout=DEFAULT_TIMEOUT)

    def build_payload(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> dict:
        wire: list[dict[str, Any]] = []

        def push(role: str, blocks: list[dict]) -> None:
            # The API expects alternating roles; merge consecutive same-role turns.
            if wire and wire[-1]["role"] == role:
                wire[-1]["content"].extend(blocks)
            else:
                wire.append({"role": role, "content": blocks})

        for m in messages:
            if m.role == "tool":
                block: dict[str, Any] = {"type": "tool_result", "tool_use_id": m.tool_call_id,
                                         "content": m.content}
                if m.is_error:
                    block["is_error"] = True
                push("user", [block])
            elif m.role == "assistant":
                blocks: list[dict] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                           for c in m.tool_calls]
                push("assistant", blocks)
            elif m.role == "user":
                push("user", [{"type": "text", "text": m.content}])
            elif m.role == "system":  # extra system messages are folded into the prompt
                system = f"{system}\n\n{m.content}".strip()

        payload: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens,
                                   "messages": wire}
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [{"name": t.name, "description": t.description,
                                 "input_schema": t.parameters} for t in tools]
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return payload

    @staticmethod
    def parse_response(data: dict) -> Message:
        if data.get("type") == "error":
            raise ProviderError(f"anthropic error: {data.get('error')}")
        texts, calls = [], []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append(ToolCall(id=block["id"], name=block["name"],
                                      arguments=block.get("input") or {}))
        return Message(role="assistant", content="\n".join(t for t in texts if t), tool_calls=calls)

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Message:
        resp = self.client.post(f"{self.base_url}/v1/messages", headers=self.headers,
                                json=self.build_payload(system, messages, tools))
        _raise_for_status(resp, self.name)
        return self.parse_response(resp.json())


# --------------------------------------------------------------------------- #
# Scripted mock provider
# --------------------------------------------------------------------------- #
Step = Message | Callable[[list[Message]], Message]


def tool_step(*calls: tuple[str, dict[str, Any]], text: str = "") -> Message:
    """A scripted assistant turn that requests one or more tools."""
    return Message(
        role="assistant",
        content=text,
        tool_calls=[ToolCall(id=f"mock_{i}_{name}", name=name, arguments=args)
                    for i, (name, args) in enumerate(calls)],
    )


def final_step(text_or_fn: str | Callable[[list[Message]], str]) -> Step:
    """A scripted final answer. Pass a function to build it from the conversation."""
    if callable(text_or_fn):
        fn = text_or_fn
        return lambda messages: Message(role="assistant", content=fn(messages))
    return Message(role="assistant", content=text_or_fn)


def tool_results(messages: list[Message], name: str) -> list[Any]:
    """All (JSON-decoded when possible) results of tool ``name`` so far."""
    out = []
    for m in messages:
        if m.role == "tool" and m.name == name:
            try:
                out.append(json.loads(m.content))
            except json.JSONDecodeError:
                out.append(m.content)
    return out


class ScriptedProvider:
    """Deterministic offline model: returns the next scripted step on each call.

    Steps are ``Message`` objects or callables ``(messages) -> Message``. Every
    request is recorded in ``self.requests`` so tests can assert on what the
    agent sent. When the script runs out, a fixed final answer is returned.
    """

    def __init__(self, script: Sequence[Step]) -> None:
        self.name = "mock"
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Message:
        self.requests.append({"system": system, "messages": list(messages),
                              "tools": [t.name for t in tools]})
        idx = len(self.requests) - 1
        if idx >= len(self.script):
            return Message(role="assistant", content="[mock] script exhausted.")
        step = self.script[idx]
        return step(messages) if callable(step) else step


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
PROVIDERS = ("mock", "openai", "azure", "ollama", "vllm", "openai-compatible", "anthropic")

DEFAULT_MODELS = {
    "openai": "gpt-6-luna",
    "ollama": "llama3.1",
    "anthropic": "claude-sonnet-5",
}


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value else default


def _require(name: str, value: str | None) -> str:
    if not value:
        raise ProviderError(f"{name} is not set. See .env.example for provider configuration.")
    return value


def get_provider(
    name: str,
    model: str | None = None,
    script: Sequence[Step] | None = None,
    client: httpx.Client | None = None,
):
    """Build a provider from its name and environment variables (see ``.env.example``)."""
    name = name.lower()
    if name == "mock":
        return ScriptedProvider(script or [])
    if name == "openai":
        return OpenAICompatibleProvider(
            model=model or _env("OPENAI_MODEL", DEFAULT_MODELS["openai"]),
            base_url=_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=_require("OPENAI_API_KEY", _env("OPENAI_API_KEY")),
            name="openai", client=client,
        )
    if name == "azure":
        endpoint = _require("AZURE_OPENAI_ENDPOINT", _env("AZURE_OPENAI_ENDPOINT")).rstrip("/")
        deployment = model or _require("AZURE_OPENAI_DEPLOYMENT", _env("AZURE_OPENAI_DEPLOYMENT"))
        api_key = _require("AZURE_OPENAI_API_KEY", _env("AZURE_OPENAI_API_KEY"))
        api_version = _env("AZURE_OPENAI_API_VERSION")
        if api_version:  # classic per-deployment endpoint
            base_url = f"{endpoint}/openai/deployments/{deployment}"
            params = {"api-version": api_version}
        else:  # v1 endpoint: OpenAI-compatible, deployment name goes in "model"
            base_url, params = f"{endpoint}/openai/v1", {}
        return OpenAICompatibleProvider(model=deployment, base_url=base_url, name="azure",
                                        headers={"api-key": api_key}, params=params, client=client)
    if name == "ollama":
        return OpenAICompatibleProvider(
            model=model or _env("OLLAMA_MODEL", DEFAULT_MODELS["ollama"]),
            base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            name="ollama", client=client,
        )
    if name == "vllm":
        return OpenAICompatibleProvider(
            model=model or _require("VLLM_MODEL", _env("VLLM_MODEL")),
            base_url=_env("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=_env("VLLM_API_KEY"), name="vllm", client=client,
        )
    if name == "openai-compatible":
        return OpenAICompatibleProvider(
            model=model or _require("LLM_MODEL", _env("LLM_MODEL")),
            base_url=_require("LLM_BASE_URL", _env("LLM_BASE_URL")),
            api_key=_env("LLM_API_KEY"), name="openai-compatible", client=client,
        )
    if name == "anthropic":
        return AnthropicProvider(
            model=model or _env("ANTHROPIC_MODEL", DEFAULT_MODELS["anthropic"]),
            api_key=_require("ANTHROPIC_API_KEY", _env("ANTHROPIC_API_KEY")),
            base_url=_env("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            client=client,
        )
    raise ValueError(f"unknown provider {name!r}; choose from {', '.join(PROVIDERS)}")
