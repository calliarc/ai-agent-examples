"""A tiny, provider-agnostic tool-calling agent loop.

Pieces:

* ``Message`` / ``ToolCall`` - a neutral chat format that every provider
  adapter translates to and from its own wire format.
* ``ToolRegistry`` - turns plain, type-annotated Python functions into tools.
  Argument schemas are generated with Pydantic, and arguments coming back from
  the model are validated before the function runs.
* ``Agent`` - the loop: ask the model, run any requested tools, feed results
  back, stop on a final answer or after ``max_steps``.
* ``Trace`` - a structured record of every model call and tool call, also
  mirrored to the ``agent`` logger. Can be written out as JSON Lines.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import inspect
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, get_type_hints

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

logger = logging.getLogger("agent")

Role = Literal["system", "user", "assistant", "tool"]


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    """A request from the model to run one tool."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """One chat turn in the neutral format shared by all providers."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set on role="tool"
    name: str | None = None  # tool name, set on role="tool"
    is_error: bool = False  # set on role="tool" when the tool failed


@dataclass
class ToolSpec:
    """What a provider needs to advertise a tool to the model."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema (type: object)


class Provider(Protocol):
    """Anything that can turn a conversation into the next assistant message."""

    name: str

    def complete(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> Message:  # pragma: no cover - protocol
        ...


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def to_jsonable(value: Any) -> Any:
    """Convert tool return values (Pydantic models, dataclasses, dates...) to JSON types."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.isoformat()
    if isinstance(value, float):
        return round(value, 4)
    return value


def _strip_titles(schema: Any) -> Any:
    """Pydantic adds ``title`` everywhere; models don't need it."""
    if isinstance(schema, dict):
        return {k: _strip_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


def _param_hints(fn: Callable[..., Any]) -> dict[str, Any]:
    """Resolve parameter annotations (not the return type, which may be a local class).

    Works with ``from __future__ import annotations`` and with tools defined
    inside a function, by using the function's globals plus its closure.
    """
    ann = {k: v for k, v in getattr(fn, "__annotations__", {}).items() if k != "return"}
    localns: dict[str, Any] = {}
    code = getattr(fn, "__code__", None)
    freevars = code.co_freevars if code else ()
    for var, cell in zip(freevars, getattr(fn, "__closure__", None) or (), strict=False):
        try:
            localns[var] = cell.cell_contents
        except ValueError:  # empty cell
            pass

    def _shim() -> None: ...

    _shim.__annotations__ = ann
    return get_type_hints(_shim, globalns=getattr(fn, "__globals__", {}), localns=localns,
                          include_extras=True)


class Tool:
    """A Python function exposed to the model, with a Pydantic argument model."""

    def __init__(
        self, fn: Callable[..., Any], name: str | None = None, description: str | None = None
    ) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        self.description = (description or inspect.getdoc(fn) or self.name).strip()
        self.args_model = self._build_args_model(fn)

    def _build_args_model(self, fn: Callable[..., Any]) -> type[BaseModel]:
        hints = _param_hints(fn)
        fields: dict[str, Any] = {}
        for pname, param in inspect.signature(fn).parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                raise TypeError(f"tool {self.name!r}: *args/**kwargs are not supported")
            if pname not in hints:
                raise TypeError(f"tool {self.name!r}: parameter {pname!r} needs a type hint")
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[pname] = (hints[pname], default)
        model_name = "".join(p.capitalize() for p in self.name.split("_")) + "Args"
        return create_model(  # type: ignore[call-overload]
            model_name, __config__=ConfigDict(extra="forbid"), **fields
        )

    @property
    def spec(self) -> ToolSpec:
        schema = _strip_titles(self.args_model.model_json_schema())
        schema.setdefault("properties", {})
        schema["type"] = "object"
        return ToolSpec(name=self.name, description=self.description, parameters=schema)

    def __call__(self, arguments: dict[str, Any]) -> Any:
        args = self.args_model.model_validate(arguments or {})
        return self.fn(**{k: getattr(args, k) for k in type(args).model_fields})


class ToolRegistry:
    """A named collection of tools.

    >>> registry = ToolRegistry()
    >>> @registry.tool
    ... def add(a: int, b: int) -> int:
    ...     '''Add two integers.'''
    ...     return a + b
    >>> registry.execute(ToolCall(id="1", name="add", arguments={"a": 2, "b": 3}))
    ('5', False)
    """

    def __init__(self, max_result_chars: int = 20_000) -> None:
        self._tools: dict[str, Tool] = {}
        self.max_result_chars = max_result_chars

    def tool(self, fn: Callable[..., Any] | None = None, *, name: str | None = None,
             description: str | None = None):
        """Decorator: ``@registry.tool`` or ``@registry.tool(name=..., description=...)``."""

        def register(f: Callable[..., Any]) -> Callable[..., Any]:
            self.add(Tool(f, name=name, description=description))
            return f

        return register(fn) if fn is not None else register

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __getitem__(self, name: str) -> Tool:
        return self._tools[name]

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    @property
    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def execute(self, call: ToolCall) -> tuple[str, bool]:
        """Run a tool call. Returns ``(result_text, is_error)``; never raises.

        Errors (unknown tool, bad arguments, exceptions) are returned to the
        model as text so it can correct itself on the next step.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            return f"Error: unknown tool {call.name!r}. Available: {', '.join(self.names)}", True
        try:
            result = tool(call.arguments)
        except ValidationError as exc:
            errors = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
                for e in exc.errors()
            )
            return f"Error: invalid arguments for {call.name}: {errors}", True
        except Exception as exc:  # noqa: BLE001 - surfaced to the model on purpose
            return f"Error: {type(exc).__name__}: {exc}", True
        text = result if isinstance(result, str) else json.dumps(to_jsonable(result))
        if len(text) > self.max_result_chars:
            text = text[: self.max_result_chars] + "... [truncated]"
        return text, False


# --------------------------------------------------------------------------- #
# Tracing
# --------------------------------------------------------------------------- #
@dataclass
class TraceEvent:
    step: int
    kind: Literal["llm", "tool", "final", "max_steps"]
    data: dict[str, Any]
    duration_ms: float = 0.0


@dataclass
class Trace:
    events: list[TraceEvent] = field(default_factory=list)

    def add(self, event: TraceEvent) -> None:
        self.events.append(event)
        if event.kind == "tool":
            logger.info(
                "step %d tool %s(%s) -> %s%s",
                event.step,
                event.data["name"],
                json.dumps(event.data["arguments"]),
                "ERROR " if event.data["is_error"] else "",
                _short(event.data["result"]),
            )
        elif event.kind == "llm":
            logger.info(
                "step %d llm %s: %d tool call(s), %.0f ms",
                event.step,
                event.data["provider"],
                len(event.data["tool_calls"]),
                event.duration_ms,
            )
        else:
            logger.info("step %d %s", event.step, event.kind)

    @property
    def tool_calls(self) -> list[str]:
        return [e.data["name"] for e in self.events if e.kind == "tool"]

    def to_jsonl(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for e in self.events:
                fh.write(json.dumps(to_jsonable(e)) + "\n")


def _short(text: str, n: int = 160) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[:n] + "..."


# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #
@dataclass
class AgentResult:
    output: str
    messages: list[Message]
    trace: Trace
    steps: int
    stop_reason: Literal["final", "max_steps"]


class Agent:
    """Minimal tool-calling loop.

    Each step: send the conversation to the provider; if the reply has tool
    calls, execute them and append results; otherwise the reply is the final
    answer. Stops after ``max_steps`` model calls.
    """

    def __init__(
        self,
        provider: Provider,
        tools: ToolRegistry,
        system_prompt: str = "You are a helpful assistant.",
        max_steps: int = 8,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.provider = provider
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    def run(self, task: str, history: list[Message] | None = None) -> AgentResult:
        messages: list[Message] = list(history or []) + [Message(role="user", content=task)]
        trace = Trace()
        specs = self.tools.specs

        for step in range(1, self.max_steps + 1):
            t0 = time.perf_counter()
            reply = self.provider.complete(self.system_prompt, messages, specs)
            trace.add(TraceEvent(
                step=step,
                kind="llm",
                data={
                    "provider": getattr(self.provider, "name", type(self.provider).__name__),
                    "content": reply.content,
                    "tool_calls": [dataclasses.asdict(c) for c in reply.tool_calls],
                },
                duration_ms=(time.perf_counter() - t0) * 1000,
            ))
            messages.append(reply)

            if not reply.tool_calls:
                trace.add(TraceEvent(step=step, kind="final", data={"content": reply.content}))
                return AgentResult(reply.content, messages, trace, step, "final")

            for call in reply.tool_calls:
                t1 = time.perf_counter()
                result, is_error = self.tools.execute(call)
                trace.add(TraceEvent(
                    step=step,
                    kind="tool",
                    data={"id": call.id, "name": call.name, "arguments": call.arguments,
                          "result": result, "is_error": is_error},
                    duration_ms=(time.perf_counter() - t1) * 1000,
                ))
                messages.append(Message(role="tool", content=result, tool_call_id=call.id,
                                        name=call.name, is_error=is_error))

        trace.add(TraceEvent(step=self.max_steps, kind="max_steps", data={}))
        last_text = next((m.content for m in reversed(messages)
                          if m.role == "assistant" and m.content), "")
        output = last_text or f"Stopped after {self.max_steps} steps without a final answer."
        return AgentResult(output, messages, trace, self.max_steps, "max_steps")
