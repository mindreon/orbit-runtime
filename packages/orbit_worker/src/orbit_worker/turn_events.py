"""Translate one Activity's AgentScope stream into Orbit event fields.

``assistant.delta`` is coalesced to one event per 100 ms or 200 characters
per text block, and every chunk passes through a ``StreamRedactor``.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from agentscope.event import (
    ModelCallEndEvent,
    ModelCallStartEvent,
    TextBlockDeltaEvent,
    TextBlockEndEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultEndEvent,
    ToolResultStartEvent,
    ToolResultTextDeltaEvent,
)

from orbit_worker.secrets import StreamRedactor, redact_text, redact_value

DELTA_INTERVAL_S = 0.1
DELTA_CHARS = 200
ARGS_PREVIEW_CHARS = 256

_TOOL_STATES = {"success", "error", "denied", "interrupted"}

Emission = tuple[str, dict[str, object]]


@dataclass
class _Block:
    flushed_at: float
    seq: int = 0
    redactor: StreamRedactor = field(default_factory=StreamRedactor)


class TurnEvents:
    """Per-Activity state. ``tool_names`` seeds calls parked by an earlier Activity."""

    def __init__(
        self,
        tool_names: dict[str, str],
        activity_attempt: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._names = dict(tool_names)
        self._args: dict[str, list[str]] = {}
        self._outputs: dict[str, list[str]] = {}
        self._blocks: dict[str, _Block] = {}
        self._attempt = activity_attempt
        self._clock = clock
        self._model = ""
        self._model_started = 0.0
        self.in_model_call = False

    def observe(self, event: object) -> list[Emission]:
        if isinstance(event, ModelCallStartEvent):
            self.in_model_call = True
            self._model = event.model_name
            self._model_started = self._clock()
        elif isinstance(event, ModelCallEndEvent):
            self.in_model_call = False
            return [("usage", self._usage(event))]
        elif isinstance(event, TextBlockDeltaEvent):
            block = self._blocks.get(event.block_id)
            if block is None:
                block = self._blocks[event.block_id] = _Block(flushed_at=self._clock())
            block.redactor.feed(event.delta)
            due = self._clock() - block.flushed_at >= DELTA_INTERVAL_S
            if due or block.redactor.pending >= DELTA_CHARS:
                return self._flush(event.block_id, final=False)
        elif isinstance(event, TextBlockEndEvent):
            return self._flush(event.block_id, final=True)
        elif isinstance(event, ToolCallStartEvent):
            self._names[event.tool_call_id] = event.tool_call_name
            self._args[event.tool_call_id] = []
        elif isinstance(event, ToolCallDeltaEvent):
            self._args.setdefault(event.tool_call_id, []).append(event.delta)
        elif isinstance(event, ToolCallEndEvent):
            call_id = event.tool_call_id
            raw = "".join(self._args.pop(call_id, []))
            return [
                (
                    "tool.call",
                    {
                        "tool_name": self._names.get(call_id, ""),
                        "call_id": call_id,
                        "args_preview": args_preview(raw),
                    },
                )
            ]
        elif isinstance(event, ToolResultStartEvent):
            self._names[event.tool_call_id] = event.tool_call_name
            self._outputs[event.tool_call_id] = []
        elif isinstance(event, ToolResultTextDeltaEvent):
            self._outputs.setdefault(event.tool_call_id, []).append(event.delta)
        elif isinstance(event, ToolResultEndEvent):
            call_id = event.tool_call_id
            state = getattr(event.state, "value", event.state)
            return [
                (
                    "tool.result",
                    {
                        "text": "".join(self._outputs.pop(call_id, [])),
                        "tool_name": self._names.get(call_id, ""),
                        "call_id": call_id,
                        "tool_state": state if state in _TOOL_STATES else None,
                    },
                )
            ]
        return []

    def flush(self) -> list[Emission]:
        emissions: list[Emission] = []
        for block_id in list(self._blocks):
            emissions.extend(self._flush(block_id, final=True))
        return emissions

    def _flush(self, block_id: str, final: bool) -> list[Emission]:
        block = self._blocks.get(block_id)
        if block is None:
            return []
        delta = block.redactor.take(final=final)
        block.flushed_at = self._clock()
        if final:
            del self._blocks[block_id]
        if not delta:
            return []
        seq = block.seq
        block.seq += 1
        return [
            (
                "assistant.delta",
                {
                    "block_id": block_id,
                    "seq": seq,
                    "delta": delta,
                    "activity_attempt": self._attempt,
                },
            )
        ]

    def _usage(self, event: ModelCallEndEvent) -> dict[str, object]:
        return {
            "model": self._model,
            "input_tokens": event.input_tokens,
            "output_tokens": event.output_tokens,
            "cache_input_tokens": event.cache_input_tokens,
            "cache_creation_input_tokens": event.cache_creation_input_tokens,
            "latency_ms": max(0, round((self._clock() - self._model_started) * 1000)),
        }


def args_preview(raw: str) -> str:
    """Tool arguments for display: secrets redacted, then cut to 256 characters."""

    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict | list):
        text = json.dumps(redact_value(parsed), ensure_ascii=False)
    else:
        text = redact_text(raw)
    return text[:ARGS_PREVIEW_CHARS]
