"""End-to-end sign-off check for the A1 events (E-A1-1 .. E-A1-5) and E-LOG-1.

``run`` starts a fresh local Temporal dev server and three orbit-orch and
orbit-worker pairs as subprocesses, each pair on its own task queue:

- ``mock``: ``ORBIT_MODEL_MODE=mock``; the mock model's ``stream:`` script
  streams provider deltas and ``echo:`` asks for ``gated_echo``.
- ``real``: ``ORBIT_MODEL_MODE=real`` against an in-process OpenAI-compatible
  stub, which counts provider requests and can answer 200 with null usage.
- ``log``: ``real`` mode against the same stub, used only by E-LOG-1, which
  stops the pair and reads its complete stdout and stderr.

Every process runs unbuffered and writes stdout and stderr to separate files
under ``--logs``. Every worker posts its events to an in-process recording
ingest stub that requires the internal bearer token. Each scenario gets its
own RoomWorkflow, driven with the ``runTurn`` and ``decide`` Updates the way
control drives it. Assertions read the JSON bodies the workers posted, in
arrival order, and for E-LOG-1 the process output.

The report has the commit, component versions, and per case: id, steps,
expected, actual, pass. It has no timestamps, ports, or random ids, so two
runs on one commit write the same bytes. Planted secrets appear in it only as
a SHA-256 prefix. ``run`` exits 1 when a case fails.

``scan`` checks report and log files for planted values, key and token
formats, and database URLs, writes its findings, and exits 1 on any finding.

    uv run python scripts/e2e_a1_events.py run --out artifacts/e2e-a1-events.json
    uv run python scripts/e2e_a1_events.py scan artifacts/e2e-a1-events.json e2e-logs/*.log
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from aiohttp import web
from orbit_contracts.models import RoomCommand, RoomWorkflowInput
from orbit_orch.workflows import RoomWorkflow
from orbit_worker.mock_model import CHUNK_SEPARATOR
from temporalio.api.workflowservice.v1 import GetSystemInfoRequest
from temporalio.client import Client, WorkflowHandle
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment

INGEST_TOKEN = "e2e-internal-ingest-token"
MODEL_KEY = "e2e-stub-model-key-not-real"
MODEL_NAME = "e2e-stub-model"
QUEUES = {"mock": "orbit-e2e-mock", "real": "orbit-e2e-real", "log": "orbit-e2e-log"}
REDACTED = "[REDACTED]"
TOOL_RESULT_BYTES = 4096
PROVIDER_ERROR_MESSAGE = "模型服务暂时出错，这一轮没跑完。"
CASE_TIMEOUT_S = 60
STUB_USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
NULL_USAGE_MARKER = "E2E-NULL-USAGE"

# Pure Chinese, no spaces or ASCII; each part is longer than one 200-character delta batch.
CJK_PARAGRAPH = "模型正在逐字输出一段没有任何空格的中文回答，" * 10
CJK_FILLER = "这是一段没有任何空格的中文说明文字，" * 12
FILLER = "word " * 45


@dataclass(frozen=True)
class SplitSecret:
    room: str
    context: str
    before: str
    first_half: str
    second_half: str
    after: str
    expected: str

    @property
    def secret(self) -> str:
        return self.first_half + self.second_half


SPLIT_SECRETS = [
    SplitSecret(
        "e-a1-2-en-split-key",
        "English",
        FILLER + "the key is ",
        "sk-live-",
        "4f9a2b77c",
        " then stop",
        FILLER + f"the key is {REDACTED} then stop",
    ),
    SplitSecret(
        "e-a1-2-en-split-bearer",
        "English",
        FILLER + "send it with Bearer ",
        "",
        "q7wz19kx",
        " then stop",
        FILLER + f"send it with Bearer {REDACTED} then stop",
    ),
    SplitSecret(
        "e-a1-2-zh-split-key",
        "Chinese",
        CJK_FILLER + "密钥是 ",
        "sk-live-",
        "4f9a2b77c",
        "，然后停",
        CJK_FILLER + f"密钥是 {REDACTED}，然后停",
    ),
    SplitSecret(
        "e-a1-2-zh-split-random",
        "Chinese",
        CJK_FILLER + "复制",
        "Zk8Qw3Rt7Yp2Lm9X",
        "c4Vb6Nj1Hg5Fd0Sa",
        "，结束",
        CJK_FILLER + f"复制{REDACTED}，结束",
    ),
]
LOG_MARKER = "E2E-LOG-1"
LOG_HALVES = ("sk-log1-Vq8R", "m3Tz6Wk1Pn4J")
LOG_SECRET = "".join(LOG_HALVES)
# Conversation content with no secret shape, so no redactor would touch it.
LOG_CANARY = "蓝色长颈鹿在月球背面数星星"
# As written, JSON-escaped (\u84dd...), and JSON-escaped twice (a JSON string
# inside a JSON document, as span attributes are when printed).
LOG_CANARY_FORMS = (
    LOG_CANARY,
    json.dumps(LOG_CANARY)[1:-1],
    json.dumps(json.dumps(LOG_CANARY))[2:-2],
)
LOG_MESSAGE = f"{LOG_MARKER} {LOG_CANARY}. Echo my key {LOG_SECRET} back to me."
LOG_TOOL_TEXT = f"{LOG_CANARY} {LOG_SECRET}"
LOG_REPLY = f"{LOG_CANARY}: your key was {LOG_SECRET}"
LOG_STARTUP_LINE = f"chat model: real model={MODEL_NAME}"

PLANTED = [INGEST_TOKEN, MODEL_KEY, LOG_SECRET, *LOG_HALVES, *LOG_CANARY_FORMS] + [
    part
    for split in SPLIT_SECRETS
    for part in (split.secret, split.first_half, split.second_half)
    if len(part) >= 6
]


class Recorder:
    """Recording ingest stub and OpenAI-compatible model stub."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self.unauthorized = 0
        self.model_requests: list[str] = []
        self.tool_messages: list[str] = []

    async def ingest(self, request: web.Request) -> web.Response:
        if request.headers.get("Authorization") != f"Bearer {INGEST_TOKEN}":
            self.unauthorized += 1
            return web.Response(status=401)
        raw = await request.text()
        self.events.append((raw, json.loads(raw)))
        return web.Response(status=202)

    async def chat(self, request: web.Request) -> web.Response:
        if request.headers.get("Authorization") != f"Bearer {MODEL_KEY}":
            return web.json_response({"error": {"message": "bad key"}}, status=401)
        body = await request.json()
        prompt = _user_text(body.get("messages", []))
        self.model_requests.append(prompt)
        usage: dict[str, object] = dict(STUB_USAGE)
        if NULL_USAGE_MARKER in prompt:
            usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        message: dict[str, object] = {"role": "assistant", "content": "stub reply"}
        finish_reason = "stop"
        if LOG_MARKER in prompt:
            tool_messages = [
                _content_text(item.get("content"))
                for item in body.get("messages", [])
                if item.get("role") == "tool"
            ]
            self.tool_messages.extend(tool_messages)
            if tool_messages:
                message = {"role": "assistant", "content": LOG_REPLY}
            else:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_e_log_1",
                            "type": "function",
                            "function": {
                                "name": "gated_echo",
                                "arguments": json.dumps({"text": LOG_TOOL_TEXT}),
                            },
                        }
                    ],
                }
                finish_reason = "tool_calls"
        return web.json_response(
            {
                "id": f"chatcmpl-{len(self.model_requests)}",
                "object": "chat.completion",
                "created": 1,
                "model": MODEL_NAME,
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": usage,
            }
        )

    def room(self, room_id: str) -> list[dict]:
        return [body for _, body in self.events if body.get("roomId") == room_id]

    def raw(self, room_id: str | None = None) -> list[str]:
        return [
            raw for raw, body in self.events if room_id is None or body.get("roomId") == room_id
        ]


def _user_text(messages: list[dict]) -> str:
    # AgentScope appends a <system-reminder> user message after the real one,
    # so markers are looked for in every user message.
    return "\n".join(
        _content_text(message.get("content"))
        for message in messages
        if message.get("role") == "user"
    )


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def _summary(text: str | None) -> dict[str, object] | None:
    if text is None:
        return None
    encoded = text.encode("utf-8")
    return {
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest()[:16],
        "head": text[:12],
        "tail": text[-12:],
    }


def _fingerprint(secret: str) -> str:
    return "sha256:" + hashlib.sha256(secret.encode("utf-8")).hexdigest()[:12]


def _of(events: list[dict], kind: str) -> list[dict]:
    return [body for body in events if body["type"] == kind]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Harness:
    def __init__(
        self,
        client: Client,
        recorder: Recorder,
        processes: dict[str, list[subprocess.Popen]],
        logs: Path,
    ) -> None:
        self.client = client
        self.recorder = recorder
        self.processes = processes
        self.logs = logs

    def stop(self, mode: str) -> bool:
        """Stop one orch and worker pair; True when every process has exited."""

        _stop(self.processes[mode])
        return all(process.poll() is not None for process in self.processes[mode])

    def output(self, name: str) -> dict[str, str]:
        return {
            stream: (self.logs / f"{name}.{stream}.log").read_text(
                encoding="utf-8", errors="replace"
            )
            for stream in ("stdout", "stderr")
        }

    async def open_room(self, room_id: str, mode: str) -> WorkflowHandle:
        handle = await self.client.start_workflow(
            RoomWorkflow.run,
            RoomWorkflowInput(room_id=room_id),
            id=f"room:{room_id}",
            task_queue=QUEUES[mode],
        )
        await handle.signal(RoomWorkflow.command, RoomCommand(kind="open", turn_id="t-open"))
        await self.wait_status(handle, "running")
        return handle

    async def wait_status(self, handle: WorkflowHandle, status: str) -> None:
        for _ in range(CASE_TIMEOUT_S * 10):
            if (await handle.query(RoomWorkflow.snapshot)).status == status:
                return
            await asyncio.sleep(0.1)
        raise TimeoutError(f"{handle.id} never reached {status}")

    async def stream(self, room_id: str, parts: list[str]) -> list[dict]:
        handle = await self.open_room(room_id, "mock")
        message = "stream:" + CHUNK_SEPARATOR.join(parts)
        await handle.execute_update("runTurn", {"turnId": "tn-1", "message": message})
        return self.recorder.room(room_id)

    async def echo(self, room_id: str, text: str) -> tuple[dict, list[dict]]:
        handle = await self.open_room(room_id, "mock")
        parked = await handle.execute_update(
            "runTurn", {"turnId": "tn-1", "message": "echo:" + text}
        )
        approval = parked.get("approval") or {}
        await handle.execute_update(
            "decide",
            {
                "decision": "allow",
                "approvalRequestId": approval.get("approvalRequestId", ""),
                "resumeTurnId": "tn-2",
            },
        )
        return parked, self.recorder.room(room_id)


async def case_e_a1_1(h: Harness) -> dict:
    room = "e-a1-1"
    reply = CJK_PARAGRAPH * 3
    events = await h.stream(room, [CJK_PARAGRAPH] * 3)
    message_at = next(
        (i for i, body in enumerate(events) if body["type"] == "assistant.message"), len(events)
    )
    before = [body for body in events[:message_at] if body["type"] == "assistant.delta"]
    messages = _of(events, "assistant.message")
    return {
        "id": "E-A1-1",
        "title": "A long pure-Chinese reply streams at least 2 assistant.delta before completion",
        "steps": [
            f"Open room {room} on the mock worker.",
            ("runTurn tn-1: the streaming mock model sends the reply as 3 provider deltas of "
            f"{len(CJK_PARAGRAPH)} Chinese characters each (no spaces, no ASCII)."),
            "Count assistant.delta events posted before the turn's assistant.message.",
        ],
        "expected": {
            "atLeastTwoDeltasBeforeCompletion": True,
            "deltasBeforeCompletion": 3,
            "deltaSeq": [0, 1, 2],
            "streamed": _summary(reply),
            "assistantMessage": _summary(reply),
        },
        "actual": {
            "atLeastTwoDeltasBeforeCompletion": len(before) >= 2,
            "deltasBeforeCompletion": len(before),
            "deltaSeq": [body["seq"] for body in before],
            "streamed": _summary("".join(body["delta"] for body in before)),
            "assistantMessage": _summary(messages[-1]["text"] if messages else None),
        },
    }


async def case_e_a1_2(h: Harness) -> dict:
    steps: list[str] = []
    expected: dict[str, object] = {}
    actual: dict[str, object] = {}
    for split in SPLIT_SECRETS:
        events = await h.stream(
            split.room,
            [split.before + split.first_half, split.second_half + split.after],
        )
        raws = h.recorder.raw(split.room)
        deltas = _of(events, "assistant.delta")
        messages = _of(events, "assistant.message")
        fragments = [part for part in (split.first_half, split.second_half) if len(part) >= 6]
        steps.append(
            f"Room {split.room} ({split.context} context): chunk 1 ends with "
            f"{'the first half of the secret' if split.first_half else 'the prefix'}, chunk 2 "
            f"starts with the rest; secret {_fingerprint(split.secret)}."
        )
        expected[split.room] = {
            "context": split.context,
            "atLeastTwoDeltas": True,
            "streamed": split.expected,
            "assistantMessage": split.expected,
            "secretVerbatimInAnyEvent": False,
            "secretHalvesInAnyEvent": [],
        }
        actual[split.room] = {
            "context": split.context,
            "atLeastTwoDeltas": len(deltas) >= 2,
            "streamed": "".join(body["delta"] for body in deltas),
            "assistantMessage": messages[-1]["text"] if messages else None,
            "secretVerbatimInAnyEvent": any(split.secret in raw for raw in raws),
            "secretHalvesInAnyEvent": [
                _fingerprint(part) for part in fragments if any(part in raw for raw in raws)
            ],
        }
    return {
        "id": "E-A1-2",
        "title": "A secret split across two chunks never appears verbatim in any captured event",
        "steps": [
            ("Each room runs on the mock worker; runTurn tn-1 streams exactly two provider "
            "deltas with the secret split across them."),
            *steps,
            "Search every raw event body posted for that room for the secret and its halves.",
        ],
        "expected": expected,
        "actual": actual,
    }


async def case_e_a1_3(h: Harness) -> dict:
    scenarios = [
        ("e-a1-3-zh-over-4kb", "中" * 3000, "echo:" + "中" * 1363, True),
        ("e-a1-3-ascii-over-4kb", "x" * 5000, "echo:" + "x" * (TOOL_RESULT_BYTES - 5), True),
        ("e-a1-3-short", "short", "echo:short", False),
    ]
    steps = []
    expected: dict[str, object] = {}
    actual: dict[str, object] = {}
    for room, text, result_text, truncated in scenarios:
        parked, events = await h.echo(room, text)
        results = _of(events, "tool.result")
        result = results[-1] if results else {}
        steps.append(
            f"Room {room}: runTurn tn-1 'echo:' + {len(text)} characters "
            f"({len(('echo:' + text).encode())} bytes of tool output); decide allow as tn-2."
        )
        expected[room] = {
            "parkedStatus": "needs_approval",
            "toolResults": 1,
            "toolName": "gated_echo",
            "toolState": "success",
            "truncated": truncated,
            "textAtMost4096Bytes": True,
            "text": _summary(result_text),
        }
        actual[room] = {
            "parkedStatus": parked.get("status"),
            "toolResults": len(results),
            "toolName": result.get("toolName"),
            "toolState": result.get("toolState"),
            "truncated": result.get("truncated", False),
            "textAtMost4096Bytes": len(result.get("text", "").encode()) <= TOOL_RESULT_BYTES,
            "text": _summary(result.get("text", "")),
        }
    return {
        "id": "E-A1-3",
        "title": "tool.result over 4KB is truncated with truncated: true",
        "steps": [
            ("Each room runs on the mock worker; the mock model asks for gated_echo, which "
            "parks for approval and then runs inside the Activity."),
            *steps,
            "Read the tool.result event; the short room is the untruncated control.",
        ],
        "expected": expected,
        "actual": actual,
    }


# Model calls per room on the mock worker: one per streamed reply; gated_echo
# takes one call that asks for the tool and one after its result.
MOCK_MODEL_CALLS = {
    "e-a1-1": 1,
    **{split.room: 1 for split in SPLIT_SECRETS},
    "e-a1-3-zh-over-4kb": 2,
    "e-a1-3-ascii-over-4kb": 2,
    "e-a1-3-short": 2,
}


async def case_e_a1_4(h: Harness) -> dict:
    room = "e-a1-4-real"
    handle = await h.open_room(room, "real")
    before = len(h.recorder.model_requests)
    for turn in ("tn-1", "tn-2"):
        await handle.execute_update("runTurn", {"turnId": turn, "message": f"E2E-USAGE {turn}"})
    provider_calls = len(h.recorder.model_requests) - before
    real_usage = _of(h.recorder.room(room), "usage")
    everything = [body for _, body in h.recorder.events]
    return {
        "id": "E-A1-4",
        "title": "Every model call emits a usage event; modelMode is only mock or real",
        "steps": [
            (f"Open room {room} on the real-mode worker (OpenAI-compatible stub); runTurn "
            "tn-1 and tn-2, one model call each. The stub counts provider requests."),
            ("For every mock-worker room above, compare usage events with its scripted "
            "model calls."),
            "Collect modelMode from every event posted during the run (all cases).",
            ("E-A1-5's call is not counted here: its usage cannot be read, which is the "
            "failure that case checks."),
        ],
        "expected": {
            "realProviderCalls": 2,
            "realUsageEvents": 2,
            "realUsage": [
                {"model": MODEL_NAME, "inputTokens": 11, "outputTokens": 7, "modelMode": "real"}
            ]
            * 2,
            "mockUsageEventsPerRoom": MOCK_MODEL_CALLS,
            "modelModesSeen": ["mock", "real"],
            "modelModeOutsideMockOrReal": 0,
        },
        "actual": {
            "realProviderCalls": provider_calls,
            "realUsageEvents": len(real_usage),
            "realUsage": [
                {
                    "model": body.get("model"),
                    "inputTokens": body.get("inputTokens"),
                    "outputTokens": body.get("outputTokens"),
                    "modelMode": body.get("modelMode"),
                }
                for body in real_usage
            ],
            "mockUsageEventsPerRoom": {
                name: len(_of(h.recorder.room(name), "usage")) for name in MOCK_MODEL_CALLS
            },
            "modelModesSeen": sorted({body.get("modelMode") for body in everything}),
            "modelModeOutsideMockOrReal": sum(
                body.get("modelMode") not in ("mock", "real") for body in everything
            ),
        },
    }


async def case_e_a1_5(h: Harness) -> dict:
    room = "e-a1-5"
    handle = await h.open_room(room, "real")
    before = len(h.recorder.model_requests)
    result = await handle.execute_update(
        "runTurn", {"turnId": "tn-1", "message": f"{NULL_USAGE_MARKER} please"}
    )
    snapshot = await handle.query(RoomWorkflow.snapshot)
    events = h.recorder.room(room)
    failed = _of(events, "turn.failed")
    return {
        "id": "E-A1-5",
        "title": "S-RM-3: HTTP 200 with null usage token counts is one retryable turn.failed",
        "steps": [
            f"Open room {room} on the real-mode worker.",
            ("runTurn tn-1: the stub answers HTTP 200 with a valid message and usage whose "
            "token counts are null."),
            ("Read the Update result, the room snapshot, the posted events, and the stub's "
            "request count (a Temporal retry would call the stub again)."),
        ],
        "expected": {
            "updateStatus": "failed",
            "updateErrorCode": "provider_error",
            "updateRetryable": True,
            "turnFailedEvents": 1,
            "failure": {
                "turnId": "tn-1",
                "agentId": "main",
                "errorCode": "provider_error",
                "retryable": True,
                "message": PROVIDER_ERROR_MESSAGE,
            },
            "providerCalls": 1,
            "assistantMessages": 0,
            "roomStatus": "running",
        },
        "actual": {
            "updateStatus": result.get("status"),
            "updateErrorCode": result.get("errorCode"),
            "updateRetryable": result.get("retryable"),
            "turnFailedEvents": len(failed),
            "failure": failed[0].get("failure") if failed else None,
            "providerCalls": len(h.recorder.model_requests) - before,
            "assistantMessages": len(_of(events, "assistant.message")),
            "roomStatus": snapshot.status,
        },
    }


def _leaks(text: str) -> dict[str, object]:
    return {
        "secret": LOG_SECRET in text,
        "secretHalves": [_fingerprint(half) for half in LOG_HALVES if half in text],
        "conversationCanary": any(form in text for form in LOG_CANARY_FORMS),
    }


async def case_e_log_1(h: Harness) -> dict:
    room = "e-log-1"
    handle = await h.open_room(room, "log")
    before = len(h.recorder.model_requests)
    parked = await handle.execute_update("runTurn", {"turnId": "tn-1", "message": LOG_MESSAGE})
    approval = parked.get("approval") or {}
    decided = await handle.execute_update(
        "decide",
        {
            "decision": "allow",
            "approvalRequestId": approval.get("approvalRequestId", ""),
            "resumeTurnId": "tn-2",
        },
    )
    requests = h.recorder.model_requests[before:]
    raws = h.recorder.raw(room)
    stopped = h.stop("log")
    worker = h.output("orbit-worker-log")
    orch = h.output("orbit-orch-log")
    clean = {"secret": False, "secretHalves": [], "conversationCanary": False}
    return {
        "id": "E-LOG-1",
        "title": "No conversation content or planted secret reaches the worker's stdout or stderr",
        "steps": [
            (f"Open room {room} on its own orbit-orch and orbit-worker pair (task queue "
            f"{QUEUES['log']}, ORBIT_MODEL_MODE=real, OpenAI-compatible stub). Both processes "
            "run with PYTHONUNBUFFERED=1; stdout and stderr go to separate files."),
            (f"runTurn tn-1: the user message carries the planted secret "
            f"{_fingerprint(LOG_SECRET)} (halves {_fingerprint(LOG_HALVES[0])} and "
            f"{_fingerprint(LOG_HALVES[1])}) and a conversation canary "
            f"{_fingerprint(LOG_CANARY)} with no secret shape."),
            ("The stub answers with a gated_echo tool call whose arguments carry the canary and "
            "the secret; the turn parks for approval."),
            ("decide allow as tn-2: gated_echo runs inside the Activity, its result goes back to "
            "the stub, and the stub's final reply repeats the canary and the secret."),
            ("Stop the pair (SIGTERM, then wait), read the worker's and the orch's complete "
            "stdout and stderr, and search them for the secret, each half, and the canary "
            "(as written, JSON-escaped, and JSON-escaped twice)."),
        ],
        "expected": {
            "parkedStatus": "needs_approval",
            "approvalTool": "gated_echo",
            "finalStatus": "completed",
            "providerCalls": 2,
            "secretReachedProvider": True,
            "toolResultReachedProvider": True,
            "secretOrHalvesInAnyEvent": False,
            "processesStopped": True,
            "workerStderrHasStartupLine": True,
            "workerStdout": clean,
            "workerStderr": clean,
            "orchStdout": clean,
            "orchStderr": clean,
        },
        "actual": {
            "parkedStatus": parked.get("status"),
            "approvalTool": approval.get("toolName"),
            "finalStatus": (decided.get("turn") or {}).get("status"),
            "providerCalls": len(requests),
            "secretReachedProvider": bool(requests) and LOG_SECRET in requests[0],
            "toolResultReachedProvider": any(
                f"echo:{LOG_TOOL_TEXT}" in text for text in h.recorder.tool_messages
            ),
            "secretOrHalvesInAnyEvent": any(
                value in raw for raw in raws for value in (LOG_SECRET, *LOG_HALVES)
            ),
            "processesStopped": stopped,
            "workerStderrHasStartupLine": LOG_STARTUP_LINE in worker["stderr"],
            "workerStdout": _leaks(worker["stdout"]),
            "workerStderr": _leaks(worker["stderr"]),
            "orchStdout": _leaks(orch["stdout"]),
            "orchStderr": _leaks(orch["stderr"]),
        },
    }


CASES: list[Callable[[Harness], Awaitable[dict]]] = [
    case_e_a1_1,
    case_e_a1_2,
    case_e_a1_3,
    case_e_a1_4,
    case_e_a1_5,
    case_e_log_1,
]


def _start(module: str, env: dict[str, str], logs: Path, name: str) -> subprocess.Popen:
    with (
        (logs / f"{name}.stdout.log").open("w", encoding="utf-8") as out,
        (logs / f"{name}.stderr.log").open("w", encoding="utf-8") as err,
    ):
        return subprocess.Popen([sys.executable, "-m", module], env=env, stdout=out, stderr=err)


def _stop(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _commit(explicit: str) -> str:
    if explicit:
        return explicit
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _versions(server_version: str) -> dict[str, str]:
    versions = {"python": platform.python_version(), "temporalServer": server_version}
    for name in (
        "orbit-contracts",
        "orbit-orch",
        "orbit-worker",
        "agentscope",
        "temporalio",
        "pydantic",
        "openai",
        "aiohttp",
    ):
        versions[name] = importlib.metadata.version(name)
    return versions


async def run(out: Path, logs: Path, commit: str) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    recorder = Recorder()
    app = web.Application()
    app.router.add_post("/internal/events", recorder.ingest)
    app.router.add_post("/v1/chat/completions", recorder.chat)
    runner = web.AppRunner(app)
    await runner.setup()
    port = _free_port()
    await web.TCPSite(runner, "127.0.0.1", port).start()

    rows: list[dict] = []
    async with await WorkflowEnvironment.start_local(
        data_converter=pydantic_data_converter
    ) as temporal:
        info = await temporal.client.workflow_service.get_system_info(GetSystemInfoRequest())
        base = {key: value for key, value in os.environ.items() if not key.startswith("ORBIT_")}
        base.update(
            {
                "TEMPORAL_ADDRESS": temporal.client.service_client.config.target_host,
                "TEMPORAL_NAMESPACE": temporal.client.namespace,
                "ORBIT_EVENT_INGEST_URL": f"http://127.0.0.1:{port}/internal/events",
                "ORBIT_INTERNAL_TOKEN": INGEST_TOKEN,
                "ORBIT_WORKER_BIND": "127.0.0.1",
                # Every write reaches the log file at once, so a stopped
                # process leaves nothing behind in a buffer.
                "PYTHONUNBUFFERED": "1",
            }
        )
        real = {
            "ORBIT_MODEL_MODE": "real",
            "ORBIT_MODEL_BASE_URL": f"http://127.0.0.1:{port}/v1",
            "ORBIT_MODEL_API_KEY": MODEL_KEY,
            "ORBIT_MODEL_NAME": MODEL_NAME,
            "ORBIT_MODEL_TIMEOUT_SECONDS": "10",
        }
        modes = {"mock": {"ORBIT_MODEL_MODE": "mock"}, "real": real, "log": real}
        processes: dict[str, list[subprocess.Popen]] = {}
        for mode, extra in modes.items():
            env = {
                **base,
                **extra,
                "TEMPORAL_TASK_QUEUE": QUEUES[mode],
                "ORBIT_WORKER_PORT": str(_free_port()),
            }
            processes[mode] = [
                _start("orbit_orch.main", env, logs, f"orbit-orch-{mode}"),
                _start("orbit_worker.main", env, logs, f"orbit-worker-{mode}"),
            ]
        harness = Harness(temporal.client, recorder, processes, logs)
        try:
            for case in CASES:
                try:
                    row = await asyncio.wait_for(case(harness), CASE_TIMEOUT_S)
                    row["pass"] = row["expected"] == row["actual"]
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "id": case.__name__.replace("case_", "").replace("_", "-").upper(),
                        "steps": [],
                        "expected": "the case completes",
                        "actual": type(exc).__name__,
                        "pass": False,
                    }
                rows.append(row)
        finally:
            _stop([process for pair in processes.values() for process in pair])
    await runner.cleanup()

    failed = [row["id"] for row in rows if not row["pass"]]
    report = {
        "suite": "e2e-a1-events",
        "commit": _commit(commit),
        "components": _versions(info.server_version),
        "setup": [
            "Fresh local Temporal dev server (in-memory).",
            ("orbit-orch and orbit-worker subprocesses on task queue orbit-e2e-mock "
            "(ORBIT_MODEL_MODE=mock, streaming mock model)."),
            ("orbit-orch and orbit-worker subprocesses on task queue orbit-e2e-real "
            "(ORBIT_MODEL_MODE=real, OpenAI-compatible stub)."),
            ("orbit-orch and orbit-worker subprocesses on task queue orbit-e2e-log "
            "(ORBIT_MODEL_MODE=real, OpenAI-compatible stub), used only by E-LOG-1."),
            ("Every process runs with PYTHONUNBUFFERED=1 and writes stdout and stderr to "
            "separate files."),
            ("Workers post events to a recording ingest stub that requires the internal "
            "bearer token."),
            "Rooms are driven with the runTurn and decide Updates, as control drives them.",
        ],
        "ingest": {
            "eventsRecorded": bool(recorder.events),
            "unauthorizedPosts": recorder.unauthorized,
        },
        "cases": rows,
        "passed": len(rows) - len(failed),
        "failed": failed,
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{'PASS' if row['pass'] else 'FAIL'}  {row['id']}")
    print(f"{report['passed']}/{len(rows)} passed; report: {out}")
    return 1 if failed or recorder.unauthorized else 0


# Independent of the redactor under test: formats of real keys and tokens.
_SCAN_PATTERNS = {
    "database-url": re.compile(
        r"\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis|rediss|amqp|mssql)://",
        re.IGNORECASE,
    ),
    "url-credentials": re.compile(r"\b[a-z][a-z0-9+.-]*://[^/\s:@\"]+:[^/\s@\"]+@", re.IGNORECASE),
    "openai-key": re.compile(r"\bsk-[A-Za-z0-9_-]{6,}"),
    "github-token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|\bgithub_pat_\w{20,}"),
    "gitlab-token": re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}"),
    "aws-access-key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "google-api-key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}"),
    "slack-token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "bearer-credential": re.compile(r"\bBearer\s+(?!\[REDACTED\])[A-Za-z0-9._~+/=-]{8,}"),
    "private-key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


def scan(paths: list[Path], out: Path | None) -> int:
    findings = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for value in PLANTED:
            if value in text:
                findings.append({"file": str(path), "rule": "planted", "match": _fingerprint(value)})
        for rule, pattern in _SCAN_PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append(
                    {"file": str(path), "rule": rule, "match": _fingerprint(match.group(0))}
                )
    result = {
        "scanned": [str(path) for path in paths],
        "rules": ["planted", *_SCAN_PATTERNS],
        "findings": findings,
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"secret scan: {len(findings)} finding(s) in {len(paths)} file(s)")
    return 1 if findings else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--out", type=Path, default=Path("artifacts/e2e-a1-events.json"))
    run_parser.add_argument("--logs", type=Path, default=Path("e2e-logs"))
    run_parser.add_argument("--commit", default="", help="defaults to git rev-parse HEAD")
    scan_parser = sub.add_parser("scan")
    scan_parser.add_argument("paths", type=Path, nargs="+")
    scan_parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "scan":
        return scan(args.paths, args.out)
    return asyncio.run(run(args.out, args.logs, args.commit))


if __name__ == "__main__":
    sys.exit(main())
