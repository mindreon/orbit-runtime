"""Real-model smoke E2E (E-RM-1, E-RM-2) against the real Qwen endpoint.

``run`` reuses the A1/E-LOG harness in ``e2e_a1_events.py``: a fresh local
Temporal dev server, a recording ingest stub, and per case one orbit-workflows
(package ``orbit-orch``) and orbit-worker pair on its own task queue. Workers
run ``ORBIT_MODEL_MODE=real`` with the Postgres state store and a per-run
Fernet state key.

- ``rm-1``: ``ORBIT_MODEL_STREAM=false``. E-RM-1: one short turn completes with
  ``assistant.message`` and one ``usage`` event whose token counts match the
  provider's non-null usage.
- ``rm-2``: ``ORBIT_MODEL_STREAM=true``. E-RM-2: the gate counts at least 2
  provider content chunks. ``assistant.delta`` events are ordered and
  contiguous and join to the final ``assistant.message``. How many deltas the
  worker posted is recorded under ``observed`` and is not asserted.

The worker's ``ORBIT_MODEL_BASE_URL`` is a local budget gate that forwards
``/v1/chat/completions`` to the upstream unchanged (the worker's own
``Authorization`` header included). Before it forwards, the gate requires an
allowlisted upstream (or the in-process stub), ``model`` equal to
``ORBIT_MODEL_NAME``, and ``max_tokens`` equal to 64. It forwards at most
``PROVIDER_BUDGET`` requests per run and answers any further one with HTTP 400.
It records, per request, only the request's model, ``max_tokens``, ``stream``,
and ``include_usage``, the HTTP status, the number of streamed content chunks,
and the provider's token counts. Nothing it sees is logged.

``ORBIT_SMOKE_STUB_UPSTREAM=1`` (set only by the pull-request job in
``ci.yml``) does not relax the allowlist. The harness ignores
``ORBIT_MODEL_BASE_URL`` and forwards only to an in-process loopback stub.
The real workflow refuses to run when that variable is set.

After each case the pair is stopped; its complete stdout and stderr, every
Failure in the workflow history, every history event, the posted events, and
the stored state are searched for the model key, each half (as written,
JSON-escaped, and JSON-escaped twice), and, in logs and Failures, for the
prompt and its canary.

Real model output varies, so two real runs do not write the same bytes. The
report is structurally deterministic: the same keys, cases, steps, and
``expected`` values for one commit. ``structural`` prints the report with each
case's ``observed`` object removed, which is what ``cmp`` compares. ``observed``
holds ``usageEvent``, ``providerUsage``, ``providerContentChunks``,
``assistantDeltas``, ``assistantMessage``, ``historyFailureEntries``, and
``capturedBytes``. Top-level ``usage`` stays in that view.

``scan`` checks the report and every file under the log directory for the
same key and prompt forms, key and token formats, and database URLs, and
fails on a missing or 0-byte log. It reads the key from the environment and
never prints it.

    uv run python scripts/e2e_real_model_smoke.py run
    uv run python scripts/e2e_real_model_smoke.py scan --report artifacts-smoke/real-model-smoke.json --logs smoke-logs

Environment: ``ORBIT_MODEL_API_KEY``, ``ORBIT_MODEL_BASE_URL`` (allowlisted
unless the stub flag is set), ``ORBIT_MODEL_NAME``, ``ORBIT_SMOKE_POSTGRES_URL``
(an empty database).

    uv run python scripts/e2e_real_model_smoke.py structural --report <file>
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
import asyncpg
from aiohttp import web
from cryptography.fernet import Fernet
from e2e_a1_events import (
    _SCAN_PATTERNS,
    INGEST_TOKEN,
    LOG_PROCESSES,
    ORCH_STARTUP_LINE,
    PROCESS_NOTE,
    STREAMS,
    Harness,
    Recorder,
    _commit,
    _event_type,
    _failures,
    _free_port,
    _of,
    _start,
    _stop,
    _versions,
)
from orbit_contracts.models import RoomCommand, RoomWorkflowInput
from orbit_orch.workflows import RoomWorkflow
from orbit_worker.postgres_store import PostgresStateStore, StateCipher, decode_blob
from temporalio.api.workflowservice.v1 import GetSystemInfoRequest
from temporalio.client import WorkflowHandle, WorkflowUpdateFailedError
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment

SUITE = "real-model-smoke"
API_KEY_VAR = "ORBIT_MODEL_API_KEY"
BASE_URL_VAR = "ORBIT_MODEL_BASE_URL"
NAME_VAR = "ORBIT_MODEL_NAME"
POSTGRES_VAR = "ORBIT_SMOKE_POSTGRES_URL"
STUB_FLAG = "ORBIT_SMOKE_STUB_UPSTREAM"
# Exact hosts. https only. The workflow config step uses this same pair.
ALLOWED_HOSTS = frozenset({"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"})
MAX_TOKENS = 64
PROVIDER_BUDGET = 2
# 30s is tight for a US-hosted runner calling Beijing. 60s matches the worker default.
MODEL_TIMEOUT_S = 60
CASE_TIMEOUT_S = 180
# Halves of a shorter key would match unrelated text.
MIN_KEY_CHARS = 16
MODES = ("rm-1", "rm-2")
QUEUES = {mode: f"orbit-smoke-{mode}" for mode in MODES}
STREAM = {"rm-1": False, "rm-2": True}
CASE_IDS = {"rm-1": "E-RM-1", "rm-2": "E-RM-2"}
PROMPT_CANARY = "紫色企鹅在冰川上弹钢琴"
PROMPTS = {
    "rm-1": f"Smoke test E-RM-1 ({PROMPT_CANARY}). Reply with exactly one word: pong",
    # Longer than MAX_TOKENS on purpose, so the reply streams for the whole cap.
    "rm-2": (
        f"Smoke test E-RM-2 ({PROMPT_CANARY}). Count from 1 to 100 in digits, "
        "separated by single spaces, and write nothing else."
    ),
}
WORKER_STDOUT_LINE = "orbit-worker: starting"
ORCH_STDOUT_LINE = "orbit-orch: starting"
STATE_TABLES = ("orbit_agent_state", "orbit_agent_idempotency")
DETERMINISM = (
    "Structurally deterministic, not byte-identical. The real workflow does not compare "
    "reruns. structural removes each case's observed object for cmp. observed holds "
    "usageEvent (inputTokens, outputTokens, latencyMs), providerUsage, "
    "providerContentChunks, assistantDeltas, assistantMessage (bytes, sha256), "
    "historyFailureEntries, and capturedBytes. Top-level usage stays in the compared view."
)
STUB_REPLY = "pong"
STUB_STREAM_PARTS = ("one two three", " four five six")
STUB_USAGE = {
    "rm-1": {"prompt_tokens": 11, "completion_tokens": 1, "total_tokens": 12},
    "rm-2": {"prompt_tokens": 20, "completion_tokens": 6, "total_tokens": 26},
}


def _fingerprint(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _forms(label: str, value: str) -> list[tuple[str, str]]:
    """The value as written, JSON-escaped, and JSON-escaped twice."""

    return [
        (label, value),
        (f"{label}-json-escaped", json.dumps(value)[1:-1]),
        (f"{label}-json-escaped-twice", json.dumps(json.dumps(value))[2:-2]),
    ]


def _dedupe(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    kept = []
    for label, value in pairs:
        if value and value not in seen:
            seen.add(value)
            kept.append((label, value))
    return kept


def key_needles(key: str) -> list[tuple[str, str]]:
    half = len(key) // 2
    return _dedupe(
        [
            *_forms("model-key", key),
            *_forms("model-key-first-half", key[:half]),
            *_forms("model-key-second-half", key[half:]),
        ]
    )


def prompt_needles() -> list[tuple[str, str]]:
    return _dedupe(
        [
            *_forms("prompt-e-rm-1", PROMPTS["rm-1"]),
            *_forms("prompt-e-rm-2", PROMPTS["rm-2"]),
            *_forms("prompt-canary", PROMPT_CANARY),
        ]
    )


def _found(text: str, needles: list[tuple[str, str]]) -> list[str]:
    return sorted({label for label, value in needles if value in text})


def _found_bytes(data: bytes, needles: list[tuple[str, str]]) -> bool:
    return any(value.encode("utf-8") in data for _, value in needles)


def _tokens(usage: object) -> dict[str, int | None]:
    source = usage if isinstance(usage, dict) else {}
    counts: dict[str, int | None] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = source.get(name)
        counts[name] = value if isinstance(value, int) and not isinstance(value, bool) else None
    return counts


def _response_evidence(raw: bytes, stream: bool) -> dict[str, object]:
    usage: object = None
    content_chunks = 0
    items: list[object] = []
    if stream:
        for line in raw.splitlines():
            if not line.startswith(b"data:"):
                continue
            data = line[len(b"data:"):].strip()
            if data == b"[DONE]":
                continue
            try:
                items.append(json.loads(data))
            except ValueError:
                continue
    else:
        try:
            items.append(json.loads(raw))
        except ValueError:
            pass
    for item in items:
        if not isinstance(item, dict):
            continue
        for choice in item.get("choices") or []:
            if isinstance(choice, dict) and (choice.get("delta") or {}).get("content"):
                content_chunks += 1
        if item.get("usage"):
            usage = item["usage"]
    return {"contentChunks": content_chunks, "usage": _tokens(usage)}


def stub_requested() -> bool:
    return os.environ.get(STUB_FLAG, "").strip().lower() in {"1", "true", "yes"}


def base_url_allowed(url: str) -> bool:
    """HTTPS and an exact DashScope host. ``http`` and every other host fail."""

    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    return (
        parts.scheme.lower() == "https"
        and not parts.username
        and not parts.password
        and host in ALLOWED_HOSTS
        and parts.port in (None, 443)
    )


def loopback_url(url: str) -> bool:
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    return (
        parts.scheme.lower() in {"http", "https"}
        and not parts.username
        and not parts.password
        and host in {"127.0.0.1", "localhost", "::1"}
    )


def _sse(payload: dict[str, object]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


async def stub_chat(request: web.Request) -> web.StreamResponse:
    """OpenAI-compatible stub. Non-stream JSON, or SSE with two content chunks.

    The body is not logged. The reply is fixed text, never the prompt or the key.
    """

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    if not isinstance(body, dict):
        body = {}
    model = body.get("model") if isinstance(body.get("model"), str) else "stub"
    if body.get("stream") is True:
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await response.prepare(request)
        for index, text in enumerate(STUB_STREAM_PARTS):
            delta: dict[str, str] = {"content": text}
            if index == 0:
                delta["role"] = "assistant"
            await response.write(
                _sse(
                    {
                        "id": "chatcmpl-stub",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    }
                )
            )
        await response.write(
            _sse(
                {
                    "id": "chatcmpl-stub",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": model,
                    "choices": [],
                    "usage": STUB_USAGE["rm-2"],
                }
            )
        )
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response
    return web.json_response(
        {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": STUB_REPLY},
                    "finish_reason": "stop",
                }
            ],
            "usage": STUB_USAGE["rm-1"],
        }
    )


class BudgetGate:
    """Forwards at most ``budget`` chat requests, after checking them.

    ``model`` and ``max_tokens`` are checked before any upstream connection.
    ``stub`` is true only when this process created the loopback stub: the
    upstream then has to be loopback. Otherwise the upstream has to be an
    allowlisted DashScope URL.
    """

    def __init__(self, upstream: str, budget: int, *, model: str, stub: bool) -> None:
        self._upstream = upstream.rstrip("/")
        self._budget = budget
        self._model = model
        self._stub = stub
        self._session: aiohttp.ClientSession | None = None
        self.requests: list[dict[str, object]] = []

    def set_upstream(self, upstream: str) -> None:
        self._upstream = upstream.rstrip("/")

    def _upstream_allowed(self) -> bool:
        if self._stub:
            return loopback_url(self._upstream)
        return base_url_allowed(self._upstream)

    def _reject_before_forward(self, payload: dict[str, object]) -> bool:
        if not self._upstream_allowed():
            return True
        if payload.get("model") != self._model:
            return True
        return payload.get("max_tokens") != MAX_TOKENS

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=MODEL_TIMEOUT_S + 10)
        )

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()

    @property
    def forwarded(self) -> int:
        return sum(1 for record in self.requests if record["forwarded"])

    async def chat(self, request: web.Request) -> web.StreamResponse:
        body = await request.read()
        try:
            payload = json.loads(body)
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        stream = payload.get("stream") is True
        options = payload.get("stream_options")
        include_usage = isinstance(options, dict) and options.get("include_usage") is True
        record: dict[str, object] = {
            "forwarded": False,
            "status": None,
            "model": payload.get("model"),
            "maxTokens": payload.get("max_tokens"),
            "stream": stream,
            "includeUsage": include_usage,
            "contentChunks": 0,
            "usage": _tokens(None),
        }
        # Model name, max_tokens, and the upstream host are decided here, before
        # the post. A rejected request is not forwarded and does not use budget.
        if self._reject_before_forward(payload):
            record["status"] = 400
            self.requests.append(record)
            return web.json_response(
                {"error": {"message": "smoke gate rejected the request before forwarding"}},
                status=400,
            )
        if self.forwarded >= self._budget:
            self.requests.append(record)
            return web.json_response(
                {"error": {"message": "smoke provider budget exhausted"}}, status=400
            )
        record["forwarded"] = True
        self.requests.append(record)
        headers = {
            name: request.headers[name]
            for name in ("Authorization", "Content-Type", "Accept")
            if name in request.headers
        }
        headers["Accept-Encoding"] = "identity"
        response = web.StreamResponse()
        chunks: list[bytes] = []
        assert self._session is not None
        try:
            async with self._session.post(
                f"{self._upstream}/chat/completions", data=body, headers=headers
            ) as upstream:
                record["status"] = upstream.status
                response.set_status(upstream.status)
                response.headers["Content-Type"] = upstream.headers.get(
                    "Content-Type", "application/json"
                )
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    chunks.append(chunk)
                    await response.write(chunk)
            await response.write_eof()
        # The exception text can name the upstream host; none of it is kept.
        except Exception:  # noqa: BLE001
            if record["status"] is None:
                record["status"] = "unreachable"
            if not response.prepared:
                return web.Response(status=502)
            return response
        record.update(_response_evidence(b"".join(chunks), stream))
        return response


class SmokeHarness(Harness):
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


def _state_cipher(state_key: str) -> StateCipher:
    """The production cipher for this run's Fernet key. Plaintext stays off."""

    return StateCipher(fernet=Fernet(state_key.encode("ascii")), allow_plaintext=False)


async def _stored_state(
    postgres_url: str, room: str, state_key: str, needles: list[tuple[str, str]]
) -> dict[str, object]:
    conn = await asyncpg.connect(postgres_url)
    try:
        rows = await conn.fetch(
            "SELECT state_version, blob FROM orbit_agent_state WHERE room_id = $1", room
        )
    finally:
        await conn.close()
    blobs = [bytes(row["blob"]) for row in rows]
    return {
        "rows": len(rows),
        "stateVersion": max((int(row["state_version"]) for row in rows), default=None),
        "encrypted": bool(blobs) and all(blob.startswith(b"fernet:") for blob in blobs),
        "keyInStoredState": any(
            _found(decode_blob(blob, _state_cipher(state_key)).model_dump_json(), needles)
            for blob in blobs
        ),
    }


def _summary(text: str | None) -> dict[str, object] | None:
    if text is None:
        return None
    encoded = text.encode("utf-8")
    return {"bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()[:16]}


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _steps(mode: str, model: str) -> list[str]:
    stream = "true" if STREAM[mode] else "false"
    ask = (
        "a one-word reply"
        if mode == "rm-1"
        else "a counting reply longer than the output cap, so it streams for the whole cap"
    )
    return [
        (f"Open room {CASE_IDS[mode].lower()} on its own orbit-workflows and orbit-worker pair "
         f"(task queue {QUEUES[mode]}, ORBIT_MODEL_MODE=real, ORBIT_MODEL_NAME={model}, "
         f"ORBIT_MODEL_STREAM={stream}, ORBIT_MODEL_MAX_TOKENS={MAX_TOKENS}, "
         "ORBIT_MODEL_MAX_RETRIES=0, Postgres state store, encrypted state). Only the worker "
         "gets the model key, from the environment."),
        (f"runTurn tn-1 with a one-sentence prompt ({_fingerprint(PROMPTS[mode])}, canary "
         f"{_fingerprint(PROMPT_CANARY)}) asking for {ask}. The worker calls the upstream "
         f"once through the budget gate (at most {PROVIDER_BUDGET} forwarded requests per run, "
         f"timeout {MODEL_TIMEOUT_S}s). The gate checks the model name and max_tokens "
         "before it forwards."),
        ("Read the Update result, the events the worker posted for the room, the gate's record "
         "of the request (model, max_tokens, stream, include_usage, HTTP status, streamed "
         "content chunks, provider token counts), and the room's row in Postgres."),
        ("Fetch the workflow history. Search every Failure in it for the model key, each half, "
         "the prompt, and the canary, and every history event and posted event for the key and "
         "each half (each value as written, JSON-escaped, and JSON-escaped twice)."),
        ("Stop the pair, read orbit-worker's and orbit-workflows' complete stdout and stderr, "
         "and search them for the same values. A 0-byte log fails the case; each process writes "
         "a startup line to stdout and to stderr."),
    ]


async def run_case(
    h: SmokeHarness,
    gate: BudgetGate,
    mode: str,
    model: str,
    postgres_url: str,
    state_key: str,
    keys: list[tuple[str, str]],
) -> dict:
    case_id = CASE_IDS[mode]
    room = case_id.lower()
    handle = await h.open_room(room, mode)
    before = len(gate.requests)
    try:
        result = await handle.execute_update(
            "runTurn", {"turnId": "tn-1", "message": PROMPTS[mode]}
        )
    except WorkflowUpdateFailedError:
        result = {"status": "update failed"}
    requests = gate.requests[before:]
    request = requests[0] if requests else {}
    provider = request.get("usage") or _tokens(None)
    events = h.recorder.room(room)
    raws = h.recorder.raw(room)
    messages = _of(events, "assistant.message")
    text = messages[-1].get("text") if messages else None
    usage_events = _of(events, "usage")
    usage = usage_events[0] if usage_events else {}
    state = await _stored_state(postgres_url, room, state_key, keys)
    history = await handle.fetch_history()
    failures = [
        (_event_type(event), failure) for event in history.events for failure in _failures(event)
    ]
    leaks = keys + prompt_needles()
    stopped = h.stop(mode)
    names = {label: f"{name}-{mode}" for label, name in LOG_PROCESSES.items()}
    outputs = {label: h.output(name) for label, name in names.items()}
    sizes = {label: h.sizes(name) for label, name in names.items()}
    startup = {
        "orbit-worker": {"stdout": WORKER_STDOUT_LINE, "stderr": f"chat model: real model={model}"},
        "orbit-workflows": {"stdout": ORCH_STDOUT_LINE, "stderr": ORCH_STARTUP_LINE},
    }

    expected: dict[str, object] = {
        "updateStatus": "completed",
        "updateErrorCode": "",
        "updateModelMode": "real",
        "updateModelName": model,
        "updateResultHasKey": False,
        "providerRequestsForwarded": 1,
        "providerRequestsRefused": 0,
        "providerStatuses": [200],
        "requestModel": model,
        "requestMaxTokens": MAX_TOKENS,
        "requestStream": STREAM[mode],
        "requestIncludeUsage": STREAM[mode],
        "assistantMessages": 1,
        "assistantMessageNonEmpty": True,
        "turnFailedEvents": 0,
        "usageEvents": 1,
        "usageModel": model,
        "usageModelMode": "real",
        "usageInputTokensPositive": True,
        "usageOutputTokensPositive": True,
        "usageOutputTokensAtMostMaxTokens": True,
        "providerUsageNonNull": {"prompt_tokens": True, "completion_tokens": True,
                                 "total_tokens": True},
        "usageMatchesProvider": True,
    }
    actual: dict[str, object] = {
        "updateStatus": result.get("status"),
        "updateErrorCode": result.get("errorCode"),
        "updateModelMode": result.get("modelMode"),
        "updateModelName": result.get("modelName"),
        "updateResultHasKey": bool(_found(json.dumps(result), keys)),
        "providerRequestsForwarded": sum(1 for record in requests if record["forwarded"]),
        "providerRequestsRefused": sum(1 for record in requests if not record["forwarded"]),
        "providerStatuses": [record["status"] for record in requests if record["forwarded"]],
        "requestModel": request.get("model"),
        "requestMaxTokens": request.get("maxTokens"),
        "requestStream": request.get("stream"),
        "requestIncludeUsage": request.get("includeUsage"),
        "assistantMessages": len(messages),
        "assistantMessageNonEmpty": bool(text),
        "turnFailedEvents": len(_of(events, "turn.failed")),
        "usageEvents": len(usage_events),
        "usageModel": usage.get("model"),
        "usageModelMode": usage.get("modelMode"),
        "usageInputTokensPositive": _positive_int(usage.get("inputTokens")),
        "usageOutputTokensPositive": _positive_int(usage.get("outputTokens")),
        "usageOutputTokensAtMostMaxTokens": _positive_int(usage.get("outputTokens"))
        and usage["outputTokens"] <= MAX_TOKENS,
        "providerUsageNonNull": {name: value is not None for name, value in provider.items()},
        "usageMatchesProvider": provider["prompt_tokens"] is not None
        and usage.get("inputTokens") == provider["prompt_tokens"]
        and usage.get("outputTokens") == provider["completion_tokens"],
    }

    deltas = _of(events, "assistant.delta")
    if STREAM[mode]:
        message_at = next(
            (i for i, body in enumerate(events) if body["type"] == "assistant.message"),
            len(events),
        )
        before_message = [body for body in events[:message_at] if body["type"] == "assistant.delta"]
        expected.update(
            {
                "providerContentChunksAtLeastTwo": True,
                "deltaBlocks": 1,
                "deltaSeqContiguous": True,
                "deltaActivityAttempts": [1],
                "allDeltasBeforeMessage": True,
                "deltasConcatenateToMessage": True,
            }
        )
        actual.update(
            {
                "providerContentChunksAtLeastTwo": int(request.get("contentChunks") or 0) >= 2,
                "deltaBlocks": len({body.get("blockId") for body in deltas}),
                "deltaSeqContiguous": [body.get("seq") for body in deltas]
                == list(range(len(deltas))),
                "deltaActivityAttempts": sorted({body.get("activityAttempt") for body in deltas}),
                "allDeltasBeforeMessage": bool(deltas) and len(before_message) == len(deltas),
                "deltasConcatenateToMessage": bool(text)
                and "".join(body.get("delta", "") for body in deltas) == text,
            }
        )

    expected.update(
        {
            "stateStore": {"rows": 1, "stateVersion": 2, "encrypted": True,
                           "keyInStoredState": False},
            "keyInAnyEvent": [],
            "historyEventsWithKey": [],
            "historyFailuresWithKeyOrPrompt": [],
            "processesStopped": True,
            "zeroByteLogs": [],
            "startupLines": {label: {stream: True for stream in STREAMS} for label in names},
            **{label: {stream: [] for stream in STREAMS} for label in names},
        }
    )
    actual.update(
        {
            "stateStore": state,
            "keyInAnyEvent": sorted({label for raw in raws for label in _found(raw, keys)}),
            "historyEventsWithKey": sorted(
                {
                    _event_type(event)
                    for event in history.events
                    if _found_bytes(event.SerializeToString(), keys)
                }
            ),
            "historyFailuresWithKeyOrPrompt": sorted(
                {kind for kind, failure in failures
                 if _found_bytes(failure.SerializeToString(), leaks)}
            ),
            "processesStopped": stopped,
            "zeroByteLogs": sorted(
                f"{names[label]}.{stream}.log"
                for label in names
                for stream in STREAMS
                if sizes[label][stream] == 0
            ),
            "startupLines": {
                label: {stream: startup[label][stream] in outputs[label][stream]
                        for stream in STREAMS}
                for label in names
            },
            **{
                label: {stream: _found(outputs[label][stream], leaks) for stream in STREAMS}
                for label in names
            },
        }
    )
    return {
        "id": case_id,
        "title": (
            "One short real turn completes with assistant.message and a usage event with "
            "non-null token counts"
            if mode == "rm-1"
            else "Streaming: the gate sees at least 2 provider content chunks, and the "
            "assistant.delta events join to the final assistant.message"
        ),
        "steps": _steps(mode, model),
        "expected": expected,
        "actual": actual,
        "observed": {
            "usageEvent": {
                "inputTokens": usage.get("inputTokens"),
                "outputTokens": usage.get("outputTokens"),
                "latencyMs": usage.get("latencyMs"),
            },
            "providerUsage": provider,
            "providerContentChunks": request.get("contentChunks"),
            "assistantDeltas": len(deltas),
            "assistantMessage": _summary(text),
            "historyFailureEntries": len(failures),
            "capturedBytes": sizes,
        },
    }


def _missing_config(stub: bool) -> list[str]:
    required = [API_KEY_VAR, NAME_VAR, POSTGRES_VAR]
    if not stub:
        required.insert(1, BASE_URL_VAR)
    return [name for name in required if not os.environ.get(name, "").strip()]


async def _state_tables_empty(postgres_url: str) -> bool:
    conn = await asyncpg.connect(postgres_url)
    try:
        for table in STATE_TABLES:
            exists = await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", table)
            if exists and await conn.fetchval(f"SELECT count(*) FROM {table}"):
                return False
    finally:
        await conn.close()
    return True


def _usage_totals(rows: list[dict]) -> dict[str, object]:
    per_case = {
        row["id"]: dict((row.get("observed") or {}).get("providerUsage") or _tokens(None))
        for row in rows
    }
    summed: dict[str, int | None] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [counts.get(name) for counts in per_case.values()]
        summed[name] = (
            sum(v for v in values if isinstance(v, int))
            if values and all(isinstance(v, int) for v in values)
            else None
        )
    return {
        "source": ("Provider-reported usage per forwarded request, read by the budget gate; each "
                   "case also checks that its usage event carries the same prompt and completion "
                   "counts. null when a count is missing."),
        **per_case,
        "summed": summed,
    }


async def run(out: Path, logs: Path) -> int:
    stub = stub_requested()
    missing = _missing_config(stub)
    if missing:
        print(f"{SUITE}: required variable(s) are not set: {', '.join(missing)}", file=sys.stderr)
        return 2
    # Popped so the Temporal dev server and orbit-workflows never inherit it.
    key = os.environ.pop(API_KEY_VAR).strip()
    if len(key) < MIN_KEY_CHARS:
        print(f"{SUITE}: {API_KEY_VAR} is too short to search for its halves", file=sys.stderr)
        return 2
    model = os.environ[NAME_VAR].strip()
    postgres_url = os.environ[POSTGRES_VAR].strip()
    # Checked before Postgres and before the gate exists, so a bad URL never
    # becomes a forwarded request. The stub flag ignores this variable.
    upstream = ""
    if stub:
        print(
            f"{SUITE}: {STUB_FLAG} is set; the only upstream is the in-process stub",
            file=sys.stderr,
        )
    else:
        upstream = os.environ[BASE_URL_VAR].strip()
        if not base_url_allowed(upstream):
            print(
                f"{SUITE}: {BASE_URL_VAR} is not an allowlisted https DashScope host",
                file=sys.stderr,
            )
            return 2
    try:
        empty = await _state_tables_empty(postgres_url)
    except Exception:  # noqa: BLE001
        print(f"{SUITE}: cannot connect to the database in {POSTGRES_VAR}", file=sys.stderr)
        return 2
    if not empty:
        print(f"{SUITE}: the database in {POSTGRES_VAR} already holds agent state; "
              "use an empty database", file=sys.stderr)
        return 2
    # Two workers creating the tables at once on an empty database can fail
    # with UniqueViolationError (https://github.com/mindreon/orbit-runtime/issues/8).
    # The harness creates the schema before they start. The product race is not fixed here.
    state_key = Fernet.generate_key().decode("ascii")
    await PostgresStateStore(
        lambda: asyncpg.connect(postgres_url), _state_cipher(state_key)
    ).ensure_schema()
    keys = key_needles(key)
    out.parent.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    recorder = Recorder()
    gate = BudgetGate("", PROVIDER_BUDGET, model=model, stub=stub)
    await gate.start()
    app = web.Application()
    app.router.add_post("/internal/events", recorder.ingest)
    app.router.add_post("/v1/chat/completions", gate.chat)
    if stub:
        app.router.add_post("/stub/v1/chat/completions", stub_chat)
    runner = web.AppRunner(app)
    await runner.setup()
    port = _free_port()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    if stub:
        gate.set_upstream(f"http://127.0.0.1:{port}/stub/v1")
    else:
        gate.set_upstream(upstream)

    rows: list[dict] = []
    log_files: list[str] = []
    try:
        async with await WorkflowEnvironment.start_local(
            data_converter=pydantic_data_converter
        ) as temporal:
            info = await temporal.client.workflow_service.get_system_info(GetSystemInfoRequest())
            base = {k: v for k, v in os.environ.items() if not k.startswith("ORBIT_")}
            base.update(
                {
                    "TEMPORAL_ADDRESS": temporal.client.service_client.config.target_host,
                    "TEMPORAL_NAMESPACE": temporal.client.namespace,
                    "PYTHONUNBUFFERED": "1",
                }
            )
            processes: dict[str, list[subprocess.Popen]] = {}
            for mode in MODES:
                orch_env = {**base, "TEMPORAL_TASK_QUEUE": QUEUES[mode]}
                worker_env = {
                    **orch_env,
                    "ORBIT_EVENT_INGEST_URL": f"http://127.0.0.1:{port}/internal/events",
                    "ORBIT_INTERNAL_TOKEN": INGEST_TOKEN,
                    "ORBIT_WORKER_BIND": "127.0.0.1",
                    "ORBIT_WORKER_PORT": str(_free_port()),
                    "ORBIT_MODEL_MODE": "real",
                    "ORBIT_MODEL_BASE_URL": f"http://127.0.0.1:{port}/v1",
                    API_KEY_VAR: key,
                    "ORBIT_MODEL_NAME": model,
                    "ORBIT_MODEL_TIMEOUT_SECONDS": str(MODEL_TIMEOUT_S),
                    "ORBIT_MODEL_MAX_TOKENS": str(MAX_TOKENS),
                    "ORBIT_MODEL_MAX_RETRIES": "0",
                    "ORBIT_MODEL_STREAM": "true" if STREAM[mode] else "false",
                    "ORBIT_STATE_STORE_URL": postgres_url,
                    "ORBIT_STATE_KEY": state_key,
                }
                processes[mode] = [
                    _start("orbit_orch.main", orch_env, logs, f"orbit-orch-{mode}"),
                    _start("orbit_worker.main", worker_env, logs, f"orbit-worker-{mode}"),
                ]
                log_files += [
                    f"{name}-{mode}.{stream}.log"
                    for name in ("orbit-orch", "orbit-worker")
                    for stream in STREAMS
                ]
            harness = SmokeHarness(temporal.client, recorder, processes, logs)
            try:
                for mode in MODES:
                    try:
                        row = await asyncio.wait_for(
                            run_case(harness, gate, mode, model, postgres_url, state_key, keys),
                            CASE_TIMEOUT_S,
                        )
                        row["pass"] = row["expected"] == row["actual"]
                    except Exception as exc:  # noqa: BLE001
                        row = {
                            "id": CASE_IDS[mode],
                            "steps": _steps(mode, model),
                            "expected": "the case completes",
                            "actual": type(exc).__name__,
                            "pass": False,
                        }
                    rows.append(row)
            finally:
                _stop([process for pair in processes.values() for process in pair])
    finally:
        await runner.cleanup()
        await gate.close()

    forwarded = gate.forwarded
    refused = len(gate.requests) - forwarded
    failed = [row["id"] for row in rows if not row["pass"]]
    if recorder.unauthorized or forwarded > PROVIDER_BUDGET:
        failed.append("run")
    report = {
        "suite": SUITE,
        "gitSha": _commit(""),
        "components": _versions(info.server_version),
        "model": {
            "name": model,
            "mode": "real",
            "endpoint": (
                "in-process OpenAI-compatible stub; ORBIT_MODEL_BASE_URL is not an upstream"
                if stub
                else f"allowlisted HTTPS endpoint from {BASE_URL_VAR} (host not recorded)"
            ),
            "maxTokens": MAX_TOKENS,
            "requestTimeoutSeconds": MODEL_TIMEOUT_S,
            "clientRetries": 0,
            "providerRequestBudget": PROVIDER_BUDGET,
        },
        "determinism": DETERMINISM,
        "setup": [
            "Fresh local Temporal dev server (in-memory).",
            ("Per case, orbit-workflows (current package/entrypoint name orbit-orch) and "
             "orbit-worker subprocesses on their own task queue; the worker runs "
             "ORBIT_MODEL_MODE=real with the Postgres state store and a per-run Fernet state key."),
            ("The worker's ORBIT_MODEL_BASE_URL is a local budget gate. Before forwarding, "
             "the gate checks the upstream, the model name, and max_tokens. It forwards at most "
             f"{PROVIDER_BUDGET} chat requests per run to "
             + (
                 "an in-process loopback stub. ORBIT_MODEL_BASE_URL is not an upstream."
                 if stub
                 else "the allowlisted HTTPS endpoint from ORBIT_MODEL_BASE_URL."
             )
             + " It records only request parameters, HTTP status, chunk counts, and token counts."),
            ("Only orbit-worker gets the model key, from the environment; the harness removes it "
             "from its own environment before starting Temporal and orbit-workflows."),
            ("Every process runs with PYTHONUNBUFFERED=1 and writes stdout and stderr to "
             "separate files."),
            ("Workers post events to a recording ingest stub that requires the internal "
             "bearer token."),
            "Rooms are driven with the runTurn Update, as control drives them.",
        ],
        "processNames": {
            "orbit-workflows": PROCESS_NOTE,
            "orbit-worker": "Package and entrypoint orbit-worker (python -m orbit_worker.main).",
        },
        "logFiles": log_files,
        "ingest": {
            "eventsRecorded": bool(recorder.events),
            "unauthorizedPosts": recorder.unauthorized,
        },
        "providerRequests": {
            "budget": PROVIDER_BUDGET,
            "forwarded": forwarded,
            "refused": refused,
        },
        "cases": rows,
        "usage": _usage_totals(rows),
        "passed": len(rows) - len([row for row in rows if not row["pass"]]),
        "failed": failed,
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{'PASS' if row['pass'] else 'FAIL'}  {row['id']}")
    print(f"{report['passed']}/{len(rows)} passed; provider requests forwarded {forwarded}, "
          f"refused {refused}; report: {out}")
    return 1 if failed else 0


def scan(report: Path, logs: Path, out: Path | None) -> int:
    key = os.environ.get(API_KEY_VAR, "").strip()
    if len(key) < MIN_KEY_CHARS:
        print(f"{SUITE} scan: {API_KEY_VAR} is not set or too short", file=sys.stderr)
        return 2
    needles = key_needles(key) + prompt_needles()
    findings: list[dict[str, str]] = []
    paths = [report, *sorted(path for path in logs.rglob("*") if path.is_file())]
    if report.is_file():
        try:
            named = json.loads(report.read_text(encoding="utf-8")).get("logFiles") or []
        except ValueError:
            named = []
            findings.append({"file": str(report), "rule": "unreadable-report"})
        for name in named:
            if not (logs / name).is_file():
                findings.append({"file": str(logs / name), "rule": "missing-file"})
    for path in paths:
        if not path.is_file():
            findings.append({"file": str(path), "rule": "missing-file"})
            continue
        if path.stat().st_size == 0:
            findings.append({"file": str(path), "rule": "empty-file"})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label in _found(text, needles):
            findings.append({"file": str(path), "rule": label})
        for rule, pattern in _SCAN_PATTERNS.items():
            if pattern.search(text):
                findings.append({"file": str(path), "rule": rule})
    result = {
        "scanned": [str(path) for path in paths],
        "rules": [
            "missing-file",
            "empty-file",
            *sorted({label for label, _ in needles}),
            *_SCAN_PATTERNS,
        ],
        "findings": findings,
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{SUITE} scan: {len(findings)} finding(s) in {len(paths)} file(s)")
    return 1 if findings else 0


def structural(report_path: Path) -> int:
    """Print the report with each case's ``observed`` object removed.

    The pull-request job writes this twice and compares the bytes with ``cmp``.
    """

    if not report_path.is_file():
        print(f"{SUITE} structural: report is missing", file=sys.stderr)
        return 2
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except ValueError:
        print(f"{SUITE} structural: report is not JSON", file=sys.stderr)
        return 2
    if isinstance(report, dict):
        cases = report.get("cases")
        if isinstance(cases, list):
            for case in cases:
                if isinstance(case, dict):
                    case.pop("observed", None)
    sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument(
        "--out", type=Path, default=Path("artifacts-smoke/real-model-smoke.json")
    )
    run_parser.add_argument("--logs", type=Path, default=Path("smoke-logs"))
    scan_parser = sub.add_parser("scan")
    scan_parser.add_argument("--report", type=Path, required=True)
    scan_parser.add_argument("--logs", type=Path, required=True)
    scan_parser.add_argument("--out", type=Path)
    structural_parser = sub.add_parser("structural")
    structural_parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "scan":
        return scan(args.report, args.logs, args.out)
    if args.command == "structural":
        return structural(args.report)
    return asyncio.run(run(args.out, args.logs))


if __name__ == "__main__":
    sys.exit(main())
