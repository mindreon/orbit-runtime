"""End-to-end sign-off check for the production state key (E-SK-1 .. E-SK-5).

``run`` reuses the ``e2e_a1_events.py`` harness: a fresh local Temporal dev
server, one orbit-orch subprocess, the recording ingest stub, and rooms
driven with the ``runTurn`` and ``decide`` Updates. orbit-worker runs the
mock model against the Postgres state store at ``ORBIT_TEST_POSTGRES_URL``,
in a schema this script drops and recreates. Each case starts, stops, and
restarts worker subprocesses with the state key configuration it needs; one
worker polls at a time. ``--control-bin`` adds an orbit-control process
(in-memory storage, local dev principal) on the same Temporal queue.

- E-SK-1: production configurations without a usable key stop the worker.
- E-SK-2: a ``plain:`` blob read in production fails the turn once.
- E-SK-3: a blob written with key A read with key B fails the turn once, and
  ``decide`` (resolveApproval) fails with the same fixed message.
- E-SK-4: blobs are ``fernet:``; context survives worker restarts.
- E-SK-5: abort and control DELETE close a room whose state is unreadable.

The report has the commit (``git rev-parse HEAD``, no override), component
versions, and per case: id, steps, expected, actual, pass. It has no
timestamps, ports, session ids, or key material, so two runs on one commit
write the same bytes. ``run`` exits 1
when a case fails.

``scan`` checks reports and process logs for the test keys, any 6-character
fragment of them, the ingest token, Fernet-key-shaped strings, and the A1
key, token, and database URL formats. It exits 1 on any finding.

    uv run python scripts/e2e_state_key.py run --out artifacts/e2e-state-key.json \\
        --control-bin /tmp/orbit-control
    uv run python scripts/e2e_state_key.py scan artifacts/e2e-state-key.json e2e-logs/state-key
"""

import argparse
import asyncio
import base64
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import aiohttp
import asyncpg
import e2e_a1_events as a1
from aiohttp import web
from cryptography.fernet import Fernet
from orbit_contracts.models import DecideOutcome, resume_turn_id
from orbit_orch.workflows import RoomWorkflow
from temporalio.api.workflowservice.v1 import GetSystemInfoRequest
from temporalio.client import WorkflowHandle
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment

POSTGRES_VAR = "ORBIT_TEST_POSTGRES_URL"
KEY_VAR = "ORBIT_STATE_KEY"
FLAG_VAR = "ORBIT_ALLOW_PLAINTEXT_STATE"
SCHEMA = "orbit_e2e_state_key"
QUEUE = a1.QUEUES["mock"]
CASE_TIMEOUT_S = 240
STARTUP_TIMEOUT_S = 60
FRAGMENT = 6
CONTROL_ORIGIN = "http://orbit-e2e.local"


def _key(label: str) -> str:
    digest = hashlib.sha256(f"orbit-e2e-state-key:{label}".encode()).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii")


# Deterministic, so two runs write the same report; never printed in it.
KEY_A = _key("A")
KEY_B = _key("B")
# 30 bytes: base64 of the wrong length, random-looking like a real key.
MALFORMED_KEY = base64.urlsafe_b64encode(
    hashlib.sha256(b"orbit-e2e-state-key:malformed").digest()[:30]
).decode("ascii")
KEYS = {"A": KEY_A, "B": KEY_B, "malformed": MALFORMED_KEY}
PLANTED = [KEY_A, KEY_B, MALFORMED_KEY, a1.INGEST_TOKEN]

UNREADABLE_CODE = "state_unreadable"
UNREADABLE_MESSAGE = "此任务的运行状态已失效，无法继续。你可以查看记录，或新建任务继续工作。"
MISSING_KEY_EXIT = (
    "orbit-worker: ORBIT_STATE_KEY is not set. The Postgres state store needs a Fernet key "
    "(only ORBIT_ALLOW_PLAINTEXT_STATE=1 allows plaintext state, for local development)."
)
MALFORMED_KEY_EXIT = (
    "orbit-worker: ORBIT_STATE_KEY is not a valid Fernet key "
    "(32 url-safe base64-encoded bytes)."
)
ENCRYPTED_START = "state store: postgres, encrypted"
PLAINTEXT_START = (
    "state store: postgres, not encrypted, plaintext state allowed "
    "(ORBIT_ALLOW_PLAINTEXT_STATE=1)"
)
# No "sk-" in it: the event redactor would treat that as a provider key.
CONTEXT_MARKER = "context-marker-case-four"


def _fragments(secret: str) -> set[str]:
    return {secret[i : i + FRAGMENT] for i in range(len(secret) - FRAGMENT + 1)}


def _leaks(text: str) -> dict[str, int]:
    """Per test key, how many of its fragments appear in ``text``."""

    return {
        name: sum(fragment in text for fragment in _fragments(secret))
        for name, secret in KEYS.items()
    }


NO_LEAKS = {name: 0 for name in KEYS}


class Worker:
    def __init__(self, process: subprocess.Popen, logs: Path, name: str) -> None:
        self.process = process
        self.logs = logs
        self.name = name

    def stream(self, stream: str) -> str:
        return (self.logs / f"{self.name}.{stream}.log").read_text(encoding="utf-8")

    def text(self) -> str:
        return self.stream("stderr") + self.stream("stdout")

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


class Control:
    """orbit-control's public HTTP API, as the web client calls it."""

    def __init__(self, process: subprocess.Popen, port: int) -> None:
        self.process = process
        self.base = f"http://127.0.0.1:{port}"

    async def call(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        headers = {"X-Orbit-Request": "1", "Origin": CONTROL_ORIGIN}
        timeout = aiohttp.ClientTimeout(total=CASE_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session, session.request(
            method, self.base + path, json=body, headers=headers
        ) as response:
            text = await response.text()
            return response.status, json.loads(text) if text.strip() else {}

    async def wait_healthy(self) -> None:
        for _ in range(STARTUP_TIMEOUT_S * 10):
            if self.process.poll() is not None:
                raise RuntimeError("orbit-control exited during startup")
            try:
                if (await self.call("GET", "/health"))[0] == 200:
                    return
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(0.1)
        raise TimeoutError("orbit-control never became healthy")


class Stack:
    """Starts orbit-worker subprocesses; the harness drives the rooms."""

    def __init__(
        self,
        harness: a1.Harness,
        env: dict[str, str],
        logs: Path,
        dsn: str,
        control: Control | None,
    ) -> None:
        self.harness = harness
        self.env = env
        self.logs = logs
        self.dsn = dsn
        self.control = control

    async def worker(self, name: str, extra: dict[str, str]) -> tuple[Worker, str]:
        """Start a worker; return it and ``running`` or ``exited``."""

        port = a1._free_port()
        env = {**self.env, **extra, "ORBIT_WORKER_PORT": str(port)}
        name = f"orbit-worker-{name}"
        worker = Worker(a1._start("orbit_worker.main", env, self.logs, name), self.logs, name)
        timeout = aiohttp.ClientTimeout(total=1)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for _ in range(STARTUP_TIMEOUT_S * 10):
                if worker.process.poll() is not None:
                    return worker, "exited"
                try:
                    async with session.get(f"http://127.0.0.1:{port}/") as response:
                        if response.status == 200:
                            return worker, "running"
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(0.1)
        worker.stop()
        raise TimeoutError(f"worker {name} neither started nor exited")

    async def running(self, name: str, extra: dict[str, str]) -> Worker:
        worker, outcome = await self.worker(name, extra)
        if outcome != "running":
            raise RuntimeError(f"worker {name} exited during startup")
        return worker

    async def blob(self, handle: WorkflowHandle) -> bytes:
        return await self.blob_of((await handle.query(RoomWorkflow.snapshot)).session_id)

    async def blob_of(self, session_id: str | None) -> bytes:
        conn = await asyncpg.connect(self.dsn)
        try:
            row = await conn.fetchrow(
                "SELECT blob FROM orbit_agent_state WHERE session_id = $1", session_id
            )
        finally:
            await conn.close()
        return bytes(row["blob"]) if row is not None else b""


def _prefix(blob: bytes) -> str:
    for prefix in ("fernet:", "plain:"):
        if blob.startswith(prefix.encode()):
            return prefix
    return "unknown"


async def _activity_runs(
    handle: WorkflowHandle, activity_type: str, turn_id: str | None = None
) -> dict:
    """What Temporal recorded for an Activity: of one turn, or every run of it."""

    scheduled: set[int] = set()
    attempts: list[int] = []
    outcomes = {"completed": 0, "failed": 0, "timedOut": 0, "canceled": 0}
    history = await handle.fetch_history()
    for event in history.events:
        if event.HasField("activity_task_scheduled_event_attributes"):
            attrs = event.activity_task_scheduled_event_attributes
            payload = json.loads(attrs.input.payloads[0].data) if attrs.input.payloads else {}
            if attrs.activity_type.name == activity_type and (
                turn_id is None or payload.get("turn_id") == turn_id
            ):
                scheduled.add(event.event_id)
        elif event.HasField("activity_task_started_event_attributes"):
            attrs = event.activity_task_started_event_attributes
            if attrs.scheduled_event_id in scheduled:
                attempts.append(attrs.attempt)
        for field, name in (
            ("activity_task_completed_event_attributes", "completed"),
            ("activity_task_failed_event_attributes", "failed"),
            ("activity_task_timed_out_event_attributes", "timedOut"),
            ("activity_task_canceled_event_attributes", "canceled"),
        ):
            if event.HasField(field) and getattr(event, field).scheduled_event_id in scheduled:
                outcomes[name] += 1
    return {"activity": activity_type, "scheduled": len(scheduled), "attempts": attempts, **outcomes}


def _failure(turn_id: str) -> dict:
    return {
        "turnId": turn_id,
        "agentId": "main",
        "errorCode": UNREADABLE_CODE,
        "retryable": False,
        "message": UNREADABLE_MESSAGE,
    }


def _exactly_once(activity_type: str = "runTurn") -> dict:
    return {
        "activity": activity_type,
        "scheduled": 1,
        "attempts": [1],
        "completed": 1,
        "failed": 0,
        "timedOut": 0,
        "canceled": 0,
    }


def _worker_log_facts(worker: Worker) -> dict:
    text = worker.text()
    return {
        "tracebacks": text.count("Traceback"),
        "unreadableLogLines": sum(
            f"[{UNREADABLE_CODE}]" in line for line in text.splitlines()
        ),
        "keyFragments": _leaks(text),
    }


# E-SK-1 rows: label, extra environment, expected startup.
STARTUP_ROWS: list[tuple[str, dict[str, str], dict]] = [
    ("flag unset, key unset", {}, {"outcome": "exited", "lastLine": MISSING_KEY_EXIT}),
    (
        "flag unset, key malformed",
        {KEY_VAR: MALFORMED_KEY},
        {"outcome": "exited", "lastLine": MALFORMED_KEY_EXIT},
    ),
    (
        "flag name misspelled (ORBIT_ALLOW_PLAINTEXT_STATES=1), key unset",
        {"ORBIT_ALLOW_PLAINTEXT_STATES": "1"},
        {"outcome": "exited", "lastLine": MISSING_KEY_EXIT},
    ),
    (
        "flag name misspelled (orbit_allow_plaintext_state=1), key unset",
        {"orbit_allow_plaintext_state": "1"},
        {"outcome": "exited", "lastLine": MISSING_KEY_EXIT},
    ),
    (
        "flag value misspelled ('true'), key unset",
        {FLAG_VAR: "true"},
        {"outcome": "exited", "lastLine": MISSING_KEY_EXIT},
    ),
    (
        "flag value misspelled ('1 ' with a trailing space), key unset",
        {FLAG_VAR: "1 "},
        {"outcome": "exited", "lastLine": MISSING_KEY_EXIT},
    ),
    (
        "flag empty, key unset",
        {FLAG_VAR: ""},
        {"outcome": "exited", "lastLine": MISSING_KEY_EXIT},
    ),
    (
        "flag empty, key malformed",
        {FLAG_VAR: "", KEY_VAR: MALFORMED_KEY},
        {"outcome": "exited", "lastLine": MALFORMED_KEY_EXIT},
    ),
    (
        "flag 1, key malformed (a set key must be valid)",
        {FLAG_VAR: "1", KEY_VAR: MALFORMED_KEY},
        {"outcome": "exited", "lastLine": MALFORMED_KEY_EXIT},
    ),
    (
        "control: flag unset, valid key",
        {KEY_VAR: KEY_A},
        {"outcome": "running", "startupLine": ENCRYPTED_START},
    ),
    (
        "control: flag 1, key unset",
        {FLAG_VAR: "1"},
        {"outcome": "running", "startupLine": PLAINTEXT_START},
    ),
]


async def case_e_sk_1(stack: Stack) -> dict:
    expected: dict[str, object] = {}
    actual: dict[str, object] = {}
    for index, (label, extra, want) in enumerate(STARTUP_ROWS, start=1):
        worker, outcome = await stack.worker(f"e-sk-1-{index:02d}", extra)
        if outcome == "running":
            worker.stop()
        text = worker.text()
        lines = [line for line in worker.stream("stderr").splitlines() if line.strip()]
        if want["outcome"] == "exited":
            expected[label] = {
                "outcome": "exited",
                "exitCode": 1,
                "lastLine": want["lastLine"],
                "tracebacks": 0,
                "keyFragmentsInLog": NO_LEAKS,
            }
            actual[label] = {
                "outcome": outcome,
                "exitCode": worker.process.returncode if outcome == "exited" else None,
                "lastLine": lines[-1] if lines else "",
                "tracebacks": text.count("Traceback"),
                "keyFragmentsInLog": _leaks(text),
            }
        else:
            expected[label] = {
                "outcome": "running",
                "startupLine": True,
                "keyFragmentsInLog": NO_LEAKS,
            }
            actual[label] = {
                "outcome": outcome,
                "startupLine": want["startupLine"] in lines,
                "keyFragmentsInLog": _leaks(text),
            }
    return {
        "id": "E-SK-1",
        "title": "Production is the default: no usable key stops the worker at startup",
        "steps": [
            (f"For each row, start orbit-worker with ORBIT_STATE_STORE_URL set (Postgres), "
             f"ORBIT_MODEL_MODE=mock, and only the row's {FLAG_VAR} / {KEY_VAR} settings."),
            ("Wait until the process exits or its health port answers; stop a running one. "
             "The malformed key is 40 url-safe base64 characters (30 bytes)."),
            ("Read the exit code, the last stderr line, and in stdout and stderr the "
             "'Traceback' count and how "
             f"many {FRAGMENT}-character fragments of each test key (A, B, malformed) appear."),
            "The two control rows show that only a valid key or the exact flag 1 starts it.",
        ],
        "expected": expected,
        "actual": actual,
    }


async def _unreadable_case(
    stack: Stack,
    room: str,
    seed_env: dict[str, str],
    seed_label: str,
    read_env: dict[str, str],
    read_label: str,
    seed_prefix: str,
    also_decide: bool = False,
) -> dict:
    h = stack.harness
    # decide is legal only while the room is awaiting_approval. tn-2 leaves its
    # room running, so the resolveApproval path is a second room that key A parked.
    # Not "{room}-decide": that id contains "sk-" plus a long token, which the
    # secret scan treats as an OpenAI key.
    decide_room = "unreadable-decide"
    decide_handle: WorkflowHandle | None = None
    parked: dict[str, object] | None = None
    decided: DecideOutcome | None = None
    decide_turn_id = ""
    seeder = await stack.running(f"{room}-seed", seed_env)
    try:
        handle = await h.open_room(room, "mock")
        first = await handle.execute_update("runTurn", {"turnId": "tn-1", "message": "hello"})
        if also_decide:
            decide_handle = await h.open_room(decide_room, "mock")
            parked = await decide_handle.execute_update(
                "runTurn", {"turnId": "tn-1", "message": "echo:decide-unreadable"}
            )
    finally:
        seeder.stop()
    seeded = await stack.blob(handle)
    decide_seeded = await stack.blob(decide_handle) if decide_handle is not None else b""
    prior = [dict(body) for body in h.recorder.room(room)]
    reader = await stack.running(f"{room}-read", read_env)
    decide_runs: dict | None = None
    decide_failure: dict | None = None
    try:
        result = await handle.execute_update(
            "runTurn", {"turnId": "tn-2", "message": "hello again"}
        )
        if also_decide and decide_handle is not None and parked is not None:
            approval = parked.get("approval") or {}
            if not isinstance(approval, dict):
                approval = {}
            prior_decide = [dict(body) for body in h.recorder.room(decide_room)]
            approval_id = str(approval.get("approvalRequestId") or "")
            decide_turn_id = resume_turn_id(approval_id)
            decided = await decide_handle.execute_update(
                "decide",
                {
                    "decision": "allow",
                    "approvalRequestId": approval_id,
                },
                result_type=DecideOutcome,
            )
            decide_runs = await _activity_runs(decide_handle, "resolveApproval", decide_turn_id)
            added_decide = h.recorder.room(decide_room)[len(prior_decide) :]
            failed_decide = [body for body in added_decide if body["type"] == "turn.failed"]
            decide_failure = failed_decide[0].get("failure") if failed_decide else None
        snapshot = await handle.query(RoomWorkflow.snapshot)
        runs = await _activity_runs(handle, "runTurn", "tn-2")
    finally:
        reader.stop()
    after = await stack.blob(handle)
    decide_after = await stack.blob(decide_handle) if decide_handle is not None else b""
    events = h.recorder.room(room)
    added = events[len(prior) :]
    failed = [body for body in added if body["type"] == "turn.failed"]
    expected = {
        "seedTurn": {"status": "completed", "texts": ["hello"]},
        "seededBlobPrefix": seed_prefix,
        "update": {
            "status": "failed",
            "errorCode": UNREADABLE_CODE,
            "retryable": False,
            "error": UNREADABLE_MESSAGE,
        },
        "newEventTypes": ["turn.failed", "session.status"],
        "failure": _failure("tn-2"),
        "temporalRunTurnTn2": _exactly_once(),
        "priorEventsUnchanged": True,
        "priorAssistantMessages": ["hello"],
        "blobUnchanged": True,
        "roomStatus": "running",
        "roomError": UNREADABLE_MESSAGE,
        "readerLog": {"tracebacks": 0, "unreadableLogLines": 1, "keyFragments": NO_LEAKS},
    }
    actual = {
        "seedTurn": {"status": first.get("status"), "texts": first.get("texts")},
        "seededBlobPrefix": _prefix(seeded),
        "update": {
            "status": result.get("status"),
            "errorCode": result.get("errorCode"),
            "retryable": result.get("retryable"),
            "error": result.get("error"),
        },
        "newEventTypes": [body["type"] for body in added],
        "failure": failed[0].get("failure") if failed else None,
        "temporalRunTurnTn2": runs,
        "priorEventsUnchanged": events[: len(prior)] == prior,
        "priorAssistantMessages": [
            body.get("text") for body in prior if body["type"] == "assistant.message"
        ],
        "blobUnchanged": bool(seeded) and seeded == after,
        "roomStatus": snapshot.status,
        "roomError": snapshot.error,
        "readerLog": _worker_log_facts(reader),
    }
    steps = [
        f"Start orbit-worker with {seed_label}.",
        f"Open room {room}; runTurn tn-1 'hello' (mock model answers 'hello').",
        ("Stop the worker. Read the session's row in orbit_agent_state: the stored blob "
         f"starts with '{seed_prefix}'. Keep a copy of the blob and of the recorded events."),
        f"Start orbit-worker with {read_label}; runTurn tn-2 'hello again'.",
        ("Read the Update result, the room snapshot, the events recorded after tn-1, the "
         "Temporal history of the runTurn Activity for tn-2 (scheduled, attempts, outcome), "
         "the stored blob again, and the worker log."),
    ]
    if also_decide:
        seed_approval = parked.get("approval") if isinstance(parked, dict) else None
        if not isinstance(seed_approval, dict):
            seed_approval = {}
        expected["decideSeed"] = {"status": "needs_approval", "toolName": "gated_echo"}
        # DecideOutcome has no reply text. The fixed message is on turn.failed.
        expected["decide"] = {
            "turnStatus": "failed",
            "errorCode": UNREADABLE_CODE,
        }
        expected["decideFailure"] = _failure(decide_turn_id)
        expected["decideActivity"] = _exactly_once("resolveApproval")
        expected["decideBlobUnchanged"] = True
        expected["readerLog"] = {"tracebacks": 0, "unreadableLogLines": 2, "keyFragments": NO_LEAKS}
        actual["decideSeed"] = {
            "status": parked.get("status") if isinstance(parked, dict) else None,
            "toolName": seed_approval.get("toolName"),
        }
        actual["decide"] = {
            "turnStatus": decided.turn_status if decided is not None else None,
            "errorCode": decided.error_code if decided is not None else None,
        }
        actual["decideFailure"] = decide_failure
        actual["decideActivity"] = decide_runs
        actual["decideBlobUnchanged"] = bool(decide_seeded) and decide_seeded == decide_after
        steps.append(
            f"Still on that key-B worker, after tn-2: room {decide_room} was parked on "
            "gated_echo by the first worker (decide is only legal from awaiting_approval, "
            "and tn-2 leaves its own room running). decide allow derives the resume turn "
            "id and runs resolveApproval once. The turn.failed message equals the fixed "
            "text; turnStatus is failed and errorCode is state_unreadable. The parked "
            "blob is not rewritten."
        )
    return {"steps": steps, "expected": expected, "actual": actual}


async def case_e_sk_2(stack: Stack) -> dict:
    body = await _unreadable_case(
        stack,
        "e-sk-2",
        {FLAG_VAR: "1"},
        f"{FLAG_VAR}=1 and no key (plaintext state, local development)",
        {KEY_VAR: KEY_A},
        f"key A and {FLAG_VAR} unset (production)",
        "plain:",
    )
    return {
        "id": "E-SK-2",
        "title": "A plain: blob read in production fails the turn once with state_unreadable",
        **body,
    }


async def case_e_sk_3(stack: Stack) -> dict:
    body = await _unreadable_case(
        stack,
        "e-sk-3",
        {KEY_VAR: KEY_A},
        f"key A and {FLAG_VAR} unset (production)",
        {KEY_VAR: KEY_B},
        f"key B and {FLAG_VAR} unset (production)",
        "fernet:",
        also_decide=True,
    )
    return {
        "id": "E-SK-3",
        "title": "A blob written with key A and read with key B fails runTurn and decide once",
        **body,
    }


async def case_e_sk_4(stack: Stack) -> dict:
    h = stack.harness
    room = "e-sk-4"
    production = {KEY_VAR: KEY_A}
    fernet = Fernet(KEY_A.encode("ascii"))
    blobs: list[bytes] = []
    workers: list[Worker] = []

    async def phase(name: str, drive):  # type: ignore[no-untyped-def]
        worker = await stack.running(f"{room}-{name}", production)
        workers.append(worker)
        try:
            return await drive()
        finally:
            worker.stop()

    handle: WorkflowHandle | None = None

    async def first() -> dict:
        nonlocal handle
        handle = await h.open_room(room, "mock")
        return await handle.execute_update(
            "runTurn", {"turnId": "tn-1", "message": "echo:" + CONTEXT_MARKER}
        )

    parked = await phase("1", first)
    assert handle is not None
    blobs.append(await stack.blob(handle))
    approval = parked.get("approval") or {}
    decided = await phase(
        "2",
        lambda: handle.execute_update(
            "decide",
            {
                "decision": "allow",
                "approvalRequestId": approval.get("approvalRequestId", ""),
            },
            result_type=DecideOutcome,
        ),
    )
    blobs.append(await stack.blob(handle))
    third = await phase(
        "3", lambda: handle.execute_update("runTurn", {"turnId": "tn-3", "message": "hello"})
    )
    blobs.append(await stack.blob(handle))
    events = h.recorder.room(room)
    results = [body for body in events if body["type"] == "tool.result"]

    def opened(blob: bytes) -> str:
        return fernet.decrypt(blob.removeprefix(b"fernet:")).decode("utf-8")

    return {
        "id": "E-SK-4",
        "title": "Blobs are fernet:; the conversation continues across worker restarts",
        "steps": [
            f"Every worker runs with key A and {FLAG_VAR} unset (production).",
            (f"Worker 1: open room {room}; runTurn tn-1 'echo:{CONTEXT_MARKER}' parks on "
             "gated_echo. Stop the worker; read the stored blob."),
            ("Worker 2 (new process): decide allow. The workflow derives the resume turn id; "
             "gated_echo runs from the parked call in the saved state. Stop the worker; "
             "read the stored blob."),
            ("Worker 3 (new process): runTurn tn-3 'hello'. The mock model answers 'done' "
             "only when a tool result is in the saved context (a fresh context answers "
             "'hello', as tn-1 in E-SK-2 shows). Read the stored blob."),
            ("Each blob must start with 'fernet:', must not contain the marker in clear, and "
             "must decrypt with key A to state that contains it."),
        ],
        "expected": {
            "tn1": {"status": "needs_approval", "toolName": "gated_echo"},
            "tn2": {"status": "completed"},
            "tn2ToolResult": {"toolName": "gated_echo", "text": "echo:" + CONTEXT_MARKER},
            "tn3": {"status": "completed", "texts": ["done"]},
            "blobPrefixes": ["fernet:", "fernet:", "fernet:"],
            "markerInStoredBytes": [False, False, False],
            "markerInDecryptedState": [True, True, True],
            "turnFailedEvents": 0,
            "workerLogs": [{"tracebacks": 0, "unreadableLogLines": 0, "keyFragments": NO_LEAKS}]
            * 3,
        },
        "actual": {
            "tn1": {
                "status": parked.get("status"),
                "toolName": approval.get("toolName"),
            },
            "tn2": {"status": decided.turn_status},
            "tn2ToolResult": {
                "toolName": results[-1].get("toolName") if results else None,
                "text": results[-1].get("text") if results else None,
            },
            "tn3": {"status": third.get("status"), "texts": third.get("texts")},
            "blobPrefixes": [_prefix(blob) for blob in blobs],
            "markerInStoredBytes": [CONTEXT_MARKER.encode() in blob for blob in blobs],
            "markerInDecryptedState": [CONTEXT_MARKER in opened(blob) for blob in blobs],
            "turnFailedEvents": sum(body["type"] == "turn.failed" for body in events),
            "workerLogs": [_worker_log_facts(worker) for worker in workers],
        },
    }


# E-SK-5 rows: label, and whether control's abort is sent before the DELETE.
CLOSE_ROWS = [("abort via control, then DELETE", True), ("DELETE only", False)]


async def _close_row(stack: Stack, control: Control, index: int, abort_first: bool) -> tuple:
    client = stack.harness.client
    seeder = await stack.running(f"e-sk-5-{index}-seed", {KEY_VAR: KEY_A})
    try:
        created, room = await control.call(
            "POST", "/v1/rooms", {"kind": "solo", "permissionPreset": "workspace-write"}
        )
        room_id = room["id"]
        posted, message = await control.call(
            "POST", f"/v1/rooms/{room_id}/messages", {"message": "hello"}
        )
        handle = client.get_workflow_handle_for(RoomWorkflow.run, f"room:{room_id}")
        opened = await handle.query(RoomWorkflow.snapshot)
    finally:
        seeder.stop()
    before = await stack.blob_of(opened.session_id)
    reader = await stack.running(f"e-sk-5-{index}-read", {KEY_VAR: KEY_B})
    try:
        aborted: dict[str, object] = {}
        if abort_first:
            status, body = await control.call("POST", f"/v1/rooms/{room_id}/abort")
            await asyncio.wait_for(handle.result(), CASE_TIMEOUT_S)
            aborted = {"abort": status, "abortBody": body}
        deleted, _ = await control.call("DELETE", f"/v1/rooms/{room_id}")
        final = await asyncio.wait_for(handle.result(), CASE_TIMEOUT_S)
        after_delete, _ = await control.call("GET", f"/v1/rooms/{room_id}")
        description = await handle.describe()
        runs = await _activity_runs(handle, "closeSession")
    finally:
        reader.stop()
    after = await stack.blob_of(opened.session_id)
    actual = {
        "create": created,
        "seedMessage": posted,
        "seedRoomState": (message.get("room") or {}).get("state"),
        "seededBlobPrefix": _prefix(before),
        **aborted,
        "delete": deleted,
        "getAfterDelete": after_delete,
        "workflowStatus": description.status.name if description.status else None,
        "roomStatus": final.status,
        "stateVersion": {"beforeRestart": opened.state_version, "final": final.state_version},
        "temporalCloseSession": runs,
        "blobUnchanged": bool(before) and before == after,
        "readerLog": _worker_log_facts(reader),
    }
    expected = {
        "create": 200,
        "seedMessage": 200,
        "seedRoomState": "running",
        "seededBlobPrefix": "fernet:",
        **({"abort": 200, "abortBody": {"aborted": True}} if abort_first else {}),
        "delete": 204,
        "getAfterDelete": 404,
        "workflowStatus": "COMPLETED",
        "roomStatus": "closed",
        "stateVersion": {"beforeRestart": 2, "final": 2},
        "temporalCloseSession": _exactly_once("closeSession"),
        "blobUnchanged": True,
        "readerLog": {"tracebacks": 0, "unreadableLogLines": 1, "keyFragments": NO_LEAKS},
    }
    return expected, actual


async def case_e_sk_5(stack: Stack) -> dict:
    row: dict[str, object] = {
        "id": "E-SK-5",
        "title": "Abort and control DELETE close a room whose state is unreadable",
        "steps": [
            ("orbit-control (in-memory storage, local dev principal) drives rooms through "
             f"RoomWorkflow on task queue {QUEUE}; DELETE sends X-Orbit-Request: 1 and an "
             "allowed Origin."),
            ("Per row: worker with key A; POST /v1/rooms, POST /v1/rooms/{id}/messages "
             "'hello'. Stop it; the stored blob starts with 'fernet:'."),
            "Start a worker with key B, so the room's state is unreadable.",
            ("Row 'abort via control, then DELETE': POST /v1/rooms/{id}/abort (the abort "
             "signal) and wait for the workflow result; then DELETE /v1/rooms/{id}, whose "
             "own abort signal finds the workflow already closed. Row 'DELETE only': "
             "DELETE /v1/rooms/{id} is the abort."),
            ("Read the DELETE status, GET /v1/rooms/{id} afterwards, the workflow result "
             "and status, every closeSession in its Temporal history, the stored blob, "
             "and the worker log."),
        ],
    }
    control = stack.control
    if control is None:
        return {
            **row,
            "expected": {"controlBinary": "provided with --control-bin"},
            "actual": {"controlBinary": "not provided; the control DELETE cannot run"},
        }
    expected: dict[str, object] = {}
    actual: dict[str, object] = {}
    for index, (label, abort_first) in enumerate(CLOSE_ROWS, start=1):
        expected[label], actual[label] = await _close_row(stack, control, index, abort_first)
    return {**row, "expected": expected, "actual": actual}


CASES = [case_e_sk_1, case_e_sk_2, case_e_sk_3, case_e_sk_4, case_e_sk_5]


def _dsn(url: str) -> str:
    return f"{url}{'&' if '?' in url else '?'}search_path={SCHEMA}"


async def _reset_schema(url: str) -> str:
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
        await conn.execute(f"CREATE SCHEMA {SCHEMA}")
        version = await conn.fetchval("SHOW server_version")
    finally:
        await conn.close()
    return str(version).split()[0]


def _control_revision(binary: Path) -> str:
    """The commit Go stamped into the control binary (``go version -m``)."""

    try:
        info = subprocess.run(
            ["go", "version", "-m", str(binary)], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    stamps = dict(
        line.split()[1].split("=", 1)
        for line in info.splitlines()
        if line.strip().startswith("build") and "=" in line
    )
    revision = stamps.get("vcs.revision", "unknown")
    return revision + (" (modified)" if stamps.get("vcs.modified") == "true" else "")


def _start_control(binary: Path, base: dict[str, str], log: Path) -> Control:
    port = a1._free_port()
    env = {
        **base,
        "PORT": str(port),
        "ORBIT_ALLOWED_ORIGINS": CONTROL_ORIGIN,
        # Its audit trail; kept with the logs so it is scanned too.
        "ORBIT_DATA_DIR": str(log.parent / "orbit-control-data"),
        # Unused: TEMPORAL_ADDRESS routes every room call through RoomWorkflow.
        "ORBIT_WORKER_URL": "http://127.0.0.1:9",
    }
    with log.open("w", encoding="utf-8") as out:
        process = subprocess.Popen(
            [str(binary)], env=env, stdout=out, stderr=subprocess.STDOUT
        )
    return Control(process, port)


def _head_sha() -> str:
    """The checked-out commit. There is deliberately no override."""

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


async def run(out: Path, logs: Path, control_bin: Path | None) -> int:
    url = os.environ.get(POSTGRES_VAR, "")
    if not url:
        print(f"{POSTGRES_VAR} is not set", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    postgres_version = await _reset_schema(url)
    dsn = _dsn(url)
    recorder = a1.Recorder()
    app = web.Application()
    app.router.add_post("/internal/events", recorder.ingest)
    runner = web.AppRunner(app)
    await runner.setup()
    port = a1._free_port()
    await web.TCPSite(runner, "127.0.0.1", port).start()

    rows: list[dict] = []
    async with await WorkflowEnvironment.start_local(
        data_converter=pydantic_data_converter
    ) as temporal:
        info = await temporal.client.workflow_service.get_system_info(GetSystemInfoRequest())
        base = {key: value for key, value in os.environ.items() if not key.startswith("ORBIT_")}
        base = {key: value for key, value in base.items() if key.upper() != FLAG_VAR}
        base.update(
            {
                "TEMPORAL_ADDRESS": temporal.client.service_client.config.target_host,
                "TEMPORAL_NAMESPACE": temporal.client.namespace,
                "TEMPORAL_TASK_QUEUE": QUEUE,
            }
        )
        orch = a1._start("orbit_orch.main", base, logs, "orbit-orch")
        worker_env = {
            **base,
            "ORBIT_EVENT_INGEST_URL": f"http://127.0.0.1:{port}/internal/events",
            "ORBIT_INTERNAL_TOKEN": a1.INGEST_TOKEN,
            "ORBIT_WORKER_BIND": "127.0.0.1",
            "ORBIT_MODEL_MODE": "mock",
            "ORBIT_STATE_STORE_URL": dsn,
        }
        control = None
        if control_bin is not None:
            control = _start_control(control_bin, base, logs / "orbit-control.log")
        harness = a1.Harness(temporal.client, recorder, {}, logs)
        stack = Stack(harness, worker_env, logs, dsn, control)
        try:
            if control is not None:
                await control.wait_healthy()
            for case in CASES:
                try:
                    row = await asyncio.wait_for(case(stack), CASE_TIMEOUT_S)
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
            for process in [orch] + ([control.process] if control is not None else []):
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
    await runner.cleanup()

    components = a1._versions(info.server_version)
    components["postgres"] = postgres_version
    for name in ("cryptography", "asyncpg"):
        components[name] = importlib.metadata.version(name)
    components["orbit-control"] = (
        _control_revision(control_bin) if control_bin is not None else "not run"
    )
    failed = [row["id"] for row in rows if not row["pass"]]
    report = {
        "suite": "e2e-state-key",
        "commit": _head_sha(),
        "components": components,
        "setup": [
            "Fresh local Temporal dev server (in-memory).",
            f"One orbit-orch subprocess on task queue {QUEUE}.",
            ("orbit-worker subprocesses on the same queue, started and stopped per case, one "
             "at a time; ORBIT_MODEL_MODE=mock."),
            (f"Postgres state store ({POSTGRES_VAR}) in schema {SCHEMA}, dropped and created "
             "at the start of the run."),
            ("Test keys are derived from fixed labels (A, B, malformed); the report carries "
             "none of their characters."),
            ("Workers post events to the e2e_a1_events.py recording ingest stub, which "
             "requires the internal bearer token."),
            "Rooms are driven with the runTurn and decide Updates, as control drives them.",
            ("E-SK-5 adds orbit-control (the commit under components) with in-memory "
             "storage and the local dev principal, on the same Temporal queue."),
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


_SCAN_PATTERNS = {
    **a1._SCAN_PATTERNS,
    "fernet-key": re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{43}="),
}


def _files(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        found.extend(sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path])
    return found


def scan(paths: list[Path], out: Path | None) -> int:
    findings = []
    files = _files(paths)
    for path in files:
        text = path.read_text(encoding="utf-8")
        for value in PLANTED:
            if value in text:
                findings.append(
                    {"file": str(path), "rule": "planted", "match": a1._fingerprint(value)}
                )
        for name, secret in KEYS.items():
            hits = [fragment for fragment in _fragments(secret) if fragment in text]
            if hits:
                findings.append(
                    {"file": str(path), "rule": f"key-fragment:{name}", "count": len(hits)}
                )
        for rule, pattern in _SCAN_PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append(
                    {"file": str(path), "rule": rule, "match": a1._fingerprint(match.group(0))}
                )
    result = {
        "scanned": [str(path) for path in files],
        "rules": ["planted", *(f"key-fragment:{name}" for name in KEYS), *_SCAN_PATTERNS],
        "findings": findings,
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"secret scan: {len(findings)} finding(s) in {len(files)} file(s)")
    return 1 if findings else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--out", type=Path, default=Path("artifacts/e2e-state-key.json"))
    run_parser.add_argument("--logs", type=Path, default=Path("e2e-logs/state-key"))
    run_parser.add_argument(
        "--control-bin", type=Path, help="orbit-control binary for E-SK-5 (required to pass)"
    )
    scan_parser = sub.add_parser("scan")
    scan_parser.add_argument("paths", type=Path, nargs="+", help="files or directories")
    scan_parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "scan":
        return scan(args.paths, args.out)
    return asyncio.run(run(args.out, args.logs, args.control_bin))


if __name__ == "__main__":
    sys.exit(main())
