"""End-to-end check of the A1 events with real orbit-orch and orbit-worker processes.

Starts a fresh local Temporal dev server, runs ``orbit_orch.main`` and
``orbit_worker.main`` as subprocesses on the mock model, and points
``ORBIT_EVENT_INGEST_URL`` at an in-process capture endpoint. Each case gets
its own RoomWorkflow, driven with the ``runTurn`` and ``decide`` Updates the
way control drives it. Assertions read the JSON bodies the worker posted.

Writes one row per case (case, expected, actual, pass) to ``--out`` and
exits 1 when any case fails. Expected values are literals, not computed by
the redactor under test. The output has no timestamps or random ids, so a
rerun on the same commit writes the same file.

    uv run python scripts/e2e_a1_events.py --out artifacts/e2e-a1-events.json
"""

import argparse
import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import web
from orbit_contracts.models import RoomCommand, RoomWorkflowInput
from orbit_orch.workflows import RoomWorkflow
from orbit_worker.mock_model import CHUNK_SEPARATOR
from temporalio.client import Client, WorkflowHandle
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment

QUEUE = "orbit-e2e"
TOKEN = "e2e-internal-token"
REDACTED = "[REDACTED]"
TOOL_RESULT_BYTES = 4096
CASE_TIMEOUT_S = 60

FILLER = "word " * 45
# Chinese without spaces; each is longer than one 200-character delta batch.
CJK_FILLER = "这是一段没有任何空格的中文说明文字，" * 12
CJK_PARAGRAPH = "模型正在逐字输出一段没有任何空格的中文回答，" * 10


@dataclass
class StreamCase:
    name: str
    chunks: list[str]
    expected_text: str
    secrets: list[str] = field(default_factory=list)
    # Exact deltas, when the batching itself is what the case checks.
    expected_deltas: list[str] | None = None


@dataclass
class ToolCase:
    name: str
    echo: str
    expected_text: str
    truncated: bool
    secrets: list[str] = field(default_factory=list)


STREAM_CASES = [
    StreamCase(
        "stream-en-bearer-prefix",
        [FILLER + "send it with Bearer ", "q7wz19kx then stop"],
        FILLER + f"send it with Bearer {REDACTED} then stop",
        ["q7wz19kx"],
    ),
    StreamCase(
        "stream-en-key-prefix",
        [FILLER + "config api_key= ", "hunter2 then stop"],
        FILLER + f"config api_key= {REDACTED} then stop",
        ["hunter2"],
    ),
    StreamCase(
        "stream-en-split-token",
        [FILLER + "the key is sk-live-", "4f9a2b77c then stop"],
        FILLER + f"the key is {REDACTED} then stop",
        ["sk-live-4f9a2b77c", "4f9a2b77c"],
    ),
    StreamCase(
        "stream-en-split-random-token",
        [FILLER + "copy Zk8Qw3Rt7Yp2Lm9X", "c4Vb6Nj1Hg5Fd0Sa then stop"],
        FILLER + f"copy {REDACTED} then stop",
        ["Zk8Qw3Rt7Yp2Lm9X", "c4Vb6Nj1Hg5Fd0Sa"],
    ),
    StreamCase(
        "stream-cjk-no-spaces",
        [CJK_PARAGRAPH, CJK_PARAGRAPH, CJK_PARAGRAPH],
        CJK_PARAGRAPH * 3,
        expected_deltas=[CJK_PARAGRAPH, CJK_PARAGRAPH, CJK_PARAGRAPH],
    ),
    StreamCase(
        "stream-cjk-split-token",
        [CJK_FILLER + "密钥是 sk-live-", "4f9a2b77c，然后停"],
        CJK_FILLER + f"密钥是 {REDACTED}，然后停",
        ["sk-live-4f9a2b77c", "4f9a2b77c"],
    ),
    StreamCase(
        "stream-cjk-key-prefix",
        [CJK_FILLER + "配置里 api_key= ", "hunter2，然后继续"],
        CJK_FILLER + f"配置里 api_key= {REDACTED}，然后继续",
        ["hunter2"],
    ),
    StreamCase(
        "stream-cjk-split-random-token",
        [CJK_FILLER + "复制Zk8Qw3Rt7Yp2Lm9X", "c4Vb6Nj1Hg5Fd0Sa，结束"],
        CJK_FILLER + f"复制{REDACTED}，结束",
        ["Zk8Qw3Rt7Yp2Lm9X", "c4Vb6Nj1Hg5Fd0Sa"],
    ),
]

_ECHO_REDACTED_PREFIX = f"echo:token={REDACTED} "
TOOL_CASES = [
    ToolCase("tool-result-short", "short", "echo:short", truncated=False),
    ToolCase(
        # 3-byte characters: 1363 of them fit after "echo:" (5 + 4089 = 4094 bytes).
        "tool-result-cjk-truncated",
        "中" * 3000,
        "echo:" + "中" * 1363,
        truncated=True,
    ),
    ToolCase(
        "tool-result-redacted-then-truncated",
        "token=abc123 " + "x" * 5000,
        _ECHO_REDACTED_PREFIX + "x" * (TOOL_RESULT_BYTES - len(_ECHO_REDACTED_PREFIX)),
        truncated=True,
        secrets=["abc123"],
    ),
]


class Capture:
    """The control ingest endpoint, reduced to recording what the worker posts."""

    def __init__(self) -> None:
        self.bodies: list[tuple[str, dict]] = []
        self.unauthorized = 0

    async def handle(self, request: web.Request) -> web.Response:
        if request.headers.get("Authorization") != f"Bearer {TOKEN}":
            self.unauthorized += 1
            return web.Response(status=401)
        raw = await request.text()
        self.bodies.append((raw, json.loads(raw)))
        return web.Response(status=202)

    def for_room(self, room_id: str) -> list[tuple[str, dict]]:
        return [(raw, body) for raw, body in self.bodies if body.get("roomId") == room_id]


def _summary(text: str) -> dict[str, object]:
    encoded = text.encode("utf-8")
    return {
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest()[:16],
        "head": text[:16],
        "tail": text[-16:],
    }


def _found(secrets: list[str], events: list[tuple[str, dict]]) -> list[str]:
    return [secret for secret in secrets if any(secret in raw for raw, _ in events)]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _open_room(client: Client, room_id: str) -> WorkflowHandle:
    handle = await client.start_workflow(
        RoomWorkflow.run,
        RoomWorkflowInput(room_id=room_id),
        id=f"room:{room_id}",
        task_queue=QUEUE,
    )
    await handle.signal(RoomWorkflow.command, RoomCommand(kind="open", turn_id="t-open"))
    for _ in range(CASE_TIMEOUT_S * 10):
        snapshot = await handle.query(RoomWorkflow.snapshot)
        if snapshot.status == "running":
            return handle
        await asyncio.sleep(0.1)
    raise TimeoutError(f"room {room_id} did not open")


async def _stream_case(client: Client, capture: Capture, case: StreamCase) -> dict:
    handle = await _open_room(client, case.name)
    message = "stream:" + CHUNK_SEPARATOR.join(case.chunks)
    await handle.execute_update("runTurn", {"turnId": "tn-1", "message": message})
    events = capture.for_room(case.name)
    deltas = [body for _, body in events if body["type"] == "assistant.delta"]
    messages = [body["text"] for _, body in events if body["type"] == "assistant.message"]
    expected: dict[str, object] = {
        "streamed": case.expected_text,
        "assistantMessage": case.expected_text,
        "atLeastTwoDeltas": True,
        "deltaSeq": list(range(len(case.expected_deltas or deltas))),
        "deltaAgentIds": ["main"],
        "deltaTurnIds": ["tn-1"],
        "secretsFound": [],
    }
    actual: dict[str, object] = {
        "streamed": "".join(body["delta"] for body in deltas),
        "assistantMessage": messages[-1] if messages else None,
        "atLeastTwoDeltas": len(deltas) >= 2,
        "deltaSeq": [body["seq"] for body in deltas],
        "deltaAgentIds": sorted({body["agentId"] for body in deltas}),
        "deltaTurnIds": sorted({body["turnId"] for body in deltas}),
        "secretsFound": _found(case.secrets, events),
    }
    if case.expected_deltas is not None:
        expected["deltas"] = case.expected_deltas
        actual["deltas"] = [body["delta"] for body in deltas]
    return {"case": case.name, "expected": expected, "actual": actual}


async def _tool_case(client: Client, capture: Capture, case: ToolCase) -> dict:
    handle = await _open_room(client, case.name)
    parked = await handle.execute_update(
        "runTurn", {"turnId": "tn-1", "message": "echo:" + case.echo}
    )
    approval = parked["approval"] or {}
    await handle.execute_update(
        "decide",
        {
            "decision": "allow",
            "approvalRequestId": approval.get("approvalRequestId", ""),
            "resumeTurnId": "tn-2",
        },
    )
    events = capture.for_room(case.name)
    results = [body for _, body in events if body["type"] == "tool.result"]
    calls = [body for _, body in events if body["type"] == "tool.call"]
    result = results[-1] if results else {}
    expected = {
        "parkedStatus": "needs_approval",
        "toolResultCount": 1,
        "toolName": "gated_echo",
        "toolState": "success",
        "truncated": case.truncated,
        "text": _summary(case.expected_text),
        "argsPreviewAtMost256": True,
        "secretsFound": [],
    }
    actual = {
        "parkedStatus": parked["status"],
        "toolResultCount": len(results),
        "toolName": result.get("toolName"),
        "toolState": result.get("toolState"),
        "truncated": result.get("truncated", False),
        "text": _summary(result.get("text", "")),
        "argsPreviewAtMost256": bool(calls) and all(
            len(body["argsPreview"]) <= 256 for body in calls
        ),
        "secretsFound": _found(case.secrets, events),
    }
    return {"case": case.name, "expected": expected, "actual": actual}


async def _run_case(run: Callable[[], object], name: str) -> dict:
    try:
        row = await asyncio.wait_for(run(), CASE_TIMEOUT_S)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        return {"case": name, "expected": "case completes", "actual": repr(exc), "pass": False}
    row["pass"] = row["expected"] == row["actual"]
    return row


def _start(module: str, env: dict[str, str], log: Path) -> subprocess.Popen:
    with log.open("w", encoding="utf-8") as out:
        return subprocess.Popen(
            [sys.executable, "-m", module], env=env, stdout=out, stderr=subprocess.STDOUT
        )


async def main(out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    capture = Capture()
    app = web.Application()
    app.router.add_post("/internal/events", capture.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    ingest_port = _free_port()
    await web.TCPSite(runner, "127.0.0.1", ingest_port).start()

    rows: list[dict] = []
    async with await WorkflowEnvironment.start_local(
        data_converter=pydantic_data_converter
    ) as temporal:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("ORBIT_MODEL_", "ORBIT_STATE_"))
        }
        env.update(
            {
                "TEMPORAL_ADDRESS": temporal.client.service_client.config.target_host,
                "TEMPORAL_NAMESPACE": temporal.client.namespace,
                "TEMPORAL_TASK_QUEUE": QUEUE,
                "ORBIT_EVENT_INGEST_URL": f"http://127.0.0.1:{ingest_port}/internal/events",
                "ORBIT_INTERNAL_TOKEN": TOKEN,
                "ORBIT_MODEL_MODE": "mock",
                "ORBIT_WORKER_BIND": "127.0.0.1",
                "ORBIT_WORKER_PORT": str(_free_port()),
            }
        )
        processes = [
            _start("orbit_orch.main", env, out.parent / "e2e-orbit-orch.log"),
            _start("orbit_worker.main", env, out.parent / "e2e-orbit-worker.log"),
        ]
        try:
            client = temporal.client
            for stream_case in STREAM_CASES:
                rows.append(
                    await _run_case(
                        lambda c=stream_case: _stream_case(client, capture, c), stream_case.name
                    )
                )
            for tool_case in TOOL_CASES:
                rows.append(
                    await _run_case(
                        lambda c=tool_case: _tool_case(client, capture, c), tool_case.name
                    )
                )
        finally:
            for process in processes:
                process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
    await runner.cleanup()

    rows.append(
        {
            "case": "ingest-auth",
            "expected": {"unauthorizedPosts": 0, "anyEventPosted": True},
            "actual": {
                "unauthorizedPosts": capture.unauthorized,
                "anyEventPosted": bool(capture.bodies),
            },
        }
    )
    rows[-1]["pass"] = rows[-1]["expected"] == rows[-1]["actual"]
    failed = [row["case"] for row in rows if not row["pass"]]
    report = {
        "suite": "e2e-a1-events",
        "model": "mock",
        "path": "RoomWorkflow runTurn/decide -> orbit-worker -> HTTP ingest",
        "cases": rows,
        "passed": len(rows) - len(failed),
        "failed": failed,
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{'PASS' if row['pass'] else 'FAIL'}  {row['case']}")
    print(f"{report['passed']}/{len(rows)} passed; report: {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("artifacts/e2e-a1-events.json"))
    sys.exit(asyncio.run(main(parser.parse_args().out)))
