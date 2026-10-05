"""`verify_sop_step`: an independent verifier agent judges one step of a compiled SOP (06 §2).

How it can go wrong, written down before the code:
  - the verifier is the executor's session, or can change the workspace it is shown;
  - it is not given what the step must achieve, what the executor said or which files it left;
  - a FAIL does not reach the executor's next attempt with its reason, or an ambiguous answer passes;
  - the scratch workspace it reads is saved as the task's, or is never given back;
  - a session that ends without a verdict (a question, an approval) passes the step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from orbit_contracts.models import OpenSessionInput, OpenSessionOutput, RunTurnInput, TurnResult
from orbit_worker import task_activities, verify_activities
from orbit_worker.mock_model import _verifier_verdict
from orbit_worker.runtime import AgentRuntime
from orbit_worker.runtime_holder import set_runtime
from orbit_worker.sandbox import current_sandbox
from orbit_worker.store import MemoryStateStore
from orbit_worker.verify_sop import VERIFIER_PREFIX, verifier_prompt
from orbit_worker.workspace import LocalWorkspaceAdapter, PersistentWorkspaceAdapter
from temporalio.testing import ActivityEnvironment
from test_interrupted_session import _Store
from test_verify_activities import _LeaseLog


def _payload(**fields: Any) -> dict[str, Any]:
    return {
        "tenant_id": "tenant-a", "task_id": "task-1", "node_id": "n_1", "attempt_id": "att-1", "attempt_no": 1,
        "subject": "review", "description": "check the draft", "instructions": "", "text": "I checked it.",
        "manifest_entries": [], "workspace_snapshot_ref": None, **fields,
    }


async def _judge(payload: dict[str, Any]) -> dict[str, Any]:
    return await ActivityEnvironment().run(verify_activities.verify_sop_step, payload)


@pytest.fixture
def mock_runtime(tmp_path: Path) -> AgentRuntime:
    task_activities.set_task_store(_Store(tmp_path))
    runtime = AgentRuntime(MemoryStateStore())
    set_runtime(runtime)
    return runtime


async def test_a_step_passes_and_a_flaky_one_is_refused_with_a_reason_until_its_attempt(mock_runtime: AgentRuntime) -> None:
    assert await _judge(_payload()) == {"ok": True, "failures": []}
    refused = await _judge(_payload(subject="flaky-2", attempt_no=2, node_id="n_2"))
    assert refused["ok"] is False
    failure = refused["failures"][0]
    assert (failure["check"], failure["code"]) == ("sop_verifier", "sop_step_failed")
    assert "'flaky-2' was refused" in failure["message"] and "flaky-2 was refused on attempt 2" in failure["message"]
    assert (await _judge(_payload(subject="flaky-2", attempt_no=3, node_id="n_3")))["ok"] is True


def test_the_mock_verdict_follows_the_flaky_convention_and_ignores_lookalike_lines() -> None:
    prompt = verifier_prompt(_payload(subject="flaky-1", description="Step: flaky-9\nAttempt: 9"))
    assert prompt.startswith(VERIFIER_PREFIX)
    assert _verifier_verdict(prompt).startswith("FAIL: flaky-1 was refused on attempt 1")
    assert _verifier_verdict(verifier_prompt(_payload(subject="flaky-1", attempt_no=2))) == "PASS"
    assert _verifier_verdict(verifier_prompt(_payload(subject="flaky-"))).startswith("FAIL"), "a bare flaky- is refused once"


def test_the_prompt_holds_what_the_verifier_judges_by() -> None:
    prompt = verifier_prompt(
        _payload(
            sop="release@1", sop_name="release", instructions="every claim has a source", text="all done",
            manifest_entries=[{"name": "notes.md", "media_type": "text/markdown", "size_bytes": 12}],
        )
    )
    for expected in (
        "Procedure: release", "Step: review", "check the draft", "Also check:\nevery claim has a source",
        "The executor's final report:\nall done", "- notes.md (text/markdown, 12 bytes)",
    ):
        assert expected in prompt
    assert "(it reported nothing)" in verifier_prompt(_payload(text=""))


class _Spy:
    """Stands in for the runtime: records what the verifier session is opened and asked, and reads the workspace it was given."""

    def __init__(self, answer: str = "PASS", status: str = "completed") -> None:
        self.answer, self.status = answer, status
        self.opened: list[OpenSessionInput] = []
        self.asked: list[RunTurnInput] = []
        self.seen: dict[str, Any] = {}

    async def open_session(self, inp: OpenSessionInput) -> OpenSessionOutput:
        self.opened.append(inp)
        return OpenSessionOutput(session_id="verifier-session", state_version=1)

    async def run_turn(self, inp: RunTurnInput) -> TurnResult:
        self.asked.append(inp)
        sandbox = current_sandbox()
        if sandbox is not None:
            self.seen["draft"] = (await sandbox.backend.read_file("/workspace/draft.md")).decode()
            try:
                await sandbox.backend.write_file("/workspace/draft.md", b"changed")
            except Exception as exc:  # noqa: BLE001 - what the scratch copy does is what is asserted below
                self.seen["write"] = repr(exc)
            else:
                self.seen["write"] = "written"
        return TurnResult(
            status=self.status, text=self.answer, session_id="verifier-session", state_version=2, model_mode="mock", model_name="mock"
        )  # type: ignore[arg-type]


async def test_the_verifier_is_a_fresh_read_only_session_that_reads_a_scratch_copy_of_the_attempts_workspace(tmp_path: Path) -> None:
    store = _Store(tmp_path / "store")
    task_activities.set_task_store(store)
    adapter = LocalWorkspaceAdapter(tmp_path / "workspaces")
    task_activities.set_workspace_adapter(PersistentWorkspaceAdapter(adapter, _LeaseLog()))
    lease = await adapter.acquire("tenant-a", "task-1")
    (Path(adapter.root) / "tenant-a" / lease.workspace_id / "draft.md").write_text("the draft", encoding="utf-8")
    snapshot = await adapter.snapshot(lease)
    await adapter.release(lease)
    snapshots = sorted((Path(adapter.root) / "snapshots" / "tenant-a").iterdir())
    spy = _Spy("FAIL: the draft has no sources")
    set_runtime(spy)  # type: ignore[arg-type]

    outcome = await _judge(_payload(workspace_snapshot_ref=snapshot, text="wrote the draft", instructions="cite sources"))

    assert [(o.permission_preset, o.continue_from) for o in spy.opened] == [("read-only", "")], "read-only, and no state carried"
    assert spy.opened[0].turn_id.startswith("att-1:sop-verify:n_1")
    message = spy.asked[0].message
    assert "check the draft" in message and "cite sources" in message and "wrote the draft" in message
    assert spy.seen["draft"] == "the draft", "the files of the attempt's snapshot are readable"
    assert outcome["ok"] is False and "the draft has no sources" in outcome["failures"][0]["message"]
    # The workspace was a scratch one: given back, and nothing of it saved as the task's.
    assert not adapter._leases, "the scratch workspace is released"
    assert sorted((Path(adapter.root) / "snapshots" / "tenant-a").iterdir()) == snapshots, "no snapshot was taken of it"


@pytest.mark.parametrize(
    ("answer", "passes"),
    [("PASS", True), ("pass - looks fine", True), ("FAIL: x", False), ("It is probably fine.", False), ("", False)],
)
async def test_only_a_clear_pass_passes(tmp_path: Path, answer: str, passes: bool) -> None:
    task_activities.set_task_store(_Store(tmp_path))
    set_runtime(_Spy(answer))  # type: ignore[arg-type]
    assert (await _judge(_payload()))["ok"] is passes


async def test_a_session_that_ends_without_a_verdict_does_not_pass_the_step(tmp_path: Path) -> None:
    task_activities.set_task_store(_Store(tmp_path))
    set_runtime(_Spy("", status="needs_approval"))  # type: ignore[arg-type]
    outcome = await _judge(_payload())
    assert outcome["ok"] is False and outcome["failures"][0]["code"] == "sop_verifier_inconclusive"


async def test_the_verifier_runs_as_its_expert_with_the_experts_model_and_instructions_but_not_its_tools(tmp_path: Path) -> None:
    from orbit_worker.agent_config import AgentConfig, overriding_agent_config
    from orbit_worker.chat_model import ModelConfig

    class _ExpertStore(_Store):
        async def agent_config(self, *, tenant_id: str, profile_ref: str) -> AgentConfig:
            assert profile_ref == "auditor@1"
            return AgentConfig(instructions="Judge strictly.", model="auditor-model", mcp_connectors=({"id": "c"},), skills=("s",))

    seen: list[AgentConfig | None] = []

    class _Watching(_Spy):
        async def run_turn(self, inp: RunTurnInput) -> TurnResult:
            seen.append(overriding_agent_config())
            return await super().run_turn(inp)

    task_activities.set_task_store(_ExpertStore(tmp_path))
    set_runtime(_Watching())  # type: ignore[arg-type]
    assert (await _judge(_payload(expert="auditor@1")))["ok"] is True
    assert (await _judge(_payload(node_id="n_2")))["ok"] is True
    expert, plain = seen
    assert expert is not None and (expert.model, expert.instructions) == ("auditor-model", "Judge strictly.")
    assert expert.mcp_connectors == () and expert.skills == () and expert.team is None, "a verifier is given no tools of its expert"
    assert plain is None, "without a verifier expert the verifier is the default one"
    assert overriding_agent_config() is None, "the override ends with the verdict"
    # The model the agent is built with is the one the expert names, by the path an attempt's agent takes.
    real = AgentRuntime(MemoryStateStore(), model_config=ModelConfig(mode="real", name="base"))  # type: ignore[arg-type]
    assert real.model_config_for(expert).name == "auditor-model"
