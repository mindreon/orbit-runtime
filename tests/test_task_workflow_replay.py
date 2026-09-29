import gzip
from pathlib import Path

import pytest
from orbit_orch.sandbox import sandbox_runner
from orbit_orch.task_workflow import AttemptWorkflow, TaskWorkflow
from temporalio.client import WorkflowHistory
from temporalio.worker import Replayer

FIXTURES = Path(__file__).parent / "fixtures" / "workflow_histories"
CASES = {
    "01-complete.json": "task/tenant-a/golden-complete",
    "02-approval.json": "task/tenant-a/golden-approval",
    "03-interrupt.json": "task/tenant-a/golden-interrupt",
    "04-plan-change.json": "task/tenant-a/golden-plan",
    "05-continue-as-new.json.gz": "task/tenant-a/golden-can",
    "06-completion-verification.json": "task/tenant-a/golden-verify",
    # Recorded before the completion checks existed: a run that started then must replay on the patched code.
    "07-completion-before-verification.json": "task/tenant-a/golden-verify-legacy",
}


@pytest.mark.asyncio
@pytest.mark.parametrize(("filename", "workflow_id"), CASES.items())
async def test_golden_history_replays(filename: str, workflow_id: str) -> None:
    fixture = FIXTURES / filename
    payload = (
        gzip.decompress(fixture.read_bytes()).decode("utf-8")
        if fixture.suffix == ".gz"
        else fixture.read_text(encoding="utf-8")
    )
    history = WorkflowHistory.from_json(workflow_id, payload)
    replayer = Replayer(
        workflows=[TaskWorkflow, AttemptWorkflow],
        workflow_runner=sandbox_runner(),
    )
    result = await replayer.replay_workflow(history)
    assert result.replay_failure is None
