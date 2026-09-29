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
    # A task with a checkpoint node, with `checkpoint-node-explicit-identity` (17 G9).
    "07-checkpoint-node.json": "task/tenant-a/golden-checkpoint",
    # The same, recorded before the patch: the old payload, without an identity (17 G9).
    "08-checkpoint-node-before-identity.json": "task/tenant-a/golden-checkpoint-old",
    # An AttemptWorkflow that parked on an approval and finished, with `commit-attempt-checkpoints` (17 G1).
    "06-attempt-commit-checkpoints.json": (
        "attempt/task_2A4N3XNAKXYV66VE23NMWGBD9C/n_52M7RQ2NZFWEZQR7JT87P30SAQ/1"
    ),
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
