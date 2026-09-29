import pytest
from orbit_worker.checkpoint_activities import commit_checkpoints
from orbit_worker.task_activities import set_task_store
from temporalio.exceptions import ApplicationError

TENANT = "tenant-a"


@pytest.mark.parametrize(
    "payload",
    [
        {"attempt_id": "att_1"},
        {"tenant_id": "", "attempt_id": "att_1"},
        {"tenant_id": TENANT},
        {"tenant_id": TENANT, "attempt_id": "att_1", "unexpected": 1},
    ],
)
async def test_an_incomplete_commit_fails_without_retries(payload) -> None:
    with pytest.raises(ApplicationError) as failed:
        await commit_checkpoints(payload)
    assert failed.value.non_retryable
    assert failed.value.type == "invalid_activity_input"


async def test_the_activity_commits_through_the_store(clean_db, task_store) -> None:
    set_task_store(task_store)
    ref = await task_store.put_checkpoint(
        tenant_id=TENANT,
        task_id="task-1",
        node_id="n_1",
        attempt_id="att_1",
        seq=1,
        kind="agent_state",
        payload=b"state",
    )
    payload = {"tenant_id": TENANT, "attempt_id": "att_1", "checkpoint_ref": ref}
    assert await commit_checkpoints(payload) == {"ok": True, "committed": 1}
    assert await commit_checkpoints(payload) == {"ok": True, "committed": 0}
