"""checkpoint_commit takes no defaults (17 G9): a payload with a field missing fails once, without retries."""

from __future__ import annotations

from typing import Any

import pytest
from orbit_worker.task_activities import checkpoint_commit, set_task_store
from temporalio.exceptions import ApplicationError

VALID: dict[str, Any] = {
    "tenant_id": "tenant-a",
    "task_id": "task_1",
    "node_id": "n_1",
    "attempt_id": "att_1",
    "seq": 0,
    "kind": "plan",
}


class RecordingStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def put_checkpoint(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "sha256:" + "0" * 64


@pytest.fixture
def store() -> RecordingStore:
    recording = RecordingStore()
    set_task_store(recording)  # type: ignore[arg-type]
    return recording


@pytest.mark.parametrize("missing", sorted(VALID))
async def test_a_missing_field_fails_without_retries_and_stores_nothing(store, missing) -> None:
    payload = {key: value for key, value in VALID.items() if key != missing}
    with pytest.raises(ApplicationError) as failed:
        await checkpoint_commit(payload)
    assert failed.value.non_retryable
    assert missing in str(failed.value)
    assert store.calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("tenant_id", ""), ("seq", -1), ("kind", "notes"), ("attempt_id", "")],
)
async def test_a_field_that_does_not_fit_fails_without_retries(store, field, value) -> None:
    with pytest.raises(ApplicationError) as failed:
        await checkpoint_commit({**VALID, field: value})
    assert failed.value.non_retryable and store.calls == []


async def test_an_unknown_field_is_refused(store) -> None:
    with pytest.raises(ApplicationError):
        await checkpoint_commit({**VALID, "attemptId": "att_2"})


async def test_a_complete_payload_is_stored_as_given(store) -> None:
    result = await checkpoint_commit(VALID)
    assert result == {"ok": True, "checkpoint_ref": "sha256:" + "0" * 64}
    call = store.calls[0]
    assert {key: call[key] for key in VALID} == VALID
