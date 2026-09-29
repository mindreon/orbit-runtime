import pytest
from orbit_orch.maintenance import RuntimeMaintenanceWorkflow
from orbit_orch.schedules import maintenance_schedule
from orbit_worker.maintenance import maintenance_tick, set_maintenance_store
from orbit_worker.task_store import TaskStore
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker


@pytest.mark.asyncio
async def test_maintenance_store_without_database_is_noop(tmp_path):
    store = TaskStore(url="", root=str(tmp_path))
    assert await store.maintenance("reap_leases") == 0


@pytest.mark.asyncio
async def test_maintenance_activity_dispatches_operation():
    class FakeStore:
        async def maintenance(self, operation, *, tenant_id):
            assert operation == "gc_checkpoints"
            assert tenant_id == "tenant-a"
            return 3

    set_maintenance_store(FakeStore())
    result = await maintenance_tick(
        {"operation": "gc_checkpoints", "tenant_id": "tenant-a"}
    )
    assert result == {
        "operation": "gc_checkpoints",
        "tenants": 1,
        "tenant_id": "tenant-a",
        "removed": 3,
    }


@pytest.mark.asyncio
async def test_maintenance_workflow_executes_io_activity():
    class FakeStore:
        async def maintenance(self, operation, *, tenant_id):
            assert operation == "gc_checkpoints"
            assert tenant_id == "default"
            return 7

    set_maintenance_store(FakeStore())
    async with await WorkflowEnvironment.start_time_skipping() as env, Worker(
        env.client, task_queue="orbit.orch", workflows=[RuntimeMaintenanceWorkflow]
    ), Worker(env.client, task_queue="orbit.io", activities=[maintenance_tick]):
        result = await env.client.execute_workflow(
            RuntimeMaintenanceWorkflow.run,
            {"operation": "gc_checkpoints", "tenant_id": "default", "task_queue": "orbit.io"},
            id="maintenance-test",
            task_queue="orbit.orch",
        )
    assert result["removed"] == 7


def test_maintenance_schedule_targets_io_queue():
    schedule = maintenance_schedule(
        "reap_leases",
        every=__import__("datetime").timedelta(minutes=5),
        io_task_queue="orbit.io",
        tenant_id="default",
    )
    assert schedule.spec.intervals[0].every.total_seconds() == 300
    assert schedule.state.note == "Orbit maintenance: reap_leases"
    assert schedule.action.args[0] == {
        "operation": "reap_leases",
        "task_queue": "orbit.io",
        "tenant_id": "default",
    }


def test_maintenance_schedule_covers_every_tenant_by_default():
    schedule = maintenance_schedule(
        "gc_checkpoints",
        every=__import__("datetime").timedelta(days=1),
        io_task_queue="orbit.io",
    )
    assert schedule.action.args[0] == {
        "operation": "gc_checkpoints",
        "task_queue": "orbit.io",
        "all_tenants": True,
    }
    assert schedule.action.id == "maintenance:all:gc_checkpoints"


class _TenantStore:
    """A store that lists tenants and can fail for some of them."""

    def __init__(self, tenants, failing=()):
        self.tenants = list(tenants)
        self.failing = set(failing)
        self.seen = []

    async def list_tenants(self):
        return list(self.tenants)

    async def maintenance(self, operation, *, tenant_id):
        self.seen.append((operation, tenant_id))
        if tenant_id in self.failing:
            raise RuntimeError("database says no")
        return 2


@pytest.mark.asyncio
async def test_all_tenants_runs_once_per_tenant_that_exists_now():
    store = _TenantStore(["tenant-a", "tenant-b"])
    set_maintenance_store(store)
    payload = {"operation": "gc_checkpoints", "all_tenants": True}
    first = await maintenance_tick(payload)
    store.tenants.append("tenant-new")  # created after the schedule was made
    second = await maintenance_tick(payload)
    assert (first["tenants"], first["removed"]) == (2, 4)
    assert (second["tenants"], second["removed"]) == (3, 6)
    assert store.seen[-1] == ("gc_checkpoints", "tenant-new")


@pytest.mark.asyncio
async def test_one_tenant_failing_does_not_stop_the_others():
    from orbit_worker.maintenance import MaintenanceError

    store = _TenantStore(["tenant-a", "tenant-b", "tenant-c"], failing={"tenant-b"})
    set_maintenance_store(store)
    with pytest.raises(MaintenanceError, match="tenant-b"):
        await maintenance_tick({"operation": "gc_checkpoints", "all_tenants": True})
    assert [tenant for _, tenant in store.seen] == ["tenant-a", "tenant-b", "tenant-c"]


@pytest.mark.parametrize(
    "payload",
    [
        {"operation": "gc_checkpoints"},
        {"operation": "gc_checkpoints", "tenant_id": "a", "all_tenants": True},
        {"operation": "gc_checkpoints", "tenant_id": ""},
        {"tenant_id": "tenant-a"},
        {"operation": "drop_everything", "tenant_id": "tenant-a"},
    ],
)
@pytest.mark.asyncio
async def test_a_payload_without_a_clear_scope_fails_without_retries(payload):
    from temporalio.exceptions import ApplicationError

    set_maintenance_store(_TenantStore([]))
    with pytest.raises(ApplicationError) as failed:
        await maintenance_tick(payload)
    assert failed.value.non_retryable


def test_maintenance_workflow_is_registered_name():
    definition = getattr(RuntimeMaintenanceWorkflow, "__temporal_workflow_definition")
    assert definition.name == "RuntimeMaintenanceWorkflow"


@pytest.mark.asyncio
async def test_reap_leases_needs_a_workspace_backend(monkeypatch):
    from orbit_worker import maintenance

    monkeypatch.setattr(maintenance, "_store", object())
    monkeypatch.setattr(maintenance, "_workspaces", None)
    with pytest.raises(RuntimeError, match="workspace backend"):
        await maintenance_tick({"operation": "reap_leases", "tenant_id": "tenant-a"})
