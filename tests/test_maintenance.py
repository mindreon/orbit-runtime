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
    assert result == {"operation": "gc_checkpoints", "tenant_id": "tenant-a", "removed": 3}


@pytest.mark.asyncio
async def test_maintenance_workflow_executes_io_activity():
    class FakeStore:
        async def maintenance(self, operation, *, tenant_id):
            assert operation == "reap_leases"
            assert tenant_id == "default"
            return 7

    set_maintenance_store(FakeStore())
    async with await WorkflowEnvironment.start_time_skipping() as env, Worker(
        env.client, task_queue="orbit.orch", workflows=[RuntimeMaintenanceWorkflow]
    ), Worker(env.client, task_queue="orbit.io", activities=[maintenance_tick]):
        result = await env.client.execute_workflow(
            RuntimeMaintenanceWorkflow.run,
            {"operation": "reap_leases", "tenant_id": "default", "task_queue": "orbit.io"},
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


def test_maintenance_workflow_is_registered_name():
    definition = getattr(RuntimeMaintenanceWorkflow, "__temporal_workflow_definition")
    assert definition.name == "RuntimeMaintenanceWorkflow"
