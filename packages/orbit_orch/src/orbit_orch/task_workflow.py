"""TaskWorkflow and AttemptWorkflow for the v3 runtime.

The workflows contain only deterministic scheduling and state transitions.  All
Postgres, object store, workspace, model and event projection work is delegated
to named activities on the worker queues.

The workflow class is assembled from layers, each in its own module: `task_base` (state), `task_events`,
`task_plan`, `task_completion`, `task_attempts`, `task_commands`. `attempt_workflow` holds AttemptWorkflow.
Workflow, signal, update and query names and `workflow.patched` ids are part of recorded histories and do not
change with the file layout.
"""

from __future__ import annotations

from temporalio import workflow
from temporalio.common import VersioningBehavior

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.v3 import InboxMessage, TaskView, TaskWorkflowInput

    from orbit_orch.plan_engine import initial_plan
    from orbit_orch.workflow_common import (
        DRAIN_BEFORE_CONTINUE_AS_NEW,
        HELD_STAYS_HELD,
        OPERATOR_HELD,
        SESSION_STAYS_OPEN,
        SIGNAL_CLOSED_CHILD,
        VERIFY_FINISHED_ATTEMPTS,
        versioning_behavior,
    )

from orbit_orch.attempt_workflow import AttemptWorkflow
from orbit_orch.task_commands import TaskCommands


@workflow.defn(name="TaskWorkflow", versioning_behavior=versioning_behavior(VersioningBehavior.AUTO_UPGRADE))
class TaskWorkflow(TaskCommands):
    @workflow.run
    async def run(self, inp: TaskWorkflowInput) -> TaskView:
        workflow.patched("taskworkflow-v3")
        self._load(inp)
        if inp.carry is not None:
            self._rearm_after_load()
        self._status = "PLANNING" if inp.carry is None else self._status
        if self._plan is None:
            self._plan = initial_plan(inp.task_id, inp.goal, inp.profile)
            self._emit("task.created", {"title": inp.title, "goal": inp.goal})
        if inp.carry is None or self._status in {"CREATED", "PLANNING", "RUNNING"}:
            self._set_status("RUNNING", "started")
        while not self._stop:
            await self._drain_commands()
            await self._schedule_ready()
            if self._status in OPERATOR_HELD:
                pass  # pause and takeover stay until an operator resumes; a running attempt does not undo them
            elif any(item.get("status") == "RUNNING" for item in self._attempts.values()):
                self._set_status("RUNNING", "attempt_running")
            elif self._plan and any(
                state.status in {"AWAITING_APPROVAL", "AWAITING_INPUT"}
                for state in self._plan.nodes.values()
            ):
                self._set_status("WAITING", "awaiting_user")
            await self._flush_events()
            held = workflow.patched(HELD_STAYS_HELD) and (self._status in OPERATOR_HELD or self._status == "CANCELLED")
            if self._all_nodes_completed() and not held:
                if self._status != "COMPLETED":
                    self._status = "COMPLETED"
                    self._emit("task.completed", {})
                    await self._flush_events()
                    if self._inbox and self._unsent and workflow.patched(SIGNAL_CLOSED_CHILD):
                        # A message that could not be handed to an attempt that had closed is answered now, as a follow-up.
                        self._commands.append(("message", self._inbox[0]))
                    self._unsent = False
                if not workflow.patched(SESSION_STAYS_OPEN):
                    break
            if self._should_continue_as_new():
                await workflow.wait_condition(workflow.all_handlers_finished)
                if workflow.patched(DRAIN_BEFORE_CONTINUE_AS_NEW):
                    # Nothing may be left behind in the run that ends: a message or a completion an update accepted is still
                    # to be handled (the next run would never hear of it), the plan is compacted once more (what is finished
                    # and nothing needs goes into the archive), and the events are out. Each of these can let another update
                    # in, so it goes round until a pass finds nothing to do, and then continues as new without yielding.
                    while True:
                        await workflow.wait_condition(workflow.all_handlers_finished)
                        if self._commands:
                            await self._drain_commands()
                            continue
                        self._compact_plan(force=True)
                        if self._events:
                            await self._flush_events()
                            continue
                        break
                workflow.continue_as_new(self._carry_input())
            observed_wake = self._wake
            # A report being verified settles after the loop last flushed: its events must not wait for the next command.
            flush_settled = workflow.patched(VERIFY_FINISHED_ATTEMPTS)
            if not self._commands and not self._has_ready_node():
                try:
                    await workflow.wait_condition(
                        lambda observed_wake=observed_wake, flush_settled=flush_settled: bool(self._commands)
                        or self._has_ready_node()
                        or self._wake != observed_wake
                        or self._stop
                        or (flush_settled and bool(self._events)),
                        # A node that waits out the backoff of a retry is not ready yet: the loop wakes when it is.
                        timeout=self._next_backoff(),
                    )
                except TimeoutError:
                    pass
        if self._status == "CANCELLED":
            # Every child has to be cancelled before the task ends (04 §6).
            await workflow.wait_condition(lambda: not self._attempts)
        await self._flush_events()
        return self._task_view()

    @workflow.query(name="getTaskView")
    def get_task_view(self) -> TaskView:
        return self._task_view()

    @workflow.query(name="getInbox")
    def get_inbox(self, after_seq: int = 0) -> list[InboxMessage]:
        return [item for item in self._inbox if item.message_seq > after_seq]


__all__ = ["AttemptWorkflow", "TaskWorkflow"]
