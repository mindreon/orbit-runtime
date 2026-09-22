# orbit-runtime

Python runtime for Orbit: Temporal workflows and the AgentScope Framework
adapter. Decisions live in
[orbit-infra `docs/adr`](https://github.com/mindreon/orbit-infra/blob/main/docs/adr/README.md).

`orbit-orch` and `orbit-worker` (TypeScript) are not where new code goes.
They stay only until the Compose cutover, then they are archived read-only.

## Packages

| Package | May import | Process |
| --- | --- | --- |
| `orbit_contracts` | pydantic | none — types only |
| `orbit_orch` | temporalio, orbit_contracts | `orbit-orch` polls the workflow queue |
| `orbit_worker` | temporalio, orbit_contracts, agentscope (not `agentscope.app`) | `orbit-worker` polls the activity queue |

Both processes use task queue `orbit` (`TEMPORAL_TASK_QUEUE`). A workflow
worker registers workflows only; an activity worker registers activities
only. Sharing the queue is safe because each worker polls its own task type.

`agentscope` is pinned to `2.0.8`. Upgrades are their own change.

## Develop

```bash
uv sync
uv run pytest
uv run lint-imports
```

The default chat model is an in-process mock, so tests and a local worker
need no API key. `AgentState` is kept in a memory store for this wave; the
Postgres store from ADR-010 replaces it without changing Activity names.

## Layout of a run

`RoomWorkflow` (orch) owns the room FSM. `openSession`, `runTurn`,
`resolveApproval`, and `closeSession` (worker) rebuild an AgentScope `Agent`
from the saved state, run one step, and save the state before returning.
A human approval parks the workflow; the agent object does not stay alive.
