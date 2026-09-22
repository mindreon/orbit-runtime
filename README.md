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
need no API key. Set `ORBIT_STATE_STORE_URL` to use the Postgres store from
ADR-010. Without it, state stays in memory. Set `ORBIT_STATE_KEY` (a Fernet
key) so blobs are encrypted before they are written.

`ORBIT_ISOLATION_MODE=bwrap` builds `BubblewrapBackend` with `share_net=False`.
`docker` and `k8s` require `ORBIT_SANDBOX_IMAGE` (a pre-baked digest). The
worker also polls `{TEMPORAL_TASK_QUEUE}-gateway` for Tool Gateway activities.

## Layout of a run

`RoomWorkflow` owns the room FSM. `openSession`, `runTurn`,
`resolveApproval`, `deliverToolResult`, and `closeSession` rebuild an
AgentScope `Agent` from the saved state, run one step, and save the state
before returning. A human approval parks the workflow. External tools
(`gateway_*`, `agent_spawn`, `agent_send`, `agent_wait`, `team_dissolve`)
park the same way; the workflow runs the gateway Activity or a child
`AgentRunWorkflow`, then delivers the result. `CloudAgentJob` clones a
workspace, runs one turn, pushes a branch marker, and returns a pull-request
URL. The agent object does not stay alive across Activities.

JSON Schema for control lives in `schema/`. Regenerate with
`uv run python -m orbit_contracts.schema_export`.
