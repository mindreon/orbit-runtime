# orbit-runtime

Python runtime for Orbit: the Temporal `TaskWorkflow` and the AgentScope adapter. Decisions live in
[orbit-infra `docs/adr`](https://github.com/mindreon/orbit-infra/blob/main/docs/adr/README.md); the design is in
[orbit-infra `docs/architecture`](https://github.com/mindreon/orbit-infra/tree/main/docs/architecture).

## Packages

| Package | May import | Process |
| --- | --- | --- |
| `orbit_contracts` | pydantic | none — types only (contract v3 under `orbit_contracts.v3`) |
| `orbit_orch` | temporalio, orbit_contracts | `orbit-orch` polls the workflow queue: `TaskWorkflow` (auto-upgrade), `AttemptWorkflow` (pinned), `SopStepWorkflow`, maintenance; a `sop_stage` node is compiled into plan nodes (`task_sop`) |
| `orbit_worker` | temporalio, orbit_contracts, agentscope (not `agentscope.app`) | `orbit-worker` polls the agent and io queues: `agent_turn`, `verify_sop_step` (the verifier of a compiled SOP step), checkpoints, workspace and maintenance activities; the old `sop_step` is deprecated |

Task queues: `orbit.orch` (workflows), `orbit.agent` (agent activities) and `orbit.io` (workspace and maintenance
activities). A workflow worker registers workflows only; an activity worker registers activities only.

`agentscope` is pinned to `2.0.9`, embedded as a library (no `create_app` service). Upgrades are their own change.

## Develop

```bash
uv sync
uv run pytest
uv run lint-imports
```

The default chat model is an in-process mock, so tests and a local worker need no API key. The full stack (Temporal,
control, Postgres, MinIO, orbit-web) is exercised by orbit-web's `pnpm test:stack`, including the real-model cases.

The `TaskStore` tests (`tests/test_*_pg.py`) run against a real Postgres with the orbit-control migrations, connected as
`orbit_worker` so row-level security and the column grants apply. They skip unless `ORBIT_TEST_POSTGRES_URL` (a
superuser URL, e.g. `postgresql://orbit:orbit@127.0.0.1:5432/orbit` of a `postgres:16-alpine` container) is set;
`ORBIT_CONTROL_DIR` names the orbit-control checkout (default: the sibling `orbit-control`). CI sets
`ORBIT_REQUIRE_PG_TESTS=1`, so there a missing database fails instead of skipping.

Both processes read their environment through `pydantic-settings` (`orbit_orch.settings`, `orbit_worker.settings`),
so a bad value stops startup with the variable named. Logs go through structlog on stderr: `ORBIT_LOG_FORMAT=json` prints
one JSON object per line (like orbit-control), `ORBIT_LOG_LEVEL` sets the level for libraries (default `WARNING`; our
own loggers always show `INFO`). Every line of an activity carries `tenant_id`, `task_id` and `attempt_id`.

JSON Schema for control lives in `schema/v3`. Regenerate with `uv run python -m orbit_contracts.schema_export`.

## Agent session state

The agent state of a task attempt is a checkpoint (`kind = agent_state`, 08 §3): after every turn the worker writes the
session, with the cached result of each turn, into the `checkpoints` table and the object store. The session id of an
attempt is its attempt id, so whichever worker gets the next activity of the attempt (after a crash, an approval or a
user reply) reads the state back. There is no separate state database. Sessions outside an attempt (SOP step agents,
local development without a database) stay in memory.

| Variable | Meaning |
| --- | --- |
| `ORBIT_CHECKPOINT_FERNET_KEY` | Optional Fernet key for every checkpoint: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`. Unset stores them in clear. |

- **There is no key rotation.** Losing or changing the key makes the checkpoints written with the old one unreadable.
- A checkpoint the key cannot decrypt, or that is not a session, ends the turn as `status: "failed"`, `errorCode:
  "state_unreadable"`, `retryable: false` and emits `turn.failed`; nothing is rewritten.

| `error_code` | Retryable | `error` / `message` |
| --- | --- | --- |
| `state_unreadable` | no | 此任务的运行状态已失效，无法继续。你可以查看记录，或新建任务继续工作。 |
| `budget` | no | The attempt spent the budget its task reserved for it (05 §4); `error` names the limit. Not shown as text: the attempt fails as `failure_class: budget`, its node is blocked and the task asks for a review until budget is granted. |

`ORBIT_ISOLATION_MODE=bwrap` builds `BubblewrapBackend` with `share_net=False`. `docker` and `k8s` require
`ORBIT_SANDBOX_IMAGE` (a pre-baked digest).

## Chat model

`ORBIT_MODEL_MODE` picks the chat model when `orbit-worker` starts.

| Variable | Required when `real` | Meaning |
| --- | --- | --- |
| `ORBIT_MODEL_MODE` | — | `mock` (default when unset) or `real` |
| `ORBIT_MODEL_BASE_URL` | yes | OpenAI-compatible endpoint, e.g. `https://host/v1` |
| `ORBIT_MODEL_API_KEY` | yes | API key for that endpoint |
| `ORBIT_MODEL_NAME` | yes | Model name sent to the endpoint |
| `ORBIT_MODEL_TIMEOUT_SECONDS` | no | Per-request timeout, default `60` |
| `ORBIT_MODEL_MAX_TOKENS` | no | Output cap per model call, sent as `max_tokens`; unset sends none |
| `ORBIT_MODEL_STREAM` | no | `true` streams the response (usage via `stream_options.include_usage`); default `false` |
| `ORBIT_MODEL_MAX_RETRIES` | no | OpenAI-client retries on timeouts, connection errors, 429, 5xx; default `2` |
| `ORBIT_MODEL_PRICE_INPUT_PER_MTOK`, `ORBIT_MODEL_PRICE_OUTPUT_PER_MTOK` | no | What the model costs, in micro-dollars per million input and output tokens (3 USD per million is `3000000`). Both set makes an attempt's cost known, so a task's `cost_usd_micros` budget is enforced; with either missing the cost is recorded as unknown and not enforced. A profile's `model_params.price_input_per_mtok` / `price_output_per_mtok` override them |
| `ORBIT_MODEL_CONTEXT_SIZE` | no | The model's context window in tokens (at least `1024`). The agent compresses its context when it nears this, so set the real window; a profile's `model_params.context_size` overrides it. Unset keeps AgentScope's default (`128000` for the OpenAI-compatible model) |

- `mock` uses `MockChatModel`. If any real variable is set, the worker logs
  which names are set and which are missing, then still uses the mock.
- `real` with a required variable unset stops the worker at startup. The
  error names the missing variables.
- The key and base URL are read inside the worker process when the model is
  built. They are not in Temporal inputs or outputs, events, logs, or error
  text. Only the mode and model name leave the worker.
- Every `TurnResult` carries `model_mode` (`mock` or `real`) and `model_name`; every event carries them as
  `modelMode` and `modelName`.
- A failed real request (network, timeout, 4xx, 5xx) returns a turn with
  `status: "failed"`, an `error_code`, and a fixed `error` text. The saved
  state stays at the previous version. The worker never falls back to the
  mock. Retry with a new turn id; the same id returns the cached failure.
- The worker logs one WARNING at startup: `chat model: mock`, or
  `chat model: real model=<name>`.

A failed model call returns a turn result with an `error_code`, `retryable` and a fixed message; the attempt reports it
as a `turn.failed` event. Clients map the code to text and never parse `error` or `message`.

`error` and `message` are the fixed text for the code below
(`FAILURE_MESSAGES` in `chat_model.py`). They are never built from the
provider response or the exception, so they cannot carry a key, host, URL, or
request body. The exception class and HTTP status go to the worker log only.

| `error_code` | Source | Retryable | `error` / `message` |
| --- | --- | --- | --- |
| `timeout` | request timed out (`APITimeoutError`) | yes | 模型响应超时，这一轮没跑完。 |
| `auth` | HTTP 401, 403 | no | 模型配置有问题，请联系管理员。 |
| `rate_limited` | HTTP 429 | yes | 模型当前请求太多，请稍后再试。 |
| `provider_error` | HTTP 5xx, connection errors, any other non-HTTP error, a response that cannot be read | yes | 模型服务暂时出错，这一轮没跑完。 |
| `config` | HTTP 400, 404 (e.g. wrong model name), other 4xx | no | 模型配置有问题，请联系管理员。 |

`auth` and `config` share one user text on purpose. Tell them apart by
`errorCode`, or by the exception class and HTTP status in the worker log.

The OpenAI client has already retried timeouts, connection errors, 429, and
5xx twice (`ORBIT_MODEL_MAX_RETRIES`) before a turn reports one of these codes. Temporal does not retry a
failed turn, because tools may already have run in it; `retryable` tells the
client whether sending a new turn is worth it.

See `.env.example`. Tests that call a real endpoint skip unless
`ORBIT_MODEL_MODE=real` and the three required variables are set.

## Tracing

Neither process exports spans by default: no exporter and no SDK
`TracerProvider` are installed, so AgentScope's `TracingMiddleware` skips
building span attributes. `orbit_worker.tracing.configure_tracing(exporter)`
installs a provider only for a given exporter and wraps it in
`RedactingSpanExporter`, which drops the conversation-content attributes
(`gen_ai.input.messages`, `gen_ai.output.messages`,
`gen_ai.system_instructions`, `gen_ai.tool.call.arguments`,
`gen_ai.tool.call.result`, `gen_ai.prompt`, `gen_ai.completion`) and passes
every other string (attributes, event and link attributes such as
`exception.message`, span names, status text) through the event redactor.
Do not add a console exporter: it writes span content to process stdout.

## Workspaces, maintenance and workflow versions

### Workspace backends

The worker selects a workspace backend with `ORBIT_ISOLATION_MODE` (or
`ORBIT_WORKSPACE_BACKEND`). `local` is the default, `docker` starts an isolated
container, and `opensandbox` uses the pinned `opensandbox==0.1.16` SDK. For the
OpenSandbox backend set `ORBIT_OPENSANDBOX_DOMAIN` and
`ORBIT_OPENSANDBOX_IMAGE`; snapshots are stored through the configured object
store under the content addressed `snapshots/<tenant>/<sha256>` prefix.

`orbit-orch` registers three Temporal Schedules for the maintenance workflow (`ORBIT_MAINTENANCE_ENABLED`,
`ORBIT_MAINTENANCE_REAP_SECONDS`, `ORBIT_MAINTENANCE_GC_SECONDS`, `ORBIT_MAINTENANCE_ATTEMPTS_SECONDS`). Each tick
runs for every tenant that exists at that moment (the worker lists tenant ids, then works on each tenant in its own
transaction), so a new tenant needs no configuration:

- `reap_leases` kills the sandbox of every expired workspace lease, then releases the lease;
- `gc_checkpoints` deletes checkpoints no history references (the newest one of an attempt always stays) and the
  blobs nothing else references;
- `cleanup_attempts` closes out attempts left STARTING or RUNNING for a day only when Temporal says their
  AttemptWorkflow is gone or closed; a parked attempt is left alone and no row is deleted.

A tick that arrives while the previous run is still open is skipped. Schedules made by older releases, one per
tenant (`orbit-maintenance-<tenant>-<operation>`), keep working and can be deleted.

Workflow history changes go behind `workflow.patched` change ids in
`orbit_orch.versioning`. Do not reuse an id, and do not drop the old branch
while an open execution can still replay it.

### Releasing a new build (Worker Versioning)

Set `ORBIT_USE_WORKER_VERSIONING=1` on `orbit-orch` and `orbit-worker`, with `ORBIT_WORKER_DEPLOYMENT` (default
`orbit`) and a distinct `ORBIT_WORKER_BUILD_ID` per release. `TaskWorkflow` auto-upgrades: an open task follows the
deployment's current version and relies on `workflow.patched` ids for replay. `AttemptWorkflow` is pinned: an attempt
finishes on the build it started on.
Both processes read the switch from `orbit_orch.settings` at startup, and each refuses to start when a process that
polls the queue it depends on is versioned differently (or is in another deployment); the message names the queue and
the variable.


1. Start the new build's `orbit-orch` and `orbit-worker` next to the old ones. Do not stop the old build.
2. Once the new build's pollers have registered, make it current:
   `temporal worker deployment set-current-version --deployment-name orbit --build-id <new>`.
   New tasks, and open tasks at their next workflow task, move to the new build; running attempts stay on the old one.
3. Retire the old build when `temporal workflow list --query 'ExecutionStatus="Running"'` shows nothing pinned to it.
4. To roll back, set the old build current again while it is still up.

The acceptance suite exercises this (`e2e/stack` E13 in orbit-web): a long attempt keeps running on `v1` while `v2` is
released and becomes current, and a task started afterwards runs its attempt on `v2`.
