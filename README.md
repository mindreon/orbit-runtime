# orbit-runtime

Python runtime for Orbit: the Temporal `TaskWorkflow` and the AgentScope adapter. Decisions live in
[orbit-infra `docs/adr`](https://github.com/mindreon/orbit-infra/blob/main/docs/adr/README.md); the design is in
[orbit-infra `docs/architecture`](https://github.com/mindreon/orbit-infra/tree/main/docs/architecture).

## Packages

| Package | May import | Process |
| --- | --- | --- |
| `orbit_contracts` | pydantic | none — types only (contract v3 under `orbit_contracts.v3`) |
| `orbit_orch` | temporalio, orbit_contracts | `orbit-orch` polls the workflow queue: `TaskWorkflow` (auto-upgrade), `AttemptWorkflow` (pinned), `SopStepWorkflow`, maintenance |
| `orbit_worker` | temporalio, orbit_contracts, agentscope (not `agentscope.app`) | `orbit-worker` polls the agent and io queues: `agent_turn`, SOP steps, checkpoints, workspace and maintenance activities |

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

Both processes read their environment through `pydantic-settings` (`orbit_orch.settings`, `orbit_worker.settings`),
so a bad value stops startup with the variable named. Logs go through structlog on stderr: `ORBIT_LOG_FORMAT=json` prints
one JSON object per line (like orbit-control), `ORBIT_LOG_LEVEL` sets the level for libraries (default `WARNING`; our
own loggers always show `INFO`). Every line of an activity carries `tenant_id`, `task_id` and `attempt_id`.

JSON Schema for control lives in `schema/v3`. Regenerate with `uv run python -m orbit_contracts.schema_export`.

## State key

Production is the default. With `ORBIT_STATE_STORE_URL` set, every blob is
written as `fernet:` with `ORBIT_STATE_KEY`, and only blobs that key decrypts
can be read.

| Variable | Meaning |
| --- | --- |
| `ORBIT_STATE_KEY` | Fernet key: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `ORBIT_ALLOW_PLAINTEXT_STATE` | Exactly `1` allows `plain:` blobs, for local development only. Unset, empty, or any other value (`true`, `1 `) is production. |

- In production, a missing key, or a key that is not 44 url-safe base64
  characters decoding to 32 bytes, stops the worker at startup with exit
  code 1. The error names the variable, never the key. A set key must be
  valid even when plaintext state is allowed.
- The worker logs one WARNING at startup: `state store: memory`,
  `state store: postgres, encrypted`, or, with the flag,
  `state store: postgres, <encrypted|not encrypted>, plaintext state allowed (ORBIT_ALLOW_PLAINTEXT_STATE=1)`.
- **There is no key rotation** and no migration. **Losing or changing
  `ORBIT_STATE_KEY` invalidates those sessions' agent state**: every blob
  written with the old key becomes unreadable for good. The same holds for
  `plain:` blobs once a worker runs in production.
- In production a `plain:` blob, a `fernet:` blob the current key cannot decrypt, or a blob with no known prefix is
  unreadable: the agent state of that session is void. The turn returns `status: "failed"`, `errorCode:
  "state_unreadable"`, `retryable: false` and emits `turn.failed`; the stored blob is not rewritten.
- The worker log names the session, the turn, and the case (plaintext not allowed, does not decrypt with the current
  key, no key, unknown prefix).

| `error_code` | Retryable | `error` / `message` |
| --- | --- | --- |
| `state_unreadable` | no | 此任务的运行状态已失效，无法继续。你可以查看记录，或新建任务继续工作。 |

Deploy order: generate a key, set `ORBIT_STATE_KEY` on the worker, then
deploy. Losing or changing the key later invalidates the agent state of every
session written with it; P0 has no rotation to recover from that.

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

`orbit-orch` registers Temporal Schedules for the maintenance workflow (`ORBIT_MAINTENANCE_ENABLED`,
`ORBIT_MAINTENANCE_TENANTS`, `ORBIT_MAINTENANCE_REAP_SECONDS`, `ORBIT_MAINTENANCE_GC_SECONDS`,
`ORBIT_MAINTENANCE_ATTEMPTS_SECONDS`): expired workspace leases are reaped, stale checkpoints collected and attempts
left running for a day cleaned up. A tick that arrives while the previous run is still open is skipped.

Workflow history changes go behind `workflow.patched` change ids in
`orbit_orch.versioning`. Do not reuse an id, and do not drop the old branch
while an open execution can still replay it.

### Releasing a new build (Worker Versioning)

Set `ORBIT_USE_WORKER_VERSIONING=1` on `orbit-orch` and `orbit-worker`, with `ORBIT_WORKER_DEPLOYMENT` (default
`orbit`) and a distinct `ORBIT_WORKER_BUILD_ID` per release. `TaskWorkflow` auto-upgrades: an open task follows the
deployment's current version and relies on `workflow.patched` ids for replay. `AttemptWorkflow` is pinned: an attempt
finishes on the build it started on.

1. Start the new build's `orbit-orch` and `orbit-worker` next to the old ones. Do not stop the old build.
2. Once the new build's pollers have registered, make it current:
   `temporal worker deployment set-current-version --deployment-name orbit --build-id <new>`.
   New tasks, and open tasks at their next workflow task, move to the new build; running attempts stay on the old one.
3. Retire the old build when `temporal workflow list --query 'ExecutionStatus="Running"'` shows nothing pinned to it.
4. To roll back, set the old build current again while it is still up.

The acceptance suite exercises this (`e2e/stack` E13 in orbit-web): a long attempt keeps running on `v1` while `v2` is
released and becomes current, and a task started afterwards runs its attempt on `v2`.
