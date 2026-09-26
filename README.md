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
ADR-010. Without it, state stays in memory and needs no key.

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
- **P0 has no key rotation** and no migration. **Losing or changing
  `ORBIT_STATE_KEY` invalidates those sessions' agent state**: every blob
  written with the old key becomes unreadable for good. The same holds for
  `plain:` blobs once a worker runs in production.
- In production a `plain:` blob, a `fernet:` blob the current key cannot
  decrypt, or a blob with no known prefix is unreadable: that session's agent
  runtime state is void. Chat history and artifacts in control are not
  affected.
- A turn on an unreadable session (`runTurn`, `decide`/`resolveApproval`,
  `deliverToolResult`, `steer`) returns `status: "failed"`, `errorCode:
  "state_unreadable"`, `retryable: false`, and emits `turn.failed` with the
  fixed message below. The Activity completes, so Temporal does not retry it,
  and the stored blob is not rewritten. The room stays `running`.
- `closeSession` and `abort` on an unreadable session succeed: they return
  `closed: true` with the row's stored `state_version` unchanged, write
  nothing, and log one line with the case. A repeat returns the same answer.
  So an abort or a control DELETE still takes `RoomWorkflow` to `closed`, and
  `AgentRunWorkflow` and `CloudAgentJob` finish (a failed `CloudAgentJob`
  still reports its own failure).
- `openSession` fails once with a non-retryable `ApplicationError` of type
  `state_unreadable` and the same text. **This only occurs when a session is
  reopened via its idempotency key over an old blob** (the same room and open
  turn id, e.g. a retried `openSession`, after the key or the mode changed).
  A new session never reads an old blob.
- The worker log names the session, the turn, and the case (plaintext not
  allowed, does not decrypt with the current key, no key, unknown prefix).

| `error_code` | Retryable | `error` / `message` |
| --- | --- | --- |
| `state_unreadable` | no | 这个会话的运行状态已无法读取，无法继续对话；历史记录仍可查看。 |

Deploy order: generate a key, set `ORBIT_STATE_KEY` on the worker, then
deploy. Losing or changing the key later invalidates the agent state of every
session written with it; P0 has no rotation to recover from that.

`ORBIT_ISOLATION_MODE=bwrap` builds `BubblewrapBackend` with `share_net=False`.
`docker` and `k8s` require `ORBIT_SANDBOX_IMAGE` (a pre-baked digest). The
worker also polls `{TEMPORAL_TASK_QUEUE}-gateway` for Tool Gateway activities.

## Chat model

`ORBIT_MODEL_MODE` picks the chat model when `orbit-worker` starts.

| Variable | Required when `real` | Meaning |
| --- | --- | --- |
| `ORBIT_MODEL_MODE` | — | `mock` (default when unset) or `real` |
| `ORBIT_MODEL_BASE_URL` | yes | OpenAI-compatible endpoint, e.g. `https://host/v1` |
| `ORBIT_MODEL_API_KEY` | yes | API key for that endpoint |
| `ORBIT_MODEL_NAME` | yes | Model name sent to the endpoint |
| `ORBIT_MODEL_TIMEOUT_SECONDS` | no | Per-request timeout, default `60` |

- `mock` uses `MockChatModel`. If any real variable is set, the worker logs
  which names are set and which are missing, then still uses the mock.
- `real` with a required variable unset stops the worker at startup. The
  error names the missing variables.
- The key and base URL are read inside the worker process when the model is
  built. They are not in Temporal inputs or outputs, events, logs, or error
  text. Only the mode and model name leave the worker.
- Every `TurnResult` carries `model_mode` (`mock` or `real`) and
  `model_name`; every event carries them as `modelMode` and `modelName`. The
  `runTurn` and `decide` Updates return them as `modelMode` and `modelName`.
- A failed real request (network, timeout, 4xx, 5xx) returns a turn with
  `status: "failed"`, an `error_code`, and a fixed `error` text. The saved
  state stays at the previous version. The worker never falls back to the
  mock. Retry with a new turn id; the same id returns the cached failure.
- The worker logs one WARNING at startup: `chat model: mock`, or
  `chat model: real model=<name>`.

`TurnResult` carries `error_code` and `retryable`. The `runTurn` and `decide`
Updates return them as `errorCode` and `retryable`. Each failed turn also
emits a `turn.failed` event whose `failure` object has `turnId`, `agentId`
(the contract agent id, `main` for the room agent), `errorCode`, `retryable`,
and `message`. A
human-readable `session.status` event `turn failed: …` is still emitted.
Clients map the code to text and never parse `error`, `message`, or
`session.status` text.

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
5xx twice before a turn reports one of these codes. Temporal does not retry a
failed turn, because tools may already have run in it; `retryable` tells the
client whether sending a new turn is worth it.

See `.env.example`. Tests that call a real endpoint skip unless
`ORBIT_MODEL_MODE=real` and the three required variables are set.

## Events

The worker posts each `OrbitEvent` to `ORBIT_EVENT_INGEST_URL` as camelCase
JSON (`by_alias=True`, `exclude_none=True`). Every event carries `turnId`,
`agentId`, and `agentPath` (`main` for the room agent) plus the parent fields.

| Type | When | Fields |
| --- | --- | --- |
| `tool.call` | the model finishes a tool call, in-Activity or external | `toolName`, `callId`, `argsPreview` |
| `tool.result` | a tool result lands, including a delivered external result | `toolName`, `callId`, `toolState`, `text` (redacted, at most 4096 UTF-8 bytes), `truncated` |
| `assistant.delta` | streamed text, at most every 100 ms or 200 characters per block | `blockId`, `seq`, `delta`, `activityAttempt` |
| `usage` | each model call ends | `model`, `inputTokens`, `outputTokens`, `cacheInputTokens`, `cacheCreationInputTokens`, `latencyMs` |

`assistant.delta` is for live display only; `assistant.message` still carries
the final text. `activityAttempt` grows when Temporal retries the Activity,
so a client drops the draft from a lower attempt.

Suspected secrets in `text`, `delta`, and `argsPreview` become `[REDACTED]`;
the event is still sent. A value counts as secret when its key, lower-cased
with `-` and `_` removed, contains `apikey`, `secret`, `privatekey`, `token`,
`cookie`, `authorization`, or `password`; when it starts with `sk-`, `ghp_`,
`github_pat_`, `glpat-`, `AKIA`, `AIza`, or `xox[abprs]-`; when it follows
`Bearer`; when it is 40-character lower-case hex right after a secret-named
key; or when it is a random-looking string of 32 or more characters. A delta chunk is released up to its last character
that cannot be part of a token (`A-Z a-z 0-9 . _ ~ + / = -`), so CJK text
streams while an unfinished token waits; a token run over 512 characters is
released anyway. The tail of released text is rescanned with the next chunk,
so a secret split across two chunks is still caught. `argsPreview` is
redacted, then cut to 256 characters.

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

`scripts/e2e_a1_events.py run` is the A1 sign-off check (E-A1-1 to E-A1-5)
plus E-LOG-1 to E-LOG-3.

- It starts a local Temporal dev server and five `orbit-orch` plus
  `orbit-worker` pairs. One pair runs the streaming mock model; one runs
  `real` mode against an OpenAI-compatible stub, which counts requests and can
  answer 200 with null usage; each E-LOG case has its own `real`-mode pair.
- Every process runs with `PYTHONUNBUFFERED=1` and writes stdout and stderr to
  separate files under `--logs`. Both processes write a startup line to
  stderr.
- Workers post events to a recording ingest stub. Rooms are driven with the
  `runTurn` and `decide` Updates.
- Each E-LOG case sends a user message carrying a planted secret and a
  conversation canary, then stops its pair. E-LOG-1 is the happy path: the
  secret also goes through a `gated_echo` tool call, its result, and the
  reply. In E-LOG-2 the stub answers HTTP 500 with the prompt echoed in the
  error body. In E-LOG-3 it answers 200 with null usage (`provider_error`).
- Every E-LOG case asserts that neither the secret, either half, nor the
  canary is in the worker's or the orch's stdout or stderr, or in any Failure
  in the workflow history. It records the byte count of every capture, and a
  process whose stdout and stderr are both empty fails the case. E-LOG-2 and
  E-LOG-3 also assert that the secret appears only in the history events that
  carry the user's message as input.
- The E-LOG report labels the workflow process `orbit-workflows`, its planned
  name; its package and entrypoint are still `orbit-orch`, and its log files
  are `orbit-orch-*.log`.
- It writes `artifacts/e2e-a1-events.json` with the commit, component
  versions, and per case the id, steps, expected, actual, and pass. The file
  has no timestamps or ports, so two runs on one commit match byte for byte.
- `scripts/e2e_a1_events.py scan <files>` checks reports and logs for the
  planted test values, key and token formats, and database URLs.
- CI runs the check twice, compares the two reports, scans the reports and
  every process log with `scan` and gitleaks, and uploads `artifacts/` and
  `e2e-logs/`.

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

The mock model's `stream:` and `echo:` scripts exist for this check.

`scripts/e2e_state_key.py run` is the state key sign-off check (E-SK-1 to
E-SK-4). It reuses the A1 harness and needs `ORBIT_TEST_POSTGRES_URL`; it
drops and recreates the `orbit_e2e_state_key` schema there.

- E-SK-1 starts `orbit-worker` under production configurations with no
  usable key and checks exit code 1, the fixed error line, and that no
  6-character fragment of a test key reaches the log.
- E-SK-2 and E-SK-3 seed a session (plaintext, or key A), restart the worker
  in production (key A, or key B), and send a turn: `state_unreadable`, one
  Activity attempt in the Temporal history, the stored blob and earlier
  events unchanged.
- E-SK-4 checks that blobs are `fernet:` and that the conversation continues
  across two worker restarts.
- E-SK-5 creates rooms through `orbit-control` (a binary passed with
  `--control-bin`, built from the pinned control commit), makes their state
  unreadable by restarting the worker with key B, then aborts one and
  DELETEs both through control: the workflow completes with the room
  `closed`, `closeSession` runs once, the blob is unchanged, and DELETE
  returns 204. Without `--control-bin` the case fails and says so.
- `scripts/e2e_state_key.py scan <files or directories>` checks reports and
  process logs. CI runs the check twice, compares the reports, scans reports
  and logs with `scan` and gitleaks, and uploads both.
- The report's `commit` is `git rev-parse HEAD` of the checkout, with no
  override; CI checks out the PR head and fails if any E2E report names
  another commit.
- CI checks out full history (`fetch-depth: 0`) and runs the pinned gitleaks
  over the repository and all of its history, all reports, and all process
  logs. `.gitleaksignore` lists single findings by fingerprint; today it has
  one, a fake GitLab token in the redactor test.

JSON Schema for control lives in `schema/`. Regenerate with
`uv run python -m orbit_contracts.schema_export`.

## Recurring runs and workflow versions

`orbit-orch` can register one Temporal Schedule at startup. Set
`ORBIT_RECURRING_SCHEDULE_ID` and `ORBIT_RECURRING_REPO_URL`. Optional:
`ORBIT_RECURRING_EVERY_SECONDS` (default `86400`), `ORBIT_RECURRING_BRANCH`,
`ORBIT_RECURRING_PROMPT`. Each tick starts `CloudAgentJob`. A tick that
arrives while that job is still open is skipped.

Workflow history changes go behind `workflow.patched` change ids in
`orbit_orch.versioning`. Do not reuse an id, and do not drop the old branch
while an open execution can still replay it.
