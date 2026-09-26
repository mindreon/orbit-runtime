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

On a successful turn, `TurnResult.error_code` is `None`. The `runTurn` and
`decide` Updates do not return JSON null for this field. The workflow sets
`errorCode` to `result.error_code or ""`, so a completed Update has
`errorCode` equal to the empty string. A failed Update has one of the codes
in the table.

The OpenAI client has already retried timeouts, connection errors, 429, and
5xx twice (`ORBIT_MODEL_MAX_RETRIES`) before a turn reports one of these codes. Temporal does not retry a
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
  stdout (`orbit-worker: starting`, `orbit-orch: starting`) and one to
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

## Real-model smoke

`scripts/e2e_real_model_smoke.py run` calls the real Qwen endpoint through the
same harness: a local Temporal dev server and one `orbit-orch`
(`orbit-workflows` in the report) plus `orbit-worker` pair per case, in `real`
mode with the Postgres state store. The real workflow needs
`ORBIT_MODEL_API_KEY`, an allowlisted `ORBIT_MODEL_BASE_URL`,
`ORBIT_MODEL_NAME`, and `ORBIT_SMOKE_POSTGRES_URL` (an empty database). With
`ORBIT_SMOKE_STUB_UPSTREAM=1` the base URL is not read; the only upstream is
the in-process stub.

| Case | Worker | Checks |
| --- | --- | --- |
| E-RM-1 | `ORBIT_MODEL_STREAM=false` | One short turn completes with `assistant.message` and one `usage` event; its token counts equal the provider's non-null prompt, completion, and total usage. `updateErrorCode` is `""`. |
| E-RM-2 | `ORBIT_MODEL_STREAM=true` | The gate counts at least 2 provider content chunks. `assistant.delta` events are ordered, contiguous, and join to the final message. The delta count is recorded under `observed` and is not asserted (the worker coalesces every 100 ms or 200 characters, and 64 tokens is about 130 characters). |

- Workers run with `ORBIT_MODEL_MAX_TOKENS=64`, `ORBIT_MODEL_MAX_RETRIES=0`,
  and `ORBIT_MODEL_TIMEOUT_SECONDS=60`. Their base URL is a local budget gate
  that contacts the upstream at most 2 times per real run and refuses the
  rest, so a real run makes at most 2 model calls. Before it forwards, the gate
  checks the allowlist, the model name, and `max_tokens`. The upstream post
  uses `allow_redirects=False`. A 3xx is not followed and is not forwarded;
  the turn fails as `provider_error`.
- After each case the harness searches both processes' stdout and stderr,
  every Failure and event in the workflow history, the posted events, and the
  stored state for the key and each half (as written, JSON-escaped, and
  JSON-escaped twice), and logs and Failures for the prompt and its canary. A
  0-byte log fails the case.
- `scan --report <file> --logs <dir>` repeats the key and prompt search over
  the report and every log, and fails on a missing or 0-byte log.
- The report (`artifacts-smoke/real-model-smoke.json`) has `gitSha`
  (`git rev-parse HEAD`), component versions, the model name, per case steps,
  expected, actual, and pass, and prompt, completion, and total tokens per case
  and summed.
- **Exception to byte-identical reruns:** real output varies, so the real
  workflow does not compare reruns. The report is structurally deterministic:
  keys, cases, steps, and `expected` are fixed for a commit, and `actual`
  equals `expected` on a pass. Each case's `observed` object is not asserted.
  It holds `usageEvent` (`inputTokens`, `outputTokens`, `latencyMs`),
  `providerUsage`, `providerContentChunks`, `assistantDeltas`,
  `assistantMessage` (`bytes`, `sha256`), `historyFailureEntries`, and
  `capturedBytes`. Top-level `usage` repeats provider token counts outside
  `observed`. The pull-request stub job (below) runs twice and `cmp`s the
  reports after removing `observed`; that job does compare top-level `usage`,
  because the stub's token counts are fixed. The real workflow compares
  neither.

`.github/workflows/real-model-smoke.yml` does not run from a pull request.
Merge the change to `main` first, then start it by hand from `main`: on the
Actions page, choose real-model-smoke and run the workflow on `main`, or run
`gh workflow run real-model-smoke.yml --ref main`. The job uses the GitHub
Environment `qwen-smoke`, and a `qwen-smoke` approver must approve that
deployment. Until an approver approves, no step runs and the model is not
called.

| Name | Kind | Required |
| --- | --- | --- |
| `ORBIT_MODEL_API_KEY` | `qwen-smoke` environment secret only. Not a repository secret and not an organization secret | yes. The job fails at the key step when empty, before any model call |
| `ORBIT_MODEL_NAME` | `qwen-smoke` environment variable | yes. No default. The job fails in the config step, before the key step, when it is empty |
| `ORBIT_MODEL_BASE_URL` | `qwen-smoke` environment variable | yes. No default. The job fails in the config step, before the key step, when it is empty or not allowlisted |

DashScope keys are region-bound. Ops set the base URL from this table, and
set the model name for that account. A mismatched key returns HTTP 401 and
the smoke fails the case as `auth`.

| Key region | Base URL |
| --- | --- |
| Beijing (China mainland) | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| International (Singapore) | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |

The allowlist is those two hosts, `https` only, no userinfo, port empty or
443. A raw value that contains whitespace or a control character is rejected
before it is parsed, so `urllib` and `aiohttp` cannot disagree. Any other
host, and any `http` URL, fails the config step before the key is read. The
harness and the gate reject it again before forwarding.

Ops checklist, before the key is created and again if the environment is
recreated:

1. Environment `qwen-smoke` exists.
2. Deployment branches are `main` only.
3. Required reviewers are set.
4. The key is only the `qwen-smoke` environment secret. It is not a
   repository secret and not an organization secret.
5. Base URL and model name are set on that environment, with no reliance on
   a workflow default.
6. `ORBIT_SMOKE_STUB_UPSTREAM` is not set on the repository or the
   environment.

The real control is that environment rule plus the required reviewers. A
dispatch from branch X runs branch X's workflow file, and anyone who can push
can delete the `GITHUB_REF` step from that file. The step is defense-in-depth
only: it exits when `GITHUB_REF` is not `refs/heads/main`, and it runs before
any step receives the key, but it does not by itself stop another branch once
the environment rule is gone. `SMOKE_SHA` is `github.sha`. The key is masked
before any later step and set only on the steps that use it. The report and
logs are uploaded only when the leak scan and gitleaks (pinned,
checksum-checked) both find nothing.

Guards and failure modes:
[`docs/real-model-smoke.md`](docs/real-model-smoke.md).

Pull-request CI does not call Qwen and does not use the `qwen-smoke`
environment. Job `real-model-smoke-stub` in `ci.yml` sets
`ORBIT_SMOKE_STUB_UPSTREAM=1` and runs this harness twice against an
in-process OpenAI-compatible stub (one non-streamed response, one SSE
response, and one 302 to a different host), through the same gate, Temporal,
orbit-workflows, orbit-worker, and Postgres. The 302 case passes only when
the turn fails as `provider_error`, the other host accepts no connection, and
nothing is forwarded. That job's budget is 3; a real run stays at 2 and does
not start the 302 case. Before the runs, `config-checks` rejects raw base
URLs that contain whitespace or a control character. After each run,
`jq -e --arg s "$SHA" '.gitSha == $s'` checks that report, where `SHA` is
`git rev-parse HEAD`; an empty `SHA` fails the step. It checks out the PR
head, uses `permissions: contents: read`, pins actions by SHA, sets
`persist-credentials: false`, scans the reports and logs with pinned
gitleaks, and uploads the report with `if: always()`. The two reports are
compared with `cmp` after each case's `observed` object is removed. The
real workflow asserts the stub flag is unset, in the config step, before the
key step, and sets `enable-cache: false` on `setup-uv`. The flag does not add
hosts to the allowlist: the stub job never reads `ORBIT_MODEL_BASE_URL` as an
upstream. `actionlint` still lints the workflow files; that job pins its
actions by SHA and sets `persist-credentials: false`. `postgres:16-alpine`
is pinned by digest.

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
