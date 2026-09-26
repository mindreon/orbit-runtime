# Real-model smoke: guards and failure modes

`.github/workflows/real-model-smoke.yml` runs `scripts/e2e_real_model_smoke.py`
against the real Qwen endpoint with a spending-capped key. This file lists
every guard in that workflow and harness, what it protects, and what a run
looks like when the guard fires. It was written before the guards.

The workflow starts only by a manual `workflow_dispatch`. It is not triggered
by a pull request, and it never uses `pull_request_target`. It deploys only
from `main`, through the `qwen-smoke` environment, and a `qwen-smoke` approver
must approve that deployment. Until an approver approves, no step runs and no
model call is made.

A dispatch from branch X runs the workflow file **on branch X**. The
`GITHUB_REF` step is defense-in-depth only. Anyone who can push to a branch
can delete that step from that branch's file. The control that actually keeps
the key on `main` is the `qwen-smoke` environment: deployment branches limited
to `main`, plus required reviewers. Ops verify that before the key exists.
See the checklist below.

## What must never happen

1. The key reaches a process log, the report, a Temporal failure, an event,
   or an uploaded artifact.
2. The key is exposed to code from a fork, or to code from any ref other than
   `main`. The environment's deployment-branch rule and required reviewers are
   what enforce this. The `GITHUB_REF` step does not.
3. A run makes more than 2 model calls.
4. A run passes without a key, without real model output, or with a log
   capture that silently recorded nothing.

## Ops checklist

Do this before the first real run, and again if `qwen-smoke` is recreated.

1. The environment `qwen-smoke` exists.
2. Its deployment branches are `main` only.
3. It has required reviewers. A run with no approval makes no model call.
4. `ORBIT_MODEL_API_KEY` is a secret **of the `qwen-smoke` environment only**.
   It is not a repository secret and not an organization secret. A secret at
   those scopes can be read by workflows that do not use this environment.
5. `ORBIT_MODEL_BASE_URL` and `ORBIT_MODEL_NAME` are environment variables on
   `qwen-smoke`. Both are required. The workflow has no default for either.
   The base URL is the one for the key's region (table below).
6. `ORBIT_SMOKE_STUB_UPSTREAM` is unset on the repository and on
   `qwen-smoke`. That flag belongs only to the pull-request stub job in
   `ci.yml`.

| Key region | `ORBIT_MODEL_BASE_URL` |
| --- | --- |
| Beijing (China mainland) | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| International (Singapore) | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |

DashScope keys are region-bound. A key sent to the other region's host
returns HTTP 401, and the smoke fails the case as `auth`
(`updateErrorCode: auth`). Ops set the base URL and the model name; the
workflow does not choose them. `qwen-flash` is one model that accepts
`max_tokens` and has thinking off. It is an example, not a default.

## Guards

| Guard | Where | Protects | When it fires |
| --- | --- | --- | --- |
| Trigger filter | workflow `on:` | 2 | Only a manual `workflow_dispatch`. There is no `pull_request` trigger. `pull_request_target` is never used, so fork code never runs with the environment's secrets. |
| Environment `qwen-smoke` | job `environment:` | 1, 2 | The key is an environment secret, not a repository or organization secret. Deployment branches are `main` only, and a `qwen-smoke` approver must approve. A dispatch from any other ref is rejected before any step. A run that nobody approves keeps waiting. A run an approver rejects fails. In every one of those cases no step runs and the key is not provided. This is the control that keeps the key on `main`. |
| Main-only ref | first step, `GITHUB_REF` | 2 | If `GITHUB_REF` is not `refs/heads/main`, the step prints `::error` and exits 1. No later step runs, and none of them receive the key. Defense-in-depth only, and only while this step is still in the workflow file that GitHub is running. A dispatch from branch X runs branch X's file, so a push can delete this step. Losing the environment's deployment-branch rule does **not** mean other branches still cannot get the key. |
| Model config | second step, before the key | 3, 4 | No defaults. `ORBIT_MODEL_NAME` empty, `ORBIT_MODEL_BASE_URL` empty, or the base URL not allowlisted: the step prints `::error` and exits 1. The same step exits 1 when `ORBIT_SMOKE_STUB_UPSTREAM` is set in the environment or the process. This step does not receive the key. |
| Base URL allowlist | config step, harness, and gate | 1, 3 | HTTPS only, exact host `dashscope.aliyuncs.com` or `dashscope-intl.aliyuncs.com`, no userinfo, port empty or 443. `http` and any other host are rejected. The gate checks again and returns HTTP 400 without forwarding. |
| Empty-key check | third step | 4 | `ORBIT_MODEL_API_KEY` empty or whitespace: the step prints `::error` naming the secret and environment and exits 1. The run is red; no later step runs. |
| Key masking | third step | 1 | `::add-mask::` for the key and each half, after the ref and config checks and before anything else that could see the key. The key and its length are never printed. |
| Key scope | step `env:` | 1 | The secret is set only on the steps that need it (check, harness, scan), not job-wide. The ref check, the config check, `uv sync`, and checkout never see it. |
| Exact ref | checkout step | 4 | `SMOKE_SHA` is `github.sha` only. The ref check has already required `refs/heads/main`. The step fails if `git rev-parse HEAD` differs. The SHA is the report's `gitSha`. |
| Budget gate | harness | 3 | The worker's `ORBIT_MODEL_BASE_URL` is a local gate that forwards to the real endpoint. It forwards at most 2 requests per run and answers any further request with HTTP 400 (non-retryable `config`). OpenAI-client retries are 0 (`ORBIT_MODEL_MAX_RETRIES=0`). A refused request fails its case: each case expects exactly 1 forwarded request and 0 refused. |
| Request check before forward | gate | 3 | Before the gate opens an upstream connection it requires `model` equal to `ORBIT_MODEL_NAME` and `max_tokens` equal to 64. A mismatch is HTTP 400 and is not forwarded. The check is not deferred until after the provider responds. |
| Output cap | worker | 3 | `ORBIT_MODEL_MAX_TOKENS=64`, sent as `max_tokens`. The per-request timeout is `ORBIT_MODEL_TIMEOUT_SECONDS=60` (a US-hosted runner calling Beijing needs more than 30s). |
| Harness start check | harness | 4 | Missing key, model name, or Postgres URL; a base URL that is missing or not allowlisted; a key under 16 characters (its halves would match unrelated text); or a database that already holds agent state (a stale row would replay a cached turn with no model call): exit 2 before anything starts. No report is written, so the scan step fails too. |
| Stub upstream flag | `ci.yml` only | 1, 3 | `ORBIT_SMOKE_STUB_UPSTREAM=1` is set only on the pull-request job `real-model-smoke-stub`. That job has no secrets and no GitHub environment. The flag does not open the allowlist to an arbitrary host. The harness ignores `ORBIT_MODEL_BASE_URL` and forwards only to an in-process loopback stub (SSE and non-streamed). The real workflow's config step fails if the flag is set, before the key step. |
| Per-case leak check | harness | 1 | After each case: stop the pair, read the worker's and orbit-workflows' stdout and stderr, fetch the workflow history. Search logs, every Failure in the history, every history event, and every posted event for the key and each half (raw, JSON-escaped, JSON-escaped twice) and for the prompt and its canary (same three forms; not in events or history, where the prompt is the turn's input). Any hit fails the case. |
| Zero-byte capture | harness and scan | 4 | Every process writes a startup line to stdout and to stderr. A 0-byte log file fails its case and the scan. |
| Report and log scan | scan step | 1 | `e2e_real_model_smoke.py scan` reads the report and every file under the log directory (including the harness's own output) for the same key and prompt forms plus key/token formats and database URLs. It also fails when a log the report names is missing or 0 bytes. |
| gitleaks | scan step | 1 | Pinned `v8.30.1`, checksum verified, over the report and log directories. Any finding fails the job. |
| Withheld artifacts | upload steps | 1 | The scan results (rule names and file names only, never values) are always uploaded. The report and logs are uploaded only when both scans pass, so a leaked key never lands in a public artifact. |
| Concurrency | workflow | 3 | One run at a time (`concurrency: real-model-smoke`, not cancelled). |

## Failure modes and what the run shows

| Situation | Result |
| --- | --- |
| Dispatched from a ref other than `main` | The `qwen-smoke` deployment-branch rule rejects the run before any step, so the key is never available. That rule is the control. The `GITHUB_REF` step, when it is still present in the file GitHub runs, is also red and exits 1 before any step receives the key. A branch that deleted the step is stopped only by the environment rule. |
| Nobody approves the `qwen-smoke` deployment | The job waits. No step runs, and no model call is made. |
| A `qwen-smoke` approver rejects the deployment | The job fails before any step. The key is not provided, and no model call is made. |
| `ORBIT_MODEL_BASE_URL` or `ORBIT_MODEL_NAME` unset, or the base URL is not allowlisted (`http`, a non-DashScope host, userinfo, or a port other than 443) | Config step red, exit 1, before any step receives the key. If the harness is started anyway it exits 2, and the gate returns HTTP 400 without forwarding. |
| `ORBIT_SMOKE_STUB_UPSTREAM` set where the real workflow can see it | Config step red, exit 1, before the key step. The real run never switches to the stub. |
| Key stored as a repository or organization secret | Out of band. The workflow cannot see that mistake. The ops checklist forbids it: the key exists only as the `qwen-smoke` environment secret. |
| Secret not created, or empty | Key step red: `ORBIT_MODEL_API_KEY is empty in environment qwen-smoke`. Nothing else runs. |
| Environment `qwen-smoke` missing | GitHub creates it on first use with no protection rules and no secret, so the empty-key check fires. Recreate it from the ops checklist before adding the key. |
| Wrong key (401/403), or a key whose region does not match the base URL (401) | Case red, `updateErrorCode: auth`, gate `providerStatuses: [401]` (or 403). Report and logs are uploaded if the scans pass. |
| Wrong model name (400/404) | Case red, `updateErrorCode: config`. The gate also refuses, before forwarding, a request whose `model` is not `ORBIT_MODEL_NAME` or whose `max_tokens` is not 64. |
| Rate limited (429), provider 5xx, timeout | Case red with `rate_limited`, `provider_error`, or `timeout`. No retry. The per-request timeout is 60s. |
| Spending cap reached | Usually 403 or 429 from the provider: case red with `auth` or `rate_limited`. |
| Model answers with a tool call or empty text | Case red: `updateStatus` is not `completed`, or no `assistant.message`. |
| Response has null usage | Case red: the worker fails the turn as `provider_error` (S-RM-3). |
| Successful turn | `updateStatus` is `completed` and `updateErrorCode` is `""` (an empty string, not JSON null). See below. |
| Stream body has fewer than 2 content chunks | E-RM-2 red: `providerContentChunksAtLeastTwo` is false. The count is measured at the gate, on provider content chunks, before the worker coalesces them. |
| Worker posts fewer than 2 `assistant.delta` events | Not a failure. The worker coalesces deltas every 100 ms or 200 characters, and 64 tokens is about 130 characters, so one delta is normal. The count is stored under `observed.assistantDeltas` and is not asserted. The case is red only when the deltas that did arrive are out of order, are not one contiguous `seq` block, or do not join to the final `assistant.message`. |
| Temporal retries the turn Activity | Second request refused by the gate; the case is red (`providerRequestsRefused: 1`). |
| Key or prompt in a log, the report, or a Failure | Case red and scan red. Report and logs are not uploaded; the scan result names the file and rule only. |
| A process wrote nothing | Case red (`zeroByteLogs`), scan red. |
| Harness crashes | Run step red; the scan step still runs over whatever logs exist and fails on the missing report. |
| Two workers call `ensure_schema` on an empty Postgres | Known product bug, not fixed in this PR. One worker can crash with `asyncpg.exceptions.UniqueViolationError` (SQLSTATE 23505) from `PostgresStateStore.ensure_schema`. The harness creates the schema before it starts workers. Tracked in [mindreon/orbit-runtime#8](https://github.com/mindreon/orbit-runtime/issues/8). |

## `updateErrorCode` on success

`TurnResult.error_code` is `None` when the turn completes. The `runTurn` and
`decide` Updates do not pass that null through. `_turn_payload` in
`orbit_orch.workflows` sets `errorCode` to `result.error_code or ""`, so a
successful Update has `errorCode: ""` (empty string). A failed Update has one
of `timeout`, `auth`, `rate_limited`, `provider_error`, or `config`. The smoke
asserts `updateErrorCode` is `""` on a pass. It is never JSON null on the
Update.

## Determinism exception

The A1/E-LOG report must be byte-identical across reruns. This report is
not: real model output, token counts, and delta counts change between runs.
It is structurally deterministic instead: the same keys, cases, steps, and
`expected` values on every run for one commit; `actual` equals `expected`
when the run passes.

`observed` on each case is not asserted and is not part of a structural
compare. Its fields are:

| Field | What varies |
| --- | --- |
| `usageEvent` | `inputTokens`, `outputTokens`, `latencyMs` |
| `providerUsage` | provider `prompt_tokens`, `completion_tokens`, `total_tokens` |
| `providerContentChunks` | exact chunk count (the assertion is only `>= 2`) |
| `assistantDeltas` | how many `assistant.delta` events the worker posted |
| `assistantMessage` | `bytes` and `sha256` of the reply |
| `historyFailureEntries` | Failure messages seen in history |
| `capturedBytes` | stdout and stderr sizes |

Top-level `usage` repeats the provider token counts outside `observed`. The
real workflow does not compare reruns, so those counts are allowed to change.
The pull-request stub job runs the harness twice against the in-process stub
and `cmp`s the two reports after each case's `observed` object is removed.
The stub returns fixed token counts, so top-level `usage` is part of that
compare. `gitSha` is `git rev-parse HEAD` of the checked-out PR head.

## Pull-request stub job

`.github/workflows/ci.yml` job `real-model-smoke-stub` runs on pull requests.
It has `permissions: contents: read`, no `environment:`, and no secrets. It
checks out the PR head (`github.event.pull_request.head.sha`) with
`persist-credentials: false`, sets `ORBIT_SMOKE_STUB_UPSTREAM=1`, and runs
`scripts/e2e_real_model_smoke.py` twice through the same gate, Temporal,
orbit-workflows, orbit-worker, and Postgres. The report is uploaded with
`if: always()`. A pinned gitleaks scan covers the reports and the logs.
Actions are pinned by SHA. `postgres:16-alpine` is pinned by digest.

The allowlist on the real workflow stays as written above. The stub flag is
not an allowlist bypass: the real workflow fails the config step when the
flag is set, and the harness, when the flag is set, never reads
`ORBIT_MODEL_BASE_URL` as an upstream.
