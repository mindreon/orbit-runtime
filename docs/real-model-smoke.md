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

## What must never happen

1. The key reaches a process log, the report, a Temporal failure, an event,
   or an uploaded artifact.
2. The key is exposed to code from a fork, or to code from any ref other than
   `main`.
3. A run makes more than 2 model calls.
4. A run passes without a key, without real model output, or with a log
   capture that silently recorded nothing.

## Guards

| Guard | Where | Protects | When it fires |
| --- | --- | --- | --- |
| Trigger filter | workflow `on:` | 2 | Only a manual `workflow_dispatch`. There is no `pull_request` trigger. `pull_request_target` is never used, so fork code never runs with the environment's secrets. |
| Environment `qwen-smoke` | job `environment:` | 1, 2 | The key is an environment secret. Deployment branches are `main` only, and a `qwen-smoke` approver must approve. A dispatch from any other ref is rejected before any step. A run that nobody approves keeps waiting. A run an approver rejects fails. In every one of those cases no step runs and the key is not provided. |
| Main-only ref | first step, `GITHUB_REF` | 2 | If `GITHUB_REF` is not `refs/heads/main`, the step prints `::error` and exits 1. No later step runs, and none of them receive the key. This is the backstop when `qwen-smoke` is deleted and recreated without its deployment-branch rule: code from any other branch still cannot obtain the key. |
| Empty-key check | second step | 4 | `ORBIT_MODEL_API_KEY` empty or whitespace: the step prints `::error` naming the secret and environment and exits 1. The run is red; no later step runs. |
| Key masking | second step | 1 | `::add-mask::` for the key and each half, after the ref check and before anything else that could see the key. The key and its length are never printed. |
| Key scope | step `env:` | 1 | The secret is set only on the steps that need it (check, harness, scan), not job-wide. The ref check, `uv sync`, and checkout never see it. |
| Exact ref | checkout step | 4 | `SMOKE_SHA` is `github.sha` only. The ref check has already required `refs/heads/main`. The step fails if `git rev-parse HEAD` differs. The SHA is the report's `gitSha`. |
| Budget gate | harness | 3 | The worker's `ORBIT_MODEL_BASE_URL` is a local gate that forwards to the real endpoint. It forwards at most 2 requests per run and answers any further request with HTTP 400 (non-retryable `config`). OpenAI-client retries are 0 (`ORBIT_MODEL_MAX_RETRIES=0`). A refused request fails its case: each case expects exactly 1 forwarded request and 0 refused. |
| Output cap | worker | 3 | `ORBIT_MODEL_MAX_TOKENS=64`. The gate records the `max_tokens` in each forwarded request; a request without it fails the case. |
| Harness start check | harness | 4 | Missing key, model name, base URL, or Postgres URL, a key under 16 characters (its halves would match unrelated text), or a database that already holds agent state (a stale row would replay a cached turn with no model call): exit 2 before anything starts. No report is written, so the scan step fails too. |
| Per-case leak check | harness | 1 | After each case: stop the pair, read the worker's and orbit-workflows' stdout and stderr, fetch the workflow history. Search logs, every Failure in the history, every history event, and every posted event for the key and each half (raw, JSON-escaped, JSON-escaped twice) and for the prompt and its canary (same three forms; not in events or history, where the prompt is the turn's input). Any hit fails the case. |
| Zero-byte capture | harness and scan | 4 | Every process writes a startup line to stdout and to stderr. A 0-byte log file fails its case and the scan. |
| Report and log scan | scan step | 1 | `e2e_real_model_smoke.py scan` reads the report and every file under the log directory (including the harness's own output) for the same key and prompt forms plus key/token formats and database URLs. It also fails when a log the report names is missing or 0 bytes. |
| gitleaks | scan step | 1 | Pinned `v8.30.1`, checksum verified, over the report and log directories. Any finding fails the job. |
| Withheld artifacts | upload steps | 1 | The scan results (rule names and file names only, never values) are always uploaded. The report and logs are uploaded only when both scans pass, so a leaked key never lands in a public artifact. |
| Concurrency | workflow | 3 | One run at a time (`concurrency: real-model-smoke`, not cancelled). |

## Failure modes and what the run shows

| Situation | Result |
| --- | --- |
| Dispatched from a ref other than `main` | The `qwen-smoke` deployment-branch rule rejects the run before any step, so the key is never available. If that environment was deleted and recreated without the branch rule, the `GITHUB_REF` step is red and exits 1 before any step receives the key. No model call either way. |
| Nobody approves the `qwen-smoke` deployment | The job waits. No step runs, and no model call is made. |
| A `qwen-smoke` approver rejects the deployment | The job fails before any step. The key is not provided, and no model call is made. |
| Secret not created, or empty | Second step red: `ORBIT_MODEL_API_KEY is empty in environment qwen-smoke`. Nothing else runs. |
| Environment `qwen-smoke` missing | GitHub creates it on first use with no secret, so the empty-key check fires. |
| Wrong key (401/403) | Case red, `updateErrorCode: auth`, gate `providerStatuses: [401]`. Report and logs are uploaded if the scans pass. |
| Wrong model name (400/404) | Case red, `updateErrorCode: config`. |
| Rate limited (429), provider 5xx, timeout | Case red with `rate_limited`, `provider_error`, or `timeout`. No retry. |
| Spending cap reached | Usually 403 or 429 from the provider: case red with `auth` or `rate_limited`. |
| Model answers with a tool call or empty text | Case red: `updateStatus` is not `completed`, or no `assistant.message`. |
| Response has null usage | Case red: the worker fails the turn as `provider_error` (S-RM-3). |
| Stream arrives in one chunk | E-RM-2 red: `providerContentChunksAtLeastTwo` or `assistantDeltasAtLeastTwo` is false. |
| Temporal retries the turn Activity | Second request refused by the gate; the case is red (`providerRequestsRefused: 1`). |
| Key or prompt in a log, the report, or a Failure | Case red and scan red. Report and logs are not uploaded; the scan result names the file and rule only. |
| A process wrote nothing | Case red (`zeroByteLogs`), scan red. |
| Harness crashes | Run step red; the scan step still runs over whatever logs exist and fails on the missing report. |

## Determinism exception

The A1/E-LOG report must be byte-identical across reruns. This report is
not: real model output, token counts, and delta counts change between runs.
It is structurally deterministic instead: the same keys, cases, steps, and
`expected` values on every run for one commit; `actual` equals `expected`
when the run passes. Values that vary (token counts, reply size and hash,
delta count, log byte counts) live under `observed` and `usage`, which are
never compared.
