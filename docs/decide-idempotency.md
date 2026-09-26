# Decide delivery idempotency (C34 §2.4, runtime)

Source of truth: `docs/contracts/orbit-contract-v2.md` in mindreon/orbit-control,
commit `79b185a4` (PR #18, merge `17baead`). This note was written before the
runtime code. It is the failure-mode list for the runtime half of §2.4.

Scope is the room workflow, the child `resolve` signal, the contract models,
and an end-to-end suite. There is no backward-compatible read of old decide
payloads or old workflow histories. `DecideRequest.resume_turn_id` is gone.
A payload that still sends `resumeTurnId` is rejected (`extra="forbid"`).

## What “delivered” means

Delivered means the room workflow **accepted** the `decide` Update. It does
not mean the resume turn finished. The resume turn id is not taken from the
request. The workflow computes:

```text
resumeTurnId = "t-" + sha256("decide:" + approvalRequestId).hexdigest()[:24]
```

The digest is lowercase hex. Control’s Update id for a decide is
`approvalRequestId`.

## Handler order (QA F1)

§2.4.1 item 4 says to write the `running` entry at the start of the handler,
before any await. That sentence is a known error and will be corrected in the
contract. The order implemented here is:

1. If this `approvalRequestId` is already decided, do not execute again.
   `done` returns the stored `DecideOutcome`. `running` waits until it is
   `done`, then returns that same outcome (N9). A second Update id does not
   get a second resume turn.
2. Prune expired entries. A `running` entry is never pruned, even when its
   `decided_at` is older than the TTL.
3. If the id is not in the table and 1024 unexpired entries remain
   (`MAX_DECIDED_APPROVALS`, not configurable), do **not** write `running`
   and do **not** resume. Set `_fatal = "DECIDED_APPROVALS_LIMIT"` and fail
   the Update with `ApplicationError(type="DECIDED_APPROVALS_LIMIT", non_retryable=True)`.
4. Otherwise write `{state: "running", decided_at: workflow.now()}` and drop
   the id from `pending_approvals`. Steps 2–4 happen before the handler’s
   first await.
5. Run the resume. A `finally` sets `state="done"` and stores the outcome.
   The `finally` also runs when the handler is cancelled or the activity
   fails, so a waiter cannot stay blocked on `running`.

Writing `running` before the cap check would make the new id “already in the
table”, and the cap check would let the 1025th decision through. The S-ID-8
order case is built so that wrong order passes the cap; the correct order
fails it.

## Cap failure

The main loop waits on `_fatal`. Before the workflow execution becomes
Failed it:

1. Emits `room.failed` through the `ingestRoomEvent` activity (same ingest
   path as other worker events). The activity’s own retries apply. If those
   retries are exhausted, the workflow still fails.
2. Cancels child workflows.
3. Raises `ApplicationError(type="DECIDED_APPROVALS_LIMIT", non_retryable=True)`
   from `run`.

The event failure is `{code: "DECIDED_APPROVALS_LIMIT", message: <fixed>}`.
The fixed message is the §10.2 / §13 item 10 text, not the different sentence
in §2.4.1 item 6 (that sentence is also wrong and will be corrected):

```text
此任务的审批次数已达上限，无法继续。你可以查看记录，或新建任务继续工作。
```

`RoomFailure` in `schema/RoomFailure.json` carries this code and message as
constants so web can import them from generated types. `decidedApprovalIds`
still returns the 1024 ids that were already stored. The 1025th id is not
added. Its tool runs zero times.

When the unexpired count reaches 512, the workflow logs one warning per run:

```text
decided_approvals_high_watermark{roomId=<id>,count=<n>}
```

The warning is not an event and is not shown to the product UI. Continue-as-new
starts a new run, which may log once more.

## Continue-as-new

`RoomWorkflowInput.carryOver` is a `RoomCarryOver`. It includes
`decidedApprovals: list[DecidedApproval]`. Continue-as-new runs from the main
coroutine only when `all_handlers_finished()` is true, so a `running` entry
is not carried. Expired entries are pruned at that point. Child workflows
still running postpone continue-as-new.

The carried rows are loaded in `@workflow.init`, which runs before any
`decide` handler in that workflow task. `run` does not load them again.
A decide that is delivered in the new run's first task still sees the
table. Loading them only inside `run` rejects that decide as
`APPROVAL_UNKNOWN`, because the update task can run before `run`.

`ORBIT_CAN_TURN_THRESHOLD` (default 200) counts completed decide resumes.
`is_continue_as_new_suggested()` also triggers it. S-ID-7 sets the threshold
to 1.

## Queries

| Query | Returns |
| --- | --- |
| `decideOutcome(approvalRequestId)` | The stored `DecideOutcome` when `state=="done"`, otherwise null |
| `decidedApprovalIds()` | Every id still in the table, including `running` |
| `decideConfig()` | `{ttlS, maxDecided}` |

`ttlS` is `ORBIT_DECIDED_APPROVAL_TTL_S` (default 86400). Values below 60
are raised to 60 unless `ORBIT_E2E=1` (S-ID-10 uses 5). `maxDecided` is 1024.
Age is `workflow.now() - decided_at`. An entry is expired when that age is
greater than `ttlS`.

## Validator

The validator does not write state. Order:

1. Id is in `decided_approvals`: accept. The handler returns the first outcome.
2. Id is in `pending_approvals`: accept.
3. Otherwise reject with `ApplicationError(type="APPROVAL_UNKNOWN")`.

## Child `resolve`

The room must not signal a child for an id already in `decided_approvals`.
The child keeps its own processed-id set with the same TTL, the same 1024
cap, and the same prune-then-cap-then-running order. A repeated `resolve`
for an id already running or done is ignored. The workflow logs:

```text
duplicate resolve ignored approvalRequestId=<id>
```

An id the child is not waiting on is ignored with the same warning. The tool
runs once. At the child cap the signal is ignored and the tool does not run;
a signal cannot fail the room Update. The room Update is what fails the room
when the room table is full.

## Failure modes

| # | Condition | Result |
| --- | --- | --- |
| F1 | Same Update id delivered twice in one run | Temporal runs the handler once. Second call returns the first `DecideOutcome`. |
| F2 | Different Update id, same approval, `state=done` | Stored outcome returned. `resolveApproval` is not scheduled again. One `turn.started` for that resume turn id. |
| F3 | Different Update id, same approval, `state=running` (N9) | Caller waits. It gets the first outcome. The tool runs once. |
| F4 | Id in neither table | Validator rejects `APPROVAL_UNKNOWN`. Table unchanged. |
| F5 | Expired `done` entries, then a new id, still under 1024 after prune | Expired rows dropped. New decision runs. Cap is not hit. |
| F6 | Prune would drop a `running` row whose `decided_at` is old, and dropping it would make the new id fit | `running` stays. The new id hits the cap and fails. |
| F7 | 1024 unexpired rows, new id | Update fails `DECIDED_APPROVALS_LIMIT`. New id is absent. Tool runs 0 times. |
| F8 | F7, workflow loop | `room.failed` ingested with the fixed message. Children cancelled. Execution status Failed. |
| F9 | `ingestRoomEvent` keeps failing | Activity retries, then the workflow fails anyway. |
| F10 | Unexpired count reaches 512 | One WARN per run, text above. Not an OrbitEvent. |
| F11 | Resume activity raises, or the handler is cancelled | `finally` stores `done` so N9 waiters unblock. Cancel still propagates. |
| F12 | Continue-as-new after a decide | New run’s table is the unexpired rows only. Same Update id returns the first outcome and does not resume again. |
| F13 | Second `resolve` signal on a child | Warning once. Tool once. |
| F14 | `resumeTurnId` on the request | Validation error. The field does not exist. |
| F15 | `ORBIT_DECIDED_APPROVAL_TTL_S` below 60 in production | `decideConfig.ttlS` is 60. |
| F16 | In-flight workflow started on older code | Not replay-compatible. No patch branch keeps the old decide shape. |

## End-to-end

`scripts/e2e_decide_idempotency.py` reuses the `scripts/e2e_a1_events.py`
harness: real Temporal dev server, `orbit-orch`, `orbit-worker`, mock model,
real Postgres (`ORBIT_TEST_POSTGRES_URL` / `ORBIT_STATE_STORE_URL`). No new
unit tests. The report commit is `git rev-parse HEAD` with no override. Two
runs on one commit must be byte-identical.

| Case | Proves |
| --- | --- |
| S-ID-2 | F1, F2, and the N9 wait (F3) |
| S-ID-7 | F12, with `ORBIT_CAN_TURN_THRESHOLD=1` |
| S-ID-8 | F7, F8, F10. The 1025th id must fail. A wrong “write running, then check the cap” order would accept it. |
| S-ID-8-order | F6. One `running` row is older than the TTL and 1023 `done` rows are fresh. Pruning `running` would pass the cap; the new id must fail, and the `running` id must remain. |
| S-ID-10 | F5. `ORBIT_E2E=1`, TTL 5 seconds, 1024 rows already older than 5 seconds. The new id succeeds. After continue-as-new only the new id remains. |
| S-ID-11 | F13. Two `resolve` signals, one tool execution, one warning. |
| S-ID-13 | Resume activity longer than `ORBIT_DECISION_DELIVERY_TIMEOUT` (harness uses 1s vs a 2s activity). The Update was accepted, the turn result is stored, the room leaves `awaiting_approval`. |

S-ID-3 through S-ID-6, S-ID-9, and S-ID-12 are orbit-control #16 phase 2
(delivery_state transitions, HTTP 202/503/409, SQL `P0001`). This repo cannot
prove them. S-ID-13’s “HTTP is not 502” line is the same control response;
the runtime suite proves only the workflow half.

`scripts/e2e_state_key.py` is not on `main`. It lives on open orbit-runtime
#6. This branch does not copy that PR. When #6 merges, its `decide` call must
stop sending `resumeTurnId` or `extra="forbid"` will reject it.
