# Decide delivery idempotency (C35 §2.4, runtime)

Source of truth: `docs/contracts/orbit-contract-v2.md` in mindreon/orbit-control,
commit `53ee38c053b48a58e7035940bbf050510554eac2` (C35, on `main`). This note
is the failure-mode list for the runtime half of §2.4.

Scope is the room workflow, the child `resolve` signal, the contract models,
and an end-to-end suite. There is no backward-compatible read of old decide
payloads or old workflow histories. `DecideRequest.resume_turn_id` is gone.
`DecideRequest` is `extra="forbid"`. A payload that still sends `resumeTurnId`,
or any other unknown field, is rejected before the handler runs.

Do not deploy this runtime until orbit-control#16 has dropped `resumeTurnId`,
switched delivery to `approvalRequestId`, and merged. Until that merge, this
change stays unreleased.

## What “delivered” means

Delivered means the room workflow **accepted** the `decide` Update. It does
not mean the resume turn finished. The resume turn id is not taken from the
request. The workflow computes:

```text
resumeTurnId = "t-" + sha256("decide:" + approvalRequestId).hexdigest()[:24]
```

The digest is lowercase hex. Control’s Update id for a decide is
`approvalRequestId`.

## Handler order (F1)

§2.4.1 item 4. The handler does this before its first await:

1. If this `approvalRequestId` is already decided, do not execute again.
   `done` returns the stored `DecideOutcome`. `running` waits until it is
   `done`, then returns that same outcome (N9). A second Update id does not
   get a second resume turn.
2. Prune expired entries. Only `state=="done"` rows are pruned. A `running`
   entry is never pruned, even when its `decided_at` is older than the TTL.
3. If the id is not in the table and 1024 unexpired entries remain
   (`MAX_DECIDED_APPROVALS`, not configurable), do **not** write `running`
   and do **not** resume. Set `_fatal = "DECIDED_APPROVALS_LIMIT"` and fail
   the Update with `ApplicationError(type="DECIDED_APPROVALS_LIMIT", non_retryable=True)`.
4. Otherwise write `{state: "running", decided_at: workflow.now()}` and drop
   the id from `pending_approvals`.

5. Run the resume. A `finally` sets `state="done"` and stores the outcome.
   The `finally` also runs when the handler is cancelled or the activity
   fails, so a waiter cannot stay blocked on `running`. Cancel still
   propagates after `finally`.

Writing `running` before the cap check would make the new id “already in the
table”, and the cap check would let the 1025th decision through. The S-ID-8
order case is built so that wrong order passes the cap; the correct order
fails it.

## Cap failure

The main loop waits on `_fatal`. If a turn handler is still running, the loop
waits until `all_handlers_finished()` before it emits anything. It then, once:

1. Emits `room.failed` through the `ingestRoomEvent` activity (same ingest
   path as other worker events). The activity’s own retries apply. If those
   retries are exhausted, the workflow still fails.
2. Cancels every child workflow the room started.
3. Raises `ApplicationError(type="DECIDED_APPROVALS_LIMIT", non_retryable=True)`
   from `run`.

The event is emitted after the in-flight turn ends, and the event stream
contains exactly one `room.failed`. The failure body is:

```text
{code: "DECIDED_APPROVALS_LIMIT", message: "此任务的审批次数已达上限，无法继续。你可以查看记录，或新建任务继续工作。"}
```

That message is the §2.4.1 item 6 / §10.2 text, character for character.
`RoomFailure` carries both fields as constants. `decidedApprovalIds` still
returns the 1024 ids that were already stored. The 1025th id is not added.
Its tool runs zero times. After the event, the workflow execution is
**Failed**. The mock model’s request count does not increase after that.

### After `_fatal`, before the workflow ends

§2.4.1 item 6 (C35):

- The validator rejects an id that is not already in `decided_approvals`.
  The error is `DECIDED_APPROVALS_LIMIT` and `non_retryable`. This includes
  ids that are pending and ids the room has never seen.
- An id that is already `done` still returns the stored `DecideOutcome`.
  The tool is not run again.
- The main loop does not start another `runTurn`. A `runTurn` Update that
  arrives in this window fails with the same `DECIDED_APPROVALS_LIMIT`
  error and does not schedule the activity.

When the unexpired count reaches 512, the workflow logs one warning per run:

```text
decided_approvals_high_watermark{roomId=<id>,count=<n>}
```

The warning is not an event and is not shown to the product UI. Continue-as-new
starts a new run, which may log once more. Continue-as-new does not run after
`_fatal` is set.

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
2. `_fatal` is set: reject with `ApplicationError(type="DECIDED_APPROVALS_LIMIT", non_retryable=True)`.
3. Id is in `pending_approvals`: accept.
4. Otherwise reject with `ApplicationError(type="APPROVAL_UNKNOWN")`.

## Child `resolve` and the shared cap

The room and every child agent share one cap of 1024. The count lives only
in the room’s `decided_approvals`. A child decision is counted by the room,
with the same prune-then-cap-then-`running` order as item 4 and item 6.
The child’s own id set is a dedupe set. It does not allocate cap slots.

The room does not signal `resolve` for an id already in `decided_approvals`.
On the child, a repeated `resolve` for an id already seen, or an id the child
is not waiting on, is ignored. The workflow logs:

```text
duplicate resolve ignored approvalRequestId=<id>
```

The tool runs once (S-ID-11).

When the room table already holds 1024 unexpired rows, the child’s next new
decision sets `_fatal` and follows the cap-failure section above. The whole
task fails with the same code and the same message, whether the 1025th
decision came from the main agent or from a child. The room cancels every
child. The child’s tool for that 1025th id does not run.

If the child’s dedupe set itself reaches 1024 (it should not, because the
room count fails the task first), the child workflow fails with
`ApplicationError(type="DECIDED_APPROVALS_LIMIT", non_retryable=True)`.
The room treats that child failure as the same cap: it sets `_fatal` and
follows item 6. The child must not log a warning and return false and keep
running.

## Failure modes

| # | Condition | Result |
| --- | --- | --- |
| F1 | Same Update id delivered twice in one run | Temporal runs the handler once. Second call returns the first `DecideOutcome`. |
| F2 | Different Update id, same approval, `state=done` | Stored outcome returned. `resolveApproval` is not scheduled again. One `turn.started` for that resume turn id. |
| F3 | Different Update id, same approval, `state=running` (N9) | Caller waits. It gets the first outcome. The tool runs once. |
| F4 | Id in neither table, and `_fatal` is not set | Validator rejects `APPROVAL_UNKNOWN`. Table unchanged. |
| F5 | Expired `done` entries, then a new id, still under 1024 after prune | Expired rows dropped. New decision runs. Cap is not hit. |
| F6 | Prune would drop a `running` row whose `decided_at` is old, and dropping it would make the new id fit | `running` stays. The new id hits the cap and fails. |
| F7 | 1024 unexpired rows, new id | Update fails `DECIDED_APPROVALS_LIMIT`. New id is absent. Tool runs 0 times. |
| F8 | F7, workflow loop | `room.failed` ingested once, with the fixed message. Children cancelled. Execution status Failed. |
| F9 | `ingestRoomEvent` keeps failing | Activity retries, then the workflow fails anyway. |
| F10 | Unexpired count reaches 512 | One WARN per run, text above. Not an OrbitEvent. |
| F11 | Resume activity raises, or the handler is cancelled | `finally` stores `done` so N9 waiters unblock. Cancel still propagates. A later `decideOutcome` query returns that stored outcome. |
| F12 | Continue-as-new after a decide | New run’s table is the unexpired rows only. Same Update id returns the first outcome and does not resume again. |
| F13 | Second `resolve` signal on a child | Warning once. Tool once. The child’s set did not allocate a second cap slot. |
| F14 | Extra field on `DecideRequest`, including `resumeTurnId` | Rejected by `extra="forbid"` before the handler. The id is not stored. The tool does not run. |
| F15 | `ORBIT_DECIDED_APPROVAL_TTL_S` below 60 in production | `decideConfig.ttlS` is 60. |
| F16 | In-flight workflow started on older code | Not replay-compatible. No patch branch keeps the old decide shape. |
| F17 | Room table holds 1024. A child triggers the 1025th decision | Same fatal path as F8. `room.failed` message matches §10.2. Every child is cancelled. Execution Failed. Mock model request count does not increase. |
| F18 | Main and child decisions together fill 1024. Either side triggers the 1025th | Both runs fail with `DECIDED_APPROVALS_LIMIT`. The code does not depend on which side went over. |
| F19 | `_fatal` is set and a turn handler is still running | A new id’s `decide` is rejected `DECIDED_APPROVALS_LIMIT`, `non_retryable`. A new `runTurn` is rejected the same way and does not call the model. A `done` id returns the stored outcome. |
| F20 | F19, then the in-flight turn ends | `room.failed` is ingested after that turn’s events, and the stream contains exactly one. |

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
| S-ID-13 | Resume activity longer than `ORBIT_DECISION_DELIVERY_TIMEOUT` (harness uses 1s vs the slow worker’s sleep). The Update was accepted, the turn result is stored, the room leaves `awaiting_approval`. |
| S-ID-14 | F17. 1024 room rows already stored. A child agent’s next decision is the 1025th on that room table. One `room.failed` with the fixed message, children cancelled, execution Failed, mock `usage` count unchanged. |
| S-ID-15 | F18. Two rooms. Each starts with 512 main rows and 512 child rows. One room’s 1025th is a main `decide`. The other’s is a child decision. Both errors are `DECIDED_APPROVALS_LIMIT`. |
| S-ID-16 | F19. `_fatal` is set while a resume activity is still inside its e2e sleep. New `decide` and new `runTurn` are rejected non-retryable. The `done` id returns the stored outcome. The rejected `runTurn` adds no `usage` event. |
| S-ID-17 | F20. Same window as S-ID-16. After the in-flight turn’s `tool.result`, the stream has exactly one `room.failed`. |
| F11 | Cancel the workflow while the resume activity is in its e2e sleep. `decideOutcome` on the closed workflow returns the stored `done` outcome. Execution status is Canceled. |
| F14 | `decide` payload includes `resumeTurnId`. The update is rejected. The approval id is not stored. `gated_echo` does not run. |

S-ID-3 through S-ID-6, S-ID-9, S-ID-12, and S-ID-18 are orbit-control #16
phase 2 (delivery_state transitions, HTTP 202/503/409, SQL `P0001`, F4’s
zero-row HTTP answers). This repo cannot prove them. S-ID-13’s “HTTP is not
502” line is the same control response; the runtime suite proves only the
workflow half.

`scripts/e2e_state_key.py` came in with the state-key work on main. Its
`decide` calls no longer send `resumeTurnId`. The resume turn id is the
derived value, and the Update result is a `DecideOutcome`.
