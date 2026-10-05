"""What the task and attempt workflow modules share: timeouts, limits, the versioning switch and pure helpers.

Everything here is deterministic. `versioning_behavior` reads the process's versioning switch once, when a workflow
module is imported to build its `@workflow.defn`; it is never called while a workflow runs.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy, VersioningBehavior

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.v3 import Budget

    from orbit_orch.settings import versioning_settings

RETRY = RetryPolicy(maximum_attempts=3)
IO_TIMEOUT = timedelta(minutes=2)
AGENT_TIMEOUT = timedelta(hours=1)
# Restoring the snapshot and starting the sandbox come on top of a command's own timeout.
COMMAND_SETUP_S = 300
HEARTBEAT = timedelta(seconds=30)
OPERATOR_HELD = frozenset({"PAUSED", "PAUSED_NEEDS_REVIEW", "TAKEN_OVER"})
MAX_UPDATES_BEFORE_CAN = 1000
MAX_COMPLETIONS_BEFORE_CAN = 50
# An attempt that ends `completed` goes through the node's completion checks before the node is frozen, and an attempt
# that was rejected or replaced cannot complete the node by reporting late (17 G15).
VERIFY_FINISHED_ATTEMPTS = "task-attempt-completion-verification"
# A task is a conversation: when every node is done it rests instead of ending, and the next message starts a follow-up
# that carries on the agent's own session. Histories from before this ended the workflow there.
SESSION_STAYS_OPEN = "task-session-stays-open"


# Each of these ids guards one behaviour change, so histories recorded before it replay as they ran. A new run takes
# every one of them.
#
# A failed attempt counts against its node and retries with a backoff up to `retry.max_attempts`; a retry carries on the
# agent session of the attempt before it and is told why that one was rejected (04 §2). Then the node is BLOCKED.
RETRY_POLICY = "task-retry-policy"
# An interrupted or stopped attempt is replaced by one that carries on its agent session, with the new messages as input.
INTERRUPT_CONTINUES_SESSION = "task-interrupt-continues-session"
# A node keeps who created it through every status change, so an agent can still change its own unstarted node.
NODE_KEEPS_AUTHOR = "task-node-keeps-created-by"
# A wait node's timer is a deadline in the carried state, and is armed again after a Continue-As-New.
WAIT_TIMER_CARRY = "task-wait-timer-carry"
# A cancelled attempt's node is held until the attempt reports its own end, not for a fixed second.
CANCEL_WAITS_FOR_CHILD = "task-cancel-waits-for-child"
# The pending approvals of an attempt that ended are CANCELLED.
APPROVALS_CANCELLED = "task-approvals-cancelled-with-attempt"
# A message leaves the task inbox once an attempt has it.
INBOX_CONSUME = "task-inbox-consume"
# What a Continue-As-New carries is bounded: the dedup window, decided approvals, follow-up bookkeeping.
STATE_BOUNDS = "task-state-bounds"
# A follow-up node's goal is bounded; the attempt gets the whole message as its input message.
FOLLOW_UP_BOUNDED = "task-follow-up-bounded-goal"
# Completed, frozen nodes nothing depends on are compacted out of the live plan into an archive summary.
PLAN_COMPACTION = "task-plan-compaction"
# A follow-up that cannot be added to the plan is reported, and the task asks for a review.
FOLLOW_UP_REJECTION_VISIBLE = "task-follow-up-rejection-visible"
# The attempt reports the failure class and whether it can be retried, and reports an activity that gave up.
ATTEMPT_FAILURE_CLASS = "attempt-failure-class"
# What a message or a completion command asked for is done before the workflow continues as new, not lost with the run.
DRAIN_BEFORE_CONTINUE_AS_NEW = "task-drain-before-continue-as-new"
# The attempt hands its parent the messages it was given and did not get to when it ends.
ATTEMPT_RETURNS_MESSAGES = "attempt-returns-unconsumed-messages"
# The checkpoint commit is not cancelled with the attempt (it is abandoned: it is short and harmless to finish), so a cancel
# that comes while it is scheduled does not leave the attempt waiting on a cancelled activity.
ATTEMPT_COMMIT_ABANDON = "attempt-commit-abandon"
# An AttemptWorkflow continues as new when its history is suggested to be cut (04 §6).
ATTEMPT_CONTINUE_AS_NEW = "attempt-continue-as-new"

# What the plan asked for in 05 §4 and §6 and 04 §3 and what the design gave to people and agents, one id each so a history
# from before it replays as it ran.
#
# An attempt reserves its node's budget from the task's remaining budget when it starts and is settled by what it spent;
# a node the task cannot cover is not started. `grantBudget` resumes only a task that was held for budget.
BUDGET_ENFORCEMENT = "task-budget-enforcement"
# Ready nodes start in plan order, at most `max_concurrency` at a time; a node that has to wait does not keep the loop awake.
SCHEDULE_CONCURRENCY = "task-schedule-concurrency"
# An agent's plan changes, and `getPlan`, stay within what it may see (05 §6).
PLAN_VISIBILITY = "task-plan-visibility"
# Depth is the `parent_node_id` nesting (03 §3, default 1); the chain of dependencies is only bounded for safety.
PLAN_NESTING = "task-plan-nesting-depth"
# A profile switch is applied to the node's next attempt, directly or after an approval (11 §3).
PROFILE_SWITCH = "task-profile-switch"
# A takeover ends the attempts that are running, as a stop does; a handback is only for a task that was taken over.
TAKEOVER_INTERRUPTS = "task-takeover-interrupts"
# The attempt is handed what is left of its reserved budget for each turn, adds up what the turns spent and reports it.
ATTEMPT_BUDGET = "attempt-budget-enforcement"

# A `sop_stage` node is compiled into a subgraph of the plan when it becomes schedulable (one `agent_turn` node per step, approval
# nodes for `human_approval`) instead of running as one attempt on the SOP engine (`task_sop`). The old attempt path stays for
# histories that ran it.
SOP_EXPANSION = "task-sop-expansion"
# A node whose completion contract holds a `sop_verifier` verification is judged by an independent verifier agent
# (`verify_sop_step`); before this the verification passed without a check.
SOP_VERIFIER = "task-sop-verifier"

# Every approval node is a question to a person (`decideApproval` completes or blocks it), not only those an SOP made.
NODE_APPROVALS = "task-node-approvals"

# `attempt.finished` carries the attempt's structured output, bounded (`ATTEMPT_OUTPUT_EVENT_BYTES`).
ATTEMPT_OUTPUT_EVENT = "attempt-finished-carries-output"
ATTEMPT_OUTPUT_EVENT_BYTES = 16 * 1024

# What an event says about its node, plan version and attempt is enough for a projection to build its rows from the events
# alone (`node.status_changed`, `plan.version_committed`, `attempt.finished`).
EVENT_PAYLOADS = "task-event-payloads-v2"

# An attempt that has already closed cannot be signalled: the parent keeps the message for the next attempt instead of failing
# the whole task with "Unable to signal external workflow because it was not found".
SIGNAL_CLOSED_CHILD = "task-signal-closed-child-tolerated"

# A task that a person paused, took over or that waits for a review is not completed by the main loop because its nodes are
# done: it stays as it is until the person resumes it, and then completes.
HELD_STAYS_HELD = "task-held-status-kept"

# node.* events belong to the node: entity {kind: node, id}, with a version of their own (09 §1), not the task's.
NODE_ENTITY_VERSIONS = "task-node-entity-versions"

# When a leader's attempt (the exploration node, a follow-up, a node the team's leader owns) created tasks and they are all done, the
# leader is given a review node that carries on its session and sees what each of them produced (`task_review`, 05 §7). The
# number of rounds is bounded by `Policy.max_review_rounds`.
LEADER_REVIEW = "task-leader-review"
# `team_stage` nodes are allowed in a plan and scheduled (07). Its attempt runs the stage in `AttemptWorkflow._run_team`.
TEAM_STAGE = "task-team-stage"
# Nodes say their review round and their stage's limits (`review_round`, `team`) in `node.status_changed` and `getPlan`; team events carry
# the member's `label` and the stage's limits and message counter.
NODE_ROUND_FACTS = "task-node-round-facts"
ATTEMPT_TEAM_FACTS = "attempt-team-facts"
# A cancel of a team stage's attempt cancels the members' turns it started and waits for them (`attempt_team._run_assignments`).
ATTEMPT_TEAM_CANCEL = "attempt-team-cancel"
# A member @-mentioned by a note or by the user takes a turn with it as input (`attempt_team`, 07 §6a); a message that mentions a member
# of a team becomes a follow-up node the member owns, which carries on that member's own session (`task_commands`).
ATTEMPT_TEAM_MENTIONS = "attempt-team-mentions"
MENTION_FOLLOW_UPS = "task-mention-follow-ups"
# Mention follow-up nodes are read-only (a replica of the head snapshot each), so several mentioned members run in parallel.
MENTION_READ_ONLY = "task-mention-follow-ups-read-only"
# Nodes say which role owns them (`owner_role`, `owner_label`); a task with a team says what its members are given and answer at plan level
# (`team.message` of kinds assign, reply and review).
NODE_OWNER_FACTS = "task-node-owner-facts"
TEAM_MESSAGES = "task-team-messages"
ATTEMPT_TEAM_STAGE = "attempt-team-stage"

# A message signalled to an attempt is not consumed by the signal: the parent keeps it on the attempt record, the attempt reports which
# messages a turn was really given (`heard_message_seqs`), and the rest go back to the inbox when the attempt ends.
HANDED_MESSAGES = "task-handed-messages"
ATTEMPT_HEARD_MESSAGES = "attempt-heard-messages"
# How many handed messages the parent keeps per attempt, and how many heard sequence numbers the child reports.
MAX_HANDED = 16
MAX_HEARD = 64

# Attempts of one task that run at once, unless the task policy says otherwise; and the design's nesting depth; and the
# longest chain of dependencies a plan may hold (a safety bound).
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_MAX_DEPTH = 1
MAX_PLAN_CHAIN = 256
# What a handover or a final answer keeps when it is carried to another attempt.
HANDOVER_CHARS = 2000
# Reviews of a leader's work: rounds unless the policy says otherwise, the tasks one review follows and how much of what they
# produced its prompt holds.
DEFAULT_MAX_REVIEW_ROUNDS = 5
MAX_REVIEW_CHILDREN = 40
REVIEW_PROMPT_CHARS = 12000
# How long the verifier agent of one SOP step may take (`verify_sop_step`).
SOP_VERIFIER_TIMEOUT_S = 900

# Seconds a node waits before the retry that follows its 1st, 2nd, 3rd... failed attempt (the last one repeats).
RETRY_BACKOFF_S = (5, 30, 120)
DEFAULT_MAX_ATTEMPTS = 3
# A cancelled child normally reports its own end. This is only for one that never does: a bit more than the heartbeat
# timeout an activity that stopped heartbeating is declared lost after.
CANCEL_FALLBACK_S = 120
# The cancel request to a child is waited for at most this long (`task-bounded-cancel`).
BOUNDED_CANCEL = "task-bounded-cancel"
CANCEL_REQUEST_S = 10
# The dedup window (04 §4), the decided approvals kept beside the pending ones, and what a follow-up keeps as its goal.
MAX_DEDUP = 512
MAX_DECIDED_APPROVALS = 64
FOLLOW_UP_GOAL_CHARS = 500
# A plan this big (half of the size a plan change may reach) has its finished nodes compacted; the newest few stay.
PLAN_COMPACT_BYTES = 128 * 1024
KEEP_RECENT_COMPLETED = 16
ARCHIVE_RECENT_TITLES = 10
MAX_ARCHIVED_IDS = 2000


def versioning_behavior(behavior: VersioningBehavior) -> VersioningBehavior:
    """Use deployment versioning only when the worker is registered for it. Evaluated when the module is imported,
    from the process's one reading of the switch; never while a workflow runs."""
    if versioning_settings().enabled:
        return behavior
    return VersioningBehavior.UNSPECIFIED


def sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def reasons(result: dict[str, Any], activity_name: str) -> list[dict[str, Any]]:
    """The structured failure reasons of a verification activity's answer. An activity that says `ok: false` without
    naming a reason still refuses the completion."""
    failures = list(result.get("failures", []))
    if not failures and not result.get("ok", True):
        failures = [{"check": "verification", "code": "verification_failed", "message": f"{activity_name} refused the completion", "detail": {}}]
    return failures


def closed(status: str) -> bool:
    """Only a cancel ends a task. COMPLETED means this round is done, and the next message starts another."""
    return status == "CANCELLED"


def budget_add(a: Budget, b: Budget) -> Budget:
    values: dict[str, int | None] = {}
    for name in ("tokens", "tool_calls", "wall_s", "cost_usd_micros"):
        left, right = getattr(a, name), getattr(b, name)
        values[name] = None if left is None and right is None else (left or 0) + (right or 0)
    return Budget(**values)
