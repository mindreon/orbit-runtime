"""The attempt of a `team_stage` node (07): a leader and its members as activities, driven by this workflow.

`AttemptWorkflow._run_team` is a loop of rounds. A round is one turn of the leader (`agent_turn`, as the leader's expert, with
the Orbit tool `team_assign`):

  * the leader answers: that is the stage's result, and the attempt ends;
  * it fails: the attempt fails as any attempt does;
  * it opens `team_assign` calls (`{member, task}`): each is a turn of that member, an `agent_turn` of its own on a session of its
    own (the id of the session is derived from the attempt and the role, `team_stage.member_attempt_id`), as its executor, with
    the permissions of the task, in a copy of the task's workspace that is thrown away, against a share of the attempt's
    budget. Turns of different members run side by side; two for the same member run one after the other, in the order the
    leader called them (`TeamPipeline`'s rule). Each member's answer is the result of its call, and when every call has one the
    leader goes on in the next round with all of them at once.

Whichever agent needs a person (an approval, a question from `ask_user`) parks the attempt: `attemptParked` says who, and the
decision or the answer resumes that agent only. The attempt is parked while any agent is; the others go on.

Members never call each other. What they have to tell the team goes through the mailbox (`team_note`): a bounded list of notes in
`TeamRun`, shown to each agent at its next turn as a short "团队消息" block. Limits (`TeamStageSpec.limits`) bound rounds,
assignments, results and notes; a limit or the attempt's budget ends the stage as a failure with the reason, never silently.
Events (`team.*`, entity `team`) say what happened round by round and member by member.

A Continue-As-New is taken between two rounds, when no activity is running, and carries `TeamRun`: the sessions of the leader
and the members, the mailbox, the counters and the answers waiting for the leader.
"""

from __future__ import annotations

import asyncio
from typing import Any

from temporalio import workflow
from temporalio.exceptions import ActivityError
from temporalio.exceptions import CancelledError as TemporalCancelledError

with workflow.unsafe.imports_passed_through():
    from orbit_contracts.v3 import AttemptWorkflowInput
    from orbit_contracts.v3.common import Budget, Usage

    from orbit_orch import budgets
    from orbit_orch.plan_engine import deterministic_id
    from orbit_orch.team_stage import (
        MESSAGE_CHARS,
        NOTES_PER_TURN,
        SUMMARY_PREVIEW_CHARS,
        TASK_PREVIEW_CHARS,
        Participant,
        TeamRun,
        answer_text,
        display_names,
        member_attempt_id,
        member_prompt,
        resolve_mentions,
        user_block,
        wake_prompt,
        with_blocks,
    )
    from orbit_orch.workflow_common import (
        AGENT_TIMEOUT,
        ATTEMPT_BUDGET,
        ATTEMPT_CONTINUE_AS_NEW,
        ATTEMPT_RETURNS_MESSAGES,
        ATTEMPT_TEAM_CANCEL,
        ATTEMPT_TEAM_FACTS,
        ATTEMPT_TEAM_MENTIONS,
        HEARTBEAT,
        IO_TIMEOUT,
        RETRY,
    )

ASSIGN = "team_assign"
ASK = "ask_user"


def _usage_since(now: Usage, before: Usage) -> Usage:
    cost = None if now.cost_usd_micros is None else max(0, now.cost_usd_micros - (before.cost_usd_micros or 0))
    return Usage(
        tokens_in=max(0, now.tokens_in - before.tokens_in),
        tokens_out=max(0, now.tokens_out - before.tokens_out),
        tool_calls=max(0, now.tool_calls - before.tool_calls),
        wall_s=max(0, now.wall_s - before.wall_s),
        cost_usd_micros=cost,
    )


class _Stop(Exception):
    """The stage ends here, as a failure of `failure_class` with `message` (a limit, the budget)."""

    def __init__(self, failure_class: str, message: str) -> None:
        super().__init__(message)
        self.failure_class = failure_class
        self.message = message


class AttemptTeam:
    # What `AttemptWorkflow` provides: `_messages`, `_decisions`, `_usage`, `_ran_activity`, `_carry`, `_should_continue_as_new`,
    # `_count_usage`, `_commit_checkpoints`, `_budget_payload`, `_notify_parent_parked` and `_notify_parent_finished`.
    _team: TeamRun
    _team_events: list[dict[str, Any]]
    _team_flush: asyncio.Lock
    _round_usage: Usage
    _mention_seen: set[int]

    async def _run_team(self, inp: AttemptWorkflowInput) -> None:
        spec = inp.team
        assert spec is not None
        carried = (inp.carry or {}).get("team")
        team = self._team = TeamRun.from_json(dict(carried)) if carried else TeamRun.start(inp.attempt_id, spec, inp.profile)  # type: ignore[arg-type]
        while True:
            if self._ran_activity and self._should_continue_as_new() and workflow.patched(ATTEMPT_CONTINUE_AS_NEW):
                # Between two rounds nothing is running and every handler is done; the parent is told nothing (04 §6).
                await self._flush_team_events(inp)
                await workflow.wait_condition(workflow.all_handlers_finished)
                workflow.continue_as_new(inp.model_copy(update={"messages": [], "carry": {**self._carry(), "team": team.to_json()}}))
            round_open = False
            try:
                if team.round >= spec.limits.max_rounds:
                    raise _Stop("policy", f"the team stage reached its limit of {spec.limits.max_rounds} rounds without a final answer")
                self._team_budget_check(inp)
                team.round += 1
                round_open = True
                before = self._usage
                self._team_emit(inp, team, "team.round_started", {
                    "node_id": inp.node_id, "attempt_id": inp.attempt_id, "round": team.round, "max_rounds": spec.limits.max_rounds,
                    **self._team_start_facts(spec, team),
                })
                outcome = await self._team_round(inp, team)
                self._team_emit(inp, team, "team.round_finished", {
                    "node_id": inp.node_id, "attempt_id": inp.attempt_id, "round": team.round,
                    "outcome": outcome["outcome"], "assignments": outcome.get("assignments", 0),
                    **self._team_count(team),
                    "usage": _usage_since(self._usage, before).model_dump(mode="json", exclude_none=True),
                })
                round_open = False
                await self._flush_team_events(inp)
                if outcome["outcome"] == "completed":
                    await self._notify_parent_finished(inp, "completed", outcome["result"])
                    return
                if outcome["outcome"] == "failed":
                    await self._notify_parent_finished(inp, "failed", outcome["result"])
                    return
            except _Stop as stop:
                if round_open:
                    self._team_emit(inp, team, "team.round_finished", {
                        "node_id": inp.node_id, "attempt_id": inp.attempt_id, "round": team.round, "outcome": "stopped",
                        "reason": stop.message[:300], **self._team_count(team),
                    })
                await self._flush_team_events(inp)
                await self._notify_parent_finished(
                    inp, "failed", {"error": stop.message, "failure_class": stop.failure_class, "retryable": False}
                )
                return

    # ---- one round -----------------------------------------------------------------------------------------------

    async def _team_round(self, inp: AttemptWorkflowInput, team: TeamRun) -> dict[str, Any]:
        """A turn of the leader and, when it assigned work, the turns of the members it named. The outcome says how the round
        ended: `completed` or `failed` (with the leader's result), or `assigned` (the leader goes on in the next round)."""
        spec = inp.team
        assert spec is not None
        leader = team.leader
        # Members the user @-mentioned while the stage was between turns take theirs first (07 §6a).
        await self._run_assignments(inp, team, [], {})
        resume, team.pending_results = team.pending_results, None
        if resume is None:
            delivered = list(self._messages)
            self._in_flight = len(delivered)
            self._heard |= {message.message_seq for message in delivered}
            result, _ = await self._team_turn(inp, team, leader, goal=inp.goal, messages=delivered)
            self._consume(result, delivered)
        else:
            # The answers, with what else the leader should see now: the team's notes and what the user wrote meanwhile.
            delivered = list(self._messages)
            self._in_flight = len(delivered)
            self._heard |= {message.message_seq for message in delivered}
            resume = [dict(item) for item in resume]
            resume[-1]["output"] = with_blocks(
                resume[-1]["output"], team.unread(leader.role, display_names(spec)), user_block([message.text for message in delivered])
            )
            result, _ = await self._team_turn(inp, team, leader, goal="", results=resume)
            self._consume(result, delivered)
        self._in_flight = 0
        self._team_notes(inp, team, leader.role, result, 0)  # the leader's notes address people; assigning is how it wakes them
        status = result.get("status")
        if status == "completed":
            self._team_message(
                inp, team, kind="review", from_role=leader.role, to_roles=[], text=str(result.get("text", "")), hop=0
            )
            return {"outcome": "completed", "result": result}
        if status != "team_external":
            return {"outcome": "failed", "result": result}
        calls = list(result["externals"])
        assigns = [call for call in calls if call["tool_name"] == ASSIGN]
        asks = [call for call in calls if call["tool_name"] == ASK]
        answers: dict[str, dict[str, Any]] = {}
        valid: list[dict[str, Any]] = []
        for call in assigns:
            member = str(call["arguments"].get("member", ""))
            if member not in team.members:
                known = ", ".join(sorted(team.members))
                answers[call["call_id"]] = _answer(call, f"there is no member {member!r}: the members are {known}", "error")
            else:
                valid.append({
                    "call_id": call["call_id"], "tool_name": ASSIGN, "member": member, "task": str(call["arguments"].get("task", "")),
                })
        if valid and team.round >= spec.limits.max_rounds:
            raise _Stop(
                "policy",
                f"the team stage reached its limit of {spec.limits.max_rounds} rounds: the leader assigned work it has no round left to use",
            )
        if team.messages_used + 2 * len(valid) > spec.limits.max_messages:
            raise _Stop(
                "policy",
                f"the team stage would pass its limit of {spec.limits.max_messages} messages "
                f"({team.messages_used} used, {len(valid)} assignments asked for)",
            )
        await self._run_assignments(inp, team, valid, answers)
        for call in asks:
            reply = await self._team_ask(inp, team, leader, [call])
            answers[call["call_id"]] = {"call_id": call["call_id"], "tool_name": ASK, "output": reply, "result_state": "success"}
        team.pending_results = [answers[call["call_id"]] for call in calls if call["call_id"] in answers]
        if team.messages_used > spec.limits.max_messages:
            raise _Stop("policy", f"the team stage passed its limit of {spec.limits.max_messages} messages")
        return {"outcome": "assigned", "assignments": len(valid)}

    def _consume(self, result: dict[str, Any], delivered: list[Any]) -> None:
        """What the leader's turn was given is consumed, unless the turn failed: it did not hear it, so it stays and goes back
        to the task with the attempt."""
        if result.get("status") != "failed" or not workflow.patched(ATTEMPT_RETURNS_MESSAGES):
            self._messages = self._messages[len(delivered):]
        else:
            self._heard -= {message.message_seq for message in delivered}

    async def _run_assignments(
        self, inp: AttemptWorkflowInput, team: TeamRun, assigns: list[dict[str, Any]], answers: dict[str, dict[str, Any]]
    ) -> None:
        """Run the turns that are due: the leader's assignments, the wakes of members a note or the user @-mentioned. Different
        members side by side, the same member one after the other, in the order the work reached it (`TeamPipeline`'s rule).
        The work a turn makes (a note that mentions someone) joins the queues, so this returns when nothing is left."""
        queues: dict[str, list[dict[str, Any]]] = {}
        tasks: dict[str, asyncio.Task[None]] = {}
        stops: list[_Stop] = []
        mentions_on = workflow.patched(ATTEMPT_TEAM_MENTIONS)

        def enqueue(role: str, job: dict[str, Any]) -> None:
            queues.setdefault(role, []).append(job)
            if role not in tasks or tasks[role].done():
                tasks[role] = asyncio.create_task(drain(role))

        def pull_user_wakes() -> None:
            if mentions_on:
                for job in self._take_user_wakes(inp, team):
                    enqueue(job["member"], job)

        async def drain(role: str) -> None:
            while queues.get(role):
                job = queues[role].pop(0)
                # Members that run together share what is left; a member that starts alone has all of it.
                parts = max(1, sum(1 for task in tasks.values() if not task.done()))
                try:
                    await self._member_job(inp, team, role, job, parts, answers, enqueue)
                except _Stop as stop:
                    stops.append(stop)
                    for queue in queues.values():
                        queue.clear()  # the stage is ending: nothing more is started
                    return
                pull_user_wakes()

        # Members that start together are all given their share of what was left when the round began, not of what is left
        # after whichever of them finishes first.
        self._round_usage = self._usage
        for call in assigns:
            enqueue(call["member"], {"kind": "assign", "call": call, "hop": 0})
        pull_user_wakes()

        async def watch_user() -> None:
            """A message the user @-mentions a member in while turns run is answered now, not when the turn that was running ends."""
            while True:
                await workflow.wait_condition(lambda: self._has_new_mention(inp, team))
                pull_user_wakes()

        watcher = asyncio.create_task(watch_user()) if mentions_on else None
        try:
            while True:
                live = [task for task in tasks.values() if not task.done()]
                if not live:
                    break
                await workflow.wait(live, return_when=asyncio.ALL_COMPLETED)
        finally:
            if watcher is not None:
                watcher.cancel()
            if workflow.patched(ATTEMPT_TEAM_CANCEL):
                # A cancel of the attempt reaches only this coroutine: the members' turns are tasks of their own, and their
                # activities would run on, unobserved. Each is cancelled and waited for (the activities are cancelled with
                # WAIT_CANCELLATION_COMPLETED), so the attempt closes only once every agent has stopped.
                running = [task for task in tasks.values() if not task.done()]
                for task in running:
                    task.cancel()
                if running:
                    await asyncio.gather(*running, return_exceptions=True)
        if stops:
            raise stops[0]

    def _has_new_mention(self, inp: AttemptWorkflowInput, team: TeamRun) -> bool:
        spec = inp.team
        assert spec is not None
        return any(
            message.message_seq not in self._mention_seen
            and any(role in team.members for role in resolve_mentions(spec, message.text, list(message.mentions), ""))
            for message in self._messages
        )

    def _take_user_wakes(self, inp: AttemptWorkflowInput, team: TeamRun) -> list[dict[str, Any]]:
        """The user's messages that @-mention members of the team become wake jobs, one per mentioned member. The message goes
        into the mailbox (the leader and the others see it at their next turn). A message that mentions only the leader, or
        nobody, is left for the leader's own turn as before."""
        spec = inp.team
        assert spec is not None
        jobs: list[dict[str, Any]] = []
        for message in list(self._messages):
            if message.message_seq in self._mention_seen:
                continue
            roles = resolve_mentions(spec, message.text, list(message.mentions), "")
            members = [role for role in roles if role in team.members]
            if not members:
                continue
            self._mention_seen.add(message.message_seq)
            if team.messages_used + 2 * len(members) > spec.limits.max_messages:
                self._team_message(
                    inp, team, kind="system", from_role="system", to_roles=[], hop=0,
                    text=f"nobody was woken by the user's message: the stage is at its limit of {spec.limits.max_messages} messages",
                )
                continue
            if team.leader.role not in roles:
                self._messages.remove(message)  # the wake is how it is answered; the leader hears it from the mailbox
            self._heard.add(message.message_seq)
            team.post("user", message.text)
            self._team_message(inp, team, kind="user", from_role="user", to_roles=roles, text=message.text, hop=0)
            jobs += [
                {"kind": "wake", "member": role, "from": "user", "text": message.text, "hop": 0} for role in members
            ]
        return jobs

    def _wake_from_note(
        self, inp: AttemptWorkflowInput, team: TeamRun, role: str, text: str, mentioned: list[str], hop: int, enqueue: Any
    ) -> None:
        """A note @-mentions members: each takes a turn with it, unless the chain is too long or the stage out of messages. A
        refusal is said in the group. Nobody wakes itself, and the leader is not woken (it is parked on its assignments and
        reads everything at its next turn)."""
        spec = inp.team
        assert spec is not None
        if not workflow.patched(ATTEMPT_TEAM_MENTIONS):
            return
        for target in mentioned:
            if target == role or target not in team.members:
                continue
            if hop + 1 > spec.limits.max_hops:
                why = f"{target} was not woken: the chain of mentions would be {hop + 1} members long and the limit is {spec.limits.max_hops}"
            elif team.messages_used + 2 > spec.limits.max_messages:
                why = f"{target} was not woken: the stage is at its limit of {spec.limits.max_messages} messages"
            else:
                enqueue(target, {"kind": "wake", "member": target, "from": role, "text": text, "hop": hop + 1})
                continue
            self._team_message(inp, team, kind="system", from_role="system", to_roles=[role], text=why, hop=hop)

    async def _member_job(
        self,
        inp: AttemptWorkflowInput,
        team: TeamRun,
        role: str,
        job: dict[str, Any],
        parts: int,
        answers: dict[str, dict[str, Any]],
        enqueue: Any,
    ) -> None:
        """One turn of a member: an assignment of the leader (its answer is the result of the leader's call) or a wake (its
        answer goes to the group, addressed to whoever woke it)."""
        member = team.members[role]
        hop = int(job["hop"])
        call = job.get("call")
        leader = team.leader.role
        if call is not None:
            task = str(call["task"])
            self._team_message(inp, team, kind="assign", from_role=leader, to_roles=[role], text=task, hop=0)
            said = task
            requester = leader
        else:
            task = str(job["text"])
            requester = str(job["from"])
            said = wake_prompt(self._name_of(inp, requester), task)
        team.messages_used += 1  # the assignment or the wake
        prompt = member_prompt(said, team.unread(role, display_names(inp.team)))
        self._team_emit(inp, team, "team.member_turn_started", {
            "node_id": inp.node_id, "attempt_id": inp.attempt_id, "round": team.round, "role": role, "executor": member.executor,
            "member_attempt_id": member.attempt_id, "task": task[:TASK_PREVIEW_CHARS],
            **self._team_label(inp, role),
        })
        await self._flush_team_events(inp)
        finished: dict[str, Any] = {
            "node_id": inp.node_id, "attempt_id": inp.attempt_id, "round": team.round, "role": role, "executor": member.executor,
            "member_attempt_id": member.attempt_id, **self._team_label(inp, role),
        }
        try:
            result, spent = await self._team_turn(inp, team, member, goal=prompt, parts=parts)
        except ActivityError as exc:
            if isinstance(exc.cause, TemporalCancelledError):
                raise asyncio.CancelledError from exc
            # The activity gave up after its retries (a lost worker): the leader hears that, and decides.
            self._team_emit(inp, team, "team.member_turn_finished", {**finished, "outcome": "failed", "summary": str(exc.cause or exc)[:SUMMARY_PREVIEW_CHARS]})
            await self._flush_team_events(inp)
            if call is not None:
                answers[call["call_id"]] = _answer(call, f"{role} could not be reached: {exc.cause or exc}", "error")
            return
        for text, mentioned in self._team_notes(inp, team, role, result, hop):
            self._wake_from_note(inp, team, role, text, mentioned, hop, enqueue)
        if result.get("status") == "completed":
            files = list(result.get("team_files", []))
            team.remember_files(role, files)
            text = str(result.get("text", ""))
            if call is not None:
                team.messages_used += 1  # the answer
                answers[call["call_id"]] = _answer(call, answer_text(text, role, files), "success")
            else:
                team.post(role, text)  # the group, and the leader at its next turn, read the answer
            self._team_message(
                inp, team, kind="reply", from_role=role, to_roles=[requester], text=text, hop=hop, artifacts=files
            )
            self._team_emit(inp, team, "team.member_turn_finished", {
                **finished, "outcome": "completed", "summary": text[:SUMMARY_PREVIEW_CHARS],
                "usage": spent.model_dump(mode="json", exclude_none=True), "artifacts": [str(item["name"]) for item in files][:20],
            })
        else:
            error = str(result.get("error", "the turn failed"))
            self._team_emit(inp, team, "team.member_turn_finished", {
                **finished, "outcome": "failed", "summary": error[:SUMMARY_PREVIEW_CHARS],
                "usage": spent.model_dump(mode="json", exclude_none=True),
            })
            if call is not None:
                answers[call["call_id"]] = _answer(call, f"{role} could not do it: {error}", "error")
            if result.get("failure_class") == "budget":
                await self._flush_team_events(inp)
                raise _Stop("budget", f"the stage's budget is spent: {error}")
        await self._flush_team_events(inp)

    # ---- one turn of one agent -----------------------------------------------------------------------------------

    async def _team_turn(
        self,
        inp: AttemptWorkflowInput,
        team: TeamRun,
        agent: Participant,
        *,
        goal: str,
        messages: list[Any] | None = None,
        results: list[dict[str, Any]] | None = None,
        parts: int = 1,
    ) -> tuple[dict[str, Any], Usage]:
        """Run `agent` until its turn ends: an `agent_turn` activity, again for every approval or answer it parks on. Returns
        the result that ended it (completed, failed, or the open `team_assign` calls of a leader) and what it spent."""
        spent = Usage()
        approval: dict[str, Any] | None = None
        while True:
            payload = self._team_payload(inp, team, agent, goal=goal, messages=messages or [], approval=approval, results=results, parts=parts)
            result = await workflow.execute_activity(
                "agent_turn",
                payload,
                task_queue="orbit.agent",
                result_type=dict,
                start_to_close_timeout=AGENT_TIMEOUT,
                heartbeat_timeout=HEARTBEAT,
                cancellation_type=workflow.ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
                retry_policy=RETRY,
            )
            self._ran_activity = True
            self._count_usage(result)
            if result.get("usage") and workflow.patched(ATTEMPT_BUDGET):
                spent = budgets.usage_add(spent, Usage.model_validate(result["usage"]))
            await self._commit_checkpoints(inp, result, agent.attempt_id)
            if result.get("session_id"):
                agent.session_id = str(result["session_id"])
            if result.get("state_version") is not None:
                agent.state_version = int(result["state_version"])
            agent.retry_calls = result.get("retry_calls")
            results, approval = None, None
            state = result.get("status")
            if state == "parked_approval":
                approval = await self._team_park_approval(inp, agent, result)
                continue
            if state == "team_external":
                asks = [call for call in result["externals"] if call["tool_name"] == ASK]
                if len(asks) < len(result["externals"]):
                    return result, spent  # a leader's assignments: the round runs them
                reply = await self._team_ask(inp, team, agent, asks)
                results = [{"call_id": call["call_id"], "tool_name": ASK, "output": reply, "result_state": "success"} for call in asks]
                continue
            return result, spent

    def _team_payload(
        self,
        inp: AttemptWorkflowInput,
        team: TeamRun,
        agent: Participant,
        *,
        goal: str,
        messages: list[Any],
        approval: dict[str, Any] | None,
        results: list[dict[str, Any]] | None,
        parts: int,
    ) -> dict[str, Any]:
        spec = inp.team
        assert spec is not None
        leader = agent.leader
        # The stage's agents are not shown the plan's team (a stage is its own): only the task's expert and skills.
        config = inp.config.model_copy(update={"team": None})
        return {
            "task_id": inp.task_id,
            "tenant_id": inp.tenant_id,
            "node_id": inp.node_id,
            "attempt_id": agent.attempt_id,
            "attempt_no": inp.attempt_no,
            "profile": agent.executor,
            "config": config.model_dump(mode="json"),
            "allow_rules": [rule.model_dump(mode="json") for rule in inp.allow_rules],
            # An attempt that carries on an earlier one (a retry) carries on each agent's session there.
            "continue_from": inp.continue_from if leader else (member_attempt_id(inp.continue_from, agent.role) if inp.continue_from else None),
            "retry_reason": inp.retry_reason if leader else "",
            "goal": goal,
            "checkpoint_ref": inp.checkpoint_ref,
            # Only the leader writes the task's workspace; a member works in a copy of it, or in none (07 §8).
            "workspace_access": inp.workspace_access if leader else "none",
            "policy": inp.policy.model_dump(mode="json"),
            "messages": [message.model_dump(mode="json") for message in messages],
            "external": None,
            "retry_calls": agent.retry_calls,
            "approval": approval,
            "session_id": agent.session_id,
            "state_version": agent.state_version,
            "switched_from": None,
            "handover": "",
            "output_schema_ref": inp.output_schema_ref if leader else None,
            "team": {
                "role": agent.role,
                "label": self._name_of(inp, agent.role) if self._name_of(inp, agent.role) != agent.role else "",
                "leader_label": self._name_of(inp, spec.leader),
                "leader": leader,
                "leader_role": spec.leader,
                "stage_attempt_id": inp.attempt_id,
                "workspace": inp.workspace_access if leader else spec.workspace_access,
                "members": (
                    [[member.role, member.description, member.label] for member in spec.members if member.role != spec.leader]
                    if leader else []
                ),
                "files": team.files_for_leader() if leader else [],
            },
            "team_results": results,
            **(self._team_budget(inp, 1 if leader else parts, leader)),
        }

    def _team_budget(self, inp: AttemptWorkflowInput, parts: int, leader: bool = True) -> dict[str, Any]:
        """What is left of the attempt's reservation for one agent's turn. Members that run side by side share it (time is not
        split: they use it together), so what they spend together stays within what is reserved."""
        if inp.budget is None or not workflow.patched(ATTEMPT_BUDGET):
            return {}
        left = budgets.after(inp.budget, self._usage if leader else self._round_usage)
        share = Budget(**{
            field: None if value is None else (value if field == "wall_s" else value // max(1, parts))
            for field, value in left.model_dump().items()
        })
        return {"budget": share.model_dump(mode="json", exclude_none=True)}

    def _team_budget_check(self, inp: AttemptWorkflowInput) -> None:
        """No round starts on a reservation that is spent."""
        if inp.budget is None or not workflow.patched(ATTEMPT_BUDGET):
            return
        left = budgets.after(inp.budget, self._usage).model_dump(exclude_none=True)
        gone = sorted(field for field, value in left.items() if value == 0)
        if gone:
            raise _Stop("budget", f"the stage's budget is spent ({', '.join(gone)})")

    # ---- people --------------------------------------------------------------------------------------------------

    async def _team_park_approval(self, inp: AttemptWorkflowInput, agent: Participant, result: dict[str, Any]) -> dict[str, Any]:
        """The agent's turn stopped for approvals: say so to the parent (with the role, so the person knows who is asking), wait
        for a decision on each call, and hand them back to resume that agent."""
        shown = []
        for item in result.get("approvals", []):
            subject = dict(item["subject"])
            subject["role"] = agent.role
            subject["summary"] = f"{agent.role}: {subject.get('summary', '')}"
            label = self._name_of(inp, agent.role)
            if label != agent.role:
                subject["role_label"] = label
            shown.append({"tool_call_id": f"{agent.role}:{item['tool_call_id']}", "subject": subject})
        agent.awaiting = {item["tool_call_id"] for item in shown}
        agent.approval_request_id = str(result.get("approval_request_id", ""))
        await self._notify_parent_parked(inp, "approval", {**result, "approvals": shown})
        await workflow.wait_condition(lambda: agent.awaiting <= self._decisions.keys())
        chosen = {key: self._decisions.pop(key) for key in agent.awaiting}
        agent.awaiting = set()

        def strip(key: str) -> str:
            return key.split(":", 1)[1]  # the id of the call, without the role it was given

        return {
            "approval_id": list(chosen.values())[-1].approval_id,
            "approval_request_id": agent.approval_request_id,
            "decision": "approve" if all(item.decision == "approve" for item in chosen.values()) else "reject",
            "decisions": {strip(key): item.decision for key, item in chosen.items()},
            "rules": {strip(key): item.rule.model_dump(mode="json") for key, item in chosen.items() if item.rule is not None and item.decision == "approve"},
        }

    async def _team_ask(self, inp: AttemptWorkflowInput, team: TeamRun, agent: Participant, calls: list[dict[str, Any]]) -> str:
        """The agent asked the user: park the attempt on the question and answer with what the user writes next."""
        question = " / ".join(str(call["arguments"].get("question", "")) for call in calls)
        await self._notify_parent_parked(inp, "input", {"question": f"[{agent.role}] {question}"})
        await workflow.wait_condition(lambda: bool(self._messages))
        delivered = list(self._messages)
        self._heard |= {message.message_seq for message in delivered}
        self._messages = self._messages[len(delivered):]
        return "\n".join(message.text for message in delivered)

    # ---- the mailbox and the events ------------------------------------------------------------------------------

    def _team_notes(
        self, inp: AttemptWorkflowInput, team: TeamRun, role: str, result: dict[str, Any], hop: int
    ) -> list[tuple[str, list[str]]]:
        """The notes a turn posted: into the mailbox, and as `team.message` events. Returns each note's text and the roles it
        addresses (the `mentions` it listed, else the @role and @label in the text)."""
        spec = inp.team
        assert spec is not None
        listed = list(result.get("note_mentions") or [])
        notes: list[tuple[str, list[str]]] = []
        for index, text in enumerate(list(result.get("notes") or [])[:NOTES_PER_TURN]):
            entry = team.post(role, str(text))
            explicit = [str(item) for item in listed[index]] if index < len(listed) else []
            mentioned = resolve_mentions(spec, entry["text"], explicit, role)
            self._team_message(inp, team, kind="note", from_role=role, to_roles=mentioned, text=entry["text"], hop=hop)
            notes.append((entry["text"], mentioned))
        return notes

    def _team_message(
        self,
        inp: AttemptWorkflowInput,
        team: TeamRun,
        *,
        kind: str,
        from_role: str,
        to_roles: list[str],
        text: str,
        hop: int,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> None:
        """One utterance of the group conversation as a durable `team.message` (`role` and `label` repeat the sender, for the
        first shape of the event)."""
        team.event_seq += 1
        payload: dict[str, Any] = {
            "node_id": inp.node_id, "attempt_id": inp.attempt_id, "seq": team.event_seq, "role": from_role, "from_role": from_role,
            "to_roles": to_roles, "text": text[:MESSAGE_CHARS], "kind": kind, "round": team.round, "hop": hop,
            **self._team_label(inp, from_role),
        }
        if "label" in payload:
            payload["from_label"] = payload["label"]
        if artifacts:
            payload["artifacts"] = [
                {"name": str(item["name"]), "blob_ref": item.get("blob_ref")} for item in artifacts[:20]
            ]
        self._team_emit(inp, team, "team.message", payload)

    @staticmethod
    def _name_of(inp: AttemptWorkflowInput, role: str) -> str:
        """What a role is called when it is addressed or quoted to an agent: its label, else 领队 for the leader, else the role."""
        assert inp.team is not None
        return display_names(inp.team).get(role, role)

    @staticmethod
    def _team_label(inp: AttemptWorkflowInput, role: str) -> dict[str, Any]:
        """The label a person gave the role, beside `role` in team events (only when there is one)."""
        label = next((m.label for m in (inp.team.members if inp.team else []) if m.role == role and m.label), "")
        return {"label": label} if label and workflow.patched(ATTEMPT_TEAM_FACTS) else {}

    @staticmethod
    def _team_start_facts(spec: Any, team: TeamRun) -> dict[str, Any]:
        if not workflow.patched(ATTEMPT_TEAM_FACTS):
            return {}
        return {
            "max_messages": spec.limits.max_messages, "max_members": spec.limits.max_members, "max_hops": spec.limits.max_hops,
            "messages": team.messages_used,
        }

    @staticmethod
    def _team_count(team: TeamRun) -> dict[str, Any]:
        return {"messages": team.messages_used} if workflow.patched(ATTEMPT_TEAM_FACTS) else {}

    def _team_emit(self, inp: AttemptWorkflowInput, team: TeamRun, event_type: str, payload: dict[str, Any]) -> None:
        team.version += 1
        self._team_events.append({
            "schema": "orbit.event/3",
            "event_id": deterministic_id(str(workflow.uuid4()), "evt"),
            "tenant_id": inp.tenant_id,
            "task_id": inp.task_id,
            "type": event_type,
            "retention": "durable",
            "source": {"kind": "workflow", "id": workflow.info().workflow_id, "attempt_id": inp.attempt_id},
            "entity": {"kind": "team", "id": inp.attempt_id, "version": team.version},
            "visibility": "tenant",
            "occurred_at": workflow.now().isoformat(),
            "payload": payload,
        })

    async def _flush_team_events(self, inp: AttemptWorkflowInput) -> None:
        """Publish what the stage has to say. Losing them is not worth failing the stage: it is logged."""
        if not self._team_events:
            return
        # One batch at a time, so what members say side by side reaches the projection in the order it was said.
        async with self._team_flush:
            if not self._team_events:
                return
            events, self._team_events = self._team_events, []
            try:
                await workflow.execute_activity(
                    "publish_events",
                    events,
                    task_queue="orbit.io",
                    result_type=dict,
                    start_to_close_timeout=IO_TIMEOUT,
                    retry_policy=RETRY,
                )
            except ActivityError as exc:
                if isinstance(exc.cause, TemporalCancelledError):
                    raise asyncio.CancelledError from exc
                workflow.logger.warning("team events of %s were not published: %s", inp.attempt_id, exc)


def _answer(call: dict[str, Any], output: str, state: str) -> dict[str, Any]:
    return {"call_id": call["call_id"], "tool_name": call["tool_name"], "output": output, "result_state": state}
