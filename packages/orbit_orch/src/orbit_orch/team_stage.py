"""A team stage's state and the rules of its protocol that need no workflow (07).

A team stage is one attempt (an `AttemptWorkflow`) whose leader and members are `agent_turn` activities. The protocol is
AgentScope's `TeamPipeline`, driven by Temporal instead of by its executor: the leader's turn ends with `team_assign` calls
open, the workflow runs a turn of each member the calls name (side by side across members, one after another for the same
member) and hands every answer back to the leader as the call's result. Members do not call each other. What they need to
tell the team goes through a mailbox: a short, bounded list in this state that each participant is shown at its next turn.

Everything here is plain data and pure functions, so the workflow can carry it across a Continue-As-New (`TeamRun.to_json`)
and tests can check the rules without a Temporal server.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from orbit_contracts.v3.nodes import TeamStageSpec

from orbit_orch.handover import fit_handover
from orbit_orch.plan_engine import deterministic_id

# What an event and the mailbox keep of a note, of the task a member was given and of what it answered.
NOTE_CHARS = 500
TASK_PREVIEW_CHARS = 300
SUMMARY_PREVIEW_CHARS = 500
# What the leader is handed of one member's answer, and of all the files members have left.
RESULT_CHARS = 6000
FILES_LISTED = 20
MAX_TEAM_FILES = 100
# How many notes of one turn count, and how many entries the mailbox keeps and shows.
NOTES_PER_TURN = 5
MAILBOX_KEPT = 200
MAILBOX_SHOWN = 10
MAILBOX_BLOCK_CHARS = 4000
MAILBOX_HEADING = "团队消息"
USER_HEADING = "用户消息"


def member_attempt_id(attempt_id: str, role: str) -> str:
    """The id a member runs under: it is the id of the member's session too (a session is its attempt's id), so it is derived
    from the stage's attempt and the member's role and any worker finds the session again."""
    return deterministic_id(f"{attempt_id}:member:{role}", "att")


@dataclass
class Participant:
    """The leader or one member: who runs it, its session and how far it has read the mailbox."""

    role: str
    executor: str
    attempt_id: str
    leader: bool = False
    session_id: str = ""
    state_version: int = 0
    mail_cursor: int = 0
    turns: int = 0
    # The call ids (with their role) a parked turn waits for a decision on; and what it needs to be asked again after a
    # worker died with calls of unknown outcome.
    awaiting: set[str] = field(default_factory=set)
    approval_request_id: str = ""
    retry_calls: list[dict[str, Any]] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "role": self.role, "executor": self.executor, "attempt_id": self.attempt_id, "leader": self.leader,
            "session_id": self.session_id, "state_version": self.state_version, "mail_cursor": self.mail_cursor,
            "turns": self.turns,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Participant:
        return cls(
            role=str(raw["role"]), executor=str(raw["executor"]), attempt_id=str(raw["attempt_id"]),
            leader=bool(raw.get("leader", False)), session_id=str(raw.get("session_id", "")),
            state_version=int(raw.get("state_version", 0)), mail_cursor=int(raw.get("mail_cursor", 0)),
            turns=int(raw.get("turns", 0)),
        )


@dataclass
class TeamRun:
    """What a team stage carries from one round to the next, and across a Continue-As-New."""

    leader: Participant
    members: dict[str, Participant]
    round: int = 0
    # Assignments, results and notes so far, against `limits.max_messages`.
    messages_used: int = 0
    mailbox: list[dict[str, Any]] = field(default_factory=list)
    next_seq: int = 1
    # The answers to the leader's open assignments, once they are all in: the next leader turn is given them.
    pending_results: list[dict[str, Any]] | None = None
    # The files members left, `role/name` -> entry, for the leader to read and merge.
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    # The version of the stage's events: they are one entity (`team`, the stage's attempt).
    version: int = 0
    # The number of the last `team.message` of the stage.
    event_seq: int = 0

    @classmethod
    def start(cls, attempt_id: str, spec: TeamStageSpec, leader_profile: str) -> TeamRun:
        """A new stage."""
        members = {
            member.role: Participant(
                role=member.role, executor=member.executor, attempt_id=member_attempt_id(attempt_id, member.role)
            )
            for member in spec.members
            if member.role != spec.leader
        }
        leader = Participant(role=spec.leader, executor=leader_profile, attempt_id=attempt_id, leader=True)
        return cls(leader=leader, members=members)

    def participant(self, role: str) -> Participant:
        return self.leader if role == self.leader.role else self.members[role]

    def to_json(self) -> dict[str, Any]:
        return {
            "leader": self.leader.to_json(),
            "members": {role: member.to_json() for role, member in self.members.items()},
            "round": self.round,
            "messages_used": self.messages_used,
            "mailbox": self.mailbox[-MAILBOX_KEPT:],
            "next_seq": self.next_seq,
            "pending_results": self.pending_results,
            "files": dict(list(self.files.items())[-MAX_TEAM_FILES:]),
            "version": self.version,
            "event_seq": self.event_seq,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> TeamRun:
        return cls(
            leader=Participant.from_json(raw["leader"]),
            members={str(role): Participant.from_json(item) for role, item in dict(raw["members"]).items()},
            round=int(raw.get("round", 0)),
            messages_used=int(raw.get("messages_used", 0)),
            mailbox=[dict(item) for item in raw.get("mailbox", [])],
            next_seq=int(raw.get("next_seq", 1)),
            pending_results=raw.get("pending_results"),
            files={str(key): dict(value) for key, value in dict(raw.get("files", {})).items()},
            version=int(raw.get("version", 0)),
            event_seq=int(raw.get("event_seq", 0)),
        )

    # ---- the mailbox ---------------------------------------------------------------------------------------------

    def post(self, role: str, text: str) -> dict[str, Any]:
        """A note for the team, from `role`. Returns the entry (with its `seq`)."""
        entry = {"seq": self.next_seq, "role": role, "text": " ".join(text.split())[:NOTE_CHARS]}
        self.next_seq += 1
        self.mailbox.append(entry)
        del self.mailbox[:-MAILBOX_KEPT]
        self.messages_used += 1
        return entry

    def unread(self, role: str, names: dict[str, str] | None = None) -> str:
        """The mailbox block `role` is shown at its next turn: the notes of the others it has not seen, newest last, bounded.
        It is read from then on (the cursor moves past everything posted so far). Empty when there is nothing new."""
        participant = self.participant(role)
        entries = [item for item in self.mailbox if item["seq"] > participant.mail_cursor and item["role"] != role]
        participant.mail_cursor = self.next_seq - 1
        if not entries:
            return ""
        lines: list[str] = []
        used = 0
        for item in reversed(entries[-MAILBOX_SHOWN:]):
            line = f"- [{(names or {}).get(item['role'], item['role'])}] {item['text']}"
            if used + len(line) > MAILBOX_BLOCK_CHARS:
                break
            lines.append(line)
            used += len(line) + 1
        return f"{MAILBOX_HEADING}:\n" + "\n".join(reversed(lines))

    def remember_files(self, role: str, entries: list[dict[str, Any]]) -> None:
        """What a member left in its copy of the workspace. A newer file of the same name replaces the older one."""
        for item in entries:
            key = f"{role}/{item['name']}"
            self.files.pop(key, None)
            self.files[key] = {
                "role": role, "name": item["name"], "blob_ref": item["blob_ref"],
                "media_type": item.get("media_type", "application/octet-stream"), "size_bytes": int(item.get("size_bytes", 0)),
            }
        for key in list(self.files)[: max(0, len(self.files) - MAX_TEAM_FILES)]:
            del self.files[key]

    def files_for_leader(self) -> list[dict[str, Any]]:
        return list(self.files.values())


def member_prompt(task: str, mail: str) -> str:
    """What a member is told for one assignment: the task, then what the team has said since it last looked."""
    return task if not mail else f"{task}\n\n{mail}"


def answer_text(output: str, role: str, files: list[dict[str, Any]]) -> str:
    """A member's answer as the leader reads it (the result of its `team_assign` call): its words, cut, and the names of the
    files it left, which are in the leader's workspace under `.team/<role>/`. A cut keeps the end of a reply that ends with a
    handover block (`fit_handover`)."""
    text = fit_handover((output or "").strip(), RESULT_CHARS) or "(the member answered nothing)"
    names = [str(item["name"]) for item in files][:FILES_LISTED]
    if names:
        text += f"\n\nFiles {role} left, readable in your workspace under .team/{role}/: " + ", ".join(names)
    return text


def with_blocks(text: str, mail: str, user: str) -> str:
    """The leader's tool result with what else it should see at this turn: the team's notes and the user's messages."""
    return "\n\n".join(part for part in (text, mail, user) if part)


def user_block(texts: list[str]) -> str:
    return f"{USER_HEADING}:\n" + "\n".join(f"- {' '.join(text.split())[:2000]}" for text in texts) if texts else ""


MESSAGE_CHARS = 2000


def resolve_mentions(spec: TeamStageSpec, text: str, explicit: list[str] | None, speaker: str = "") -> list[str]:
    """The roles an utterance is addressed to. The explicit list wins (a role or a label each); without one, `@role` and `@label`
    in the text, found by name (not as the start of a longer word). Only roles of the team count; the speaker is never one."""
    members = list(spec.members)
    by_name = {name: member.role for member in members for name in (member.role, member.label) if name}
    if explicit:
        found = [by_name[name] for name in explicit if name in by_name]
    else:
        found = [
            member.role
            for member in members
            if any(name and re.search(rf"@{re.escape(name)}(?![A-Za-z0-9_-])", text) for name in (member.role, member.label))
        ]
    return [role for role in dict.fromkeys(found) if role != speaker]


def wake_prompt(sender: str, text: str) -> str:
    """What a member is told when it is @-mentioned: who said it, and what."""
    return f"团队成员 {sender} 在群里 @ 了你: {text}"


LEADER_NAME = "领队"
USER_NAME = "用户"


def display_names(spec: TeamStageSpec) -> dict[str, str]:
    """What each role is called to the agents: its label, else 领队 for the leader, else the role id; the user is 用户."""
    names = {m.role: m.label or (LEADER_NAME if m.role == spec.leader else m.role) for m in spec.members}
    names["user"] = USER_NAME
    return names
