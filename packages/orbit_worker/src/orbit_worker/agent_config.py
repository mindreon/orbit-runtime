"""What an agent profile contributes to a task attempt's Agent (15 T8.3).

A profile spec is versioned and immutable, so reading it at the start of an attempt is replay-safe. Three keys matter
here: `instructions` extend the system prompt, `model` renames the model of a real provider, and `mcp_connectors` are
specs that hold names only, never secret values. Anything unusable is skipped with a log line: a bad profile must not
fail every attempt that runs under it.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from dataclasses import dataclass
from typing import Any

from orbit_contracts.v3 import Team
from pydantic import ValidationError

from orbit_worker.mcp_connectors import parse_stored, specs_for_storage

logger = logging.getLogger(__name__)

MAX_INSTRUCTIONS_CHARS = 20_000
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,99}")


@dataclass(frozen=True)
class AgentConfig:
    instructions: str = ""
    model: str = ""
    mcp_connectors: tuple[dict[str, Any], ...] = ()
    # Catalog skill ids to stage for the attempt (15 T8.4). Staging skips what it cannot use.
    skills: tuple[str, ...] = ()
    # Only the team's leader has one: it is what lets the leader give a node to a member (15 T8.6).
    team: Team | None = None


def agent_config_from_spec(spec: dict[str, Any]) -> AgentConfig:
    return AgentConfig(
        instructions=_instructions(spec.get("instructions")),
        model=_model(spec.get("model")),
        mcp_connectors=_connectors(spec.get("mcp_connectors")),
        skills=_skill_ids(spec.get("skills")),
    )


def with_task_config(base: AgentConfig, raw: dict[str, Any] | None, profile_ref: str = "") -> AgentConfig:
    """The task's own choices on top of the expert's. A connector list the task sets replaces the expert's (an empty
    list removes them); one it leaves open (None) keeps the expert's. The connectors get the same checks as a profile's:
    the task config comes from control, but a worker does not take a launch target on trust."""
    if not raw:
        return base
    config = base
    if raw.get("connectors") is not None:
        config = dataclasses.replace(config, mcp_connectors=_connectors(raw["connectors"]))
    if raw.get("skills") is not None:
        config = dataclasses.replace(config, skills=_skill_ids(raw["skills"]))
    team = _team_for(raw.get("team"), profile_ref)
    if team is not None:
        config = dataclasses.replace(config, team=team, instructions=f"{config.instructions}\n\n{team_prompt(team)}".strip())
    return config


def permission_preset_for(raw: dict[str, Any] | None) -> str:
    """Questions only ("ask") runs read-only. Every other mode, and none, keeps write permission."""
    return "read-only" if raw and raw.get("mode") == "ask" else "workspace-write"


def _instructions(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if len(text) > MAX_INSTRUCTIONS_CHARS:
        logger.warning("profile instructions cut to %d characters", MAX_INSTRUCTIONS_CHARS)
        return text[:MAX_INSTRUCTIONS_CHARS]
    return text


def _model(value: Any) -> str:
    if isinstance(value, str) and _MODEL_NAME.fullmatch(value):
        return value
    if value:
        logger.warning("profile model name ignored: not a model name")
    return ""


def _team_for(raw: Any, profile_ref: str) -> Team | None:
    """The team, for the attempt that is the leader's. A member's attempt, and any other, is not shown it."""
    if not isinstance(raw, dict):
        return None
    try:
        team = Team.model_validate(raw)
    except ValidationError:
        logger.warning("task team ignored: not a valid team")
        return None
    leader = next((member for member in team.members if member.role == team.leader), None)
    if leader is None:
        logger.warning("task team ignored: its leader is not a member")
        return None
    return team if leader.expert == profile_ref else None


def team_prompt(team: Team) -> str:
    """What the leader is told about its team. A description is put on one line so that it cannot add structure."""
    lines = [
        (
            "You lead a team. When you create a task with TaskCreate, give it to the member who should do it by "
            'setting its metadata to {"owner": "<role>"}; a task without an owner is yours.'
        ),
        "Your team:",
    ]
    for member in team.members:
        description = " ".join(member.description.split())
        lines.append(f"- {member.role}: {description}" if description else f"- {member.role}")
    return "\n".join(lines)


def owner_profile_for(team: Team | None, metadata: Any) -> tuple[str | None, str]:
    """The expert of the role a TaskCreate names in its metadata, or an error saying what is wrong. No owner is no error:
    the node is the leader's."""
    if not isinstance(metadata, dict) or "owner" not in metadata:
        return None, ""
    owner = metadata["owner"]
    if team is None:
        return None, "there is no team: leave out the owner, the task is yours"
    roles = ", ".join(member.role for member in team.members)
    if not isinstance(owner, str) or not owner:
        return None, f"owner must be one of the team's roles: {roles}"
    member = next((item for item in team.members if item.role == owner), None)
    if member is None:
        return None, f"unknown role {owner!r}: the team's roles are {roles}"
    return member.expert, ""


def _skill_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(dict.fromkeys(item for item in value if isinstance(item, str) and item))


def _connectors(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    items = [item for item in value if isinstance(item, dict)]
    return tuple(specs_for_storage(parse_stored(items)))
