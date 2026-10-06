"""What an agent profile contributes to a task attempt's Agent (15 T8.3).

A profile spec is versioned and immutable, so reading it at the start of an attempt is replay-safe. Three keys matter
here: `instructions` extend the system prompt, `model` renames the model of a real provider, and `mcp_connectors` are
specs that hold names only, never secret values. Anything unusable is skipped with a log line: a bad profile must not
fail every attempt that runs under it.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from orbit_contracts.v3 import Team
from pydantic import ValidationError

from orbit_worker.mcp_connectors import parse_stored, specs_for_storage
from orbit_worker.settings import MIN_CONTEXT_SIZE

logger = logging.getLogger(__name__)

_override: contextvars.ContextVar[AgentConfig | None] = contextvars.ContextVar("orbit_agent_config_override", default=None)


@contextlib.contextmanager
def agent_config_override(config: AgentConfig) -> Iterator[None]:
    """Run the agents built inside as `config` (its instructions and model) instead of the task attempt's own: a SOP step's
    verifier is its own expert's, whatever the attempt it judges ran as."""
    token = _override.set(config)
    try:
        yield
    finally:
        _override.reset(token)


def overriding_agent_config() -> AgentConfig | None:
    return _override.get()

MAX_INSTRUCTIONS_CHARS = 20_000
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,99}")


@dataclass(frozen=True)
class AgentConfig:
    instructions: str = ""
    # SOUL.md of the expert's bundle (ADR-0013): persona and tone, put before the instructions in the system prompt.
    soul: str = ""
    model: str = ""
    mcp_connectors: tuple[dict[str, Any], ...] = ()
    # Catalog skill ids to stage for the attempt (15 T8.4). Staging skips what it cannot use.
    skills: tuple[str, ...] = ()
    # Names of the skills in the expert's own bundle, in load order, and the profile ref ("id@version") they are fetched
    # from. A bundle skill is staged exactly like a catalog skill (ADR-0013).
    bundle_skills: tuple[str, ...] = ()
    bundle_ref: str = ""
    # Only the team's leader has one: it is what lets the leader give a node to a member (15 T8.6).
    team: Team | None = None
    # The task has a team and this attempt is not its leader's (a member's node of the plan): it has no `team`, yet it is not a
    # single agent either, so what only the leader and a lone agent have (the checklist the user sees) is not its.
    in_team_member: bool = False
    # The attempt is the leader's own work node, a task the leader gave itself: it executes instead of planning, so it is told
    # `OWN_TASK_PROMPT` and has no tool that changes the plan. `team` stays set (it is still the leader's expert).
    own_task: bool = False
    # The model's context window, when the profile sets one in `model_params` (it overrides ORBIT_MODEL_CONTEXT_SIZE).
    context_size: int | None = None
    # What the profile's model costs, in micro-dollars per million input and output tokens (`model_params`); it replaces the
    # worker's `ORBIT_MODEL_PRICE_*`. Known only when both are set.
    price_input_per_mtok: int | None = None
    price_output_per_mtok: int | None = None


def agent_config_from_spec(spec: dict[str, Any]) -> AgentConfig:
    soul, instructions = _prompt_texts(spec.get("soul"), spec.get("instructions"))
    return AgentConfig(
        instructions=instructions,
        soul=soul,
        bundle_skills=_skill_ids(spec.get("bundle_skills")),
        model=_model(spec.get("model")),
        mcp_connectors=_connectors(spec.get("mcp_connectors")),
        skills=_skill_ids(spec.get("skills")),
        context_size=_context_size(spec.get("model_params")),
        price_input_per_mtok=_price(spec.get("model_params"), "price_input_per_mtok"),
        price_output_per_mtok=_price(spec.get("model_params"), "price_output_per_mtok"),
    )


OWN_TASK_PROMPT = (
    "This is a task you gave yourself: do it now in your workspace and answer with a short handover (what you did, files, "
    "open issues). Do not create tasks."
)


def with_task_config(
    base: AgentConfig, raw: dict[str, Any] | None, profile_ref: str = "", *, own_task: bool = False
) -> AgentConfig:
    """The task's own choices on top of the expert's. A connector list the task sets replaces the expert's (an empty
    list removes them); one it leaves open (None) keeps the expert's. The connectors get the same checks as a profile's:
    the task config comes from control, but a worker does not take a launch target on trust. `own_task` says the attempt is the
    leader's own work node (`is_leader_work_node`): it gets `OWN_TASK_PROMPT` instead of the leader's."""
    if not raw:
        return base
    config = base
    if raw.get("connectors") is not None:
        config = dataclasses.replace(config, mcp_connectors=_connectors(raw["connectors"]))
    if raw.get("skills") is not None:
        config = dataclasses.replace(config, skills=_skill_ids(raw["skills"]))
    if raw.get("model") and (model := _model(raw["model"])):
        # An invalid name reads as empty and keeps the expert's model rather than clearing it.
        config = dataclasses.replace(config, model=model)
    team, is_member = _team_for(raw.get("team"), profile_ref)
    if team is not None:
        prompt = OWN_TASK_PROMPT if own_task else team_prompt(team)
        config = dataclasses.replace(
            config, team=team, own_task=own_task, instructions=f"{config.instructions}\n\n{prompt}".strip()
        )
    elif is_member:
        config = dataclasses.replace(config, in_team_member=True)
    return config


def permission_preset_for(raw: dict[str, Any] | None) -> str:
    """The preset of the attempt's own expert's mode, for everyone alike, a team's leader too: questions only ("ask") runs
    read-only, every other mode, and none, keeps write permission. Whether a leader does work itself is its expert's
    configuration, not a rule of the team."""
    if raw and raw.get("mode") == "ask":
        return "read-only"
    return "workspace-write"


def _prompt_texts(soul: Any, instructions: Any) -> tuple[str, str]:
    """SOUL.md and AGENTS.md share one limit. Control refuses a longer pair when it is saved; this only guards: the soul
    is kept first, the instructions get what is left."""
    soul_text = soul.strip() if isinstance(soul, str) else ""
    text = instructions.strip() if isinstance(instructions, str) else ""
    if len(soul_text) + len(text) <= MAX_INSTRUCTIONS_CHARS:
        return soul_text, text
    logger.warning("profile soul and instructions cut to %d characters together", MAX_INSTRUCTIONS_CHARS)
    soul_text = soul_text[:MAX_INSTRUCTIONS_CHARS]
    return soul_text, text[: MAX_INSTRUCTIONS_CHARS - len(soul_text)]


def _model(value: Any) -> str:
    if isinstance(value, str) and _MODEL_NAME.fullmatch(value):
        return value
    if value:
        logger.warning("profile model name ignored: not a model name")
    return ""


def _context_size(params: Any) -> int | None:
    value = params.get("context_size") if isinstance(params, dict) else None
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= MIN_CONTEXT_SIZE:
        return value
    logger.warning("profile context_size ignored: not a whole number of at least %d", MIN_CONTEXT_SIZE)
    return None


def _price(params: Any, key: str) -> int | None:
    value = params.get(key) if isinstance(params, dict) else None
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    logger.warning("profile %s ignored: not a whole number of micro-dollars", key)
    return None


def _team_for(raw: Any, profile_ref: str) -> tuple[Team | None, bool]:
    """The team, for the attempt that is the leader's; a member's attempt, and any other, is not shown it. The flag says the
    task has a valid team and the attempt is not the leader's."""
    if not isinstance(raw, dict):
        return None, False
    try:
        team = Team.model_validate(raw)
    except ValidationError:
        logger.warning("task team ignored: not a valid team")
        return None, False
    leader = next((member for member in team.members if member.role == team.leader), None)
    if leader is None:
        logger.warning("task team ignored: its leader is not a member")
        return None, False
    if leader.expert == profile_ref:
        return team, False
    return None, True


def team_prompt(team: Team) -> str:
    """What the leader is told about its team. A description is put on one line so that it cannot add structure."""
    lines = [
        (
            "You lead a team. Clarify the goal, plan it, and split it into tasks with TaskCreate. "
            "Do work yourself when it fits your own role and skills or is too small to be worth handing off, and give a member "
            "what fits its role better. Set each task's metadata to "
            '{"owner": "<role>"}: a member\'s role, or your own role for work you do yourself. '
            "If your workspace is read-only you only plan, coordinate and review, so give every task to a member. "
            "Integration, documentation, scripts and verification are tasks too: give them to whoever is best suited. "
            "Hands-on verification is split by area: each one verifies and reports on its own part (the backend member runs "
            "its API checks, the frontend member runs install and build). A cross-cutting piece, such as an end-to-end check or "
            "a README, goes to the one best suited to it, and its brief says which parts belong to the others. Do not "
            "bundle unrelated work into one task. Write the brief of a member's task in full (members cannot see this "
            "conversation). Accepting the whole result is your own review turn, not a member's task. "
            "In anything you write (to the user, in briefs, in reviews) call members by their label; the owner id is only for "
            "TaskCreate metadata, never shown. "
            "In a planning turn write one short message at the end (what was split and to whom), not a line per step. "
            "Tasks run after you end your turn, so create them and end your turn. "
            "When they are done you review the results in your review turn and, if something is missing "
            "or wrong, create further tasks."
        ),
        "Your team (you are the first):",
    ]
    ordered = sorted(team.members, key=lambda member: member.role != team.leader)
    for member in ordered:
        description = " ".join(member.description.split())
        label = " ".join(member.label.split())
        you = " (you)" if member.role == team.leader else ""
        # The label first, as the user knows the member; the id only where TaskCreate needs it.
        name = f"{label} (owner id: {member.role})" if label else f"{member.role}"
        lines.append(f"- {name}{you}: {description}" if description else f"- {name}{you}")
    return "\n".join(lines)


def owner_profile_for(team: Team | None, metadata: Any) -> tuple[str | None, str]:
    """The expert of the role a TaskCreate names in its metadata, or an error saying what is wrong. Without a team no owner
    is no error: the node is the attempt's own. In a team the owner is required: a member's role, or the leader's own role
    for work the leader does itself (it runs as the leader's expert)."""
    if team is None:
        if isinstance(metadata, dict) and "owner" in metadata:
            return None, "there is no team: leave out the owner, the task is yours"
        return None, ""
    roles = ", ".join(member.role for member in team.members)
    owner = metadata.get("owner") if isinstance(metadata, dict) else None
    if not isinstance(owner, str) or not owner:
        return None, f"in a team every task has an owner: set metadata {{\"owner\": ...}} to one of {roles}"
    member = next((item for item in team.members if item.role == owner), None)
    if member is None:
        return None, f"unknown role {owner!r}: the owner is one of {roles}"
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
