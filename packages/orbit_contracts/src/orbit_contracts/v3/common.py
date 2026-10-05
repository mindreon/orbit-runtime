"""Shared building blocks for contract v3.

Wire names are snake_case. Every model rejects unknown fields and is frozen;
build a new instance instead of changing one. Unions carry ``x-go-type`` so
orbit-control can generate a named Go type for them.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

CONTRACT_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    populate_by_name=True,
    serialize_by_alias=True,
)


class ContractModel(BaseModel):
    model_config = CONTRACT_CONFIG


def go_union(name: str, discriminator: str) -> Any:
    """Field metadata for a discriminated union that Go generates as ``name``."""
    return Field(discriminator=discriminator, json_schema_extra={"x-go-type": name})


_ULID = r"[0-9A-HJKMNP-TV-Z]{26}"
_SHA256_HEX = r"[0-9a-f]{64}"


def _prefixed(prefix: str) -> Any:
    return StringConstraints(pattern=rf"^{prefix}_{_ULID}$")


# Control mints task ids as a canonical UUIDv7 (time-ordered). A ULID task id is still accepted: workflows started
# before that change, and the golden histories that prove their replay, carry one. Every other id stays a ULID.
_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
TaskId = Annotated[str, StringConstraints(pattern=rf"^task_({_ULID}|{_UUID})$")]
NodeId = Annotated[str, _prefixed("n")]
AttemptId = Annotated[str, _prefixed("att")]
ApprovalId = Annotated[str, _prefixed("apr")]
EventId = Annotated[str, _prefixed("evt")]
ManifestId = Annotated[str, _prefixed("man")]
CheckpointId = Annotated[str, _prefixed("ckpt")]
# A node id, or a placeholder "tmp:<n>" that the plan engine maps in id_map.
NodeRef = Annotated[str, StringConstraints(pattern=rf"^(n_{_ULID}|tmp:[1-9][0-9]*)$")]
# Control mints client command ids as a UUIDv7 (a ULID is still accepted); agent commands use
# sha256(attempt_id | tool_call_id).
CommandId = Annotated[str, StringConstraints(pattern=rf"^({_ULID}|{_UUID}|{_SHA256_HEX})$")]
Sha256Ref = Annotated[str, StringConstraints(pattern=rf"^sha256:{_SHA256_HEX}$")]
# "<id>@<version>" for profiles and SOPs, e.g. "coder@3".
VersionedRef = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*@[1-9][0-9]*$")
]
ArtifactUri = Annotated[str, StringConstraints(pattern=r"^artifact://\S+$")]
NonEmptyText = Annotated[str, StringConstraints(min_length=1, max_length=32_000)]
Count = Annotated[int, Field(ge=0)]

ActorKind = Literal["user", "agent", "system"]
TaskStatus = Literal[
    "CREATED",
    "PLANNING",
    "RUNNING",
    "WAITING",
    "PAUSED",
    "PAUSED_NEEDS_REVIEW",
    "TAKEN_OVER",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
]
NodeStatus = Literal[
    "PENDING",
    "READY",
    "RUNNING",
    "AWAITING_APPROVAL",
    "AWAITING_INPUT",
    "PROPOSED",
    "VERIFYING",
    "COMPLETED",
    "RETRY_PENDING",
    "BLOCKED",
    "SKIPPED",
    "CANCELLED",
]
AttemptStatus = Literal[
    "STARTING",
    "RUNNING",
    "PARKED_HITL",
    "PARKED_INPUT",
    "HANDOVER",
    "VERIFYING",
    "ACCEPTED",
    "REJECTED",
    "ABORTED",
    "LOST",
]
FailureClass = Literal["transient", "model", "tool", "policy", "budget", "verification", "lost"]
Risk = Literal["low", "medium", "high"]


class Actor(ContractModel):
    kind: ActorKind
    id: str = Field(min_length=1)
    attempt_id: AttemptId | None = None
    profile: VersionedRef | None = None


class Budget(ContractModel):
    """Limits. A missing field means no limit at this level.

    Money is in micro-dollars so both languages keep it exact.
    """

    tokens: Count | None = None
    tool_calls: Count | None = None
    wall_s: Count | None = None
    cost_usd_micros: Count | None = None


class Policy(ContractModel):
    """What an attempt may do (05 §6). Every layer (tenant, task, profile) can carry one; layers only tighten each
    other: `denied_tools` add up and the smallest `exploration_max_tool_calls` wins. A missing field limits nothing."""

    denied_tools: list[str] = Field(default_factory=list)
    exploration_max_tool_calls: Count | None = None
    # How many attempts of the task may run at once (04 §3). Missing means the orchestrator's default (4).
    max_concurrency: int | None = Field(default=None, ge=1)


class PermissionRuleSpec(ContractModel):
    """A permission rule a person allowed for the rest of a task: this tool, and what it is called with (a command
    prefix for Bash, a path pattern for Write). It is what AgentScope suggests with the approval."""

    tool_name: str = Field(min_length=1)
    rule_content: str | None = None


class HeaderRef(ContractModel):
    name: str = Field(min_length=1)
    env: str = Field(min_length=1)  # an environment variable name, never its value


class ConnectorSnapshot(ContractModel):
    """An MCP connector as the worker connects to it (15 M8): names and launch targets, never secret values.
    Control resolves the tenant's connector to this when the configuration is set, so a running task keeps the
    connector it was given even if the tenant's list changes."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    transport: Literal["stdio", "streamable_http"] = "stdio"
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env_refs: list[str] = Field(default_factory=list)
    url: str = ""
    header_refs: list[HeaderRef] = Field(default_factory=list)


ConfigMode = Literal["default", "plan", "ask"]


class TeamMember(ContractModel):
    """One member of a team (15 M8, T8.6): a role the leader can give work to, and the expert who does it."""

    role: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")
    expert: VersionedRef
    description: str = Field(default="", max_length=300)


class Team(ContractModel):
    """A leader and members. The leader's expert plans the task; a node the leader gives to a role runs as that member's
    expert, and one it gives to nobody runs as the leader. Teams do not nest (07 §1: depth 1)."""

    leader: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")
    members: list[TeamMember] = Field(min_length=1, max_length=8)
    # How deep nodes may nest under a parent node (03 §3 invariant 4): 1 lets the leader give nodes to members.
    max_depth: int = Field(default=1, ge=1, le=8)


class TaskConfig(ContractModel):
    """What a task runs with, beside its goal (15 M8). `expert` replaces the task's profile for nodes that do not
    name their own; `skills` and `connectors` are the complete sets to use, and None means the expert's defaults.
    A change takes effect from the next attempt, never in the one that is running (11 §3)."""

    config_version: int = Field(default=1, ge=1)
    expert: VersionedRef | None = None
    skills: list[str] | None = None
    connectors: list[ConnectorSnapshot] | None = None
    mode: ConfigMode = "default"
    # Set when `expert` is a team: control resolves it, so the workflow needs no database to know the members.
    team: Team | None = None


class Usage(ContractModel):
    tokens_in: Count = 0
    tokens_out: Count = 0
    tool_calls: Count = 0
    wall_s: Count = 0
    # None is "unknown": the model has no price the worker knows, so a cost is neither recorded nor enforced (05 §4).
    cost_usd_micros: Count | None = None


class Failure(ContractModel):
    failure_class: FailureClass
    retryable: bool
    message: str
