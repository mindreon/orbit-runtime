"""Task nodes as proposed in a plan change (03 §4, node-type registry v1 in 05 §5)."""

from typing import Annotated, Literal, Self

from pydantic import Field, JsonValue, model_validator

from orbit_contracts.v3.common import (
    ArtifactUri,
    Budget,
    ContractModel,
    Count,
    NodeRef,
    Risk,
    VersionedRef,
    go_union,
)

NodeType = Literal["agent_turn", "sop_stage", "team_stage", "approval", "wait", "checkpoint"]
WorkspaceAccess = Literal["write", "read", "none"]
VerificationKind = Literal["command", "schema", "human", "sop_verifier"]


class ArtifactRequirement(ContractModel):
    name: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    min_count: int = Field(default=1, ge=1)


class Verification(ContractModel):
    kind: VerificationKind
    # Kind-specific settings, e.g. {"command": "pytest -q", "timeout_s": 600}. A `sop_verifier` (an independent verifier agent
    # judges the step, 06 §2) carries {"sop", "sop_name", "step_id", "subject", "description", "instructions", "expert"?}.
    spec: dict[str, JsonValue] = Field(default_factory=dict)


class CompletionContract(ContractModel):
    output_schema_ref: Annotated[str, Field(pattern=r"^schema://\S+/[1-9][0-9]*$")] | None = None
    required_artifacts: list[ArtifactRequirement] = Field(default_factory=list)
    verifications: list[Verification] = Field(default_factory=list)
    allowed_side_effects: list[str] = Field(default_factory=list)


class NodeTimeout(ContractModel):
    attempt_s: Count | None = None
    node_s: Count | None = None


class RetryPolicy(ContractModel):
    max_attempts: int = Field(default=3, ge=1)
    backoff_s: Count = 30
    # Switch to this profile after repeated verification failures (11 §3).
    repair_profile: VersionedRef | None = None


class AgentTurnSpec(ContractModel):
    goal: str = Field(min_length=1)
    inputs: list[ArtifactUri] = Field(default_factory=list)


class SopStageSpec(ContractModel):
    sop: VersionedRef
    inputs: list[ArtifactUri] = Field(default_factory=list)


_ROLE = r"^[a-z][a-z0-9_-]{0,31}$"


class TeamStageMember(ContractModel):
    """One member of a team stage: the role the leader assigns work to, and the profile (an expert) that does it."""

    role: Annotated[str, Field(pattern=_ROLE)]
    executor: VersionedRef
    description: str = Field(default="", max_length=300)
    label: str = Field(default="", max_length=40)


class TeamStageLimits(ContractModel):
    """What bounds one team stage (07 §4). Whichever is reached first ends the stage with a reason, never silently."""

    # Members of the team, the leader included.
    max_members: int = Field(default=4, ge=1, le=8)
    # Turns of the leader.
    max_rounds: int = Field(default=10, ge=1, le=30)
    # Assignments, results and notes exchanged inside the stage.
    max_messages: int = Field(default=100, ge=1, le=200)
    # Longest chain of member-to-member wakes (a member @-mentioning another, who mentions another...) inside one leader round.
    max_hops: int = Field(default=3, ge=0, le=8)


class TeamStageSpec(ContractModel):
    """A bounded team inside one attempt (07): a leader and its members work on `goal` together. The leader runs as the
    `leader` member's executor and hands work to the others; what it finishes with is the stage's result. `workspace_access`
    is what the members get of the task workspace: a copy of its latest snapshot that is thrown away (`read`), or nothing
    (`none`). Only the leader writes the task's workspace (the node's own `workspace_access`)."""

    goal: str = Field(min_length=1)
    members: list[TeamStageMember] = Field(min_length=2, max_length=8)
    leader: Annotated[str, Field(pattern=_ROLE)]
    limits: TeamStageLimits = TeamStageLimits()
    workspace_access: Literal["read", "none"] = "read"
    inputs: list[ArtifactUri] = Field(default_factory=list)

    @model_validator(mode="after")
    def _a_team(self) -> Self:
        roles = [member.role for member in self.members]
        if len(set(roles)) != len(roles):
            raise ValueError("the roles of a team stage are unique")
        if self.leader not in roles:
            raise ValueError("the leader of a team stage is one of its members")
        if len(roles) > self.limits.max_members:
            raise ValueError(f"a team stage of at most {self.limits.max_members} members has {len(roles)}")
        return self


class ApprovalSpec(ContractModel):
    summary: str = Field(min_length=1)
    risk: Risk = "medium"


class WaitSpec(ContractModel):
    wait_key: str | None = Field(default=None, min_length=1)
    timer_s: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _needs_a_condition(self) -> Self:
        if self.wait_key is None and self.timer_s is None:
            raise ValueError("a wait node needs wait_key or timer_s")
        return self


class CheckpointSpec(ContractModel):
    label: str = Field(min_length=1)


class SopStepInfo(ContractModel):
    """Which part of a compiled SOP a node is (the plan keeps no other trace of the SOP): `role` `sop` is the `sop_stage`
    node itself, `step` one step's agent node, `approval_before` and `approval_after` the approval nodes of a step with
    `human_approval`. `index` (1-based) and `total` count the steps: "SOP X: step i/n"."""

    sop: VersionedRef
    role: Literal["sop", "step", "approval_before", "approval_after"]
    total: int = Field(ge=1)
    step_id: str | None = None
    index: int | None = Field(default=None, ge=1)
    subject: str | None = None


class TeamStageInfo(ContractModel):
    """What a view of a `team_stage` node says about its stage: the limits it runs under (from its spec, so nobody assumes the
    defaults). The counters ({round, messages}) are on the stage's `team.*` events (`team.round_started` and `team.round_finished`)."""

    max_members: int = Field(ge=1)
    max_rounds: int = Field(ge=1)
    max_messages: int = Field(ge=1)
    max_hops: int = Field(default=3, ge=0)


class _NodeDraftBase(ContractModel):
    node_id: NodeRef
    type_version: int = Field(default=1, ge=1)
    title: str = Field(min_length=1, max_length=200)
    depends_on: list[NodeRef] = Field(default_factory=list)
    # The node this one is nested under (a team member's work under the leader's node). Its chain is the node's depth
    # (03 §3 invariant 4); `depends_on` is about order, not nesting.
    parent_node_id: NodeRef | None = None
    # None means the registry default for the node type (08 §1).
    workspace_access: WorkspaceAccess | None = None
    owner_profile: VersionedRef | None = None
    budget: Budget = Budget()
    timeout: NodeTimeout | None = None
    retry: RetryPolicy | None = None
    completion_contract: CompletionContract = CompletionContract()


class AgentTurnNode(_NodeDraftBase):
    type: Literal["agent_turn"] = "agent_turn"
    spec: AgentTurnSpec


class SopStageNode(_NodeDraftBase):
    type: Literal["sop_stage"] = "sop_stage"
    spec: SopStageSpec


class TeamStageNode(_NodeDraftBase):
    type: Literal["team_stage"] = "team_stage"
    spec: TeamStageSpec


class ApprovalNode(_NodeDraftBase):
    type: Literal["approval"] = "approval"
    spec: ApprovalSpec


class WaitNode(_NodeDraftBase):
    type: Literal["wait"] = "wait"
    spec: WaitSpec


class CheckpointNode(_NodeDraftBase):
    type: Literal["checkpoint"] = "checkpoint"
    spec: CheckpointSpec


TaskNodeDraft = Annotated[
    AgentTurnNode | SopStageNode | TeamStageNode | ApprovalNode | WaitNode | CheckpointNode,
    go_union("TaskNodeDraft", "type"),
]
