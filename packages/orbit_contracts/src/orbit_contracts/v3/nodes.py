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
    # Kind-specific settings, e.g. {"command": "pytest -q", "timeout_s": 600}.
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


class TeamStageSpec(ContractModel):
    """Registered but disabled in phase 1 (07, A20)."""

    objective: str = Field(min_length=1)
    member_profiles: list[VersionedRef] = Field(default_factory=list)


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


class _NodeDraftBase(ContractModel):
    node_id: NodeRef
    type_version: int = Field(default=1, ge=1)
    title: str = Field(min_length=1, max_length=200)
    depends_on: list[NodeRef] = Field(default_factory=list)
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
