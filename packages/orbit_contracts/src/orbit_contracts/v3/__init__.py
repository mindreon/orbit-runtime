"""Contract v3: TaskWorkflow commands, messages, views and events.

Snake_case on the wire, unknown fields rejected, models frozen. The v2 room
contracts in ``orbit_contracts.models`` stay until the cutover (ADR-0011).
"""

from orbit_contracts.v3.common import Actor, Budget, Failure, Usage
from orbit_contracts.v3.events import (
    DURABLE_EVENT_TYPES,
    EPHEMERAL_EVENT_TYPES,
    EntityRef,
    Event,
    EventSource,
)
from orbit_contracts.v3.messages import (
    ApprovalDecidedSignal,
    AttemptFinishedSignal,
    AttemptParkedSignal,
    CompletionProposal,
    CompletionResult,
    DecideApprovalInput,
    DeliverMessagesSignal,
    ExternalEventSignal,
    GrantBudgetInput,
    RequestProfileSwitchInput,
    SendMessageInput,
    SendMessageResult,
    TaskControlInput,
)
from orbit_contracts.v3.nodes import TaskNodeDraft, WaitSpec
from orbit_contracts.v3.plan import PlanChangeCommand, PlanChangeResult, PlanOp
from orbit_contracts.v3.views import InboxView, PlanView, TaskView, TaskWorkflowInput

__all__ = [
    "DURABLE_EVENT_TYPES",
    "EPHEMERAL_EVENT_TYPES",
    "Actor",
    "ApprovalDecidedSignal",
    "AttemptFinishedSignal",
    "AttemptParkedSignal",
    "Budget",
    "CompletionProposal",
    "CompletionResult",
    "DecideApprovalInput",
    "DeliverMessagesSignal",
    "EntityRef",
    "Event",
    "EventSource",
    "ExternalEventSignal",
    "Failure",
    "GrantBudgetInput",
    "InboxView",
    "PlanChangeCommand",
    "PlanChangeResult",
    "PlanOp",
    "PlanView",
    "RequestProfileSwitchInput",
    "SendMessageInput",
    "SendMessageResult",
    "TaskControlInput",
    "TaskNodeDraft",
    "TaskView",
    "TaskWorkflowInput",
    "Usage",
    "WaitSpec",
]
