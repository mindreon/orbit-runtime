"""Workflow sandbox restrictions shared by the orch process and tests."""

from temporalio.worker.workflow_sandbox import (
    SandboxedWorkflowRunner,
    SandboxRestrictions,
)


def sandbox_runner() -> SandboxedWorkflowRunner:
    """Allow pydantic and the contract package inside the workflow sandbox.

    The pass-through list is reviewed: it must stay free of agentscope,
    LLM SDKs, and the worker package.
    """

    return SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(
            "pydantic",
            "orbit_contracts",
        )
    )
