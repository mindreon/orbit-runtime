"""Tool Gateway Activity. Credential references only; never secret values."""

from orbit_contracts.models import GatewayExecuteInput, GatewayExecuteOutput

from orbit_worker.secrets import reject_secret_values


async def execute_gateway(inp: GatewayExecuteInput) -> GatewayExecuteOutput:
    reject_secret_values(inp.arguments)
    if inp.credential_ref.startswith("sk-"):
        raise ValueError("credential reference must not be a secret value")
    if inp.tool_name == "gateway_charge":
        amount = inp.arguments.get("amount", "")
        return GatewayExecuteOutput(
            output=f"charged {amount}",
            metadata={"ok": "true"},
        )
    if inp.tool_name == "gateway_lookup":
        return GatewayExecuteOutput(
            output="lookup-ok",
            metadata={"ok": "true", "value": inp.arguments.get("query", "")},
        )
    return GatewayExecuteOutput(
        output=f"unsupported tool {inp.tool_name}",
        metadata={},
        result_state="error",
    )
