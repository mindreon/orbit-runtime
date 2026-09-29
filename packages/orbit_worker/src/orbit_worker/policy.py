"""The layers of policy and how they combine (05 §6): tenant, then task, then profile.

Each layer can only tighten the ones before it. The outermost layer is not a default the inner ones override: a tool
any layer denies stays denied, and the smallest exploration cap of all layers is the cap.
"""

from __future__ import annotations

from typing import Any

from orbit_contracts.v3 import Policy


def merge(*layers: Policy) -> Policy:
    """The effective policy of layers given outermost first: denials add up, the tightest cap wins."""
    denied: list[str] = []
    caps: list[int] = []
    for layer in layers:
        denied.extend(name for name in layer.denied_tools if name not in denied)
        if layer.exploration_max_tool_calls is not None:
            caps.append(layer.exploration_max_tool_calls)
    return Policy(denied_tools=denied, exploration_max_tool_calls=min(caps) if caps else None)


def from_profile_spec(spec: dict[str, Any]) -> Policy:
    """The profile layer: `tools.denied` and `exploration.max_tool_calls` of an agent profile."""
    cap = (spec.get("exploration") or {}).get("max_tool_calls")
    return Policy(
        denied_tools=[str(name) for name in (spec.get("tools") or {}).get("denied") or []],
        exploration_max_tool_calls=int(cap) if isinstance(cap, int) else None,
    )
