"""What an agent may do without asking: the one place the permission presets of a task are given their meaning.

Control and the workflow only carry a `PermissionSpec` (`orbit_contracts.v3.PermissionSpec`); every semantic below lives here.
A session's plan is fixed when it opens, so a changed preset takes effect with the next attempt.

    preset    engine mode    edits in /workspace   Bash commands                              skill scripts
    default   ACCEPT_EDITS   auto                  AgentScope decides (read-only auto)        ask
    request   DEFAULT        ask                   AgentScope decides (read-only auto)        ask
    auto      ACCEPT_EDITS   auto                  low and medium auto, high asks             auto
    full      BYPASS         auto, anywhere        auto, except what the notes below say      auto
    custom    from the switches (see `plan_for`)

Notes for every preset:
  * A catastrophic command (`command_risk`) is denied, never asked. It is checked in `OrbitPermissionMiddleware`, whatever the
    engine decided, so no mode, rule or earlier approval lets it through.
  * `mode: "ask"` of the task or expert config opens the session read-only (`read-only`, EXPLORE); that always wins over the
    preset, which is then ignored.
  * `full` runs in BYPASS, which skips the safety asks of AgentScope's tools. Two of them are kept: a write to a dangerous file
    or directory (`.env`, `.ssh`, `.git`, shell start-up files, ... AgentScope's own lists) still asks, for Write and Edit as for a
    Bash command that writes one. The other safety asks (`$(...)`, `rm -rf` patterns, sed) are skipped: the classifier already
    reads those, and it denies what is catastrophic.
  * `auto` and `auto_commands` allow a command the classifier calls low or medium, and ask for a high one. Medium is the
    ordinary work of an agent (builds, installs into the workspace, git commit, python script.py); high is what leaves it
    (privilege, global installs, force and recursive deletes, pipes into a shell, uploads, dotfiles).
  * With `write_scope: "none"` Write and Edit are denied by rule, a command known to write is denied, and `auto_commands`
    allows only low commands, because a medium one (`python x.py`) may write.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from agentscope.permission import PermissionBehavior, PermissionMode, PermissionRule
from orbit_contracts.v3 import PermissionSpec

from orbit_worker.command_risk import assess, is_protected_path

logger = logging.getLogger(__name__)

Risk = Literal["low", "medium", "high"]
# The preset names of the session contract (`OpenSessionInput.permission_preset`). A task's preset is `workspace-write` plus
# its `PermissionSpec`; `read-only` is the mode-"ask" override; `danger-full-access` is kept for callers that name it.
SESSION_PRESETS = frozenset({"workspace-write", "read-only", "danger-full-access"})
FILE_TOOLS = frozenset({"Write", "Edit"})
CATASTROPHIC_DENIED = "已拒绝：灾难性命令（{reason}）"
WRITES_DENIED = "已拒绝：此任务不允许写入文件（{reason}）"
PROTECTED_ASK = "需要确认：此操作会修改受保护的文件或目录（{reason}）"


@dataclass(frozen=True)
class PermissionPlan:
    mode: PermissionMode
    # No Write, no Edit, no command known to write.
    write_scope: Literal["none", "workspace"] = "workspace"
    auto_edits: bool = True
    auto_commands: bool = False
    auto_builtin: bool = False
    # BYPASS: everything is auto except what the middleware denies or asks.
    full: bool = False
    read_only: bool = False

    def deny_rules(self) -> dict[str, list[PermissionRule]]:
        """The engine's own rules for what is refused outright: file tools of a task that may not write."""
        if self.write_scope != "none" or self.read_only:
            return {}
        return {
            name: [PermissionRule(tool_name=name, rule_content=None, behavior=PermissionBehavior.DENY, source="task-permissions")]
            for name in sorted(FILE_TOOLS)
        }


def plan_for(session_preset: str, spec: PermissionSpec | dict[str, Any] | None = None) -> PermissionPlan:
    """The plan of a session. `session_preset` is the session contract's (`SESSION_PRESETS`); `spec` is the task's
    `PermissionSpec` (absent: preset "default"). Raises `ValueError` for an unknown session preset."""
    if session_preset not in SESSION_PRESETS:
        raise ValueError(f"unknown permission preset: {session_preset}")
    if session_preset == "read-only":
        return PermissionPlan(mode=PermissionMode.EXPLORE, write_scope="none", auto_edits=False, read_only=True)
    if session_preset == "danger-full-access":
        return PermissionPlan(mode=PermissionMode.BYPASS, full=True, auto_commands=True, auto_builtin=True)
    parsed = _spec(spec)
    match parsed.preset:
        case "request":
            return PermissionPlan(mode=PermissionMode.DEFAULT, auto_edits=False)
        case "auto":
            return PermissionPlan(mode=PermissionMode.ACCEPT_EDITS, auto_commands=True, auto_builtin=True)
        case "full":
            return PermissionPlan(mode=PermissionMode.BYPASS, full=True, auto_commands=True, auto_builtin=True)
        case "custom":
            # Nothing is auto-edited when nothing may be written, and a mode that auto-allows filesystem commands would
            # only be one more thing to reason about.
            edits = parsed.auto_edits and parsed.write_scope == "workspace"
            return PermissionPlan(
                mode=PermissionMode.ACCEPT_EDITS if edits else PermissionMode.DEFAULT,
                write_scope=parsed.write_scope,
                auto_edits=parsed.auto_edits,
                auto_commands=parsed.auto_commands,
                auto_builtin=parsed.auto_builtin,
            )
        case _:
            return PermissionPlan(mode=PermissionMode.ACCEPT_EDITS)


def _spec(spec: PermissionSpec | dict[str, Any] | None) -> PermissionSpec:
    if spec is None:
        return PermissionSpec(preset="default")
    if isinstance(spec, PermissionSpec):
        return spec
    try:
        return PermissionSpec.model_validate(spec)
    except ValueError:
        # Control validates the config; a worker that cannot read it takes the preset every task had before.
        logger.warning("task permissions ignored: not a valid permission spec")
        return PermissionSpec(preset="default")


def call_risk(tool_name: str, arguments: dict[str, str]) -> Risk:
    """The risk shown on the approval of a tool call. A Bash command is classified; a write to a protected file is high;
    everything else (MCP and custom tools, ordinary file edits) is medium. Catastrophic never reaches an approval, but if
    one is asked about it reads as high."""
    if tool_name == "Bash":
        level = assess(arguments.get("command", "")).level
        return "high" if level == "catastrophic" else level
    if tool_name in FILE_TOOLS and is_protected_path(arguments.get("file_path", "")):
        return "high"
    return "medium"
