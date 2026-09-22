"""Workspace isolation. Bubblewrap must not share the host network."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IsolationSnapshot:
    mode: str
    share_net: bool
    backend: str
    image_digest: str
    cpu_max: str
    memory_max: str
    cgroup_applied: bool


def prepare_isolation(
    *,
    mode: str,
    share_net: bool,
    strict: bool,
    root: Path,
    image_digest: str = "",
    cpu_max: str = "",
    memory_max: str = "",
    apply_cgroup: bool = False,
) -> IsolationSnapshot:
    """Build the workspace backend and record what the session will run under.

    ``bwrap`` always passes ``share_net=False`` into BubblewrapBackend.
    AgentScope's own default is shared networking; Orbit does not use it.
    """

    root.mkdir(parents=True, exist_ok=True)
    applied = False
    if mode == "local":
        backend = "local"
    elif mode == "bwrap":
        if share_net:
            raise RuntimeError("Bubblewrap must run with share_net=False")
        if strict and shutil.which("bwrap") is None:
            raise RuntimeError("bwrap is not installed and ORBIT_ISOLATION_STRICT=1")
        _build_bubblewrap(root)
        backend = "bwrap"
        share_net = False
    elif mode in ("docker", "k8s"):
        if not image_digest:
            raise RuntimeError(f"{mode} isolation requires a pre-baked image digest")
        backend = mode
    else:
        raise RuntimeError(f"unknown isolation mode: {mode}")
    if apply_cgroup:
        applied = _write_cgroup(root / "cgroup", cpu_max, memory_max)
    return IsolationSnapshot(
        mode=mode,
        share_net=share_net,
        backend=backend,
        image_digest=image_digest,
        cpu_max=cpu_max,
        memory_max=memory_max,
        cgroup_applied=applied,
    )


def _build_bubblewrap(root: Path) -> None:
    from agentscope.workspace import BubblewrapBackend

    BubblewrapBackend(
        host_workdir=str(root / "workspace"),
        host_tmpdir=str(root / "tmp"),
        share_net=False,
    )


def _write_cgroup(path: Path, cpu_max: str, memory_max: str) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        if cpu_max:
            (path / "cpu.max").write_text(cpu_max + "\n", encoding="utf-8")
        if memory_max:
            (path / "memory.max").write_text(memory_max + "\n", encoding="utf-8")
    except OSError:
        return False
    return True


def isolation_from_env(root: Path | None = None) -> IsolationSnapshot:
    mode = os.environ.get("ORBIT_ISOLATION_MODE", "local")
    share_net = os.environ.get("ORBIT_BWRAP_SHARE_NET", "0") == "1"
    strict = os.environ.get("ORBIT_ISOLATION_STRICT", "0") == "1"
    base = root or Path(os.environ.get("ORBIT_WORK_ROOT", "/tmp/orbit-workspaces"))
    return prepare_isolation(
        mode=mode,
        share_net=share_net,
        strict=strict,
        root=base / "_probe",
        image_digest=os.environ.get("ORBIT_SANDBOX_IMAGE", ""),
        cpu_max=os.environ.get("ORBIT_CGROUP_CPU", ""),
        memory_max=os.environ.get("ORBIT_CGROUP_MEMORY", ""),
        apply_cgroup=os.environ.get("ORBIT_CGROUP_APPLY", "0") == "1",
    )
