"""The one AgentRuntime of a worker process, installed at startup and read by the activities."""

from orbit_worker.runtime import AgentRuntime

_runtime: AgentRuntime | None = None


def set_runtime(runtime: AgentRuntime) -> None:
    """One runtime per worker process. Tests install their own before polling."""

    global _runtime
    _runtime = runtime


def get_runtime() -> AgentRuntime:
    if _runtime is None:
        raise RuntimeError("AgentRuntime is not installed on this worker")
    return _runtime
