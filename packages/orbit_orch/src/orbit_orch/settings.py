"""Environment settings of the orchestrator, and the ones it shares with the worker.

Every process reads its environment here and nowhere else, once at startup, so a bad or missing value stops the
process before it polls a queue. The variable names are the deployment contract and do not change.
"""

from functools import lru_cache
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TemporalSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    address: str = Field("localhost:7233", validation_alias="TEMPORAL_ADDRESS")
    namespace: str = Field("default", validation_alias="TEMPORAL_NAMESPACE")
    # orbit-orch polls this queue; orbit-worker never does.
    orch_queue: str = Field("orbit.orch", validation_alias="TEMPORAL_TASK_QUEUE")
    agent_queue: str = Field("orbit.agent", validation_alias="ORBIT_AGENT_TASK_QUEUE")
    io_queue: str = Field("orbit.io", validation_alias="ORBIT_IO_TASK_QUEUE")


class VersioningSettings(BaseSettings):
    """Worker Versioning. orbit-orch and orbit-worker must agree on all three values (see `orbit_orch.versioning`)."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    enabled: bool = Field(False, validation_alias="ORBIT_USE_WORKER_VERSIONING")
    deployment: str = Field("orbit", validation_alias="ORBIT_WORKER_DEPLOYMENT")
    build_id: str = Field("dev", validation_alias="ORBIT_WORKER_BUILD_ID")

    @model_validator(mode="after")
    def _names_are_set_when_enabled(self) -> Self:
        if self.enabled and not (self.deployment.strip() and self.build_id.strip()):
            raise ValueError(
                "ORBIT_USE_WORKER_VERSIONING=1 needs non-empty ORBIT_WORKER_DEPLOYMENT and ORBIT_WORKER_BUILD_ID"
            )
        return self


@lru_cache(maxsize=1)
def versioning_settings() -> VersioningSettings:
    """The process's one reading of the versioning switch.

    `task_workflow` calls it when it is imported to choose each workflow's versioning behavior, and the worker
    registration reads the same object, so the two cannot disagree inside a process. It is read once, never while a
    workflow runs, so it cannot make a replay differ.
    """
    return VersioningSettings()


class MaintenanceSettings(BaseSettings):
    """The three maintenance Schedules orbit-orch registers (17 G7)."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    enabled: bool = Field(True, validation_alias="ORBIT_MAINTENANCE_ENABLED")
    reap_seconds: int = Field(300, validation_alias="ORBIT_MAINTENANCE_REAP_SECONDS", gt=0)
    gc_seconds: int = Field(86400, validation_alias="ORBIT_MAINTENANCE_GC_SECONDS", gt=0)
    attempts_seconds: int = Field(86400, validation_alias="ORBIT_MAINTENANCE_ATTEMPTS_SECONDS", gt=0)
