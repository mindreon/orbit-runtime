"""Environment settings both processes share: where Temporal is and which queues carry what."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TemporalSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    address: str = Field("localhost:7233", validation_alias="TEMPORAL_ADDRESS")
    namespace: str = Field("default", validation_alias="TEMPORAL_NAMESPACE")
    # orbit-orch polls this queue; orbit-worker never does.
    orch_queue: str = Field("orbit.orch", validation_alias="TEMPORAL_TASK_QUEUE")
    agent_queue: str = Field("orbit.agent", validation_alias="ORBIT_AGENT_TASK_QUEUE")
    io_queue: str = Field("orbit.io", validation_alias="ORBIT_IO_TASK_QUEUE")
