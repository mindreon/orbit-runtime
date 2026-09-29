"""Environment settings of the worker process, read and validated once at startup."""

from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    # The health listener.
    bind: str = Field("0.0.0.0", validation_alias="ORBIT_WORKER_BIND")
    port: int = Field(8090, validation_alias="ORBIT_WORKER_PORT")
    # An activity cancellation (an interrupt, a cancel) reaches a running turn on its next heartbeat, so this
    # bounds how long an interrupt takes to land.
    heartbeat_throttle_s: float = Field(5.0, validation_alias="ORBIT_HEARTBEAT_THROTTLE_S", gt=0)
    # Where events go; empty keeps them in this process.
    event_ingest_url: str = Field("", validation_alias="ORBIT_EVENT_INGEST_URL")
    internal_token: str = Field("", validation_alias="ORBIT_INTERNAL_TOKEN", repr=False)
    # `schema://<name>/<version>` is the file `<dir>/<name>/<version>.json`. Empty means no schema can be resolved,
    # so a node that names an output schema cannot be completed (04 §5).
    output_schema_dir: str = Field("", validation_alias="ORBIT_OUTPUT_SCHEMA_DIR")


class WorkspaceSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    backend: Literal["local", "docker", "opensandbox"] = Field(
        "local", validation_alias=AliasChoices("ORBIT_WORKSPACE_BACKEND", "ORBIT_ISOLATION_MODE")
    )
    ttl_seconds: int = Field(300, validation_alias="ORBIT_WORKSPACE_TTL_SECONDS", gt=0)
    root: str = Field("/tmp/orbit-workspaces", validation_alias="ORBIT_WORKSPACE_ROOT")
    image: str = Field("python:3.11-slim", validation_alias="ORBIT_WORKSPACE_IMAGE")
    opensandbox_domain: str | None = Field(None, validation_alias="ORBIT_OPENSANDBOX_DOMAIN")
    opensandbox_api_key: str | None = Field(
        None, validation_alias="ORBIT_OPENSANDBOX_API_KEY", repr=False
    )
    opensandbox_protocol: str = Field("http", validation_alias="ORBIT_OPENSANDBOX_PROTOCOL")
    opensandbox_server_proxy: bool = Field(False, validation_alias="ORBIT_OPENSANDBOX_SERVER_PROXY")
    opensandbox_image: str = Field(
        "ghcr.io/mindreon/orbit-sandbox:latest",
        validation_alias=AliasChoices("ORBIT_OPENSANDBOX_IMAGE", "ORBIT_SANDBOX_IMAGE"),
    )

    @field_validator("backend", mode="before")
    @classmethod
    def _normalize_backend(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("opensandbox_domain", "opensandbox_api_key", mode="before")
    @classmethod
    def _empty_is_unset(cls, value: object) -> object:
        return value or None
