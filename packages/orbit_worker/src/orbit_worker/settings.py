"""Environment settings of the worker process, read and validated once at startup.

The variable names are the deployment contract. This module is the only place the worker reads its environment,
apart from `process_environment()` for values whose *names* come from data.
"""

import os
import re
from collections.abc import Mapping
from typing import Annotated, Literal

from cryptography.fernet import Fernet
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from orbit_worker.workspace import (
    DEFAULT_DOCKER_CPUS,
    DEFAULT_DOCKER_MEMORY,
    DEFAULT_DOCKER_PIDS_LIMIT,
)

# `docker run --memory`: a positive number with an optional b, k, m or g.
_DOCKER_MEMORY = re.compile(r"[1-9][0-9]*[bkmg]?")


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
    # Limits of one docker sandbox container (17 G5).
    docker_cpus: float = Field(DEFAULT_DOCKER_CPUS, validation_alias="ORBIT_WORKSPACE_DOCKER_CPUS", gt=0)
    docker_memory: str = Field(DEFAULT_DOCKER_MEMORY, validation_alias="ORBIT_WORKSPACE_DOCKER_MEMORY")
    docker_pids_limit: int = Field(
        DEFAULT_DOCKER_PIDS_LIMIT, validation_alias="ORBIT_WORKSPACE_DOCKER_PIDS_LIMIT", gt=0
    )
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

    @field_validator("docker_memory", mode="after")
    @classmethod
    def _memory_is_a_docker_size(cls, value: str) -> str:
        size = value.strip().lower()
        if not _DOCKER_MEMORY.fullmatch(size):
            raise ValueError("must be a positive number with an optional b, k, m or g, like 512m or 2g")
        return size

    @field_validator("opensandbox_domain", "opensandbox_api_key", mode="before")
    @classmethod
    def _empty_is_unset(cls, value: object) -> object:
        return value or None


class IsolationSettings(BaseSettings):
    """The isolation mode the agent turns run under (`orbit_worker.isolation`)."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    mode: str = Field("local", validation_alias="ORBIT_ISOLATION_MODE")
    share_net: bool = Field(False, validation_alias="ORBIT_BWRAP_SHARE_NET")
    strict: bool = Field(False, validation_alias="ORBIT_ISOLATION_STRICT")
    work_root: str = Field("/tmp/orbit-workspaces", validation_alias="ORBIT_WORK_ROOT")
    sandbox_image: str = Field("", validation_alias="ORBIT_SANDBOX_IMAGE")
    cgroup_cpu: str = Field("", validation_alias="ORBIT_CGROUP_CPU")
    cgroup_memory: str = Field("", validation_alias="ORBIT_CGROUP_MEMORY")
    cgroup_apply: bool = Field(False, validation_alias="ORBIT_CGROUP_APPLY")


class StoreSettings(BaseSettings):
    """Where checkpoints and their blobs live (`orbit_worker.task_store`)."""

    # The keys and the database URL are secrets: a validation error must not echo them.
    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True, hide_input_in_errors=True)

    # Empty runs without Postgres, for a process that only needs the blob directory.
    control_worker_db_url: str = Field("", validation_alias="ORBIT_CONTROL_WORKER_DB_URL", repr=False)
    checkpoint_dir: str = Field(".orbit-checkpoints", validation_alias="ORBIT_CHECKPOINT_DIR")
    # Empty stores blobs unencrypted.
    checkpoint_fernet_key: str = Field("", validation_alias="ORBIT_CHECKPOINT_FERNET_KEY", repr=False)
    # Empty keeps blobs in `checkpoint_dir`; otherwise they go to this S3-compatible endpoint.
    object_store_endpoint: str = Field("", validation_alias="ORBIT_OBJECT_STORE_ENDPOINT")
    object_store_bucket: str = Field("orbit", validation_alias="ORBIT_OBJECT_STORE_BUCKET")
    object_store_access_key: str = Field("", validation_alias="ORBIT_OBJECT_STORE_ACCESS_KEY", repr=False)
    object_store_secret_key: str = Field("", validation_alias="ORBIT_OBJECT_STORE_SECRET_KEY", repr=False)
    object_store_secure: bool = Field(False, validation_alias="ORBIT_OBJECT_STORE_SECURE")
    # A fixed region skips the GetBucketLocation probe, which the worker's prefix-limited policy denies.
    object_store_region: str = Field("us-east-1", validation_alias="ORBIT_OBJECT_STORE_REGION")

    @field_validator("checkpoint_fernet_key", mode="after")
    @classmethod
    def _key_is_a_fernet_key(cls, value: str) -> str:
        if value:
            try:
                Fernet(value.encode("ascii"))
            except (ValueError, UnicodeEncodeError):
                # Never echo the key.
                raise ValueError("must be a url-safe base64 Fernet key (32 bytes)") from None
        return value


class MockSettings(BaseSettings):
    """The mock model and its test knobs. Read where they are used, so a test can set them per call."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    model_mode: Literal["mock", "real"] = Field("mock", validation_alias="ORBIT_MODEL_MODE")
    tool_log: str = Field("", validation_alias="ORBIT_MOCK_TOOL_LOG")
    tool_delay_ms: int = Field(0, validation_alias="ORBIT_MOCK_TOOL_DELAY_MS", ge=0)
    turn_delay_ms: int = Field(0, validation_alias="ORBIT_MOCK_TURN_DELAY_MS", ge=0)
    # Unset means the turn delay, which a SOP step also uses.
    sop_step_delay_ms: int | None = Field(None, validation_alias="ORBIT_MOCK_SOP_STEP_DELAY_MS", ge=0)
    stream_delay_ms: int = Field(0, validation_alias="ORBIT_MOCK_STREAM_DELAY_MS", ge=0)
    # E2E only. Unset in production, so a resume is not delayed.
    e2e_resolve_delay_s: float = Field(0.0, validation_alias="ORBIT_E2E_RESOLVE_DELAY_S", ge=0)

    @field_validator("model_mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: object) -> object:
        # An empty value means the default, as `chat_model.resolve_model_config` reads it.
        return (value.strip().lower() or "mock") if isinstance(value, str) else value

    @property
    def mock(self) -> bool:
        return self.model_mode == "mock"


class McpSettings(BaseSettings):
    """Which environment variables an MCP connector may name as its secrets."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    allowed_env_prefixes: Annotated[tuple[str, ...], NoDecode] = Field(
        ("ORBIT_MCP_",), validation_alias="ORBIT_MCP_ALLOWED_ENV_PREFIXES"
    )

    @field_validator("allowed_env_prefixes", mode="before")
    @classmethod
    def _split(cls, value: object) -> object:
        if isinstance(value, str):
            items = tuple(item.strip() for item in value.split(",") if item.strip())
            return items or ("ORBIT_MCP_",)
        return value


def process_environment() -> Mapping[str, str]:
    """The raw process environment, for values whose names are data and so cannot be settings fields: the
    environment variables an MCP connector names as its secrets, the model endpoint variables `chat_model` checks
    by name, and `PATH` for a child process."""
    return os.environ
