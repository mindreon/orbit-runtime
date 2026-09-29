from __future__ import annotations

import pytest
from orbit_orch.settings import MaintenanceSettings, TemporalSettings, VersioningSettings
from orbit_worker.isolation import isolation_from_settings
from orbit_worker.settings import (
    IsolationSettings,
    McpSettings,
    MockSettings,
    StoreSettings,
    WorkerSettings,
    WorkspaceSettings,
)
from pydantic import ValidationError


def test_defaults_need_no_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "TEMPORAL_ADDRESS",
        "ORBIT_WORKSPACE_BACKEND",
        "ORBIT_ISOLATION_MODE",
        "ORBIT_WORKER_PORT",
    ):
        monkeypatch.delenv(name, raising=False)
    assert TemporalSettings().address == "localhost:7233"
    assert TemporalSettings().io_queue == "orbit.io"
    assert WorkspaceSettings().backend == "local"
    assert WorkerSettings().port == 8090


def test_workspace_backend_falls_back_to_the_isolation_mode_and_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ORBIT_WORKSPACE_BACKEND", raising=False)
    monkeypatch.setenv("ORBIT_ISOLATION_MODE", " Docker ")
    assert WorkspaceSettings().backend == "docker"
    monkeypatch.setenv("ORBIT_WORKSPACE_BACKEND", "opensandbox")
    assert WorkspaceSettings().backend == "opensandbox"


def test_an_unknown_backend_or_bad_number_stops_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORBIT_WORKSPACE_BACKEND", "firecracker")
    with pytest.raises(ValidationError):
        WorkspaceSettings()
    monkeypatch.delenv("ORBIT_WORKSPACE_BACKEND")
    monkeypatch.setenv("ORBIT_WORKSPACE_TTL_SECONDS", "0")
    with pytest.raises(ValidationError):
        WorkspaceSettings()


def test_empty_opensandbox_values_are_unset_and_secrets_are_not_printed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORBIT_OPENSANDBOX_DOMAIN", "")
    monkeypatch.setenv("ORBIT_OPENSANDBOX_API_KEY", "super-secret-key")
    monkeypatch.setenv("ORBIT_INTERNAL_TOKEN", "tok-secret")
    settings = WorkspaceSettings()
    assert settings.opensandbox_domain is None
    assert "super-secret-key" not in repr(settings)
    assert "tok-secret" not in repr(WorkerSettings())


def test_output_schema_dir_is_optional_and_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORBIT_OUTPUT_SCHEMA_DIR", raising=False)
    assert WorkerSettings().output_schema_dir == ""
    monkeypatch.setenv("ORBIT_OUTPUT_SCHEMA_DIR", "/etc/orbit/schemas")
    assert WorkerSettings().output_schema_dir == "/etc/orbit/schemas"


def test_docker_workspace_limits_have_defaults_and_can_be_set(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ORBIT_WORKSPACE_DOCKER_CPUS",
        "ORBIT_WORKSPACE_DOCKER_MEMORY",
        "ORBIT_WORKSPACE_DOCKER_PIDS_LIMIT",
    ):
        monkeypatch.delenv(name, raising=False)
    default = WorkspaceSettings()
    assert (default.docker_cpus, default.docker_memory, default.docker_pids_limit) == (1.0, "1g", 256)
    monkeypatch.setenv("ORBIT_WORKSPACE_DOCKER_CPUS", "0.5")
    monkeypatch.setenv("ORBIT_WORKSPACE_DOCKER_MEMORY", "512M")
    monkeypatch.setenv("ORBIT_WORKSPACE_DOCKER_PIDS_LIMIT", "64")
    tuned = WorkspaceSettings()
    assert (tuned.docker_cpus, tuned.docker_memory, tuned.docker_pids_limit) == (0.5, "512m", 64)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ORBIT_WORKSPACE_DOCKER_CPUS", "0"),
        ("ORBIT_WORKSPACE_DOCKER_CPUS", "-1"),
        ("ORBIT_WORKSPACE_DOCKER_MEMORY", "lots"),
        ("ORBIT_WORKSPACE_DOCKER_MEMORY", "0"),
        ("ORBIT_WORKSPACE_DOCKER_MEMORY", ""),
        ("ORBIT_WORKSPACE_DOCKER_PIDS_LIMIT", "0"),
        ("ORBIT_WORKSPACE_DOCKER_PIDS_LIMIT", "-1"),
    ],
)
def test_a_bad_docker_limit_stops_startup(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        WorkspaceSettings()


def test_store_settings_reject_a_bad_fernet_key_without_printing_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORBIT_CHECKPOINT_FERNET_KEY", "not-a-key-secret")
    with pytest.raises(ValidationError) as caught:
        StoreSettings()
    assert "not-a-key-secret" not in str(caught.value)
    monkeypatch.delenv("ORBIT_CHECKPOINT_FERNET_KEY")
    assert StoreSettings().checkpoint_fernet_key == ""


def test_store_settings_have_the_documented_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ORBIT_CONTROL_WORKER_DB_URL",
        "ORBIT_CHECKPOINT_DIR",
        "ORBIT_OBJECT_STORE_ENDPOINT",
        "ORBIT_OBJECT_STORE_BUCKET",
        "ORBIT_OBJECT_STORE_SECURE",
        "ORBIT_OBJECT_STORE_REGION",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = StoreSettings()
    assert (settings.checkpoint_dir, settings.object_store_bucket, settings.object_store_region) == (
        ".orbit-checkpoints",
        "orbit",
        "us-east-1",
    )
    assert not settings.object_store_secure
    monkeypatch.setenv("ORBIT_OBJECT_STORE_SECURE", "1")
    assert StoreSettings().object_store_secure


def test_isolation_settings_default_to_local_and_reject_bad_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    for name in ("ORBIT_ISOLATION_MODE", "ORBIT_BWRAP_SHARE_NET", "ORBIT_ISOLATION_STRICT", "ORBIT_CGROUP_APPLY"):
        monkeypatch.delenv(name, raising=False)
    settings = IsolationSettings()
    assert (settings.mode, settings.share_net, settings.strict, settings.cgroup_apply) == ("local", False, False, False)
    assert isolation_from_settings(settings, tmp_path).backend == "local"
    monkeypatch.setenv("ORBIT_ISOLATION_STRICT", "sometimes")
    with pytest.raises(ValidationError):
        IsolationSettings()


def test_mock_settings_fall_back_and_normalize(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ORBIT_MODEL_MODE", "ORBIT_MOCK_SOP_STEP_DELAY_MS", "ORBIT_MOCK_TURN_DELAY_MS"):
        monkeypatch.delenv(name, raising=False)
    assert MockSettings().mock
    assert MockSettings().sop_step_delay_ms is None
    monkeypatch.setenv("ORBIT_MODEL_MODE", " Real ")
    assert not MockSettings().mock
    monkeypatch.setenv("ORBIT_MODEL_MODE", "")
    assert MockSettings().mock
    monkeypatch.setenv("ORBIT_MODEL_MODE", "maybe")
    with pytest.raises(ValidationError):
        MockSettings()
    monkeypatch.delenv("ORBIT_MODEL_MODE")
    monkeypatch.setenv("ORBIT_MOCK_TURN_DELAY_MS", "-5")
    with pytest.raises(ValidationError):
        MockSettings()


def test_mcp_prefixes_split_on_commas_and_default_when_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORBIT_MCP_ALLOWED_ENV_PREFIXES", raising=False)
    assert McpSettings().allowed_env_prefixes == ("ORBIT_MCP_",)
    monkeypatch.setenv("ORBIT_MCP_ALLOWED_ENV_PREFIXES", " A_ , B_,, ")
    assert McpSettings().allowed_env_prefixes == ("A_", "B_")
    monkeypatch.setenv("ORBIT_MCP_ALLOWED_ENV_PREFIXES", " , ")
    assert McpSettings().allowed_env_prefixes == ("ORBIT_MCP_",)


def test_maintenance_and_versioning_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ORBIT_MAINTENANCE_ENABLED",
        "ORBIT_MAINTENANCE_REAP_SECONDS",
        "ORBIT_USE_WORKER_VERSIONING",
        "ORBIT_WORKER_DEPLOYMENT",
        "ORBIT_WORKER_BUILD_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    maintenance = MaintenanceSettings()
    assert (maintenance.enabled, maintenance.reap_seconds, maintenance.gc_seconds) == (True, 300, 86400)
    monkeypatch.setenv("ORBIT_MAINTENANCE_ENABLED", "0")
    monkeypatch.setenv("ORBIT_MAINTENANCE_REAP_SECONDS", "0")
    with pytest.raises(ValidationError):
        MaintenanceSettings()
    versioning = VersioningSettings()
    assert (versioning.enabled, versioning.deployment, versioning.build_id) == (False, "orbit", "dev")
    monkeypatch.setenv("ORBIT_USE_WORKER_VERSIONING", "1")
    monkeypatch.setenv("ORBIT_WORKER_BUILD_ID", " ")
    with pytest.raises(ValidationError):
        VersioningSettings()
    monkeypatch.setenv("ORBIT_USE_WORKER_VERSIONING", "maybe")
    with pytest.raises(ValidationError):
        VersioningSettings()
