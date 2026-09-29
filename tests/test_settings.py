from __future__ import annotations

import pytest
from orbit_orch.settings import TemporalSettings
from orbit_worker.settings import WorkerSettings, WorkspaceSettings
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
