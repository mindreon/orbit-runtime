from pathlib import Path

import pytest
from orbit_worker.workspace import (
    LocalWorkspaceAdapter,
    OpenSandboxWorkspaceAdapter,
    PersistentWorkspaceAdapter,
)


class FakeLeaseStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def acquire_workspace_lease(self, **kwargs):
        self.calls.append(("acquire", kwargs))

    async def renew_workspace_lease(self, **kwargs):
        self.calls.append(("renew", kwargs))

    async def release_workspace_lease(self, **kwargs):
        self.calls.append(("release", kwargs))


@pytest.mark.asyncio
async def test_workspace_lease_snapshot_restore_and_archive_roundtrip(tmp_path: Path) -> None:
    adapter = LocalWorkspaceAdapter(tmp_path)
    lease = await adapter.acquire("tenant-a", "task-a")
    workspace = tmp_path / "tenant-a" / lease.workspace_id
    (workspace / "src").mkdir()
    (workspace / "src" / "hello.txt").write_text("one", encoding="utf-8")
    snapshot = await adapter.snapshot(lease)
    (workspace / "src" / "hello.txt").write_text("two", encoding="utf-8")
    await adapter.restore(lease, snapshot)
    assert (workspace / "src" / "hello.txt").read_text(encoding="utf-8") == "one"
    archive = await adapter.get_archive(lease)
    (workspace / "src" / "hello.txt").unlink()
    await adapter.put_archive(lease, archive)
    assert (workspace / "src" / "hello.txt").read_text(encoding="utf-8") == "one"
    await adapter.release(lease)


@pytest.mark.asyncio
async def test_read_only_workspace_has_an_independent_id(tmp_path: Path) -> None:
    adapter = LocalWorkspaceAdapter(tmp_path)
    writer = await adapter.acquire("tenant-a", "task-a")
    reader = await adapter.acquire("tenant-a", "task-a", read_only=True)
    assert writer.workspace_id != reader.workspace_id
    assert reader.read_only is True
    await adapter.release(reader)
    await adapter.release(writer)


@pytest.mark.asyncio
async def test_workspace_lease_is_persisted(tmp_path: Path) -> None:
    store = FakeLeaseStore()
    adapter = PersistentWorkspaceAdapter(LocalWorkspaceAdapter(tmp_path), store)
    lease = await adapter.acquire("tenant-a", "task-a")
    await adapter.renew(lease, ttl_s=60)
    await adapter.release(lease)
    assert [name for name, _ in store.calls] == ["acquire", "renew", "release"]
    assert store.calls[0][1]["lease_key"].startswith("task-a/w/ws_"), "a workspace has a key of its own"


@pytest.mark.asyncio
async def test_opensandbox_uses_sdk_lifecycle_and_content_addressed_snapshots(monkeypatch) -> None:
    import opensandbox

    class FakeStatus:
        state = "Running"

    class FakeInfo:
        id = "sandbox-1"
        status = FakeStatus()

    class FakeExecution:
        exit_code = 0

    class FakeFiles:
        def __init__(self) -> None:
            self.payload = b"archive"

        async def write_file(self, _path, data):
            self.payload = data

        async def read_bytes_stream(self, _path):
            yield self.payload

    class FakeCommands:
        async def run(self, _command):
            return FakeExecution()

    class FakeSandbox:
        def __init__(self):
            self.id = "sandbox-1"
            self.files = FakeFiles()
            self.commands = FakeCommands()
            self.paused = False
            self.renewed = False

        async def renew(self, _timeout):
            self.renewed = True

        async def pause(self):
            self.paused = True

        async def kill(self):
            self.paused = True

        @classmethod
        async def create(cls, **_kwargs):
            return cls()

        @classmethod
        async def connect(cls, _sandbox_id, _config):
            return cls()

        @classmethod
        async def resume(cls, _sandbox_id, _config):
            return cls()

    class FakeManager:
        async def list_sandbox_infos(self, _filter):
            return type("Page", (), {"sandbox_infos": []})()

        @classmethod
        async def create(cls, _config):
            return cls()

    monkeypatch.setattr(opensandbox, "Sandbox", FakeSandbox)
    monkeypatch.setattr(opensandbox, "SandboxManager", FakeManager)

    class SnapshotStore:
        def __init__(self) -> None:
            self.items = {}

        async def put_snapshot(self, *, tenant_id, digest, payload):
            self.items[(tenant_id, digest)] = payload

        async def get_snapshot(self, *, tenant_id, digest):
            return self.items[(tenant_id, digest)]

    adapter = OpenSandboxWorkspaceAdapter(
        connection_config=object(), image="orbit:test", snapshot_store=SnapshotStore()
    )
    lease = await adapter.acquire("tenant-a", "task-a")
    renewed = await adapter.renew(lease, ttl_s=60)
    assert renewed.expires_at != lease.expires_at
    snapshot = await adapter.snapshot(renewed)
    assert snapshot.startswith("sha256:")
    await adapter.restore(renewed, snapshot)
    await adapter.release(renewed)
