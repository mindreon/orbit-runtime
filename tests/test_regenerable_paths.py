"""Dependencies, build output and caches are neither artifacts nor part of a workspace snapshot."""

import io
import tarfile
from pathlib import Path

import pytest
from orbit_worker.sandbox import files_in_archive
from orbit_worker.workspace import LocalWorkspaceAdapter, _pack
from orbit_worker.workspace_paths import TAR_EXCLUDES, is_regenerable


@pytest.mark.parametrize(
    "path",
    [
        "node_modules/react/index.js", "web/node_modules/.bin/vite", "a/.venv/lib/x.py", "venv/bin/python",
        "pkg/__pycache__/m.cpython-311.pyc", "m.pyc", ".git/HEAD", ".cache/x", ".pytest_cache/v/x", ".mypy_cache/x",
        ".ruff_cache/x", "dist/app.js", "web/build/out.js", ".next/server.js", ".turbo/log", "coverage/lcov.info",
        "lib/python3.11/site-packages/x.py", "package-lock.json", "web/pnpm-lock.yaml", "yarn.lock", "uv.lock",
        "backend/poetry.lock",
    ],
)
def test_regenerable_paths_are_recognised(path: str) -> None:
    assert is_regenerable(path)


@pytest.mark.parametrize("path", ["src/app.tsx", "README.md", "backend/main.py", "docs/build-notes.md", "distance.txt"])
def test_deliverables_are_not_regenerable(path: str) -> None:
    assert not is_regenerable(path)


def _archive(names: list[str]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as tar:
        for name in names:
            info = tarfile.TarInfo(name=name)
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
    return output.getvalue()


def test_the_artifact_list_leaves_out_dependencies_and_keeps_the_rest() -> None:
    names = ["src/app.tsx", "node_modules/react/index.js", "dist/app.js", "package-lock.json", "api/main.py", "api/x.pyc"]
    assert [file.name for file in files_in_archive(_archive(names))] == ["api/main.py", "src/app.tsx"]


def _write(root: Path, name: str, data: bytes = b"x") -> None:
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def test_pack_leaves_regenerable_paths_out(tmp_path: Path) -> None:
    for name in ["src/app.tsx", "node_modules/esbuild/bin/esbuild.node", "api/__pycache__/m.pyc", "dist/a.js", "uv.lock", "api/m.py"]:
        _write(tmp_path, name)
    with tarfile.open(fileobj=io.BytesIO(_pack(tmp_path)), mode="r:gz") as tar:
        assert sorted(tar.getnames()) == ["api/m.py", "src/app.tsx"]


async def test_a_resumed_workspace_has_the_sources_without_the_dependencies(tmp_path: Path) -> None:
    adapter = LocalWorkspaceAdapter(tmp_path)
    lease = await adapter.acquire("tenant-a", "task-a")
    workspace = tmp_path / "tenant-a" / lease.workspace_id
    _write(workspace, "src/app.tsx", b"app")
    _write(workspace, "node_modules/big/x.node", b"\0" * 1024)
    snapshot = await adapter.snapshot(lease)
    # What the snapshot holds is only the sources, and restoring it still works: the dependencies are installed again.
    assert len(await adapter.load_snapshot("tenant-a", snapshot)) < 400
    await adapter.release(lease)
    again = await adapter.acquire("tenant-a", "task-a")
    await adapter.restore(again, snapshot)
    restored = tmp_path / "tenant-a" / again.workspace_id
    assert (restored / "src" / "app.tsx").read_bytes() == b"app"
    assert not (restored / "node_modules").exists()
    await adapter.release(again)


def test_a_sandbox_that_tars_its_own_workspace_is_told_what_to_skip() -> None:
    assert "--exclude=node_modules" in TAR_EXCLUDES
    assert "--exclude=uv.lock" in TAR_EXCLUDES
    assert "--exclude=*.pyc" in TAR_EXCLUDES
