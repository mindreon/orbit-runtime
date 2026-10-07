"""Which paths of a workspace are dependencies, build output or caches: the one rule for the artifact list and the snapshot.

Such a path is regenerable (`npm install`, `uv sync`, a build), so it is neither a deliverable nor worth keeping in the
workspace snapshot: a `node_modules` alone is hundreds of files and tens of MB. The files stay in the live workspace; they are
only not listed as artifacts (`files_in_archive`) and not carried from one attempt to the next (every `get_archive`). A resumed
task finds them absent and installs or builds again.
"""

from __future__ import annotations

# A path with any of these as a segment is left out. `.git`, `.venv` and the other dot-directories are also hidden from the
# artifact list by `_visible`; they are named here because the snapshot has to leave them out too.
REGENERABLE_DIRS = frozenset(
    {
        "node_modules", ".venv", "venv", "__pycache__", ".git", ".cache", ".pytest_cache", ".mypy_cache",
        ".ruff_cache", "dist", "build", ".next", ".turbo", "coverage", "site-packages",
        "node-compile-cache", ".npm", ".pnpm-store", ".vite",
    }
)
REGENERABLE_FILES = frozenset({"package-lock.json", "pnpm-lock.yaml", "yarn.lock", "uv.lock", "poetry.lock"})
REGENERABLE_SUFFIXES = (".pyc",)


def is_regenerable_name(name: str, *, is_dir: bool) -> bool:
    """Whether one directory or file name is regenerable by itself (what a directory walk prunes on)."""
    if is_dir:
        return name in REGENERABLE_DIRS
    return name in REGENERABLE_DIRS or name in REGENERABLE_FILES or name.endswith(REGENERABLE_SUFFIXES)


def is_regenerable(path: str) -> bool:
    """Whether `path` (relative, `/`-separated) is, or is under, a regenerable directory, or is a regenerable file."""
    parts = [part for part in path.split("/") if part]
    return any(part in REGENERABLE_DIRS for part in parts[:-1]) or (
        bool(parts) and is_regenerable_name(parts[-1], is_dir=False)
    )


# `tar --exclude` patterns for a sandbox that archives its own workspace. GNU tar matches a pattern against any path segment,
# so each one drops the directory and everything under it. The artifact list filters again (`is_regenerable`), so a tar that
# ignores them only makes a bigger snapshot, never a noisier artifact list.
TAR_EXCLUDES = tuple(
    f"--exclude={name}" for name in (*sorted(REGENERABLE_DIRS), *sorted(REGENERABLE_FILES), *(f"*{s}" for s in REGENERABLE_SUFFIXES))
)
