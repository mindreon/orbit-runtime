"""Cloud Agent workspace Activities.

These record the clone / branch / pull-request steps on the local work
root. They do not talk to a git host. A later executor can replace the
file markers without changing the workflow names.
"""

import os
from pathlib import Path

from orbit_contracts.models import (
    CloneRepoInput,
    CloneRepoOutput,
    OpenPrInput,
    OpenPrOutput,
    PushBranchInput,
    PushBranchOutput,
)


def _root(session_id: str) -> Path:
    base = Path(os.environ.get("ORBIT_WORK_ROOT", "/tmp/orbit-workspaces"))
    path = base / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


async def clone_repo(inp: CloneRepoInput) -> CloneRepoOutput:
    workdir = _root(inp.session_id)
    (workdir / "REPO").write_text(f"{inp.repo_url}@{inp.revision}\n", encoding="utf-8")
    return CloneRepoOutput(workdir=str(workdir))


async def push_branch(inp: PushBranchInput) -> PushBranchOutput:
    path = Path(inp.workdir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "BRANCH").write_text(inp.branch + "\n", encoding="utf-8")
    return PushBranchOutput(branch=inp.branch)


async def open_pr(inp: OpenPrInput) -> OpenPrOutput:
    slug = inp.repo_url.rstrip("/").removesuffix(".git")
    return OpenPrOutput(pr_url=f"{slug}/pull/1")
