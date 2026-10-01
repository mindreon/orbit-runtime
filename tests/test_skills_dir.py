"""The worker reading skills straight from the mounted library, without asking control.

Same library as control's (`<root>/<handle>/<slug>/SKILL.md ...`). The files are copied into the attempt's own directory
afterwards (see test_skills.py); here is only how they are read.

How it can go wrong, written down before the code:
  - an id that leaves the root, a symlinked file or a symlinked skill directory, reads something that is not a skill;
  - a binary, non-UTF-8 or oversized file is passed on as text; a skill with too many files or bytes is staged whole;
  - the name and description are lost when SKILL.md has no frontmatter, and the skill is dropped;
  - a skill the library lacks is an error, instead of None so that the next source (control) gets its turn;
  - a source that fails stops the chain, so one broken mount hides every skill;
  - a skill replaced while it is read crashes the attempt.
"""

import os
from pathlib import Path

import pytest
from orbit_worker.skills import (
    MAX_SKILL_BYTES,
    ChainSkillSource,
    DirSkillSource,
    SkillBundle,
    SkillSourceError,
)


def put(root: Path, rel: str, body: str | bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body if isinstance(body, bytes) else body.encode())


@pytest.mark.parametrize(
    "skill_id",
    ["", "ok", "/ok/skill", "ok//skill", "../x/y", "ok/..", "./ok/skill", "ok/skill/", "ok/skill/extra", "ok\\skill", "ok/sk\x00ill", "h" * 300 + "/s", ".git/config"],
)
async def test_an_id_that_could_leave_the_root_is_not_found(tmp_path: Path, skill_id: str) -> None:
    put(tmp_path, "ok/skill/SKILL.md", "fine")
    put(tmp_path.parent, "x/y/SKILL.md", "outside the library")
    assert await DirSkillSource(tmp_path).fetch(skill_id) is None


async def test_a_symlink_is_never_followed(tmp_path: Path) -> None:
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    put(outside, "secret.txt", "outside the library")
    put(outside, "elsewhere/SKILL.md", "another skill, outside")
    root = tmp_path / "lib"
    put(root, "h/file-link/SKILL.md", "ok")
    os.symlink(outside / "secret.txt", root / "h/file-link/leak.txt")
    os.symlink(outside / "elsewhere", root / "h/dir-link")
    source = DirSkillSource(root)
    found = await source.fetch("h/file-link")
    assert found is not None and [path for path, _ in found.files] == ["SKILL.md"]
    assert await source.fetch("h/dir-link") is None


async def test_only_usable_text_is_read_and_it_comes_back_sorted(tmp_path: Path) -> None:
    put(tmp_path, "h/s/SKILL.md", "---\nname: Writer\ndescription: writes\n---\nbody")
    put(tmp_path, "h/s/binary.bin", b"a\x00b")
    put(tmp_path, "h/s/latin1.txt", b"caf\xe9")
    put(tmp_path, "h/s/big.md", "x" * (256 * 1024 + 1))
    put(tmp_path, "h/s/refs/b.md", "b")
    put(tmp_path, "h/s/refs/a.md", "a")
    found = await DirSkillSource(tmp_path).fetch("h/s")
    assert found is not None
    assert [path for path, _ in found.files] == ["SKILL.md", "refs/a.md", "refs/b.md"]
    assert (found.id, found.name, found.description) == ("h/s", "Writer", "writes")


async def test_a_skill_without_frontmatter_is_named_after_its_directory(tmp_path: Path) -> None:
    put(tmp_path, "@scope/pirate/SKILL.md", "Just the body.")
    found = await DirSkillSource(tmp_path).fetch("@scope/pirate")
    assert found is not None and (found.name, found.description) == ("pirate", "")


async def test_a_skill_that_is_too_big_is_not_read(tmp_path: Path) -> None:
    for number in range(501):
        put(tmp_path, f"h/many/f{number}.md", "x")
    put(tmp_path, "h/many/SKILL.md", "x")
    chunk = "y" * (256 * 1024)
    for number in range(MAX_SKILL_BYTES // len(chunk) + 1):
        put(tmp_path, f"h/heavy/p{number}.md", chunk)
    source = DirSkillSource(tmp_path)
    assert await source.fetch("h/many") is None
    assert await source.fetch("h/heavy") is None


async def test_a_missing_or_empty_skill_is_none(tmp_path: Path) -> None:
    (tmp_path / "h/empty").mkdir(parents=True)
    put(tmp_path, "h/file-not-dir", "x")
    source = DirSkillSource(tmp_path)
    for skill_id in ("h/missing", "nobody/none", "h/empty", "h/file-not-dir"):
        assert await source.fetch(skill_id) is None


class Fixed:
    def __init__(self, result: SkillBundle | None | Exception) -> None:
        self.result, self.calls = result, 0

    async def fetch(self, skill_id: str) -> SkillBundle | None:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


BUNDLE = SkillBundle(id="h/s", name="N", description="d", files=(("SKILL.md", "x"),))


async def test_the_chain_asks_the_next_source_when_the_first_has_nothing() -> None:
    first, second = Fixed(None), Fixed(BUNDLE)
    assert await ChainSkillSource(first, second).fetch("h/s") == BUNDLE
    assert (first.calls, second.calls) == (1, 1)


async def test_the_first_answer_ends_the_chain() -> None:
    first, second = Fixed(BUNDLE), Fixed(None)
    assert await ChainSkillSource(first, second).fetch("h/s") == BUNDLE
    assert second.calls == 0


async def test_a_failing_source_does_not_hide_the_next(caplog: pytest.LogCaptureFixture) -> None:
    broken, good = Fixed(SkillSourceError("mount gone")), Fixed(BUNDLE)
    assert await ChainSkillSource(broken, good).fetch("h/s") == BUNDLE
    assert "mount gone" in caplog.text
    assert await ChainSkillSource(Fixed(OSError("disk")), Fixed(None)).fetch("h/s") is None


async def test_a_skill_replaced_atomically_while_it_is_read_is_found_or_none(tmp_path: Path) -> None:
    import asyncio
    import shutil

    put(tmp_path, "h/s/SKILL.md", "v1")
    source = DirSkillSource(tmp_path)

    async def churn() -> None:
        for _ in range(100):
            # Written beside, then renamed into place: the way a skill is updated (rsync does the same).
            put(tmp_path, "h/.next/SKILL.md", "v2")
            shutil.rmtree(tmp_path / "h/s", ignore_errors=True)
            (tmp_path / "h/.next").rename(tmp_path / "h/s")
            await asyncio.sleep(0)

    task = asyncio.create_task(churn())
    for _ in range(100):
        found = await source.fetch("h/s")
        assert found is None or dict(found.files)["SKILL.md"] in {"v1", "v2"}
    await task
