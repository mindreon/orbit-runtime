"""Skills a task was given, staged for the agent (15 T8.4).

A skill is third-party text. The worker fetches it from control, writes it under a directory that exists for one attempt,
and registers it with AgentScope, which tells the agent its name, description and place and lets it read the files. The
files are data: nothing here executes them.

How it can go wrong, written down before the code:
  - no skills are configured and the worker still calls control;
  - a skill control refuses (not usable, unknown) or cannot deliver (control down) stops the attempt, or hides the others;
  - a file name that climbs out of the skill's directory (`..`, an absolute path, a backslash, NUL) writes outside it, or
    a skill with one such file is half written;
  - a skill so large it would fill the worker's disk is staged;
  - the SKILL.md has no name or description and the skill is dropped, instead of falling back to the catalog's;
  - two skills with one name replace each other silently (AgentScope keeps the last);
  - the staged files outlive the attempt, also when the attempt raises;
  - a staged file is executable, or a skill's directory name carries characters that mean something to a path;
  - the control token shows up in a log line or an error;
  - the agent is not told about a staged skill.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

import pytest
from aiohttp import web
from orbit_worker.skills import (
    MAX_SKILL_BYTES,
    ControlSkillSource,
    SkillBundle,
    SkillSourceError,
    safe_relative_path,
    staged_skills,
)


class FakeSource:
    def __init__(self, bundles: dict[str, SkillBundle | Exception | None]) -> None:
        self.bundles = bundles
        self.calls: list[str] = []

    async def fetch(self, skill_id: str) -> SkillBundle | None:
        self.calls.append(skill_id)
        found = self.bundles.get(skill_id)
        if isinstance(found, Exception):
            raise found
        return found


def bundle(skill_id: str = "h/s", *, name: str = "Writer", description: str = "writes", files=None) -> SkillBundle:
    return SkillBundle(
        id=skill_id,
        name=name,
        description=description,
        files=tuple(files if files is not None else [("SKILL.md", "---\nname: Writer\ndescription: writes\n---\nUse me.")]),
    )


@pytest.mark.parametrize(
    "path", ["", "/etc/passwd", "../x", "a/../../x", "a\\b.md", "a\x00b", "./a", "a//b", "a/./b", "..", "a/.."]
)
def test_a_file_name_that_could_leave_the_directory_is_refused(path: str) -> None:
    assert safe_relative_path(path) is None


@pytest.mark.parametrize("path", ["SKILL.md", "references/api.md", "a/b/c.json", "x.tar.gz"])
def test_an_ordinary_relative_name_is_kept(path: str) -> None:
    assert safe_relative_path(path) == path


async def test_without_skills_control_is_not_asked(tmp_path: Path) -> None:
    source = FakeSource({})
    async with staged_skills(source, (), root=tmp_path) as skills:
        assert skills == ()
    assert source.calls == []
    assert list(tmp_path.iterdir()) == []


async def test_a_refused_or_failing_skill_does_not_hide_the_others(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = FakeSource({"a/gone": None, "b/down": SkillSourceError("control answered 502"), "c/ok": bundle("c/ok")})
    with caplog.at_level(logging.WARNING):
        async with staged_skills(source, ("a/gone", "b/down", "c/ok"), root=tmp_path) as skills:
            assert [skill.name for skill in skills] == ["Writer"]
    assert "a/gone" in caplog.text and "b/down" in caplog.text


async def test_a_skill_with_one_escaping_file_is_refused_whole(tmp_path: Path) -> None:
    evil = bundle("e/vil", files=[("SKILL.md", "ok"), ("../outside.md", "pwned")])
    source = FakeSource({"e/vil": evil})
    root = tmp_path / "stage"
    root.mkdir()
    async with staged_skills(source, ("e/vil",), root=root) as skills:
        assert skills == ()
        assert [p for p in root.rglob("*") if p.is_file()] == []
    assert not (tmp_path / "outside.md").exists()


async def test_a_skill_over_the_size_limit_is_skipped(tmp_path: Path) -> None:
    big = bundle("b/ig", files=[("SKILL.md", "x"), ("big.txt", "y" * (MAX_SKILL_BYTES + 1))])
    async with staged_skills(FakeSource({"b/ig": big}), ("b/ig",), root=tmp_path) as skills:
        assert skills == ()


async def test_a_skill_without_a_skill_md_is_skipped(tmp_path: Path) -> None:
    only_notes = bundle("n/o", files=[("README.md", "hi")])
    async with staged_skills(FakeSource({"n/o": only_notes}), ("n/o",), root=tmp_path) as skills:
        assert skills == ()


async def test_a_missing_name_and_description_fall_back_to_the_catalogs(tmp_path: Path) -> None:
    bare = bundle("h/bare", name="Catalog name", description="Catalog text", files=[("SKILL.md", "Just the body.")])
    async with staged_skills(FakeSource({"h/bare": bare}), ("h/bare",), root=tmp_path) as skills:
        assert (skills[0].name, skills[0].description) == ("Catalog name", "Catalog text")
        assert skills[0].markdown == "Just the body."


async def test_two_skills_with_one_name_both_stay(tmp_path: Path) -> None:
    source = FakeSource({"a/one": bundle("a/one", name="Same"), "b/two": bundle("b/two", name="Same")})
    async with staged_skills(source, ("a/one", "b/two"), root=tmp_path) as skills:
        assert len({skill.name for skill in skills}) == 2
        assert len({skill.dir for skill in skills}) == 2


async def test_files_are_written_as_plain_data_inside_their_own_directory(tmp_path: Path) -> None:
    files = [("SKILL.md", "---\nname: W\ndescription: d\n---\nbody"), ("scripts/run.sh", "#!/bin/sh\necho hi\n"), ("références.md", "é")]
    async with staged_skills(FakeSource({"@sc ope/we ird": bundle("@sc ope/we ird", files=files)}), ("@sc ope/we ird",), root=tmp_path) as skills:
        [skill] = skills
        directory = Path(skill.dir)
        assert directory.parent == tmp_path or tmp_path in directory.parents
        assert (directory / "scripts" / "run.sh").read_text() == "#!/bin/sh\necho hi\n"
        assert (directory / "références.md").read_text(encoding="utf-8") == "é"
        assert not os.access(directory / "scripts" / "run.sh", os.X_OK)
        assert stat.S_IMODE((directory / "SKILL.md").stat().st_mode) == 0o644
        assert " " not in directory.name and "@" not in directory.name and "/" not in directory.name


async def test_the_staged_files_are_gone_afterwards_also_when_the_attempt_raises(tmp_path: Path) -> None:
    source = FakeSource({"h/s": bundle()})
    with pytest.raises(RuntimeError, match="attempt failed"):
        async with staged_skills(source, ("h/s",), root=tmp_path) as skills:
            assert Path(skills[0].dir).exists()
            raise RuntimeError("attempt failed")
    assert [p for p in tmp_path.rglob("*")] == []


# ---- the HTTP client against a control that answers like the real one ---------------------------------------------

TOKEN = "t0ken-must-not-leak"


async def _control(handler) -> tuple[web.AppRunner, str]:
    app = web.Application()
    app.router.add_get("/internal/skills/{handle}/{slug}", handler)
    app.router.add_get("/internal/skills/{slug}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return runner, f"http://127.0.0.1:{port}"


async def test_the_client_reads_a_bundle_and_sends_the_token() -> None:
    seen: dict[str, str] = {}

    async def handler(request: web.Request) -> web.Response:
        seen["auth"] = request.headers.get("Authorization", "")
        seen["path"] = request.path
        return web.json_response(
            {"id": "@sc/w", "name": "Writer", "description": "d", "files": [{"path": "SKILL.md", "body": "hi"}]}
        )

    runner, base = await _control(handler)
    try:
        found = await ControlSkillSource(base, TOKEN).fetch("@sc/w")
    finally:
        await runner.cleanup()
    assert found == SkillBundle(id="@sc/w", name="Writer", description="d", files=(("SKILL.md", "hi"),))
    assert seen == {"auth": f"Bearer {TOKEN}", "path": "/internal/skills/@sc/w"}


async def test_a_skill_without_an_author_uses_the_short_path() -> None:
    seen: dict[str, str] = {}

    async def handler(request: web.Request) -> web.Response:
        seen["path"] = request.path
        return web.json_response({"id": "solo", "name": "Solo", "description": "d", "files": []})

    runner, base = await _control(handler)
    try:
        await ControlSkillSource(base, TOKEN).fetch("solo")
    finally:
        await runner.cleanup()
    assert seen["path"] == "/internal/skills/solo"


async def test_not_found_is_none_and_other_failures_raise_without_the_token() -> None:
    async def not_found(_: web.Request) -> web.Response:
        return web.json_response({"error": "NOT_FOUND"}, status=404)

    async def broken(_: web.Request) -> web.Response:
        return web.Response(status=502, text=f"gateway echoed {TOKEN}")

    runner, base = await _control(not_found)
    try:
        assert await ControlSkillSource(base, TOKEN).fetch("a/b") is None
    finally:
        await runner.cleanup()
    runner, base = await _control(broken)
    try:
        with pytest.raises(SkillSourceError) as failure:
            await ControlSkillSource(base, TOKEN).fetch("a/b")
    finally:
        await runner.cleanup()
    assert "502" in str(failure.value) and TOKEN not in str(failure.value)


async def test_an_unreachable_control_raises_a_source_error() -> None:
    with pytest.raises(SkillSourceError):
        await ControlSkillSource("http://127.0.0.1:9", TOKEN).fetch("a/b")
