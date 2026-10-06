"""Skills a task was given, staged for the agent (15 T8.4).

A skill is third-party text. The worker fetches it from control, writes it under a directory that exists for one attempt,
and registers it with AgentScope, which tells the agent its name, description and place and lets it read the files. The
files are data: nothing here executes them. A skill that cannot be used (refused, unreachable, unsafe, too big) is
skipped with a log line and never stops the attempt.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import re
import shutil
import stat
import tempfile
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import aiohttp
import frontmatter
from agentscope.skill import Skill

logger = logging.getLogger(__name__)

MAX_SKILL_BYTES = 8 * 1024 * 1024
MAX_FILE_BYTES = 256 * 1024
MAX_SKILL_FILES = 500
MAX_ID_PART = 200
SKILL_FILE = "SKILL.md"
_FETCH_TIMEOUT_S = 15


class SkillSourceError(RuntimeError):
    """A skill could not be fetched. The message never carries the token or the response body."""


@dataclass(frozen=True)
class SkillBundle:
    id: str
    name: str
    description: str
    files: tuple[tuple[str, str], ...]


class SkillSource(Protocol):
    async def fetch(self, skill_id: str) -> SkillBundle | None: ...


def safe_relative_path(path: str) -> str | None:
    """The path if it stays inside the directory it is joined to, else None."""
    if not path or path.startswith("/") or "\\" in path or "\x00" in path:
        return None
    segments = path.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        return None
    return path


class DirSkillSource:
    """Reads skills straight from the mounted library, the same tree control reads: <root>/<handle>/<slug>/SKILL.md and the
    files beside it. The skill id is the mapping. Nothing outside <root> is read, a symlink is never followed, and a skill
    that is too big, or has no readable text, is None so that the next source gets its turn."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()

    async def fetch(self, skill_id: str) -> SkillBundle | None:
        return await asyncio.to_thread(self._read, skill_id)

    def _directory(self, skill_id: str) -> Path | None:
        parts = skill_id.split("/")
        if len(parts) != 2 or any(not _id_part_ok(part) for part in parts):
            return None
        directory = self._root / parts[0] / parts[1]
        return directory if _plain_dir(directory.parent) and _plain_dir(directory) else None

    def _read(self, skill_id: str) -> SkillBundle | None:
        directory = self._directory(skill_id)
        if directory is None:
            return None
        try:
            files = _read_tree(directory)
        except FileNotFoundError:
            return None  # replaced while it was read
        if not files:
            return None
        parsed = frontmatter.loads(dict(files).get(SKILL_FILE, ""))
        slug = skill_id.split("/")[1]
        return SkillBundle(
            id=skill_id,
            name=str(parsed.get("name") or slug),
            description=str(parsed.get("description") or ""),
            files=tuple(files),
        )


def _id_part_ok(part: str) -> bool:
    return bool(part) and len(part) <= MAX_ID_PART and not part.startswith(".") and "\\" not in part and "\x00" not in part


def _plain_dir(path: Path) -> bool:
    try:
        return stat.S_ISDIR(os.lstat(path).st_mode)
    except OSError:
        return False


def _read_tree(directory: Path) -> list[tuple[str, str]] | None:
    """The regular UTF-8 files of a skill, sorted. None when it has too many files or too many bytes."""
    files: list[tuple[str, str]] = []
    seen = total = 0
    for current, _dirs, names in os.walk(directory, followlinks=False):
        for name in names:
            path = Path(current) / name
            listed = os.lstat(path)
            if not stat.S_ISREG(listed.st_mode):
                continue
            seen += 1
            if seen > MAX_SKILL_FILES:
                return None
            if listed.st_size > MAX_FILE_BYTES:
                continue
            total += listed.st_size
            if total > MAX_SKILL_BYTES:
                return None
            text = _read_text(path, listed)
            if text is not None:
                files.append((path.relative_to(directory).as_posix(), text))
    return sorted(files)


def _read_text(path: Path, listed: os.stat_result) -> str | None:
    """The file's text if it is still the file that was listed (not swapped for a link) and is UTF-8 without NUL."""
    with open(path, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_ino, opened.st_dev) != (listed.st_ino, listed.st_dev):
            return None
        raw = handle.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES or b"\x00" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


BUNDLE_PREFIX = "expert:"


def bundle_skill_ids(names: tuple[str, ...]) -> tuple[str, ...]:
    """The ids under which `staged_skills` stages an expert's bundle skills, beside the catalog skill ids."""
    return tuple(f"{BUNDLE_PREFIX}{name}" for name in names)


class ExpertScopedSource:
    """Routes the skill ids of `bundle_skill_ids` to the bundle of one expert version (ADR-0013) and every other id to the
    catalog source. A bundle skill is staged exactly like a catalog skill once it is fetched."""

    def __init__(self, source: SkillSource, tenant_id: str, expert_ref: str) -> None:
        self._source = source
        self._tenant_id = tenant_id
        self._expert_ref = expert_ref

    async def fetch(self, skill_id: str) -> SkillBundle | None:
        if not skill_id.startswith(BUNDLE_PREFIX):
            return await self._source.fetch(skill_id)
        fetch_expert = getattr(self._source, "fetch_expert", None)
        if fetch_expert is None:
            return None
        return await fetch_expert(self._tenant_id, self._expert_ref, skill_id[len(BUNDLE_PREFIX) :])


class ChainSkillSource:
    """The first source that has the skill wins. One that fails is logged and the next is tried, so a broken mount does
    not hide a skill control could still serve."""

    def __init__(self, *sources: SkillSource) -> None:
        self._sources = sources

    async def fetch(self, skill_id: str) -> SkillBundle | None:
        for source in self._sources:
            try:
                found = await source.fetch(skill_id)
            except Exception as exc:  # noqa: BLE001 - a source must never stop the attempt
                logger.warning("skill %s: a source failed (%s): %s", skill_id, type(exc).__name__, exc)
                continue
            if found is not None:
                return found
        return None


    async def fetch_expert(self, tenant_id: str, expert_ref: str, name: str) -> SkillBundle | None:
        """Only control holds the bundles of experts: the sources that can serve one have `fetch_expert`."""
        for source in self._sources:
            fetch = getattr(source, "fetch_expert", None)
            if fetch is None:
                continue
            try:
                found = await fetch(tenant_id, expert_ref, name)
            except Exception as exc:  # noqa: BLE001 - a source must never stop the attempt
                logger.warning("bundle skill %s: a source failed (%s): %s", name, type(exc).__name__, exc)
                continue
            if found is not None:
                return found
        return None


class ControlSkillSource:
    """Reads a skill's files from control's internal listener (the same one the worker posts live events to)."""

    def __init__(self, base_url: str, token: str) -> None:
        self._base = base_url.rstrip("/")
        self._token = token

    async def fetch(self, skill_id: str) -> SkillBundle | None:
        return await self._get(self._url(skill_id))

    async def fetch_expert(self, tenant_id: str, expert_ref: str, name: str) -> SkillBundle | None:
        """A skill of an expert version's own bundle (ADR-0013): `/internal/experts/<id>/<version>/skills/<name>`, for the
        tenant the attempt runs in."""
        expert_id, _, version = expert_ref.rpartition("@")
        if not expert_id or not version.isdigit() or not safe_relative_path(name) or "/" in name:
            return None
        parts = "/".join(quote(part, safe="") for part in (expert_id, version, "skills", name))
        return await self._get(f"{self._base}/internal/experts/{parts}?tenant_id={quote(tenant_id, safe='')}")

    async def _get(self, url: str) -> SkillBundle | None:
        timeout = aiohttp.ClientTimeout(total=_FETCH_TIMEOUT_S)
        headers = {"Authorization": f"Bearer {self._token}"}
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.get(url, headers=headers) as response,
            ):
                if response.status == 404:
                    return None
                if response.status != 200:
                    raise SkillSourceError(f"control answered {response.status}")
                return _bundle(await response.json())
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise SkillSourceError(f"control unreachable ({type(exc).__name__})") from None
        except ValueError:
            raise SkillSourceError("control sent a skill that is not valid JSON") from None

    def _url(self, skill_id: str) -> str:
        handle, slash, slug = skill_id.partition("/")
        parts = [handle, slug] if slash else [handle]
        return f"{self._base}/internal/skills/" + "/".join(quote(part, safe="@") for part in parts)


def _bundle(raw: Any) -> SkillBundle:
    try:
        files = tuple((str(item["path"]), str(item["body"])) for item in raw["files"])
        return SkillBundle(id=str(raw["id"]), name=str(raw["name"]), description=str(raw["description"]), files=files)
    except (KeyError, TypeError):
        raise SkillSourceError("control sent a skill in an unknown shape") from None


@contextlib.asynccontextmanager
async def staged_skills(
    source: SkillSource, skill_ids: tuple[str, ...], *, root: Path | None = None
) -> AsyncIterator[tuple[Skill, ...]]:
    """The skills that could be staged for one attempt. Their files are removed when the block ends, however it ends."""
    if not skill_ids:
        yield ()
        return
    stage = Path(tempfile.mkdtemp(prefix="orbit-skills-", dir=root))
    try:
        yield tuple(await _stage_all(source, skill_ids, stage))
    finally:
        shutil.rmtree(stage, ignore_errors=True)


async def _stage_all(source: SkillSource, skill_ids: tuple[str, ...], stage: Path) -> list[Skill]:
    staged: list[Skill] = []
    names: set[str] = set()
    for skill_id in dict.fromkeys(skill_ids):
        bundle = await _fetch(source, skill_id)
        if bundle is None:
            continue
        reason = _unusable(bundle)
        if reason:
            logger.warning("skill %s skipped: %s", skill_id, reason)
            continue
        staged.append(_write(bundle, stage, names))
    return staged


async def _fetch(source: SkillSource, skill_id: str) -> SkillBundle | None:
    try:
        bundle = await source.fetch(skill_id)
    except SkillSourceError as exc:
        logger.warning("skill %s skipped: %s", skill_id, exc)
        return None
    except Exception as exc:  # noqa: BLE001 - a skill must never stop the attempt
        logger.warning("skill %s skipped: %s", skill_id, type(exc).__name__)
        return None
    if bundle is None:
        logger.warning("skill %s skipped: control does not offer it", skill_id)
    return bundle


def _unusable(bundle: SkillBundle) -> str:
    if any(safe_relative_path(path) is None for path, _ in bundle.files):
        return "a file name leaves the skill's directory"
    if sum(len(body.encode("utf-8")) for _, body in bundle.files) > MAX_SKILL_BYTES:
        return "it is over the size limit"
    if SKILL_FILE not in {path for path, _ in bundle.files}:
        return f"it has no {SKILL_FILE}"
    return ""


def _write(bundle: SkillBundle, stage: Path, names: set[str]) -> Skill:
    directory = stage / _directory_name(bundle.id)
    directory.mkdir(mode=0o755)
    root = directory.resolve()
    for path, body in bundle.files:
        target = (directory / path).resolve()
        if root not in target.parents:
            raise ValueError("unsafe path passed the check")  # the check above makes this unreachable
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        target.chmod(0o644)
    skill_md = dict(bundle.files)[SKILL_FILE]
    parsed = frontmatter.loads(skill_md)
    name = _unique(str(parsed.get("name") or bundle.name), names)
    description = str(parsed.get("description") or bundle.description)
    return Skill(name=name, description=description, dir=str(directory), markdown=parsed.content, updated_at=time.time())


def _directory_name(skill_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", skill_id).strip("-.") or "skill"
    return f"{slug[:60]}-{hashlib.sha256(skill_id.encode()).hexdigest()[:8]}"


def _unique(name: str, taken: set[str]) -> str:
    candidate, number = name, 1
    while candidate in taken:
        number += 1
        candidate = f"{name}-{number}"
    taken.add(candidate)
    return candidate


_source: SkillSource | None = None


def set_skill_source(source: SkillSource | None) -> None:
    global _source
    _source = source


def get_skill_source() -> SkillSource | None:
    return _source
