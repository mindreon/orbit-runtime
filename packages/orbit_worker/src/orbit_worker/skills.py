"""Skills a task was given, staged for the agent (15 T8.4).

A skill is third-party text. The worker fetches it from control, writes it under a directory that exists for one attempt,
and registers it with AgentScope, which tells the agent its name, description and place and lets it read the files. The
files are data: nothing here executes them. A skill that cannot be used (refused, unreachable, unsafe, too big) is
skipped with a log line and never stops the attempt.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
import shutil
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


class ControlSkillSource:
    """Reads a skill's files from control's internal listener (the same one the worker posts live events to)."""

    def __init__(self, base_url: str, token: str) -> None:
        self._base = base_url.rstrip("/")
        self._token = token

    async def fetch(self, skill_id: str) -> SkillBundle | None:
        timeout = aiohttp.ClientTimeout(total=_FETCH_TIMEOUT_S)
        headers = {"Authorization": f"Bearer {self._token}"}
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.get(self._url(skill_id), headers=headers) as response,
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
