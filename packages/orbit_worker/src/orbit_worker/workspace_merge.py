"""Merging one attempt's workspace into the head snapshot of its task.

Attempts of one task may run side by side, each in a workspace of its own that starts as a copy of the head (the task's latest
snapshot). When the second of them saves, the head already holds what the first one added, so saving the second's whole
workspace would drop the first's files. Instead only what the attempt itself changed (against the snapshot it started from) is
applied onto the head as it is at that moment. The caller does this while it holds the task's commit lock.

A file both attempts changed goes to the one that saves last. Directories, regular files and symlinks are carried; the
modification time is not part of what counts as a change.
"""

from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass


@dataclass(frozen=True)
class _Entry:
    info: tarfile.TarInfo
    data: bytes

    def same_as(self, other: _Entry) -> bool:
        a, b = self.info, other.info
        if a.type != b.type:
            return False
        if a.isreg():
            return self.data == other.data and (a.mode & 0o111) == (b.mode & 0o111)
        if a.issym() or a.islnk():
            return a.linkname == b.linkname
        return True


def _read(archive: bytes | None) -> dict[str, _Entry]:
    entries: dict[str, _Entry] = {}
    if not archive:
        return entries
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as tar:
        for member in tar.getmembers():
            name = member.name.removeprefix("./").rstrip("/")
            if not name or name == ".":
                continue
            handle = tar.extractfile(member) if member.isreg() else None
            entries[name] = _Entry(member, handle.read() if handle is not None else b"")
    return entries


def merge_archives(base: bytes | None, mine: bytes, head: bytes) -> bytes | None:
    """`head` with this attempt's changes applied: what `mine` added or changed against `base` (what the attempt started from;
    None for an empty workspace) is put in, and what it removed is taken out. None when that changes nothing in `head`."""
    before, now, latest = _read(base), _read(mine), _read(head)
    merged = dict(latest)
    for name, entry in now.items():
        if name not in before or not entry.same_as(before[name]):
            merged[name] = entry
    for name in before:
        if name not in now:
            merged.pop(name, None)
    if merged.keys() == latest.keys() and all(merged[name] is latest[name] or merged[name].same_as(latest[name]) for name in merged):
        return None
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as tar:
        for name in sorted(merged):
            entry = merged[name]
            info = entry.info
            info.name = name
            tar.addfile(info, io.BytesIO(entry.data) if info.isreg() else None)
    return output.getvalue()
