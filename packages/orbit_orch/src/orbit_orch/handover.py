"""The handover block a team member ends its final reply with, as plain strings.

A member is told (`orbit_worker.agent_config.MEMBER_HANDOVER_PROMPT`) to end its reply with a line `## Handover` and a few fixed
headings. The reply is carried to others in a bounded summary (what a leader's review reads, what a team reply shows, what a
`team_assign` call returns), and the block must survive the cut without the answer above it being dropped: `fit_handover` keeps
the start of the reply and the block. No framework is imported, so the workflow, the worker and the tests all use it.
"""

from __future__ import annotations

import re

HANDOVER_MARKER = "## Handover"
ELLIPSIS = "…"
# The block keeps at most this share of the room when there is a reply above it: the answer is not given up for it.
_HEAD_SHARE = 4
_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]*Handover[ \t]*:?[ \t]*$", re.IGNORECASE | re.MULTILINE)
_SECTION_START = re.compile(r"^(?=[ \t]*#{2,6}[ \t]*\S)", re.MULTILINE)


def has_handover_marker(text: str) -> bool:
    """Whether a `Handover` heading line is anywhere in `text`."""
    return _HEADING.search(text) is not None


def _split(text: str) -> tuple[str, str] | None:
    """The reply above its last handover heading and what is under it, or None when there is no heading or nothing under it."""
    found = list(_HEADING.finditer(text))
    if not found:
        return None
    body = text[found[-1].end():].strip()
    return (text[: found[-1].start()].rstrip(), body) if body else None


def _fit_sections(body: str, room: int) -> str:
    """`body` cut to `room` characters by giving every section (a heading and what is under it) the same share, the short
    ones keeping all they have: a long Evidence must not push the limits and open issues out."""
    parts = [part.rstrip() for part in _SECTION_START.split(body) if part.strip()]
    room = max(0, room - (len(parts) - 1))  # the newlines between them
    low, high = 0, room
    while low < high:  # the largest share that fits
        share = (low + high + 1) // 2
        if sum(min(len(part), share) for part in parts) <= room:
            low = share
        else:
            high = share - 1
    return "\n".join(part[:low].rstrip() for part in parts)


def fit_handover(text: str, limit: int) -> str:
    """`text` within `limit` characters. It is returned as it is when it fits. A reply with a handover block that does not
    fit keeps the block (its sections cut evenly, with a quarter of the room at most given up to it for the answer) and, in
    what room is left, the start of the reply, then `…` where it was cut. A reply without a block keeps its first `limit`."""
    if len(text) <= limit:
        return text
    split = _split(text)
    if split is None:
        return text[:limit]
    head, body = split
    block = f"{HANDOVER_MARKER}\n{body}"
    block_room = limit - (min(len(head), limit // _HEAD_SHARE) + len(ELLIPSIS) + 2 if head else 0)
    if len(block) > block_room:
        block = f"{HANDOVER_MARKER}\n{_fit_sections(body, block_room - len(HANDOVER_MARKER) - 1)}"
    room = limit - len(block) - len(ELLIPSIS) - 2
    if not head or room <= 0:
        return block
    if len(head) <= room:
        return f"{head}\n\n{block}"
    return f"{head[:room].rstrip()}{ELLIPSIS}\n\n{block}"[:limit]


def without_handover_block(text: str) -> str:
    """What a person who reads a reply directly is shown: the reply above its handover block. A reply that is nothing but the
    block is shown as the block's content, without the English heading."""
    split = _split(text)
    if split is None:
        return text
    head, body = split
    return head or body
