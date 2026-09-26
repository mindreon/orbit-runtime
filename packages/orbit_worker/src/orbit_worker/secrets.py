"""Refuse credential values inside AgentState (ADR-010); redact them in events."""

import math
import re
import string
from collections import Counter

_SECRET_KEYS = {"password", "secret", "api_key", "token", "authorization", "access_token"}

REDACTED = "[REDACTED]"

# A key is secret-named when its name, lower-cased with "-" and "_" removed,
# contains one of these: apiKey, x-api-key, client_secret, refresh_token, ...
_SECRET_KEY_PARTS = (
    "apikey",
    "secret",
    "privatekey",
    "token",
    "cookie",
    "authorization",
    "password",
)
# The same rule inside free text: "-" or "_" may sit between any two letters.
_KEY_PART = "|".join("[-_]*".join(part) for part in _SECRET_KEY_PARTS)
_KEY_NAME = rf"(?<![A-Za-z0-9_\-])[A-Za-z0-9_\-]*?(?:{_KEY_PART})[A-Za-z0-9_\-]*"
# Only the ``secret`` group is replaced, so "Bearer " or "token: " stays readable.
# A key's value is printable ASCII except quotes , ; }, so CJK prose after
# "token: abc" is not swallowed into the secret.
_TEXT_PATTERNS = (
    re.compile(
        rf"(?i){_KEY_NAME}[\"']?\s*[:=]\s*[\"']?(?P<secret>[!#-&(-+\--:<-|~]+)"
    ),
    # A 40-character lower-case hex token (a GitHub classic token, for one)
    # right after a secret-named key, with or without ":" or "=".
    re.compile(
        rf"(?i){_KEY_NAME}[^A-Za-z0-9\n]{{1,6}}(?P<secret>(?-i:[0-9a-f]{{40}}))(?![A-Za-z0-9])"
    ),
    re.compile(r"(?i)(?<![A-Za-z0-9_])bearer\s+(?P<secret>[A-Za-z0-9._~+/=\-]+)"),
    re.compile(
        r"(?<![A-Za-z0-9_])(?P<secret>"
        r"sk-[A-Za-z0-9_\-]+"
        r"|ghp_[A-Za-z0-9]{8,}"
        r"|github_pat_[A-Za-z0-9_]{8,}"
        r"|glpat-[A-Za-z0-9_\-]{8,}"
        r"|AKIA[0-9A-Z]{16,}"
        r"|AIza[0-9A-Za-z_\-]{20,}"
        r"|xox[abprs]-[A-Za-z0-9\-]{8,}"
        r")"
    ),
)
_LONG_TOKEN = re.compile(r"(?<![A-Za-z0-9_\-+/=])(?P<secret>[A-Za-z0-9_\-+/=]{32,})")
_MIN_ENTROPY_BITS = 3.5
# Enough context to see "access_token = " or "Authorization: Bearer " in the
# text before a chunk boundary.
_OVERLAP = 64
# Characters a token can be made of. A chunk may be cut before any other
# character (space, CJK, punctuation) without splitting a token.
_TOKEN_CHARS = frozenset(string.ascii_letters + string.digits + "._~+/=-")
# A token run longer than this is released anyway and judged as a long token.
_MAX_HOLD = 512


def reject_secret_values(value: object, path: str = "") -> None:
    """Raise when a serialized blob carries a credential.

    Keys that name a secret, and strings that look like a provider key,
    are rejected before the blob is written. Credential references (an id
    with no secret material) are ordinary strings and stay.
    """

    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key)
            if name.lower() in _SECRET_KEYS and isinstance(item, str) and item:
                raise ValueError(f"secret value at {path}{name}")
            reject_secret_values(item, f"{path}{name}.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            reject_secret_values(item, f"{path}{index}.")
        return
    if isinstance(value, str) and value.startswith("sk-"):
        raise ValueError(f"secret token at {path[:-1] or 'value'}")


def is_secret_key(name: str) -> bool:
    normalized = name.lower().replace("-", "").replace("_", "")
    return any(part in normalized for part in _SECRET_KEY_PARTS)


def redact_text(text: str) -> str:
    """Replace suspected secrets in free text with ``[REDACTED]``. Never raises."""

    return _replace(text, _secret_spans(text), 0, len(text))


def redact_value(value: object) -> object:
    """Redact a JSON-like value: secret-named keys lose their value, strings are scanned."""

    if isinstance(value, dict):
        return {
            key: REDACTED
            if is_secret_key(str(key)) and value[key] not in (None, "")
            else redact_value(value[key])
            for key in value
        }
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


class StreamRedactor:
    """Redacts one text stream that arrives in chunks.

    A chunk is released only up to its last non-token character, so an
    unfinished token waits for the rest of itself while CJK text, which has
    no spaces, still streams. The last ``_OVERLAP`` characters
    already released stay as context for the next scan, so a secret whose
    prefix ("Bearer ", "token=") was in the previous chunk is still caught.
    """

    def __init__(self) -> None:
        self._context = ""
        self._pending = ""

    @property
    def pending(self) -> int:
        return len(self._pending)

    def feed(self, chunk: str) -> None:
        self._pending += chunk

    def take(self, final: bool = False) -> str:
        window = self._context + self._pending
        offset = len(self._context)
        cut = len(window)
        if not final:
            while cut > offset and window[cut - 1] in _TOKEN_CHARS:
                cut -= 1
        spans = _secret_spans(window)
        for start, end in spans:
            if start < cut < end:
                cut = start
        if len(window) - cut > _MAX_HOLD:
            cut = len(window)
        if cut <= offset:
            return ""
        released = _replace(window, spans, offset, cut)
        self._context = window[:cut][-_OVERLAP:]
        self._pending = window[cut:]
        return released


def _secret_spans(text: str) -> list[tuple[int, int]]:
    spans = [
        match.span("secret") for pattern in _TEXT_PATTERNS for match in pattern.finditer(text)
    ]
    spans.extend(
        match.span("secret")
        for match in _LONG_TOKEN.finditer(text)
        if _looks_random(match.group("secret"))
    )
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _replace(text: str, spans: list[tuple[int, int]], lo: int, hi: int) -> str:
    parts: list[str] = []
    position = lo
    for start, end in spans:
        start, end = max(start, lo), min(end, hi)
        if start >= end:
            continue
        parts.append(text[position:start])
        parts.append(REDACTED)
        position = end
    parts.append(text[position:hi])
    return "".join(parts)


def _looks_random(token: str) -> bool:
    # Hex ids (session ids, digests) lack upper case and are not flagged.
    if not (
        any(c.isupper() for c in token)
        and any(c.islower() for c in token)
        and any(c.isdigit() for c in token)
    ):
        return False
    counts = Counter(token)
    entropy = -sum(n / len(token) * math.log2(n / len(token)) for n in counts.values())
    return entropy >= _MIN_ENTROPY_BITS
