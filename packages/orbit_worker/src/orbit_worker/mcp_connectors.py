"""Attach saved MCP connectors to an AgentScope toolkit.

The shape follows Octop's custom MCP path (stdio or streamable HTTP, a
per-server lock, a fingerprint cache) and Orbit's rule that secret values
are never stored. Header and environment values are read from the worker
process environment at connect time.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import re
from typing import Any
from urllib.parse import parse_qsl, urlparse

from agentscope.mcp import HttpMCPConfig, MCPClient, StdioMCPConfig
from orbit_contracts.models import McpConnectorSpec, McpHeaderRef

from orbit_worker.secrets import redact_text, reject_secret_values
from orbit_worker.settings import McpSettings, process_environment

logger = logging.getLogger(__name__)

_ACCEPT = "application/json, text/event-stream"
_CONNECT_TIMEOUT_S = 25.0
_NAME = re.compile(r"[A-Za-z0-9_-]+")
_HEADER_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LOCAL_HOSTS = frozenset({"localhost", "host.docker.internal", "gateway.docker.internal"})
_LOCAL_SUFFIXES = (".local", ".localhost", ".internal")
_SECRET_QUERY = frozenset(
    {"token", "key", "access_token", "api_key", "secret", "password", "apikey"}
)


class McpRegistry:
    """One live client per connector id. A changed fingerprint reconnects."""

    def __init__(self) -> None:
        self._clients: dict[str, MCPClient] = {}
        self._fingerprints: dict[str, str] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()

    async def clients_for(self, specs: list[McpConnectorSpec]) -> list[MCPClient]:
        ready: list[MCPClient] = []
        for spec in specs:
            client = await self._one(spec)
            if client is not None:
                ready.append(client)
        return ready

    async def close_all(self) -> None:
        async with self._guard:
            clients = list(self._clients.values())
            self._clients.clear()
            self._fingerprints.clear()
        for client in clients:
            await client.close()

    async def _lock(self, name: str) -> asyncio.Lock:
        async with self._guard:
            lock = self._locks.get(name)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[name] = lock
            return lock

    async def _one(self, spec: McpConnectorSpec) -> MCPClient | None:
        try:
            normalized = normalize_spec(spec)
        except ValueError as exc:
            logger.warning("mcp connector %s skipped: %s", spec.id or spec.name, exc)
            return None
        name = client_name(normalized)
        env, headers = _resolved(normalized, log_missing=False)
        fingerprint = _fingerprint(normalized, env, headers)
        lock = await self._lock(name)
        async with lock:
            cached = self._clients.get(name)
            same = cached is not None and self._fingerprints.get(name) == fingerprint
            if same and cached is not None and cached.is_connected:
                return cached
            if cached is not None:
                await cached.close()
                self._clients.pop(name, None)
            env, headers = _resolved(normalized, log_missing=True)
            client = _build_client(name, normalized, env, headers)
            try:
                await asyncio.wait_for(client.connect(), _CONNECT_TIMEOUT_S)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "mcp connector %s connect failed (%s): %s",
                    name,
                    type(exc).__name__,
                    redact_text(str(exc))[:200],
                )
                if client.is_connected:
                    await client.close()
                return None
            _serialize_session(client)
            self._clients[name] = client
            self._fingerprints[name] = fingerprint
            logger.info(
                "mcp connector ready id=%s transport=%s target=%s fingerprint=%s",
                normalized.id,
                normalized.transport,
                _log_target(normalized),
                fingerprint,
            )
            return client


def specs_for_storage(specs: list[McpConnectorSpec]) -> list[dict[str, Any]]:
    """Drop invalid connectors. The stored dicts have names, never values."""

    stored: list[dict[str, Any]] = []
    for spec in specs:
        try:
            normalized = normalize_spec(spec)
            dumped = normalized.model_dump(mode="json")
            reject_secret_values(dumped)
        except ValueError as exc:
            logger.warning("mcp connector %s not stored: %s", spec.id or spec.name, exc)
            continue
        stored.append(dumped)
    return stored


def parse_stored(raw: list[dict[str, Any]] | None) -> list[McpConnectorSpec]:
    if not raw:
        return []
    specs: list[McpConnectorSpec] = []
    for item in raw:
        try:
            specs.append(normalize_spec(McpConnectorSpec.model_validate(item)))
        except ValueError as exc:
            logger.warning("stored mcp connector skipped: %s", exc)
    return specs


async def attach_mcp_clients(toolkit: Any, raw: list[dict[str, Any]] | None, registry: McpRegistry) -> None:
    """Connect the room's connectors and register them on the basic tool group.

    A connector that cannot be reached is skipped. The turn still runs.
    """

    groups = getattr(toolkit, "tool_groups", None)
    if not groups:
        return
    clients = await registry.clients_for(parse_stored(raw))
    if not clients:
        return
    group = groups[0]
    present = {client.name for client in group.mcps}
    for client in clients:
        if client.name not in present:
            group.mcps.append(client)
            present.add(client.name)


def normalize_spec(spec: McpConnectorSpec) -> McpConnectorSpec:
    name = spec.name.strip()
    if not name:
        raise ValueError("name is required")
    transport = spec.transport
    if transport not in ("stdio", "streamable_http"):
        raise ValueError("transport must be stdio or streamable_http")
    args = [item for item in spec.args if item != ""]
    allowed_prefixes = _allowed_env_prefixes()
    env_refs = _names(spec.env_refs, "env ref", allowed_prefixes)
    header_refs = _headers(spec.header_refs, allowed_prefixes)
    if transport == "stdio":
        command = spec.command.strip()
        if not command or "\x00" in command or "\n" in command:
            raise ValueError("command is required")
        if urlparse(command).scheme in ("http", "https"):
            raise ValueError("use the remote address transport for an http url")
        return spec.model_copy(
            update={
                "name": name,
                "command": command,
                "args": args,
                "env_refs": env_refs,
                "url": "",
                "header_refs": header_refs,
            }
        )
    return spec.model_copy(
        update={
            "name": name,
            "command": "",
            "args": [],
            "env_refs": env_refs,
            "url": validate_mcp_http_url(spec.url),
            "header_refs": header_refs,
        }
    )


def validate_mcp_http_url(url: str) -> str:
    """Allow HTTP on a local or LAN host. Public hosts must be HTTPS.

    Query parameters that usually carry a secret are rejected so the saved
    connector cannot become a place to store a token.
    """

    text = url.strip()
    parsed = urlparse(text)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("url must be http or https and include a host")
    if parsed.username or parsed.password:
        raise ValueError("url must not include a username or password")
    for key, _value in parse_qsl(parsed.query, keep_blank_values=True):
        if key.lower() in _SECRET_QUERY:
            raise ValueError("url must not include a secret query parameter")
    host = parsed.hostname.lower().rstrip(".")
    if _is_private_or_local_host(host):
        return text
    if parsed.scheme != "https":
        raise ValueError("public MCP server URLs must use https")
    if _public_ip_rejected(host):
        raise ValueError("public MCP server URLs must not use a private address")
    return text


def client_name(spec: McpConnectorSpec) -> str:
    candidate = spec.id.strip()
    if _NAME.fullmatch(candidate):
        return candidate
    slug = re.sub(r"[^A-Za-z0-9_-]", "", candidate) or "mcp"
    return slug[:64]


def _allowed_env_prefixes() -> tuple[str, ...]:
    return McpSettings().allowed_env_prefixes


def _validate_env_ref(name: str, prefixes: tuple[str, ...]) -> None:
    if not _ENV_NAME.fullmatch(name):
        raise ValueError("env ref must be a name, not a value")
    if not any(name.startswith(prefix) for prefix in prefixes):
        raise ValueError("env ref is outside the MCP environment prefix allowlist")


def _names(items: list[str], label: str, prefixes: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for item in items:
        name = item.strip()
        if not name:
            continue
        _validate_env_ref(name, prefixes)
        out.append(name)
    return out


def _headers(items: list[McpHeaderRef], prefixes: tuple[str, ...]) -> list[McpHeaderRef]:
    out: list[McpHeaderRef] = []
    seen: set[str] = set()
    for item in items:
        name = item.name.strip()
        env = item.env.strip()
        if not _HEADER_NAME.fullmatch(name):
            raise ValueError("header name is invalid")
        _validate_env_ref(env, prefixes)
        folded = name.lower()
        if folded in seen:
            raise ValueError("duplicate header name")
        seen.add(folded)
        out.append(McpHeaderRef(name=name, env=env))
    return out


def _is_private_or_local_host(host: str) -> bool:
    if host in _LOCAL_HOSTS or host.endswith(_LOCAL_SUFFIXES):
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return addr.is_loopback or addr.is_private or addr.is_link_local


def _public_ip_rejected(host: str) -> bool:
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def _resolved(
    spec: McpConnectorSpec, *, log_missing: bool
) -> tuple[dict[str, str], dict[str, str]]:
    environ = process_environment()
    env = {name: environ.get(name, "") for name in spec.env_refs}
    if log_missing:
        missing = [name for name, value in env.items() if not value]
        if missing:
            logger.info("mcp connector %s missing env refs: %s", spec.id, ",".join(missing))
    headers = {"Accept": _ACCEPT}
    for ref in spec.header_refs:
        value = environ.get(ref.env, "")
        if not value:
            if log_missing:
                logger.info("mcp connector %s missing header env %s", spec.id, ref.env)
            continue
        headers[ref.name] = value
    return env, headers


def _fingerprint(spec: McpConnectorSpec, env: dict[str, str], headers: dict[str, str]) -> str:
    payload = {
        "transport": spec.transport,
        "command": spec.command,
        "args": list(spec.args),
        "url": spec.url,
        "env": {key: env[key] for key in sorted(env)},
        "headers": {key: headers[key] for key in sorted(headers)},
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _log_target(spec: McpConnectorSpec) -> str:
    if spec.transport != "streamable_http":
        return "stdio"
    parsed = urlparse(spec.url)
    return f"{parsed.scheme}://{parsed.hostname or ''}"


def _build_client(
    name: str,
    spec: McpConnectorSpec,
    env: dict[str, str],
    headers: dict[str, str],
) -> MCPClient:
    if spec.transport == "stdio":
        # Keep process secrets out of the child. PATH and locale are enough for
        # command lookup and diagnostics; connector secrets must be explicitly
        # declared through the prefix allowlist.
        environ = process_environment()
        child_env = {key: environ[key] for key in ("PATH", "HOME", "TMPDIR", "LANG") if key in environ}
        for key, value in env.items():
            if value:
                child_env[key] = value
        config: StdioMCPConfig | HttpMCPConfig = StdioMCPConfig(
            command=spec.command,
            args=spec.args,
            env=child_env,
        )
    else:
        config = HttpMCPConfig(url=spec.url, headers=headers, timeout=20.0)
    return MCPClient(name=name, is_stateful=True, mcp_config=config, execution_timeout=30.0)


def _serialize_session(client: MCPClient) -> None:
    """One MCP session cannot serve two calls at once. Share it under a lock."""

    session = client._session
    if session is None:
        return
    lock = asyncio.Lock()
    call_tool = session.call_tool
    list_tools = session.list_tools

    async def locked_call(*args: Any, **kwargs: Any) -> Any:
        async with lock:
            return await call_tool(*args, **kwargs)

    async def locked_list(*args: Any, **kwargs: Any) -> Any:
        async with lock:
            return await list_tools(*args, **kwargs)

    session.call_tool = locked_call
    session.list_tools = locked_list
