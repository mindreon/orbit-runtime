"""A throwaway Postgres database with the orbit-control schema, for tests that need the real tables and roles.

The schema is not copied here: the goose migrations of an orbit-control checkout are applied as they are, as the
`orbit_owner` role, and the code under test connects as `orbit_worker` (no BYPASSRLS, not an owner), so row-level
security and the column grants are exactly what production runs with. Two variables select the setup:

- `ORBIT_TEST_POSTGRES_URL`: a superuser URL (creates roles and one database per test session).
- `ORBIT_CONTROL_DIR`: an orbit-control checkout; the default is the sibling `orbit-control` of the infra workspace.

Without the URL the tests skip. CI sets `ORBIT_REQUIRE_PG_TESTS=1`, and then a missing URL or checkout is a failure.
"""

from __future__ import annotations

import asyncio
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import asyncpg

ROLE_PASSWORD = "orbit-test"
_ROLES = {
    "orbit_owner": "LOGIN NOSUPERUSER BYPASSRLS",
    "orbit_app": "LOGIN NOSUPERUSER NOBYPASSRLS",
    "orbit_ops": "LOGIN NOSUPERUSER NOBYPASSRLS",
    "orbit_worker": "LOGIN NOSUPERUSER NOBYPASSRLS",
}
DEFAULT_CONTROL_DIR = Path(__file__).resolve().parents[2] / "orbit-control"
_MIGRATIONS = Path("internal/store/migrations/sql")


@dataclass(frozen=True)
class ControlDb:
    admin_url: str
    database: str
    migrations_dir: Path

    def url(self, role: str) -> str:
        parts = urlsplit(self.admin_url)
        host = parts.netloc.rpartition("@")[2]
        netloc = f"{role}:{quote(ROLE_PASSWORD)}@{host}"
        return urlunsplit((parts.scheme, netloc, f"/{self.database}", "", ""))

    async def owner(self) -> asyncpg.Connection:
        return await asyncpg.connect(self.url("orbit_owner"))


def unavailable_reason() -> str | None:
    """Why the database tests cannot run here, or None."""
    if not os.environ.get("ORBIT_TEST_POSTGRES_URL"):
        return "ORBIT_TEST_POSTGRES_URL is not set"
    if not _migrations_dir().is_dir():
        return f"no orbit-control migrations at {_migrations_dir()} (set ORBIT_CONTROL_DIR)"
    return None


def required() -> bool:
    return os.environ.get("ORBIT_REQUIRE_PG_TESTS") == "1"


def _migrations_dir() -> Path:
    return Path(os.environ.get("ORBIT_CONTROL_DIR") or DEFAULT_CONTROL_DIR) / _MIGRATIONS


def up_section(sql: str) -> str:
    """The part of a goose migration that runs on the way up, without goose's own comment directives."""
    up = sql.split("-- +goose Down", 1)[0]
    return re.sub(r"^-- \+goose .*$", "", up, flags=re.MULTILINE)


async def _create(admin_url: str, database: str) -> None:
    admin = await asyncpg.connect(admin_url)
    try:
        for role, attributes in _ROLES.items():
            exists = await admin.fetchval("SELECT 1 FROM pg_roles WHERE rolname=$1", role)
            verb = "ALTER" if exists else "CREATE"
            await admin.execute(f"{verb} ROLE {role} {attributes} PASSWORD '{ROLE_PASSWORD}'")
        await admin.execute(f"CREATE DATABASE {database} OWNER orbit_owner")
        await admin.execute(f"REVOKE ALL ON DATABASE {database} FROM PUBLIC")
        await admin.execute(
            f"GRANT CONNECT ON DATABASE {database} TO orbit_app, orbit_ops, orbit_worker"
        )
    finally:
        await admin.close()


async def _migrate(db: ControlDb) -> None:
    owner = await db.owner()
    try:
        for path in sorted(db.migrations_dir.glob("*.sql")):
            await owner.execute(up_section(path.read_text()))
        await owner.execute(
            "INSERT INTO tenants(id) VALUES ('default'), ('tenant-a'), ('tenant-b')"
        )
    finally:
        await owner.close()


async def _drop(db: ControlDb) -> None:
    admin = await asyncpg.connect(db.admin_url)
    try:
        await admin.execute(f"DROP DATABASE IF EXISTS {db.database} WITH (FORCE)")
    finally:
        await admin.close()


def create_control_db() -> ControlDb:
    db = ControlDb(
        admin_url=os.environ["ORBIT_TEST_POSTGRES_URL"],
        database=f"orbit_it_{uuid.uuid4().hex[:12]}",
        migrations_dir=_migrations_dir(),
    )
    asyncio.run(_create(db.admin_url, db.database))
    try:
        asyncio.run(_migrate(db))
    except BaseException:
        asyncio.run(_drop(db))
        raise
    return db


def drop_control_db(db: ControlDb) -> None:
    asyncio.run(_drop(db))
