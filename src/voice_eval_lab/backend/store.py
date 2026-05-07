"""SessionStore Protocol with InMemorySessionStore and PostgresSessionStore implementations.

Factory: ``make_session_store()`` reads ``BACKEND_DSN`` env var and returns
the appropriate backend — Postgres if set, in-memory otherwise.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Protocol


class SessionStore(Protocol):
    """Structural protocol for session persistence backends."""

    async def create(
        self,
        session_id: str,
        user_id: str,
        scenario_id: str | None,
        started_at: datetime,
    ) -> None:
        """Persist a new session record."""
        ...

    async def get(self, session_id: str) -> dict[str, Any] | None:
        """Return session dict or None if not found."""
        ...

    async def end(self, session_id: str, ended_at: datetime) -> dict[str, Any] | None:
        """Mark a session ended, return updated state or None if not found."""
        ...

    async def healthz(self) -> bool:
        """Return True when the backend is reachable, False otherwise."""
        ...


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------


class InMemorySessionStore:
    """Thread-safe, in-process session store backed by a plain dict.

    No external dependencies. Suitable for development and testing.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    async def create(
        self,
        session_id: str,
        user_id: str,
        scenario_id: str | None,
        started_at: datetime,
    ) -> None:
        self._sessions[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "scenario_id": scenario_id,
            "started_at": started_at,
            "ended_at": None,
        }

    async def get(self, session_id: str) -> dict[str, Any] | None:
        return self._sessions.get(session_id)

    async def end(self, session_id: str, ended_at: datetime) -> dict[str, Any] | None:
        record = self._sessions.get(session_id)
        if record is None:
            return None
        record["ended_at"] = ended_at
        return dict(record)

    async def healthz(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Postgres implementation
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    scenario_id TEXT,
    started_at  TIMESTAMPTZ NOT NULL,
    ended_at    TIMESTAMPTZ
)
"""

_INSERT_SESSION = """
INSERT INTO sessions (session_id, user_id, scenario_id, started_at)
VALUES ($1, $2, $3, $4)
"""

_SELECT_SESSION = """
SELECT session_id, user_id, scenario_id, started_at, ended_at
FROM sessions WHERE session_id = $1
"""

_UPDATE_ENDED = """
UPDATE sessions SET ended_at = $2 WHERE session_id = $1
RETURNING session_id, user_id, scenario_id, started_at, ended_at
"""


def _require_asyncpg() -> Any:
    try:
        import asyncpg

        return asyncpg
    except ImportError as exc:
        raise ImportError(
            "asyncpg is required for PostgresSessionStore. "
            "Install with: pip install 'voice-eval-lab[real]'"
        ) from exc


class PostgresSessionStore:
    """Session store backed by Postgres.

    The connection is lazy — the first call opens it and creates the table.
    Args:
        dsn: Postgres connection string (defaults to ``BACKEND_DSN`` env var).
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn: Any = None

    async def _get_conn(self) -> Any:
        if self._conn is None:
            asyncpg = _require_asyncpg()
            self._conn = await asyncpg.connect(self._dsn)
            await self._conn.execute(_CREATE_TABLE)
        return self._conn

    async def create(
        self,
        session_id: str,
        user_id: str,
        scenario_id: str | None,
        started_at: datetime,
    ) -> None:
        conn = await self._get_conn()
        await conn.execute(_INSERT_SESSION, session_id, user_id, scenario_id, started_at)

    async def get(self, session_id: str) -> dict[str, Any] | None:
        conn = await self._get_conn()
        row = await conn.fetchrow(_SELECT_SESSION, session_id)
        if row is None:
            return None
        return dict(row)

    async def end(self, session_id: str, ended_at: datetime) -> dict[str, Any] | None:
        conn = await self._get_conn()
        row = await conn.fetchrow(_UPDATE_ENDED, session_id, ended_at)
        if row is None:
            return None
        return dict(row)

    async def healthz(self) -> bool:
        try:
            conn = await self._get_conn()
            await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_session_store() -> InMemorySessionStore | PostgresSessionStore:
    """Return the best available SessionStore for the current environment.

    When ``BACKEND_DSN`` is set, returns a :class:`PostgresSessionStore`.
    Otherwise returns an :class:`InMemorySessionStore`.
    """
    dsn = os.environ.get("BACKEND_DSN", "")
    if dsn:
        return PostgresSessionStore(dsn=dsn)
    return InMemorySessionStore()


__all__ = [
    "InMemorySessionStore",
    "PostgresSessionStore",
    "SessionStore",
    "make_session_store",
]
