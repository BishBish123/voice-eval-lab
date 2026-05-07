"""Tests for SessionStore implementations and factory."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voice_eval_lab.backend.store import InMemorySessionStore, make_session_store

# ---------------------------------------------------------------------------
# InMemorySessionStore
# ---------------------------------------------------------------------------

_NOW = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture()
def store() -> InMemorySessionStore:
    return InMemorySessionStore()


async def test_in_memory_create_and_get(store: InMemorySessionStore) -> None:
    await store.create("s1", "user-a", "scenario-x", _NOW)
    record = await store.get("s1")
    assert record is not None
    assert record["session_id"] == "s1"
    assert record["user_id"] == "user-a"
    assert record["scenario_id"] == "scenario-x"
    assert record["started_at"] == _NOW
    assert record["ended_at"] is None


async def test_in_memory_get_missing(store: InMemorySessionStore) -> None:
    result = await store.get("does-not-exist")
    assert result is None


async def test_in_memory_end(store: InMemorySessionStore) -> None:
    await store.create("s2", "user-b", None, _NOW)
    ended_at = datetime(2025, 1, 1, 13, 0, 0, tzinfo=UTC)
    result = await store.end("s2", ended_at)
    assert result is not None
    assert result["ended_at"] == ended_at
    # Verify the in-store copy is also updated
    fetched = await store.get("s2")
    assert fetched is not None
    assert fetched["ended_at"] == ended_at


async def test_in_memory_end_missing(store: InMemorySessionStore) -> None:
    result = await store.end("no-such-session", _NOW)
    assert result is None


async def test_in_memory_healthz(store: InMemorySessionStore) -> None:
    assert await store.healthz() is True


# ---------------------------------------------------------------------------
# PostgresSessionStore — mocked asyncpg
# ---------------------------------------------------------------------------


def _make_mock_conn(row: dict[str, Any] | None = None) -> MagicMock:
    """Return a mock asyncpg connection with configurable fetchrow/execute."""
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=None)
    if row is not None:
        conn.fetchrow = AsyncMock(return_value=row)
    else:
        conn.fetchrow = AsyncMock(return_value=None)
    conn.fetchval = AsyncMock(return_value=1)
    return conn


async def test_postgres_store_insert(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_eval_lab.backend.store import PostgresSessionStore

    mock_conn = _make_mock_conn()
    mock_asyncpg = MagicMock()
    mock_asyncpg.connect = AsyncMock(return_value=mock_conn)

    with patch.dict("sys.modules", {"asyncpg": mock_asyncpg}):
        pg_store = PostgresSessionStore(dsn="postgresql://localhost/test")
        pg_store._conn = mock_conn  # bypass lazy _get_conn
        await pg_store.create("s3", "user-c", "sc-1", _NOW)

    mock_conn.execute.assert_called()


async def test_postgres_store_get(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_eval_lab.backend.store import PostgresSessionStore

    row = {
        "session_id": "s4",
        "user_id": "user-d",
        "scenario_id": None,
        "started_at": _NOW,
        "ended_at": None,
    }
    mock_conn = _make_mock_conn(row=row)
    mock_asyncpg = MagicMock()
    mock_asyncpg.connect = AsyncMock(return_value=mock_conn)

    with patch.dict("sys.modules", {"asyncpg": mock_asyncpg}):
        pg_store = PostgresSessionStore(dsn="postgresql://localhost/test")
        pg_store._conn = mock_conn
        result = await pg_store.get("s4")

    assert result is not None
    assert result["session_id"] == "s4"


async def test_postgres_store_end(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_eval_lab.backend.store import PostgresSessionStore

    ended_at = datetime(2025, 1, 1, 15, 0, 0, tzinfo=UTC)
    row = {
        "session_id": "s5",
        "user_id": "user-e",
        "scenario_id": "sc-2",
        "started_at": _NOW,
        "ended_at": ended_at,
    }
    mock_conn = _make_mock_conn(row=row)
    mock_asyncpg = MagicMock()
    mock_asyncpg.connect = AsyncMock(return_value=mock_conn)

    with patch.dict("sys.modules", {"asyncpg": mock_asyncpg}):
        pg_store = PostgresSessionStore(dsn="postgresql://localhost/test")
        pg_store._conn = mock_conn
        result = await pg_store.end("s5", ended_at)

    assert result is not None
    assert result["ended_at"] == ended_at


async def test_postgres_store_healthz_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_eval_lab.backend.store import PostgresSessionStore

    mock_conn = _make_mock_conn()
    with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
        pg_store = PostgresSessionStore(dsn="postgresql://localhost/test")
        pg_store._conn = mock_conn
        ok = await pg_store.healthz()

    assert ok is True


async def test_postgres_store_healthz_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_eval_lab.backend.store import PostgresSessionStore

    mock_conn = MagicMock()
    mock_conn.execute = AsyncMock(return_value=None)
    mock_conn.fetchval = AsyncMock(side_effect=OSError("connection refused"))

    with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
        pg_store = PostgresSessionStore(dsn="postgresql://localhost/test")
        pg_store._conn = mock_conn
        ok = await pg_store.healthz()

    assert ok is False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_make_session_store_inmemory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BACKEND_DSN", raising=False)
    result = make_session_store()
    assert isinstance(result, InMemorySessionStore)


def test_make_session_store_postgres(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_eval_lab.backend.store import PostgresSessionStore

    monkeypatch.setenv("BACKEND_DSN", "postgresql://localhost/testdb")
    result = make_session_store()
    assert isinstance(result, PostgresSessionStore)
