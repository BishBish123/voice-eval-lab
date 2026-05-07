"""End-to-end API tests for the voice-eval-lab backend.

Uses FastAPI's TestClient (sync WSGI wrapper) so no real network port is bound.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from voice_eval_lab.backend.app import create_app
from voice_eval_lab.backend.store import InMemorySessionStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client(store: InMemorySessionStore | None = None, **env: str) -> TestClient:
    """Create a TestClient with the given env overrides."""
    with patch.dict(os.environ, env, clear=False):
        return TestClient(create_app(store=store or InMemorySessionStore()))


# ---------------------------------------------------------------------------
# POST /sessions — no auth required
# ---------------------------------------------------------------------------


def test_create_session_no_auth_returns_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BACKEND_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)

    with _client() as client:
        resp = client.post("/sessions", json={"user_id": "alice"})

    assert resp.status_code == 201
    body = resp.json()
    assert "session_id" in body
    assert body["livekit_token"] is None
    assert "expires_at" in body


def test_create_session_with_scenario(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BACKEND_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)

    with _client() as client:
        resp = client.post("/sessions", json={"user_id": "bob", "scenario_id": "sc-1"})

    assert resp.status_code == 201
    assert resp.json()["livekit_token"] is None


# ---------------------------------------------------------------------------
# POST /sessions — auth required, no header → 401
# ---------------------------------------------------------------------------


def test_create_session_auth_required_no_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)

    with TestClient(create_app(store=InMemorySessionStore(), auth_token="secret-token")) as client:
        resp = client.post("/sessions", json={"user_id": "eve"})

    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /sessions — auth required, valid token → 200
# ---------------------------------------------------------------------------


def test_create_session_auth_required_valid_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)

    with TestClient(create_app(store=InMemorySessionStore(), auth_token="mytoken")) as client:
        resp = client.post(
            "/sessions",
            json={"user_id": "carol"},
            headers={"Authorization": "Bearer mytoken"},
        )

    assert resp.status_code == 201
    assert resp.json()["session_id"] is not None


# ---------------------------------------------------------------------------
# POST /sessions — LIVEKIT_API_KEY set → non-null JWT token
# ---------------------------------------------------------------------------


def test_create_session_livekit_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BACKEND_AUTH_TOKEN", raising=False)
    # Use a 32-byte secret so pyjwt doesn't emit InsecureKeyLengthWarning
    monkeypatch.setenv("LIVEKIT_API_KEY", "testkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "a" * 32)

    with TestClient(create_app(store=InMemorySessionStore())) as client:
        resp = client.post("/sessions", json={"user_id": "dave"})

    assert resp.status_code == 201
    body = resp.json()
    token = body["livekit_token"]
    assert token is not None

    # Decode and verify sub claim (no signature verification needed in test)
    import jwt as pyjwt

    decoded: Any = pyjwt.decode(token, options={"verify_signature": False})
    assert decoded["sub"] == "dave"


# ---------------------------------------------------------------------------
# GET /sessions/{id} round-trip
# ---------------------------------------------------------------------------


def test_get_session_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BACKEND_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)

    store = InMemorySessionStore()
    with _client(store=store) as client:
        create_resp = client.post("/sessions", json={"user_id": "eve", "scenario_id": "s1"})
        assert create_resp.status_code == 201
        session_id = create_resp.json()["session_id"]

        get_resp = client.get(f"/sessions/{session_id}")

    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["session_id"] == session_id
    assert body["user_id"] == "eve"
    assert body["scenario_id"] == "s1"
    assert body["ended_at"] is None


def test_get_session_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BACKEND_AUTH_TOKEN", raising=False)

    with _client() as client:
        resp = client.get("/sessions/nonexistent-id")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /sessions/{id}/end → ended state
# ---------------------------------------------------------------------------


def test_end_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BACKEND_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)

    store = InMemorySessionStore()
    with _client(store=store) as client:
        create_resp = client.post("/sessions", json={"user_id": "frank"})
        assert create_resp.status_code == 201
        session_id = create_resp.json()["session_id"]

        end_resp = client.post(f"/sessions/{session_id}/end")

    assert end_resp.status_code == 200
    body = end_resp.json()
    assert body["session_id"] == session_id
    assert body["ended_at"] is not None


def test_end_session_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BACKEND_AUTH_TOKEN", raising=False)

    with _client() as client:
        resp = client.post("/sessions/no-such-id/end")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /healthz
# ---------------------------------------------------------------------------


def test_healthz_always_200(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BACKEND_AUTH_TOKEN", raising=False)

    with _client() as client:
        resp = client.get("/healthz")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /readyz — 503 when Postgres backend unreachable
# ---------------------------------------------------------------------------


def test_readyz_503_when_postgres_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """readyz returns 503 when PostgresSessionStore.healthz() returns False."""
    monkeypatch.delenv("BACKEND_AUTH_TOKEN", raising=False)

    from voice_eval_lab.backend.store import PostgresSessionStore

    # Build a store whose healthz() always fails
    pg_store = PostgresSessionStore(dsn="postgresql://bad-host/nodb")
    pg_store._conn = MagicMock()
    pg_store._conn.fetchval = AsyncMock(side_effect=OSError("connection refused"))

    with TestClient(create_app(store=pg_store)) as client:
        resp = client.get("/readyz")

    assert resp.status_code == 503


def test_readyz_200_when_inmemory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BACKEND_AUTH_TOKEN", raising=False)

    with _client() as client:
        resp = client.get("/readyz")

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
