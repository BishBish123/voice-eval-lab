"""Tests for the browser frontend and SSE endpoint.

Covers:
- Static index.html served at GET /
- Expected DOM hooks present in index.html
- API routes still work after static mount
- SSE endpoint content-type and event shape
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from voice_eval_lab.backend.app import _STATIC_DIR, create_app
from voice_eval_lab.backend.store import InMemorySessionStore

# ---------------------------------------------------------------------------
# Shared client factory (no auth, no LiveKit)
# ---------------------------------------------------------------------------


def _client(**env: str) -> TestClient:
    with patch.dict(os.environ, env, clear=False):
        return TestClient(create_app(store=InMemorySessionStore()))


# ---------------------------------------------------------------------------
# 1. Static files — index.html served at GET /
# ---------------------------------------------------------------------------


def test_static_dir_exists() -> None:
    assert _STATIC_DIR.is_dir(), f"static dir missing: {_STATIC_DIR}"


def test_static_index_html_exists() -> None:
    assert (_STATIC_DIR / "index.html").is_file()


def test_static_app_js_exists() -> None:
    assert (_STATIC_DIR / "app.js").is_file()


def test_static_styles_css_exists() -> None:
    assert (_STATIC_DIR / "styles.css").is_file()


def test_get_root_returns_index_html(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BACKEND_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)

    with _client() as client:
        resp = client.get("/")

    assert resp.status_code == 200
    ct = resp.headers.get("content-type", "")
    assert "text/html" in ct


# ---------------------------------------------------------------------------
# 2. index.html contains expected DOM hooks
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def index_html_content() -> str:
    return (_STATIC_DIR / "index.html").read_text()


def test_index_has_start_session_button(index_html_content: str) -> None:
    assert 'id="start-session"' in index_html_content


def test_index_has_end_session_button(index_html_content: str) -> None:
    assert 'id="end-session"' in index_html_content


def test_index_has_transcript_div(index_html_content: str) -> None:
    assert 'id="transcript"' in index_html_content


def test_index_has_no_backend_banner(index_html_content: str) -> None:
    assert 'id="no-backend-banner"' in index_html_content


def test_index_has_waveform_canvas(index_html_content: str) -> None:
    assert 'id="waveform"' in index_html_content


def test_index_has_connection_status(index_html_content: str) -> None:
    assert 'id="connection-status"' in index_html_content


def test_index_loads_app_js(index_html_content: str) -> None:
    assert "app.js" in index_html_content


# ---------------------------------------------------------------------------
# 3. API routes still work after static mount
# ---------------------------------------------------------------------------


def test_healthz_still_works_with_static_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BACKEND_AUTH_TOKEN", raising=False)

    with _client() as client:
        resp = client.get("/healthz")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_post_sessions_still_works_with_static_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BACKEND_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)

    with _client() as client:
        resp = client.post("/sessions", json={"user_id": "frontend-test"})

    assert resp.status_code == 201
    body = resp.json()
    assert "session_id" in body
    assert body["livekit_token"] is None


# ---------------------------------------------------------------------------
# 4. SSE endpoint — content-type and event shape
# ---------------------------------------------------------------------------


def test_sse_endpoint_content_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /sessions/{id}/events returns text/event-stream."""
    monkeypatch.delenv("BACKEND_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)

    session_id = "sse-test-session"
    with _client() as client, client.stream("GET", f"/sessions/{session_id}/events") as resp:
        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "")
        assert "text/event-stream" in ct


def test_sse_endpoint_emits_turn_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """SSE stream emits at least one ``event: turn`` line with JSON payload."""
    monkeypatch.delenv("BACKEND_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)

    # Patch the mock interval so the test does not wait 2 s per event.
    import voice_eval_lab.backend.app as app_module

    original_interval = app_module._MOCK_EVENT_INTERVAL_S
    app_module._MOCK_EVENT_INTERVAL_S = 0.0
    try:
        session_id = "sse-turn-test"
        with _client() as client, client.stream(
            "GET", f"/sessions/{session_id}/events"
        ) as resp:
            assert resp.status_code == 200
            lines: list[str] = []
            for raw in resp.iter_lines():
                line = raw.strip()
                lines.append(line)
                # Stop after the first turn event data line.
                if line.startswith("data:"):
                    break

        turn_lines = [ln for ln in lines if ln.startswith("event: turn")]
        data_lines = [ln for ln in lines if ln.startswith("data:")]
        assert turn_lines, f"No 'event: turn' line found. Lines: {lines}"
        assert data_lines, f"No 'data:' line found. Lines: {lines}"

        # Verify the data payload is valid JSON with role and text.
        payload = json.loads(data_lines[0].removeprefix("data:").strip())
        assert "role" in payload
        assert "text" in payload
        assert payload["role"] in ("user", "agent")
    finally:
        app_module._MOCK_EVENT_INTERVAL_S = original_interval
