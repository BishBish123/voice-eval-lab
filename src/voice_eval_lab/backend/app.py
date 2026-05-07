"""FastAPI application for the voice-eval-lab backend.

Endpoints:
  POST   /sessions                      — mint a session + optional LiveKit JWT
  GET    /sessions/{session_id}         — retrieve session state
  POST   /sessions/{session_id}/end     — mark session ended
  GET    /sessions/{session_id}/events  — SSE stream of transcript turn events
  GET    /healthz                       — liveness probe
  GET    /readyz                        — readiness probe (checks persistence backend)
  GET    /                              — static browser frontend (index.html)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from voice_eval_lab.backend.auth import configure_auth, load_auth_from_env, require_auth
from voice_eval_lab.backend.models import CreateSessionRequest, SessionResponse, SessionState
from voice_eval_lab.backend.store import SessionStore, make_session_store

logger = logging.getLogger(__name__)

_SESSION_TTL_HOURS = 24

# Per-session asyncio queues for SSE turn events.
# Key: session_id, Value: asyncio.Queue of serialised JSON strings.
_event_queues: dict[str, asyncio.Queue[str]] = {}

_STATIC_DIR = Path(__file__).parent / "static"

# Interval (seconds) between mock turn events in the SSE demo stream.
_MOCK_EVENT_INTERVAL_S = 2.0

# Mock turns emitted by the SSE demo stream when no real pipeline is wired.
_MOCK_TURNS = [
    {"role": "user", "text": "Hello, can you help me?"},
    {"role": "agent", "text": "Of course! What do you need?"},
    {"role": "user", "text": "What metrics does voice-eval-lab track?"},
    {
        "role": "agent",
        "text": (
            "Turn latency, WER, faithfulness, barge-in success, "
            "false-trigger rate, and four diagnostic metrics."
        ),
    },
]


# ---------------------------------------------------------------------------
# LiveKit JWT signing
# ---------------------------------------------------------------------------


def _mint_livekit_token(user_id: str, room_name: str) -> str | None:
    """Mint a LiveKit access token JWT when credentials are available.

    Returns ``None`` when ``LIVEKIT_API_KEY`` or ``LIVEKIT_API_SECRET`` are
    not set.  Uses ``pyjwt`` for signing; when pyjwt is not installed, returns
    ``None`` with a warning (install ``voice-eval-lab[real]`` to enable).

    The JWT payload follows the LiveKit access-token spec:
      https://docs.livekit.io/home/server/generating-tokens/
    """
    api_key = os.environ.get("LIVEKIT_API_KEY", "")
    api_secret = os.environ.get("LIVEKIT_API_SECRET", "")
    if not api_key or not api_secret:
        return None

    try:
        import jwt as pyjwt
    except ImportError:
        logger.warning(
            "pyjwt is not installed — LiveKit JWT minting is unavailable. "
            "Install with: pip install 'voice-eval-lab[real]'"
        )
        return None

    now = datetime.now(tz=UTC)
    payload = {
        "iss": api_key,
        "sub": user_id,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=_SESSION_TTL_HOURS)).timestamp()),
        "nbf": int(now.timestamp()),
        "video": {
            "room": room_name,
            "roomJoin": True,
            "canPublish": True,
            "canSubscribe": True,
        },
    }
    token: str = pyjwt.encode(payload, api_secret, algorithm="HS256")
    return token


# ---------------------------------------------------------------------------
# SSE helpers (module-level so create_app stays within statement budget)
# ---------------------------------------------------------------------------


async def _mock_producer(queue: asyncio.Queue[str]) -> None:
    """Push demo turns onto *queue* at :data:`_MOCK_EVENT_INTERVAL_S` intervals."""
    for turn in _MOCK_TURNS:
        await asyncio.sleep(_MOCK_EVENT_INTERVAL_S)
        await queue.put(json.dumps(turn))
    # Empty string is the sentinel that signals end-of-stream.
    await asyncio.sleep(_MOCK_EVENT_INTERVAL_S)
    await queue.put("")


async def _sse_event_generator(session_id: str) -> AsyncIterator[str]:
    """Drain the per-session queue and yield SSE-formatted strings."""
    queue: asyncio.Queue[str] = asyncio.Queue()
    _event_queues[session_id] = queue
    producer = asyncio.create_task(_mock_producer(queue))
    try:
        while True:
            payload = await queue.get()
            if not payload:
                break
            yield f"event: turn\ndata: {payload}\n\n"
    finally:
        producer.cancel()
        _event_queues.pop(session_id, None)


def _make_sse_response(session_id: str) -> StreamingResponse:
    """Build the SSE StreamingResponse for a session."""
    return StreamingResponse(
        _sse_event_generator(session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    store: SessionStore | None = None,
    auth_token: str | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        store: Explicit session store to use (for testing). When ``None``,
            ``make_session_store()`` is called to select the backend from env.
        auth_token: Explicit auth token to use (for testing). When ``None``,
            ``BACKEND_AUTH_TOKEN`` env var is read.

    Returns:
        Configured :class:`FastAPI` instance.
    """
    _store: SessionStore = store if store is not None else make_session_store()
    _token = auth_token if auth_token is not None else load_auth_from_env()

    @asynccontextmanager
    async def _lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
        configure_auth(_token)
        store_type = type(_store).__name__
        lk_enabled = bool(
            os.environ.get("LIVEKIT_API_KEY") and os.environ.get("LIVEKIT_API_SECRET")
        )
        logger.info(
            "voice-eval-lab backend starting — store=%s auth=%s livekit=%s",
            store_type,
            "enabled" if _token else "DISABLED",
            "enabled" if lk_enabled else "gated (LIVEKIT_API_KEY/SECRET not set)",
        )
        yield

    app = FastAPI(title="voice-eval-lab backend", version="0.1.0", lifespan=_lifespan)

    @app.post("/sessions", response_model=SessionResponse, status_code=201)
    async def create_session(
        body: CreateSessionRequest,
        _: None = Depends(require_auth),
    ) -> SessionResponse:
        session_id = str(uuid.uuid4())
        now = datetime.now(tz=UTC)
        expires_at = now + timedelta(hours=_SESSION_TTL_HOURS)
        room_name = f"room-{session_id}"
        livekit_token = _mint_livekit_token(body.user_id, room_name)
        await _store.create(
            session_id=session_id,
            user_id=body.user_id,
            scenario_id=body.scenario_id,
            started_at=now,
        )
        return SessionResponse(
            session_id=session_id,
            livekit_token=livekit_token,
            expires_at=expires_at.isoformat(),
        )

    @app.get("/sessions/{session_id}", response_model=SessionState)
    async def get_session(
        session_id: str,
        _: None = Depends(require_auth),
    ) -> SessionState:
        record = await _store.get(session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return _record_to_state(record)

    @app.post("/sessions/{session_id}/end", response_model=SessionState)
    async def end_session(
        session_id: str,
        _: None = Depends(require_auth),
    ) -> SessionState:
        ended_at = datetime.now(tz=UTC)
        record = await _store.end(session_id, ended_at)
        if record is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return _record_to_state(record)

    @app.get("/sessions/{session_id}/events")
    async def session_events(session_id: str) -> StreamingResponse:
        """SSE stream of transcript turn events (mock demo when no pipeline is wired)."""
        return _make_sse_response(session_id)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(response: Response) -> dict[str, Any]:
        healthy = await _store.healthz()
        if not healthy:
            response.status_code = 503
            return {"status": "unavailable", "store": type(_store).__name__}
        return {"status": "ok", "store": type(_store).__name__}

    # Mounted LAST so API routes always take precedence over the static catch-all.
    if _STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record_to_state(record: dict[str, Any]) -> SessionState:
    return SessionState(
        session_id=record["session_id"],
        user_id=record["user_id"],
        scenario_id=record.get("scenario_id"),
        started_at=record["started_at"],
        ended_at=record.get("ended_at"),
    )


__all__ = ["create_app"]
