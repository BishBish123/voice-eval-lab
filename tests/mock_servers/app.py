"""FastAPI mock server that responds with documented provider JSON/bytes shapes.

Each endpoint:
- Validates that the relevant auth header is present (any non-empty value).
- Returns the fixture JSON / bytes corresponding to the documented API shape.
- Supports ``?force=500`` query parameter: first call returns 500, second returns 200.
  Counter resets after each cycle so the pattern repeats.

Endpoints mounted:
    POST /groq/openai/v1/chat/completions      — OpenAI-compatible
    POST /deepgram/v1/listen                   — Deepgram Nova STT
    POST /cartesia/tts/bytes                   — Cartesia TTS (raw PCM bytes)
    POST /elevenlabs/v1/text-to-speech/{vid}   — ElevenLabs TTS (raw PCM bytes)
    POST /anthropic/v1/messages                — Anthropic Messages API
    POST /openai/v1/chat/completions           — OpenAI Chat Completions
"""

from __future__ import annotations

import io
import json
import wave
from collections import defaultdict

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response

# ---------------------------------------------------------------------------
# Silence WAV generator
# ---------------------------------------------------------------------------

def _make_silence_wav(n_frames: int = 1600, framerate: int = 16000) -> bytes:
    """Return a minimal 16-bit mono silence WAV blob."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(framerate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


_SILENCE_WAV = _make_silence_wav()


# ---------------------------------------------------------------------------
# force=500 helper
# ---------------------------------------------------------------------------

def _maybe_force_500(app: FastAPI, key: str, force: str | None) -> Response | None:
    """Return a 500 Response the first time ``?force=500`` is seen for *key*.

    On the second call, resets the counter and returns None (→ 200).
    Pattern repeats: call N=1 → 500, call N=2 → 200, call N=3 → 500, ...
    """
    if force != "500":
        return None
    counters: dict[str, int] = app.state.force_500_counters
    counters[key] += 1
    if counters[key] == 1:
        return Response(
            content='{"error":"forced 500"}',
            status_code=500,
            media_type="application/json",
        )
    counters[key] = 0
    return None


def _json(payload: object, status: int = 200) -> Response:
    return Response(
        content=json.dumps(payload),
        status_code=status,
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# Fixture payloads
# ---------------------------------------------------------------------------

_GROQ_RESPONSE = {
    "choices": [{"message": {"content": "mock groq reply", "role": "assistant"}}]
}

_DEEPGRAM_RESPONSE = {
    "results": {
        "channels": [
            {"alternatives": [{"transcript": "hello world", "confidence": 0.95}]}
        ]
    }
}

# Anthropic shape — JSON embedded in text content (as LLMJudge expects)
_ANTHROPIC_RESPONSE = {
    "content": [
        {"type": "text", "text": '{"score":0.7,"rationale":"ok"}'}
    ]
}

# OpenAI shape used by LLMJudge fallback
_OPENAI_RESPONSE = {
    "choices": [
        {"message": {"content": '{"score":0.8,"rationale":"fine"}', "role": "assistant"}}
    ]
}


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_mock_app() -> FastAPI:
    """Return a fresh FastAPI app with all voice-eval-lab mock endpoints mounted.

    The app carries a ``force_500_counters`` dict (keyed by route) on
    ``app.state`` so the first ``?force=500`` call returns 500 and the second
    returns 200.  The counter resets so the cycle can repeat.
    """
    app = FastAPI(title="voice-eval-lab-mock-api")
    app.state.force_500_counters: dict[str, int] = defaultdict(int)  # type: ignore[assignment]

    # -----------------------------------------------------------------------
    # Groq — POST /groq/openai/v1/chat/completions
    # Auth: Authorization: Bearer <key>
    # -----------------------------------------------------------------------

    @app.post("/groq/openai/v1/chat/completions")
    async def groq_chat(
        request: Request,
        force: str | None = Query(default=None),
        authorization: str | None = Header(default=None),
    ) -> Response:
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        forced = _maybe_force_500(app, "groq_chat", force)
        if forced is not None:
            return forced
        return _json(_GROQ_RESPONSE)

    # -----------------------------------------------------------------------
    # Deepgram — POST /deepgram/v1/listen
    # Auth: Authorization: Token <key>
    # -----------------------------------------------------------------------

    @app.post("/deepgram/v1/listen")
    async def deepgram_listen(
        request: Request,
        force: str | None = Query(default=None),
        authorization: str | None = Header(default=None),
    ) -> Response:
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        forced = _maybe_force_500(app, "deepgram_listen", force)
        if forced is not None:
            return forced
        return _json(_DEEPGRAM_RESPONSE)

    # -----------------------------------------------------------------------
    # Cartesia — POST /cartesia/tts/bytes
    # Auth: Authorization: Bearer <key>
    # Returns: raw PCM/WAV bytes
    # -----------------------------------------------------------------------

    @app.post("/cartesia/tts/bytes")
    async def cartesia_tts(
        request: Request,
        force: str | None = Query(default=None),
        authorization: str | None = Header(default=None),
    ) -> Response:
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        forced = _maybe_force_500(app, "cartesia_tts", force)
        if forced is not None:
            return forced
        return Response(
            content=_SILENCE_WAV,
            status_code=200,
            media_type="audio/wav",
        )

    # -----------------------------------------------------------------------
    # ElevenLabs — POST /elevenlabs/v1/text-to-speech/{voice_id}
    # Auth: xi-api-key: <key>
    # Returns: raw PCM/WAV bytes
    # -----------------------------------------------------------------------

    @app.post("/elevenlabs/v1/text-to-speech/{voice_id}")
    async def elevenlabs_tts(
        voice_id: str,
        request: Request,
        force: str | None = Query(default=None),
        xi_api_key: str | None = Header(default=None, alias="xi-api-key"),
    ) -> Response:
        if not xi_api_key:
            raise HTTPException(status_code=401, detail="Missing xi-api-key header")
        forced = _maybe_force_500(app, "elevenlabs_tts", force)
        if forced is not None:
            return forced
        return Response(
            content=_SILENCE_WAV,
            status_code=200,
            media_type="audio/wav",
        )

    # -----------------------------------------------------------------------
    # Anthropic — POST /anthropic/v1/messages
    # Auth: x-api-key: <key>
    # Returns: Anthropic Messages API shape with JSON-embedded score
    # -----------------------------------------------------------------------

    @app.post("/anthropic/v1/messages")
    async def anthropic_messages(
        request: Request,
        force: str | None = Query(default=None),
        x_api_key: str | None = Header(default=None, alias="x-api-key"),
    ) -> Response:
        if not x_api_key:
            raise HTTPException(status_code=401, detail="Missing x-api-key header")
        forced = _maybe_force_500(app, "anthropic_messages", force)
        if forced is not None:
            return forced
        return _json(_ANTHROPIC_RESPONSE)

    # -----------------------------------------------------------------------
    # OpenAI — POST /openai/v1/chat/completions
    # Auth: Authorization: Bearer <key>
    # Returns: OpenAI chat shape with JSON-embedded score
    # -----------------------------------------------------------------------

    @app.post("/openai/v1/chat/completions")
    async def openai_chat(
        request: Request,
        force: str | None = Query(default=None),
        authorization: str | None = Header(default=None),
    ) -> Response:
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        forced = _maybe_force_500(app, "openai_chat", force)
        if forced is not None:
            return forced
        return _json(_OPENAI_RESPONSE)

    return app
