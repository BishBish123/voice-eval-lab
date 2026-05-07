"""Deepgram STT adapter — implements the ``STT`` Protocol.

Key:     ``DEEPGRAM_API_KEY``
Endpoint: POST https://api.deepgram.com/v1/listen
Model:   ``nova-3`` with ``smart_format=true``

Mock path (no key / API failure):
    Delegates to ``MockSTT`` with deterministic jitter enabled.

Real path (future / fixture-gated):
    Deepgram pre-recorded STT requires actual audio bytes (e.g. a valid
    WAV file).  The mock pipeline operates on text-only ``TurnInput``
    objects and therefore has no audio source to send.  Posting UTF-8
    text bytes as ``audio/wav`` would be rejected by Deepgram and is
    never attempted.

    ``_real_transcribe`` is gated: if no ``audio_bytes`` attribute is
    present on the turn (or if it is ``None``), the adapter logs a
    warning and falls back to the mock path.  This prevents the silent
    "text-as-WAV" bug in production.  Real-mode Deepgram requires audio
    bytes; future fixture support (e.g. an ``_audio_fixtures`` dict keyed
    by ``(conv_id, turn_index)``) should populate ``turn.audio_bytes``
    before calling this adapter.
"""

from __future__ import annotations

import logging
import os
import time
import warnings

try:
    import httpx as _httpx
except ImportError:  # pragma: no cover
    _httpx = None  # type: ignore[assignment]

from voice_eval_lab.models import PipelineSpan, Turn
from voice_eval_lab.pipeline import MockSTT

logger = logging.getLogger(__name__)

_DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"
_DEEPGRAM_MODEL = "nova-3"
_TIMEOUT_S = 10.0
_DEEPGRAM_BASE_URL_ENV = "DEEPGRAM_API_BASE_URL"


class DeepgramSTT:
    """STT adapter backed by Deepgram Nova-3.  Falls back to ``MockSTT`` when the key is absent.

    Real-call shape:
        POST /v1/listen?model=nova-3&smart_format=true
        body: raw audio bytes — a valid WAV/PCM file (NOT UTF-8 text)
        header: Authorization: Token <DEEPGRAM_API_KEY>

    IMPORTANT: Real-mode requires actual audio bytes on the turn input.
    The mock pipeline operates on text-only ``Turn`` objects; there is no
    audio source, so the real path is gated behind ``turn.audio_bytes``.
    When ``audio_bytes`` is absent or ``None``, a warning is logged and
    the adapter falls back to mock — it never POSTs text as ``audio/wav``.

    The adapter is constructed without making any network call so it is safe
    to instantiate in test environments.
    """

    def __init__(self, wer_substitution_rate: float = 0.0, base_url: str | None = None) -> None:
        self._api_key: str | None = os.environ.get("DEEPGRAM_API_KEY")
        self._mock: bool = self._api_key is None
        self._inner: MockSTT = MockSTT(wer_substitution_rate=wer_substitution_rate)
        # URL override: constructor param > env var > hard-coded default.
        self._api_url: str = (
            base_url
            or os.environ.get(_DEEPGRAM_BASE_URL_ENV, "")
            or _DEEPGRAM_API_URL
        )
        # Turn-context fields for deterministic jitter on the mock path.
        self._conv_id: str = ""
        self._turn_index: int = -1

    async def transcribe(self, turn: Turn) -> tuple[str, list[PipelineSpan]]:
        self._inner._conv_id = self._conv_id
        self._inner._turn_index = self._turn_index

        if self._mock:
            return await self._inner.transcribe(turn)

        return await self._real_transcribe(turn)

    async def _real_transcribe(self, turn: Turn) -> tuple[str, list[PipelineSpan]]:
        # Guard: real Deepgram STT requires actual audio bytes.  The mock
        # pipeline operates on text-only Turn objects, so audio_bytes is
        # normally absent.  Posting turn.text.encode() as audio/wav would
        # send invalid data to Deepgram and is never permitted.
        audio_bytes: bytes | None = getattr(turn, "audio_bytes", None)
        if audio_bytes is None:
            logger.warning(
                "DeepgramSTT real mode requires audio bytes (turn.audio_bytes), "
                "but none are present on this turn (conv_id=%r, turn_index=%r). "
                "The mock pipeline operates on text only — real-mode Deepgram requires "
                "actual audio. Falling back to mock.",
                self._conv_id,
                self._turn_index,
            )
            return await self._inner.transcribe(turn)

        if _httpx is None:  # pragma: no cover
            warnings.warn(
                "httpx is not installed; install voice-eval-lab[real] or add httpx. "
                "Falling back to mock.",
                stacklevel=2,
            )
            return await self._inner.transcribe(turn)

        t0 = time.monotonic()
        try:
            async with _httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
                resp = await client.post(
                    self._api_url,
                    params={"model": _DEEPGRAM_MODEL, "smart_format": "true"},
                    headers={
                        "Authorization": f"Token {self._api_key}",
                        "Content-Type": "audio/wav",
                    },
                    content=audio_bytes,
                )
                resp.raise_for_status()
                data = resp.json()
                alternatives = (
                    data.get("results", {})
                    .get("channels", [{}])[0]
                    .get("alternatives", [{}])
                )
                transcript: str = alternatives[0].get("transcript", turn.text) if alternatives else turn.text
                elapsed_ms = round((time.monotonic() - t0) * 1000)
        except Exception as exc:
            logger.warning(
                "DeepgramSTT real call failed (%s: %s); falling back to mock.",
                type(exc).__name__,
                exc,
            )
            return await self._inner.transcribe(turn)

        spans = [
            PipelineSpan(
                name="stt.transcribe",
                started_at_ms=turn.ended_at_ms,
                ended_at_ms=turn.ended_at_ms + elapsed_ms,
                attrs={
                    "engine": "deepgram",
                    "model": _DEEPGRAM_MODEL,
                    "wer_injected": "0.0",
                    "source": "deepgram",
                },
            )
        ]
        return transcript, spans
