"""Cartesia TTS adapter — implements the ``TTS`` Protocol.

Key:     ``CARTESIA_API_KEY``
Endpoint: POST https://api.cartesia.ai/tts/bytes
Model:   ``sonic-2`` voice ``a0e99841-438c-4a64-b679-ae501e7d6091`` (English)

Mock path (no key / API failure):
    Delegates to ``MockTTS`` with deterministic jitter enabled.

Real path:
    Single ``httpx`` POST requesting audio/pcm output; measures wall-clock
    to first response byte as the first-byte latency.  On failure logs a
    warning and falls back to mock output.
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

from voice_eval_lab.models import PipelineSpan
from voice_eval_lab.pipeline import MockTTS

logger = logging.getLogger(__name__)

_CARTESIA_API_URL = "https://api.cartesia.ai/tts/bytes"
_CARTESIA_MODEL_ID = "sonic-2"
_CARTESIA_VOICE_ID = "a0e99841-438c-4a64-b679-ae501e7d6091"
_CARTESIA_API_VERSION = "2024-06-10"
_TIMEOUT_S = 15.0
_CARTESIA_BASE_URL_ENV = "CARTESIA_API_BASE_URL"


class CartesiaTTS:
    """TTS adapter backed by Cartesia Sonic.  Falls back to ``MockTTS`` when the key is absent.

    Real-call shape:
        POST /tts/bytes
        body: {"model_id": "sonic-2", "transcript": "...",
               "voice": {"mode": "id", "id": "<voice_id>"},
               "output_format": {"container": "raw", "encoding": "pcm_f32le", "sample_rate": 44100}}
        headers: Authorization: Bearer <CARTESIA_API_KEY>
                 Cartesia-Version: 2024-06-10

    Auth format verified against https://docs.cartesia.ai/api-reference/tts/bytes
    on 2026-05-06: canonical header is ``Authorization: Bearer <key>``.
    The older ``X-API-Key`` form is not documented as supported.

    Latency is measured as wall-clock from request start to first response byte.
    The adapter is constructed without making any network call.
    """

    def __init__(self, first_byte_ms: int = 75, base_url: str | None = None) -> None:
        self._api_key: str | None = os.environ.get("CARTESIA_API_KEY")
        self._mock: bool = self._api_key is None
        self._inner: MockTTS = MockTTS(first_byte_ms=first_byte_ms)
        # URL override: constructor param > env var > hard-coded default.
        self._api_url: str = (
            base_url
            or os.environ.get(_CARTESIA_BASE_URL_ENV, "")
            or _CARTESIA_API_URL
        )
        # Turn-context fields for deterministic jitter on the mock path.
        self._conv_id: str = ""
        self._turn_index: int = -1

    async def synthesize(self, text: str) -> tuple[int, list[PipelineSpan]]:
        self._inner._conv_id = self._conv_id
        self._inner._turn_index = self._turn_index

        if self._mock:
            return await self._inner.synthesize(text)

        return await self._real_synthesize(text)

    async def _real_synthesize(self, text: str) -> tuple[int, list[PipelineSpan]]:
        if _httpx is None:  # pragma: no cover
            warnings.warn(
                "httpx is not installed; install voice-eval-lab[real] or add httpx. "
                "Falling back to mock.",
                stacklevel=2,
            )
            return await self._inner.synthesize(text)

        t0 = time.monotonic()
        try:
            async with _httpx.AsyncClient(timeout=_TIMEOUT_S) as client, client.stream(
                "POST",
                self._api_url,
                headers={
                    # Canonical auth as per https://docs.cartesia.ai/api-reference/tts/bytes
                    # (verified 2026-05-06): Authorization: Bearer <key>.
                    "Authorization": f"Bearer {self._api_key}",
                    "Cartesia-Version": _CARTESIA_API_VERSION,
                    "Content-Type": "application/json",
                },
                json={
                    "model_id": _CARTESIA_MODEL_ID,
                    "transcript": text,
                    "voice": {"mode": "id", "id": _CARTESIA_VOICE_ID},
                    "output_format": {
                        "container": "raw",
                        "encoding": "pcm_f32le",
                        "sample_rate": 44100,
                    },
                },
            ) as resp:
                resp.raise_for_status()
                # Consume first chunk to measure first-byte latency.
                async for _ in resp.aiter_bytes(chunk_size=256):
                    break
                elapsed_ms = round((time.monotonic() - t0) * 1000)
        except Exception as exc:
            logger.warning(
                "CartesiaTTS real call failed (%s: %s); falling back to mock.",
                type(exc).__name__,
                exc,
            )
            return await self._inner.synthesize(text)

        spans = [
            PipelineSpan(
                name="tts.synthesize",
                started_at_ms=0,
                ended_at_ms=elapsed_ms,
                attrs={
                    "engine": "cartesia",
                    "model": _CARTESIA_MODEL_ID,
                    "chars": str(len(text)),
                    "source": "cartesia",
                },
            )
        ]
        return elapsed_ms, spans
