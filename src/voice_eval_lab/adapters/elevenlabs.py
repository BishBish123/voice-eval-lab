"""ElevenLabs TTS adapter — implements the ``TTS`` Protocol.

Key:      ``ELEVENLABS_API_KEY``
Voice:    ``ELEVENLABS_VOICE_ID`` (default: ``21m00Tcm4TlvDq8ikWAM`` — Rachel)
Endpoint: ``POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}``
Model:    ``eleven_turbo_v2_5``

Auth verified against https://elevenlabs.io/docs/api-reference/text-to-speech/convert
on 2026-05-06: canonical header is ``xi-api-key: <key>`` (not Bearer).

Response: binary audio (MP3 by default; can request PCM via output_format).
The adapter requests ``pcm_16000`` to match the PCM expectation of the
``TTS`` Protocol and avoid a conversion step.

Mock path (no key / API failure):
    Delegates to ``MockTTS`` with deterministic jitter enabled.

Real path:
    Single ``httpx`` POST requesting audio/pcm output; measures wall-clock
    to first response byte as the first-byte latency. On failure logs a
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

_ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
_ELEVENLABS_MODEL_ID = "eleven_turbo_v2_5"
_ELEVENLABS_DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel — ElevenLabs' default
_TIMEOUT_S = 15.0
_ELEVENLABS_BASE_URL_ENV = "ELEVENLABS_API_BASE_URL"


class ElevenLabsTTS:
    """TTS adapter backed by ElevenLabs.  Falls back to ``MockTTS`` when the key is absent.

    Real-call shape:
        POST /v1/text-to-speech/{voice_id}?output_format=pcm_16000
        body: {"text": "...", "model_id": "eleven_turbo_v2_5",
               "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
        headers: xi-api-key: <ELEVENLABS_API_KEY>

    Auth format verified against https://elevenlabs.io/docs/api-reference/text-to-speech/convert
    on 2026-05-06: canonical header is ``xi-api-key: <key>`` (not ``Authorization: Bearer``).

    Latency is measured as wall-clock from request start to first response byte.
    The adapter is constructed without making any network call.
    """

    def __init__(
        self,
        voice_id: str | None = None,
        first_byte_ms: int = 75,
        base_url: str | None = None,
    ) -> None:
        self._api_key: str | None = os.environ.get("ELEVENLABS_API_KEY")
        self._mock: bool = self._api_key is None
        self._voice_id: str = (
            voice_id
            or os.environ.get("ELEVENLABS_VOICE_ID", "")
            or _ELEVENLABS_DEFAULT_VOICE_ID
        )
        self._inner: MockTTS = MockTTS(first_byte_ms=first_byte_ms)
        # URL override: constructor param > env var > hard-coded default.
        # When overriding, the caller supplies the full base URL without the
        # path; the voice_id path segment is appended at call time.
        self._base_url: str = (
            base_url
            or os.environ.get(_ELEVENLABS_BASE_URL_ENV, "")
            or "https://api.elevenlabs.io"
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

        url = f"{self._base_url}/v1/text-to-speech/{self._voice_id}"
        # _real_synthesize is only called when _mock is False, which means
        # _api_key is non-None. Assert here to satisfy mypy's type checker.
        assert self._api_key is not None
        t0 = time.monotonic()
        try:
            async with _httpx.AsyncClient(timeout=_TIMEOUT_S) as client, client.stream(
                "POST",
                url,
                params={"output_format": "pcm_16000"},
                headers={
                    # Canonical auth as per https://elevenlabs.io/docs/api-reference/text-to-speech/convert
                    # (verified 2026-05-06): xi-api-key header (not Authorization: Bearer).
                    "xi-api-key": self._api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": _ELEVENLABS_MODEL_ID,
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
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
                "ElevenLabsTTS real call failed (%s: %s); falling back to mock.",
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
                    "engine": "elevenlabs",
                    "model": _ELEVENLABS_MODEL_ID,
                    "voice_id": self._voice_id,
                    "chars": str(len(text)),
                    "source": "elevenlabs",
                },
            )
        ]
        return elapsed_ms, spans
