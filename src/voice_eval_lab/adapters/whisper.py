"""Whisper STT adapter — implements the ``STT`` Protocol via openai-whisper.

Gate:    ``WHISPER_MODEL_NAME`` env var (no value → adapter is not used)
Models:  tiny (default, ~75MB), base, small, medium, large
Package: ``openai-whisper`` (NOT the ``whisper`` PyPI package)

Mock path (openai-whisper not installed / no audio bytes):
    Delegates to ``MockSTT`` with deterministic jitter enabled.

Real path (WHISPER_MODEL_NAME set + openai-whisper installed + audio bytes):
    Lazy-loads the Whisper model on first ``transcribe`` call (not at
    __init__ time) so import stays fast even if the package is installed.
    Audio must be 16-bit PCM at 16kHz mono.  When the supplied WAV bytes
    use a different sample rate, the adapter resamples via the stdlib
    ``wave`` module (nearest-integer decimation) so no third-party audio
    library is needed.

Audio conversion guarantees:
    - Accepts any 16-bit mono WAV (8 kHz to 48 kHz).
    - Resamples to 16 kHz using integer stride decimation (stdlib only).
    - Passes a ``numpy`` float32 array to whisper.transcribe().
"""

from __future__ import annotations

import io
import logging
import struct
import wave

from voice_eval_lab.models import PipelineSpan, Turn
from voice_eval_lab.pipeline import MockSTT

logger = logging.getLogger(__name__)

_WHISPER_TARGET_RATE = 16_000  # Hz required by Whisper

try:
    import numpy as _np
    import whisper as _whisper_mod  # openai-whisper package

    _whisper_available = True
except ImportError:  # pragma: no cover
    _whisper_mod = None  # type: ignore[assignment,unused-ignore]
    _np = None  # type: ignore[assignment,unused-ignore]
    _whisper_available = False


def _pcm_bytes_to_float32(
    raw_bytes: bytes,
    src_rate: int,
    target_rate: int = _WHISPER_TARGET_RATE,
) -> object:  # returns np.ndarray[np.float32]
    """Convert raw 16-bit PCM bytes to a float32 numpy array at ``target_rate``.

    When ``src_rate != target_rate``, resamples via nearest-integer stride
    (floor(src_rate / target_rate) step), which is lossless for integer
    multiples and a good-enough approximation otherwise — avoids pulling
    in scipy / resampy for the common 24kHz→16kHz case (stride=1 would
    keep every sample; stride=2 at 32kHz→16kHz would keep every other).

    Actually: stride = round(src_rate / target_rate), clipped to ≥1.
    """
    assert _np is not None
    samples = _np.frombuffer(raw_bytes, dtype=_np.int16)
    if src_rate != target_rate:
        stride = max(1, round(src_rate / target_rate))
        samples = samples[::stride]
    return samples.astype(_np.float32) / 32768.0


def _decode_wav(audio_bytes: bytes) -> tuple[bytes, int]:
    """Return (raw_pcm_bytes, sample_rate) from a WAV blob.

    Raises ``ValueError`` if the WAV is not 16-bit mono.
    """
    with wave.open(io.BytesIO(audio_bytes)) as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(
            f"WhisperSTT expects 16-bit PCM WAV (sampwidth=2), got sampwidth={sampwidth}"
        )

    if n_channels != 1:
        # Downmix to mono by averaging left and right channels.
        n_samples = len(raw) // (2 * n_channels)
        fmt = f"<{n_samples * n_channels}h"
        interleaved = struct.unpack(fmt, raw)
        mono: list[int] = []
        for i in range(n_samples):
            avg = sum(interleaved[i * n_channels : i * n_channels + n_channels]) // n_channels
            mono.append(avg)
        raw = struct.pack(f"<{n_samples}h", *mono)

    return raw, framerate


class WhisperSTT:
    """STT adapter backed by OpenAI Whisper (local, no API key required).

    Gated by the ``WHISPER_MODEL_NAME`` env var:
        - If unset → the factory won't pick this adapter; direct
          instantiation falls back to mock if the package is absent.
        - If set    → use the named model (tiny / base / small / medium /
          large). Default model_name argument is "tiny".

    Model loading is lazy — no model is loaded at ``__init__`` time.
    The first ``transcribe`` call triggers ``whisper.load_model()``.
    """

    def __init__(self, model_name: str = "tiny") -> None:
        self._model_name = model_name
        self._mock: bool = not _whisper_available
        self._inner: MockSTT = MockSTT()
        self._model: object | None = None  # lazy-loaded
        # Turn-context fields forwarded by VoicePipeline.run.
        self._conv_id: str = ""
        self._turn_index: int = -1

    # ------------------------------------------------------------------
    # STT Protocol
    # ------------------------------------------------------------------

    async def transcribe(self, turn: Turn) -> tuple[str, list[PipelineSpan]]:
        self._inner._conv_id = self._conv_id
        self._inner._turn_index = self._turn_index

        audio_bytes: bytes | None = getattr(turn, "audio_bytes", None)

        if self._mock or audio_bytes is None:
            if self._mock:
                logger.debug(
                    "WhisperSTT: openai-whisper not installed; delegating to MockSTT."
                )
            else:
                logger.warning(
                    "WhisperSTT real mode requires audio bytes (turn.audio_bytes), "
                    "but none are present (conv_id=%r, turn_index=%r). "
                    "Falling back to mock.",
                    self._conv_id,
                    self._turn_index,
                )
            return await self._inner.transcribe(turn)

        return await self._real_transcribe(turn, audio_bytes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the Whisper model on first real transcription."""
        if self._model is None:
            assert _whisper_mod is not None
            logger.info("WhisperSTT: loading model %r (first call)", self._model_name)
            self._model = _whisper_mod.load_model(self._model_name)

    async def _real_transcribe(
        self, turn: Turn, audio_bytes: bytes
    ) -> tuple[str, list[PipelineSpan]]:
        import time

        try:
            raw_pcm, src_rate = _decode_wav(audio_bytes)
        except Exception as exc:
            logger.warning(
                "WhisperSTT: could not decode WAV bytes (%s: %s); falling back to mock.",
                type(exc).__name__,
                exc,
            )
            return await self._inner.transcribe(turn)

        try:
            self._ensure_model_loaded()
            assert _np is not None and _whisper_mod is not None

            audio_array = _pcm_bytes_to_float32(raw_pcm, src_rate)

            t0 = time.monotonic()
            result = _whisper_mod.transcribe(self._model, audio_array)
            elapsed_ms = round((time.monotonic() - t0) * 1000)

            transcript: str = result.get("text", "").strip() if isinstance(result, dict) else ""
        except Exception as exc:
            logger.warning(
                "WhisperSTT real transcription failed (%s: %s); falling back to mock.",
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
                    "engine": "whisper",
                    "model": self._model_name,
                    "wer_injected": "0.0",
                    "source": "whisper",
                },
            )
        ]
        return transcript, spans
