"""Pipecat FrameProcessor wrappers for the STT / LLM / TTS Protocol adapters.

Each processor subclasses Pipecat's ``FrameProcessor`` when the ``pipecat``
package is importable, and falls back to a pure-Python shim that mirrors the
same ``process_frame`` contract when it is not (CI / base-install).

Frame translation:

    STTProcessor  : AudioRawFrame  → calls adapter.transcribe() → TextFrame (user)
    LLMProcessor  : TextFrame user → calls adapter.reply()      → TextFrame (agent)
    TTSProcessor  : TextFrame agent→ calls adapter.synthesize() → AudioRawFrame chunks

Frames that a processor does not own are forwarded downstream unchanged via
``await self.push_frame(frame)``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from voice_eval_lab.models import Turn, TurnRole
from voice_eval_lab.pipeline import LLM, STT, TTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft-import pipecat — processors work as stubs when the package is absent
# ---------------------------------------------------------------------------

try:
    from pipecat.frames.frames import (  # type: ignore[import-untyped]
        AudioRawFrame,
        Frame,
        TextFrame,
    )
    from pipecat.processors.frame_processor import (  # type: ignore[import-untyped]
        FrameDirection,
        FrameProcessor,
    )

    _PIPECAT_AVAILABLE = True
except ImportError:  # pragma: no cover — pipecat present in real env
    _PIPECAT_AVAILABLE = False

    # ---- minimal shims so the module is importable without pipecat ----

    class Frame:  # type: ignore[no-redef]
        """Minimal Frame shim used when pipecat is not installed."""

    class TextFrame(Frame):  # type: ignore[misc,no-redef]
        """Shim: wraps a text string."""

        def __init__(self, text: str) -> None:
            self.text = text

    class AudioRawFrame(Frame):  # type: ignore[misc,no-redef]
        """Shim: wraps raw PCM bytes."""

        def __init__(
            self,
            audio: bytes,
            sample_rate: int = 16000,
            num_channels: int = 1,
        ) -> None:
            self.audio = audio
            self.sample_rate = sample_rate
            self.num_channels = num_channels

    class FrameDirection:  # type: ignore[no-redef]
        DOWNSTREAM = "downstream"
        UPSTREAM = "upstream"

    class FrameProcessor:  # type: ignore[no-redef]
        """Minimal FrameProcessor shim used when pipecat is not installed."""

        def __init__(self) -> None:
            self._downstream: FrameProcessor | None = None

        async def push_frame(
            self,
            frame: Frame,
            direction: Any = None,
        ) -> None:
            """Forward frame to the next processor in the chain."""
            if self._downstream is not None:
                await self._downstream.process_frame(frame, direction)

        async def process_frame(self, frame: Frame, direction: Any = None) -> None:
            """Override in subclasses; default is a no-op passthrough."""
            await self.push_frame(frame, direction)


if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# STTProcessor
# ---------------------------------------------------------------------------


class STTProcessor(FrameProcessor):  # type: ignore[misc]
    """Wraps any ``STT`` Protocol adapter as a Pipecat FrameProcessor.

    Consumes ``AudioRawFrame`` and emits ``TextFrame`` (user transcript).
    All other frames are forwarded downstream unchanged.

    The ``Turn`` passed to the adapter is synthesised from the audio frame;
    ``started_at_ms`` and ``ended_at_ms`` default to 0 when the frame carries
    no timing metadata (the shim path). Real LiveKit frames can carry a
    timestamp via the ``AudioRawFrame`` subclass; override ``_frame_to_turn``
    to extract it.
    """

    def __init__(self, adapter: STT) -> None:
        super().__init__()
        self._adapter = adapter
        self._history: list[Turn] = []

    def _frame_to_turn(self, frame: AudioRawFrame) -> Turn:
        """Build a minimal Turn from an audio frame.

        Override in subclasses to propagate real timing metadata
        (e.g. from a LiveKit AudioRawFrame timestamp field).
        """
        return Turn(
            role=TurnRole.USER,
            text="",  # STT adapter will transcribe from audio_bytes
            started_at_ms=0,
            ended_at_ms=0,
            # Attach raw audio bytes so real STT adapters can use them.
            # The mock adapter ignores this field; real adapters (DeepgramSTT)
            # read it via getattr(turn, "audio_bytes", None).
        )

    async def process_frame(self, frame: Frame, direction: Any = None) -> None:
        if isinstance(frame, AudioRawFrame):
            turn = self._frame_to_turn(frame)
            # Attach audio bytes so real adapters (DeepgramSTT) can use them.
            object.__setattr__(turn, "audio_bytes", getattr(frame, "audio", b""))
            try:
                text, _spans = await self._adapter.transcribe(turn)
            except Exception:
                logger.exception("STTProcessor: transcribe failed; forwarding empty text")
                text = ""
            if text:
                await self.push_frame(TextFrame(text=text), direction)
        else:
            await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# LLMProcessor
# ---------------------------------------------------------------------------


class LLMProcessor(FrameProcessor):  # type: ignore[misc]
    """Wraps any ``LLM`` Protocol adapter as a Pipecat FrameProcessor.

    Consumes ``TextFrame`` (user transcript) and emits ``TextFrame``
    (agent reply). All other frames are forwarded downstream unchanged.

    ``gold_facts`` can be set after construction to pass facts to the
    adapter on each call (useful for testing and for the in-memory
    eval harness).
    """

    def __init__(self, adapter: LLM, gold_facts: list[str] | None = None) -> None:
        super().__init__()
        self._adapter = adapter
        self._history: list[Turn] = []
        self.gold_facts: list[str] = gold_facts or []

    async def process_frame(self, frame: Frame, direction: Any = None) -> None:
        if isinstance(frame, TextFrame):
            user_text = frame.text
            try:
                reply_text, _spans = await self._adapter.reply(
                    self._history,
                    user_text,
                    self.gold_facts,
                )
            except Exception:
                logger.exception("LLMProcessor: reply failed")
                reply_text = ""
            # Record both sides so subsequent turns have full history.
            self._history.append(
                Turn(role=TurnRole.USER, text=user_text, started_at_ms=0, ended_at_ms=0)
            )
            self._history.append(
                Turn(role=TurnRole.AGENT, text=reply_text, started_at_ms=0, ended_at_ms=0)
            )
            if reply_text:
                await self.push_frame(TextFrame(text=reply_text), direction)
        else:
            await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# TTSProcessor
# ---------------------------------------------------------------------------

# Default PCM chunk size — 20 ms of 16-kHz mono 16-bit audio.
_DEFAULT_CHUNK_BYTES = 640  # 16000 Hz * 0.020 s * 2 bytes/sample = 640


class TTSProcessor(FrameProcessor):  # type: ignore[misc]
    """Wraps any ``TTS`` Protocol adapter as a Pipecat FrameProcessor.

    Consumes ``TextFrame`` (agent reply) and emits one or more
    ``AudioRawFrame`` chunks. All other frames are forwarded downstream.

    The mock ``TTS`` Protocol returns ``(first_byte_ms, spans)``; there
    are no actual audio bytes. ``TTSProcessor`` synthesises a silent PCM
    buffer whose length corresponds to ``first_byte_ms`` of 16-kHz mono
    audio, split into ``chunk_bytes``-sized ``AudioRawFrame`` chunks. Real
    adapters (CartesiaTTS) should be extended to return raw PCM; the
    ``chunk_bytes`` parameter controls the streaming granularity.
    """

    def __init__(
        self,
        adapter: TTS,
        sample_rate: int = 16000,
        num_channels: int = 1,
        chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._chunk_bytes = chunk_bytes

    async def process_frame(self, frame: Frame, direction: Any = None) -> None:
        if isinstance(frame, TextFrame):
            text = frame.text
            try:
                first_byte_ms, _spans = await self._adapter.synthesize(text)
            except Exception:
                logger.exception("TTSProcessor: synthesize failed")
                await self.push_frame(frame, direction)
                return
            # Produce a silent PCM buffer whose length maps to first_byte_ms.
            # bytes_per_ms = sample_rate * channels * 2 (16-bit) / 1000
            bytes_per_ms = (self._sample_rate * self._num_channels * 2) // 1000
            total_bytes = max(self._chunk_bytes, first_byte_ms * bytes_per_ms)
            audio_buf = bytes(total_bytes)
            # Emit in chunk_bytes slices to simulate streaming TTS.
            for start in range(0, total_bytes, self._chunk_bytes):
                chunk = audio_buf[start : start + self._chunk_bytes]
                await self.push_frame(
                    AudioRawFrame(
                        audio=chunk,
                        sample_rate=self._sample_rate,
                        num_channels=self._num_channels,
                    ),
                    direction,
                )
        else:
            await self.push_frame(frame, direction)


__all__ = [
    "AudioRawFrame",
    "Frame",
    "FrameProcessor",
    "LLMProcessor",
    "STTProcessor",
    "TTSProcessor",
    "TextFrame",
]
