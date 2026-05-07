"""Pipecat pipeline builder + in-memory test driver.

``build_pipeline`` wires the three FrameProcessor wrappers (STT, LLM, TTS)
into a Pipecat ``Pipeline``. When ``pipecat`` is not importable a thin
``Pipeline`` shim is returned that implements the same ``run`` method.

``run_pipeline`` drives the pipeline against an in-memory audio source and
collects the resulting ``Turn`` objects. It is used by the CLI ``pipeline run``
command and by the test suite — no LiveKit room required.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

from voice_eval_lab.models import Turn, TurnRole
from voice_eval_lab.pipecat.processors import (
    AudioRawFrame,
    Frame,
    LLMProcessor,
    STTProcessor,
    TextFrame,
    TTSProcessor,
)
from voice_eval_lab.pipecat.turn_detector import SmartTurnDetector
from voice_eval_lab.pipeline import LLM, STT, TTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft-import pipecat Pipeline
# ---------------------------------------------------------------------------

try:
    from pipecat.pipeline.pipeline import (
        Pipeline as _PipecatPipeline,  # type: ignore[import-untyped]
    )

    _PIPECAT_PIPELINE_AVAILABLE = True
except ImportError:
    _PIPECAT_PIPELINE_AVAILABLE = False
    _PipecatPipeline = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Shim Pipeline (used when pipecat is not installed)
# ---------------------------------------------------------------------------


class _ShimPipeline:
    """Minimal pipeline shim: chains processors in order and drives frames through.

    Mirrors just enough of the pipecat.pipeline.Pipeline surface for the
    in-memory test driver to work without the real SDK installed.
    """

    def __init__(self, processors: Sequence[Any]) -> None:
        self._processors = list(processors)
        # Wire the chain: each processor's push_frame forwards to the next.
        for i, proc in enumerate(self._processors[:-1]):
            proc._downstream = self._processors[i + 1]

    def processors(self) -> list[Any]:
        return list(self._processors)

    async def run_frame(self, frame: Frame) -> None:
        """Push a single frame through the head of the chain."""
        if self._processors:
            await self._processors[0].process_frame(frame)


# ---------------------------------------------------------------------------
# Turn-detector + barge-in stub
# ---------------------------------------------------------------------------


class _TurnDetectorStub:
    """Minimal VAD turn-detector stub (kept for backwards compatibility).

    Replaced by :class:`~voice_eval_lab.pipecat.turn_detector.SmartTurnDetector`
    as the default.  This class remains so that ``turn_detector='none'`` mode
    can instantiate it explicitly and so that existing test code that checks
    ``isinstance(stt_proc._turn_detector, _TurnDetectorStub)`` continues to
    work.
    """

    async def process(self, frame: Frame) -> Frame:
        """Return the frame unchanged (no VAD applied)."""
        return frame


class _BargeInStub:
    """Minimal barge-in / interruption handler stub.

    Production implementation would monitor incoming ``AudioRawFrame``
    activity while TTS is playing and cancel the ``TTSProcessor`` on
    energy above a threshold. Not implemented here.
    """

    def cancel_tts(self) -> None:
        """No-op — real implementation would signal TTSProcessor to stop."""


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

#: Type alias for the union of the real Pipecat Pipeline and the shim.
Pipeline = Any


def build_pipeline(
    stt: STT,
    llm: LLM,
    tts: TTS,
    *,
    turn_detector: str = "smart",
) -> Pipeline:
    """Wire STT / LLM / TTS adapters into a Pipecat-compatible pipeline.

    Returns a real ``pipecat.pipeline.Pipeline`` when the SDK is importable,
    otherwise a ``_ShimPipeline`` with the same processor chain.

    The pipeline shape is:

        AudioRawFrame → STTProcessor → TextFrame
                                     → LLMProcessor → TextFrame
                                                     → TTSProcessor → AudioRawFrame

    A turn-detector and ``_BargeInStub`` are attached to the STT processor.
    The turn-detector is a :class:`SmartTurnDetector` by default (``"smart"``
    mode) which soft-imports Pipecat's ``SmartTurnAnalyzer`` and falls back to
    an energy-based silence detector when the SDK is not installed.  Pass
    ``turn_detector="none"`` to use the no-op ``_TurnDetectorStub`` instead.

    Args:
        stt:            STT adapter.
        llm:            LLM adapter.
        tts:            TTS adapter.
        turn_detector:  ``"smart"`` (default) or ``"none"``.

    Returns:
        A pipeline object with a ``processors()`` method listing the active
        FrameProcessor instances in order.
    """
    stt_proc = STTProcessor(adapter=stt)
    llm_proc = LLMProcessor(adapter=llm)
    tts_proc = TTSProcessor(adapter=tts)

    # Wire the turn detector.
    if turn_detector == "none":
        stt_proc._turn_detector = _TurnDetectorStub()  # type: ignore[attr-defined]
    else:
        # Default: "smart" — SmartTurnDetector with energy-based fallback.
        stt_proc._turn_detector = SmartTurnDetector()  # type: ignore[attr-defined]

    stt_proc._barge_in = _BargeInStub()  # type: ignore[attr-defined]

    processors = [stt_proc, llm_proc, tts_proc]

    if _PIPECAT_PIPELINE_AVAILABLE and _PipecatPipeline is not None:
        try:
            return _PipecatPipeline(processors)
        except Exception:
            logger.warning(
                "Failed to construct pipecat.Pipeline; falling back to shim.",
                exc_info=True,
            )

    return _ShimPipeline(processors)


# ---------------------------------------------------------------------------
# In-memory pipeline driver
# ---------------------------------------------------------------------------


async def run_pipeline(
    pipeline: Pipeline,
    *,
    audio_source: Sequence[bytes] | None = None,
    audio_sink: list[bytes] | None = None,
) -> AsyncIterator[Turn]:
    """Drive a pipeline against an in-memory audio source.

    Wraps each chunk in ``audio_source`` in an ``AudioRawFrame``, pushes it
    through the pipeline, and collects ``TextFrame`` outputs from the LLM
    processor. Each LLM reply is yielded as an agent ``Turn``.

    Args:
        pipeline:     A pipeline returned by ``build_pipeline``.
        audio_source: Sequence of raw PCM byte strings to feed as input.
                      When ``None``, a single empty chunk is used.
        audio_sink:   Optional list to collect raw ``AudioRawFrame`` bytes
                      from the TTS processor (for recording / verification).

    Yields:
        ``Turn`` objects for each agent reply produced during the run.
    """
    return _run_pipeline_impl(pipeline, audio_source=audio_source, audio_sink=audio_sink)


async def _run_pipeline_impl(
    pipeline: Pipeline,
    *,
    audio_source: Sequence[bytes] | None = None,
    audio_sink: list[bytes] | None = None,
) -> AsyncIterator[Turn]:
    """Internal async generator that drives the pipeline frame by frame."""
    chunks: Sequence[bytes] = audio_source if audio_source is not None else [b""]

    # Collect output turns from the LLM processor.
    collected_turns: list[Turn] = []

    # Locate the processors in the chain.
    procs = pipeline.processors() if hasattr(pipeline, "processors") else []

    stt_proc: STTProcessor | None = next(
        (p for p in procs if isinstance(p, STTProcessor)), None
    )
    llm_proc: LLMProcessor | None = next(
        (p for p in procs if isinstance(p, LLMProcessor)), None
    )
    tts_proc: TTSProcessor | None = next(
        (p for p in procs if isinstance(p, TTSProcessor)), None
    )

    if stt_proc is None or llm_proc is None or tts_proc is None:
        logger.warning("run_pipeline: pipeline missing expected processors; yielding nothing.")
        return

    # Monkey-patch a collector onto the TTS processor's push_frame so we can
    # capture TextFrame outputs from the LLM before they hit TTS, and capture
    # AudioRawFrame outputs from TTS for the audio_sink.
    _orig_llm_push = llm_proc.push_frame
    _orig_tts_push = tts_proc.push_frame

    async def _llm_push_interceptor(frame: Frame, direction: Any = None) -> None:
        if isinstance(frame, TextFrame):
            collected_turns.append(
                Turn(
                    role=TurnRole.AGENT,
                    text=frame.text,
                    started_at_ms=0,
                    ended_at_ms=0,
                )
            )
        await _orig_llm_push(frame, direction)

    async def _tts_push_interceptor(frame: Frame, direction: Any = None) -> None:
        if audio_sink is not None and isinstance(frame, AudioRawFrame):
            audio_sink.append(frame.audio)
        # Don't forward TTS output further — no downstream processor.

    llm_proc.push_frame = _llm_push_interceptor  # type: ignore[method-assign]
    tts_proc.push_frame = _tts_push_interceptor  # type: ignore[method-assign]

    try:
        for chunk in chunks:
            audio_frame = AudioRawFrame(audio=chunk)
            if isinstance(pipeline, _ShimPipeline):
                await pipeline.run_frame(audio_frame)
            else:
                # Real pipecat Pipeline: use its own run mechanism.
                # The Pipeline.run() method expects a task runner; for the
                # in-memory driver we push directly to the first processor.
                head = procs[0] if procs else None
                if head is not None:
                    await head.process_frame(audio_frame)
    finally:
        # Restore original push_frame methods.
        llm_proc.push_frame = _orig_llm_push  # type: ignore[method-assign]
        tts_proc.push_frame = _orig_tts_push  # type: ignore[method-assign]

    for turn in collected_turns:
        yield turn


__all__ = [
    "Pipeline",
    "build_pipeline",
    "run_pipeline",
]
