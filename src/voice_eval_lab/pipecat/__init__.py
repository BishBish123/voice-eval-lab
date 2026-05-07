"""Pipecat pipeline scaffolding for voice-eval-lab.

Exports:

- ``STTProcessor``       — Pipecat FrameProcessor wrapping any ``STT`` Protocol
- ``LLMProcessor``       — Pipecat FrameProcessor wrapping any ``LLM`` Protocol
- ``TTSProcessor``       — Pipecat FrameProcessor wrapping any ``TTS`` Protocol
- ``build_pipeline``     — wire the three processors into a Pipecat Pipeline
- ``run_pipeline``       — drive the pipeline against an in-memory source (testing)
- ``serve_on_livekit``   — connect the pipeline to a LiveKit room
- ``make_pipecat_pipeline`` — convenience factory: MockSTT + MockLLM + MockTTS wired up

Without LiveKit credentials ``serve_on_livekit`` logs a warning and returns
immediately. Without ``pipecat-ai`` installed the processors fall back to
pure-Python shims that implement the same ``process_frame`` / ``push_frame``
contract.
"""

from __future__ import annotations

from voice_eval_lab.pipecat.livekit import serve_on_livekit
from voice_eval_lab.pipecat.pipeline import Pipeline, build_pipeline, run_pipeline
from voice_eval_lab.pipecat.processors import (
    AudioRawFrame,
    Frame,
    LLMProcessor,
    STTProcessor,
    TextFrame,
    TTSProcessor,
)
from voice_eval_lab.pipecat.turn_detector import SmartTurnDetector, TurnState


def make_pipecat_pipeline() -> Pipeline:
    """Return a wired Pipecat pipeline backed by the three mock adapters.

    Convenience factory for CLI smoke tests and quick local experimentation.
    The mock adapters produce deterministic latency and a canned transcript
    so the pipeline runs end-to-end without any API keys or audio devices.

    Returns:
        A ``Pipeline`` (real Pipecat or shim) with ``STTProcessor``,
        ``LLMProcessor``, and ``TTSProcessor`` wired in order.
    """
    from voice_eval_lab.pipeline import MockLLM, MockSTT, MockTTS  # noqa: PLC0415

    return build_pipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())


__all__ = [
    "AudioRawFrame",
    "Frame",
    "LLMProcessor",
    "Pipeline",
    "STTProcessor",
    "SmartTurnDetector",
    "TTSProcessor",
    "TextFrame",
    "TurnState",
    "build_pipeline",
    "make_pipecat_pipeline",
    "run_pipeline",
    "serve_on_livekit",
]
