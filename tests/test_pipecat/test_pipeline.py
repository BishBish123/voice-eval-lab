"""Unit tests for build_pipeline and run_pipeline (in-memory driver).

No LiveKit connection. No real pipecat-ai required — the shim is used when
the SDK is not installed.
"""

from __future__ import annotations

import pytest

from voice_eval_lab.models import TurnRole
from voice_eval_lab.pipecat import make_pipecat_pipeline
from voice_eval_lab.pipecat.pipeline import (
    _ShimPipeline,
    build_pipeline,
    run_pipeline,
)
from voice_eval_lab.pipecat.processors import LLMProcessor, STTProcessor, TTSProcessor
from voice_eval_lab.pipeline import MockLLM, MockSTT, MockTTS


class TestBuildPipeline:
    def test_returns_pipeline_with_three_processors(self) -> None:
        pipeline = build_pipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())
        assert hasattr(pipeline, "processors")
        procs = pipeline.processors()
        assert len(procs) == 3

    def test_processor_types_in_order(self) -> None:
        pipeline = build_pipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())
        procs = pipeline.processors()
        assert isinstance(procs[0], STTProcessor)
        assert isinstance(procs[1], LLMProcessor)
        assert isinstance(procs[2], TTSProcessor)

    def test_shim_pipeline_returned_when_pipecat_absent(self) -> None:
        """When the pipecat SDK is not installed, a _ShimPipeline is used."""
        from voice_eval_lab.pipecat import pipeline as _pm

        original = _pm._PIPECAT_PIPELINE_AVAILABLE
        try:
            _pm._PIPECAT_PIPELINE_AVAILABLE = False
            result = build_pipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())
            assert isinstance(result, _ShimPipeline)
        finally:
            _pm._PIPECAT_PIPELINE_AVAILABLE = original

    def test_turn_detector_and_barge_in_attached(self) -> None:
        pipeline = build_pipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())
        stt_proc = pipeline.processors()[0]
        assert hasattr(stt_proc, "_turn_detector")
        assert hasattr(stt_proc, "_barge_in")

    def test_make_pipecat_pipeline_factory(self) -> None:
        pipeline = make_pipecat_pipeline()
        assert hasattr(pipeline, "processors")
        procs = pipeline.processors()
        assert len(procs) == 3


class TestRunPipeline:
    @pytest.mark.asyncio
    async def test_run_with_mock_adapters_yields_turns(self) -> None:
        """run_pipeline with mock adapters should yield at least one agent Turn."""
        pipeline = build_pipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())

        # Patch STTProcessor._frame_to_turn to return a turn with real text.
        from voice_eval_lab.models import Turn
        from voice_eval_lab.pipecat.processors import AudioRawFrame

        stt_proc = pipeline.processors()[0]

        def _fake_turn(frame: AudioRawFrame) -> Turn:
            return Turn(
                role=TurnRole.USER,
                text="hnsw ef_search parameter",
                started_at_ms=0,
                ended_at_ms=100,
            )

        stt_proc._frame_to_turn = _fake_turn  # type: ignore[method-assign]

        turns = []
        async for turn in await run_pipeline(
            pipeline, audio_source=[b"\x00" * 320]
        ):
            turns.append(turn)

        assert len(turns) >= 1
        assert all(t.role == TurnRole.AGENT for t in turns)

    @pytest.mark.asyncio
    async def test_audio_sink_collects_bytes(self) -> None:
        """Audio sink should receive PCM bytes from TTSProcessor."""
        pipeline = build_pipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())

        stt_proc = pipeline.processors()[0]
        from voice_eval_lab.models import Turn
        from voice_eval_lab.pipecat.processors import AudioRawFrame

        def _fake_turn(frame: AudioRawFrame) -> Turn:
            return Turn(
                role=TurnRole.USER,
                text="hello",
                started_at_ms=0,
                ended_at_ms=100,
            )

        stt_proc._frame_to_turn = _fake_turn  # type: ignore[method-assign]

        sink: list[bytes] = []
        async for _ in await run_pipeline(
            pipeline, audio_source=[b"\x00"], audio_sink=sink
        ):
            pass

        assert len(sink) >= 1
        assert all(isinstance(b, bytes) for b in sink)

    @pytest.mark.asyncio
    async def test_empty_audio_source_uses_single_empty_chunk(self) -> None:
        """Passing audio_source=None should still drive the pipeline once."""
        pipeline = build_pipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())

        stt_proc = pipeline.processors()[0]
        from voice_eval_lab.models import Turn
        from voice_eval_lab.pipecat.processors import AudioRawFrame

        def _fake_turn(frame: AudioRawFrame) -> Turn:
            return Turn(
                role=TurnRole.USER,
                text="test utterance",
                started_at_ms=0,
                ended_at_ms=50,
            )

        stt_proc._frame_to_turn = _fake_turn  # type: ignore[method-assign]

        turns = []
        async for turn in await run_pipeline(pipeline, audio_source=None):
            turns.append(turn)

        # One audio chunk → one STT call → one LLM reply → one agent turn.
        assert len(turns) == 1

    @pytest.mark.asyncio
    async def test_multiple_audio_chunks_produce_multiple_turns(self) -> None:
        pipeline = build_pipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())

        stt_proc = pipeline.processors()[0]
        from voice_eval_lab.models import Turn
        from voice_eval_lab.pipecat.processors import AudioRawFrame

        call_count = 0

        def _fake_turn(frame: AudioRawFrame) -> Turn:
            nonlocal call_count
            call_count += 1
            return Turn(
                role=TurnRole.USER,
                text=f"utterance {call_count}",
                started_at_ms=0,
                ended_at_ms=100,
            )

        stt_proc._frame_to_turn = _fake_turn  # type: ignore[method-assign]

        turns = []
        async for turn in await run_pipeline(
            pipeline, audio_source=[b"a", b"b", b"c"]
        ):
            turns.append(turn)

        assert len(turns) == 3
