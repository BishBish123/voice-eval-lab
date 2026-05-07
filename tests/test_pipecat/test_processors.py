"""Unit tests for the Pipecat FrameProcessor wrappers.

All tests run without a real pipecat-ai install — the shim FrameProcessor
is used automatically when the SDK is absent. No LiveKit connection is made.
"""

from __future__ import annotations

import pytest

from voice_eval_lab.models import Turn, TurnRole
from voice_eval_lab.pipecat.processors import (
    AudioRawFrame,
    Frame,
    LLMProcessor,
    STTProcessor,
    TextFrame,
    TTSProcessor,
)
from voice_eval_lab.pipeline import MockLLM, MockSTT, MockTTS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CollectingProcessor:
    """Minimal downstream that records every frame pushed to it."""

    def __init__(self) -> None:
        self.frames: list[Frame] = []
        self._downstream = None

    async def process_frame(self, frame: Frame, direction: object = None) -> None:
        self.frames.append(frame)

    async def push_frame(self, frame: Frame, direction: object = None) -> None:
        self.frames.append(frame)


def _wire(upstream: object, downstream: _CollectingProcessor) -> None:
    """Connect upstream._downstream = downstream."""
    upstream._downstream = downstream  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# STTProcessor
# ---------------------------------------------------------------------------


class TestSTTProcessor:
    """AudioRawFrame in → TextFrame out (via mock STT)."""

    @pytest.mark.asyncio
    async def test_audio_raw_frame_produces_text_frame(self) -> None:
        mock_stt = MockSTT()
        # Give the mock STT a canned turn so it has text to return.
        # MockSTT.transcribe uses turn.text as the gold transcript.
        proc = STTProcessor(adapter=mock_stt)

        # Patch _frame_to_turn to return a turn with known text.
        def _fake_turn(frame: AudioRawFrame) -> Turn:
            return Turn(
                role=TurnRole.USER,
                text="hello world",
                started_at_ms=0,
                ended_at_ms=100,
            )

        proc._frame_to_turn = _fake_turn  # type: ignore[method-assign]

        collector = _CollectingProcessor()
        _wire(proc, collector)

        audio = AudioRawFrame(audio=b"\x00" * 320)
        await proc.process_frame(audio)

        assert len(collector.frames) == 1
        frame = collector.frames[0]
        assert isinstance(frame, TextFrame)
        assert frame.text == "hello world"

    @pytest.mark.asyncio
    async def test_non_audio_frame_forwarded_unchanged(self) -> None:
        proc = STTProcessor(adapter=MockSTT())
        collector = _CollectingProcessor()
        _wire(proc, collector)

        text_frame = TextFrame(text="pass-through")
        await proc.process_frame(text_frame)

        assert len(collector.frames) == 1
        assert collector.frames[0] is text_frame

    @pytest.mark.asyncio
    async def test_empty_transcript_not_forwarded(self) -> None:
        """When STT returns empty string no TextFrame should be emitted."""

        class _SilentSTT:
            async def transcribe(self, turn: Turn) -> tuple[str, list]:
                return "", []

        proc = STTProcessor(adapter=_SilentSTT())
        collector = _CollectingProcessor()
        _wire(proc, collector)

        await proc.process_frame(AudioRawFrame(audio=b""))
        assert collector.frames == []

    @pytest.mark.asyncio
    async def test_stt_exception_does_not_propagate(self) -> None:
        """A crashing STT adapter should not raise; an empty transcript is emitted silently."""

        class _CrashSTT:
            async def transcribe(self, turn: Turn) -> tuple[str, list]:
                raise RuntimeError("network error")

        proc = STTProcessor(adapter=_CrashSTT())
        collector = _CollectingProcessor()
        _wire(proc, collector)

        # Should not raise.
        await proc.process_frame(AudioRawFrame(audio=b"x"))
        # Empty transcript → no frame forwarded.
        assert collector.frames == []

    @pytest.mark.asyncio
    async def test_wer_injection_propagates(self) -> None:
        """MockSTT with wer_substitution_rate=1.0 should substitute all words."""
        mock_stt = MockSTT(wer_substitution_rate=1.0)
        proc = STTProcessor(adapter=mock_stt)

        def _fake_turn(frame: AudioRawFrame) -> Turn:
            return Turn(
                role=TurnRole.USER,
                text="the quick brown fox",
                started_at_ms=0,
                ended_at_ms=100,
            )

        proc._frame_to_turn = _fake_turn  # type: ignore[method-assign]
        collector = _CollectingProcessor()
        _wire(proc, collector)

        await proc.process_frame(AudioRawFrame(audio=b""))
        assert len(collector.frames) == 1
        assert isinstance(collector.frames[0], TextFrame)
        # All four words substituted with "WERR".
        assert collector.frames[0].text == "WERR WERR WERR WERR"


# ---------------------------------------------------------------------------
# LLMProcessor
# ---------------------------------------------------------------------------


class TestLLMProcessor:
    """TextFrame (user) in → TextFrame (agent reply) out."""

    @pytest.mark.asyncio
    async def test_text_frame_produces_reply_text_frame(self) -> None:
        proc = LLMProcessor(adapter=MockLLM(), gold_facts=["hnsw ef_search parameter"])
        collector = _CollectingProcessor()
        _wire(proc, collector)

        user_frame = TextFrame(text="hnsw ef_search parameter")
        await proc.process_frame(user_frame)

        assert len(collector.frames) == 1
        reply = collector.frames[0]
        assert isinstance(reply, TextFrame)
        assert len(reply.text) > 0

    @pytest.mark.asyncio
    async def test_non_text_frame_forwarded_unchanged(self) -> None:
        proc = LLMProcessor(adapter=MockLLM())
        collector = _CollectingProcessor()
        _wire(proc, collector)

        audio = AudioRawFrame(audio=b"\x00")
        await proc.process_frame(audio)

        assert len(collector.frames) == 1
        assert collector.frames[0] is audio

    @pytest.mark.asyncio
    async def test_history_grows_per_turn(self) -> None:
        """Each call should append a user + agent turn to the internal history."""
        proc = LLMProcessor(adapter=MockLLM())
        collector = _CollectingProcessor()
        _wire(proc, collector)

        assert len(proc._history) == 0
        await proc.process_frame(TextFrame(text="turn one"))
        assert len(proc._history) == 2  # user + agent
        await proc.process_frame(TextFrame(text="turn two"))
        assert len(proc._history) == 4

    @pytest.mark.asyncio
    async def test_llm_exception_does_not_propagate(self) -> None:
        class _CrashLLM:
            async def reply(
                self, history: list, last_user_text: str, gold_facts: list
            ) -> tuple[str, list]:
                raise RuntimeError("model overloaded")

        proc = LLMProcessor(adapter=_CrashLLM())
        collector = _CollectingProcessor()
        _wire(proc, collector)

        await proc.process_frame(TextFrame(text="hello"))
        # Empty reply → no frame forwarded.
        assert collector.frames == []

    @pytest.mark.asyncio
    async def test_gold_facts_passed_to_adapter(self) -> None:
        """gold_facts set on the processor should reach the LLM adapter."""
        received_facts: list[list[str]] = []

        class _SpyLLM:
            async def reply(
                self, history: list, last_user_text: str, gold_facts: list
            ) -> tuple[str, list]:
                received_facts.append(list(gold_facts))
                return "ok", []

        proc = LLMProcessor(adapter=_SpyLLM(), gold_facts=["fact-a", "fact-b"])
        collector = _CollectingProcessor()
        _wire(proc, collector)

        await proc.process_frame(TextFrame(text="question"))
        assert received_facts == [["fact-a", "fact-b"]]


# ---------------------------------------------------------------------------
# TTSProcessor
# ---------------------------------------------------------------------------


class TestTTSProcessor:
    """TextFrame (agent) in → AudioRawFrame chunks out."""

    @pytest.mark.asyncio
    async def test_text_frame_produces_audio_frames(self) -> None:
        proc = TTSProcessor(adapter=MockTTS())
        collector = _CollectingProcessor()
        _wire(proc, collector)

        await proc.process_frame(TextFrame(text="hello"))

        assert len(collector.frames) >= 1
        for frame in collector.frames:
            assert isinstance(frame, AudioRawFrame)

    @pytest.mark.asyncio
    async def test_audio_chunks_contain_bytes(self) -> None:
        proc = TTSProcessor(adapter=MockTTS(), chunk_bytes=320)
        collector = _CollectingProcessor()
        _wire(proc, collector)

        await proc.process_frame(TextFrame(text="hello"))

        for frame in collector.frames:
            assert isinstance(frame.audio, bytes)
            assert len(frame.audio) > 0

    @pytest.mark.asyncio
    async def test_non_text_frame_forwarded_unchanged(self) -> None:
        proc = TTSProcessor(adapter=MockTTS())
        collector = _CollectingProcessor()
        _wire(proc, collector)

        audio_in = AudioRawFrame(audio=b"\x01\x02")
        await proc.process_frame(audio_in)

        assert len(collector.frames) == 1
        assert collector.frames[0] is audio_in

    @pytest.mark.asyncio
    async def test_tts_exception_forwards_original_frame(self) -> None:
        class _CrashTTS:
            async def synthesize(self, text: str) -> tuple[int, list]:
                raise RuntimeError("synthesis failed")

        proc = TTSProcessor(adapter=_CrashTTS())
        collector = _CollectingProcessor()
        _wire(proc, collector)

        text_frame = TextFrame(text="hello")
        await proc.process_frame(text_frame)

        # On TTS failure the original TextFrame is forwarded rather than crashing.
        assert len(collector.frames) == 1
        assert isinstance(collector.frames[0], TextFrame)

    @pytest.mark.asyncio
    async def test_chunk_size_controls_frame_count(self) -> None:
        """Smaller chunk_bytes → more AudioRawFrame chunks emitted."""
        proc_small = TTSProcessor(adapter=MockTTS(first_byte_ms=100), chunk_bytes=32)
        proc_large = TTSProcessor(adapter=MockTTS(first_byte_ms=100), chunk_bytes=3200)

        col_small = _CollectingProcessor()
        col_large = _CollectingProcessor()
        _wire(proc_small, col_small)
        _wire(proc_large, col_large)

        await proc_small.process_frame(TextFrame(text="hi"))
        await proc_large.process_frame(TextFrame(text="hi"))

        assert len(col_small.frames) >= len(col_large.frames)
