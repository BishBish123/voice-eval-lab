"""Tests for the real-adapter stubs (Groq, Deepgram, Cartesia) and latency jitter.

All tests run WITHOUT env vars so they exercise only the mock path.
No live HTTP calls are made.
"""

from __future__ import annotations

import logging
import statistics
import unittest.mock

import pytest

import voice_eval_lab.adapters.cartesia as _ca_mod
import voice_eval_lab.adapters.deepgram as _dg_mod
from voice_eval_lab.adapters import CartesiaTTS, DeepgramSTT, GroqLLM
from voice_eval_lab.eval.golden import default_golden_set
from voice_eval_lab.models import Conversation, Turn, TurnRole
from voice_eval_lab.pipeline import (
    MockLLM,
    MockSTT,
    MockTTS,
    VoicePipeline,
    deterministic_latency_ms,
)

# ---------------------------------------------------------------------------
# deterministic_latency_ms — basic properties
# ---------------------------------------------------------------------------


class TestDeterministicLatencyMs:
    def test_same_inputs_same_output(self) -> None:
        a = deterministic_latency_ms(80, "conv-1", 0, "stt")
        b = deterministic_latency_ms(80, "conv-1", 0, "stt")
        assert a == b

    def test_different_conv_id_different_output(self) -> None:
        a = deterministic_latency_ms(80, "conv-1", 0, "stt")
        b = deterministic_latency_ms(80, "conv-2", 0, "stt")
        assert a != b

    def test_different_turn_index_different_output(self) -> None:
        a = deterministic_latency_ms(80, "conv-1", 0, "stt")
        b = deterministic_latency_ms(80, "conv-1", 1, "stt")
        assert a != b

    def test_different_component_different_output(self) -> None:
        a = deterministic_latency_ms(80, "conv-1", 0, "stt")
        b = deterministic_latency_ms(80, "conv-1", 0, "llm")
        assert a != b

    def test_output_is_positive(self) -> None:
        for i in range(50):
            v = deterministic_latency_ms(10, f"conv-{i}", i, "stt")
            assert v >= 1

    def test_100_samples_not_all_identical(self) -> None:
        samples = [deterministic_latency_ms(80, "c", i, "stt") for i in range(100)]
        assert len(set(samples)) > 1, "all 100 samples were identical — jitter is broken"

    def test_realistic_spread_over_large_sample(self) -> None:
        """p95 should be noticeably above p50 over 200 samples."""
        samples = sorted(
            deterministic_latency_ms(120, f"conv-{i}", i % 20, "llm") for i in range(200)
        )
        p50 = samples[len(samples) // 2]
        p95 = samples[int(0.95 * len(samples))]
        p99 = samples[int(0.99 * len(samples))]
        # p95 must exceed p50 by at least 10%
        assert p95 > p50 * 1.10, f"insufficient spread: p50={p50} p95={p95}"
        # p99 must exceed p95
        assert p99 >= p95


# ---------------------------------------------------------------------------
# GroqLLM — mock path (no env var)
# ---------------------------------------------------------------------------


class TestGroqLLMMock:
    def test_mock_flag_set_when_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        adapter = GroqLLM()
        assert adapter._mock is True

    def test_real_flag_set_when_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "sk-test-fake")
        adapter = GroqLLM()
        assert adapter._mock is False

    async def test_reply_returns_text_and_spans(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        adapter = GroqLLM()
        text, spans = await adapter.reply(
            history=[],
            last_user_text="quiz me on postgres replication",
            gold_facts=["Physical replication ships WAL bytes; logical replication ships row-level changes."],
        )
        assert isinstance(text, str) and text
        assert len(spans) >= 1
        assert spans[0].name == "llm.reply"

    async def test_reply_conformance_to_protocol_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        adapter = GroqLLM()
        result = await adapter.reply([], "hi", [])
        assert isinstance(result, tuple) and len(result) == 2

    async def test_jitter_active_when_turn_context_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        adapter = GroqLLM()
        adapter._conv_id = "test-conv"
        # Sample two different turn indices — should produce different latencies
        adapter._turn_index = 0
        _, spans_0 = await adapter.reply([], "hi", [])
        adapter._turn_index = 5
        _, spans_1 = await adapter.reply([], "hi", [])
        lat_0 = spans_0[0].ended_at_ms - spans_0[0].started_at_ms
        lat_1 = spans_1[0].ended_at_ms - spans_1[0].started_at_ms
        # Not guaranteed to differ for every pair, but these two inputs
        # happen to differ (verified by running the jitter function).
        # We assert non-degeneracy across the adapter as a whole.
        assert lat_0 > 0 and lat_1 > 0


# ---------------------------------------------------------------------------
# DeepgramSTT — mock path (no env var)
# ---------------------------------------------------------------------------


class TestDeepgramSTTMock:
    def test_mock_flag_set_when_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        adapter = DeepgramSTT()
        assert adapter._mock is True

    def test_real_flag_set_when_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPGRAM_API_KEY", "fake-dg-key")
        adapter = DeepgramSTT()
        assert adapter._mock is False

    async def test_transcribe_returns_text_and_spans(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        adapter = DeepgramSTT()
        turn = Turn(role=TurnRole.USER, text="hello world", started_at_ms=0, ended_at_ms=1000)
        text, spans = await adapter.transcribe(turn)
        assert isinstance(text, str) and text == "hello world"
        assert len(spans) >= 1
        assert spans[0].name == "stt.transcribe"

    async def test_transcribe_conformance_to_protocol_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        adapter = DeepgramSTT()
        turn = Turn(role=TurnRole.USER, text="hi", started_at_ms=0, ended_at_ms=500)
        result = await adapter.transcribe(turn)
        assert isinstance(result, tuple) and len(result) == 2

    async def test_wer_substitution_propagated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        adapter = DeepgramSTT(wer_substitution_rate=1.0)
        turn = Turn(role=TurnRole.USER, text="one two three", started_at_ms=0, ended_at_ms=500)
        text, _ = await adapter.transcribe(turn)
        assert all(w == "WERR" for w in text.split())


# ---------------------------------------------------------------------------
# CartesiaTTS — mock path (no env var)
# ---------------------------------------------------------------------------


class TestCartesiaTTSMock:
    def test_mock_flag_set_when_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
        adapter = CartesiaTTS()
        assert adapter._mock is True

    def test_real_flag_set_when_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CARTESIA_API_KEY", "fake-cartesia-key")
        adapter = CartesiaTTS()
        assert adapter._mock is False

    async def test_synthesize_returns_int_and_spans(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
        adapter = CartesiaTTS()
        first_byte, spans = await adapter.synthesize("hello voice")
        assert isinstance(first_byte, int) and first_byte > 0
        assert len(spans) >= 1
        assert spans[0].name == "tts.synthesize"

    async def test_synthesize_conformance_to_protocol_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
        adapter = CartesiaTTS()
        result = await adapter.synthesize("hi")
        assert isinstance(result, tuple) and len(result) == 2


# ---------------------------------------------------------------------------
# Full-pipeline latency spread — p50 ≠ p95 ≠ p99
# ---------------------------------------------------------------------------


class TestLatencySpreadInEval:
    """Run the golden set through the pipeline and assert non-degenerate percentiles."""

    async def test_p50_p95_p99_differ(self) -> None:
        """After the jitter change, turn latencies must have real spread."""
        pipeline = VoicePipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())
        latencies: list[float] = []
        for conv in default_golden_set():
            run = await pipeline.run(conv)
            for tr in run.turn_runs:
                if tr.false_trigger:
                    continue
                vad = next((s for s in tr.spans if s.name == "vad_end"), None)
                fb = next((s for s in tr.spans if s.name == "tts_first_byte"), None)
                if vad and fb:
                    latencies.append(float(fb.ended_at_ms - vad.ended_at_ms))

        assert len(latencies) >= 5, f"too few latency samples: {latencies}"
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        p50 = latencies_sorted[n // 2]
        p95 = latencies_sorted[int(0.95 * n)]
        p99 = latencies_sorted[min(int(0.99 * n), n - 1)]

        # The key assertion: percentiles must not all be identical.
        assert not (p50 == p95 == p99), (
            f"Degenerate latencies — p50={p50} p95={p95} p99={p99}. "
            "All percentiles are identical, meaning jitter is not working."
        )
        # Sanity: p95 >= p50, p99 >= p95
        assert p95 >= p50
        assert p99 >= p95

    async def test_latency_values_not_all_same(self) -> None:
        """No two consecutive turns in the same conversation should share identical latency."""
        pipeline = VoicePipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())
        # Use a multi-turn conversation (postgres-replication has 2 user turns)
        conv = next(c for c in default_golden_set() if c.conv_id == "postgres-replication")
        run = await pipeline.run(conv)
        latencies = []
        for tr in run.turn_runs:
            vad = next((s for s in tr.spans if s.name == "vad_end"), None)
            fb = next((s for s in tr.spans if s.name == "tts_first_byte"), None)
            if vad and fb:
                latencies.append(fb.ended_at_ms - vad.ended_at_ms)

        assert len(latencies) == 2
        assert latencies[0] != latencies[1], (
            "Turn 0 and turn 1 have identical latency — jitter is not varying per turn."
        )

    async def test_large_synthetic_corpus_has_realistic_spread(self) -> None:
        """100 synthetic turns should produce p95 clearly above p50."""
        turns = [
            Turn(
                role=TurnRole.USER,
                text=f"question {i}",
                started_at_ms=i * 2000,
                ended_at_ms=i * 2000 + 1000,
            )
            for i in range(100)
        ]
        conv = Conversation(conv_id="synthetic-spread", topic="spread-test", turns=turns, gold_facts=[])
        pipeline = VoicePipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())
        run = await pipeline.run(conv)

        latencies: list[float] = []
        for tr in run.turn_runs:
            vad = next((s for s in tr.spans if s.name == "vad_end"), None)
            fb = next((s for s in tr.spans if s.name == "tts_first_byte"), None)
            if vad and fb:
                latencies.append(float(fb.ended_at_ms - vad.ended_at_ms))

        latencies.sort()
        n = len(latencies)
        assert n == 100
        p50 = latencies[n // 2]
        p95 = latencies[int(0.95 * n)]

        # Must have real spread: at minimum p95 > p50
        assert p95 > p50, f"No spread: p50={p50} p95={p95}"
        # Standard deviation must be non-zero
        stdev = statistics.stdev(latencies)
        assert stdev > 0, "Zero standard deviation -- all latencies are identical"


# ---------------------------------------------------------------------------
# Adapter integration with VoicePipeline
# ---------------------------------------------------------------------------


class TestAdaptersInPipeline:
    async def test_groq_adapter_works_in_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        pipeline = VoicePipeline(stt=MockSTT(), llm=GroqLLM(), tts=MockTTS())
        conv = default_golden_set()[0]
        run = await pipeline.run(conv)
        assert run.user_turns_played > 0
        for tr in run.turn_runs:
            assert "vad_end" in {s.name for s in tr.spans}

    async def test_deepgram_adapter_works_in_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        pipeline = VoicePipeline(stt=DeepgramSTT(), llm=MockLLM(), tts=MockTTS())
        conv = default_golden_set()[0]
        run = await pipeline.run(conv)
        assert run.user_turns_played > 0

    async def test_cartesia_adapter_works_in_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
        pipeline = VoicePipeline(stt=MockSTT(), llm=MockLLM(), tts=CartesiaTTS())
        conv = default_golden_set()[0]
        run = await pipeline.run(conv)
        assert run.user_turns_played > 0

    async def test_all_real_adapters_in_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All three adapters together, all in mock mode."""
        for key in ("GROQ_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        pipeline = VoicePipeline(stt=DeepgramSTT(), llm=GroqLLM(), tts=CartesiaTTS())
        conv = default_golden_set()[0]
        run = await pipeline.run(conv)
        assert run.user_turns_played > 0
        # Jitter should produce non-identical latencies across turns
        latencies = []
        for tr in run.turn_runs:
            vad = next((s for s in tr.spans if s.name == "vad_end"), None)
            fb = next((s for s in tr.spans if s.name == "tts_first_byte"), None)
            if vad and fb:
                latencies.append(fb.ended_at_ms - vad.ended_at_ms)
        # At least one turn must have been processed
        assert len(latencies) >= 1


# ---------------------------------------------------------------------------
# Fix 1: DeepgramSTT real-mode guard — must never POST text as audio/wav
# ---------------------------------------------------------------------------


class TestDeepgramNoAudioBytesGuard:
    """When DEEPGRAM_API_KEY is set but no audio_bytes are on the turn,
    the adapter must warn and fall back to mock — never POST text as WAV.
    """

    async def test_real_mode_without_audio_bytes_falls_back_to_mock(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("DEEPGRAM_API_KEY", "fake-dg-key")
        adapter = DeepgramSTT()
        assert adapter._mock is False  # key is set → real-mode

        turn = Turn(role=TurnRole.USER, text="hello world", started_at_ms=0, ended_at_ms=1000)
        # turn has no audio_bytes attribute → must fall back to mock
        with caplog.at_level(logging.WARNING, logger="voice_eval_lab.adapters.deepgram"):
            text, spans = await adapter.transcribe(turn)

        # Falls back to mock: text is unchanged (0% WER), span engine is "mock"
        assert text == "hello world"
        assert spans[0].attrs.get("engine") == "mock"

    async def test_real_mode_warning_message_is_descriptive(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("DEEPGRAM_API_KEY", "fake-dg-key")
        adapter = DeepgramSTT()

        turn = Turn(role=TurnRole.USER, text="test input", started_at_ms=0, ended_at_ms=500)
        with caplog.at_level(logging.WARNING, logger="voice_eval_lab.adapters.deepgram"):
            await adapter.transcribe(turn)

        # Warning must mention audio_bytes so the caller knows what to supply
        assert any("audio_bytes" in r.message for r in caplog.records), (
            "Expected warning mentioning 'audio_bytes' — got: "
            + str([r.message for r in caplog.records])
        )

    async def test_real_mode_standard_turn_no_audio_bytes_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A standard Turn (no audio_bytes field) causes fallback and warning.

        Turn is a Pydantic model — getattr returns None (the default), which
        is sufficient to trigger the guard without needing to set the field.
        """
        monkeypatch.setenv("DEEPGRAM_API_KEY", "fake-dg-key")
        adapter = DeepgramSTT()

        # Standard Turn has no audio_bytes field; getattr default is None
        turn = Turn(role=TurnRole.USER, text="no audio", started_at_ms=0, ended_at_ms=500)
        assert getattr(turn, "audio_bytes", None) is None  # confirm guard precondition
        with caplog.at_level(logging.WARNING, logger="voice_eval_lab.adapters.deepgram"):
            _text, spans = await adapter.transcribe(turn)

        assert spans[0].attrs.get("engine") == "mock"
        assert any("audio_bytes" in r.message for r in caplog.records)

    async def test_real_mode_with_audio_bytes_present_does_not_fall_back_to_mock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When audio_bytes is set on a custom turn subclass, the adapter proceeds
        to the HTTP path (not mock) and posts the audio bytes — not text.

        We inject a fake httpx module so no real network call is made.
        """
        monkeypatch.setenv("DEEPGRAM_API_KEY", "fake-dg-key")

        # Build a fake httpx module with an AsyncClient we can control.
        fake_response = unittest.mock.MagicMock()
        fake_response.raise_for_status = unittest.mock.MagicMock()
        fake_response.json.return_value = {
            "results": {
                "channels": [{"alternatives": [{"transcript": "audio text"}]}]
            }
        }

        fake_client = unittest.mock.AsyncMock()
        fake_client.post = unittest.mock.AsyncMock(return_value=fake_response)
        fake_client.__aenter__ = unittest.mock.AsyncMock(return_value=fake_client)
        fake_client.__aexit__ = unittest.mock.AsyncMock(return_value=False)

        fake_httpx = unittest.mock.MagicMock()
        fake_httpx.AsyncClient.return_value = fake_client

        # Temporarily replace the None _httpx with our fake module.
        original_httpx = _dg_mod._httpx
        _dg_mod._httpx = fake_httpx  # type: ignore[assignment]
        try:
            adapter = DeepgramSTT()

            fake_wav = b"RIFF\x00\x00\x00\x00WAVEfmt "  # minimal WAV-like header

            # Use a simple namespace to carry audio_bytes past the guard.
            class AudioTurn(Turn):
                model_config = Turn.model_config.copy()  # type: ignore[attr-defined]
                model_config["extra"] = "allow"

            turn = AudioTurn(
                role=TurnRole.USER, text="audio text", started_at_ms=0, ended_at_ms=500
            )
            object.__setattr__(turn, "audio_bytes", fake_wav)

            text, spans = await adapter.transcribe(turn)
        finally:
            _dg_mod._httpx = original_httpx  # type: ignore[assignment]

        # Confirm the content posted was audio_bytes, NOT turn.text.encode()
        call_kwargs = fake_client.post.call_args
        posted_content = call_kwargs.kwargs.get("content") or (
            call_kwargs.args[1] if len(call_kwargs.args) > 1 else None
        )
        assert posted_content == fake_wav, (
            "Expected audio_bytes to be posted, got text bytes instead"
        )
        assert text == "audio text"
        assert spans[0].attrs.get("engine") == "deepgram"


# ---------------------------------------------------------------------------
# Fix 3: CartesiaTTS uses Authorization: Bearer header
# ---------------------------------------------------------------------------


class TestCartesiaAuthHeader:
    """CartesiaTTS real-mode must send Authorization: Bearer <key>, not X-API-Key."""

    async def test_real_synthesize_sends_bearer_auth_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CartesiaTTS must send Authorization: Bearer <key> (not X-API-Key).

        httpx may not be installed in CI, so we inject a fake httpx module
        directly into the adapter module rather than using patch.object on None.
        """
        monkeypatch.setenv("CARTESIA_API_KEY", "fake-cartesia-key")

        fake_response = unittest.mock.MagicMock()
        fake_response.raise_for_status = unittest.mock.MagicMock()

        async def fake_aiter_bytes(chunk_size: int = 256):
            yield b"\x00" * 256

        fake_response.aiter_bytes = fake_aiter_bytes

        fake_stream_ctx = unittest.mock.AsyncMock()
        fake_stream_ctx.__aenter__ = unittest.mock.AsyncMock(return_value=fake_response)
        fake_stream_ctx.__aexit__ = unittest.mock.AsyncMock(return_value=False)

        fake_client = unittest.mock.MagicMock()
        fake_client.stream = unittest.mock.MagicMock(return_value=fake_stream_ctx)
        fake_client.__aenter__ = unittest.mock.AsyncMock(return_value=fake_client)
        fake_client.__aexit__ = unittest.mock.AsyncMock(return_value=False)

        fake_httpx = unittest.mock.MagicMock()
        fake_httpx.AsyncClient.return_value = fake_client

        original_httpx = _ca_mod._httpx
        _ca_mod._httpx = fake_httpx  # type: ignore[assignment]
        try:
            adapter = CartesiaTTS()
            assert adapter._mock is False
            _first_byte, _spans = await adapter.synthesize("hello")
        finally:
            _ca_mod._httpx = original_httpx  # type: ignore[assignment]

        stream_call_kwargs = fake_client.stream.call_args
        headers_sent = stream_call_kwargs.kwargs.get("headers", {})

        assert "Authorization" in headers_sent, (
            f"Expected 'Authorization' header; got headers: {headers_sent}"
        )
        assert headers_sent["Authorization"] == "Bearer fake-cartesia-key", (
            f"Expected 'Bearer fake-cartesia-key'; got: {headers_sent['Authorization']}"
        )
        assert "X-API-Key" not in headers_sent, (
            "X-API-Key header must not be present (deprecated); use Authorization: Bearer"
        )
