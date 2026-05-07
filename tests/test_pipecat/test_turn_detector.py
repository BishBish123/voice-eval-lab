"""Tests for SmartTurnDetector — rule-based fallback + mocked Pipecat path.

No real ONNX model is loaded. The Pipecat SmartTurnAnalyzer is always mocked.
"""

from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from voice_eval_lab.pipecat.turn_detector import (
    _ENERGY_THRESHOLD,
    SmartTurnDetector,
    TurnState,
    _chunk_duration_ms,
    _chunk_energy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16_000  # 16 kHz mono


def _make_pcm(n_samples: int, amplitude: int = 0) -> bytes:
    """Return a raw 16-bit LE PCM chunk with *n_samples* all at *amplitude*."""
    return struct.pack(f"<{n_samples}h", *([amplitude] * n_samples))


def _silence(duration_ms: int, sample_rate: int = _SAMPLE_RATE) -> bytes:
    """Return a silent PCM chunk of the given duration."""
    n = int(sample_rate * duration_ms / 1_000)
    return _make_pcm(n, amplitude=0)


def _active(duration_ms: int, sample_rate: int = _SAMPLE_RATE) -> bytes:
    """Return a loud PCM chunk (amplitude=10 000) of the given duration."""
    n = int(sample_rate * duration_ms / 1_000)
    return _make_pcm(n, amplitude=10_000)


# ---------------------------------------------------------------------------
# _chunk_energy unit tests
# ---------------------------------------------------------------------------


class TestChunkEnergy:
    def test_silence_chunk_is_zero(self) -> None:
        chunk = _silence(10)
        assert _chunk_energy(chunk) == 0.0

    def test_active_chunk_above_threshold(self) -> None:
        chunk = _active(10)
        assert _chunk_energy(chunk) > _ENERGY_THRESHOLD

    def test_empty_chunk_returns_zero(self) -> None:
        assert _chunk_energy(b"") == 0.0

    def test_single_byte_returns_zero(self) -> None:
        # Less than one 16-bit sample → treated as empty.
        assert _chunk_energy(b"\xff") == 0.0


class TestChunkDurationMs:
    def test_known_duration(self) -> None:
        # 160 samples at 16 kHz = 10 ms
        chunk = _make_pcm(160)
        assert _chunk_duration_ms(chunk, sample_rate=16_000) == pytest.approx(10.0)

    def test_zero_sample_rate_returns_zero(self) -> None:
        assert _chunk_duration_ms(b"\x00\x00", sample_rate=0) == 0.0

    def test_empty_chunk_returns_zero(self) -> None:
        assert _chunk_duration_ms(b"", sample_rate=16_000) == 0.0


# ---------------------------------------------------------------------------
# Rule-based fallback: silence detection
# ---------------------------------------------------------------------------


class TestSmartTurnDetectorFallback:
    """Pipecat unavailable → energy-based silence detector is used."""

    def _make_detector(self, min_silence_ms: int = 300) -> SmartTurnDetector:
        """Return a detector that always uses the fallback (no analyzer)."""
        d = SmartTurnDetector(min_silence_ms=min_silence_ms)
        d._analyzer = None  # Force fallback regardless of environment.
        return d

    # --- End-of-turn when silence exceeds threshold ---

    def test_silence_exceeds_threshold_returns_end_of_turn(self) -> None:
        detector = self._make_detector(min_silence_ms=300)
        # Feed 400 ms of silence: should exceed 300 ms threshold.
        state = detector.analyze(_silence(400))
        assert state.is_end_of_turn is True
        assert state.confidence == pytest.approx(0.7)

    def test_silence_below_threshold_returns_not_end_of_turn(self) -> None:
        detector = self._make_detector(min_silence_ms=500)
        # Feed 200 ms of silence: below 500 ms threshold.
        state = detector.analyze(_silence(200))
        assert state.is_end_of_turn is False
        assert state.confidence == pytest.approx(0.3)

    # --- Active audio does not trigger end-of-turn ---

    def test_active_audio_returns_not_end_of_turn(self) -> None:
        detector = self._make_detector(min_silence_ms=300)
        state = detector.analyze(_active(400))
        assert state.is_end_of_turn is False
        assert state.confidence == pytest.approx(0.3)

    # --- Mixed audio: active then silence flips correctly ---

    def test_mixed_active_then_silence_flips_correctly(self) -> None:
        detector = self._make_detector(min_silence_ms=300)
        # Active chunk — should not be end-of-turn.
        state1 = detector.analyze(_active(200))
        assert state1.is_end_of_turn is False
        # Another active chunk — still no.
        state2 = detector.analyze(_active(200))
        assert state2.is_end_of_turn is False
        # Now silence for 400 ms — should flip.
        state3 = detector.analyze(_silence(400))
        assert state3.is_end_of_turn is True

    def test_silence_resets_on_active_audio(self) -> None:
        """Accumulated silence must reset when active speech arrives."""
        detector = self._make_detector(min_silence_ms=300)
        # 200 ms silence (accumulates but doesn't exceed 300).
        detector.analyze(_silence(200))
        # Active chunk — resets accumulator.
        detector.analyze(_active(100))
        # Another 200 ms silence — still below threshold (accumulator was reset).
        state = detector.analyze(_silence(200))
        assert state.is_end_of_turn is False

    def test_accumulated_silence_across_multiple_chunks(self) -> None:
        """Silence accumulation spans across multiple successive silent chunks."""
        detector = self._make_detector(min_silence_ms=300)
        # 150 ms + 160 ms = 310 ms silence in two chunks → exceeds 300.
        state1 = detector.analyze(_silence(150))
        assert state1.is_end_of_turn is False
        state2 = detector.analyze(_silence(160))
        assert state2.is_end_of_turn is True

    # --- Return type ---

    def test_analyze_returns_turn_state_named_tuple(self) -> None:
        detector = self._make_detector()
        result = detector.analyze(_silence(100))
        assert isinstance(result, TurnState)
        assert isinstance(result.is_end_of_turn, bool)
        assert isinstance(result.confidence, float)

    # --- reset() ---

    def test_reset_clears_silence_accumulator(self) -> None:
        detector = self._make_detector(min_silence_ms=300)
        # Accumulate just below threshold.
        detector.analyze(_silence(200))
        # Reset — accumulator should go to zero.
        detector.reset()
        # 150 ms more silence — would be 350 ms total without reset, but
        # after reset it's only 150 ms and should not trigger.
        state = detector.analyze(_silence(150))
        assert state.is_end_of_turn is False


# ---------------------------------------------------------------------------
# Custom thresholds propagate
# ---------------------------------------------------------------------------


class TestSmartTurnDetectorThresholds:
    def test_min_silence_ms_stored(self) -> None:
        d = SmartTurnDetector(min_silence_ms=750)
        assert d.min_silence_ms == 750

    def test_eou_threshold_stored(self) -> None:
        d = SmartTurnDetector(eou_threshold=0.8)
        assert d.eou_threshold == pytest.approx(0.8)

    def test_custom_min_silence_ms_changes_detection(self) -> None:
        """A detector with min_silence_ms=100 fires sooner than one with 500."""
        fast = SmartTurnDetector(min_silence_ms=100)
        fast._analyzer = None

        slow = SmartTurnDetector(min_silence_ms=500)
        slow._analyzer = None

        chunk = _silence(200)  # 200 ms — enough for fast, not for slow.
        assert fast.analyze(chunk).is_end_of_turn is True
        assert slow.analyze(chunk).is_end_of_turn is False

    def test_eou_threshold_forwarded_to_analyzer(self) -> None:
        """When SmartTurnAnalyzer is mocked, eou_threshold is passed to __init__."""
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        with (
            patch(
                "voice_eval_lab.pipecat.turn_detector._SMART_TURN_AVAILABLE",
                True,
            ),
            patch(
                "voice_eval_lab.pipecat.turn_detector._SmartTurnAnalyzer",
                mock_cls,
            ),
        ):
            SmartTurnDetector(eou_threshold=0.65)

        mock_cls.assert_called_once_with(eou_threshold=0.65)


# ---------------------------------------------------------------------------
# Pipecat available path — mocked SmartTurnAnalyzer
# ---------------------------------------------------------------------------


class TestSmartTurnDetectorPipecatPath:
    """Verify delegation when SmartTurnAnalyzer IS available (mocked)."""

    def _make_detector_with_mock_analyzer(
        self, analyze_return: object
    ) -> SmartTurnDetector:
        """Build a SmartTurnDetector whose _analyzer is a mock."""
        detector = SmartTurnDetector()
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = analyze_return
        detector._analyzer = mock_analyzer
        return detector

    def test_delegates_to_analyzer_tuple_result(self) -> None:
        detector = self._make_detector_with_mock_analyzer((True, 0.9))
        state = detector.analyze(_silence(10))
        assert state.is_end_of_turn is True
        assert state.confidence == pytest.approx(0.9)

    def test_delegates_to_analyzer_dict_result(self) -> None:
        detector = self._make_detector_with_mock_analyzer(
            {"is_end_of_turn": False, "confidence": 0.2}
        )
        state = detector.analyze(_silence(10))
        assert state.is_end_of_turn is False
        assert state.confidence == pytest.approx(0.2)

    def test_delegates_to_analyzer_turn_state_result(self) -> None:
        expected = TurnState(is_end_of_turn=True, confidence=0.85)
        detector = self._make_detector_with_mock_analyzer(expected)
        state = detector.analyze(_silence(10))
        assert state == expected

    def test_fallback_on_analyzer_exception(self) -> None:
        """If the analyzer raises, fall through to energy-based detection."""
        detector = SmartTurnDetector(min_silence_ms=100)
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = RuntimeError("boom")
        detector._analyzer = mock_analyzer

        # 200 ms of silence should trigger fallback end-of-turn (threshold=100).
        state = detector.analyze(_silence(200))
        assert state.is_end_of_turn is True

    def test_fallback_on_unknown_result_shape(self) -> None:
        """If the analyzer returns an unknown shape, fall through to fallback."""
        detector = SmartTurnDetector(min_silence_ms=100)
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = "unexpected_string"
        detector._analyzer = mock_analyzer

        # 200 ms silence → fallback should fire.
        state = detector.analyze(_silence(200))
        assert state.is_end_of_turn is True

    def test_analyzer_receives_audio_chunk(self) -> None:
        """The analyzer must receive the exact bytes passed to analyze()."""
        detector = SmartTurnDetector()
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = (False, 0.3)
        detector._analyzer = mock_analyzer

        chunk = _active(50)
        detector.analyze(chunk)
        mock_analyzer.analyze.assert_called_once_with(chunk)


# ---------------------------------------------------------------------------
# Pipeline wiring: turn_detector param in build_pipeline
# ---------------------------------------------------------------------------


class TestBuildPipelineTurnDetectorParam:
    def test_smart_mode_attaches_smart_turn_detector(self) -> None:
        from voice_eval_lab.pipecat.pipeline import build_pipeline
        from voice_eval_lab.pipeline import MockLLM, MockSTT, MockTTS

        pipeline = build_pipeline(
            stt=MockSTT(), llm=MockLLM(), tts=MockTTS(), turn_detector="smart"
        )
        stt_proc = pipeline.processors()[0]
        assert isinstance(stt_proc._turn_detector, SmartTurnDetector)

    def test_none_mode_attaches_stub(self) -> None:
        from voice_eval_lab.pipecat.pipeline import _TurnDetectorStub, build_pipeline
        from voice_eval_lab.pipeline import MockLLM, MockSTT, MockTTS

        pipeline = build_pipeline(
            stt=MockSTT(), llm=MockLLM(), tts=MockTTS(), turn_detector="none"
        )
        stt_proc = pipeline.processors()[0]
        assert isinstance(stt_proc._turn_detector, _TurnDetectorStub)

    def test_default_turn_detector_is_smart(self) -> None:
        """build_pipeline() with no turn_detector arg defaults to smart."""
        from voice_eval_lab.pipecat.pipeline import build_pipeline
        from voice_eval_lab.pipeline import MockLLM, MockSTT, MockTTS

        pipeline = build_pipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())
        stt_proc = pipeline.processors()[0]
        assert isinstance(stt_proc._turn_detector, SmartTurnDetector)


# ---------------------------------------------------------------------------
# CLI: --turn-detector flag
# ---------------------------------------------------------------------------


class TestCLITurnDetectorFlag:
    def test_smart_flag_runs_successfully(self) -> None:
        from voice_eval_lab.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["pipeline", "run", "--turn-detector", "smart"])
        assert result.exit_code == 0, result.output

    def test_none_flag_runs_successfully(self) -> None:
        from voice_eval_lab.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["pipeline", "run", "--turn-detector", "none"])
        assert result.exit_code == 0, result.output

    def test_invalid_flag_exits_nonzero(self) -> None:
        from voice_eval_lab.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["pipeline", "run", "--turn-detector", "invalid"])
        assert result.exit_code != 0

    def test_smart_flag_pipeline_has_smart_turn_detector(self) -> None:
        """Verify that --turn-detector smart wires SmartTurnDetector into the pipeline."""
        import voice_eval_lab.pipecat.pipeline as _pipeline_mod
        from voice_eval_lab.cli import app

        captured_detectors: list[object] = []
        original_build = _pipeline_mod.build_pipeline

        def _capturing_build(stt: object, llm: object, tts: object, *, turn_detector: str = "smart") -> object:  # type: ignore[misc]
            pipeline = original_build(stt, llm, tts, turn_detector=turn_detector)  # type: ignore[arg-type]
            captured_detectors.append(pipeline.processors()[0]._turn_detector)
            return pipeline

        runner = CliRunner()
        with patch.object(_pipeline_mod, "build_pipeline", side_effect=_capturing_build):
            result = runner.invoke(app, ["pipeline", "run", "--turn-detector", "smart"])

        assert result.exit_code == 0, result.output
        assert len(captured_detectors) >= 1
        assert isinstance(captured_detectors[0], SmartTurnDetector)
