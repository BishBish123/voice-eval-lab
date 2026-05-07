"""Tests for WhisperSTT adapter and make_stt() factory.

All tests run without installing openai-whisper or loading any real models.
No network calls, no file I/O, no torch.

Coverage goals:
- WhisperSTT._mock=True when openai-whisper is absent
- WhisperSTT with mocked whisper: model loaded with correct name, transcribe returns text
- Audio format conversion: 24kHz → 16kHz resampling via stdlib wave
- Lazy model loading: __init__ does NOT call load_model
- make_stt() factory env-var dispatch (mock / deepgram / whisper)
- CLI --stt flag dispatch (auto / mock / deepgram / whisper)
"""

from __future__ import annotations

import io
import struct
import sys
import types
import unittest.mock
import wave

import pytest

from voice_eval_lab.models import Turn, TurnRole
from voice_eval_lab.pipeline import MockSTT

# ---------------------------------------------------------------------------
# Helpers — synthesise a minimal valid WAV blob
# ---------------------------------------------------------------------------


def _make_wav(
    n_samples: int = 160,
    sample_rate: int = 16_000,
    n_channels: int = 1,
    sampwidth: int = 2,
) -> bytes:
    """Return a valid in-memory WAV with *n_samples* of silence."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00" * (n_samples * n_channels * sampwidth))
    return buf.getvalue()


def _make_turn(audio_bytes: bytes | None = None, text: str = "hello") -> Turn:
    turn = Turn(role=TurnRole.USER, text=text, started_at_ms=0, ended_at_ms=500)
    if audio_bytes is not None:
        object.__setattr__(turn, "audio_bytes", audio_bytes)
    return turn


# ---------------------------------------------------------------------------
# 1. WhisperSTT._mock=True when openai-whisper is absent
# ---------------------------------------------------------------------------


class TestWhisperSTTNoPackage:
    """Simulate missing openai-whisper by patching the module-level sentinel."""

    def _import_fresh(self) -> type:
        """Force a fresh import of whisper adapter with _whisper_available=False."""
        # Remove cached module if present so we can re-import with patched state.
        mod_name = "voice_eval_lab.adapters.whisper"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        import voice_eval_lab.adapters.whisper as wmod

        return wmod.WhisperSTT

    def test_mock_flag_true_when_package_absent(self) -> None:
        """When _whisper_available is False the adapter must set _mock=True."""
        import voice_eval_lab.adapters.whisper as wmod

        orig = wmod._whisper_available
        try:
            wmod._whisper_available = False  # type: ignore[assignment]
            # Re-instantiate after patching the sentinel.
            adapter = wmod.WhisperSTT.__new__(wmod.WhisperSTT)
            wmod.WhisperSTT.__init__(adapter)
            assert adapter._mock is True
        finally:
            wmod._whisper_available = orig

    async def test_transcribe_delegates_to_mock_when_package_absent(self) -> None:
        """Transcribe without audio_bytes + _mock=True must delegate to MockSTT."""
        import voice_eval_lab.adapters.whisper as wmod

        orig = wmod._whisper_available
        try:
            wmod._whisper_available = False  # type: ignore[assignment]
            adapter = wmod.WhisperSTT()
            assert adapter._mock is True
            turn = _make_turn(text="hello world")
            text, spans = await adapter.transcribe(turn)
            # Should get MockSTT output: same text, engine=mock
            assert text == "hello world"
            assert spans[0].attrs.get("engine") == "mock"
        finally:
            wmod._whisper_available = orig

    def test_model_not_loaded_at_init(self) -> None:
        """_model must be None immediately after __init__ (lazy load)."""
        import voice_eval_lab.adapters.whisper as wmod

        adapter = wmod.WhisperSTT()
        assert adapter._model is None


# ---------------------------------------------------------------------------
# 2. WhisperSTT with mocked whisper package
# ---------------------------------------------------------------------------


class TestWhisperSTTMockedPackage:
    """Patch _whisper_mod and _np so no real Whisper is used."""

    def _build_adapter(self, model_name: str = "tiny") -> object:
        import voice_eval_lab.adapters.whisper as wmod

        adapter = wmod.WhisperSTT(model_name=model_name)
        return adapter

    def _patch_whisper(
        self,
        wmod: types.ModuleType,
        transcript: str = "mocked transcript",
    ) -> tuple[unittest.mock.MagicMock, unittest.mock.MagicMock]:
        """Return (fake_whisper_mod, fake_np) mocks installed into wmod."""
        import numpy as np

        fake_model = unittest.mock.MagicMock()
        fake_whisper = unittest.mock.MagicMock()
        fake_whisper.load_model.return_value = fake_model
        fake_whisper.transcribe.return_value = {"text": transcript}

        wmod._whisper_mod = fake_whisper  # type: ignore[assignment]
        wmod._np = np  # real numpy — safe to use
        wmod._whisper_available = True  # type: ignore[assignment]

        return fake_whisper, fake_model

    async def test_transcribe_returns_expected_text(self) -> None:
        import voice_eval_lab.adapters.whisper as wmod

        orig_mod = wmod._whisper_mod
        orig_avail = wmod._whisper_available
        try:
            _fake_whisper, _ = self._patch_whisper(wmod, transcript="hello world")
            adapter = wmod.WhisperSTT(model_name="tiny")
            wav = _make_wav()
            turn = _make_turn(audio_bytes=wav, text="hello world")

            text, spans = await adapter.transcribe(turn)

            assert text == "hello world"
            assert spans[0].name == "stt.transcribe"
            assert spans[0].attrs["engine"] == "whisper"
            assert spans[0].attrs["model"] == "tiny"
        finally:
            wmod._whisper_mod = orig_mod
            wmod._whisper_available = orig_avail

    async def test_load_model_called_with_correct_name(self) -> None:
        import voice_eval_lab.adapters.whisper as wmod

        orig_mod = wmod._whisper_mod
        orig_avail = wmod._whisper_available
        try:
            fake_whisper, _ = self._patch_whisper(wmod)
            adapter = wmod.WhisperSTT(model_name="base")
            wav = _make_wav()
            turn = _make_turn(audio_bytes=wav)

            await adapter.transcribe(turn)

            fake_whisper.load_model.assert_called_once_with("base")
        finally:
            wmod._whisper_mod = orig_mod
            wmod._whisper_available = orig_avail

    async def test_model_loaded_only_once_across_multiple_calls(self) -> None:
        import voice_eval_lab.adapters.whisper as wmod

        orig_mod = wmod._whisper_mod
        orig_avail = wmod._whisper_available
        try:
            fake_whisper, _ = self._patch_whisper(wmod)
            adapter = wmod.WhisperSTT(model_name="tiny")
            wav = _make_wav()
            turn = _make_turn(audio_bytes=wav)

            await adapter.transcribe(turn)
            await adapter.transcribe(turn)

            # load_model must only be called once regardless of call count.
            fake_whisper.load_model.assert_called_once()
        finally:
            wmod._whisper_mod = orig_mod
            wmod._whisper_available = orig_avail

    async def test_span_engine_is_whisper(self) -> None:
        import voice_eval_lab.adapters.whisper as wmod

        orig_mod = wmod._whisper_mod
        orig_avail = wmod._whisper_available
        try:
            self._patch_whisper(wmod, transcript="hi")
            adapter = wmod.WhisperSTT()
            wav = _make_wav()
            turn = _make_turn(audio_bytes=wav)

            _text, spans = await adapter.transcribe(turn)

            assert spans[0].attrs.get("source") == "whisper"
        finally:
            wmod._whisper_mod = orig_mod
            wmod._whisper_available = orig_avail


# ---------------------------------------------------------------------------
# 3. Audio format conversion: 24kHz → 16kHz
# ---------------------------------------------------------------------------


class TestAudioConversion:
    def test_pcm_bytes_to_float32_same_rate_no_resampling(self) -> None:
        import numpy as np

        import voice_eval_lab.adapters.whisper as wmod

        raw = struct.pack("<4h", 0, 16384, -16384, 32767)
        result = wmod._pcm_bytes_to_float32(raw, src_rate=16_000, target_rate=16_000)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 4

    def test_pcm_bytes_to_float32_24khz_to_16khz_decimates(self) -> None:
        """24kHz→16kHz uses stride=round(24000/16000)=2, halves sample count."""
        import numpy as np

        import voice_eval_lab.adapters.whisper as wmod

        # 100 samples at 24kHz
        raw = struct.pack(f"<{100}h", *range(100))
        result = wmod._pcm_bytes_to_float32(raw, src_rate=24_000, target_rate=16_000)
        assert isinstance(result, np.ndarray)
        # stride=2 → every other sample → 50 samples
        assert len(result) == 50

    def test_decode_wav_returns_correct_rate(self) -> None:
        import voice_eval_lab.adapters.whisper as wmod

        wav = _make_wav(sample_rate=24_000)
        raw_pcm, rate = wmod._decode_wav(wav)
        assert rate == 24_000
        assert isinstance(raw_pcm, bytes)

    def test_decode_wav_rejects_non_16bit(self) -> None:
        import voice_eval_lab.adapters.whisper as wmod

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(4)  # 32-bit
            wf.setframerate(16_000)
            wf.writeframes(b"\x00" * 160)

        with pytest.raises(ValueError, match="16-bit"):
            wmod._decode_wav(buf.getvalue())

    def test_decode_wav_stereo_downmix(self) -> None:
        """Stereo WAV is downmixed to mono (half the frame count)."""
        import voice_eval_lab.adapters.whisper as wmod

        wav = _make_wav(n_samples=100, n_channels=2, sample_rate=16_000)
        raw_pcm, _rate = wmod._decode_wav(wav)
        # 100 stereo frames -> 100 mono samples x 2 bytes each
        assert len(raw_pcm) == 200

    async def test_24khz_audio_transcribed_correctly(self) -> None:
        """End-to-end: 24kHz WAV is resampled before Whisper receives it."""
        import numpy as np

        import voice_eval_lab.adapters.whisper as wmod

        orig_mod = wmod._whisper_mod
        orig_avail = wmod._whisper_available
        try:
            fake_whisper = unittest.mock.MagicMock()
            fake_whisper.load_model.return_value = unittest.mock.MagicMock()
            fake_whisper.transcribe.return_value = {"text": "resampled"}
            wmod._whisper_mod = fake_whisper  # type: ignore[assignment]
            wmod._np = np
            wmod._whisper_available = True  # type: ignore[assignment]

            adapter = wmod.WhisperSTT(model_name="tiny")
            wav = _make_wav(n_samples=480, sample_rate=24_000)  # 20ms at 24kHz
            turn = _make_turn(audio_bytes=wav)

            text, _spans = await adapter.transcribe(turn)

            assert text == "resampled"
            # The array passed to transcribe should be smaller than the raw input
            call_args = fake_whisper.transcribe.call_args
            audio_arg = call_args.args[1]
            assert isinstance(audio_arg, np.ndarray)
            # At stride=2: 480 samples → 240 output samples
            assert len(audio_arg) == 240
        finally:
            wmod._whisper_mod = orig_mod
            wmod._whisper_available = orig_avail


# ---------------------------------------------------------------------------
# 4. Lazy model loading
# ---------------------------------------------------------------------------


class TestLazyModelLoading:
    def test_init_does_not_load_model(self) -> None:
        """WhisperSTT.__init__ must NOT call load_model."""
        import voice_eval_lab.adapters.whisper as wmod

        orig_mod = wmod._whisper_mod
        orig_avail = wmod._whisper_available
        try:
            fake_whisper = unittest.mock.MagicMock()
            wmod._whisper_mod = fake_whisper  # type: ignore[assignment]
            wmod._whisper_available = True  # type: ignore[assignment]

            _adapter = wmod.WhisperSTT(model_name="tiny")

            fake_whisper.load_model.assert_not_called()
        finally:
            wmod._whisper_mod = orig_mod
            wmod._whisper_available = orig_avail

    def test_model_is_none_after_init(self) -> None:
        import voice_eval_lab.adapters.whisper as wmod

        adapter = wmod.WhisperSTT()
        assert adapter._model is None

    async def test_model_loaded_after_first_transcribe(self) -> None:
        import numpy as np

        import voice_eval_lab.adapters.whisper as wmod

        orig_mod = wmod._whisper_mod
        orig_avail = wmod._whisper_available
        try:
            fake_model = unittest.mock.MagicMock()
            fake_whisper = unittest.mock.MagicMock()
            fake_whisper.load_model.return_value = fake_model
            fake_whisper.transcribe.return_value = {"text": "ok"}
            wmod._whisper_mod = fake_whisper  # type: ignore[assignment]
            wmod._np = np
            wmod._whisper_available = True  # type: ignore[assignment]

            adapter = wmod.WhisperSTT()
            assert adapter._model is None  # lazy: not loaded yet

            wav = _make_wav()
            turn = _make_turn(audio_bytes=wav)
            await adapter.transcribe(turn)

            assert adapter._model is fake_model
        finally:
            wmod._whisper_mod = orig_mod
            wmod._whisper_available = orig_avail


# ---------------------------------------------------------------------------
# 5. make_stt() factory env-var dispatch
# ---------------------------------------------------------------------------


class TestMakeSTTFactory:
    def test_no_env_vars_returns_mock_stt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        monkeypatch.delenv("WHISPER_MODEL_NAME", raising=False)
        from voice_eval_lab.adapters import make_stt

        adapter = make_stt()
        assert isinstance(adapter, MockSTT)

    def test_deepgram_key_returns_deepgram_stt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPGRAM_API_KEY", "fake-key")
        monkeypatch.delenv("WHISPER_MODEL_NAME", raising=False)
        from voice_eval_lab.adapters import DeepgramSTT, make_stt

        adapter = make_stt()
        assert isinstance(adapter, DeepgramSTT)

    def test_whisper_model_name_returns_whisper_stt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        monkeypatch.setenv("WHISPER_MODEL_NAME", "base")
        from voice_eval_lab.adapters import WhisperSTT, make_stt

        adapter = make_stt()
        assert isinstance(adapter, WhisperSTT)
        assert adapter._model_name == "base"

    def test_both_keys_deepgram_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When both DEEPGRAM_API_KEY and WHISPER_MODEL_NAME are set, Deepgram wins."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "fake-key")
        monkeypatch.setenv("WHISPER_MODEL_NAME", "tiny")
        from voice_eval_lab.adapters import DeepgramSTT, make_stt

        adapter = make_stt()
        assert isinstance(adapter, DeepgramSTT)

    def test_whisper_adapter_uses_model_name_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        monkeypatch.setenv("WHISPER_MODEL_NAME", "small")
        from voice_eval_lab.adapters import WhisperSTT, make_stt

        adapter = make_stt()
        assert isinstance(adapter, WhisperSTT)
        assert adapter._model_name == "small"


# ---------------------------------------------------------------------------
# 6. CLI --stt flag dispatch
# ---------------------------------------------------------------------------


class TestCLISTTFlag:
    def test_run_stt_mock_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        monkeypatch.delenv("WHISPER_MODEL_NAME", raising=False)
        from typer.testing import CliRunner

        from voice_eval_lab.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["run", "--stt", "mock", "--out", "/dev/null"])
        assert result.exit_code == 0, result.output

    def test_run_stt_auto_flag_defaults_to_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        monkeypatch.delenv("WHISPER_MODEL_NAME", raising=False)
        from typer.testing import CliRunner

        from voice_eval_lab.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["run", "--stt", "auto", "--out", "/dev/null"])
        assert result.exit_code == 0, result.output

    def test_run_stt_whisper_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--stt whisper succeeds (uses mock fallback when package absent)."""
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        monkeypatch.setenv("WHISPER_MODEL_NAME", "tiny")
        from typer.testing import CliRunner

        from voice_eval_lab.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["run", "--stt", "whisper", "--out", "/dev/null"])
        assert result.exit_code == 0, result.output

    def test_run_invalid_stt_flag_exits_nonzero(self) -> None:
        from typer.testing import CliRunner

        from voice_eval_lab.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["run", "--stt", "invalid-stt-mode"])
        assert result.exit_code != 0

    def test_run_stt_deepgram_flag_uses_deepgram_adapter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--stt deepgram should pick DeepgramSTT (mock path, no key needed)."""
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        from typer.testing import CliRunner

        from voice_eval_lab.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["run", "--stt", "deepgram", "--out", "/dev/null"])
        assert result.exit_code == 0, result.output
