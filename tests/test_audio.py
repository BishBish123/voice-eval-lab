"""Tests for the audio fixture infrastructure.

Covers:
- FilesystemAudioStore round-trip (write + read)
- WAV header validation (rejects non-WAV bytes)
- get_audio returns None for missing keys
- SilenceFixtureGenerator produces valid WAV bytes
- VoicePipeline.run(audio_store=...) attaches audio_bytes to Turn before STT
- CLI: voice-eval audio populate-silence writes the correct file
- CLI: voice-eval audio list returns all populated keys
- CLI: voice-eval audio import copies a WAV into the fixture tree
- CLI: voice-eval run --audio-fixtures runs to completion
"""

from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from voice_eval_lab.audio import FilesystemAudioStore, SilenceFixtureGenerator
from voice_eval_lab.audio.protocol import AudioFixtureStore
from voice_eval_lab.cli import app
from voice_eval_lab.eval.golden import default_golden_set
from voice_eval_lab.models import Conversation, Turn, TurnRole
from voice_eval_lab.pipeline import MockLLM, MockSTT, MockTTS, VoicePipeline

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_silence_wav(duration_ms: int = 100) -> bytes:
    """Return a minimal valid WAV (silence) for use in tests."""
    gen = SilenceFixtureGenerator()
    return gen.generate(duration_ms)


def _is_valid_wav(data: bytes) -> bool:
    """Return True if *data* can be opened by stdlib ``wave``."""
    try:
        buf = io.BytesIO(data)
        with wave.open(buf, "rb") as wf:
            _ = wf.getnframes()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# FilesystemAudioStore tests
# ---------------------------------------------------------------------------


class TestFilesystemAudioStoreRoundTrip:
    def test_write_then_read_returns_same_bytes(self, tmp_path: Path) -> None:
        store = FilesystemAudioStore(tmp_path)
        wav = _make_silence_wav(200)
        store.add_audio("conv-1", 0, wav)
        result = store.get_audio("conv-1", 0)
        assert result == wav

    def test_file_created_at_expected_path(self, tmp_path: Path) -> None:
        store = FilesystemAudioStore(tmp_path)
        wav = _make_silence_wav(100)
        store.add_audio("my-conv", 3, wav)
        expected = tmp_path / "my-conv" / "turn-03.wav"
        assert expected.exists()
        assert expected.read_bytes() == wav

    def test_add_creates_parent_directories(self, tmp_path: Path) -> None:
        root = tmp_path / "deep" / "nested"
        store = FilesystemAudioStore(root)
        wav = _make_silence_wav(50)
        store.add_audio("x", 0, wav)
        assert (root / "x" / "turn-00.wav").exists()

    def test_get_returns_none_for_missing_key(self, tmp_path: Path) -> None:
        store = FilesystemAudioStore(tmp_path)
        assert store.get_audio("no-such-conv", 0) is None

    def test_get_returns_none_when_root_absent(self, tmp_path: Path) -> None:
        store = FilesystemAudioStore(tmp_path / "nonexistent")
        assert store.get_audio("x", 0) is None

    def test_list_keys_empty_when_no_fixtures(self, tmp_path: Path) -> None:
        store = FilesystemAudioStore(tmp_path)
        assert store.list_keys() == []

    def test_list_keys_returns_all_stored(self, tmp_path: Path) -> None:
        store = FilesystemAudioStore(tmp_path)
        wav = _make_silence_wav(100)
        store.add_audio("conv-a", 0, wav)
        store.add_audio("conv-a", 1, wav)
        store.add_audio("conv-b", 0, wav)
        keys = store.list_keys()
        assert ("conv-a", 0) in keys
        assert ("conv-a", 1) in keys
        assert ("conv-b", 0) in keys
        assert len(keys) == 3

    def test_list_keys_sorted(self, tmp_path: Path) -> None:
        store = FilesystemAudioStore(tmp_path)
        wav = _make_silence_wav(100)
        store.add_audio("z-conv", 2, wav)
        store.add_audio("a-conv", 0, wav)
        store.add_audio("a-conv", 1, wav)
        keys = store.list_keys()
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# WAV header validation tests
# ---------------------------------------------------------------------------


class TestWavHeaderValidation:
    def test_rejects_non_wav_bytes(self, tmp_path: Path) -> None:
        store = FilesystemAudioStore(tmp_path)
        with pytest.raises(ValueError, match="RIFF"):
            store.add_audio("x", 0, b"NOT_A_WAV_FILE_AT_ALL")

    def test_rejects_empty_bytes(self, tmp_path: Path) -> None:
        store = FilesystemAudioStore(tmp_path)
        with pytest.raises(ValueError, match="too short"):
            store.add_audio("x", 0, b"")

    def test_rejects_riff_without_wave(self, tmp_path: Path) -> None:
        # Build 12 bytes with RIFF header but wrong format tag.
        bad = b"RIFF\x00\x00\x00\x00AVI "
        store = FilesystemAudioStore(tmp_path)
        with pytest.raises(ValueError, match="WAVE"):
            store.add_audio("x", 0, bad)

    def test_accepts_valid_wav(self, tmp_path: Path) -> None:
        store = FilesystemAudioStore(tmp_path)
        wav = _make_silence_wav(100)
        # Should not raise.
        store.add_audio("x", 0, wav)


# ---------------------------------------------------------------------------
# SilenceFixtureGenerator tests
# ---------------------------------------------------------------------------


class TestSilenceFixtureGenerator:
    def test_output_is_valid_wav(self) -> None:
        gen = SilenceFixtureGenerator()
        data = gen.generate(200)
        assert _is_valid_wav(data)

    def test_output_starts_with_riff_wave(self) -> None:
        gen = SilenceFixtureGenerator()
        data = gen.generate(100)
        assert data[:4] == b"RIFF"
        assert data[8:12] == b"WAVE"

    def test_duration_maps_to_frame_count(self) -> None:
        gen = SilenceFixtureGenerator(sample_rate=16_000)
        data = gen.generate(500)
        buf = io.BytesIO(data)
        with wave.open(buf, "rb") as wf:
            # 500ms @ 16 kHz = 8000 frames
            assert wf.getnframes() == 8000

    def test_zero_duration_is_valid(self) -> None:
        gen = SilenceFixtureGenerator()
        data = gen.generate(0)
        assert _is_valid_wav(data)

    def test_negative_duration_raises(self) -> None:
        gen = SilenceFixtureGenerator()
        with pytest.raises(ValueError, match=">= 0"):
            gen.generate(-1)

    def test_custom_sample_rate(self) -> None:
        gen = SilenceFixtureGenerator(sample_rate=8_000)
        data = gen.generate(1000)
        buf = io.BytesIO(data)
        with wave.open(buf, "rb") as wf:
            assert wf.getframerate() == 8_000
            assert wf.getnframes() == 8_000  # 1000ms @ 8 kHz

    def test_wav_passable_to_filesystem_store(self, tmp_path: Path) -> None:
        gen = SilenceFixtureGenerator()
        store = FilesystemAudioStore(tmp_path)
        wav = gen.generate(300)
        store.add_audio("test-conv", 0, wav)
        assert store.get_audio("test-conv", 0) == wav


# ---------------------------------------------------------------------------
# Protocol conformance test
# ---------------------------------------------------------------------------


class TestAudioFixtureStoreProtocol:
    def test_filesystem_store_conforms_to_protocol(self, tmp_path: Path) -> None:
        store = FilesystemAudioStore(tmp_path)
        # runtime_checkable Protocol check.
        assert isinstance(store, AudioFixtureStore)


# ---------------------------------------------------------------------------
# VoicePipeline audio_store integration tests
# ---------------------------------------------------------------------------


class TestVoicePipelineAudioStore:
    async def test_audio_bytes_attached_to_turn_when_store_provided(self) -> None:
        """When audio_store returns bytes, Turn.audio_bytes must be set before STT."""
        received_turns: list[Turn] = []

        class CapturingSTT:
            async def transcribe(self, turn: Turn) -> tuple[str, list[Any]]:
                received_turns.append(turn)
                return turn.text, []

        wav = _make_silence_wav(100)

        class MockStore:
            def get_audio(self, conv_id: str, turn_index: int) -> bytes | None:
                return wav

            def add_audio(self, conv_id: str, turn_index: int, wav_bytes: bytes) -> None:
                pass

            def list_keys(self) -> list[tuple[str, int]]:
                return []

        conv = Conversation(
            conv_id="test-audio",
            topic="audio fixture test",
            turns=[
                Turn(role=TurnRole.USER, text="hello", started_at_ms=0, ended_at_ms=500)
            ],
            gold_facts=[],
        )
        pipeline = VoicePipeline(stt=CapturingSTT(), llm=MockLLM(), tts=MockTTS())  # type: ignore[arg-type]
        await pipeline.run(conv, audio_store=MockStore())  # type: ignore[arg-type]

        assert len(received_turns) == 1
        assert received_turns[0].audio_bytes == wav

    async def test_audio_bytes_none_when_store_returns_none(self) -> None:
        """When audio_store returns None, Turn.audio_bytes stays None."""
        received_turns: list[Turn] = []

        class CapturingSTT:
            async def transcribe(self, turn: Turn) -> tuple[str, list[Any]]:
                received_turns.append(turn)
                return turn.text, []

        class EmptyStore:
            def get_audio(self, conv_id: str, turn_index: int) -> bytes | None:
                return None

            def add_audio(self, conv_id: str, turn_index: int, wav_bytes: bytes) -> None:
                pass

            def list_keys(self) -> list[tuple[str, int]]:
                return []

        conv = Conversation(
            conv_id="no-audio",
            topic="no audio",
            turns=[
                Turn(role=TurnRole.USER, text="hello", started_at_ms=0, ended_at_ms=500)
            ],
            gold_facts=[],
        )
        pipeline = VoicePipeline(stt=CapturingSTT(), llm=MockLLM(), tts=MockTTS())  # type: ignore[arg-type]
        await pipeline.run(conv, audio_store=EmptyStore())  # type: ignore[arg-type]

        assert received_turns[0].audio_bytes is None

    async def test_no_audio_store_leaves_turn_unchanged(self) -> None:
        """Without audio_store, Turn.audio_bytes is None (backward-compat)."""
        received_turns: list[Turn] = []

        class CapturingSTT:
            async def transcribe(self, turn: Turn) -> tuple[str, list[Any]]:
                received_turns.append(turn)
                return turn.text, []

        conv = Conversation(
            conv_id="no-store",
            topic="no store",
            turns=[
                Turn(role=TurnRole.USER, text="hi", started_at_ms=0, ended_at_ms=300)
            ],
            gold_facts=[],
        )
        pipeline = VoicePipeline(stt=CapturingSTT(), llm=MockLLM(), tts=MockTTS())  # type: ignore[arg-type]
        await pipeline.run(conv)

        assert received_turns[0].audio_bytes is None

    async def test_golden_set_runs_with_filesystem_audio_store(self, tmp_path: Path) -> None:
        """Full pipeline run with FilesystemAudioStore over the golden set."""
        store = FilesystemAudioStore(tmp_path)
        gen = SilenceFixtureGenerator()
        # Populate fixtures for first golden conversation (postgres-replication, 2 user turns).
        conv = default_golden_set()[0]
        for i in range(2):
            store.add_audio(conv.conv_id, i, gen.generate(200))

        pipeline = VoicePipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())
        run = await pipeline.run(conv, audio_store=store)
        assert run.user_turns_played == 2
        assert len(run.turn_runs) == 2


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestAudioCLI:
    def test_populate_silence_writes_file(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "audio",
                "populate-silence",
                "--conv-id", "test-conv",
                "--turn", "0",
                "--duration-ms", "100",
                "--root", str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
        wav_path = tmp_path / "test-conv" / "turn-00.wav"
        assert wav_path.exists()
        assert _is_valid_wav(wav_path.read_bytes())

    def test_populate_silence_writes_to_correct_turn_index(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "audio",
                "populate-silence",
                "--conv-id", "my-conv",
                "--turn", "5",
                "--root", str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "my-conv" / "turn-05.wav").exists()

    def test_list_returns_all_keys(self, tmp_path: Path) -> None:
        gen = SilenceFixtureGenerator()
        store = FilesystemAudioStore(tmp_path)
        store.add_audio("conv-a", 0, gen.generate(100))
        store.add_audio("conv-b", 2, gen.generate(100))

        result = runner.invoke(app, ["audio", "list", "--root", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "conv-a" in result.output
        assert "conv-b" in result.output
        assert "2 fixture(s)" in result.output

    def test_list_empty_root(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["audio", "list", "--root", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "no fixtures" in result.output

    def test_import_copies_wav(self, tmp_path: Path) -> None:
        # Write a real WAV to a source path.
        src = tmp_path / "source.wav"
        wav = _make_silence_wav(200)
        src.write_bytes(wav)

        dest_root = tmp_path / "fixtures"
        result = runner.invoke(
            app,
            [
                "audio",
                "import",
                str(src),
                "--conv-id", "imported-conv",
                "--turn", "1",
                "--root", str(dest_root),
            ],
        )
        assert result.exit_code == 0, result.output
        dest = dest_root / "imported-conv" / "turn-01.wav"
        assert dest.exists()
        assert dest.read_bytes() == wav

    def test_import_rejects_non_wav(self, tmp_path: Path) -> None:
        src = tmp_path / "bad.wav"
        src.write_bytes(b"not a wav file at all")
        result = runner.invoke(
            app,
            [
                "audio",
                "import",
                str(src),
                "--conv-id", "x",
                "--turn", "0",
                "--root", str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0
        assert "invalid WAV" in result.output

    def test_import_rejects_missing_source(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "audio",
                "import",
                str(tmp_path / "nonexistent.wav"),
                "--conv-id", "x",
                "--turn", "0",
                "--root", str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_run_with_audio_fixtures_succeeds(self, tmp_path: Path) -> None:
        """voice-eval run --audio-fixtures <dir> completes successfully."""
        out = tmp_path / "report.md"
        # Pre-populate silence fixtures so the store has something to return.
        gen = SilenceFixtureGenerator()
        store = FilesystemAudioStore(tmp_path / "audio")
        for conv in default_golden_set():
            from voice_eval_lab.models import TurnRole as TR

            user_turns = [t for t in conv.turns if t.role == TR.USER]
            for i in range(len(user_turns)):
                store.add_audio(conv.conv_id, i, gen.generate(100))

        result = runner.invoke(
            app,
            [
                "run",
                "--out", str(out),
                "--audio-fixtures", str(tmp_path / "audio"),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
