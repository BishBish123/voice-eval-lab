"""Tests for ElevenLabsTTS adapter and make_tts() factory.

All tests run WITHOUT env vars so they exercise only the mock path by
default.  Where the real HTTP path is needed, httpx is injected as a
fake module — no live ElevenLabs calls are made.
"""

from __future__ import annotations

import logging
import unittest.mock

import pytest
from typer.testing import CliRunner

import voice_eval_lab.adapters.elevenlabs as _el_mod
from voice_eval_lab.adapters import ElevenLabsTTS, make_tts
from voice_eval_lab.cli import app
from voice_eval_lab.eval.golden import default_golden_set
from voice_eval_lab.pipeline import MockLLM, MockSTT, MockTTS, VoicePipeline

runner = CliRunner()


# ---------------------------------------------------------------------------
# ElevenLabsTTS — mock path (no env var)
# ---------------------------------------------------------------------------


class TestElevenLabsTTSMockPath:
    def test_mock_flag_set_when_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        adapter = ElevenLabsTTS()
        assert adapter._mock is True

    def test_real_flag_set_when_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test-fake")
        adapter = ElevenLabsTTS()
        assert adapter._mock is False

    async def test_synthesize_returns_int_and_spans(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        adapter = ElevenLabsTTS()
        first_byte, spans = await adapter.synthesize("hello voice")
        assert isinstance(first_byte, int) and first_byte > 0
        assert len(spans) >= 1
        assert spans[0].name == "tts.synthesize"

    async def test_synthesize_conformance_to_protocol_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        adapter = ElevenLabsTTS()
        result = await adapter.synthesize("hi")
        assert isinstance(result, tuple) and len(result) == 2

    async def test_mock_path_engine_attr_is_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        adapter = ElevenLabsTTS()
        _first_byte, spans = await adapter.synthesize("test")
        assert spans[0].attrs.get("engine") == "mock"

    async def test_mock_path_delegates_to_inner_mocktts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a key the adapter must use MockTTS (not the real path)."""
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        adapter = ElevenLabsTTS()
        assert isinstance(adapter._inner, MockTTS)
        first_byte, _spans = await adapter.synthesize("delegate test")
        assert first_byte > 0


# ---------------------------------------------------------------------------
# ElevenLabsTTS — voice_id resolution
# ---------------------------------------------------------------------------


class TestElevenLabsVoiceIdResolution:
    def test_default_voice_id_used_when_nothing_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
        adapter = ElevenLabsTTS()
        assert adapter._voice_id == "21m00Tcm4TlvDq8ikWAM"

    def test_voice_id_arg_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
        adapter = ElevenLabsTTS(voice_id="custom-voice-123")
        assert adapter._voice_id == "custom-voice-123"

    def test_env_var_voice_id_used_when_no_arg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        monkeypatch.setenv("ELEVENLABS_VOICE_ID", "env-voice-456")
        adapter = ElevenLabsTTS()
        assert adapter._voice_id == "env-voice-456"

    def test_arg_takes_priority_over_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        monkeypatch.setenv("ELEVENLABS_VOICE_ID", "env-voice-456")
        adapter = ElevenLabsTTS(voice_id="arg-voice-789")
        assert adapter._voice_id == "arg-voice-789"


# ---------------------------------------------------------------------------
# ElevenLabsTTS — real path: correct request shape
# ---------------------------------------------------------------------------


class TestElevenLabsRealRequestShape:
    """Inject a fake httpx module to verify the POST shape without live calls."""

    def _make_fake_httpx(self) -> tuple[unittest.mock.MagicMock, unittest.mock.MagicMock]:
        """Return (fake_httpx_module, fake_client)."""
        fake_response = unittest.mock.MagicMock()
        fake_response.raise_for_status = unittest.mock.MagicMock()

        async def fake_aiter_bytes(chunk_size: int = 256) -> object:
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
        return fake_httpx, fake_client

    async def test_real_synthesize_sends_xi_api_key_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-el-key")
        fake_httpx, fake_client = self._make_fake_httpx()

        original = _el_mod._httpx
        _el_mod._httpx = fake_httpx  # type: ignore[assignment]
        try:
            adapter = ElevenLabsTTS()
            assert adapter._mock is False
            await adapter.synthesize("hello elevenlabs")
        finally:
            _el_mod._httpx = original  # type: ignore[assignment]

        stream_kwargs = fake_client.stream.call_args.kwargs
        headers = stream_kwargs.get("headers", {})
        assert "xi-api-key" in headers, f"Expected 'xi-api-key' header; got: {headers}"
        assert headers["xi-api-key"] == "fake-el-key"

    async def test_real_synthesize_does_not_use_bearer_auth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ElevenLabs uses xi-api-key, NOT Authorization: Bearer."""
        monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-el-key")
        fake_httpx, fake_client = self._make_fake_httpx()

        original = _el_mod._httpx
        _el_mod._httpx = fake_httpx  # type: ignore[assignment]
        try:
            adapter = ElevenLabsTTS()
            await adapter.synthesize("no bearer please")
        finally:
            _el_mod._httpx = original  # type: ignore[assignment]

        stream_kwargs = fake_client.stream.call_args.kwargs
        headers = stream_kwargs.get("headers", {})
        assert "Authorization" not in headers, (
            "ElevenLabs must NOT use Authorization header; got: " + str(headers)
        )

    async def test_real_synthesize_posts_correct_model_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-el-key")
        fake_httpx, fake_client = self._make_fake_httpx()

        original = _el_mod._httpx
        _el_mod._httpx = fake_httpx  # type: ignore[assignment]
        try:
            adapter = ElevenLabsTTS()
            await adapter.synthesize("check model id")
        finally:
            _el_mod._httpx = original  # type: ignore[assignment]

        stream_kwargs = fake_client.stream.call_args.kwargs
        body = stream_kwargs.get("json", {})
        assert body.get("model_id") == "eleven_turbo_v2_5", (
            f"Expected model_id='eleven_turbo_v2_5'; got: {body.get('model_id')!r}"
        )

    async def test_real_synthesize_uses_correct_voice_id_in_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-el-key")
        fake_httpx, fake_client = self._make_fake_httpx()

        original = _el_mod._httpx
        _el_mod._httpx = fake_httpx  # type: ignore[assignment]
        try:
            adapter = ElevenLabsTTS(voice_id="test-voice-id")
            await adapter.synthesize("voice id in url")
        finally:
            _el_mod._httpx = original  # type: ignore[assignment]

        # The URL is the second positional arg to stream("POST", <url>, ...)
        stream_args = fake_client.stream.call_args.args
        url_called = stream_args[1] if len(stream_args) > 1 else ""
        assert "test-voice-id" in url_called, (
            f"Expected voice_id in URL; got: {url_called!r}"
        )

    async def test_real_synthesize_returns_int_and_spans(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-el-key")
        fake_httpx, _fake_client = self._make_fake_httpx()

        original = _el_mod._httpx
        _el_mod._httpx = fake_httpx  # type: ignore[assignment]
        try:
            adapter = ElevenLabsTTS()
            first_byte, spans = await adapter.synthesize("test real path")
        finally:
            _el_mod._httpx = original  # type: ignore[assignment]

        assert isinstance(first_byte, int) and first_byte >= 0
        assert len(spans) == 1
        assert spans[0].name == "tts.synthesize"
        assert spans[0].attrs.get("engine") == "elevenlabs"

    async def test_real_synthesize_posts_text_in_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-el-key")
        fake_httpx, fake_client = self._make_fake_httpx()

        original = _el_mod._httpx
        _el_mod._httpx = fake_httpx  # type: ignore[assignment]
        try:
            adapter = ElevenLabsTTS()
            await adapter.synthesize("the quick brown fox")
        finally:
            _el_mod._httpx = original  # type: ignore[assignment]

        stream_kwargs = fake_client.stream.call_args.kwargs
        body = stream_kwargs.get("json", {})
        assert body.get("text") == "the quick brown fox"


# ---------------------------------------------------------------------------
# ElevenLabsTTS — API error falls back to mock
# ---------------------------------------------------------------------------


class TestElevenLabsApiErrorFallback:
    async def test_api_error_falls_back_to_mock(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-el-key")

        fake_response = unittest.mock.MagicMock()
        fake_response.raise_for_status = unittest.mock.MagicMock(
            side_effect=RuntimeError("HTTP 429 rate limit")
        )

        fake_stream_ctx = unittest.mock.AsyncMock()
        fake_stream_ctx.__aenter__ = unittest.mock.AsyncMock(return_value=fake_response)
        fake_stream_ctx.__aexit__ = unittest.mock.AsyncMock(return_value=False)

        fake_client = unittest.mock.MagicMock()
        fake_client.stream = unittest.mock.MagicMock(return_value=fake_stream_ctx)
        fake_client.__aenter__ = unittest.mock.AsyncMock(return_value=fake_client)
        fake_client.__aexit__ = unittest.mock.AsyncMock(return_value=False)

        fake_httpx = unittest.mock.MagicMock()
        fake_httpx.AsyncClient.return_value = fake_client

        original = _el_mod._httpx
        _el_mod._httpx = fake_httpx  # type: ignore[assignment]
        try:
            adapter = ElevenLabsTTS()
            with caplog.at_level(logging.WARNING, logger="voice_eval_lab.adapters.elevenlabs"):
                first_byte, spans = await adapter.synthesize("error test")
        finally:
            _el_mod._httpx = original  # type: ignore[assignment]

        # Must fall back to mock — span engine should be "mock"
        assert spans[0].attrs.get("engine") == "mock"
        assert first_byte > 0
        # Warning must have been logged
        assert any("ElevenLabsTTS" in r.message or "falling back" in r.message.lower()
                   for r in caplog.records)


# ---------------------------------------------------------------------------
# make_tts() factory — env-var dispatch
# ---------------------------------------------------------------------------


class TestMakeTtsFactory:
    def test_neither_key_returns_mock_tts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        adapter = make_tts()
        assert isinstance(adapter, MockTTS)

    def test_cartesia_key_returns_cartesia_tts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CARTESIA_API_KEY", "fake-cartesia-key")
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        from voice_eval_lab.adapters import CartesiaTTS
        adapter = make_tts()
        assert isinstance(adapter, CartesiaTTS)

    def test_elevenlabs_key_returns_elevenlabs_tts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
        monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-el-key")
        adapter = make_tts()
        assert isinstance(adapter, ElevenLabsTTS)

    def test_both_keys_set_cartesia_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When both keys are set, Cartesia takes priority (lower streaming latency)."""
        monkeypatch.setenv("CARTESIA_API_KEY", "fake-cartesia-key")
        monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-el-key")
        from voice_eval_lab.adapters import CartesiaTTS
        adapter = make_tts()
        assert isinstance(adapter, CartesiaTTS)


# ---------------------------------------------------------------------------
# ElevenLabsTTS in VoicePipeline
# ---------------------------------------------------------------------------


class TestElevenLabsInPipeline:
    async def test_elevenlabs_adapter_works_in_pipeline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        pipeline = VoicePipeline(stt=MockSTT(), llm=MockLLM(), tts=ElevenLabsTTS())
        conv = default_golden_set()[0]
        run = await pipeline.run(conv)
        assert run.user_turns_played > 0
        for tr in run.turn_runs:
            assert "vad_end" in {s.name for s in tr.spans}


# ---------------------------------------------------------------------------
# CLI --tts flag
# ---------------------------------------------------------------------------


class TestCliTtsFlag:
    def test_tts_mock_flag_works(self, tmp_path: pytest.TempPathFactory) -> None:
        out = tmp_path / "REPORT.md"
        result = runner.invoke(app, ["run", "--out", str(out), "--tts", "mock"])
        assert result.exit_code == 0, result.output

    def test_tts_auto_flag_works(self, tmp_path: pytest.TempPathFactory) -> None:
        out = tmp_path / "REPORT.md"
        result = runner.invoke(app, ["run", "--out", str(out), "--tts", "auto"])
        assert result.exit_code == 0, result.output

    def test_tts_elevenlabs_flag_works_without_key(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--tts elevenlabs without ELEVENLABS_API_KEY should mock-fallback and succeed."""
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        out = tmp_path / "REPORT.md"
        result = runner.invoke(app, ["run", "--out", str(out), "--tts", "elevenlabs"])
        assert result.exit_code == 0, result.output

    def test_tts_cartesia_flag_works_without_key(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--tts cartesia without CARTESIA_API_KEY should mock-fallback and succeed."""
        monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
        out = tmp_path / "REPORT.md"
        result = runner.invoke(app, ["run", "--out", str(out), "--tts", "cartesia"])
        assert result.exit_code == 0, result.output

    def test_tts_invalid_flag_exits_nonzero(self, tmp_path: pytest.TempPathFactory) -> None:
        out = tmp_path / "REPORT.md"
        result = runner.invoke(app, ["run", "--out", str(out), "--tts", "openai"])
        assert result.exit_code != 0
        assert "openai" in result.output.lower() or "tts" in result.output.lower()
