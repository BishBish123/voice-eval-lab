"""Unit tests for serve_on_livekit.

- No LiveKit cloud connection is made.
- livekit-agents is mocked via unittest.mock when not installed.
- Tests verify the no-credentials path (logged warning + early return) and
  the mocked call sequence (WorkerOptions constructed + logged).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from voice_eval_lab.pipecat.livekit import serve_on_livekit
from voice_eval_lab.pipecat.pipeline import build_pipeline
from voice_eval_lab.pipeline import MockLLM, MockSTT, MockTTS


@pytest.fixture()
def mock_pipeline() -> object:
    return build_pipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())


class TestServeOnLivekitMissingCredentials:
    def test_no_credentials_logs_warning_and_returns(
        self, mock_pipeline: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Missing env-var credentials → warning logged, no exception raised."""
        with (
            patch.dict(
                "os.environ",
                {},
                clear=False,
            ),
            patch(
                "voice_eval_lab.pipecat.livekit.os.environ.get",
                side_effect=lambda k, default="": "",
            ),
            caplog.at_level(logging.WARNING, logger="voice_eval_lab.pipecat.livekit"),
        ):
            serve_on_livekit(mock_pipeline, room_name="test-room")

        assert any("missing credentials" in r.message for r in caplog.records)

    def test_empty_url_logs_warning(
        self, mock_pipeline: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            patch(
                "voice_eval_lab.pipecat.livekit.os.environ.get",
                side_effect=lambda k, default="": {
                    "LIVEKIT_API_KEY": "key",
                    "LIVEKIT_API_SECRET": "secret",
                }.get(k, ""),
            ),
            caplog.at_level(logging.WARNING, logger="voice_eval_lab.pipecat.livekit"),
        ):
            serve_on_livekit(mock_pipeline, room_name="test-room")

        assert any("missing credentials" in r.message for r in caplog.records)

    def test_no_credentials_does_not_raise(self, mock_pipeline: object) -> None:
        """Must not raise even when all credentials are missing."""
        with patch(
            "voice_eval_lab.pipecat.livekit.os.environ.get",
            side_effect=lambda k, default="": "",
        ):
            serve_on_livekit(mock_pipeline, room_name="test-room")  # no exception

    def test_explicit_empty_credentials_returns_early(
        self, mock_pipeline: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="voice_eval_lab.pipecat.livekit"):
            serve_on_livekit(
                mock_pipeline,
                room_name="test-room",
                livekit_url="",
                api_key="",
                api_secret="",
            )
        assert any("missing credentials" in r.message for r in caplog.records)


class TestServeOnLivekitPackageNotInstalled:
    def test_missing_package_logs_warning_and_returns(
        self, mock_pipeline: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When livekit-agents is not installed → warning logged, no exception."""
        with (
            patch("voice_eval_lab.pipecat.livekit._LIVEKIT_AGENTS_AVAILABLE", False),
            caplog.at_level(logging.WARNING, logger="voice_eval_lab.pipecat.livekit"),
        ):
            serve_on_livekit(
                mock_pipeline,
                room_name="test-room",
                livekit_url="wss://example.livekit.cloud",
                api_key="my-key",
                api_secret="my-secret",
            )

        assert any("livekit-agents" in r.message for r in caplog.records)


class TestServeOnLivekitMockedSDK:
    def test_with_mocked_sdk_calls_worker_options(
        self, mock_pipeline: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With mocked livekit-agents, WorkerOptions is constructed correctly."""
        mock_lk = MagicMock()
        mock_worker_options_instance = MagicMock()
        mock_lk.WorkerOptions.return_value = mock_worker_options_instance

        with (
            patch("voice_eval_lab.pipecat.livekit._LIVEKIT_AGENTS_AVAILABLE", True),
            patch("voice_eval_lab.pipecat.livekit._lk_agents", mock_lk),
            caplog.at_level(logging.INFO, logger="voice_eval_lab.pipecat.livekit"),
        ):
            serve_on_livekit(
                mock_pipeline,
                room_name="my-room",
                livekit_url="wss://my-livekit.example.com",
                api_key="test-key",
                api_secret="test-secret",
            )

        # WorkerOptions must have been called exactly once.
        mock_lk.WorkerOptions.assert_called_once()
        call_kwargs = mock_lk.WorkerOptions.call_args.kwargs
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["api_secret"] == "test-secret"  # noqa: S105
        assert call_kwargs["ws_url"] == "wss://my-livekit.example.com"
        assert callable(call_kwargs["entrypoint_fnc"])

    def test_with_mocked_sdk_logs_room_name(
        self, mock_pipeline: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        mock_lk = MagicMock()
        mock_lk.WorkerOptions.return_value = MagicMock()

        with (
            patch("voice_eval_lab.pipecat.livekit._LIVEKIT_AGENTS_AVAILABLE", True),
            patch("voice_eval_lab.pipecat.livekit._lk_agents", mock_lk),
            caplog.at_level(logging.INFO, logger="voice_eval_lab.pipecat.livekit"),
        ):
            serve_on_livekit(
                mock_pipeline,
                room_name="eval-room",
                livekit_url="wss://host",
                api_key="k",
                api_secret="s",
            )

        assert any("eval-room" in r.message for r in caplog.records)

    def test_entrypoint_is_async_callable(self, mock_pipeline: object) -> None:
        """The entrypoint passed to WorkerOptions must be an async callable."""
        import inspect

        from voice_eval_lab.pipecat.livekit import _make_entrypoint

        ep = _make_entrypoint(mock_pipeline)
        assert callable(ep)
        assert inspect.iscoroutinefunction(ep)

    @pytest.mark.asyncio
    async def test_entrypoint_calls_connect(self, mock_pipeline: object) -> None:
        """The entrypoint coroutine should call ctx.connect() on the job context."""
        from voice_eval_lab.pipecat.livekit import _make_entrypoint

        ep = _make_entrypoint(mock_pipeline)
        mock_ctx = MagicMock()
        mock_ctx.connect = MagicMock(return_value=asyncio_coroutine_result())
        await ep(mock_ctx)
        mock_ctx.connect.assert_called_once()


def asyncio_coroutine_result() -> object:
    """Return an awaitable that resolves immediately to None."""

    async def _noop() -> None:
        pass

    return _noop()
