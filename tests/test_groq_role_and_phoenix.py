"""Tests for Fix 2 (GroqLLM role mapping) and Fix 4 (Phoenix no-op exporter).

Fix 2 — TurnRole.AGENT -> "assistant"
--------------------------------------
TurnRole.AGENT.value == "agent" but the OpenAI/Groq API expects "assistant".
_real_reply must use _OPENAI_ROLE lookup, not t.role.value, when building the
messages list.

Fix 4 — Phoenix OTLP exporter no-op behaviour
----------------------------------------------
* The module must be importable without opentelemetry installed.
* When OTEL_EXPORTER_OTLP_ENDPOINT is unset/empty, export_spans is a no-op.
"""

from __future__ import annotations

import builtins
import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import voice_eval_lab.observability.phoenix as phoenix_mod
from voice_eval_lab.adapters.groq import _OPENAI_ROLE, GroqLLM
from voice_eval_lab.models import PipelineSpan, Turn, TurnRole
from voice_eval_lab.observability.phoenix import export_spans

# ---------------------------------------------------------------------------
# Fix 2 — role mapping unit tests
# ---------------------------------------------------------------------------


class TestOpenAIRoleMapping:
    """_OPENAI_ROLE must map TurnRole values to what the OpenAI/Groq API expects."""

    def test_user_maps_to_user(self) -> None:
        assert _OPENAI_ROLE[TurnRole.USER] == "user"

    def test_agent_maps_to_assistant(self) -> None:
        # Core of the bug: TurnRole.AGENT.value == "agent", NOT "assistant".
        assert TurnRole.AGENT.value == "agent", (
            "Sanity check: if this fails the enum changed and the test needs updating"
        )
        assert _OPENAI_ROLE[TurnRole.AGENT] == "assistant"

    def test_mapping_covers_all_roles(self) -> None:
        """Every TurnRole must have an entry so no KeyError on future roles."""
        for role in TurnRole:
            assert role in _OPENAI_ROLE, f"TurnRole.{role.name} missing from _OPENAI_ROLE"


class TestGroqRealReplyMessageList:
    """_real_reply must build messages with "user"/"assistant", not "agent"."""

    async def test_message_roles_are_openai_compatible(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "sk-test-fake")

        captured_payload: dict | None = None

        async def fake_post(url: str, **kwargs: object) -> MagicMock:
            nonlocal captured_payload
            captured_payload = kwargs.get("json")  # type: ignore[assignment]
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "mock assistant reply"}}]
            }
            return mock_resp

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=fake_post)

        with patch("voice_eval_lab.adapters.groq._httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client

            adapter = GroqLLM()
            history = [
                Turn(role=TurnRole.USER, text="hello", started_at_ms=0, ended_at_ms=1000),
                Turn(role=TurnRole.AGENT, text="hi there", started_at_ms=1000, ended_at_ms=2000),
                Turn(role=TurnRole.USER, text="tell me more", started_at_ms=2000, ended_at_ms=3000),
            ]
            _text, _spans = await adapter._real_reply(history, "what is raft?", [])

        assert captured_payload is not None
        messages = captured_payload["messages"]

        # history roles
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"  # was "agent" before the fix
        assert messages[2]["role"] == "user"
        # appended user turn
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "what is raft?"

        # no "agent" role must appear anywhere — the API would reject it
        roles = [m["role"] for m in messages]
        assert "agent" not in roles, f"'agent' role leaked into API payload: {roles}"

    async def test_message_list_content_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Role fix must not corrupt message content."""
        monkeypatch.setenv("GROQ_API_KEY", "sk-test-fake")

        captured: dict | None = None

        async def fake_post(url: str, **kwargs: object) -> MagicMock:
            nonlocal captured
            captured = kwargs.get("json")  # type: ignore[assignment]
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
            return resp

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=fake_post)

        with patch("voice_eval_lab.adapters.groq._httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            adapter = GroqLLM()
            history = [
                Turn(role=TurnRole.AGENT, text="I am the agent", started_at_ms=0, ended_at_ms=500),
            ]
            await adapter._real_reply(history, "follow-up", [])

        assert captured is not None
        assert captured["messages"][0]["content"] == "I am the agent"
        assert captured["messages"][1]["content"] == "follow-up"


# ---------------------------------------------------------------------------
# Fix 4 — Phoenix no-op exporter
# ---------------------------------------------------------------------------


class TestPhoenixExporterImportable:
    """The module must be importable even without opentelemetry installed."""

    def test_module_importable(self) -> None:
        mod = importlib.import_module("voice_eval_lab.observability.phoenix")
        assert hasattr(mod, "export_spans")

    def test_export_spans_callable(self) -> None:
        assert callable(export_spans)


class TestPhoenixNoOpWhenEnvUnset:
    """export_spans must be a no-op when OTEL_EXPORTER_OTLP_ENDPOINT is unset."""

    def test_no_op_when_endpoint_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

        spans = [
            PipelineSpan(name="llm.reply", started_at_ms=0, ended_at_ms=100, attrs={"k": "v"})
        ]
        # Must not raise; no import of opentelemetry attempted.
        result = export_spans(spans, conv_id="test-conv")
        assert result is None

    def test_no_op_when_endpoint_empty_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")

        spans = [PipelineSpan(name="stt.transcribe", started_at_ms=0, ended_at_ms=80)]
        result = export_spans(spans, conv_id="")
        assert result is None

    def test_no_op_with_empty_span_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

        result = export_spans([], conv_id="no-spans")
        assert result is None

    def test_no_exception_when_endpoint_set_but_otel_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When endpoint is set but OTel is absent, a logger.warning is emitted (no raise)."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

        spans = [PipelineSpan(name="llm.reply", started_at_ms=0, ended_at_ms=50)]

        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name.startswith("opentelemetry"):
                raise ImportError(f"Mocked missing: {name}")
            return real_import(name, *args, **kwargs)  # type: ignore[call-arg]

        # Force _OTEL_AVAILABLE to False so the warning path is exercised.
        original_available = phoenix_mod._OTEL_AVAILABLE
        try:
            phoenix_mod._OTEL_AVAILABLE = False
            with patch("builtins.__import__", side_effect=mock_import):
                # Assert no exception is raised when OTel is absent; the function
                # logs a warning via logger.warning (not warnings.warn) and returns.
                try:
                    phoenix_mod.export_spans(spans, conv_id="missing-otel")
                except Exception as exc:
                    pytest.fail(f"export_spans raised unexpectedly: {exc}")
        finally:
            phoenix_mod._OTEL_AVAILABLE = original_available
