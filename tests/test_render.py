"""Tests for the markdown + HTML renderers."""

from __future__ import annotations

from tests.conftest import make_conv, make_user
from voice_eval_lab.eval.metrics import render_report, render_report_html, score_run
from voice_eval_lab.models import (
    ConversationRun,
    ConversationScore,
    EvalReport,
    PipelineSpan,
    TurnLatencyStats,
    TurnRun,
)


def _empty_report() -> EvalReport:
    return EvalReport(
        n_conversations=0,
        aggregate_turn_latency=TurnLatencyStats(p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, n=0),
        aggregate_wer=0.0,
        aggregate_faithfulness=0.0,
        aggregate_barge_in_success=0.0,
        aggregate_false_trigger_rate=0.0,
        per_conversation=[],
    )


def _populated_report() -> EvalReport:
    return EvalReport(
        n_conversations=2,
        aggregate_turn_latency=TurnLatencyStats(p50_ms=200.0, p95_ms=275.0, p99_ms=275.0, n=4),
        aggregate_wer=0.05,
        aggregate_faithfulness=0.85,
        aggregate_barge_in_success=1.0,
        aggregate_false_trigger_rate=0.0,
        aggregate_barge_in_latency_p95_ms=80.0,
        aggregate_tts_first_byte_jitter_ms=12.5,
        aggregate_endpointing_accuracy=0.95,
        aggregate_llm_decisiveness=0.7,
        per_conversation=[
            ConversationScore(
                conv_id="x",
                topic="topic-x",
                turn_latency=TurnLatencyStats(p50_ms=200.0, p95_ms=200.0, p99_ms=200.0, n=2),
                transcription_wer=0.0,
                response_faithfulness=1.0,
                barge_in_success_rate=1.0,
                false_trigger_rate=0.0,
                barge_in_latency_p95_ms=80.0,
                tts_first_byte_jitter_ms=10.0,
                endpointing_accuracy=1.0,
                llm_decisiveness=0.8,
            ),
            ConversationScore(
                conv_id="y",
                topic="topic-y",
                turn_latency=TurnLatencyStats(p50_ms=275.0, p95_ms=275.0, p99_ms=275.0, n=2),
                transcription_wer=0.10,
                response_faithfulness=0.7,
                barge_in_success_rate=1.0,
                false_trigger_rate=0.0,
                barge_in_latency_p95_ms=80.0,
                tts_first_byte_jitter_ms=15.0,
                endpointing_accuracy=0.9,
                llm_decisiveness=0.6,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


class TestRenderMarkdown:
    def test_starts_with_h1(self) -> None:
        out = render_report(_empty_report())
        assert out.startswith("# Voice eval report")

    def test_contains_headline_section(self) -> None:
        out = render_report(_populated_report())
        assert "## Headline" in out

    def test_contains_per_conversation_section(self) -> None:
        out = render_report(_populated_report())
        assert "## Per conversation" in out

    def test_includes_all_new_metrics(self) -> None:
        out = render_report(_populated_report())
        for label in [
            "Turn latency",
            "Transcription WER",
            "Response faithfulness",
            "Barge-in success",
            "False-trigger rate",
            "Barge-in yield p95",
            "TTS first-byte jitter",
            "Endpointing accuracy",
            "LLM decisiveness",
        ]:
            assert label in out, f"missing {label!r}"

    def test_per_conversation_rows_present(self) -> None:
        out = render_report(_populated_report())
        assert "| x | topic-x" in out
        assert "| y | topic-y" in out

    def test_no_none_in_output(self) -> None:
        out = render_report(_populated_report())
        assert "None" not in out

    def test_ends_with_newline(self) -> None:
        out = render_report(_populated_report())
        assert out.endswith("\n")


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


class TestRenderHTML:
    def test_starts_with_doctype(self) -> None:
        out = render_report_html(_empty_report())
        assert out.startswith("<!doctype html>")

    def test_has_table_tags(self) -> None:
        out = render_report_html(_populated_report())
        assert "<table>" in out
        assert "</table>" in out

    def test_includes_headline_and_per_conversation(self) -> None:
        out = render_report_html(_populated_report())
        assert "Headline" in out
        assert "Per conversation" in out

    def test_no_none(self) -> None:
        out = render_report_html(_populated_report())
        assert "None" not in out

    def test_per_conv_rows_present(self) -> None:
        out = render_report_html(_populated_report())
        assert "<td>x</td>" in out
        assert "<td>y</td>" in out

    def test_styled_with_inline_css(self) -> None:
        out = render_report_html(_populated_report())
        assert "<style>" in out
        assert "border-collapse" in out


# ---------------------------------------------------------------------------
# HTML escaping — a crafted scores.json must not inject script or attrs
# ---------------------------------------------------------------------------


def _report_with_conv(conv_id: str = "x", topic: str = "topic-x") -> EvalReport:
    return EvalReport(
        n_conversations=1,
        aggregate_turn_latency=TurnLatencyStats(p50_ms=200.0, p95_ms=200.0, p99_ms=200.0, n=1),
        aggregate_wer=0.0,
        aggregate_faithfulness=1.0,
        aggregate_barge_in_success=1.0,
        aggregate_false_trigger_rate=0.0,
        aggregate_barge_in_latency_p95_ms=80.0,
        aggregate_tts_first_byte_jitter_ms=10.0,
        aggregate_endpointing_accuracy=1.0,
        aggregate_llm_decisiveness=1.0,
        per_conversation=[
            ConversationScore(
                conv_id=conv_id,
                topic=topic,
                turn_latency=TurnLatencyStats(p50_ms=200.0, p95_ms=200.0, p99_ms=200.0, n=1),
                transcription_wer=0.0,
                response_faithfulness=1.0,
                barge_in_success_rate=1.0,
                false_trigger_rate=0.0,
                barge_in_latency_p95_ms=80.0,
                tts_first_byte_jitter_ms=10.0,
                endpointing_accuracy=1.0,
                llm_decisiveness=1.0,
            ),
        ],
    )


class TestHTMLEscape:
    def test_html_escapes_conv_id(self) -> None:
        out = render_report_html(_report_with_conv(conv_id="<script>alert(1)</script>"))
        # Raw <script> tag must NOT appear; the entity-encoded form must.
        assert "<script>alert(1)</script>" not in out
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out

    def test_html_escapes_topic(self) -> None:
        out = render_report_html(_report_with_conv(topic="<img src=x onerror=alert(1)>"))
        assert "<img src=x onerror=alert(1)>" not in out
        assert "&lt;img src=x onerror=alert(1)&gt;" in out

    def test_html_escapes_quoted_attrs(self) -> None:
        # A double quote in conv_id should be encoded so it can't break out
        # of an attribute value.
        out = render_report_html(_report_with_conv(conv_id='evil"value'))
        assert 'evil"value' not in out
        assert "evil&quot;value" in out


# ---------------------------------------------------------------------------
# score_run + render integration on a hand-rolled fixture
# ---------------------------------------------------------------------------


class TestScoreRunRender:
    def test_score_run_then_render_no_crash(self) -> None:
        conv = make_conv([make_user("hello")], gold=["hello world"])
        run = ConversationRun(
            conv_id="c",
            topic="t",
            user_turns_played=1,
            turn_runs=[
                TurnRun(
                    user_turn_index=0,
                    transcribed_text="hello",
                    agent_reply="hello world",
                    spans=[
                        PipelineSpan(name="vad_end", started_at_ms=1000, ended_at_ms=1000),
                        PipelineSpan(name="tts_first_byte", started_at_ms=1200, ended_at_ms=1200),
                    ],
                )
            ],
        )
        report = score_run([(conv, run)])
        md = render_report(report)
        html = render_report_html(report)
        assert "## Headline" in md
        assert "<title>Voice eval report</title>" in html
