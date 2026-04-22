"""Eval metric tests."""

from __future__ import annotations

import pytest

from voice_eval_lab.eval.metrics import (
    barge_in_success_rate,
    false_trigger_rate,
    response_faithfulness,
    score_run,
    transcription_wer,
    turn_latency_stats,
)
from voice_eval_lab.models import (
    Conversation,
    ConversationRun,
    PipelineSpan,
    Turn,
    TurnRole,
    TurnRun,
)


def _conv(turns: list[Turn], gold: list[str] | None = None, conv_id: str = "c") -> Conversation:
    return Conversation(conv_id=conv_id, topic="t", turns=turns, gold_facts=gold or [])


def _user(text: str, start: int = 0, end: int = 1000, interrupted: bool = False) -> Turn:
    return Turn(
        role=TurnRole.USER, text=text, started_at_ms=start, ended_at_ms=end, interrupted=interrupted
    )


def _turn_run(
    *,
    transcribed: str,
    reply: str,
    vad_end_ms: int,
    first_byte_ms: int,
    interrupted: bool = False,
    false_trigger: bool = False,
) -> TurnRun:
    return TurnRun(
        user_turn_index=0,
        transcribed_text=transcribed,
        agent_reply=reply,
        interrupted=interrupted,
        false_trigger=false_trigger,
        spans=[
            PipelineSpan(name="vad_end", started_at_ms=vad_end_ms, ended_at_ms=vad_end_ms),
            PipelineSpan(
                name="tts_first_byte", started_at_ms=first_byte_ms, ended_at_ms=first_byte_ms
            ),
        ],
    )


def _run(turn_runs: list[TurnRun], conv_id: str = "c") -> ConversationRun:
    return ConversationRun(
        conv_id=conv_id, topic="t", user_turns_played=len(turn_runs), turn_runs=turn_runs
    )


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


class TestTurnLatencyStats:
    def test_known_values(self) -> None:
        runs = [
            _turn_run(transcribed="x", reply="y", vad_end_ms=1000, first_byte_ms=1100),
            _turn_run(transcribed="x", reply="y", vad_end_ms=2000, first_byte_ms=2300),
            _turn_run(transcribed="x", reply="y", vad_end_ms=3000, first_byte_ms=3500),
        ]
        stats = turn_latency_stats(runs)
        assert stats.n == 3
        assert stats.p50_ms in (100, 300)  # depends on percentile rounding


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------


class TestTranscriptionWER:
    def test_perfect_transcription(self) -> None:
        conv = _conv([_user("hello world")])
        run = _run(
            [_turn_run(transcribed="hello world", reply="x", vad_end_ms=0, first_byte_ms=10)]
        )
        assert transcription_wer(conv, run) == 0.0

    def test_one_substitution(self) -> None:
        conv = _conv([_user("hello world")])
        run = _run([_turn_run(transcribed="hello WERR", reply="x", vad_end_ms=0, first_byte_ms=10)])
        assert transcription_wer(conv, run) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


class TestFaithfulness:
    def test_quoted_fact_scores_one(self) -> None:
        conv = _conv([_user("u")], gold=["the answer is 42"])
        run = _run(
            [_turn_run(transcribed="u", reply="The answer is 42!", vad_end_ms=0, first_byte_ms=10)]
        )
        assert response_faithfulness(conv, run) == 1.0

    def test_no_match_scores_zero(self) -> None:
        conv = _conv([_user("u")], gold=["the answer is 42"])
        run = _run([_turn_run(transcribed="u", reply="not sure", vad_end_ms=0, first_byte_ms=10)])
        assert response_faithfulness(conv, run) == 0.0

    def test_empty_gold_returns_one(self) -> None:
        conv = _conv([_user("u")], gold=[])
        run = _run([_turn_run(transcribed="u", reply="anything", vad_end_ms=0, first_byte_ms=10)])
        assert response_faithfulness(conv, run) == 1.0

    def test_false_triggers_excluded_from_faithfulness(self) -> None:
        conv = _conv([_user("u")], gold=["yes"])
        run = _run(
            [
                _turn_run(transcribed="u", reply="yes here", vad_end_ms=0, first_byte_ms=10),
                _turn_run(
                    transcribed="", reply="???", vad_end_ms=0, first_byte_ms=10, false_trigger=True
                ),
            ]
        )
        # Only the first reply counts; should be 1.0.
        assert response_faithfulness(conv, run) == 1.0


# ---------------------------------------------------------------------------
# Barge-in
# ---------------------------------------------------------------------------


class TestBargeIn:
    def test_no_interrupted_turns_returns_one(self) -> None:
        conv = _conv([_user("hi")])
        run = _run([_turn_run(transcribed="hi", reply="hi", vad_end_ms=0, first_byte_ms=10)])
        assert barge_in_success_rate(conv, run) == 1.0

    def test_interrupted_turn_yielded(self) -> None:
        conv = _conv([_user("interrupted", interrupted=True)])
        # An interrupted turn is "successful" iff a barge_in.yield span exists
        # within budget. tts_first_byte alone is not enough.
        run = ConversationRun(
            conv_id="c",
            topic="t",
            user_turns_played=1,
            turn_runs=[
                TurnRun(
                    user_turn_index=0,
                    transcribed_text="x",
                    agent_reply="ok",
                    interrupted=True,
                    spans=[
                        PipelineSpan(name="vad_end", started_at_ms=0, ended_at_ms=0),
                        PipelineSpan(name="tts_first_byte", started_at_ms=10, ended_at_ms=10),
                        PipelineSpan(name="barge_in.yield", started_at_ms=0, ended_at_ms=80),
                    ],
                )
            ],
        )
        assert barge_in_success_rate(conv, run) == 1.0

    def test_interrupted_turn_no_yield_span(self) -> None:
        conv = _conv([_user("interrupted", interrupted=True)])
        # turn run with no barge_in.yield span — pipeline never yielded.
        # tts_first_byte being present is irrelevant for barge-in success.
        run = ConversationRun(
            conv_id="c",
            topic="t",
            user_turns_played=1,
            turn_runs=[
                TurnRun(
                    user_turn_index=0,
                    transcribed_text="x",
                    agent_reply="...",
                    interrupted=True,
                    spans=[
                        PipelineSpan(name="tts_first_byte", started_at_ms=10, ended_at_ms=10),
                    ],  # no barge_in.yield
                )
            ],
        )
        assert barge_in_success_rate(conv, run) == 0.0

    def test_no_yield_span_fails_barge_in(self) -> None:
        # Interrupted turn that has tts_first_byte but no barge_in.yield —
        # the agent reached TTS but never actually yielded to the user.
        conv = _conv([_user("interrupted", interrupted=True)])
        run = ConversationRun(
            conv_id="c",
            topic="t",
            user_turns_played=1,
            turn_runs=[
                TurnRun(
                    user_turn_index=0,
                    transcribed_text="x",
                    agent_reply="ok",
                    interrupted=True,
                    spans=[
                        PipelineSpan(name="vad_end", started_at_ms=0, ended_at_ms=0),
                        PipelineSpan(name="tts_first_byte", started_at_ms=10, ended_at_ms=10),
                    ],
                )
            ],
        )
        assert barge_in_success_rate(conv, run) == 0.0

    def test_over_budget_yield_fails(self) -> None:
        # yield span exists but blows the default 200ms budget.
        conv = _conv([_user("interrupted", interrupted=True)])
        run = ConversationRun(
            conv_id="c",
            topic="t",
            user_turns_played=1,
            turn_runs=[
                TurnRun(
                    user_turn_index=0,
                    transcribed_text="x",
                    agent_reply="ok",
                    interrupted=True,
                    spans=[
                        PipelineSpan(name="barge_in.yield", started_at_ms=0, ended_at_ms=350),
                    ],
                )
            ],
        )
        assert barge_in_success_rate(conv, run) == 0.0

    def test_under_budget_yield_succeeds(self) -> None:
        conv = _conv([_user("interrupted", interrupted=True)])
        run = ConversationRun(
            conv_id="c",
            topic="t",
            user_turns_played=1,
            turn_runs=[
                TurnRun(
                    user_turn_index=0,
                    transcribed_text="x",
                    agent_reply="ok",
                    interrupted=True,
                    spans=[
                        PipelineSpan(name="barge_in.yield", started_at_ms=0, ended_at_ms=120),
                    ],
                )
            ],
        )
        assert barge_in_success_rate(conv, run) == 1.0

    def test_custom_budget(self) -> None:
        # 250ms yield: passes a 300ms budget, fails a 100ms budget.
        conv = _conv([_user("interrupted", interrupted=True)])
        run = ConversationRun(
            conv_id="c",
            topic="t",
            user_turns_played=1,
            turn_runs=[
                TurnRun(
                    user_turn_index=0,
                    transcribed_text="x",
                    agent_reply="ok",
                    interrupted=True,
                    spans=[
                        PipelineSpan(name="barge_in.yield", started_at_ms=0, ended_at_ms=250),
                    ],
                )
            ],
        )
        assert barge_in_success_rate(conv, run, barge_in_budget_ms=300) == 1.0
        assert barge_in_success_rate(conv, run, barge_in_budget_ms=100) == 0.0


# ---------------------------------------------------------------------------
# False trigger rate
# ---------------------------------------------------------------------------


class TestFalseTriggerRate:
    def test_zero_when_no_runs(self) -> None:
        run = ConversationRun(conv_id="c", topic="t", user_turns_played=0, turn_runs=[])
        assert false_trigger_rate(run) == 0.0

    def test_proportional(self) -> None:
        run = _run(
            [
                _turn_run(transcribed="x", reply="y", vad_end_ms=0, first_byte_ms=10),
                _turn_run(
                    transcribed="", reply="???", vad_end_ms=0, first_byte_ms=10, false_trigger=True
                ),
            ]
        )
        assert false_trigger_rate(run) == 0.5


# ---------------------------------------------------------------------------
# Aggregate score_run
# ---------------------------------------------------------------------------


class TestScoreRun:
    def test_empty(self) -> None:
        report = score_run([])
        assert report.n_conversations == 0

    def test_aggregate_includes_per_conversation(self) -> None:
        conv = _conv([_user("hello world")], gold=["hello"])
        run = _run(
            [
                _turn_run(
                    transcribed="hello world", reply="hello back", vad_end_ms=0, first_byte_ms=200
                )
            ]
        )
        report = score_run([(conv, run)])
        assert report.n_conversations == 1
        assert report.aggregate_wer == 0.0
        assert report.aggregate_faithfulness == 1.0
        assert report.aggregate_turn_latency.n == 1
