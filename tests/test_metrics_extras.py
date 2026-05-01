"""Tests for the v0.2 diagnostic metrics — barge-in latency, jitter, endpointing, decisiveness."""

from __future__ import annotations

import math

import pytest

from tests.conftest import make_conv, make_run, make_turn_run, make_user
from voice_eval_lab.eval.metrics import (
    barge_in_latency_p95_ms,
    endpointing_accuracy,
    llm_decisiveness,
    score_run,
    tts_first_byte_jitter_ms,
)
from voice_eval_lab.models import (
    ConversationRun,
    PipelineSpan,
    TurnRun,
)

# ---------------------------------------------------------------------------
# barge_in_latency_p95
# ---------------------------------------------------------------------------


class TestBargeInLatencyP95:
    def test_zero_when_no_barge_yield_span(self) -> None:
        run = make_run([make_turn_run()])
        assert barge_in_latency_p95_ms(run) == 0.0

    def test_single_barge_yield(self) -> None:
        run = make_run([make_turn_run(interrupted=True, barge_yield_ms=80)])
        assert barge_in_latency_p95_ms(run) == 80.0

    def test_picks_p95(self) -> None:
        # 90 samples at 50, 10 at 200 — at n=100, p95 index = int(0.95*100)=95,
        # which falls in the slow (200) bucket once sorted.
        runs: list[TurnRun] = []
        for i in range(90):
            runs.append(make_turn_run(interrupted=True, barge_yield_ms=50, user_turn_index=i))
        for i in range(10):
            runs.append(make_turn_run(interrupted=True, barge_yield_ms=200, user_turn_index=90 + i))
        run = make_run(runs)
        assert barge_in_latency_p95_ms(run) == 200.0

    def test_ignores_non_barge_spans(self) -> None:
        # Even with `barge_in.yield` missing, an interrupted turn isn't enough.
        run = make_run([make_turn_run(interrupted=True)])
        assert barge_in_latency_p95_ms(run) == 0.0

    def test_does_not_crash_on_empty_run(self) -> None:
        run = ConversationRun(conv_id="c", topic="t", user_turns_played=0, turn_runs=[])
        assert barge_in_latency_p95_ms(run) == 0.0


# ---------------------------------------------------------------------------
# tts_first_byte_jitter
# ---------------------------------------------------------------------------


class TestTTSFirstByteJitter:
    def test_zero_for_single_sample(self) -> None:
        run = make_run([make_turn_run(vad_end_ms=0, first_byte_ms=100)])
        assert tts_first_byte_jitter_ms(run) == 0.0

    def test_zero_for_constant_latency(self) -> None:
        runs = [
            make_turn_run(vad_end_ms=0, first_byte_ms=200),
            make_turn_run(vad_end_ms=1000, first_byte_ms=1200),
            make_turn_run(vad_end_ms=2000, first_byte_ms=2200),
        ]
        assert tts_first_byte_jitter_ms(make_run(runs)) == 0.0

    def test_known_stddev_two_samples(self) -> None:
        runs = [
            make_turn_run(vad_end_ms=0, first_byte_ms=100),  # latency 100
            make_turn_run(vad_end_ms=0, first_byte_ms=300),  # latency 300
        ]
        # mean=200, var=10000, stddev=100
        assert tts_first_byte_jitter_ms(make_run(runs)) == pytest.approx(100.0)

    def test_increases_with_spread(self) -> None:
        tight = make_run(
            [
                make_turn_run(vad_end_ms=0, first_byte_ms=100),
                make_turn_run(vad_end_ms=0, first_byte_ms=110),
                make_turn_run(vad_end_ms=0, first_byte_ms=90),
            ]
        )
        wide = make_run(
            [
                make_turn_run(vad_end_ms=0, first_byte_ms=100),
                make_turn_run(vad_end_ms=0, first_byte_ms=300),
                make_turn_run(vad_end_ms=0, first_byte_ms=50),
            ]
        )
        assert tts_first_byte_jitter_ms(wide) > tts_first_byte_jitter_ms(tight)

    def test_ignores_turns_without_spans(self) -> None:
        # Empty span turn should be skipped.
        run = ConversationRun(
            conv_id="c",
            topic="t",
            user_turns_played=2,
            turn_runs=[
                make_turn_run(vad_end_ms=0, first_byte_ms=100),
                TurnRun(
                    user_turn_index=1,
                    transcribed_text="",
                    agent_reply="",
                    spans=[],
                ),
            ],
        )
        # Only one valid sample => 0.0
        assert tts_first_byte_jitter_ms(run) == 0.0


# ---------------------------------------------------------------------------
# endpointing_accuracy
# ---------------------------------------------------------------------------


class TestEndpointingAccuracy:
    def test_perfect_alignment(self) -> None:
        conv = make_conv([make_user("hello", end=1000)])
        run = make_run([make_turn_run(vad_end_ms=1000, first_byte_ms=1100)])
        assert endpointing_accuracy(conv, run) == 1.0

    def test_off_by_one_within_default_tolerance(self) -> None:
        conv = make_conv([make_user("hello", end=1000)])
        run = make_run([make_turn_run(vad_end_ms=1040, first_byte_ms=1100)])
        # default tolerance is 50ms — 40ms drift is OK.
        assert endpointing_accuracy(conv, run) == 1.0

    def test_outside_tolerance(self) -> None:
        conv = make_conv([make_user("hello", end=1000)])
        run = make_run([make_turn_run(vad_end_ms=1200, first_byte_ms=1300)])
        assert endpointing_accuracy(conv, run) == 0.0

    def test_partial_alignment(self) -> None:
        conv = make_conv(
            [
                make_user("a", end=1000),
                make_user("b", start=1500, end=2500),
            ]
        )
        run = make_run(
            [
                make_turn_run(vad_end_ms=1000, first_byte_ms=1100),  # aligned
                make_turn_run(vad_end_ms=2700, first_byte_ms=2800),  # off by 200
            ]
        )
        assert endpointing_accuracy(conv, run) == 0.5

    def test_no_user_turns_is_one(self) -> None:
        # vacuously true: nothing to be wrong about.
        conv = make_conv([])
        run = ConversationRun(conv_id="c", topic="t", user_turns_played=0, turn_runs=[])
        assert endpointing_accuracy(conv, run) == 1.0

    def test_user_turns_no_vad_spans_is_zero(self) -> None:
        # Pipeline emitted user turns but never produced a vad_end span on any
        # of them — that's a broken VAD, not a perfect endpointing score.
        conv = make_conv([make_user("u1", end=1000), make_user("u2", end=2000)])
        run = ConversationRun(
            conv_id="c",
            topic="t",
            user_turns_played=2,
            turn_runs=[
                TurnRun(user_turn_index=0, transcribed_text="x", agent_reply="y", spans=[]),
                TurnRun(user_turn_index=1, transcribed_text="x", agent_reply="y", spans=[]),
            ],
        )
        assert endpointing_accuracy(conv, run) == 0.0

    def test_user_turns_with_vad_spans_uses_real_math(self) -> None:
        # Sanity-check that the existing math still runs when a VAD signal
        # is present.
        conv = make_conv(
            [
                make_user("a", end=1000),
                make_user("b", start=1500, end=2500),
            ]
        )
        run = make_run(
            [
                make_turn_run(vad_end_ms=1000, first_byte_ms=1100),  # aligned
                make_turn_run(vad_end_ms=2700, first_byte_ms=2800),  # off by 200
            ]
        )
        assert endpointing_accuracy(conv, run) == 0.5

    def test_custom_tolerance(self) -> None:
        conv = make_conv([make_user("hello", end=1000)])
        run = make_run([make_turn_run(vad_end_ms=1100, first_byte_ms=1200)])
        assert endpointing_accuracy(conv, run, tolerance_ms=200) == 1.0
        assert endpointing_accuracy(conv, run, tolerance_ms=50) == 0.0


# ---------------------------------------------------------------------------
# llm_decisiveness
# ---------------------------------------------------------------------------


class TestLLMDecisiveness:
    def test_all_decisive(self) -> None:
        run = make_run(
            [
                make_turn_run(reply="The answer is 42."),
                make_turn_run(reply="Yes, that's correct."),
            ]
        )
        assert llm_decisiveness(run) == 1.0

    def test_one_hedge(self) -> None:
        run = make_run(
            [
                make_turn_run(reply="The answer is 42."),
                make_turn_run(reply="I don't know."),
            ]
        )
        assert llm_decisiveness(run) == 0.5

    def test_mock_llm_fallback_counts_as_hedge(self) -> None:
        # The mock LLM emits "I don't have a confident answer about ..." when
        # no gold fact matches. That phrase is in HEDGING_PHRASES, so the
        # decisiveness should drop accordingly.
        run = make_run(
            [
                make_turn_run(reply="I don't have a confident answer about 'foo'."),
            ]
        )
        assert llm_decisiveness(run) == 0.0

    def test_excludes_false_triggers(self) -> None:
        run = make_run(
            [
                make_turn_run(reply="A grounded answer."),
                make_turn_run(reply="...?", false_trigger=True),
            ]
        )
        # Only the first reply counts.
        assert llm_decisiveness(run) == 1.0

    def test_empty_reply_counts_as_hedge(self) -> None:
        run = make_run([make_turn_run(reply="")])
        assert llm_decisiveness(run) == 0.0

    def test_no_replies_returns_one(self) -> None:
        run = ConversationRun(conv_id="c", topic="t", user_turns_played=0, turn_runs=[])
        assert llm_decisiveness(run) == 1.0


# ---------------------------------------------------------------------------
# Property-style checks: monotonicity invariants
# ---------------------------------------------------------------------------


class TestMonotonicity:
    @pytest.mark.parametrize(
        "samples",
        [
            [50, 60, 70, 80, 90, 100],
            [200, 200, 200, 200, 200],
            [10, 20, 30, 100, 100, 100, 200, 300],
        ],
    )
    def test_jitter_nonnegative(self, samples: list[int]) -> None:
        runs = [make_turn_run(vad_end_ms=0, first_byte_ms=s) for s in samples]
        assert tts_first_byte_jitter_ms(make_run(runs)) >= 0.0

    def test_jitter_invariant_under_constant_shift(self) -> None:
        a = make_run(
            [
                make_turn_run(vad_end_ms=0, first_byte_ms=100),
                make_turn_run(vad_end_ms=0, first_byte_ms=200),
                make_turn_run(vad_end_ms=0, first_byte_ms=300),
            ]
        )
        b = make_run(
            [
                make_turn_run(vad_end_ms=1000, first_byte_ms=1100),
                make_turn_run(vad_end_ms=1000, first_byte_ms=1200),
                make_turn_run(vad_end_ms=1000, first_byte_ms=1300),
            ]
        )
        assert tts_first_byte_jitter_ms(a) == pytest.approx(tts_first_byte_jitter_ms(b))

    def test_more_yield_spans_never_lower_p95(self) -> None:
        small = make_run([make_turn_run(interrupted=True, barge_yield_ms=80)])
        bigger = make_run(
            [
                make_turn_run(interrupted=True, barge_yield_ms=80),
                make_turn_run(interrupted=True, barge_yield_ms=400, user_turn_index=1),
            ]
        )
        # Adding a slower yield never makes p95 smaller.
        assert barge_in_latency_p95_ms(bigger) >= barge_in_latency_p95_ms(small)

    def test_decisiveness_is_a_fraction(self) -> None:
        run = make_run(
            [
                make_turn_run(reply="grounded"),
                make_turn_run(reply="I don't know"),
                make_turn_run(reply="maybe"),
            ]
        )
        d = llm_decisiveness(run)
        assert 0.0 <= d <= 1.0
        assert math.isfinite(d)

    def test_endpointing_is_a_fraction(self) -> None:
        conv = make_conv(
            [
                make_user("a", end=1000),
                make_user("b", start=2000, end=3000),
                make_user("c", start=4000, end=5000),
            ]
        )
        run = make_run(
            [
                make_turn_run(vad_end_ms=1000, first_byte_ms=1100),
                make_turn_run(vad_end_ms=3500, first_byte_ms=3600),
                make_turn_run(vad_end_ms=5000, first_byte_ms=5100),
            ]
        )
        e = endpointing_accuracy(conv, run)
        assert 0.0 <= e <= 1.0


# ---------------------------------------------------------------------------
# Span-attribute round-trip — the new spans must carry the right names
# ---------------------------------------------------------------------------


class TestSpanShape:
    def test_barge_in_yield_span_uses_dotted_name(self) -> None:
        spans = [
            PipelineSpan(name="barge_in.yield", started_at_ms=0, ended_at_ms=80),
        ]
        assert spans[0].name == "barge_in.yield"


# ---------------------------------------------------------------------------
# Aggregate barge-in p95 — pooled across the run, not the mean of per-conv
# ---------------------------------------------------------------------------


class TestAggregateBargeInP95:
    def test_aggregate_barge_in_p95_uses_global_pool(self) -> None:
        # Three conversations: two with barge-in yields and one with none.
        # Old impl averaged per-conv p95s (incl. zeros) and produced
        # a much smaller headline; the fix pools yields globally so the
        # aggregate is a true p95 of the observed durations.
        conv_a = make_conv([make_user("a", interrupted=True)], conv_id="a")
        conv_b = make_conv([make_user("b", interrupted=True)], conv_id="b")
        conv_c = make_conv([make_user("c")], conv_id="c")
        run_a = make_run(
            [make_turn_run(interrupted=True, barge_yield_ms=80)],
            conv_id="a",
        )
        run_b = make_run(
            [
                make_turn_run(interrupted=True, barge_yield_ms=120),
                make_turn_run(interrupted=True, barge_yield_ms=200, user_turn_index=1),
            ],
            conv_id="b",
        )
        run_c = make_run([make_turn_run()], conv_id="c")
        report = score_run([(conv_a, run_a), (conv_b, run_b), (conv_c, run_c)])
        # Pooled samples sorted: [80, 120, 200]; idx = int(0.95 * 3) = 2 -> 200.
        assert report.aggregate_barge_in_latency_p95_ms == 200.0

    def test_aggregate_barge_in_p95_returns_none_with_no_signal(self) -> None:
        conv_a = make_conv([make_user("a")], conv_id="a")
        conv_b = make_conv([make_user("b")], conv_id="b")
        run_a = make_run([make_turn_run()], conv_id="a")
        run_b = make_run([make_turn_run()], conv_id="b")
        report = score_run([(conv_a, run_a), (conv_b, run_b)])
        assert report.aggregate_barge_in_latency_p95_ms is None
