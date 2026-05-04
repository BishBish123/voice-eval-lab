"""Tests for the v0.2 diagnostic metrics — barge-in latency, jitter, endpointing, decisiveness."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.conftest import make_conv, make_run, make_turn_run, make_user
from voice_eval_lab.eval.metrics import (
    IncompleteRunError,
    barge_in_latency_p95_ms,
    endpointing_accuracy,
    llm_decisiveness,
    score_conversation,
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

    def test_endpointing_no_signal_is_none(self) -> None:
        # Pipeline emitted user turns but never produced a vad_end span on any
        # of them — that's "no measurement," distinct from "every measured
        # turn was wrong." Returning 0.0 conflated the two and made a
        # broken VAD invisible in the aggregate mean.
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
        assert endpointing_accuracy(conv, run) is None

    def test_endpointing_all_wrong_is_zero(self) -> None:
        # VAD emitted on every turn but every one was outside tolerance.
        # That's a measurable failure, not a no-signal case.
        conv = make_conv(
            [
                make_user("u1", end=1000),
                make_user("u2", start=1500, end=2500),
            ]
        )
        run = make_run(
            [
                make_turn_run(vad_end_ms=1300, first_byte_ms=1400),  # off by 300
                make_turn_run(vad_end_ms=2900, first_byte_ms=3000),  # off by 400
            ]
        )
        assert endpointing_accuracy(conv, run) == 0.0

    def test_endpointing_aggregate_skips_no_signal(self) -> None:
        # One conversation reports None (broken VAD); the other reports 1.0.
        # Aggregate must be 1.0, not (1.0 + 0)/2 = 0.5.
        conv_good = make_conv([make_user("good", end=1000)], conv_id="good")
        run_good = make_run(
            [make_turn_run(vad_end_ms=1000, first_byte_ms=1100)],
            conv_id="good",
        )
        conv_broken = make_conv([make_user("broken", end=2000)], conv_id="broken")
        run_broken = ConversationRun(
            conv_id="broken",
            topic="t",
            user_turns_played=1,
            turn_runs=[
                TurnRun(user_turn_index=0, transcribed_text="x", agent_reply="y", spans=[]),
            ],
        )
        report = score_run([(conv_good, run_good), (conv_broken, run_broken)])
        assert report.aggregate_endpointing_accuracy == 1.0

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

    def test_decisiveness_catches_likely_probably(self) -> None:
        # Single-token hedges that the original phrase list missed —
        # word-boundary regex catches them without matching inside
        # unrelated tokens.
        run = make_run(
            [
                make_turn_run(reply="The result is likely correct."),
                make_turn_run(reply="It's probably the LLM."),
                make_turn_run(reply="Could be the network."),
                make_turn_run(reply="The answer is 42."),
            ]
        )
        # Three of four hedge; only one decisive.
        assert llm_decisiveness(run) == pytest.approx(1 / 4)

    def test_decisiveness_word_boundary_does_not_match_inside_tokens(self) -> None:
        # "biweekly" contains the substring "weekly" but no hedge token —
        # the regex must use \b anchors so it doesn't false-fire.
        run = make_run([make_turn_run(reply="The deploy runs biweekly.")])
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


# ---------------------------------------------------------------------------
# Aggregate jitter — pooled stddev across the whole run, not mean-of-stddevs
# ---------------------------------------------------------------------------


class TestAggregateJitter:
    def test_aggregate_jitter_uses_global_pool(self) -> None:
        # Two conversations; the per-turn first-byte latencies vary across
        # the run. The aggregate should equal the population stddev of
        # the *pooled* latencies, not the mean of per-conv stddevs.
        conv_a = make_conv([make_user("a"), make_user("aa")], conv_id="a")
        conv_b = make_conv([make_user("b"), make_user("bb")], conv_id="b")
        run_a = make_run(
            [
                make_turn_run(vad_end_ms=0, first_byte_ms=100),
                make_turn_run(vad_end_ms=0, first_byte_ms=300, user_turn_index=1),
            ],
            conv_id="a",
        )
        run_b = make_run(
            [
                make_turn_run(vad_end_ms=0, first_byte_ms=200),
                make_turn_run(vad_end_ms=0, first_byte_ms=400, user_turn_index=1),
            ],
            conv_id="b",
        )
        report = score_run([(conv_a, run_a), (conv_b, run_b)])
        expected = float(np.std([100.0, 300.0, 200.0, 400.0], ddof=0))
        assert report.aggregate_tts_first_byte_jitter_ms == pytest.approx(expected)

    def test_aggregate_jitter_is_none_with_no_signal(self) -> None:
        # Conversations with user turns but turn_runs that emit no
        # vad_end/first_byte spans produce zero pooled samples.
        conv = make_conv([make_user("u")], conv_id="x")
        run = ConversationRun(
            conv_id="x",
            topic="t",
            user_turns_played=1,
            turn_runs=[
                TurnRun(user_turn_index=0, transcribed_text="x", agent_reply="y", spans=[]),
            ],
        )
        report = score_run([(conv, run)])
        assert report.aggregate_tts_first_byte_jitter_ms is None


# ---------------------------------------------------------------------------
# Turn coverage — adapters that drop turns must surface, not score better
# ---------------------------------------------------------------------------


class TestTurnCoverage:
    def test_metrics_raise_on_incomplete_run(self) -> None:
        conv = make_conv(
            [
                make_user("u1", end=1000),
                make_user("u2", start=1500, end=2500),
                make_user("u3", start=3000, end=4000),
            ],
            gold=["x"],
        )
        # Only one turn_run for three user turns — adapter dropped the rest.
        run = make_run([make_turn_run(vad_end_ms=1000, first_byte_ms=1100)])
        with pytest.raises(IncompleteRunError, match="3 user turns"):
            score_conversation(conv, run)
        with pytest.raises(IncompleteRunError, match="3 user turns"):
            score_run([(conv, run)])

    def test_run_with_full_coverage_passes(self) -> None:
        conv = make_conv(
            [
                make_user("u1", end=1000),
                make_user("u2", start=1500, end=2500),
            ],
            gold=["x"],
        )
        run = make_run(
            [
                make_turn_run(vad_end_ms=1000, first_byte_ms=1100),
                make_turn_run(vad_end_ms=2500, first_byte_ms=2600, user_turn_index=1),
            ]
        )
        # Should not raise — full coverage.
        score = score_conversation(conv, run)
        assert score.conv_id == "c"

    def test_run_with_extra_false_triggers_passes(self) -> None:
        # Pipelines emit synthetic false-trigger turn_runs *after* the
        # user turns; coverage check only flags missing turns, not extras.
        conv = make_conv([make_user("u1", end=1000)], gold=["x"])
        run = make_run(
            [
                make_turn_run(vad_end_ms=1000, first_byte_ms=1100),
                make_turn_run(false_trigger=True, user_turn_index=1),
            ]
        )
        score = score_conversation(conv, run)
        assert score.conv_id == "c"


# ---------------------------------------------------------------------------
# Aggregate WER — corpus pool, not mean of per-conversation WER
# ---------------------------------------------------------------------------


class TestAggregatePooledRatios:
    """Faithfulness, false-trigger, decisiveness, endpointing must be pooled.

    Mean-of-per-conversation overweights short conversations. A 1-turn
    conversation at 100% reads the same as a 50-turn conversation at
    50% in the mean (1.0+0.5)/2 = 0.75; the corpus rate is 26/51 ~ 0.51.
    """

    def test_aggregate_faithfulness_uses_pooled_replies(self) -> None:
        # Short conversation (1 reply, grounded) + long conversation
        # (50 replies, half grounded). Mean = 0.75, pooled = 26/51 ~= 0.51.
        gold = ["alpha"]
        conv_short = make_conv([make_user("u1")], gold=gold, conv_id="short")
        run_short = make_run(
            [make_turn_run(reply="alpha is the fact.")],
            conv_id="short",
        )
        long_user_turns = []
        long_runs = []
        for i in range(50):
            long_user_turns.append(make_user(f"u{i}", start=i * 1000, end=i * 1000 + 500))
            long_runs.append(
                make_turn_run(
                    reply="alpha is the fact." if i % 2 == 0 else "no idea",
                    user_turn_index=i,
                )
            )
        conv_long = make_conv(long_user_turns, gold=gold, conv_id="long")
        run_long = make_run(long_runs, conv_id="long")
        report = score_run([(conv_short, run_short), (conv_long, run_long)])
        # Pooled: (1 + 25) / (1 + 50) = 26/51.
        assert report.aggregate_faithfulness == pytest.approx(26 / 51)

    def test_aggregate_false_trigger_uses_pooled_turns(self) -> None:
        # Short conversation (1 turn, 1 false-trigger) -> 100% per-conv.
        # Long conversation (50 turns, 0 false-triggers) -> 0% per-conv.
        # Mean = 50%, pooled = 1/51 ~= 0.02.
        conv_short = make_conv([make_user("u1")], conv_id="short")
        run_short = make_run(
            [make_turn_run(false_trigger=True)],
            conv_id="short",
        )
        long_user_turns = [
            make_user(f"u{i}", start=i * 1000, end=i * 1000 + 500) for i in range(50)
        ]
        long_runs = [make_turn_run(user_turn_index=i) for i in range(50)]
        conv_long = make_conv(long_user_turns, conv_id="long")
        run_long = make_run(long_runs, conv_id="long")
        report = score_run([(conv_short, run_short), (conv_long, run_long)])
        assert report.aggregate_false_trigger_rate == pytest.approx(1 / 51)

    def test_aggregate_decisiveness_uses_pooled_replies(self) -> None:
        # Short: 1 decisive reply. Long: 50 replies, 10 decisive.
        # Mean = (1.0 + 0.2) / 2 = 0.6, pooled = 11/51 ~= 0.216.
        conv_short = make_conv([make_user("u1")], conv_id="short")
        run_short = make_run(
            [make_turn_run(reply="The answer is 42.")],
            conv_id="short",
        )
        long_user_turns = [
            make_user(f"u{i}", start=i * 1000, end=i * 1000 + 500) for i in range(50)
        ]
        long_runs = []
        for i in range(50):
            reply = "The answer is 42." if i < 10 else "I don't know."
            long_runs.append(make_turn_run(reply=reply, user_turn_index=i))
        conv_long = make_conv(long_user_turns, conv_id="long")
        run_long = make_run(long_runs, conv_id="long")
        report = score_run([(conv_short, run_short), (conv_long, run_long)])
        assert report.aggregate_llm_decisiveness == pytest.approx(11 / 51)

    def test_aggregate_endpointing_uses_pooled_measured_turns(self) -> None:
        # Short conv (1 turn, aligned). Long conv (50 turns, half aligned).
        # Mean = (1.0 + 0.5) / 2 = 0.75, pooled = 26/51.
        conv_short = make_conv([make_user("u1", end=1000)], conv_id="short")
        run_short = make_run(
            [make_turn_run(vad_end_ms=1000, first_byte_ms=1100)],
            conv_id="short",
        )
        long_user_turns = []
        long_runs = []
        for i in range(50):
            long_user_turns.append(
                make_user(f"u{i}", start=i * 1000, end=i * 1000 + 500)
            )
            # Even turns are aligned, odd turns are 200ms off.
            vad_drift = 0 if i % 2 == 0 else 200
            long_runs.append(
                make_turn_run(
                    vad_end_ms=i * 1000 + 500 + vad_drift,
                    first_byte_ms=i * 1000 + 600 + vad_drift,
                    user_turn_index=i,
                )
            )
        conv_long = make_conv(long_user_turns, conv_id="long")
        run_long = make_run(long_runs, conv_id="long")
        report = score_run([(conv_short, run_short), (conv_long, run_long)])
        # Pooled aligned: 1 + 25 = 26; pooled measured: 1 + 50 = 51.
        assert report.aggregate_endpointing_accuracy == pytest.approx(26 / 51)


class TestAggregateBargeInSuccessPooled:
    def test_aggregate_barge_in_pools_interrupted_turns(self) -> None:
        # Three conversations: two with no interrupts (used to inflate the
        # mean to 1.0 for free), one with two interrupts where one fails.
        # Old mean would be (1.0 + 1.0 + 0.5) / 3 = 0.833; pooled is 1/2 = 0.5.
        conv_quiet1 = make_conv([make_user("a")], conv_id="q1")
        conv_quiet2 = make_conv([make_user("b")], conv_id="q2")
        conv_interrupt = make_conv(
            [
                make_user("c", interrupted=True),
                make_user("d", start=2000, end=3000, interrupted=True),
            ],
            conv_id="i",
        )
        run_quiet1 = make_run([make_turn_run()], conv_id="q1")
        run_quiet2 = make_run([make_turn_run()], conv_id="q2")
        # First yield 50ms (under default 100ms budget — success), second 200ms (fail).
        run_interrupt = make_run(
            [
                make_turn_run(interrupted=True, barge_yield_ms=50),
                make_turn_run(interrupted=True, barge_yield_ms=200, user_turn_index=1),
            ],
            conv_id="i",
        )
        report = score_run(
            [(conv_quiet1, run_quiet1), (conv_quiet2, run_quiet2), (conv_interrupt, run_interrupt)]
        )
        assert report.aggregate_barge_in_success == pytest.approx(0.5)

    def test_aggregate_barge_in_returns_none_on_no_signal(self) -> None:
        # No conversation has any interrupted turns at all — there's no
        # signal, so the pooled aggregate is None (distinct from 1.0).
        conv_a = make_conv([make_user("a")], conv_id="a")
        conv_b = make_conv([make_user("b")], conv_id="b")
        run_a = make_run([make_turn_run()], conv_id="a")
        run_b = make_run([make_turn_run()], conv_id="b")
        report = score_run([(conv_a, run_a), (conv_b, run_b)])
        assert report.aggregate_barge_in_success is None


class TestAggregateWERCorpusPool:
    def test_aggregate_wer_uses_corpus_pool(self) -> None:
        # Two conversations: one short conversation at 0% WER and one long
        # conversation at ~10% WER. The mean-of-per-conv aggregate would
        # report 5%; the corpus-pooled WER must come out near the long
        # conversation's rate because that's where almost all the words live.

        # Short conversation: one turn, perfect transcription.
        short_user = make_user("hello world", end=1000)
        short_run_turn = make_turn_run(transcribed="hello world")
        conv_short = make_conv([short_user], conv_id="short")
        run_short = make_run([short_run_turn], conv_id="short")

        # Long conversation: 100 turns, each with WER=10% (1 wrong word in 10).
        long_turns = []
        long_runs = []
        for i in range(100):
            long_turns.append(
                make_user(
                    "alpha bravo charlie delta echo foxtrot golf hotel india juliet",
                    start=i * 1000,
                    end=i * 1000 + 500,
                )
            )
            long_runs.append(
                make_turn_run(
                    transcribed="WRONG bravo charlie delta echo foxtrot golf hotel india juliet",
                    user_turn_index=i,
                )
            )
        conv_long = make_conv(long_turns, conv_id="long")
        run_long = make_run(long_runs, conv_id="long")

        report = score_run([(conv_short, run_short), (conv_long, run_long)])
        # Corpus has 2 (short) + 1000 (long) = 1002 reference words; 100 errors
        # all in long conv; expected WER ~= 100/1002 ~= 0.0998.
        assert report.aggregate_wer == pytest.approx(100 / 1002, rel=1e-6)
        # Sanity check: this is far from the (0 + 0.1) / 2 = 0.05 a mean would give.
        assert report.aggregate_wer > 0.09
