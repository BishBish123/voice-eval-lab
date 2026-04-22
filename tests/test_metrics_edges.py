"""Edge-case + property tests for the headline metrics."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import make_conv, make_run, make_turn_run, make_user
from voice_eval_lab.eval.metrics import (
    _percentile_stats,
    llm_decisiveness,
    response_faithfulness,
    score_run,
    transcription_wer,
    turn_latency_stats,
)
from voice_eval_lab.models import (
    ConversationRun,
    PipelineSpan,
    TurnRun,
)

# ---------------------------------------------------------------------------
# WER edge cases
# ---------------------------------------------------------------------------


class TestWEREdges:
    def test_empty_hypothesis_is_full_wer(self) -> None:
        conv = make_conv([make_user("hello world")])
        run = make_run([make_turn_run(transcribed="", reply="x")])
        # Two reference words, zero match — WER should equal 1.0
        wer = transcription_wer(conv, run)
        assert wer == pytest.approx(1.0)

    def test_empty_reference_is_zero_wer(self) -> None:
        # Empty user turn is skipped; metric returns 0.0.
        conv = make_conv([make_user("")])
        run = make_run([make_turn_run(transcribed="something")])
        assert transcription_wer(conv, run) == 0.0

    def test_single_word_perfect(self) -> None:
        conv = make_conv([make_user("hello")])
        run = make_run([make_turn_run(transcribed="hello")])
        assert transcription_wer(conv, run) == 0.0

    def test_long_transcript(self) -> None:
        text = "the quick brown fox jumps over the lazy dog " * 20
        conv = make_conv([make_user(text)])
        run = make_run([make_turn_run(transcribed=text)])
        assert transcription_wer(conv, run) == 0.0

    def test_no_user_turns_returns_zero(self) -> None:
        conv = make_conv([])
        run = ConversationRun(conv_id="c", topic="t", user_turns_played=0, turn_runs=[])
        assert transcription_wer(conv, run) == 0.0


# ---------------------------------------------------------------------------
# Faithfulness edge cases
# ---------------------------------------------------------------------------


class TestFaithfulnessEdges:
    def test_all_false_triggers_returns_zero(self) -> None:
        conv = make_conv([make_user("u")], gold=["the answer"])
        run = make_run(
            [
                make_turn_run(reply="...?", false_trigger=True),
                make_turn_run(reply="...?", false_trigger=True),
            ]
        )
        # All replies excluded; the function returns 0.0 on no signal.
        assert response_faithfulness(conv, run) == 0.0

    def test_case_insensitive_match(self) -> None:
        conv = make_conv([make_user("u")], gold=["The Answer"])
        run = make_run([make_turn_run(reply="THE ANSWER IS 42")])
        assert response_faithfulness(conv, run) == 1.0

    def test_partial_substring_does_not_count(self) -> None:
        conv = make_conv([make_user("u")], gold=["the very specific answer is 42"])
        run = make_run([make_turn_run(reply="the answer is 42")])
        assert response_faithfulness(conv, run) == 0.0

    def test_multiple_replies_partial_credit(self) -> None:
        conv = make_conv(
            [make_user("u1"), make_user("u2")],
            gold=["yes"],
        )
        run = make_run(
            [
                make_turn_run(reply="yes!"),
                make_turn_run(reply="no, not really"),
            ]
        )
        assert response_faithfulness(conv, run) == 0.5

    def test_faithfulness_handles_unicode_variants(self) -> None:
        # Decomposed-form "café" (e + COMBINING ACUTE) should match the
        # composed-form gold fact after NFKC normalization.
        composed = "the café is open"
        decomposed = "The café is open!"  # e + U+0301
        conv = make_conv([make_user("u")], gold=[composed])
        run = make_run([make_turn_run(reply=decomposed)])
        assert response_faithfulness(conv, run) == 1.0


# ---------------------------------------------------------------------------
# Latency edge cases + percentile ordering
# ---------------------------------------------------------------------------


class TestLatencyOrdering:
    def test_p50_le_p95_le_p99_random(self) -> None:
        latencies = [50, 100, 200, 250, 300, 400, 500, 1000, 1200, 50, 60, 70]
        runs = [
            make_turn_run(vad_end_ms=0, first_byte_ms=lat, user_turn_index=i)
            for i, lat in enumerate(latencies)
        ]
        stats = turn_latency_stats(runs)
        assert stats.p50_ms <= stats.p95_ms <= stats.p99_ms

    def test_n_matches_input(self) -> None:
        runs = [make_turn_run(vad_end_ms=0, first_byte_ms=100, user_turn_index=i) for i in range(7)]
        assert turn_latency_stats(runs).n == 7

    def test_empty_returns_zeros(self) -> None:
        stats = turn_latency_stats([])
        assert stats.n == 0
        assert stats.p50_ms == 0.0
        assert stats.p95_ms == 0.0
        assert stats.p99_ms == 0.0

    def test_skips_turns_without_required_spans(self) -> None:
        # Turn with no spans should be skipped, not crash.
        run = ConversationRun(
            conv_id="c",
            topic="t",
            user_turns_played=2,
            turn_runs=[
                make_turn_run(vad_end_ms=0, first_byte_ms=100, user_turn_index=0),
                TurnRun(user_turn_index=1, transcribed_text="", agent_reply="", spans=[]),
            ],
        )
        stats = turn_latency_stats(run.turn_runs)
        assert stats.n == 1


# ---------------------------------------------------------------------------
# score_run edge cases
# ---------------------------------------------------------------------------


class TestScoreRunEdges:
    def test_all_interrupted(self) -> None:
        conv = make_conv(
            [
                make_user("u1", interrupted=True, end=1000),
                make_user("u2", interrupted=True, end=2000),
            ],
            gold=["x"],
        )
        # barge_in.yield spans within the 200ms default budget so the metric
        # actually counts the turns as successful.
        run = make_run(
            [
                make_turn_run(
                    interrupted=True,
                    vad_end_ms=1000,
                    first_byte_ms=1100,
                    barge_yield_ms=80,
                ),
                make_turn_run(
                    interrupted=True,
                    vad_end_ms=2000,
                    first_byte_ms=2100,
                    barge_yield_ms=80,
                    user_turn_index=1,
                ),
            ]
        )
        report = score_run([(conv, run)])
        assert report.aggregate_barge_in_success == 1.0

    def test_all_silence(self) -> None:
        conv = make_conv([make_user("", end=100), make_user("", end=200)])
        run = make_run(
            [
                make_turn_run(transcribed=""),
                make_turn_run(transcribed="", user_turn_index=1),
            ]
        )
        report = score_run([(conv, run)])
        # WER is 0 (no reference text to score against).
        assert report.aggregate_wer == 0.0

    def test_all_false_trigger(self) -> None:
        # Each user turn covered by a real TurnRun + an injected false-trigger
        # alongside it (mirroring how the pipeline appends synthetics after
        # each real turn). The rate is `false_trigger_turns / user_turns`,
        # so 2 synthetics over 2 user turns = 100%.
        conv = make_conv([make_user("u1"), make_user("u2")], gold=[])
        run = make_run(
            [
                make_turn_run(user_turn_index=0),
                make_turn_run(false_trigger=True, user_turn_index=0),
                make_turn_run(user_turn_index=1),
                make_turn_run(false_trigger=True, user_turn_index=1),
            ]
        )
        report = score_run([(conv, run)])
        assert report.aggregate_false_trigger_rate == 1.0

    def test_empty_pairs(self) -> None:
        report = score_run([])
        assert report.n_conversations == 0


# ---------------------------------------------------------------------------
# Property: faithfulness denominator excludes false triggers
# ---------------------------------------------------------------------------


class TestFalseTriggerExclusion:
    def test_adding_false_trigger_does_not_change_faithfulness(self) -> None:
        conv = make_conv([make_user("u")], gold=["yes"])
        without = make_run([make_turn_run(reply="yes here")])
        with_ft = make_run(
            [
                make_turn_run(reply="yes here"),
                make_turn_run(reply="...", false_trigger=True),
            ]
        )
        assert response_faithfulness(conv, without) == response_faithfulness(conv, with_ft)

    def test_false_trigger_between_real_turns_preserves_alignment(self) -> None:
        # Regression test for the user-turn alignment bug: the pipeline
        # interleaves synthetic false-trigger TurnRuns between real ones
        # (`pipeline.py` appends after the real turn it follows).  A naive
        # `zip(user_turns, run.turn_runs)` shifts every later user turn
        # onto the wrong TurnRun, so WER / faithfulness / endpointing /
        # decisiveness all read the wrong cell.
        #
        # Layout: 3 user turns ("apple", "banana", "cherry").  The real
        # turn for user-turn 0 is followed by a synthetic false-trigger,
        # then the real turns for user-turns 1 and 2.  Each real
        # transcribed_text matches the user_turn exactly, so a correctly
        # aligned WER is 0%.  Pre-fix, the misalignment maps user_turn 1
        # onto the empty false-trigger TurnRun -> non-zero WER.
        conv = make_conv(
            [
                make_user("apple", start=0, end=1000),
                make_user("banana", start=1500, end=2500),
                make_user("cherry", start=3000, end=4000),
            ],
            gold=["apple", "banana", "cherry"],
        )
        run = make_run(
            [
                make_turn_run(
                    transcribed="apple",
                    reply="apple is the answer",
                    vad_end_ms=1000,
                    first_byte_ms=1100,
                    user_turn_index=0,
                ),
                # Synthetic false-trigger appended right after the first
                # real turn — this is what `VoicePipeline.run` does when
                # the per-turn Bernoulli draw fires.
                TurnRun(
                    user_turn_index=0,
                    transcribed_text="",
                    agent_reply="...?",
                    interrupted=False,
                    false_trigger=True,
                    spans=[],
                ),
                make_turn_run(
                    transcribed="banana",
                    reply="banana is correct",
                    vad_end_ms=2500,
                    first_byte_ms=2600,
                    user_turn_index=1,
                ),
                make_turn_run(
                    transcribed="cherry",
                    reply="cherry is right",
                    vad_end_ms=4000,
                    first_byte_ms=4100,
                    user_turn_index=2,
                ),
            ]
        )

        # All three real transcripts match the gold text exactly.  Pre-fix
        # the second/third user turns aligned onto the false-trigger /
        # second-real TurnRun, which broke WER away from 0.
        assert transcription_wer(conv, run) == pytest.approx(0.0)
        # Each gold fact appears in its matching reply -> 100% faithfulness.
        assert response_faithfulness(conv, run) == pytest.approx(1.0)
        # No hedging in any of the three real replies -> 100% decisiveness.
        assert llm_decisiveness(run, conv) == pytest.approx(1.0)
        # score_run wires this through to the pooled aggregates as well.
        report = score_run([(conv, run)])
        assert report.aggregate_wer == pytest.approx(0.0)
        assert report.aggregate_faithfulness == pytest.approx(1.0)
        assert report.aggregate_endpointing_accuracy == pytest.approx(1.0)
        assert report.aggregate_llm_decisiveness == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Property: turn_latency_stats is monotone in n
# ---------------------------------------------------------------------------


class TestLatencyMonotonicity:
    def test_appending_constant_keeps_p50_p95_p99(self) -> None:
        a = [make_turn_run(vad_end_ms=0, first_byte_ms=100, user_turn_index=i) for i in range(5)]
        # Add identical samples — stats shouldn't change.
        b = a + [
            make_turn_run(vad_end_ms=0, first_byte_ms=100, user_turn_index=5 + i) for i in range(5)
        ]
        sa = turn_latency_stats(a)
        sb = turn_latency_stats(b)
        assert (sa.p50_ms, sa.p95_ms, sa.p99_ms) == (sb.p50_ms, sb.p95_ms, sb.p99_ms)

    def test_adding_slow_sample_never_lowers_p99(self) -> None:
        base = [
            make_turn_run(vad_end_ms=0, first_byte_ms=100, user_turn_index=i) for i in range(10)
        ]
        slow = [
            *base,
            make_turn_run(vad_end_ms=0, first_byte_ms=10_000, user_turn_index=10),
        ]
        sa = turn_latency_stats(base)
        sb = turn_latency_stats(slow)
        assert sb.p99_ms >= sa.p99_ms


# ---------------------------------------------------------------------------
# Span helper tests — smoke-test PipelineSpan construction
# ---------------------------------------------------------------------------


class TestPipelineSpanShape:
    def test_attrs_default_empty_dict(self) -> None:
        s = PipelineSpan(name="x", started_at_ms=0, ended_at_ms=10)
        assert s.attrs == {}

    def test_attrs_round_trip(self) -> None:
        s = PipelineSpan(name="x", started_at_ms=0, ended_at_ms=10, attrs={"k": "v"})
        assert s.model_dump()["attrs"] == {"k": "v"}


# ---------------------------------------------------------------------------
# Percentile helper — pin the linear-interpolation method so a future change
# can't silently shift every headline number.
# ---------------------------------------------------------------------------


class TestPercentileMethod:
    def test_percentile_n0_returns_zeros(self) -> None:
        stats = _percentile_stats([])
        assert stats.n == 0
        assert stats.p50_ms == 0.0
        assert stats.p95_ms == 0.0
        assert stats.p99_ms == 0.0

    def test_percentile_n1_returns_single_value(self) -> None:
        stats = _percentile_stats([42.0])
        assert stats.n == 1
        assert stats.p50_ms == 42.0
        assert stats.p95_ms == 42.0
        assert stats.p99_ms == 42.0

    def test_percentile_n2_uses_interpolation(self) -> None:
        # Linear interpolation on [100, 300]: p50 is the midpoint.
        # The old floor-index version would have returned 300 for p50.
        stats = _percentile_stats([100.0, 300.0])
        assert stats.n == 2
        assert stats.p50_ms == pytest.approx(200.0)

    def test_percentile_n3_well_defined(self) -> None:
        stats = _percentile_stats([10.0, 20.0, 30.0])
        assert stats.n == 3
        # Median of three equally-spaced points is the middle one.
        assert stats.p50_ms == pytest.approx(20.0)
        assert stats.p50_ms <= stats.p95_ms <= stats.p99_ms

    def test_percentile_method_is_linear(self) -> None:
        # Pin the method choice — if anyone switches to "lower"/"higher"
        # the headline numbers shift quietly. This test fails before
        # anyone notices the regression.
        samples = [50.0, 100.0, 150.0, 200.0, 250.0]
        expected = np.percentile(samples, [50.0, 95.0, 99.0], method="linear")
        stats = _percentile_stats(samples)
        assert stats.p50_ms == pytest.approx(float(expected[0]))
        assert stats.p95_ms == pytest.approx(float(expected[1]))
        assert stats.p99_ms == pytest.approx(float(expected[2]))
