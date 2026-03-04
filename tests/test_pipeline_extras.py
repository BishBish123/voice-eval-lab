"""Pipeline tests for v0.2 additions: RetryingTTS, LatencyBudget, streaming LLM, per-turn WER."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from voice_eval_lab.eval.golden import default_golden_set
from voice_eval_lab.models import (
    Conversation,
    ConversationRun,
    PipelineSpan,
    Turn,
    TurnRole,
)
from voice_eval_lab.pipeline import (
    FlakyTTS,
    LatencyBudget,
    MockLLM,
    MockSTT,
    MockTTS,
    RetryingTTS,
    VoicePipeline,
)

# ---------------------------------------------------------------------------
# RetryingTTS
# ---------------------------------------------------------------------------


class TestRetryingTTS:
    async def test_succeeds_on_retry(self) -> None:
        flaky = FlakyTTS(inner=MockTTS(), fail_n=1)
        wrapped = RetryingTTS(inner=flaky, max_attempts=3)
        first_byte, spans = await wrapped.synthesize("hello")
        assert first_byte == 75
        # one tts.retry span + one tts.synthesize span
        names = [s.name for s in spans]
        assert names.count("tts.retry") == 1
        assert "tts.synthesize" in names

    async def test_succeeds_immediately_with_zero_failures(self) -> None:
        wrapped = RetryingTTS(inner=MockTTS(), max_attempts=3)
        first_byte, spans = await wrapped.synthesize("hello")
        assert first_byte == 75
        assert all(s.name != "tts.retry" for s in spans)

    async def test_raises_after_max_attempts(self) -> None:
        flaky = FlakyTTS(inner=MockTTS(), fail_n=10)
        wrapped = RetryingTTS(inner=flaky, max_attempts=3)
        with pytest.raises(RuntimeError):
            await wrapped.synthesize("hello")

    async def test_retry_attrs_carry_attempt_number(self) -> None:
        flaky = FlakyTTS(inner=MockTTS(), fail_n=2)
        wrapped = RetryingTTS(inner=flaky, max_attempts=3, base_delay_ms=10)
        _, spans = await wrapped.synthesize("hi")
        retries = [s for s in spans if s.name == "tts.retry"]
        assert [r.attrs["attempt"] for r in retries] == ["1", "2"]

    async def test_exponential_backoff_in_span_duration(self) -> None:
        flaky = FlakyTTS(inner=MockTTS(), fail_n=2)
        wrapped = RetryingTTS(inner=flaky, max_attempts=3, base_delay_ms=10)
        _, spans = await wrapped.synthesize("hi")
        retries = [s for s in spans if s.name == "tts.retry"]
        # 10 * 2**0 = 10, 10 * 2**1 = 20
        durations = [s.ended_at_ms - s.started_at_ms for s in retries]
        assert durations == [10, 20]


# ---------------------------------------------------------------------------
# LatencyBudget middleware
# ---------------------------------------------------------------------------


class TestLatencyBudget:
    async def test_no_warning_when_under_budget(self) -> None:
        pipeline = VoicePipeline(
            stt=MockSTT(),
            llm=MockLLM(),
            tts=MockTTS(),
            latency_budget=LatencyBudget(budget_ms=10_000),
        )
        run = await pipeline.run(default_golden_set()[0])
        for tr in run.turn_runs:
            assert all(s.name != "latency_budget.exceeded" for s in tr.spans)

    async def test_emits_warning_when_over_budget(self) -> None:
        pipeline = VoicePipeline(
            stt=MockSTT(),
            llm=MockLLM(),
            tts=MockTTS(),
            latency_budget=LatencyBudget(budget_ms=50),
        )
        run = await pipeline.run(default_golden_set()[0])
        flagged = [
            s for tr in run.turn_runs for s in tr.spans if s.name == "latency_budget.exceeded"
        ]
        assert flagged, "expected at least one budget violation span"
        assert flagged[0].attrs["budget_ms"] == "50"

    def test_annotate_is_idempotent_for_runs_without_spans(self) -> None:
        # Annotating a run with empty turn_runs leaves it unchanged.
        run = ConversationRun(conv_id="c", topic="t", user_turns_played=0, turn_runs=[])
        out = LatencyBudget(budget_ms=100).annotate(run)
        assert out is run
        assert out.turn_runs == []


# ---------------------------------------------------------------------------
# Streaming LLM
# ---------------------------------------------------------------------------


class TestStreamingLLM:
    async def test_stream_yields_chunks(self) -> None:
        llm = MockLLM()
        chunks: list[str] = []
        agen: AsyncIterator[str] = llm.stream(
            history=[],
            last_user_text="quiz me on postgres replication",
            gold_facts=[
                "Physical replication ships WAL bytes; logical replication ships row-level changes.",
            ],
            chunk_size=2,
        )
        async for chunk in agen:
            chunks.append(chunk)
        assert chunks  # non-empty
        # Joined chunks should reproduce the reply.
        full, _ = await llm.reply(
            [],
            "quiz me on postgres replication",
            [
                "Physical replication ships WAL bytes; logical replication ships row-level changes.",
            ],
        )
        assert " ".join(chunks).split() == full.split()

    async def test_stream_chunk_size_default(self) -> None:
        llm = MockLLM()
        chunks: list[str] = []
        async for chunk in llm.stream(history=[], last_user_text="hi", gold_facts=[]):
            chunks.append(chunk)
        assert chunks


# ---------------------------------------------------------------------------
# Per-turn STT WER
# ---------------------------------------------------------------------------


class TestPerTurnWER:
    async def test_per_turn_override_wins(self) -> None:
        stt = MockSTT(wer_substitution_rate=0.0)
        # Override per-turn — high rate.
        turn = Turn(
            role=TurnRole.USER,
            text="one two three four",
            started_at_ms=0,
            ended_at_ms=1000,
            wer_substitution_rate=0.5,
        )
        text, _ = await stt.transcribe(turn)
        assert text.startswith("WERR WERR")

    async def test_per_turn_zero_overrides_global(self) -> None:
        stt = MockSTT(wer_substitution_rate=1.0)
        turn = Turn(
            role=TurnRole.USER,
            text="alpha beta gamma",
            started_at_ms=0,
            ended_at_ms=1000,
            wer_substitution_rate=0.0,
        )
        text, _ = await stt.transcribe(turn)
        assert text == "alpha beta gamma"

    async def test_global_used_when_per_turn_absent(self) -> None:
        stt = MockSTT(wer_substitution_rate=1.0)
        turn = Turn(
            role=TurnRole.USER,
            text="alpha beta gamma",
            started_at_ms=0,
            ended_at_ms=1000,
        )
        text, _ = await stt.transcribe(turn)
        # 100% substitution on 3 words -> 3 WERR
        assert text.split() == ["WERR", "WERR", "WERR"]

    async def test_span_records_effective_rate(self) -> None:
        stt = MockSTT(wer_substitution_rate=0.0)
        turn = Turn(
            role=TurnRole.USER,
            text="x y",
            started_at_ms=0,
            ended_at_ms=500,
            wer_substitution_rate=0.5,
        )
        _, spans = await stt.transcribe(turn)
        assert spans[0].attrs["wer_injected"] == "0.5"


# ---------------------------------------------------------------------------
# Determinism + run-shape invariants on the expanded golden set
# ---------------------------------------------------------------------------


class TestDeterminism:
    async def test_same_input_same_output(self) -> None:
        pipeline_a = VoicePipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())
        pipeline_b = VoicePipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())
        conv = default_golden_set()[0]
        run_a = await pipeline_a.run(conv)
        run_b = await pipeline_b.run(conv)
        assert run_a.model_dump() == run_b.model_dump()

    async def test_pipeline_emits_one_turn_run_per_user_turn_for_all_convs(self) -> None:
        pipeline = VoicePipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())
        for conv in default_golden_set():
            run = await pipeline.run(conv)
            n_user = sum(1 for t in conv.turns if t.role.value == "user")
            assert run.user_turns_played == n_user
            assert len(run.turn_runs) == n_user


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestPipelineEdgeCases:
    async def test_empty_conversation(self) -> None:
        pipeline = VoicePipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())
        conv = Conversation(conv_id="empty", topic="empty", turns=[])
        run = await pipeline.run(conv)
        assert run.user_turns_played == 0
        assert run.turn_runs == []

    async def test_all_agent_turns_no_user(self) -> None:
        pipeline = VoicePipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())
        conv = Conversation(
            conv_id="agent-only",
            topic="agent-only",
            turns=[
                Turn(role=TurnRole.AGENT, text="a", started_at_ms=0, ended_at_ms=500),
                Turn(role=TurnRole.AGENT, text="b", started_at_ms=600, ended_at_ms=1000),
            ],
        )
        run = await pipeline.run(conv)
        assert run.user_turns_played == 0
        assert run.turn_runs == []

    async def test_interrupted_turn_emits_barge_yield_span(self) -> None:
        pipeline = VoicePipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())
        # double-barge has two interrupted turns
        conv = next(c for c in default_golden_set() if c.conv_id == "double-barge")
        run = await pipeline.run(conv)
        yields: list[PipelineSpan] = []
        for tr in run.turn_runs:
            yields.extend([s for s in tr.spans if s.name == "barge_in.yield"])
        assert len(yields) == 2
