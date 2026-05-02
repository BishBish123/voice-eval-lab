"""Pipeline tests for v0.2 additions: RetryingTTS, LatencyBudget, streaming LLM, per-turn WER."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest

from voice_eval_lab.eval.golden import default_golden_set
from voice_eval_lab.eval.metrics import score_conversation
from voice_eval_lab.models import (
    Conversation,
    ConversationRun,
    PipelineSpan,
    Turn,
    TurnRole,
)
from voice_eval_lab.pipeline import (
    TTS,
    FlakyTTS,
    LatencyBudget,
    MockLLM,
    MockSTT,
    MockTTS,
    RetryingTTS,
    VoicePipeline,
)


@dataclass
class SlowFlakyTTS:
    """TTS adapter that takes ~`per_attempt_ms` real ms per attempt and fails the first `fail_n`.

    Used to verify RetryingTTS reports *cumulative* wall-clock latency,
    not just the last attempt's elapsed time.
    """

    inner: TTS
    fail_n: int = 0
    per_attempt_ms: int = 50
    _calls: int = field(default=0, init=False)

    async def synthesize(self, text: str) -> tuple[int, list[PipelineSpan]]:
        self._calls += 1
        await asyncio.sleep(self.per_attempt_ms / 1000.0)
        if self._calls <= self.fail_n:
            raise RuntimeError(f"SlowFlakyTTS scheduled failure #{self._calls}")
        return await self.inner.synthesize(text)

# ---------------------------------------------------------------------------
# RetryingTTS
# ---------------------------------------------------------------------------


class TestRetryingTTS:
    async def test_succeeds_on_retry(self) -> None:
        flaky = FlakyTTS(inner=MockTTS(), fail_n=1)
        wrapped = RetryingTTS(inner=flaky, max_attempts=3)
        first_byte, spans = await wrapped.synthesize("hello")
        # RetryingTTS now returns cumulative wall-clock ms — with MockTTS
        # (no real sleeps) that's ~0ms. The span timing is what matters.
        assert first_byte >= 0
        # one tts.retry span + one tts.synthesize span
        names = [s.name for s in spans]
        assert names.count("tts.retry") == 1
        assert "tts.synthesize" in names

    async def test_succeeds_immediately_with_zero_failures(self) -> None:
        wrapped = RetryingTTS(inner=MockTTS(), max_attempts=3)
        first_byte, spans = await wrapped.synthesize("hello")
        # MockTTS doesn't sleep, so cumulative wall-clock is ~0.
        assert first_byte >= 0
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

    async def test_retrying_tts_reports_cumulative_first_byte(self) -> None:
        # Two failed attempts (~50ms each) + one success (~50ms) ≈ 150ms.
        # The previous implementation only reported the success leg's
        # elapsed time, hiding the retry cost from latency budgets.
        slow = SlowFlakyTTS(inner=MockTTS(), fail_n=2, per_attempt_ms=50)
        wrapped = RetryingTTS(inner=slow, max_attempts=3, base_delay_ms=1)
        first_byte, _ = await wrapped.synthesize("hi")
        # Allow generous tolerance for async scheduler jitter; the bug
        # would have produced ~50, the fix produces ~150.
        assert 120 <= first_byte <= 400, f"expected ~150ms cumulative, got {first_byte}"

    async def test_retrying_tts_first_byte_in_pipeline_span(self) -> None:
        # End-to-end: VoicePipeline must surface the cumulative latency in
        # the tts_first_byte span so latency-budget metrics see it.
        slow = SlowFlakyTTS(inner=MockTTS(), fail_n=2, per_attempt_ms=50)
        wrapped = RetryingTTS(inner=slow, max_attempts=3, base_delay_ms=1)
        pipeline = VoicePipeline(stt=MockSTT(), llm=MockLLM(), tts=wrapped)
        # Build a one-user-turn conversation.
        conv = Conversation(
            conv_id="retry-latency",
            topic="t",
            turns=[
                Turn(role=TurnRole.USER, text="hi", started_at_ms=0, ended_at_ms=1000),
            ],
        )
        run = await pipeline.run(conv)
        assert len(run.turn_runs) == 1
        spans = run.turn_runs[0].spans
        fb = next(s for s in spans if s.name == "tts_first_byte")
        # The span ends at ended_at_ms = vad_end + stt + llm + tts. The
        # cumulative TTS leg should now be ~150ms (not ~50ms).
        # MockSTT default latency 80ms, MockLLM default 120ms, so the TTS
        # contribution is ended_at_ms - started_at_ms.
        tts_contribution = fb.ended_at_ms - fb.started_at_ms
        assert 120 <= tts_contribution <= 400, (
            f"expected ~150ms cumulative TTS leg in span, got {tts_contribution}"
        )


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


class TestMockLLMMatching:
    """The match heuristic must be tolerant of cosmetic phrasing differences."""

    async def test_mock_llm_matches_underscore_to_space(self) -> None:
        llm = MockLLM()
        text, _ = await llm.reply(
            history=[],
            last_user_text="explain ef search in hnsw",  # space form
            gold_facts=[
                "ef_search controls the size of the dynamic candidate list at query time.",
            ],
        )
        # Match found -> reply is the gold fact verbatim, not the hedge.
        assert text.startswith("ef_search controls")

    async def test_mock_llm_matches_case_insensitive(self) -> None:
        llm = MockLLM()
        text, _ = await llm.reply(
            history=[],
            last_user_text="QUIZ ME ON POSTGRES REPLICATION",
            gold_facts=[
                "Physical replication ships WAL bytes; logical replication ships row-level changes.",
            ],
        )
        assert text.startswith("Physical replication")

    async def test_mock_llm_matches_unicode_normalized(self) -> None:
        llm = MockLLM()
        # NFKC fold: fullwidth Latin letters should normalize to ASCII so
        # the user transcript "fullwidth ef" matches "ef_search" in the
        # gold fact. The fullwidth glyphs below are intentional.
        fullwidth = "explain ｅｆ search"  # noqa: RUF001
        text, _ = await llm.reply(
            history=[],
            last_user_text=fullwidth,
            gold_facts=[
                "ef_search controls the size of the dynamic candidate list at query time.",
            ],
        )
        assert text.startswith("ef_search controls")

    async def test_mock_llm_falls_back_when_no_match(self) -> None:
        # Sanity-check the regression direction — unrelated user text still
        # falls back to the hedging reply.
        llm = MockLLM()
        text, _ = await llm.reply(
            history=[],
            last_user_text="completely unrelated topic here",
            gold_facts=["Raft elects a leader by majority vote with randomized timeouts."],
        )
        assert "I don't have a confident answer" in text

    async def test_hnsw_tuning_conversation_now_grounds(self) -> None:
        # The bundled hnsw-tuning conversation phrases the lookup as
        # "explain ef search" while the fact stores it as "ef_search".
        # Faithfulness was 0% before the normalization fix; assert it's
        # >0 now after re-running the conversation through the pipeline.
        conv = next(c for c in default_golden_set() if c.conv_id == "hnsw-tuning")
        pipeline = VoicePipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())
        run = await pipeline.run(conv)
        score = score_conversation(conv, run)
        assert score.response_faithfulness > 0.0


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
