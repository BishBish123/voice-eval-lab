"""Pipeline tests — adapter contracts + run-shape invariants."""

from __future__ import annotations

from voice_eval_lab.eval.golden import default_golden_set
from voice_eval_lab.models import Conversation, PipelineSpan, Turn, TurnRole
from voice_eval_lab.pipeline import MockLLM, MockSTT, MockTTS, VoicePipeline


class TestMockSTT:
    async def test_substitutes_first_n_words(self) -> None:
        stt = MockSTT(wer_substitution_rate=0.5)
        text, _spans = await stt.transcribe(
            Turn(role=TurnRole.USER, text="one two three four", started_at_ms=0, ended_at_ms=1000)
        )
        assert text.startswith("WERR WERR")

    async def test_no_substitution_when_rate_zero(self) -> None:
        stt = MockSTT(wer_substitution_rate=0)
        text, _ = await stt.transcribe(
            Turn(role=TurnRole.USER, text="one two three", started_at_ms=0, ended_at_ms=500)
        )
        assert text == "one two three"


class TestPipelineRun:
    async def test_emits_one_turn_run_per_user_turn(self) -> None:
        pipeline = VoicePipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())
        conv = default_golden_set()[0]
        run = await pipeline.run(conv)
        # postgres-replication has 2 user turns + 1 agent turn.
        n_user = sum(1 for t in conv.turns if t.role.value == "user")
        assert run.user_turns_played == n_user
        assert len(run.turn_runs) == n_user

    async def test_spans_include_vad_and_tts_first_byte(self) -> None:
        pipeline = VoicePipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())
        run = await pipeline.run(default_golden_set()[0])
        for tr in run.turn_runs:
            names = {s.name for s in tr.spans}
            assert "vad_end" in names
            assert "tts_first_byte" in names

    async def test_false_trigger_injection(self) -> None:
        pipeline = VoicePipeline(
            stt=MockSTT(), llm=MockLLM(), tts=MockTTS(), false_trigger_rate=1.0
        )
        run = await pipeline.run(default_golden_set()[0])
        assert any(tr.false_trigger for tr in run.turn_runs)

    async def test_false_trigger_rate_zero_injects_none(self) -> None:
        pipeline = VoicePipeline(
            stt=MockSTT(), llm=MockLLM(), tts=MockTTS(), false_trigger_rate=0.0
        )
        run = await pipeline.run(default_golden_set()[0])
        assert all(not tr.false_trigger for tr in run.turn_runs)

    async def test_false_trigger_rate_one_injects_every_turn(self) -> None:
        # rate=1.0 means a false-trigger after every user turn — the
        # pipeline emits N user turn-runs interleaved with N synthetic
        # false-trigger turn-runs.
        pipeline = VoicePipeline(
            stt=MockSTT(), llm=MockLLM(), tts=MockTTS(), false_trigger_rate=1.0
        )
        conv = default_golden_set()[0]
        n_user = sum(1 for t in conv.turns if t.role.value == "user")
        run = await pipeline.run(conv)
        assert sum(1 for tr in run.turn_runs if tr.false_trigger) == n_user
        assert len(run.turn_runs) == 2 * n_user

    async def test_false_trigger_rate_injects_per_turn(self) -> None:
        # Bernoulli per turn with a fixed seed — over a long synthetic
        # conversation we expect ~rate * N injections, and the count must
        # be deterministic for that seed (the count is what the eval harness
        # exposes via false_trigger_rate, so reproducibility matters).
        n = 200
        turns = [
            Turn(role=TurnRole.USER, text=f"hi {i}", started_at_ms=i * 1000, ended_at_ms=i * 1000 + 500)
            for i in range(n)
        ]
        conv = Conversation(conv_id="long", topic="t", turns=turns, gold_facts=[])

        pipeline_a = VoicePipeline(
            stt=MockSTT(),
            llm=MockLLM(),
            tts=MockTTS(),
            false_trigger_rate=0.5,
            false_trigger_seed=42,
        )
        run_a = await pipeline_a.run(conv)
        injected_a = sum(1 for tr in run_a.turn_runs if tr.false_trigger)

        # Same seed -> same draws.
        pipeline_b = VoicePipeline(
            stt=MockSTT(),
            llm=MockLLM(),
            tts=MockTTS(),
            false_trigger_rate=0.5,
            false_trigger_seed=42,
        )
        run_b = await pipeline_b.run(conv)
        injected_b = sum(1 for tr in run_b.turn_runs if tr.false_trigger)
        assert injected_a == injected_b
        # 0.5 * 200 = 100 expected; allow generous slack for Bernoulli noise
        # while still rejecting the old "any rate > 0 -> exactly one" bug.
        assert 70 < injected_a < 130
        # And materially different from the rate=1.0 ceiling (which would
        # be 200) and rate=0 floor (0).
        assert 0 < injected_a < n

    async def test_history_includes_prior_agent_replies(self) -> None:
        # The LLM's history argument must include both user and agent turns
        # from previous exchanges.  Before the fix, only user turns were
        # appended, so the history_len span attr grew but the actual context
        # passed to llm.reply was user-only.
        captured: list[list[Turn]] = []

        class CapturingLLM:
            async def reply(
                self,
                history: list[Turn],
                last_user_text: str,
                gold_facts: list[str],
            ) -> tuple[str, list[PipelineSpan]]:
                captured.append(list(history))
                return "agent reply", []

        pipeline = VoicePipeline(stt=MockSTT(), llm=CapturingLLM(), tts=MockTTS())
        # Use postgres-replication which has 2 user turns so we get a
        # non-trivial history on the second call.
        conv = default_golden_set()[0]
        await pipeline.run(conv)

        # Second call should have seen [user_turn_0, agent_reply_0] in history.
        assert len(captured) == 2
        first_history = captured[0]
        second_history = captured[1]
        # First call: history is empty (no prior turns yet).
        assert first_history == []
        # Second call: must include both the first user turn and the agent reply.
        assert len(second_history) == 2
        assert second_history[0].role.value == "user"
        assert second_history[1].role.value == "agent"
        assert second_history[1].text == "agent reply"

    async def test_llm_surfaces_matching_gold_fact(self) -> None:
        pipeline = VoicePipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())
        # postgres-replication has facts mentioning "WAL" / "replication"
        run = await pipeline.run(default_golden_set()[0])
        # The first user turn says "quiz me on postgres replication" — the
        # LLM should pick a gold fact whose first words include "postgres" /
        # "replication". Exact text comes from the gold facts.
        assert any("replication" in tr.agent_reply.lower() for tr in run.turn_runs)
