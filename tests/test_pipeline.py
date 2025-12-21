"""Pipeline tests — adapter contracts + run-shape invariants."""

from __future__ import annotations

from voice_eval_lab.eval.golden import default_golden_set
from voice_eval_lab.models import Turn, TurnRole
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

    async def test_llm_surfaces_matching_gold_fact(self) -> None:
        pipeline = VoicePipeline(stt=MockSTT(), llm=MockLLM(), tts=MockTTS())
        # postgres-replication has facts mentioning "WAL" / "replication"
        run = await pipeline.run(default_golden_set()[0])
        # The first user turn says "quiz me on postgres replication" — the
        # LLM should pick a gold fact whose first words include "postgres" /
        # "replication". Exact text comes from the gold facts.
        assert any("replication" in tr.agent_reply.lower() for tr in run.turn_runs)
