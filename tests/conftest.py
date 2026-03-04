"""Shared test fixtures for the voice-eval-lab suite."""

from __future__ import annotations

from voice_eval_lab.models import (
    Conversation,
    ConversationRun,
    PipelineSpan,
    Turn,
    TurnRole,
    TurnRun,
)


def make_conv(
    turns: list[Turn],
    *,
    gold: list[str] | None = None,
    conv_id: str = "c",
    topic: str = "t",
) -> Conversation:
    return Conversation(conv_id=conv_id, topic=topic, turns=turns, gold_facts=gold or [])


def make_user(
    text: str,
    *,
    start: int = 0,
    end: int = 1000,
    interrupted: bool = False,
) -> Turn:
    return Turn(
        role=TurnRole.USER,
        text=text,
        started_at_ms=start,
        ended_at_ms=end,
        interrupted=interrupted,
    )


def make_turn_run(
    *,
    transcribed: str = "x",
    reply: str = "y",
    vad_end_ms: int = 0,
    first_byte_ms: int = 100,
    interrupted: bool = False,
    false_trigger: bool = False,
    barge_yield_ms: int | None = None,
    user_turn_index: int = 0,
) -> TurnRun:
    spans: list[PipelineSpan] = [
        PipelineSpan(name="vad_end", started_at_ms=vad_end_ms, ended_at_ms=vad_end_ms),
        PipelineSpan(name="tts_first_byte", started_at_ms=first_byte_ms, ended_at_ms=first_byte_ms),
    ]
    if barge_yield_ms is not None:
        spans.append(
            PipelineSpan(
                name="barge_in.yield",
                started_at_ms=vad_end_ms,
                ended_at_ms=vad_end_ms + barge_yield_ms,
            )
        )
    return TurnRun(
        user_turn_index=user_turn_index,
        transcribed_text=transcribed,
        agent_reply=reply,
        interrupted=interrupted,
        false_trigger=false_trigger,
        spans=spans,
    )


def make_run(turn_runs: list[TurnRun], *, conv_id: str = "c") -> ConversationRun:
    return ConversationRun(
        conv_id=conv_id,
        topic="t",
        user_turns_played=len(turn_runs),
        turn_runs=turn_runs,
    )


__all__ = ["make_conv", "make_run", "make_turn_run", "make_user"]
