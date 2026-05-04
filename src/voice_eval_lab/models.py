"""Pydantic types for the eval harness + reference pipeline."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class TurnRole(StrEnum):
    USER = "user"
    AGENT = "agent"


class Turn(BaseModel):
    """One side of a turn — what the user said + what the agent said back."""

    role: TurnRole
    text: str
    started_at_ms: int = Field(description="Audio frame index when speech started")
    ended_at_ms: int = Field(description="Audio frame index when speech ended")
    interrupted: bool = False
    # Per-turn WER override for the mock STT (None = use the global rate).
    wer_substitution_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Per-turn override for the mock STT word substitution rate (0..1).",
    )


class Conversation(BaseModel):
    """A whole golden-set conversation: user turns, expected agent turns, gold facts."""

    conv_id: str
    topic: str
    turns: list[Turn]
    gold_facts: list[str] = Field(
        default_factory=list,
        description="Atomic statements the agent's responses must remain consistent with.",
    )


class PipelineSpan(BaseModel):
    """One time-stamped step inside the voice pipeline.

    Maps cleanly to an OpenTelemetry span; the eval harness derives turn
    latency by joining `vad_end` -> `tts_first_byte` per turn.
    """

    name: str
    started_at_ms: int
    ended_at_ms: int
    attrs: dict[str, str] = Field(default_factory=dict)


class TurnRun(BaseModel):
    """Pipeline output for one user turn — all the spans + the final agent reply."""

    user_turn_index: int
    transcribed_text: str
    agent_reply: str
    interrupted: bool = False
    false_trigger: bool = Field(
        default=False,
        description="True if the pipeline started replying to a non-utterance (cough, room noise).",
    )
    spans: list[PipelineSpan] = Field(default_factory=list)


class ConversationRun(BaseModel):
    """Per-conversation pipeline output the eval harness scores against."""

    conv_id: str
    topic: str
    user_turns_played: int
    turn_runs: list[TurnRun]


# ---------------------------------------------------------------------------
# Eval result types
# ---------------------------------------------------------------------------


class TurnLatencyStats(BaseModel):
    p50_ms: float
    p95_ms: float
    p99_ms: float
    n: int


class ConversationScore(BaseModel):
    conv_id: str
    topic: str

    turn_latency: TurnLatencyStats
    transcription_wer: float = Field(description="Word-error rate, 0..1, lower is better")
    response_faithfulness: float = Field(
        description="Fraction of agent replies grounded in gold_facts."
    )
    barge_in_success_rate: float = Field(
        description="Of user-interrupted turns, fraction the agent yielded within 200ms."
    )
    false_trigger_rate: float = Field(
        description="Fraction of agent replies started while the user was not speaking."
    )
    barge_in_latency_p95_ms: float = Field(
        default=0.0,
        description="p95 latency from interrupt to TTS yield (ms); 0 when no barge-ins.",
    )
    tts_first_byte_jitter_ms: float = Field(
        default=0.0,
        description="Std-dev of first-byte latency across turns (ms).",
    )
    endpointing_accuracy: float | None = Field(
        default=1.0,
        description=(
            "Fraction of user turns where VAD end aligned with utterance end (±tolerance). "
            "None when no user turn produced a vad_end span (no signal — distinct from "
            "0.0 which means every measured turn was misaligned)."
        ),
    )
    llm_decisiveness: float = Field(
        default=1.0,
        description="Fraction of agent replies that don't hedge ('I don't know', 'maybe', ...).",
    )


class EvalReport(BaseModel):
    n_conversations: int
    aggregate_turn_latency: TurnLatencyStats
    aggregate_wer: float
    aggregate_faithfulness: float | None = Field(
        default=None,
        description=(
            "Pooled fraction of agent replies grounded in gold facts: "
            "sum(grounded_replies) / sum(replies_with_gold_facts). None when no "
            "conversation in the run had any gold facts (no signal — distinct from "
            "1.0 which means every reply quoted a fact)."
        ),
    )
    aggregate_barge_in_success: float | None = Field(
        default=None,
        description=(
            "Pooled fraction of interrupted turns the pipeline yielded inside the budget, "
            "computed as sum(yielded) / sum(interrupted) across the run. None when the "
            "entire run had no interrupted turns (no signal — distinct from 1.0 which "
            "means every interrupt was handled inside budget)."
        ),
    )
    aggregate_false_trigger_rate: float | None = Field(
        default=None,
        description=(
            "Pooled fraction of turns the pipeline marked as false_trigger: "
            "sum(false_trigger_turns) / sum(turn_runs). None when no conversation in "
            "the run produced any turn_runs at all."
        ),
    )
    aggregate_barge_in_latency_p95_ms: float | None = Field(
        default=None,
        description=(
            "p95 of barge_in.yield duration pooled across the whole run, in ms. "
            "None when no conversation produced a yield span (no signal)."
        ),
    )
    aggregate_tts_first_byte_jitter_ms: float | None = Field(
        default=None,
        description=(
            "Population stddev of first-byte latency pooled across the run, in ms. "
            "None when no conversation produced first-byte spans."
        ),
    )
    aggregate_endpointing_accuracy: float | None = Field(
        default=1.0,
        description=(
            "Mean of per-conversation endpointing scores, skipping conversations with no "
            "VAD signal. None when no conversation produced any vad_end span."
        ),
    )
    aggregate_llm_decisiveness: float | None = Field(
        default=None,
        description=(
            "Pooled fraction of agent replies that don't hedge: "
            "sum(decisive_replies) / sum(non_false_trigger_replies). None when no "
            "conversation had any non-false-trigger replies."
        ),
    )
    per_conversation: list[ConversationScore]
