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
