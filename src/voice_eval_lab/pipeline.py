"""Reference voice pipeline (deterministic, no real audio).

Real-pipeline shape:

    audio frames → VAD → STT (streaming) → LLM (streaming) → TTS (streaming) → audio frames

This module ships *exactly* that lifecycle but with mock STT/LLM/TTS
adapters that emit the structured `PipelineSpan` records the eval harness
consumes. It exists so the harness can be exercised end-to-end without
booking LiveKit + Deepgram + Cartesia accounts. Real adapters drop in
behind the same Protocol surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from voice_eval_lab.models import PipelineSpan, Turn


class STT(Protocol):
    async def transcribe(self, turn: Turn) -> tuple[str, list[PipelineSpan]]: ...


class LLM(Protocol):
    async def reply(
        self,
        history: list[Turn],
        last_user_text: str,
        gold_facts: list[str],
    ) -> tuple[str, list[PipelineSpan]]: ...


class TTS(Protocol):
    async def synthesize(self, text: str) -> tuple[int, list[PipelineSpan]]:
        """Returns (first-byte ms, spans)."""
        ...


# ---------------------------------------------------------------------------
# Mock adapters — deterministic, single-process
# ---------------------------------------------------------------------------


@dataclass
class MockSTT:
    """Returns the gold transcript with a configurable WER injected."""

    wer_substitution_rate: float = 0.0
    latency_ms: int = 80

    async def transcribe(self, turn: Turn) -> tuple[str, list[PipelineSpan]]:
        text = _inject_wer(turn.text, self.wer_substitution_rate)
        spans = [
            PipelineSpan(
                name="stt.transcribe",
                started_at_ms=turn.ended_at_ms,
                ended_at_ms=turn.ended_at_ms + self.latency_ms,
                attrs={"engine": "mock", "wer_injected": str(self.wer_substitution_rate)},
            )
        ]
        return text, spans


@dataclass
class MockLLM:
    """Returns a reply that mentions the first matching gold fact when present."""

    latency_ms: int = 120

    async def reply(
        self,
        history: list[Turn],
        last_user_text: str,
        gold_facts: list[str],
    ) -> tuple[str, list[PipelineSpan]]:
        match = next(
            (
                f
                for f in gold_facts
                if any(w in last_user_text.lower() for w in f.lower().split()[:3])
            ),
            None,
        )
        text = match if match else f"I don't have a confident answer about {last_user_text!r}."
        spans = [
            PipelineSpan(
                name="llm.reply",
                started_at_ms=0,
                ended_at_ms=self.latency_ms,
                attrs={"model": "mock", "history_len": str(len(history))},
            )
        ]
        return text, spans


@dataclass
class MockTTS:
    first_byte_ms: int = 75

    async def synthesize(self, text: str) -> tuple[int, list[PipelineSpan]]:
        spans = [
            PipelineSpan(
                name="tts.synthesize",
                started_at_ms=0,
                ended_at_ms=self.first_byte_ms,
                attrs={"engine": "mock", "chars": str(len(text))},
            )
        ]
        return self.first_byte_ms, spans


def _inject_wer(text: str, sub_rate: float) -> str:
    """Substitute `sub_rate` fraction of words with a fixed token to drive WER."""
    if sub_rate <= 0:
        return text
    words = text.split()
    n_sub = int(len(words) * sub_rate)
    if n_sub <= 0:
        return text
    # Substitute the first n_sub words deterministically.
    for i in range(n_sub):
        words[i] = "WERR"
    return " ".join(words)
