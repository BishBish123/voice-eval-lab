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

import asyncio
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Protocol

from voice_eval_lab.models import (
    Conversation,
    ConversationRun,
    PipelineSpan,
    Turn,
    TurnRole,
    TurnRun,
)


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
    """Returns the gold transcript with a configurable WER injected.

    The substitution rate can be overridden per-turn via
    `Turn.wer_substitution_rate`, which is useful for A/B-style scenarios
    where a single conversation should mix clean and noisy audio.
    """

    wer_substitution_rate: float = 0.0
    latency_ms: int = 80

    async def transcribe(self, turn: Turn) -> tuple[str, list[PipelineSpan]]:
        rate = (
            turn.wer_substitution_rate
            if turn.wer_substitution_rate is not None
            else self.wer_substitution_rate
        )
        text = _inject_wer(turn.text, rate)
        spans = [
            PipelineSpan(
                name="stt.transcribe",
                started_at_ms=turn.ended_at_ms,
                ended_at_ms=turn.ended_at_ms + self.latency_ms,
                attrs={"engine": "mock", "wer_injected": str(rate)},
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

    async def stream(
        self,
        history: list[Turn],
        last_user_text: str,
        gold_facts: list[str],
        chunk_size: int = 4,
    ) -> AsyncIterator[str]:
        """Token-streaming variant — yields word chunks of the same reply.

        Real LLMs interleave with TTS so the first audio byte fires before
        the LLM finishes. Tests use this to assert the streaming contract
        without coupling to wall-clock time.
        """
        text, _ = await self.reply(history, last_user_text, gold_facts)
        words = text.split()
        for i in range(0, len(words), chunk_size):
            yield " ".join(words[i : i + chunk_size])


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


@dataclass
class FlakyTTS:
    """A TTS adapter that fails the first `fail_n` calls then succeeds.

    Used to exercise the `RetryingTTS` wrapper.
    """

    inner: TTS
    fail_n: int = 1
    _calls: int = field(default=0, init=False)

    async def synthesize(self, text: str) -> tuple[int, list[PipelineSpan]]:
        self._calls += 1
        if self._calls <= self.fail_n:
            raise RuntimeError(f"FlakyTTS scheduled failure #{self._calls}")
        return await self.inner.synthesize(text)


# Transient-error types worth retrying. KeyboardInterrupt, SystemExit,
# CancelledError, programming bugs (TypeError, AttributeError) — all of
# those should propagate immediately rather than be retried.
RETRYABLE_TTS_ERRORS: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
    RuntimeError,
)


@dataclass
class RetryingTTS:
    """Decorator that wraps any TTS protocol with exponential-backoff retries.

    Real TTS streams glitch under load — Cartesia / ElevenLabs both
    document transient 5xx. The retry policy is intentionally bounded:
    voice agents that can't yield first byte under ~600ms are unusable,
    so we cap at `max_attempts` and surface the exception otherwise.

    Only a narrow set of exceptions counts as retryable
    (`RETRYABLE_TTS_ERRORS`); programmer errors (TypeError, AttributeError)
    and lifecycle signals (KeyboardInterrupt, asyncio.CancelledError)
    propagate immediately so we don't burn the retry budget on bugs.
    """

    inner: TTS
    max_attempts: int = 3
    base_delay_ms: int = 25

    async def synthesize(self, text: str) -> tuple[int, list[PipelineSpan]]:
        # Cumulative wall-clock from the first invocation. The downstream
        # tts_first_byte span uses whatever we return here, so reporting
        # only the successful attempt's elapsed time hides retry/backoff
        # cost from latency budgets and headline metrics.
        t0 = time.monotonic()
        last_exc: BaseException | None = None
        retry_spans: list[PipelineSpan] = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                _inner_first_byte, spans = await self.inner.synthesize(text)
                cumulative_ms = int((time.monotonic() - t0) * 1000)
                return cumulative_ms, [*retry_spans, *spans]
            except RETRYABLE_TTS_ERRORS as exc:
                last_exc = exc
                retry_spans.append(
                    PipelineSpan(
                        name="tts.retry",
                        started_at_ms=0,
                        ended_at_ms=self.base_delay_ms * (2 ** (attempt - 1)),
                        attrs={"attempt": str(attempt), "error": type(exc).__name__},
                    )
                )
                if attempt < self.max_attempts:
                    await asyncio.sleep(0)  # cooperative yield, no real wall delay
        assert last_exc is not None
        raise last_exc


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


@dataclass
class LatencyBudget:
    """Middleware that flags turns whose first-byte latency blows the budget.

    Voice agents have a hard ceiling — most users disengage past ~700ms
    end-to-end. The middleware emits a `latency_budget.exceeded` span on
    every turn that crosses `budget_ms`, which the metric layer can
    surface in the report later.
    """

    budget_ms: int = 700

    def annotate(self, run: ConversationRun) -> ConversationRun:
        for tr in run.turn_runs:
            vad = next((s for s in tr.spans if s.name == "vad_end"), None)
            fb = next((s for s in tr.spans if s.name == "tts_first_byte"), None)
            if vad is None or fb is None:
                continue
            latency = fb.ended_at_ms - vad.ended_at_ms
            if latency > self.budget_ms:
                tr.spans.append(
                    PipelineSpan(
                        name="latency_budget.exceeded",
                        started_at_ms=fb.ended_at_ms,
                        ended_at_ms=fb.ended_at_ms,
                        attrs={
                            "budget_ms": str(self.budget_ms),
                            "observed_ms": str(latency),
                        },
                    )
                )
        return run


@dataclass
class VoicePipeline:
    stt: STT
    llm: LLM
    tts: TTS
    barge_in_yield_ms: int = 100
    false_trigger_rate: float = 0.0
    latency_budget: LatencyBudget | None = None

    async def run(self, conversation: Conversation) -> ConversationRun:
        history: list[Turn] = []
        runs: list[TurnRun] = []
        played = 0
        for i, user_turn in enumerate(_user_turns(conversation.turns)):
            played += 1
            stt_text, stt_spans = await self.stt.transcribe(user_turn)
            llm_text, llm_spans = await self.llm.reply(history, stt_text, conversation.gold_facts)
            tts_first_byte, tts_spans = await self.tts.synthesize(llm_text)

            spans = [
                PipelineSpan(
                    name="vad_end",
                    started_at_ms=user_turn.ended_at_ms,
                    ended_at_ms=user_turn.ended_at_ms,
                    attrs={"role": "user"},
                ),
                *stt_spans,
                *llm_spans,
                *tts_spans,
                PipelineSpan(
                    name="tts_first_byte",
                    started_at_ms=user_turn.ended_at_ms
                    + sum(s.ended_at_ms - s.started_at_ms for s in stt_spans + llm_spans),
                    ended_at_ms=user_turn.ended_at_ms
                    + sum(s.ended_at_ms - s.started_at_ms for s in stt_spans + llm_spans)
                    + tts_first_byte,
                    attrs={},
                ),
            ]

            interrupted = user_turn.interrupted
            if interrupted:
                # When the user barges in, the pipeline yields shortly after
                # vad_end. The yield latency is captured as a span so the
                # `barge_in_latency_p95` metric can read it.
                spans.append(
                    PipelineSpan(
                        name="barge_in.yield",
                        started_at_ms=user_turn.ended_at_ms,
                        ended_at_ms=user_turn.ended_at_ms + self.barge_in_yield_ms,
                        attrs={"budget_ms": str(self.barge_in_yield_ms)},
                    )
                )
            runs.append(
                TurnRun(
                    user_turn_index=i,
                    transcribed_text=stt_text,
                    agent_reply=llm_text,
                    interrupted=interrupted,
                    false_trigger=False,  # mock pipeline doesn't generate false triggers
                    spans=spans,
                )
            )
            history.append(user_turn)

        # Optionally inject false triggers (one synthetic at the end) for
        # eval-harness exercise.
        if self.false_trigger_rate > 0 and runs:
            runs.append(
                TurnRun(
                    user_turn_index=len(runs),
                    transcribed_text="",
                    agent_reply="...?",
                    interrupted=False,
                    false_trigger=True,
                    spans=[],
                )
            )

        run = ConversationRun(
            conv_id=conversation.conv_id,
            topic=conversation.topic,
            user_turns_played=played,
            turn_runs=runs,
        )
        if self.latency_budget is not None:
            run = self.latency_budget.annotate(run)
        return run


def _user_turns(turns: Iterable[Turn]) -> Iterable[Turn]:
    for t in turns:
        if t.role is TurnRole.USER:
            yield t


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
