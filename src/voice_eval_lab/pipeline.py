"""Reference voice pipeline (deterministic, no real audio).

Real-pipeline shape:

    audio frames → VAD → STT → LLM → TTS → audio frames

This module ships *exactly* that lifecycle but with mock STT/LLM/TTS
adapters that emit the structured `PipelineSpan` records the eval harness
consumes. It exists so the harness can be exercised end-to-end without
booking LiveKit + Deepgram + Cartesia accounts. Real adapters drop in
behind the same Protocol surface.

Latency semantics: the mock pipeline measures *full-completion* latency
(the LLM and TTS calls return their full result before the next stage
starts), not streaming / interleaved latency.  ``MockLLM.stream``
exists as a Protocol exercise but is not wired into ``VoicePipeline.run``;
the latency numbers reported by the eval reflect the full round-trip, not
the time-to-first-token.
"""

from __future__ import annotations

import asyncio
import random
import re
import unicodedata
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
    """Returns a reply that mentions the first matching gold fact when present.

    The matching heuristic is intentionally tolerant of cosmetic
    differences between the user transcript and the gold fact:
    case, underscore-vs-space (``ef_search`` vs ``ef search``), Unicode
    NFKC variants, and runs of whitespace. The earlier raw-substring
    check produced 0% faithfulness on conversations whose users phrased
    fact tokens with spaces while the fact stored them with underscores.
    """

    latency_ms: int = 120

    async def reply(
        self,
        history: list[Turn],
        last_user_text: str,
        gold_facts: list[str],
    ) -> tuple[str, list[PipelineSpan]]:
        user_tokens = _normalize_tokens(last_user_text)
        match = next(
            (f for f in gold_facts if _user_mentions_fact_lead(user_tokens, f)),
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

        This method implements the streaming *Protocol* surface so real
        adapters can be swapped in, but ``VoicePipeline.run`` uses the
        non-streaming ``reply`` path.  Latency numbers in the eval report
        therefore reflect full-completion latency, not time-to-first-token.
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
        # The downstream tts_first_byte span uses whatever we return here,
        # so reporting only the successful attempt's elapsed time hides
        # retry/backoff cost from latency budgets and headline metrics.
        # We accumulate the configured backoff for each retry attempt and
        # add the successful inner call's reported first-byte ms — that
        # way the returned latency is deterministic (no wall-clock jitter
        # from `time.monotonic()`) and the `tts.retry` span durations
        # correspond to wall-clock time we actually slept.
        last_exc: BaseException | None = None
        retry_spans: list[PipelineSpan] = []
        total_backoff_ms = 0
        for attempt in range(1, self.max_attempts + 1):
            try:
                inner_first_byte, spans = await self.inner.synthesize(text)
                return total_backoff_ms + inner_first_byte, [*retry_spans, *spans]
            except RETRYABLE_TTS_ERRORS as exc:
                last_exc = exc
                backoff_ms = self.base_delay_ms * (2 ** (attempt - 1))
                retry_spans.append(
                    PipelineSpan(
                        name="tts.retry",
                        started_at_ms=0,
                        ended_at_ms=backoff_ms,
                        attrs={"attempt": str(attempt), "error": type(exc).__name__},
                    )
                )
                if attempt < self.max_attempts:
                    # Actually wait the configured backoff so the
                    # `tts.retry` span duration matches real wall-clock
                    # time. The previous `sleep(0)` only yielded
                    # cooperatively, so the advertised backoff was
                    # invisible to latency consumers.
                    total_backoff_ms += backoff_ms
                    await asyncio.sleep(backoff_ms / 1000.0)
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
    """Reference voice pipeline. ``false_trigger_rate`` is the per-user-turn
    Bernoulli probability of emitting a synthetic false-trigger turn (a
    turn with ``false_trigger=True`` and an empty body) right after the
    sampled user turn. ``false_trigger_seed`` makes that sampling
    reproducible for tests and for golden-set scoring; passing ``None``
    falls back to system entropy.

    Rate semantics:
      * 0.0 -> never inject (and the RNG is never consulted)
      * 1.0 -> inject one synthetic after every user turn
      * 0<r<1 -> independent Bernoulli per user turn
    """

    stt: STT
    llm: LLM
    tts: TTS
    barge_in_yield_ms: int = 100
    false_trigger_rate: float = 0.0
    false_trigger_seed: int | None = 0
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
            # Per-turn Bernoulli sample: when a seed is configured, the
            # draw is derived from (false_trigger_seed, conv_id, turn_index)
            # so draws are independent across conversations and across turns
            # within a conversation.  A single per-conversation RNG shared
            # across all turns meant that adding/removing conversations would
            # shift draws for every subsequent conversation in the corpus
            # run; deriving per-turn isolates each draw completely.
            # When false_trigger_seed is None, a fresh unseeded Random()
            # provides system-entropy draws as before.
            # random.Random only accepts None/int/float/str/bytes as seed;
            # we encode the compound key as a str so all types are covered.
            if self.false_trigger_rate > 0 and (
                random.Random(  # noqa: S311 - eval RNG, not crypto
                    None
                    if self.false_trigger_seed is None
                    else f"{self.false_trigger_seed}:{conversation.conv_id}:{i}"
                ).random()
                < self.false_trigger_rate
            ):
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
            history.append(user_turn)
            # Append the agent's reply so subsequent LLM calls see the
            # full dialogue rather than only the user side.  Omitting the
            # agent turns was the original bug: multi-turn conversations
            # fed the LLM only a user-only history, making history_len
            # grow but the actual context one-sided.
            history.append(
                Turn(
                    role=TurnRole.AGENT,
                    text=llm_text,
                    started_at_ms=user_turn.ended_at_ms,
                    ended_at_ms=user_turn.ended_at_ms + self.barge_in_yield_ms,
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


def _normalize_tokens(text: str) -> list[str]:
    """Lowercase + NFKC-normalize + split on any non-letter/digit run.

    Treats underscores, hyphens, punctuation, and whitespace as token
    boundaries so ``ef_search``, ``ef-search``, and ``ef search`` all
    produce the same token list. Empty tokens are dropped.
    """
    nfkc = unicodedata.normalize("NFKC", text)
    return [t for t in re.split(r"[^A-Za-z0-9]+", nfkc.lower()) if t]


def _user_mentions_fact_lead(user_tokens: list[str], fact: str) -> bool:
    """True if any of the first three tokens of `fact` appear in `user_tokens`.

    Match semantics mirror the original "any of fact's first 3 words is in
    user text" heuristic, but on a normalized token list rather than raw
    substring of the lowercase string. That fixes cosmetic mismatches like
    ``ef_search`` vs ``ef search`` (different whitespace, same intent)
    without changing which conversations match.
    """
    fact_lead = _normalize_tokens(fact)[:3]
    if not fact_lead:
        return False
    user_set = set(user_tokens)
    return any(tok in user_set for tok in fact_lead)


def _inject_wer(text: str, sub_rate: float) -> str:
    """Substitute `sub_rate` fraction of words with a fixed token to drive WER.

    `sub_rate` must lie in [0, 1]. Out-of-band values are rejected
    explicitly so a typo (e.g. ``--wer-rate 5`` instead of ``0.5``)
    surfaces as a `ValueError` rather than crashing inside the index
    arithmetic with a confusing IndexError or silently disabling WER.
    """
    if not 0.0 <= sub_rate <= 1.0:
        raise ValueError(
            f"wer_substitution_rate must be in [0.0, 1.0], got {sub_rate!r}"
        )
    if sub_rate == 0.0:
        return text
    words = text.split()
    # Cap defensively: at sub_rate=1.0 every word is substituted.
    n_sub = min(len(words), int(len(words) * sub_rate))
    if n_sub <= 0:
        return text
    # Substitute the first n_sub words deterministically.
    for i in range(n_sub):
        words[i] = "WERR"
    return " ".join(words)
