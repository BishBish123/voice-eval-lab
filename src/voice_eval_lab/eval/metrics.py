"""Voice-agent metrics + the per-run scorer.

(This module sits in `voice_eval_lab.eval`, the harness package — it does
not call Python's builtin code-evaluator anywhere.)
"""

from __future__ import annotations

import jiwer

from voice_eval_lab.models import (
    Conversation,
    ConversationRun,
    PipelineSpan,
    TurnLatencyStats,
    TurnRole,
    TurnRun,
)


def turn_latency_stats(turn_runs: list[TurnRun]) -> TurnLatencyStats:
    """Per-turn latency = first-byte minus vad_end, in ms."""
    samples: list[float] = []
    for tr in turn_runs:
        vad = _find_span(tr.spans, "vad_end")
        first_byte = _find_span(tr.spans, "tts_first_byte")
        if vad is None or first_byte is None:
            continue
        samples.append(float(first_byte.ended_at_ms - vad.ended_at_ms))
    return _percentile_stats(samples)


def transcription_wer(conversation: Conversation, run: ConversationRun) -> float:
    user_turns = [t for t in conversation.turns if t.role is TurnRole.USER]
    references: list[str] = []
    hypotheses: list[str] = []
    for ut, tr in zip(user_turns, run.turn_runs, strict=False):
        references.append(ut.text)
        hypotheses.append(tr.transcribed_text)
    if not references:
        return 0.0
    out = jiwer.wer(references, hypotheses)
    return float(out)


def response_faithfulness(conversation: Conversation, run: ConversationRun) -> float:
    """Fraction of agent replies that quote at least one gold fact substring.

    Proxy until an LLM judge is plugged in via the same Protocol.
    """
    if not conversation.gold_facts:
        return 1.0  # nothing to disagree with
    score = 0.0
    n = 0
    for tr in run.turn_runs:
        if tr.false_trigger:
            continue
        n += 1
        if any(fact.lower() in tr.agent_reply.lower() for fact in conversation.gold_facts):
            score += 1.0
    return score / n if n else 0.0


def barge_in_success_rate(conversation: Conversation, run: ConversationRun) -> float:
    """Fraction of user-interrupted turns the pipeline yielded inside the budget."""
    user_turns = [t for t in conversation.turns if t.role is TurnRole.USER]
    interruptible = [t for t in user_turns if t.interrupted]
    if not interruptible:
        return 1.0
    yielded = 0
    for ut in interruptible:
        idx = user_turns.index(ut)
        if idx >= len(run.turn_runs):
            continue
        tr = run.turn_runs[idx]
        if any(s.name == "tts_first_byte" for s in tr.spans):
            yielded += 1
    return yielded / len(interruptible)


def _find_span(spans: list[PipelineSpan], name: str) -> PipelineSpan | None:
    for s in spans:
        if s.name == name:
            return s
    return None


def _percentile_stats(samples: list[float]) -> TurnLatencyStats:
    if not samples:
        return TurnLatencyStats(p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, n=0)
    sorted_samples = sorted(samples)
    n = len(sorted_samples)

    def pct(p: float) -> float:
        idx = min(n - 1, int(p * n))
        return float(sorted_samples[idx])

    return TurnLatencyStats(p50_ms=pct(0.50), p95_ms=pct(0.95), p99_ms=pct(0.99), n=n)
