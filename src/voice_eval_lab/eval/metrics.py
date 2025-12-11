"""Five voice-agent metrics + the per-run scorer.

(This module sits in `voice_eval_lab.eval`, the harness package — it does
not call Python's builtin `eval()` anywhere.)

- `turn_latency_stats` — vad_end -> tts_first_byte per turn, p50/p95/p99
- `transcription_wer` — jiwer over (gold transcript, pipeline transcript)
- `response_faithfulness` — fraction of agent replies that mention any
  gold fact (proxy for grounded-ness; LLM judge would replace this in
  production while keeping the API identical)
- `barge_in_success_rate` — of user-interrupted turns, fraction the
  pipeline yielded inside `barge_in_yield_ms`
- `false_trigger_rate` — fraction of turns marked `false_trigger=True`
"""

from __future__ import annotations

import jiwer

from voice_eval_lab.models import (
    Conversation,
    ConversationRun,
    ConversationScore,
    EvalReport,
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
    """Fraction of user-interrupted turns the pipeline yielded inside the budget.

    With the deterministic mock pipeline this is 1.0 when an interrupted
    user turn has a tts_first_byte span; the test fixture exercises the
    failure path by withholding the yield span.
    """
    user_turns = [t for t in conversation.turns if t.role is TurnRole.USER]
    interruptible = [t for t in user_turns if t.interrupted]
    if not interruptible:
        return 1.0
    yielded = 0
    for ut in interruptible:
        # Find the matching turn run by index.
        idx = user_turns.index(ut)
        if idx >= len(run.turn_runs):
            continue
        tr = run.turn_runs[idx]
        if any(s.name == "tts_first_byte" for s in tr.spans):
            yielded += 1
    return yielded / len(interruptible)


def false_trigger_rate(run: ConversationRun) -> float:
    if not run.turn_runs:
        return 0.0
    return sum(1 for tr in run.turn_runs if tr.false_trigger) / len(run.turn_runs)


# ---------------------------------------------------------------------------
# Per-conversation + aggregate scoring
# ---------------------------------------------------------------------------


def score_conversation(conversation: Conversation, run: ConversationRun) -> ConversationScore:
    return ConversationScore(
        conv_id=conversation.conv_id,
        topic=conversation.topic,
        turn_latency=turn_latency_stats(run.turn_runs),
        transcription_wer=transcription_wer(conversation, run),
        response_faithfulness=response_faithfulness(conversation, run),
        barge_in_success_rate=barge_in_success_rate(conversation, run),
        false_trigger_rate=false_trigger_rate(run),
    )


def score_run(pairs: list[tuple[Conversation, ConversationRun]]) -> EvalReport:
    per_conv = [score_conversation(c, r) for c, r in pairs]
    if not per_conv:
        empty = TurnLatencyStats(p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, n=0)
        return EvalReport(
            n_conversations=0,
            aggregate_turn_latency=empty,
            aggregate_wer=0.0,
            aggregate_faithfulness=0.0,
            aggregate_barge_in_success=0.0,
            aggregate_false_trigger_rate=0.0,
            per_conversation=[],
        )

    all_latencies: list[float] = []
    for _c, r in pairs:
        for tr in r.turn_runs:
            vad = _find_span(tr.spans, "vad_end")
            fb = _find_span(tr.spans, "tts_first_byte")
            if vad and fb:
                all_latencies.append(float(fb.ended_at_ms - vad.ended_at_ms))

    return EvalReport(
        n_conversations=len(per_conv),
        aggregate_turn_latency=_percentile_stats(all_latencies),
        aggregate_wer=sum(s.transcription_wer for s in per_conv) / len(per_conv),
        aggregate_faithfulness=sum(s.response_faithfulness for s in per_conv) / len(per_conv),
        aggregate_barge_in_success=sum(s.barge_in_success_rate for s in per_conv) / len(per_conv),
        aggregate_false_trigger_rate=sum(s.false_trigger_rate for s in per_conv) / len(per_conv),
        per_conversation=per_conv,
    )


def render_report(report: EvalReport) -> str:
    lines = ["# Voice eval report", ""]
    lines.append("## Headline")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| Conversations | {report.n_conversations} |")
    lines.append(
        f"| Turn latency p50 / p95 / p99 (ms) | "
        f"{report.aggregate_turn_latency.p50_ms:.0f} / "
        f"{report.aggregate_turn_latency.p95_ms:.0f} / "
        f"{report.aggregate_turn_latency.p99_ms:.0f} |"
    )
    lines.append(f"| Transcription WER (mean) | {report.aggregate_wer:.2%} |")
    lines.append(f"| Response faithfulness (mean) | {report.aggregate_faithfulness:.2%} |")
    lines.append(f"| Barge-in success (mean) | {report.aggregate_barge_in_success:.2%} |")
    lines.append(f"| False-trigger rate (mean) | {report.aggregate_false_trigger_rate:.2%} |")
    lines.append("")
    lines.append("## Per conversation")
    lines.append("")
    lines.append("| conv_id | topic | p95 ms | WER | faithfulness | barge-in | false-trigger |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for s in report.per_conversation:
        lines.append(
            f"| {s.conv_id} | {s.topic} | "
            f"{s.turn_latency.p95_ms:.0f} | {s.transcription_wer:.2%} | "
            f"{s.response_faithfulness:.2%} | {s.barge_in_success_rate:.2%} | "
            f"{s.false_trigger_rate:.2%} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


__all__ = [
    "ConversationScore",
    "EvalReport",
    "barge_in_success_rate",
    "false_trigger_rate",
    "render_report",
    "response_faithfulness",
    "score_conversation",
    "score_run",
    "transcription_wer",
    "turn_latency_stats",
]
