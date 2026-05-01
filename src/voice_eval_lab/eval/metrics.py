"""Voice-agent metrics + the per-run scorer.

(This module sits in `voice_eval_lab.eval`, the harness package — it does
not call Python's builtin code-evaluator anywhere.)

Headline metrics:

- `turn_latency_stats` — vad_end -> tts_first_byte per turn, p50/p95/p99
- `transcription_wer` — jiwer over (gold transcript, pipeline transcript)
- `response_faithfulness` — fraction of agent replies that mention any
  gold fact (proxy for grounded-ness; LLM judge would replace this in
  production while keeping the API identical)
- `barge_in_success_rate` — of user-interrupted turns, fraction the
  pipeline yielded inside `barge_in_yield_ms`
- `false_trigger_rate` — fraction of turns marked `false_trigger=True`

Diagnostic metrics (added after the v0.1 set proved coarse for tuning):

- `barge_in_latency_p95` — p95 of the barge_in.yield span duration; the
  binary success rate hides regressions inside the budget
- `tts_first_byte_jitter` — std-dev of first-byte latency across turns;
  reveals variance the percentile stats smear out
- `endpointing_accuracy` — fraction of user turns where the VAD end span
  lined up with the gold `ended_at_ms` (within tolerance)
- `llm_decisiveness` — fraction of agent replies that don't hedge with
  "I don't have a confident answer", "maybe", "I'm not sure", etc.
"""

from __future__ import annotations

import html
import math

import jiwer
import numpy as np

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

HEDGING_PHRASES = (
    "i don't have a confident answer",
    "i'm not sure",
    "i don't know",
    "maybe",
    "perhaps",
    "i think",
    "could be",
    "i can't say",
)


class IncompleteRunError(ValueError):
    """Raised when a `ConversationRun` has fewer turn_runs than the conversation has user turns.

    Adapters that silently drop later turns must surface as an error
    rather than score better than an honest pipeline. Without this check
    a real LLM that crashed mid-conversation would look perfect on every
    user turn it managed to answer.
    """


def _check_turn_coverage(conversation: Conversation, run: ConversationRun) -> None:
    """Raise `IncompleteRunError` when the run is missing turn_runs."""
    expected = sum(1 for t in conversation.turns if t.role is TurnRole.USER)
    actual = len(run.turn_runs)
    if actual < expected:
        raise IncompleteRunError(
            f"conversation {conversation.conv_id!r} has {expected} user turns "
            f"but the run produced only {actual} turn_runs — "
            "an adapter dropped turns and would otherwise score silently better"
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
        if not ut.text.strip():
            continue
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


DEFAULT_BARGE_IN_BUDGET_MS = 200


def barge_in_success_rate(
    conversation: Conversation,
    run: ConversationRun,
    barge_in_budget_ms: int = DEFAULT_BARGE_IN_BUDGET_MS,
) -> float:
    """Fraction of user-interrupted turns the pipeline yielded inside `barge_in_budget_ms`.

    A turn is "successful" iff it has a `barge_in.yield` span AND that
    span's duration is <= `barge_in_budget_ms`. The presence of a
    `tts_first_byte` span is *not* sufficient — the agent reaching TTS
    at all says nothing about whether it cut off the previous reply when
    the user barged in.

    Returns 1.0 when there are no interrupted user turns (vacuously true).
    """
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
        yield_span = _find_span(tr.spans, "barge_in.yield")
        if yield_span is None:
            continue
        if (yield_span.ended_at_ms - yield_span.started_at_ms) <= barge_in_budget_ms:
            yielded += 1
    return yielded / len(interruptible)


def false_trigger_rate(run: ConversationRun) -> float:
    if not run.turn_runs:
        return 0.0
    return sum(1 for tr in run.turn_runs if tr.false_trigger) / len(run.turn_runs)


def barge_in_latency_p95_ms(run: ConversationRun) -> float:
    """p95 of `barge_in.yield` span duration, in ms.

    Returns 0.0 when no interrupted turns exist — there's no signal to
    summarise. The binary success rate hides regressions that happen
    *inside* the budget; this metric exposes the distribution.
    """
    samples: list[float] = []
    for tr in run.turn_runs:
        for s in tr.spans:
            if s.name == "barge_in.yield":
                samples.append(float(s.ended_at_ms - s.started_at_ms))
    if not samples:
        return 0.0
    samples.sort()
    idx = min(len(samples) - 1, int(0.95 * len(samples)))
    return float(samples[idx])


def tts_first_byte_jitter_ms(run: ConversationRun) -> float:
    """Population standard deviation of first-byte latency, in ms.

    The percentile stats fold the distribution into three numbers; this
    one number tells you whether the agent's audio start *feels* steady.
    Returns 0.0 for fewer than 2 samples.
    """
    samples: list[float] = []
    for tr in run.turn_runs:
        vad = _find_span(tr.spans, "vad_end")
        fb = _find_span(tr.spans, "tts_first_byte")
        if vad is None or fb is None:
            continue
        samples.append(float(fb.ended_at_ms - vad.ended_at_ms))
    if len(samples) < 2:
        return 0.0
    mean = sum(samples) / len(samples)
    var = sum((x - mean) ** 2 for x in samples) / len(samples)
    return math.sqrt(var)


def endpointing_accuracy(
    conversation: Conversation, run: ConversationRun, tolerance_ms: int = 50
) -> float:
    """Fraction of user turns where VAD-end aligned with the gold utterance end.

    Three cases:

    - No user turns at all → 1.0 (vacuously true; nothing to be wrong about).
    - User turns exist but zero produced a measurable `vad_end` span → 0.0.
      The metric used to return 1.0 here, which silently turned a broken
      pipeline (no VAD signal whatsoever) into a perfect score.
    - User turns exist with measurable VAD ends → fraction within tolerance.

    The mock pipeline always lines them up exactly so the headline value
    is 1.0; the metric exists so a real pipeline (whose VAD will be early
    or late) gets scored on a known axis.
    """
    user_turns = [t for t in conversation.turns if t.role is TurnRole.USER]
    if not user_turns:
        return 1.0
    aligned = 0
    counted = 0
    for ut, tr in zip(user_turns, run.turn_runs, strict=False):
        vad = _find_span(tr.spans, "vad_end")
        if vad is None:
            continue
        counted += 1
        if abs(vad.ended_at_ms - ut.ended_at_ms) <= tolerance_ms:
            aligned += 1
    if counted == 0:
        # User turns existed but the pipeline emitted no VAD-end spans —
        # treat as "no signal" rather than a perfect score.
        return 0.0
    return aligned / counted


def llm_decisiveness(run: ConversationRun) -> float:
    """Fraction of agent replies that don't contain a hedging phrase.

    False-trigger turns are excluded — the agent is *supposed* to dodge
    those. Empty replies count as hedging (no signal = no commitment).
    """
    counted = 0
    decisive = 0
    for tr in run.turn_runs:
        if tr.false_trigger:
            continue
        counted += 1
        reply = tr.agent_reply.lower()
        if not reply.strip():
            continue
        if any(phrase in reply for phrase in HEDGING_PHRASES):
            continue
        decisive += 1
    if counted == 0:
        return 1.0
    return decisive / counted


# ---------------------------------------------------------------------------
# Per-conversation + aggregate scoring
# ---------------------------------------------------------------------------


def score_conversation(conversation: Conversation, run: ConversationRun) -> ConversationScore:
    _check_turn_coverage(conversation, run)
    return ConversationScore(
        conv_id=conversation.conv_id,
        topic=conversation.topic,
        turn_latency=turn_latency_stats(run.turn_runs),
        transcription_wer=transcription_wer(conversation, run),
        response_faithfulness=response_faithfulness(conversation, run),
        barge_in_success_rate=barge_in_success_rate(conversation, run),
        false_trigger_rate=false_trigger_rate(run),
        barge_in_latency_p95_ms=barge_in_latency_p95_ms(run),
        tts_first_byte_jitter_ms=tts_first_byte_jitter_ms(run),
        endpointing_accuracy=endpointing_accuracy(conversation, run),
        llm_decisiveness=llm_decisiveness(run),
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
            aggregate_barge_in_latency_p95_ms=None,
            per_conversation=[],
        )

    all_latencies: list[float] = []
    all_barge_yields: list[float] = []
    for _c, r in pairs:
        for tr in r.turn_runs:
            vad = _find_span(tr.spans, "vad_end")
            fb = _find_span(tr.spans, "tts_first_byte")
            if vad and fb:
                all_latencies.append(float(fb.ended_at_ms - vad.ended_at_ms))
            for s in tr.spans:
                if s.name == "barge_in.yield":
                    all_barge_yields.append(float(s.ended_at_ms - s.started_at_ms))

    # Aggregate barge-in p95 must come from the *pooled* sample, not the
    # mean of per-conversation p95s — the latter folds zero-signal
    # conversations into the headline and hides the real distribution.
    if all_barge_yields:
        all_barge_yields.sort()
        idx = min(len(all_barge_yields) - 1, int(0.95 * len(all_barge_yields)))
        agg_barge_p95: float | None = float(all_barge_yields[idx])
    else:
        agg_barge_p95 = None

    return EvalReport(
        n_conversations=len(per_conv),
        aggregate_turn_latency=_percentile_stats(all_latencies),
        aggregate_wer=sum(s.transcription_wer for s in per_conv) / len(per_conv),
        aggregate_faithfulness=sum(s.response_faithfulness for s in per_conv) / len(per_conv),
        aggregate_barge_in_success=sum(s.barge_in_success_rate for s in per_conv) / len(per_conv),
        aggregate_false_trigger_rate=sum(s.false_trigger_rate for s in per_conv) / len(per_conv),
        aggregate_barge_in_latency_p95_ms=agg_barge_p95,
        aggregate_tts_first_byte_jitter_ms=(
            sum(s.tts_first_byte_jitter_ms for s in per_conv) / len(per_conv)
        ),
        aggregate_endpointing_accuracy=(
            sum(s.endpointing_accuracy for s in per_conv) / len(per_conv)
        ),
        aggregate_llm_decisiveness=(sum(s.llm_decisiveness for s in per_conv) / len(per_conv)),
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
    barge_p95_cell = (
        "n/a"
        if report.aggregate_barge_in_latency_p95_ms is None
        else f"{report.aggregate_barge_in_latency_p95_ms:.0f}"
    )
    lines.append(f"| Barge-in yield p95 (ms) | {barge_p95_cell} |")
    lines.append(
        f"| TTS first-byte jitter (ms) | {report.aggregate_tts_first_byte_jitter_ms:.1f} |"
    )
    lines.append(f"| Endpointing accuracy (mean) | {report.aggregate_endpointing_accuracy:.2%} |")
    lines.append(f"| LLM decisiveness (mean) | {report.aggregate_llm_decisiveness:.2%} |")
    lines.append("")
    lines.append("## Per conversation")
    lines.append("")
    lines.append(
        "| conv_id | topic | p95 ms | WER | faithfulness | "
        "barge-in | false-trigger | yield p95 | jitter | endpoint | decisive |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for s in report.per_conversation:
        lines.append(
            f"| {s.conv_id} | {s.topic} | "
            f"{s.turn_latency.p95_ms:.0f} | {s.transcription_wer:.2%} | "
            f"{s.response_faithfulness:.2%} | {s.barge_in_success_rate:.2%} | "
            f"{s.false_trigger_rate:.2%} | {s.barge_in_latency_p95_ms:.0f} | "
            f"{s.tts_first_byte_jitter_ms:.1f} | {s.endpointing_accuracy:.2%} | "
            f"{s.llm_decisiveness:.2%} |"
        )
    return "\n".join(lines) + "\n"


def render_report_html(report: EvalReport) -> str:
    """HTML variant of `render_report` for browser-friendly viewing.

    All dynamic strings (conv_id, topic, free-form metric labels) are
    passed through `html.escape(..., quote=True)` so a crafted scores.json
    cannot inject <script> or break out of attribute quoting in the
    rendered report.
    """

    def esc(value: object) -> str:
        return html.escape(str(value), quote=True)

    def row(metric: str, value: str) -> str:
        return f"    <tr><td>{esc(metric)}</td><td class='num'>{esc(value)}</td></tr>"

    headline_rows = [
        row("Conversations", str(report.n_conversations)),
        row(
            "Turn latency p50 / p95 / p99 (ms)",
            (
                f"{report.aggregate_turn_latency.p50_ms:.0f} / "
                f"{report.aggregate_turn_latency.p95_ms:.0f} / "
                f"{report.aggregate_turn_latency.p99_ms:.0f}"
            ),
        ),
        row("Transcription WER (mean)", f"{report.aggregate_wer:.2%}"),
        row("Response faithfulness (mean)", f"{report.aggregate_faithfulness:.2%}"),
        row("Barge-in success (mean)", f"{report.aggregate_barge_in_success:.2%}"),
        row("False-trigger rate (mean)", f"{report.aggregate_false_trigger_rate:.2%}"),
        row(
            "Barge-in yield p95 (ms)",
            "n/a"
            if report.aggregate_barge_in_latency_p95_ms is None
            else f"{report.aggregate_barge_in_latency_p95_ms:.0f}",
        ),
        row("TTS first-byte jitter (ms)", f"{report.aggregate_tts_first_byte_jitter_ms:.1f}"),
        row("Endpointing accuracy (mean)", f"{report.aggregate_endpointing_accuracy:.2%}"),
        row("LLM decisiveness (mean)", f"{report.aggregate_llm_decisiveness:.2%}"),
    ]
    per_conv_rows = []
    for s in report.per_conversation:
        per_conv_rows.append(
            "    <tr>"
            f"<td>{esc(s.conv_id)}</td>"
            f"<td>{esc(s.topic)}</td>"
            f"<td class='num'>{s.turn_latency.p95_ms:.0f}</td>"
            f"<td class='num'>{s.transcription_wer:.2%}</td>"
            f"<td class='num'>{s.response_faithfulness:.2%}</td>"
            f"<td class='num'>{s.barge_in_success_rate:.2%}</td>"
            f"<td class='num'>{s.false_trigger_rate:.2%}</td>"
            f"<td class='num'>{s.barge_in_latency_p95_ms:.0f}</td>"
            f"<td class='num'>{s.tts_first_byte_jitter_ms:.1f}</td>"
            f"<td class='num'>{s.endpointing_accuracy:.2%}</td>"
            f"<td class='num'>{s.llm_decisiveness:.2%}</td>"
            "</tr>"
        )
    headline_block = "\n".join(headline_rows)
    per_conv_block = "\n".join(per_conv_rows)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Voice eval report</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
  th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; }}
  th {{ background: #f5f5f5; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  h1 {{ margin-bottom: 0.3rem; }}
  h2 {{ margin-top: 2rem; }}
</style>
</head>
<body>
<h1>Voice eval report</h1>
<h2>Headline</h2>
<table>
  <thead><tr><th>Metric</th><th>Value</th></tr></thead>
  <tbody>
{headline_block}
  </tbody>
</table>
<h2>Per conversation</h2>
<table>
  <thead><tr>
    <th>conv_id</th><th>topic</th><th>p95 ms</th><th>WER</th>
    <th>faithfulness</th><th>barge-in</th><th>false-trigger</th>
    <th>yield p95</th><th>jitter</th><th>endpoint</th><th>decisive</th>
  </tr></thead>
  <tbody>
{per_conv_block}
  </tbody>
</table>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_span(spans: list[PipelineSpan], name: str) -> PipelineSpan | None:
    for s in spans:
        if s.name == name:
            return s
    return None


def _percentile_stats(samples: list[float]) -> TurnLatencyStats:
    """Compute p50/p95/p99 + sample count using linear-interpolated percentiles.

    Percentiles are computed with `numpy.percentile(..., method="linear")`,
    which matches the C=1 / "type 7" definition used by R, Excel, and most
    statistics libraries: percentiles linearly interpolate between adjacent
    order statistics. The earlier `int(p * n)` floor-index version produced
    unstable results for small N — for N=2 it would surface the maximum
    sample as p50, which is non-standard and broke regression detection
    on tiny golden sets.

    Returns zeros + n=0 for an empty sample list.
    """
    if not samples:
        return TurnLatencyStats(p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, n=0)
    p50, p95, p99 = np.percentile(samples, [50.0, 95.0, 99.0], method="linear")
    return TurnLatencyStats(
        p50_ms=float(p50),
        p95_ms=float(p95),
        p99_ms=float(p99),
        n=len(samples),
    )


__all__ = [
    "ConversationScore",
    "EvalReport",
    "IncompleteRunError",
    "barge_in_latency_p95_ms",
    "barge_in_success_rate",
    "endpointing_accuracy",
    "false_trigger_rate",
    "llm_decisiveness",
    "render_report",
    "render_report_html",
    "response_faithfulness",
    "score_conversation",
    "score_run",
    "transcription_wer",
    "tts_first_byte_jitter_ms",
    "turn_latency_stats",
]
