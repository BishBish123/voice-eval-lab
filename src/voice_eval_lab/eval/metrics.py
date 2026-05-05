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
import re
import unicodedata

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

HEDGING_PHRASES: tuple[str, ...] = (
    "i don't have a confident answer",
    "i'm not sure",
    "i don't know",
    "maybe",
    "perhaps",
    "i think",
    "could be",
    "i can't say",
)

# Single-token hedge words detected via word-boundary regex so they don't
# match inside other tokens (e.g. "likely" should not count toward the
# noun "biweekly"). Module-level so callers can extend the list without
# touching the metric internals.
HEDGE_WORDS: tuple[str, ...] = (
    "likely",
    "probably",
    "probable",
    "possibly",
    "maybe",
    "could",
    "might",
    "seems",
    "appears",
    "believe",
    "suspect",
)
_HEDGE_WORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in HEDGE_WORDS) + r")\b"
)


def _normalize_text(text: str) -> str:
    """NFKC-fold then lowercase for substring comparisons.

    Faithfulness and decisiveness compare gold facts and replies as
    text. Without NFKC the same logical word (e.g. fullwidth ASCII or
    composed-vs-decomposed accented forms) reads as different bytes
    and silently fails to match. Lowercase is applied after NFKC so the
    comparison is genuinely case-insensitive on every glyph.
    """
    return unicodedata.normalize("NFKC", text).lower()


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


def _faithfulness_counts(
    conversation: Conversation, run: ConversationRun
) -> tuple[int, int]:
    """Return (grounded_replies, total_replies_with_gold_facts).

    Returns (0, 0) for conversations with no gold_facts — those are
    vacuously faithful at the per-conversation layer but contribute
    nothing to the pooled aggregate (no facts to be (un)grounded
    against).
    """
    if not conversation.gold_facts:
        return 0, 0
    normalized_facts = [_normalize_text(f) for f in conversation.gold_facts]
    grounded = 0
    counted = 0
    for tr in run.turn_runs:
        if tr.false_trigger:
            continue
        counted += 1
        reply = _normalize_text(tr.agent_reply)
        if any(fact in reply for fact in normalized_facts):
            grounded += 1
    return grounded, counted


def response_faithfulness(conversation: Conversation, run: ConversationRun) -> float:
    """Fraction of agent replies that quote at least one gold fact substring.

    Both sides are NFKC-normalized + lowercased before comparison so
    cosmetic Unicode variants (fullwidth ASCII, composed vs decomposed
    accents) don't silently miss a match.

    Proxy until an LLM judge is plugged in via the same Protocol.
    """
    if not conversation.gold_facts:
        return 1.0  # nothing to disagree with
    grounded, counted = _faithfulness_counts(conversation, run)
    return grounded / counted if counted else 0.0


DEFAULT_PIPELINE_BARGE_IN_BUDGET_MS = 100
"""Canonical pipeline barge-in budget — matches `VoicePipeline.barge_in_yield_ms`.

The metric and the pipeline used to disagree: the metric scored
against a 200ms budget while the pipeline shipped a 100ms yield, so
a 150ms yield was "success" by the metric and 50ms over budget by
contract. Per-conversation `barge_in_success_rate` now requires the
budget explicitly (no default); higher-level `score_run` /
`score_conversation` thread *this* constant through unless the caller
overrides, and the CLI passes the pipeline's actual configured value.
"""


def _barge_in_counts(
    conversation: Conversation,
    run: ConversationRun,
    barge_in_budget_ms: int,
) -> tuple[int, int]:
    """Return (successful_interrupted_turns, total_interrupted_turns).

    Used both for the per-conversation rate and to pool numerators /
    denominators across the run.
    """
    user_turns = [t for t in conversation.turns if t.role is TurnRole.USER]
    interruptible = [t for t in user_turns if t.interrupted]
    if not interruptible:
        return 0, 0
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
    return yielded, len(interruptible)


def barge_in_success_rate(
    conversation: Conversation,
    run: ConversationRun,
    *,
    barge_in_budget_ms: int,
) -> float:
    """Fraction of user-interrupted turns the pipeline yielded inside `barge_in_budget_ms`.

    A turn is "successful" iff it has a `barge_in.yield` span AND that
    span's duration is <= `barge_in_budget_ms`. The presence of a
    `tts_first_byte` span is *not* sufficient — the agent reaching TTS
    at all says nothing about whether it cut off the previous reply when
    the user barged in.

    `barge_in_budget_ms` is required (no default) so the metric and the
    pipeline cannot drift apart. The metric used to default to 200ms
    while the canonical `VoicePipeline.barge_in_yield_ms` was 100ms,
    which made a 150ms yield "success" by the metric and 50ms over
    budget by contract. Callers thread the pipeline's actual configured
    value via `score_run(..., barge_in_budget_ms=...)`.

    Returns 1.0 when there are no interrupted user turns (vacuously true).
    """
    yielded, total = _barge_in_counts(conversation, run, barge_in_budget_ms)
    if total == 0:
        return 1.0
    return yielded / total


def _false_trigger_counts(run: ConversationRun) -> tuple[int, int]:
    """Return (false_trigger_turns, total_turn_runs)."""
    return sum(1 for tr in run.turn_runs if tr.false_trigger), len(run.turn_runs)


def false_trigger_rate(run: ConversationRun) -> float:
    triggers, total = _false_trigger_counts(run)
    if total == 0:
        return 0.0
    return triggers / total


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


def _endpointing_counts(
    conversation: Conversation, run: ConversationRun, tolerance_ms: int = 50
) -> tuple[int, int]:
    """Return (aligned_turns, measured_turns_with_vad_end).

    Pooling these numerator/denominator counts across the run gives the
    aggregate the right denominator: a long noisy conversation with 50
    measured turns weighs 50x a 1-turn conversation, instead of 1x.
    """
    user_turns = [t for t in conversation.turns if t.role is TurnRole.USER]
    aligned = 0
    counted = 0
    for ut, tr in zip(user_turns, run.turn_runs, strict=False):
        vad = _find_span(tr.spans, "vad_end")
        if vad is None:
            continue
        counted += 1
        if abs(vad.ended_at_ms - ut.ended_at_ms) <= tolerance_ms:
            aligned += 1
    return aligned, counted


def endpointing_accuracy(
    conversation: Conversation, run: ConversationRun, tolerance_ms: int = 50
) -> float | None:
    """Fraction of user turns where VAD-end aligned with the gold utterance end.

    Four cases:

    - No user turns at all → 1.0 (vacuously true; nothing to be wrong about).
    - User turns exist but zero produced a measurable `vad_end` span → None.
      A pipeline with no VAD signal at all is "no measurement," not "every
      measured turn was wrong" — the latter is what 0.0 should mean. Mixing
      the two cases lets a broken VAD silently disappear into the aggregate
      mean instead of standing out as missing data.
    - User turns exist with measurable VAD ends but every one was outside
      tolerance → 0.0.
    - User turns exist with measurable VAD ends → fraction within tolerance.

    The mock pipeline always lines them up exactly so the headline value
    is 1.0; the metric exists so a real pipeline (whose VAD will be early
    or late) gets scored on a known axis. The aggregator skips `None`
    entries so a single broken conversation can't drag the headline.
    """
    user_turns = [t for t in conversation.turns if t.role is TurnRole.USER]
    if not user_turns:
        return 1.0
    aligned, counted = _endpointing_counts(conversation, run, tolerance_ms)
    if counted == 0:
        # User turns existed but the pipeline emitted no VAD-end spans —
        # explicit "no signal" rather than a 0.0 / 1.0 conflation.
        return None
    return aligned / counted


def _decisiveness_counts(run: ConversationRun) -> tuple[int, int]:
    """Return (decisive_replies, total_replies_excluding_false_triggers)."""
    counted = 0
    decisive = 0
    for tr in run.turn_runs:
        if tr.false_trigger:
            continue
        counted += 1
        reply = _normalize_text(tr.agent_reply)
        if not reply.strip():
            continue
        if any(phrase in reply for phrase in HEDGING_PHRASES):
            continue
        if _HEDGE_WORD_RE.search(reply):
            continue
        decisive += 1
    return decisive, counted


def llm_decisiveness(run: ConversationRun) -> float:
    """Fraction of agent replies that don't contain a hedging phrase.

    False-trigger turns are excluded — the agent is *supposed* to dodge
    those. Empty replies count as hedging (no signal = no commitment).

    Hedging is detected in two passes:
    - long phrases via NFKC-normalized substring (``HEDGING_PHRASES``)
    - single-token hedge words (``likely``, ``probably``, ...) via a
      word-boundary regex so substrings inside unrelated tokens don't
      count.
    """
    decisive, counted = _decisiveness_counts(run)
    if counted == 0:
        return 1.0
    return decisive / counted


# ---------------------------------------------------------------------------
# Per-conversation + aggregate scoring
# ---------------------------------------------------------------------------


def score_conversation(
    conversation: Conversation,
    run: ConversationRun,
    *,
    barge_in_budget_ms: int = DEFAULT_PIPELINE_BARGE_IN_BUDGET_MS,
) -> ConversationScore:
    _check_turn_coverage(conversation, run)
    return ConversationScore(
        conv_id=conversation.conv_id,
        topic=conversation.topic,
        turn_latency=turn_latency_stats(run.turn_runs),
        transcription_wer=transcription_wer(conversation, run),
        response_faithfulness=response_faithfulness(conversation, run),
        barge_in_success_rate=barge_in_success_rate(
            conversation, run, barge_in_budget_ms=barge_in_budget_ms
        ),
        false_trigger_rate=false_trigger_rate(run),
        barge_in_latency_p95_ms=barge_in_latency_p95_ms(run),
        tts_first_byte_jitter_ms=tts_first_byte_jitter_ms(run),
        endpointing_accuracy=endpointing_accuracy(conversation, run),
        llm_decisiveness=llm_decisiveness(run),
    )


class _RunPool:
    """Collects per-turn samples pooled across the whole run.

    Aggregates that compare distributions (latencies, barge-in yields,
    jitter) or denominators (corpus WER) need the *pooled* data, not
    the mean of per-conversation aggregates. Bundling the loop here
    keeps `score_run` from sprouting nested branches per metric.
    """

    __slots__ = (
        "barge_in_total",
        "barge_in_yielded",
        "barge_yields",
        "decisive_replies",
        "decisive_total",
        "endpointing_aligned",
        "endpointing_measured",
        "false_trigger_total",
        "false_trigger_turns",
        "grounded_replies",
        "grounded_total",
        "latencies",
        "wer_hyps",
        "wer_refs",
    )

    def __init__(self) -> None:
        self.latencies: list[float] = []
        self.barge_yields: list[float] = []
        self.wer_refs: list[str] = []
        self.wer_hyps: list[str] = []
        # Numerator/denominator pools. Each headline ratio aggregate is
        # numerator-sum / denominator-sum across the run; the per-conv
        # mean overweights short conversations and dilutes signal from
        # the few that actually exercised the metric.
        self.barge_in_yielded: int = 0
        self.barge_in_total: int = 0
        self.grounded_replies: int = 0
        self.grounded_total: int = 0
        self.false_trigger_turns: int = 0
        self.false_trigger_total: int = 0
        self.decisive_replies: int = 0
        self.decisive_total: int = 0
        self.endpointing_aligned: int = 0
        self.endpointing_measured: int = 0


def _pool_run_samples(
    pairs: list[tuple[Conversation, ConversationRun]],
    *,
    barge_in_budget_ms: int,
) -> _RunPool:
    pool = _RunPool()
    for c, r in pairs:
        user_turns = [t for t in c.turns if t.role is TurnRole.USER]
        for ut, tr in zip(user_turns, r.turn_runs, strict=False):
            if ut.text.strip():
                pool.wer_refs.append(ut.text)
                pool.wer_hyps.append(tr.transcribed_text)
        for tr in r.turn_runs:
            vad = _find_span(tr.spans, "vad_end")
            fb = _find_span(tr.spans, "tts_first_byte")
            if vad and fb:
                pool.latencies.append(float(fb.ended_at_ms - vad.ended_at_ms))
            for s in tr.spans:
                if s.name == "barge_in.yield":
                    pool.barge_yields.append(float(s.ended_at_ms - s.started_at_ms))
        yielded, total = _barge_in_counts(c, r, barge_in_budget_ms)
        pool.barge_in_yielded += yielded
        pool.barge_in_total += total
        grounded, ground_total = _faithfulness_counts(c, r)
        pool.grounded_replies += grounded
        pool.grounded_total += ground_total
        ft_turns, ft_total = _false_trigger_counts(r)
        pool.false_trigger_turns += ft_turns
        pool.false_trigger_total += ft_total
        decisive, dec_total = _decisiveness_counts(r)
        pool.decisive_replies += decisive
        pool.decisive_total += dec_total
        aligned, measured = _endpointing_counts(c, r)
        pool.endpointing_aligned += aligned
        pool.endpointing_measured += measured
    return pool


def score_run(
    pairs: list[tuple[Conversation, ConversationRun]],
    *,
    barge_in_budget_ms: int = DEFAULT_PIPELINE_BARGE_IN_BUDGET_MS,
) -> EvalReport:
    per_conv = [
        score_conversation(c, r, barge_in_budget_ms=barge_in_budget_ms) for c, r in pairs
    ]
    if not per_conv:
        empty = TurnLatencyStats(p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, n=0)
        return EvalReport(
            n_conversations=0,
            aggregate_turn_latency=empty,
            aggregate_wer=0.0,
            aggregate_faithfulness=None,
            aggregate_barge_in_success=None,
            aggregate_false_trigger_rate=None,
            aggregate_barge_in_latency_p95_ms=None,
            aggregate_tts_first_byte_jitter_ms=None,
            aggregate_endpointing_accuracy=None,
            aggregate_llm_decisiveness=None,
            per_conversation=[],
        )

    pooled = _pool_run_samples(pairs, barge_in_budget_ms=barge_in_budget_ms)
    all_latencies = pooled.latencies
    all_barge_yields = pooled.barge_yields
    pooled_wer_refs = pooled.wer_refs
    pooled_wer_hyps = pooled.wer_hyps

    # Aggregate barge-in p95 must come from the *pooled* sample, not the
    # mean of per-conversation p95s — the latter folds zero-signal
    # conversations into the headline and hides the real distribution.
    if all_barge_yields:
        all_barge_yields.sort()
        idx = min(len(all_barge_yields) - 1, int(0.95 * len(all_barge_yields)))
        agg_barge_p95: float | None = float(all_barge_yields[idx])
    else:
        agg_barge_p95 = None

    # Jitter is the population stddev of first-byte latency *across all
    # turns* — see ARCHITECTURE.md. Mean-of-per-conv-stddevs underweights
    # variance that crosses conversation boundaries.
    if len(all_latencies) >= 2:
        agg_jitter: float | None = float(np.std(all_latencies, ddof=0))
    else:
        agg_jitter = None

    # Corpus WER is computed against the *pooled* refs/hyps across every
    # measurable user turn in every conversation. Mean-of-per-conv-WER
    # overweights short conversations relative to long ones — a one-turn
    # conversation at 0% WER and a 50-turn conversation at 10% should not
    # average to 5%, because the actual word-error rate over the corpus
    # is ~10%. jiwer.wer accepts the parallel lists directly.
    agg_wer = (
        float(jiwer.wer(pooled_wer_refs, pooled_wer_hyps)) if pooled_wer_refs else 0.0
    )

    # Aggregate barge-in success rate pools numerators/denominators across
    # the run. mean-of-per-conv treats every conversation with zero
    # interrupts as 1.0 ("vacuously true") and dilutes any real failures
    # in the few conversations that actually exercised barge-in. None
    # when the entire run had no interrupts (no signal — same nullable
    # pattern as endpointing).
    agg_barge_in_success: float | None = (
        pooled.barge_in_yielded / pooled.barge_in_total
        if pooled.barge_in_total > 0
        else None
    )

    # Faithfulness, false-trigger, decisiveness, endpointing all share
    # the same root cause as WER: their per-conversation denominators
    # vary (1-turn vs 50-turn conversations), so the mean of per-conv
    # scores is not the corpus-level rate. Pool numerators / denominators
    # across the run; None when the entire run had no measurable signal
    # (faithfulness with no gold facts anywhere, no replies at all,
    # broken VAD on every turn).
    agg_faithfulness: float | None = (
        pooled.grounded_replies / pooled.grounded_total
        if pooled.grounded_total > 0
        else None
    )
    agg_false_trigger_rate: float | None = (
        pooled.false_trigger_turns / pooled.false_trigger_total
        if pooled.false_trigger_total > 0
        else None
    )
    agg_decisiveness: float | None = (
        pooled.decisive_replies / pooled.decisive_total
        if pooled.decisive_total > 0
        else None
    )
    agg_endpointing: float | None = (
        pooled.endpointing_aligned / pooled.endpointing_measured
        if pooled.endpointing_measured > 0
        else None
    )

    return EvalReport(
        n_conversations=len(per_conv),
        aggregate_turn_latency=_percentile_stats(all_latencies),
        aggregate_wer=agg_wer,
        aggregate_faithfulness=agg_faithfulness,
        aggregate_barge_in_success=agg_barge_in_success,
        aggregate_false_trigger_rate=agg_false_trigger_rate,
        aggregate_barge_in_latency_p95_ms=agg_barge_p95,
        aggregate_tts_first_byte_jitter_ms=agg_jitter,
        aggregate_endpointing_accuracy=agg_endpointing,
        aggregate_llm_decisiveness=agg_decisiveness,
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
    lines.append(f"| Transcription WER (corpus) | {report.aggregate_wer:.2%} |")
    faithfulness_cell = _pct_or_na(report.aggregate_faithfulness)
    lines.append(f"| Response faithfulness (pooled) | {faithfulness_cell} |")
    barge_success_cell = _pct_or_na(report.aggregate_barge_in_success)
    lines.append(f"| Barge-in success (pooled) | {barge_success_cell} |")
    false_trigger_cell = _pct_or_na(report.aggregate_false_trigger_rate)
    lines.append(f"| False-trigger rate (pooled) | {false_trigger_cell} |")
    barge_p95_cell = (
        "n/a"
        if report.aggregate_barge_in_latency_p95_ms is None
        else f"{report.aggregate_barge_in_latency_p95_ms:.0f}"
    )
    lines.append(f"| Barge-in yield p95 (ms) | {barge_p95_cell} |")
    jitter_cell = (
        "n/a"
        if report.aggregate_tts_first_byte_jitter_ms is None
        else f"{report.aggregate_tts_first_byte_jitter_ms:.1f}"
    )
    lines.append(f"| TTS first-byte jitter (ms) | {jitter_cell} |")
    endpoint_cell = _pct_or_na(report.aggregate_endpointing_accuracy)
    lines.append(f"| Endpointing accuracy (pooled) | {endpoint_cell} |")
    decisiveness_cell = _pct_or_na(report.aggregate_llm_decisiveness)
    lines.append(f"| LLM decisiveness (pooled) | {decisiveness_cell} |")
    lines.append("")
    lines.append("## Per conversation")
    lines.append("")
    lines.append(
        "| conv_id | topic | p95 ms | WER | faithfulness | "
        "barge-in | false-trigger | yield p95 | jitter | endpoint | decisive |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for s in report.per_conversation:
        endpoint = (
            "n/a" if s.endpointing_accuracy is None else f"{s.endpointing_accuracy:.2%}"
        )
        lines.append(
            f"| {_md_cell(s.conv_id)} | {_md_cell(s.topic)} | "
            f"{s.turn_latency.p95_ms:.0f} | {s.transcription_wer:.2%} | "
            f"{s.response_faithfulness:.2%} | {s.barge_in_success_rate:.2%} | "
            f"{s.false_trigger_rate:.2%} | {s.barge_in_latency_p95_ms:.0f} | "
            f"{s.tts_first_byte_jitter_ms:.1f} | {endpoint} | "
            f"{s.llm_decisiveness:.2%} |"
        )
    return "\n".join(lines) + "\n"


def _pct_or_na(value: float | None) -> str:
    """Render a 0..1 fraction as a percentage, or ``"n/a"`` for None."""
    return "n/a" if value is None else f"{value:.2%}"


def _md_cell(value: str) -> str:
    """Escape characters that would corrupt a Markdown table row.

    A pipe (``|``) ends the cell, a newline ends the row, a backslash
    can leak into rendered output, and HTML-ish tokens (``<``, ``>``)
    let renderers that allow inline HTML (CommonMark "lenient" mode,
    GitHub-flavored Markdown) treat the cell as live HTML. None of
    those are useful inside a metric report — escape them all.
    """
    if not value:
        return value
    escaped = (
        value.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r\n", " ")
        .replace("\n", " ")
        .replace("\r", " ")
    )
    return html.escape(escaped, quote=False)


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
        row("Response faithfulness (pooled)", _pct_or_na(report.aggregate_faithfulness)),
        row("Barge-in success (pooled)", _pct_or_na(report.aggregate_barge_in_success)),
        row("False-trigger rate (pooled)", _pct_or_na(report.aggregate_false_trigger_rate)),
        row(
            "Barge-in yield p95 (ms)",
            "n/a"
            if report.aggregate_barge_in_latency_p95_ms is None
            else f"{report.aggregate_barge_in_latency_p95_ms:.0f}",
        ),
        row(
            "TTS first-byte jitter (ms)",
            "n/a"
            if report.aggregate_tts_first_byte_jitter_ms is None
            else f"{report.aggregate_tts_first_byte_jitter_ms:.1f}",
        ),
        row("Endpointing accuracy (pooled)", _pct_or_na(report.aggregate_endpointing_accuracy)),
        row("LLM decisiveness (pooled)", _pct_or_na(report.aggregate_llm_decisiveness)),
    ]
    per_conv_rows = []
    for s in report.per_conversation:
        endpoint = (
            "n/a" if s.endpointing_accuracy is None else f"{s.endpointing_accuracy:.2%}"
        )
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
            f"<td class='num'>{endpoint}</td>"
            f"<td class='num'>{s.llm_decisiveness:.2%}</td>"
            "</tr>"
        )
    headline_block = "\n".join(headline_rows)
    per_conv_block = "\n".join(per_conv_rows)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
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
