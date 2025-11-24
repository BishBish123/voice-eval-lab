"""Voice-agent metrics + the per-run scorer.

(This module sits in `voice_eval_lab.eval`, the harness package — it does
not call Python's builtin code-evaluator anywhere.)
"""

from __future__ import annotations

from voice_eval_lab.models import (
    PipelineSpan,
    TurnLatencyStats,
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
