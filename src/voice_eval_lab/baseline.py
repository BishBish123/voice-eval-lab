"""Baseline persistence + regression detection.

A baseline is a snapshot of headline + per-conversation eval scores
serialised to JSON. The compare flow:

    1. Run the eval, write `baseline.json`.
    2. Land a change.
    3. Run the eval again, compare against `baseline.json`.
    4. If any metric regressed past a threshold, exit non-zero.

Thresholds reflect the directionality of each metric: WER, jitter,
false-trigger, latency are "lower is better"; faithfulness, barge-in,
endpointing, decisiveness are "higher is better".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from voice_eval_lab.models import EvalReport


@dataclass
class RegressionThresholds:
    """Per-metric tolerance bands for `compare`. Defaults are conservative.

    All numbers are absolute deltas in the metric's native unit. Latency
    is ms; WER / faithfulness / barge-in / decisiveness / endpointing are
    fractions (0..1); jitter is ms.
    """

    latency_p95_ms: float = 10.0
    wer: float = 0.02
    faithfulness: float = 0.05
    barge_in: float = 0.05
    false_trigger: float = 0.02
    barge_in_latency_p95_ms: float = 25.0
    tts_first_byte_jitter_ms: float = 5.0
    endpointing: float = 0.02
    decisiveness: float = 0.05


@dataclass
class MetricDiff:
    metric: str
    baseline: float
    current: float
    delta: float
    regressed: bool


def write_baseline(report: EvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(), indent=2))


def read_baseline(path: Path) -> EvalReport:
    return EvalReport.model_validate(json.loads(path.read_text()))


def compare(
    baseline: EvalReport,
    current: EvalReport,
    thresholds: RegressionThresholds | None = None,
) -> list[MetricDiff]:
    """Return per-metric diffs; the `regressed` flag is set when |delta| > threshold *and* the
    direction is bad."""
    th = thresholds or RegressionThresholds()
    diffs: list[MetricDiff] = []

    def add(metric: str, b: float, c: float, threshold: float, *, lower_is_better: bool) -> None:
        delta = c - b
        regressed = delta > threshold if lower_is_better else -delta > threshold
        diffs.append(
            MetricDiff(metric=metric, baseline=b, current=c, delta=delta, regressed=regressed)
        )

    add(
        "turn_latency_p95_ms",
        baseline.aggregate_turn_latency.p95_ms,
        current.aggregate_turn_latency.p95_ms,
        th.latency_p95_ms,
        lower_is_better=True,
    )
    add(
        "wer",
        baseline.aggregate_wer,
        current.aggregate_wer,
        th.wer,
        lower_is_better=True,
    )
    add(
        "faithfulness",
        baseline.aggregate_faithfulness,
        current.aggregate_faithfulness,
        th.faithfulness,
        lower_is_better=False,
    )
    add(
        "barge_in_success",
        baseline.aggregate_barge_in_success,
        current.aggregate_barge_in_success,
        th.barge_in,
        lower_is_better=False,
    )
    add(
        "false_trigger_rate",
        baseline.aggregate_false_trigger_rate,
        current.aggregate_false_trigger_rate,
        th.false_trigger,
        lower_is_better=True,
    )
    add(
        "barge_in_latency_p95_ms",
        baseline.aggregate_barge_in_latency_p95_ms,
        current.aggregate_barge_in_latency_p95_ms,
        th.barge_in_latency_p95_ms,
        lower_is_better=True,
    )
    add(
        "tts_first_byte_jitter_ms",
        baseline.aggregate_tts_first_byte_jitter_ms,
        current.aggregate_tts_first_byte_jitter_ms,
        th.tts_first_byte_jitter_ms,
        lower_is_better=True,
    )
    add(
        "endpointing_accuracy",
        baseline.aggregate_endpointing_accuracy,
        current.aggregate_endpointing_accuracy,
        th.endpointing,
        lower_is_better=False,
    )
    add(
        "llm_decisiveness",
        baseline.aggregate_llm_decisiveness,
        current.aggregate_llm_decisiveness,
        th.decisiveness,
        lower_is_better=False,
    )
    return diffs


def render_diffs(diffs: list[MetricDiff]) -> str:
    lines = [
        "| Metric | Baseline | Current | Δ | Regressed |",
        "| --- | ---: | ---: | ---: | :---: |",
    ]
    for d in diffs:
        flag = "yes" if d.regressed else "no"
        lines.append(
            f"| {d.metric} | {d.baseline:.4f} | {d.current:.4f} | {d.delta:+.4f} | {flag} |"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "MetricDiff",
    "RegressionThresholds",
    "compare",
    "read_baseline",
    "render_diffs",
    "write_baseline",
]
