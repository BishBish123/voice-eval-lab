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

Schema versioning
-----------------

The on-disk file is wrapped with an explicit `schema_version` so a stale
baseline (written before a metric was added) cannot silently inherit
schema defaults — Pydantic would happily fill `aggregate_jitter_ms=0.0`
for a v0.1 file and produce false comparisons. `read_baseline` rejects
files whose version is missing/mismatched, or whose payload is missing
any field the current `EvalReport` schema declares.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from voice_eval_lab.models import EvalReport

CURRENT_SCHEMA_VERSION = 1


class BaselineSchemaError(ValueError):
    """Raised when a baseline file is missing/wrong schema version or fields."""


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
    baseline: float | None
    current: float | None
    delta: float | None
    regressed: bool


def _expected_report_fields() -> set[str]:
    """Field names the current EvalReport schema declares."""
    return set(EvalReport.model_fields.keys())


def write_baseline(report: EvalReport, path: Path) -> None:
    """Persist `report` wrapped with the current schema version + timestamp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "saved_at": datetime.now(UTC).isoformat(),
        "report": report.model_dump(),
    }
    path.write_text(json.dumps(payload, indent=2))


def read_baseline(path: Path) -> EvalReport:
    """Load a versioned baseline file.

    Raises `BaselineSchemaError` if the file is missing the wrapper, has
    a mismatched schema version, or is missing any expected EvalReport
    field. The strict shape stops an old baseline from quietly producing
    Pydantic-default values for metrics that didn't exist when it was
    written.
    """
    raw: Any = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise BaselineSchemaError(
            f"baseline at {path} is not a JSON object (got {type(raw).__name__})"
        )
    if "schema_version" not in raw:
        raise BaselineSchemaError(
            f"baseline at {path} is missing 'schema_version' "
            "(written by an older voice-eval-lab — re-run `voice-eval baseline --save`)"
        )
    version = raw["schema_version"]
    if version != CURRENT_SCHEMA_VERSION:
        raise BaselineSchemaError(
            f"baseline at {path} has schema_version={version!r}, "
            f"expected {CURRENT_SCHEMA_VERSION} — re-run `voice-eval baseline --save`"
        )
    if "report" not in raw or not isinstance(raw["report"], dict):
        raise BaselineSchemaError(
            f"baseline at {path} is missing the 'report' object"
        )
    report_blob = raw["report"]
    expected = _expected_report_fields()
    missing = expected - set(report_blob.keys())
    if missing:
        raise BaselineSchemaError(
            f"baseline at {path} is missing required fields: {sorted(missing)} — "
            "re-run `voice-eval baseline --save` against current code"
        )
    return EvalReport.model_validate(report_blob)


def compare(
    baseline: EvalReport,
    current: EvalReport,
    thresholds: RegressionThresholds | None = None,
) -> list[MetricDiff]:
    """Return per-metric diffs; the `regressed` flag is set when |delta| > threshold *and* the
    direction is bad."""
    th = thresholds or RegressionThresholds()
    diffs: list[MetricDiff] = []

    def add(
        metric: str,
        b: float | None,
        c: float | None,
        threshold: float,
        *,
        lower_is_better: bool,
    ) -> None:
        # When either side has no signal we can't fairly compute a delta.
        # The diff still appears in the report (so consumers can see "n/a"
        # for documentation), but it is never flagged as a regression.
        if b is None or c is None:
            diffs.append(
                MetricDiff(metric=metric, baseline=b, current=c, delta=None, regressed=False)
            )
            return
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
        baseline_cell = "n/a" if d.baseline is None else f"{d.baseline:.4f}"
        current_cell = "n/a" if d.current is None else f"{d.current:.4f}"
        delta_cell = "n/a" if d.delta is None else f"{d.delta:+.4f}"
        lines.append(
            f"| {d.metric} | {baseline_cell} | {current_cell} | {delta_cell} | {flag} |"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "BaselineSchemaError",
    "MetricDiff",
    "RegressionThresholds",
    "compare",
    "read_baseline",
    "render_diffs",
    "write_baseline",
]
