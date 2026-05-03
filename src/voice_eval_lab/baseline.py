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

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from voice_eval_lab.models import EvalReport

CURRENT_SCHEMA_VERSION = 1


def _assert_no_symlink_ancestor(path: Path) -> None:
    """Raise `OSError` if any *parent directory* of `path` is a symlink.

    Checking only the final component misses the case where an attacker
    pre-plants a symlink higher in the directory tree so that writing to
    ``/tmp/evil/baseline.json`` redirects the file into an arbitrary
    location.  Walking every parent catches that entire attack class.
    """
    for parent in path.parents:
        if parent.is_symlink():
            resolved_parent = parent.resolve()
            suggested = path.resolve()
            raise OSError(
                f"refusing to write: parent directory is a symlink: {parent} "
                f"-> {resolved_parent} (suggested workaround: pass the resolved "
                f"path instead, e.g. {suggested})"
            )


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
    """Persist `report` wrapped with the current schema version + timestamp.

    Writes are atomic and concurrent-safe: the JSON is streamed into a
    uniquely-named temp file in the destination directory (created via
    ``tempfile.NamedTemporaryFile`` so two concurrent saves cannot collide
    on a fixed ``<name>.tmp``), then renamed into place via ``os.replace``.
    A crash or Ctrl-C between the two steps leaves the previous baseline
    intact rather than truncating it.

    The destination is also rejected if it is a symlink — a symlinked
    final path could redirect the rename to an arbitrary location. The
    same check runs on the temp path so a pre-planted symlink in the
    destination directory cannot be exploited to overwrite something
    outside it. (The CLI does its own symlink check on the final path
    too; this layer enforces the invariant for any direct caller.)
    """
    _assert_no_symlink_ancestor(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise OSError(f"refusing to write to symlinked baseline path: {path}")
    payload = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "saved_at": datetime.now(UTC).isoformat(),
        "report": report.model_dump(),
    }
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp.", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        if tmp_path.is_symlink():
            # tempfile.mkstemp creates a regular file via O_EXCL, so this
            # is belt-and-suspenders — but if anything ever produces a
            # symlinked temp we abort before writing through it.
            raise OSError(f"refusing to write to symlinked temp path: {tmp_path}")
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(payload, indent=2))
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up the temp on any failure (including KeyboardInterrupt) so
        # we don't leave orphaned `.tmp.*` files in the destination dir.
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise


def read_baseline(path: Path) -> EvalReport:
    """Load a versioned baseline file.

    Raises `BaselineSchemaError` if the file is missing the wrapper, has
    a mismatched schema version, or is missing any expected EvalReport
    field. The strict shape stops an old baseline from quietly producing
    Pydantic-default values for metrics that didn't exist when it was
    written.
    """
    try:
        raw: Any = json.loads(path.read_text())
    except UnicodeDecodeError as exc:
        raise OSError(
            f"baseline at {path} is not valid UTF-8 (binary or corrupt file)"
        ) from exc
    except IsADirectoryError:
        raise OSError(f"baseline path {path} is a directory, not a file") from None
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
        if isinstance(version, int) and version < CURRENT_SCHEMA_VERSION:
            raise BaselineSchemaError(
                f"baseline at {path} has schema_version={version!r} "
                f"(older than current {CURRENT_SCHEMA_VERSION}) — "
                "rerun `voice-eval baseline --save` to refresh"
            )
        raise BaselineSchemaError(
            f"baseline at {path} has schema_version={version!r} "
            f"(newer than current {CURRENT_SCHEMA_VERSION}) — "
            "upgrade voice-eval-lab to read this baseline"
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
    direction is bad.

    Signal loss/gain semantics
    --------------------------

    A `None` value means the metric had no usable samples in that run (e.g.
    no endpointing-labelled turns, or no per-turn TTS first-byte timings to
    compute jitter from). Treating `baseline=number, current=None` as "no
    change" hides real regressions where a code change accidentally drops
    instrumentation. So:

    - baseline numeric, current None  → flagged as a regression (signal loss)
    - baseline None,    current numeric → improvement (signal gain), not regressed
    - both None                        → skipped, never regresses

    Delta is left as `None` whenever either side is None — there is no
    meaningful arithmetic delta across a missing sample.
    """
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
        # When either side has no signal we can't compute an arithmetic
        # delta, but the four cases mean different things:
        #   * both None             → neither run produced this metric; skip
        #   * baseline only None    → the metric is now being measured; gain
        #   * current only None     → instrumentation lost since baseline; regression
        if b is None and c is None:
            diffs.append(
                MetricDiff(metric=metric, baseline=None, current=None, delta=None, regressed=False)
            )
            return
        if b is None:
            diffs.append(
                MetricDiff(metric=metric, baseline=None, current=c, delta=None, regressed=False)
            )
            return
        if c is None:
            diffs.append(
                MetricDiff(metric=metric, baseline=b, current=None, delta=None, regressed=True)
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
