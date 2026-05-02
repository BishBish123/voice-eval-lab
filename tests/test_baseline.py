"""Tests for baseline serialization + regression detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voice_eval_lab.baseline import (
    CURRENT_SCHEMA_VERSION,
    BaselineSchemaError,
    RegressionThresholds,
    compare,
    read_baseline,
    render_diffs,
    write_baseline,
)
from voice_eval_lab.models import EvalReport, TurnLatencyStats


def _report(**overrides: float) -> EvalReport:
    base = {
        "n_conversations": 1,
        "aggregate_turn_latency": TurnLatencyStats(p50_ms=200.0, p95_ms=275.0, p99_ms=275.0, n=4),
        "aggregate_wer": 0.05,
        "aggregate_faithfulness": 0.8,
        "aggregate_barge_in_success": 1.0,
        "aggregate_false_trigger_rate": 0.0,
        "aggregate_barge_in_latency_p95_ms": 100.0,
        "aggregate_tts_first_byte_jitter_ms": 5.0,
        "aggregate_endpointing_accuracy": 1.0,
        "aggregate_llm_decisiveness": 0.7,
        "per_conversation": [],
    }
    base.update(overrides)  # type: ignore[arg-type]
    return EvalReport.model_validate(base)


# ---------------------------------------------------------------------------
# write/read round trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_write_read_roundtrip(self, tmp_path: Path) -> None:
        report = _report()
        path = tmp_path / "baseline.json"
        write_baseline(report, path)
        loaded = read_baseline(path)
        assert loaded.model_dump() == report.model_dump()

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        report = _report()
        path = tmp_path / "nested" / "deep" / "baseline.json"
        write_baseline(report, path)
        assert path.exists()

    def test_write_baseline_tags_schema_version(self, tmp_path: Path) -> None:
        report = _report()
        path = tmp_path / "baseline.json"
        write_baseline(report, path)
        raw = json.loads(path.read_text())
        assert raw["schema_version"] == CURRENT_SCHEMA_VERSION
        assert "report" in raw
        assert "saved_at" in raw

    def test_baseline_write_is_atomic(self, tmp_path: Path) -> None:
        # Simulate a crash between "open temp file" and "rename": write a
        # partial payload to the .tmp path without renaming, then assert
        # the baseline at the real path is unchanged.
        path = tmp_path / "baseline.json"
        write_baseline(_report(), path)
        original = path.read_text()
        # Pretend a fresh write got interrupted after the tmp was opened
        # but before os.replace ran.
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("{ partially written")
        # Original still parses fine.
        assert path.read_text() == original
        loaded = read_baseline(path)
        assert loaded.aggregate_wer == 0.05

    def test_baseline_write_does_not_leave_tmp(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        write_baseline(_report(), path)
        # Successful write removes the tmp file via os.replace.
        assert not path.with_name(path.name + ".tmp").exists()


# ---------------------------------------------------------------------------
# read_baseline strictness — stale files must NOT be silently re-defaulted
# ---------------------------------------------------------------------------


class TestReadBaselineStrict:
    def test_read_baseline_rejects_v0_blob(self, tmp_path: Path) -> None:
        # Pre-versioning shape: bare EvalReport dump, no wrapper.
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps(_report().model_dump()))
        with pytest.raises(BaselineSchemaError, match="schema_version"):
            read_baseline(path)

    def test_read_baseline_rejects_unknown_version(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        path.write_text(
            json.dumps({"schema_version": 999, "report": _report().model_dump()})
        )
        with pytest.raises(BaselineSchemaError, match="999"):
            read_baseline(path)

    def test_read_baseline_rejects_missing_field(self, tmp_path: Path) -> None:
        # Drop a v0.2 metric the current schema requires.
        report_blob = _report().model_dump()
        del report_blob["aggregate_tts_first_byte_jitter_ms"]
        path = tmp_path / "baseline.json"
        path.write_text(
            json.dumps({"schema_version": CURRENT_SCHEMA_VERSION, "report": report_blob})
        )
        with pytest.raises(BaselineSchemaError, match="aggregate_tts_first_byte_jitter_ms"):
            read_baseline(path)

    def test_read_baseline_rejects_missing_report(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps({"schema_version": CURRENT_SCHEMA_VERSION}))
        with pytest.raises(BaselineSchemaError, match="report"):
            read_baseline(path)

    def test_read_baseline_rejects_non_object(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(BaselineSchemaError):
            read_baseline(path)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


class TestCompare:
    def test_no_change_no_regression(self) -> None:
        a = _report()
        diffs = compare(a, _report())
        assert all(d.delta == 0.0 for d in diffs)
        assert all(not d.regressed for d in diffs)

    def test_latency_increase_regresses(self) -> None:
        base = _report()
        current = _report(
            aggregate_turn_latency=TurnLatencyStats(p50_ms=200.0, p95_ms=400.0, p99_ms=400.0, n=4)
        )
        diffs = compare(base, current)
        latency = next(d for d in diffs if d.metric == "turn_latency_p95_ms")
        assert latency.regressed
        assert latency.delta == 125.0

    def test_latency_decrease_does_not_regress(self) -> None:
        base = _report()
        current = _report(
            aggregate_turn_latency=TurnLatencyStats(p50_ms=200.0, p95_ms=200.0, p99_ms=200.0, n=4)
        )
        diffs = compare(base, current)
        latency = next(d for d in diffs if d.metric == "turn_latency_p95_ms")
        assert not latency.regressed
        assert latency.delta < 0

    def test_wer_increase_regresses(self) -> None:
        diffs = compare(_report(), _report(aggregate_wer=0.10))
        wer = next(d for d in diffs if d.metric == "wer")
        assert wer.regressed

    def test_faithfulness_drop_regresses(self) -> None:
        diffs = compare(_report(), _report(aggregate_faithfulness=0.5))
        f = next(d for d in diffs if d.metric == "faithfulness")
        assert f.regressed
        assert f.delta < 0

    def test_faithfulness_increase_does_not_regress(self) -> None:
        diffs = compare(_report(), _report(aggregate_faithfulness=0.95))
        f = next(d for d in diffs if d.metric == "faithfulness")
        assert not f.regressed

    def test_threshold_respected_for_small_drift(self) -> None:
        # WER drift of 0.005 should be under the default 0.02 threshold.
        diffs = compare(_report(), _report(aggregate_wer=0.055))
        wer = next(d for d in diffs if d.metric == "wer")
        assert not wer.regressed

    def test_jitter_increase_regresses(self) -> None:
        diffs = compare(_report(), _report(aggregate_tts_first_byte_jitter_ms=20.0))
        j = next(d for d in diffs if d.metric == "tts_first_byte_jitter_ms")
        assert j.regressed

    def test_decisiveness_drop_regresses(self) -> None:
        diffs = compare(_report(), _report(aggregate_llm_decisiveness=0.5))
        d = next(d for d in diffs if d.metric == "llm_decisiveness")
        assert d.regressed

    def test_endpointing_drop_regresses(self) -> None:
        diffs = compare(_report(), _report(aggregate_endpointing_accuracy=0.5))
        e = next(d for d in diffs if d.metric == "endpointing_accuracy")
        assert e.regressed

    def test_false_trigger_increase_regresses(self) -> None:
        diffs = compare(_report(), _report(aggregate_false_trigger_rate=0.10))
        ft = next(d for d in diffs if d.metric == "false_trigger_rate")
        assert ft.regressed

    def test_custom_thresholds_loosen(self) -> None:
        # With a permissive WER threshold, even a big jump shouldn't regress.
        diffs = compare(
            _report(),
            _report(aggregate_wer=0.5),
            RegressionThresholds(wer=1.0),
        )
        wer = next(d for d in diffs if d.metric == "wer")
        assert not wer.regressed

    def test_custom_thresholds_tighten(self) -> None:
        # With a 0 threshold, any drift regresses.
        diffs = compare(
            _report(),
            _report(aggregate_wer=0.051),
            RegressionThresholds(wer=0.0),
        )
        wer = next(d for d in diffs if d.metric == "wer")
        assert wer.regressed

    def test_render_diffs_shape(self) -> None:
        out = render_diffs(compare(_report(), _report()))
        assert "Metric" in out
        assert "Δ" in out
        assert "regressed" in out.lower()
        # Every metric shows up.
        for m in [
            "turn_latency_p95_ms",
            "wer",
            "faithfulness",
            "barge_in_success",
            "false_trigger_rate",
            "barge_in_latency_p95_ms",
            "tts_first_byte_jitter_ms",
            "endpointing_accuracy",
            "llm_decisiveness",
        ]:
            assert m in out
