"""Tests for baseline serialization + regression detection."""

from __future__ import annotations

import json
import threading
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
        # Successful write removes the temp file via os.replace. With the
        # tempfile-based unique naming, scan the directory for any
        # `.tmp.*` siblings rather than checking a fixed name.
        leftover = [p for p in tmp_path.iterdir() if p.name.startswith(".tmp.")]
        assert leftover == []

    def test_concurrent_baseline_writes_dont_race(self, tmp_path: Path) -> None:
        # Two concurrent saves used to share the same `.tmp` sibling: one
        # process's flush could be truncated by the other's open(), and
        # whichever os.replace() landed last would win with a half-written
        # body. The fix uses unique tempfile.mkstemp names — both writes
        # complete, no `.tmp.*` orphans remain, and the final file is one
        # of the two valid payloads.
        path = tmp_path / "baseline.json"
        report_a = _report(aggregate_wer=0.01)
        report_b = _report(aggregate_wer=0.99)
        errors: list[BaseException] = []

        def writer(report: EvalReport) -> None:
            try:
                for _ in range(20):
                    write_baseline(report, path)
            except BaseException as exc:  # pragma: no cover - bubble up
                errors.append(exc)

        ta = threading.Thread(target=writer, args=(report_a,))
        tb = threading.Thread(target=writer, args=(report_b,))
        ta.start()
        tb.start()
        ta.join()
        tb.join()
        assert errors == []
        leftover = [p for p in tmp_path.iterdir() if p.name.startswith(".tmp.")]
        assert leftover == []
        loaded = read_baseline(path)
        assert loaded.aggregate_wer in {0.01, 0.99}

    def test_symlinked_parent_directory_rejected(self, tmp_path: Path) -> None:
        # Pre-plant a symlink at the parent level so that a write to
        # link_dir/baseline.json would redirect into real_dir/baseline.json
        # — which could be outside the intended write scope on a shared box.
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        link_dir = tmp_path / "link"
        link_dir.symlink_to(real_dir)
        with pytest.raises(OSError, match="symlink"):
            write_baseline(_report(), link_dir / "baseline.json")
        assert not (real_dir / "baseline.json").exists()

    def test_temp_path_symlink_rejected(self, tmp_path: Path) -> None:
        # A pre-planted symlink at the final destination must be rejected
        # so the rename cannot redirect to an arbitrary location.
        elsewhere = tmp_path / "elsewhere.json"
        path = tmp_path / "baseline.json"
        path.symlink_to(elsewhere)
        with pytest.raises(OSError, match="symlink"):
            write_baseline(_report(), path)
        # Symlink target must not have been created.
        assert not elsewhere.exists()


# ---------------------------------------------------------------------------
# read_baseline strictness — stale files must NOT be silently re-defaulted
# ---------------------------------------------------------------------------


class TestReadBaselineReadErrors:
    """read_baseline must map file-read errors to clean OSError messages."""

    def test_unicode_decode_error_raises_oserror(self, tmp_path: Path) -> None:
        bad = tmp_path / "baseline.json"
        bad.write_bytes(b"\xff\xfe binary garbage \x80\x81")
        with pytest.raises(OSError, match=r"utf-8|unicode|UTF-8"):
            read_baseline(bad)

    def test_is_a_directory_raises_oserror(self, tmp_path: Path) -> None:
        # Passing a directory path should raise OSError with a clear message.
        with pytest.raises(OSError, match="directory"):
            read_baseline(tmp_path)

    def test_permission_error_propagates(self, tmp_path: Path) -> None:
        bad = tmp_path / "baseline.json"
        bad.write_text('{"schema_version": 1, "report": {}}')
        bad.chmod(0o000)
        try:
            with pytest.raises((OSError, PermissionError)):
                read_baseline(bad)
        finally:
            bad.chmod(0o644)


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

    def test_read_baseline_too_old_message_suggests_rerun(self, tmp_path: Path) -> None:
        # A version lower than CURRENT_SCHEMA_VERSION should say "rerun baseline".
        path = tmp_path / "baseline.json"
        path.write_text(
            json.dumps({"schema_version": CURRENT_SCHEMA_VERSION - 1, "report": {}})
        )
        with pytest.raises(BaselineSchemaError, match=r"rerun|re-run|refresh"):
            read_baseline(path)

    def test_read_baseline_too_new_message_suggests_upgrade(self, tmp_path: Path) -> None:
        # A version higher than CURRENT_SCHEMA_VERSION should say "upgrade voice-eval-lab".
        path = tmp_path / "baseline.json"
        path.write_text(
            json.dumps({"schema_version": CURRENT_SCHEMA_VERSION + 1, "report": {}})
        )
        with pytest.raises(BaselineSchemaError, match=r"upgrade"):
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

    def test_compare_treats_signal_loss_as_regression(self) -> None:
        # Endpointing or jitter that had a number in the baseline run but
        # has dropped to None in the current run means the eval lost signal
        # — almost always a code regression that nuked the instrumentation.
        base = _report(aggregate_endpointing_accuracy=0.92)
        current = _report(aggregate_endpointing_accuracy=None)
        diffs = compare(base, current)
        endp = next(d for d in diffs if d.metric == "endpointing_accuracy")
        assert endp.regressed
        assert endp.baseline == 0.92
        assert endp.current is None
        assert endp.delta is None

    def test_compare_treats_signal_gain_as_improvement(self) -> None:
        # Inverse: baseline had no signal (metric did not exist or had no
        # samples), current run produces one. That is strictly better and
        # must not be flagged as a regression.
        base = _report(aggregate_tts_first_byte_jitter_ms=None)
        current = _report(aggregate_tts_first_byte_jitter_ms=4.0)
        diffs = compare(base, current)
        jitter = next(d for d in diffs if d.metric == "tts_first_byte_jitter_ms")
        assert not jitter.regressed
        assert jitter.baseline is None
        assert jitter.current == 4.0
        assert jitter.delta is None

    def test_compare_skips_both_none(self) -> None:
        base = _report(aggregate_llm_decisiveness=None)
        current = _report(aggregate_llm_decisiveness=None)
        diffs = compare(base, current)
        d = next(d for d in diffs if d.metric == "llm_decisiveness")
        assert not d.regressed
        assert d.baseline is None
        assert d.current is None
        assert d.delta is None

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
