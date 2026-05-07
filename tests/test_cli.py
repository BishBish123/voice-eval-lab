"""CLI smoke tests via typer.testing.CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from voice_eval_lab.cli import app

runner = CliRunner()


class TestRunCommand:
    def test_run_writes_markdown(self, tmp_path: Path) -> None:
        out = tmp_path / "REPORT.md"
        result = runner.invoke(app, ["run", "--out", str(out)])
        assert result.exit_code == 0, result.output
        assert out.exists()
        body = out.read_text()
        assert body.startswith("# Voice eval report")

    def test_run_writes_json_when_requested(self, tmp_path: Path) -> None:
        out = tmp_path / "REPORT.md"
        scores = tmp_path / "scores.json"
        result = runner.invoke(app, ["run", "--out", str(out), "--json", str(scores)])
        assert result.exit_code == 0, result.output
        assert scores.exists()
        data = json.loads(scores.read_text())
        assert "n_conversations" in data
        assert data["n_conversations"] >= 1

    def test_run_with_wer_flag(self, tmp_path: Path) -> None:
        out = tmp_path / "REPORT.md"
        result = runner.invoke(app, ["run", "--out", str(out), "--wer-substitution-rate", "0.2"])
        assert result.exit_code == 0, result.output

    def test_run_with_false_trigger_flag(self, tmp_path: Path) -> None:
        out = tmp_path / "REPORT.md"
        result = runner.invoke(app, ["run", "--out", str(out), "--false-trigger-rate", "1.0"])
        assert result.exit_code == 0, result.output

    def test_run_rejects_wer_above_one(self, tmp_path: Path) -> None:
        # Rates above 1.0 used to crash inside `_inject_wer` with an
        # IndexError; the CLI now refuses them with a clear typer error.
        out = tmp_path / "REPORT.md"
        result = runner.invoke(
            app, ["run", "--out", str(out), "--wer-substitution-rate", "1.5"]
        )
        assert result.exit_code != 0
        assert "0.0, 1.0" in result.output

    def test_run_rejects_negative_wer(self, tmp_path: Path) -> None:
        out = tmp_path / "REPORT.md"
        result = runner.invoke(
            app, ["run", "--out", str(out), "--wer-substitution-rate", "-0.1"]
        )
        assert result.exit_code != 0
        assert "0.0, 1.0" in result.output

    def test_run_rejects_symlink_output(self, tmp_path: Path) -> None:
        # A symlink at the output path could redirect a write into an
        # unrelated file. The CLI should refuse rather than follow it.
        target = tmp_path / "real.md"
        target.write_text("placeholder")
        link = tmp_path / "REPORT.md"
        link.symlink_to(target)
        result = runner.invoke(app, ["run", "--out", str(link)])
        assert result.exit_code != 0
        assert "symlink" in result.output
        # Original target untouched.
        assert target.read_text() == "placeholder"

    def test_run_handles_oserror_gracefully(self, tmp_path: Path) -> None:
        # Writing into a directory whose parent is itself a regular file
        # produces a NotADirectoryError (an OSError subclass). The CLI
        # should map it to a typer exit-2 with a readable message rather
        # than dumping a traceback.
        not_a_dir = tmp_path / "blocker"
        not_a_dir.write_text("file, not directory")
        bad = not_a_dir / "REPORT.md"
        result = runner.invoke(app, ["run", "--out", str(bad)])
        assert result.exit_code == 2
        assert "could not write" in result.output


class TestSymlinkAncestorGuard:
    def test_run_rejects_symlinked_parent_directory(self, tmp_path: Path) -> None:
        # Pre-plant a symlink in the parent chain: real_dir is the actual
        # destination; link_dir points to it. Writing a report under
        # link_dir must be refused before any bytes hit disk.
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        link_dir = tmp_path / "link"
        link_dir.symlink_to(real_dir)
        result = runner.invoke(app, ["run", "--out", str(link_dir / "REPORT.md")])
        assert result.exit_code != 0
        assert "symlink" in result.output.lower()

    def test_baseline_rejects_symlinked_parent_directory(self, tmp_path: Path) -> None:
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        link_dir = tmp_path / "link"
        link_dir.symlink_to(real_dir)
        result = runner.invoke(app, ["baseline", "--save", str(link_dir / "baseline.json")])
        assert result.exit_code != 0
        assert "symlink" in result.output.lower()

    def test_compare_rejects_symlinked_parent_before_running_harness(
        self, tmp_path: Path
    ) -> None:
        # Establish a real baseline so --baseline is valid.
        baseline = tmp_path / "baseline.json"
        runner.invoke(app, ["baseline", "--save", str(baseline)])
        # Pre-plant a symlink as the --out parent directory.
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        link_dir = tmp_path / "link"
        link_dir.symlink_to(real_dir)
        out = link_dir / "diff.md"
        result = runner.invoke(
            app, ["compare", "--baseline", str(baseline), "--out", str(out)]
        )
        # Must fail before the harness runs; no diff file should be written.
        assert result.exit_code != 0
        assert "symlink" in result.output.lower()
        assert not out.exists()


class TestListCommand:
    def test_list_outputs_each_conversation(self) -> None:
        # Use a wide terminal so rich doesn't wrap conv_ids mid-name.
        result = runner.invoke(app, ["list"], env={"COLUMNS": "200"})
        assert result.exit_code == 0, result.output
        assert "postgres-replication" in result.output
        assert "hnsw-tuning" in result.output
        assert "agent-led-debug" in result.output

    def test_list_table_has_headers(self) -> None:
        result = runner.invoke(app, ["list"], env={"COLUMNS": "200"})
        assert "conv_id" in result.output
        assert "topic" in result.output


class TestBaselineCommand:
    def test_baseline_writes_json(self, tmp_path: Path) -> None:
        baseline = tmp_path / "baseline.json"
        result = runner.invoke(app, ["baseline", "--save", str(baseline)])
        assert result.exit_code == 0, result.output
        assert baseline.exists()
        data = json.loads(baseline.read_text())
        # Versioned wrapper: schema_version + saved_at + report blob.
        assert data["schema_version"] >= 1
        assert "aggregate_turn_latency" in data["report"]


class TestCompareCommand:
    def test_compare_no_regression(self, tmp_path: Path) -> None:
        baseline = tmp_path / "baseline.json"
        result = runner.invoke(app, ["baseline", "--save", str(baseline)])
        assert result.exit_code == 0, result.output

        result = runner.invoke(app, ["compare", "--baseline", str(baseline)])
        assert result.exit_code == 0, result.output
        assert "no regressions" in result.output

    def test_compare_detects_wer_regression(self, tmp_path: Path) -> None:
        # Establish baseline at 0% WER, then run with injected WER -> should regress.
        baseline = tmp_path / "baseline.json"
        runner.invoke(app, ["baseline", "--save", str(baseline)])
        result = runner.invoke(
            app,
            [
                "compare",
                "--baseline",
                str(baseline),
                "--wer-substitution-rate",
                "0.5",
            ],
        )
        # Exit non-zero when WER regresses past the default 2pp threshold.
        assert result.exit_code == 1, result.output
        assert "regression detected" in result.output

    def test_compare_out_writes_diff_to_file(self, tmp_path: Path) -> None:
        # `--out PATH` writes the rendered diff to disk while still
        # printing it to stdout, so a CI step can attach the artifact
        # without scraping console output.
        baseline = tmp_path / "baseline.json"
        runner.invoke(app, ["baseline", "--save", str(baseline)])
        diff_out = tmp_path / "diff.md"
        result = runner.invoke(
            app,
            ["compare", "--baseline", str(baseline), "--out", str(diff_out)],
        )
        assert result.exit_code == 0, result.output
        assert "no regressions" in result.output
        assert diff_out.exists()
        body = diff_out.read_text()
        assert body.endswith("\n")
        assert body.strip(), "diff file should not be empty"

    def test_compare_out_refuses_symlink(self, tmp_path: Path) -> None:
        baseline = tmp_path / "baseline.json"
        runner.invoke(app, ["baseline", "--save", str(baseline)])
        target = tmp_path / "real.md"
        target.write_text("placeholder")
        link = tmp_path / "link.md"
        link.symlink_to(target)
        result = runner.invoke(
            app,
            ["compare", "--baseline", str(baseline), "--out", str(link)],
        )
        # _safe_write_text raises BadParameter; typer surfaces that as
        # exit code 2.
        assert result.exit_code != 0


class TestRenderCommand:
    def _produce_scores(self, tmp_path: Path) -> Path:
        # `render` consumes the unwrapped scores.json that `run --json` emits,
        # not the versioned baseline wrapper.
        scores = tmp_path / "scores.json"
        runner.invoke(app, ["run", "--out", str(tmp_path / "REPORT.md"), "--json", str(scores)])
        return scores

    def test_render_markdown_from_json(self, tmp_path: Path) -> None:
        scores = self._produce_scores(tmp_path)
        out = tmp_path / "REPORT.md"
        result = runner.invoke(
            app,
            ["render", "--from", str(scores), "--out", str(out), "--format", "markdown"],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        assert out.read_text().startswith("# Voice eval report")

    def test_render_html_from_json(self, tmp_path: Path) -> None:
        scores = self._produce_scores(tmp_path)
        out = tmp_path / "REPORT.html"
        result = runner.invoke(
            app,
            ["render", "--from", str(scores), "--out", str(out), "--format", "html"],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        assert out.read_text().startswith("<!doctype html>")

    def test_render_unknown_format_errors(self, tmp_path: Path) -> None:
        scores = self._produce_scores(tmp_path)
        result = runner.invoke(
            app,
            ["render", "--from", str(scores), "--out", str(tmp_path / "x"), "--format", "yaml"],
        )
        assert result.exit_code != 0

    def test_render_missing_file_exits_2(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "render",
                "--from",
                str(tmp_path / "nope.json"),
                "--out",
                str(tmp_path / "x.md"),
            ],
        )
        assert result.exit_code == 2, result.output

    def test_render_invalid_json_exits_2(self, tmp_path: Path) -> None:
        bad = tmp_path / "scores.json"
        bad.write_text("{ this is not json")
        result = runner.invoke(
            app,
            ["render", "--from", str(bad), "--out", str(tmp_path / "x.md")],
        )
        assert result.exit_code == 2, result.output


class TestRenderReadErrors:
    """render command must map read-time errors to exit 2 with a clean message."""

    def _scores_path(self, tmp_path: Path) -> Path:
        scores = tmp_path / "scores.json"
        runner.invoke(app, ["run", "--out", str(tmp_path / "REPORT.md"), "--json", str(scores)])
        return scores

    def test_render_unicode_decode_error_exits_2(self, tmp_path: Path) -> None:
        bad = tmp_path / "scores.json"
        bad.write_bytes(b"\xff\xfe not utf-8 \x80\x81")
        result = runner.invoke(app, ["render", "--from", str(bad), "--out", str(tmp_path / "x.md")])
        assert result.exit_code == 2
        assert "utf-8" in result.output.lower() or "unicode" in result.output.lower()

    def test_render_is_a_directory_exits_2(self, tmp_path: Path) -> None:
        # Passing a directory as the scores file should exit 2 cleanly.
        result = runner.invoke(
            app, ["render", "--from", str(tmp_path), "--out", str(tmp_path / "x.md")]
        )
        assert result.exit_code == 2

    def test_render_oserror_exits_2(self, tmp_path: Path) -> None:
        # Simulate an unreadable file by making it a directory whose parent
        # is a regular file (triggers NotADirectoryError, an OSError subclass).
        blocker = tmp_path / "blocker"
        blocker.write_text("file")
        bad = blocker / "scores.json"
        result = runner.invoke(
            app, ["render", "--from", str(bad), "--out", str(tmp_path / "x.md")]
        )
        assert result.exit_code == 2


class TestCompareReadErrors:
    """compare command must map read-time errors in the baseline to exit 2."""

    def test_compare_unicode_decode_error_exits_2(self, tmp_path: Path) -> None:
        bad = tmp_path / "baseline.json"
        bad.write_bytes(b"\xff\xfe not utf-8 \x80\x81")
        result = runner.invoke(app, ["compare", "--baseline", str(bad)])
        assert result.exit_code == 2
        assert "utf-8" in result.output.lower() or "unicode" in result.output.lower()

    def test_compare_is_a_directory_exits_2(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["compare", "--baseline", str(tmp_path)])
        assert result.exit_code == 2


class TestCompareThresholdsConfig:
    """--thresholds-config loads all 9 metric overrides from a JSON file."""

    def _baseline(self, tmp_path: Path) -> Path:
        bl = tmp_path / "baseline.json"
        runner.invoke(app, ["baseline", "--save", str(bl)])
        return bl

    def test_thresholds_config_overrides_wer_threshold(self, tmp_path: Path) -> None:
        bl = self._baseline(tmp_path)
        # A very tight WER threshold (0.0) means even a small injection regresses.
        cfg = tmp_path / "thresholds.json"
        cfg.write_text('{"wer": 0.0}')
        result = runner.invoke(
            app,
            [
                "compare",
                "--baseline", str(bl),
                "--wer-substitution-rate", "0.5",
                "--thresholds-config", str(cfg),
            ],
        )
        assert result.exit_code == 1
        assert "regression" in result.output.lower()

    def test_thresholds_config_all_permissive_no_regression(self, tmp_path: Path) -> None:
        bl = self._baseline(tmp_path)
        # Permissive thresholds — even large WER injection should not regress.
        cfg = tmp_path / "thresholds.json"
        cfg.write_text('{"wer": 1.0, "faithfulness": 1.0, "decisiveness": 1.0}')
        result = runner.invoke(
            app,
            [
                "compare",
                "--baseline", str(bl),
                "--wer-substitution-rate", "0.5",
                "--thresholds-config", str(cfg),
            ],
        )
        assert result.exit_code == 0
        assert "no regressions" in result.output

    def test_thresholds_config_unknown_key_exits_2(self, tmp_path: Path) -> None:
        bl = self._baseline(tmp_path)
        cfg = tmp_path / "thresholds.json"
        cfg.write_text('{"unknown_metric": 0.5}')
        result = runner.invoke(app, ["compare", "--baseline", str(bl), "--thresholds-config", str(cfg)])
        assert result.exit_code == 2
        assert "unknown" in result.output.lower()

    def test_thresholds_config_non_number_value_exits_2(self, tmp_path: Path) -> None:
        bl = self._baseline(tmp_path)
        cfg = tmp_path / "thresholds.json"
        cfg.write_text('{"wer": "loose"}')
        result = runner.invoke(app, ["compare", "--baseline", str(bl), "--thresholds-config", str(cfg)])
        assert result.exit_code == 2

    def test_thresholds_config_missing_file_exits_2(self, tmp_path: Path) -> None:
        bl = self._baseline(tmp_path)
        result = runner.invoke(
            app,
            [
                "compare",
                "--baseline", str(bl),
                "--thresholds-config", str(tmp_path / "no-such.json"),
            ],
        )
        assert result.exit_code == 2

    def test_thresholds_config_invalid_json_exits_2(self, tmp_path: Path) -> None:
        bl = self._baseline(tmp_path)
        cfg = tmp_path / "bad.json"
        cfg.write_text("not json {")
        result = runner.invoke(app, ["compare", "--baseline", str(bl), "--thresholds-config", str(cfg)])
        assert result.exit_code == 2

    def test_thresholds_config_bool_value_exits_2(self, tmp_path: Path) -> None:
        # JSON ``true`` is admitted by isinstance(int, float) — Python's
        # ``True`` is an int subclass — so a bare numeric check would
        # silently coerce ``true`` -> ``1.0`` and disable the gate for
        # that metric. Reject explicitly.
        bl = self._baseline(tmp_path)
        cfg = tmp_path / "thresholds.json"
        cfg.write_text('{"wer": true}')
        result = runner.invoke(
            app, ["compare", "--baseline", str(bl), "--thresholds-config", str(cfg)]
        )
        assert result.exit_code == 2
        assert "wer" in result.output.lower()

    def test_thresholds_config_nan_value_exits_2(self, tmp_path: Path) -> None:
        # json.loads accepts NaN; comparisons against NaN are always
        # False so any later regression check would silently pass.
        bl = self._baseline(tmp_path)
        cfg = tmp_path / "thresholds.json"
        cfg.write_text('{"wer": NaN}')
        result = runner.invoke(
            app, ["compare", "--baseline", str(bl), "--thresholds-config", str(cfg)]
        )
        assert result.exit_code == 2
        assert "wer" in result.output.lower()

    def test_thresholds_config_infinity_value_exits_2(self, tmp_path: Path) -> None:
        # An infinite tolerance is just "never gate" — make the operator
        # say so explicitly instead of silently permitting a typo.
        bl = self._baseline(tmp_path)
        cfg = tmp_path / "thresholds.json"
        cfg.write_text('{"wer": Infinity}')
        result = runner.invoke(
            app, ["compare", "--baseline", str(bl), "--thresholds-config", str(cfg)]
        )
        assert result.exit_code == 2
        assert "wer" in result.output.lower()

    def test_thresholds_config_negative_value_exits_2(self, tmp_path: Path) -> None:
        # Negative tolerance has no defensible meaning for any of the
        # ratio / latency thresholds (they're all "allowed positive
        # increase / drop"). Reject so a typo can't invert the gate.
        bl = self._baseline(tmp_path)
        cfg = tmp_path / "thresholds.json"
        cfg.write_text('{"wer": -1.0}')
        result = runner.invoke(
            app, ["compare", "--baseline", str(bl), "--thresholds-config", str(cfg)]
        )
        assert result.exit_code == 2
        assert "wer" in result.output.lower()

    def test_compare_flag_nan_rejected(self, tmp_path: Path) -> None:
        # The same gate applies to direct compare-threshold flags:
        # typer parses ``--latency-threshold-ms NaN`` to a float, so the
        # callback has to reject it before it reaches RegressionThresholds.
        bl = self._baseline(tmp_path)
        result = runner.invoke(
            app,
            [
                "compare",
                "--baseline",
                str(bl),
                "--latency-threshold-ms",
                "nan",
            ],
        )
        assert result.exit_code != 0

    def test_compare_flag_negative_rejected(self, tmp_path: Path) -> None:
        bl = self._baseline(tmp_path)
        result = runner.invoke(
            app,
            [
                "compare",
                "--baseline",
                str(bl),
                "--wer-threshold",
                "-0.01",
            ],
        )
        assert result.exit_code != 0


class TestCompareErrors:
    def test_compare_missing_baseline_exits_2(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["compare", "--baseline", str(tmp_path / "missing.json")],
        )
        assert result.exit_code == 2, result.output

    def test_compare_unversioned_baseline_exits_2(self, tmp_path: Path) -> None:
        # A pre-versioning baseline file should be rejected with a clear error.
        bad = tmp_path / "old-baseline.json"
        # Bare report blob (no schema_version wrapper).
        runner.invoke(app, ["baseline", "--save", str(tmp_path / "real.json")])
        wrapped = json.loads((tmp_path / "real.json").read_text())
        bad.write_text(json.dumps(wrapped["report"]))
        result = runner.invoke(app, ["compare", "--baseline", str(bad)])
        assert result.exit_code == 2, result.output

    def test_compare_invalid_json_exits_2(self, tmp_path: Path) -> None:
        bad = tmp_path / "baseline.json"
        bad.write_text("not json {")
        result = runner.invoke(app, ["compare", "--baseline", str(bad)])
        assert result.exit_code == 2, result.output
