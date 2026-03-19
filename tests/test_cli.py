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
        assert "aggregate_turn_latency" in data


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


class TestRenderCommand:
    def test_render_markdown_from_json(self, tmp_path: Path) -> None:
        baseline = tmp_path / "scores.json"
        runner.invoke(app, ["baseline", "--save", str(baseline)])
        out = tmp_path / "REPORT.md"
        result = runner.invoke(
            app,
            ["render", "--from", str(baseline), "--out", str(out), "--format", "markdown"],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        assert out.read_text().startswith("# Voice eval report")

    def test_render_html_from_json(self, tmp_path: Path) -> None:
        baseline = tmp_path / "scores.json"
        runner.invoke(app, ["baseline", "--save", str(baseline)])
        out = tmp_path / "REPORT.html"
        result = runner.invoke(
            app,
            ["render", "--from", str(baseline), "--out", str(out), "--format", "html"],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        assert out.read_text().startswith("<!doctype html>")

    def test_render_unknown_format_errors(self, tmp_path: Path) -> None:
        baseline = tmp_path / "scores.json"
        runner.invoke(app, ["baseline", "--save", str(baseline)])
        result = runner.invoke(
            app,
            ["render", "--from", str(baseline), "--out", str(tmp_path / "x"), "--format", "yaml"],
        )
        assert result.exit_code != 0
