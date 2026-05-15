"""`voice-eval` CLI: run the harness over the bundled golden set + emit reports.

Subcommands:

- `run`       — score the bundled mock pipeline and write a markdown / JSON report
- `list`      — enumerate the bundled golden conversations
- `baseline`  — run the eval and persist the headline scores as a baseline file
- `compare`   — diff a fresh run against a baseline, exit non-zero on regression
- `render`    — re-render an existing scores.json into markdown or HTML
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from voice_eval_lab.baseline import (
    BaselineSchemaError,
    RegressionThresholds,
    read_baseline,
    render_diffs,
    write_baseline,
)
from voice_eval_lab.baseline import (
    compare as compare_reports,
)
from voice_eval_lab.eval.golden import default_golden_set
from voice_eval_lab.eval.metrics import (
    EvalReport,
    IncompleteRunError,
    render_report,
    render_report_html,
    score_run,
)
from voice_eval_lab.pipeline import MockLLM, MockSTT, MockTTS, VoicePipeline

app = typer.Typer(
    name="voice-eval",
    help="Voice agent eval harness + reference pipeline.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _validate_unit_interval(value: float) -> float:
    """Reject CLI floats outside [0.0, 1.0] with a clear typer error."""
    if not 0.0 <= value <= 1.0:
        raise typer.BadParameter(f"must be in [0.0, 1.0], got {value!r}")
    return value


def _assert_no_symlink_ancestor(path: Path) -> None:
    """Raise `typer.BadParameter` if any *parent* of `path` is a symlink.

    Checking only the final path component misses the case where an
    attacker pre-plants a symlink earlier in the chain
    (e.g. ``/tmp/evil -> /etc``) so that writing to
    ``/tmp/evil/output.md`` redirects into ``/etc/output.md``.
    Walking all parents catches that class of attack.
    """
    for parent in path.parents:
        if parent.is_symlink():
            raise typer.BadParameter(
                f"refusing to write: parent directory is a symlink: {parent}"
            )


def _safe_write_text(path: Path, body: str) -> None:
    """Write `body` to `path`, refusing symlinks and surfacing OSError cleanly.

    Refusing symlinks keeps a malicious or stale link from redirecting
    a `voice-eval` write into an unrelated file (a security paper-cut
    on shared dev boxes). OSError is mapped to `typer.Exit(2)` with a
    readable message so common failures (read-only mount, permissions,
    no space left) don't surface as raw Python tracebacks.
    """
    _assert_no_symlink_ancestor(path)
    if path.is_symlink():
        raise typer.BadParameter(f"refusing to write to symlink: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    except OSError as exc:
        console.print(f"[red]could not write {path}:[/] {exc}")
        raise typer.Exit(code=2) from None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_eval(
    *,
    wer_substitution_rate: float = 0.0,
    false_trigger_rate: float = 0.0,
) -> EvalReport:
    async def _go() -> EvalReport:
        pipeline = VoicePipeline(
            stt=MockSTT(wer_substitution_rate=wer_substitution_rate),
            llm=MockLLM(),
            tts=MockTTS(),
            false_trigger_rate=false_trigger_rate,
        )
        conversations = default_golden_set()
        runs = [await pipeline.run(c) for c in conversations]
        # Thread the pipeline's actual configured budget through to the
        # scorer so the metric and the pipeline contract cannot drift.
        return score_run(
            list(zip(conversations, runs, strict=True)),
            barge_in_budget_ms=pipeline.barge_in_yield_ms,
        )

    return asyncio.run(_go())


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def run(
    out: Path = typer.Option(Path("evals/REPORT.md"), help="Markdown report output."),
    json_out: Path | None = typer.Option(None, "--json", help="Optional JSON dump."),
    wer_substitution_rate: float = typer.Option(
        0.0,
        help="Inject WER into mock STT (0..1) to exercise the metric.",
        callback=_validate_unit_interval,
    ),
    false_trigger_rate: float = typer.Option(
        0.0,
        help=(
            "Per-user-turn Bernoulli probability of injecting a synthetic "
            "false-trigger turn (0..1). A fixed seed is used so corpus runs "
            "are reproducible."
        ),
        callback=_validate_unit_interval,
    ),
) -> None:
    """Score the bundled mock pipeline over the bundled golden set."""
    try:
        report = _run_eval(
            wer_substitution_rate=wer_substitution_rate,
            false_trigger_rate=false_trigger_rate,
        )
    except IncompleteRunError as exc:
        console.print(f"[red]incomplete run:[/] {exc}")
        raise typer.Exit(code=2) from None
    _safe_write_text(out, render_report(report))
    console.print(f"[green]wrote[/] {out}")
    if json_out is not None:
        _safe_write_text(json_out, json.dumps(report.model_dump(), indent=2))
        console.print(f"[green]wrote[/] {json_out}")
    # Echo the headline.
    console.print(render_report(report).split("## Per conversation")[0])


@app.command("list")
def list_cmd() -> None:
    """List the bundled golden conversations."""
    table = Table(title="Bundled golden set")
    table.add_column("conv_id")
    table.add_column("topic")
    table.add_column("user turns", justify="right")
    table.add_column("interrupted", justify="right")
    table.add_column("gold facts", justify="right")
    for conv in default_golden_set():
        n_user = sum(1 for t in conv.turns if t.role.value == "user")
        n_int = sum(1 for t in conv.turns if t.interrupted)
        table.add_row(
            conv.conv_id,
            conv.topic,
            str(n_user),
            str(n_int),
            str(len(conv.gold_facts)),
        )
    console.print(table)


@app.command()
def baseline(
    save: Path = typer.Option(..., "--save", help="JSON file to write the baseline to."),
    wer_substitution_rate: float = typer.Option(0.0, callback=_validate_unit_interval),
    false_trigger_rate: float = typer.Option(0.0, callback=_validate_unit_interval),
) -> None:
    """Run the eval and persist the headline scores as a baseline."""
    _assert_no_symlink_ancestor(save)
    if save.is_symlink():
        raise typer.BadParameter(f"refusing to write to symlink: {save}")
    try:
        report = _run_eval(
            wer_substitution_rate=wer_substitution_rate,
            false_trigger_rate=false_trigger_rate,
        )
    except IncompleteRunError as exc:
        console.print(f"[red]incomplete run:[/] {exc}")
        raise typer.Exit(code=2) from None
    try:
        write_baseline(report, save)
    except OSError as exc:
        console.print(f"[red]could not write baseline {save}:[/] {exc}")
        raise typer.Exit(code=2) from None
    console.print(f"[green]wrote baseline[/] {save}")


@app.command()
def compare(
    baseline_path: Path = typer.Option(
        ..., "--baseline", help="Baseline JSON file to diff against."
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help=(
            "Optional file path to write the rendered diff to. The diff is "
            "still printed to stdout regardless; --out adds a copy on disk "
            "for CI artifacts or PR comments."
        ),
    ),
    wer_substitution_rate: float = typer.Option(0.0, callback=_validate_unit_interval),
    false_trigger_rate: float = typer.Option(0.0, callback=_validate_unit_interval),
    latency_threshold_ms: float = typer.Option(10.0, help="Allowed p95 latency increase (ms)."),
    wer_threshold: float = typer.Option(0.02, help="Allowed WER increase (fraction)."),
    faithfulness_threshold: float = typer.Option(
        0.05, help="Allowed faithfulness drop (fraction)."
    ),
) -> None:
    """Diff a fresh run against a baseline; exit non-zero if any metric regressed."""
    try:
        base = read_baseline(baseline_path)
    except FileNotFoundError:
        console.print(f"[red]baseline not found:[/] {baseline_path}", style="bold")
        raise typer.Exit(code=2) from None
    except json.JSONDecodeError as exc:
        console.print(f"[red]baseline is not valid JSON:[/] {baseline_path} ({exc})")
        raise typer.Exit(code=2) from None
    except BaselineSchemaError as exc:
        console.print(f"[red]baseline schema error:[/] {exc}")
        raise typer.Exit(code=2) from None
    except ValidationError as exc:
        console.print(f"[red]baseline failed validation:[/] {exc}")
        raise typer.Exit(code=2) from None
    try:
        current = _run_eval(
            wer_substitution_rate=wer_substitution_rate,
            false_trigger_rate=false_trigger_rate,
        )
    except IncompleteRunError as exc:
        console.print(f"[red]incomplete run:[/] {exc}")
        raise typer.Exit(code=2) from None
    thresholds = RegressionThresholds(
        latency_p95_ms=latency_threshold_ms,
        wer=wer_threshold,
        faithfulness=faithfulness_threshold,
    )
    diffs = compare_reports(base, current, thresholds)
    rendered = render_diffs(diffs)
    console.print(rendered)
    if out is not None:
        _safe_write_text(out, rendered + "\n" if not rendered.endswith("\n") else rendered)
        console.print(f"[green]wrote diff to[/] {out}")
    if any(d.regressed for d in diffs):
        console.print("[red]regression detected[/]")
        sys.exit(1)
    console.print("[green]no regressions[/]")


@app.command()
def render(
    json_path: Path = typer.Option(
        Path("evals/scores.json"), "--from", help="Source JSON scores file."
    ),
    out: Path = typer.Option(Path("evals/REPORT.md"), help="Output report path."),
    fmt: str = typer.Option("markdown", "--format", help="One of: markdown, html."),
) -> None:
    """Re-render an existing scores.json into markdown or HTML."""
    try:
        report = EvalReport.model_validate(json.loads(json_path.read_text()))
    except FileNotFoundError:
        console.print(f"[red]scores file not found:[/] {json_path}")
        raise typer.Exit(code=2) from None
    except json.JSONDecodeError as exc:
        console.print(f"[red]scores file is not valid JSON:[/] {json_path} ({exc})")
        raise typer.Exit(code=2) from None
    except ValidationError as exc:
        console.print(f"[red]scores file failed validation:[/] {exc}")
        raise typer.Exit(code=2) from None
    if fmt == "markdown":
        body = render_report(report)
    elif fmt == "html":
        body = render_report_html(report)
    else:
        raise typer.BadParameter(f"unknown format: {fmt!r}")
    _safe_write_text(out, body)
    console.print(f"[green]wrote[/] {out}")
