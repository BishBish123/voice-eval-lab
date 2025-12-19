"""`voice-eval` CLI: run the harness over the bundled golden set + emit reports.

(The `eval` package this CLI lives next to is the metrics module — no
calls to Python's builtin `eval()` anywhere in this file.)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console

from voice_eval_lab.eval.golden import default_golden_set
from voice_eval_lab.eval.metrics import render_report, score_run
from voice_eval_lab.pipeline import MockLLM, MockSTT, MockTTS, VoicePipeline

app = typer.Typer(
    name="voice-eval",
    help="Voice agent eval harness + reference pipeline.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    out: Path = typer.Option(Path("evals/REPORT.md"), help="Markdown report output."),
    json_out: Path | None = typer.Option(None, "--json", help="Optional JSON dump."),
    wer_substitution_rate: float = typer.Option(
        0.0, help="Inject WER into mock STT (0..1) to exercise the metric."
    ),
    false_trigger_rate: float = typer.Option(
        0.0, help="Force the pipeline to emit a synthetic false-trigger turn (0 or 1)."
    ),
) -> None:
    """Score the bundled mock pipeline over the bundled golden set."""

    async def _go() -> None:
        pipeline = VoicePipeline(
            stt=MockSTT(wer_substitution_rate=wer_substitution_rate),
            llm=MockLLM(),
            tts=MockTTS(),
            false_trigger_rate=false_trigger_rate,
        )
        conversations = default_golden_set()
        runs = [await pipeline.run(c) for c in conversations]
        report = score_run(list(zip(conversations, runs, strict=True)))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_report(report))
        console.print(f"[green]wrote[/] {out}")
        if json_out is not None:
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps(report.model_dump(), indent=2))
            console.print(f"[green]wrote[/] {json_out}")
        # Echo the headline.
        console.print(render_report(report).split("## Per conversation")[0])

    asyncio.run(_go())
