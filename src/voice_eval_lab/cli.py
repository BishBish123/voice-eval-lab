"""`voice-eval` CLI: run the harness over the bundled golden set + emit reports.

Subcommands:

- `run`       — score the bundled mock pipeline and write a markdown / JSON report
- `list`      — enumerate the bundled golden conversations
- `baseline`  — run the eval and persist the headline scores as a baseline file
- `compare`   — diff a fresh run against a baseline, exit non-zero on regression
- `render`    — re-render an existing scores.json into markdown or HTML
- `calibrate` — run the LLM judge over a CSV of human-labelled samples and compute Cohen's kappa
- `notes`     — manage the in-memory notes store (add / lookup / clear)
- `serve`     — start the FastAPI backend with uvicorn
- `pipeline`  — Pipecat pipeline commands: run (in-memory) and serve (LiveKit)
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
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
    _real_turn_runs,
    render_report,
    render_report_html,
    score_run,
)
from voice_eval_lab.judge.factory import make_judge
from voice_eval_lab.judge.llm import LLMJudge
from voice_eval_lab.judge.protocol import JudgeProtocol
from voice_eval_lab.models import Conversation, ConversationRun, TurnRole
from voice_eval_lab.notes.memory import InMemoryNotesStore
from voice_eval_lab.notes.protocol import NotesStore
from voice_eval_lab.pipeline import STT, TTS, MockLLM, MockSTT, MockTTS, VoicePipeline

app = typer.Typer(
    name="voice-eval",
    help="Voice agent eval harness + reference pipeline.",
    add_completion=False,
    no_args_is_help=True,
)

notes_app = typer.Typer(
    name="notes",
    help="Manage the session notes store (add / lookup / clear).",
    add_completion=False,
    no_args_is_help=True,
)
app.add_typer(notes_app, name="notes")

pipeline_app = typer.Typer(
    name="pipeline",
    help="Pipecat pipeline commands: run (in-memory smoke test) and serve (LiveKit).",
    add_completion=False,
    no_args_is_help=True,
)
app.add_typer(pipeline_app, name="pipeline")

audio_app = typer.Typer(
    name="audio",
    help="Audio fixture management: populate silence, import WAV files, list keys.",
    add_completion=False,
    no_args_is_help=True,
)
app.add_typer(audio_app, name="audio")

console = Console()


def _validate_unit_interval(value: float) -> float:
    """Reject CLI floats outside [0.0, 1.0] with a clear typer error."""
    if not 0.0 <= value <= 1.0:
        raise typer.BadParameter(f"must be in [0.0, 1.0], got {value!r}")
    return value


def _validate_compare_threshold_value(key: str, val: object) -> float:
    """Validate one compare-threshold value, returning a finite non-negative float.

    Rejects:

    - ``bool`` (Python's ``True`` / ``False`` are subclasses of ``int``,
      so a bare ``isinstance(val, (int, float))`` happily coerces them
      to ``1.0`` / ``0.0`` and silently disables regression detection
      for the affected metric — check ``type(val) is bool`` *before*
      the numeric isinstance).
    - ``NaN`` / ``Infinity`` — Python's ``json.loads`` admits both as
      JSON numbers, and any later ``delta > nan`` comparison is always
      ``False``, so an attacker (or a typo) could disable a regression
      gate without an error.
    - Negative values — a negative tolerance would treat *any* drop as
      a regression (or *no* drop), neither of which is what an operator
      meant.

    Raises ``ValueError`` naming the offending key on any rejection so
    the caller can surface a clear CLI / config error.
    """
    if type(val) is bool:
        raise ValueError(
            f"threshold value for {key!r} must be a number, got bool ({val!r})"
        )
    if not isinstance(val, (int, float)):
        raise ValueError(
            f"threshold value for {key!r} must be a number, got {val!r}"
        )
    fval = float(val)
    if math.isnan(fval) or math.isinf(fval):
        raise ValueError(
            f"threshold value for {key!r} must be finite, got {val!r}"
        )
    if fval < 0:
        raise ValueError(
            f"threshold value for {key!r} must be non-negative, got {val!r}"
        )
    return fval


def _validate_compare_threshold_flag(value: float) -> float:
    """Typer callback: reject NaN, Infinity, or negative compare-threshold flags.

    typer parses ``--latency-threshold-ms`` / ``--wer-threshold`` /
    ``--faithfulness-threshold`` as ``float``, which silently accepts
    ``nan`` and ``inf`` from the shell.  Apply the same finite +
    non-negative gate the JSON config path uses.
    """
    try:
        return _validate_compare_threshold_value("<flag>", value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None


def _load_thresholds_config(path: Path) -> dict[str, float]:
    """Load a JSON thresholds-config file and return a validated {key: float} dict.

    Exits with code 2 on any read or parse error.  Unknown keys and
    non-numeric values are also rejected so a typo in the config file
    does not silently fall back to the library default.
    """
    try:
        raw_cfg = path.read_text()
    except FileNotFoundError:
        console.print(f"[red]thresholds config not found:[/] {path}")
        raise typer.Exit(code=2) from None
    except IsADirectoryError:
        console.print(f"[red]thresholds config path is a directory:[/] {path}")
        raise typer.Exit(code=2) from None
    except UnicodeDecodeError as exc:
        console.print(f"[red]thresholds config is not valid UTF-8:[/] {path} ({exc})")
        raise typer.Exit(code=2) from None
    except OSError as exc:
        console.print(f"[red]could not read thresholds config {path}:[/] {exc}")
        raise typer.Exit(code=2) from None
    try:
        cfg_data = json.loads(raw_cfg)
    except json.JSONDecodeError as exc:
        console.print(f"[red]thresholds config is not valid JSON:[/] {path} ({exc})")
        raise typer.Exit(code=2) from None
    if not isinstance(cfg_data, dict):
        console.print(f"[red]thresholds config must be a JSON object:[/] {path}")
        raise typer.Exit(code=2) from None
    valid_keys = {f.name for f in dataclasses.fields(RegressionThresholds)}
    result: dict[str, float] = {}
    for key, val in cfg_data.items():
        if key not in valid_keys:
            console.print(
                f"[red]unknown threshold key {key!r}[/] (valid: {sorted(valid_keys)})"
            )
            raise typer.Exit(code=2) from None
        try:
            result[key] = _validate_compare_threshold_value(key, val)
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=2) from None
    return result


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
            resolved_parent = parent.resolve()
            suggested = path.resolve()
            raise typer.BadParameter(
                f"refusing to write: parent directory is a symlink: {parent} "
                f"-> {resolved_parent} (suggested workaround: pass the resolved "
                f"path instead, e.g. {suggested})"
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


_VALID_JUDGE_MODES = frozenset({"auto", "llm", "substring"})
_VALID_STT_MODES = frozenset({"auto", "mock", "deepgram", "whisper"})
_VALID_TTS_MODES = frozenset({"auto", "mock", "cartesia", "elevenlabs"})
_VALID_TURN_DETECTOR_MODES = frozenset({"smart", "none"})


def _validate_judge_flag(mode: str) -> None:
    """Raise ``typer.BadParameter`` if *mode* is not a recognised judge mode."""
    if mode not in _VALID_JUDGE_MODES:
        raise typer.BadParameter(
            f"Unknown judge mode: {mode!r}. Valid values: {sorted(_VALID_JUDGE_MODES)}."
        )


def _validate_stt_flag(mode: str) -> None:
    """Raise ``typer.BadParameter`` if *mode* is not a recognised STT mode."""
    if mode not in _VALID_STT_MODES:
        raise typer.BadParameter(
            f"Unknown STT mode: {mode!r}. Valid values: {sorted(_VALID_STT_MODES)}."
        )


def _validate_tts_flag(mode: str) -> None:
    """Raise ``typer.BadParameter`` if *mode* is not a recognised TTS mode."""
    if mode not in _VALID_TTS_MODES:
        raise typer.BadParameter(
            f"Unknown TTS mode: {mode!r}. Valid values: {sorted(_VALID_TTS_MODES)}."
        )


def _validate_turn_detector_flag(mode: str) -> None:
    """Raise ``typer.BadParameter`` if *mode* is not a recognised turn-detector mode."""
    if mode not in _VALID_TURN_DETECTOR_MODES:
        raise typer.BadParameter(
            f"Unknown turn-detector mode: {mode!r}. "
            f"Valid values: {sorted(_VALID_TURN_DETECTOR_MODES)}."
        )


def _make_stt_from_flag(mode: str) -> STT:
    """Construct an STT adapter from the --stt CLI flag.

    Modes
    -----
    ``auto``     — delegate to :func:`~voice_eval_lab.adapters.make_stt` (env-var dispatch).
    ``mock``     — always :class:`~voice_eval_lab.pipeline.MockSTT`.
    ``deepgram`` — always :class:`~voice_eval_lab.adapters.DeepgramSTT`
                   (mock path when ``DEEPGRAM_API_KEY`` is absent).
    ``whisper``  — always :class:`~voice_eval_lab.adapters.WhisperSTT`
                   using ``WHISPER_MODEL_NAME`` env var (default: ``tiny``).
    """
    import os

    from voice_eval_lab.adapters import DeepgramSTT, WhisperSTT
    from voice_eval_lab.adapters import make_stt as _make_stt

    if mode == "auto":
        return _make_stt()
    if mode == "mock":
        return MockSTT()
    if mode == "deepgram":
        return DeepgramSTT()
    if mode == "whisper":
        model_name = os.environ.get("WHISPER_MODEL_NAME", "tiny")
        return WhisperSTT(model_name=model_name)
    # Unreachable after _validate_stt_flag, but be explicit.
    raise typer.BadParameter(f"Unknown STT mode: {mode!r}")


def _make_tts_from_flag(mode: str) -> TTS:
    """Construct a TTS adapter from the --tts CLI flag.

    Modes
    -----
    ``auto``       — delegate to :func:`~voice_eval_lab.adapters.make_tts` (env-var dispatch).
    ``mock``       — always :class:`~voice_eval_lab.pipeline.MockTTS`.
    ``cartesia``   — always :class:`~voice_eval_lab.adapters.CartesiaTTS`
                     (mock path when ``CARTESIA_API_KEY`` is absent).
    ``elevenlabs`` — always :class:`~voice_eval_lab.adapters.ElevenLabsTTS`
                     (mock path when ``ELEVENLABS_API_KEY`` is absent).
    """
    from voice_eval_lab.adapters import CartesiaTTS, ElevenLabsTTS
    from voice_eval_lab.adapters import make_tts as _make_tts

    if mode == "auto":
        return _make_tts()
    if mode == "mock":
        return MockTTS()
    if mode == "cartesia":
        return CartesiaTTS()
    if mode == "elevenlabs":
        return ElevenLabsTTS()
    # Unreachable after _validate_tts_flag, but be explicit.
    raise typer.BadParameter(f"Unknown TTS mode: {mode!r}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_notes_fixture(path: Path) -> InMemoryNotesStore:
    """Load a notes fixture JSON into an :class:`InMemoryNotesStore`.

    The JSON must be a list of objects with ``note_id`` and ``text`` keys.
    An optional ``embedding`` list key is also accepted.

    Runs synchronously (builds the store in a fresh event loop).
    """
    try:
        raw = path.read_text()
    except FileNotFoundError:
        console.print(f"[red]notes fixture not found:[/] {path}")
        raise typer.Exit(code=2) from None
    except OSError as exc:
        console.print(f"[red]could not read notes fixture {path}:[/] {exc}")
        raise typer.Exit(code=2) from None
    try:
        records = json.loads(raw)
    except json.JSONDecodeError as exc:
        console.print(f"[red]notes fixture is not valid JSON:[/] {path} ({exc})")
        raise typer.Exit(code=2) from None
    if not isinstance(records, list):
        console.print(f"[red]notes fixture must be a JSON array:[/] {path}")
        raise typer.Exit(code=2) from None

    store = InMemoryNotesStore()

    async def _load() -> None:
        for item in records:
            if not isinstance(item, dict) or "note_id" not in item or "text" not in item:
                console.print("[red]each notes fixture entry must have note_id and text[/]")
                raise typer.Exit(code=2)
            await store.add_note(
                note_id=item["note_id"],
                text=item["text"],
                embedding=item.get("embedding"),
            )

    asyncio.run(_load())
    return store


def _run_eval(
    *,
    wer_substitution_rate: float = 0.0,
    false_trigger_rate: float = 0.0,
    judge: JudgeProtocol | None = None,
    notes_fixture: Path | None = None,
    audio_fixtures: Path | None = None,
    stt: STT | None = None,
    tts: TTS | None = None,
) -> EvalReport:
    """Run the eval. Caller must supply the judge — judge construction may
    raise OSError/ImportError for missing keys/deps, and that should be
    handled with a tight catch at the call site, not folded into eval errors.

    When ``stt`` is ``None``, defaults to ``MockSTT(wer_substitution_rate=...)``.
    When ``tts`` is ``None``, defaults to ``MockTTS()``.
    """

    # Load the fixture synchronously before entering the event loop to avoid
    # nested asyncio.run() (asyncio.run cannot be called from a running loop).
    pre_loaded_store = _load_notes_fixture(notes_fixture) if notes_fixture is not None else None

    # Build the audio fixture store if a directory was supplied.
    audio_store = None
    if audio_fixtures is not None:
        from voice_eval_lab.audio import FilesystemAudioStore

        audio_store = FilesystemAudioStore(audio_fixtures)

    async def _go() -> EvalReport:
        base_llm: object = MockLLM()
        if pre_loaded_store is not None:
            from voice_eval_lab.notes.llm_adapter import WithNotesLLM

            llm: object = WithNotesLLM(inner=base_llm, store=pre_loaded_store)
        else:
            llm = base_llm

        # When an explicit STT adapter is provided use it; otherwise default
        # to MockSTT so that wer_substitution_rate is still honoured.
        effective_stt: STT = stt if stt is not None else MockSTT(wer_substitution_rate=wer_substitution_rate)
        # When an explicit TTS adapter is provided use it; otherwise default to MockTTS.
        effective_tts: TTS = tts if tts is not None else MockTTS()

        pipeline = VoicePipeline(
            stt=effective_stt,
            llm=llm,  # type: ignore[arg-type]
            tts=effective_tts,
            false_trigger_rate=false_trigger_rate,
        )
        conversations = default_golden_set()
        runs = [await pipeline.run(c, audio_store=audio_store) for c in conversations]
        # Thread the pipeline's actual configured budget through to the
        # scorer so the metric and the pipeline contract cannot drift.
        report = score_run(
            list(zip(conversations, runs, strict=True)),
            barge_in_budget_ms=pipeline.barge_in_yield_ms,
        )

        # Substring judge produces the same numbers already in `report`,
        # so only re-score when a real LLM judge was selected.
        if judge is not None and isinstance(judge, LLMJudge):
            report = await _apply_llm_faithfulness(judge, conversations, runs, report)

        return report

    return asyncio.run(_go())


async def _apply_llm_faithfulness(
    judge: object,
    conversations: list[Conversation],
    runs: list[ConversationRun],
    report: EvalReport,
) -> EvalReport:
    """Re-score faithfulness using the LLM judge and patch the EvalReport.

    The per-conversation and aggregate faithfulness values are replaced in
    a new EvalReport so no other metric is touched.
    """
    assert isinstance(judge, JudgeProtocol)

    grounded_total = 0
    ground_count = 0
    new_per_conv = list(report.per_conversation)

    for idx, (conv, run) in enumerate(zip(conversations, runs, strict=True)):
        if not conv.gold_facts:
            continue  # vacuously faithful — unchanged
        user_turns = [t for t in conv.turns if t.role is TurnRole.USER]
        real_turns = _real_turn_runs(run)
        grounded = 0
        counted = 0
        for i, tr in enumerate(real_turns):
            if i < len(user_turns) and not user_turns[i].text.strip():
                continue
            counted += 1
            question = user_turns[i].text if i < len(user_turns) else ""
            result = await judge.score(
                question=question,
                expected_keypoints=conv.gold_facts,
                answer=tr.agent_reply,
            )
            if result.score >= 0.5:
                grounded += 1
        conv_faith = grounded / counted if counted else 0.0
        grounded_total += grounded
        ground_count += counted
        # Patch per-conversation faithfulness.
        old = new_per_conv[idx]
        new_per_conv[idx] = old.model_copy(update={"response_faithfulness": conv_faith})

    agg_faith = grounded_total / ground_count if ground_count > 0 else report.aggregate_faithfulness
    return report.model_copy(
        update={
            "aggregate_faithfulness": agg_faith,
            "per_conversation": new_per_conv,
        }
    )


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
    judge: str = typer.Option(
        "auto",
        help=(
            "Faithfulness judge backend: 'auto' (default) uses LLMJudge when "
            "ANTHROPIC_API_KEY or OPENAI_API_KEY is set, else SubstringProxyJudge. "
            "'llm' forces LLMJudge (raises if no key is set). "
            "'substring' forces SubstringProxyJudge regardless of env keys."
        ),
    ),
    with_notes: Path | None = typer.Option(
        None,
        "--with-notes",
        help=(
            "Path to a notes fixture JSON (array of {note_id, text} objects). "
            "Loads the notes into an InMemoryNotesStore and wraps the LLM with "
            "WithNotesLLM so relevant notes are prepended to each reply."
        ),
    ),
    audio_fixtures: Path | None = typer.Option(
        None,
        "--audio-fixtures",
        help=(
            "Path to an audio fixture directory (e.g. evals/audio/). "
            "Loads a FilesystemAudioStore and attaches WAV bytes to each Turn "
            "before the STT adapter is called. When DEEPGRAM_API_KEY is set "
            "the real Deepgram adapter will receive actual audio."
        ),
    ),
    stt: str = typer.Option(
        "auto",
        "--stt",
        help=(
            "STT adapter to use: 'auto' (default) dispatches based on env vars "
            "(DEEPGRAM_API_KEY → deepgram; WHISPER_MODEL_NAME → whisper; else mock). "
            "'mock' forces MockSTT regardless of env vars. "
            "'deepgram' forces DeepgramSTT (mock path when DEEPGRAM_API_KEY is absent). "
            "'whisper' forces WhisperSTT using WHISPER_MODEL_NAME (default model: tiny)."
        ),
    ),
    tts: str = typer.Option(
        "auto",
        "--tts",
        help=(
            "TTS adapter to use: 'auto' (default) dispatches based on env vars "
            "(CARTESIA_API_KEY → cartesia; ELEVENLABS_API_KEY → elevenlabs; else mock). "
            "When both keys are set, cartesia wins (lower streaming latency). "
            "'mock' forces MockTTS regardless of env vars. "
            "'cartesia' forces CartesiaTTS (mock path when CARTESIA_API_KEY is absent). "
            "'elevenlabs' forces ElevenLabsTTS (mock path when ELEVENLABS_API_KEY is absent)."
        ),
    ),
) -> None:
    """Score the bundled mock pipeline over the bundled golden set."""
    _validate_judge_flag(judge)
    _validate_stt_flag(stt)
    _validate_tts_flag(tts)
    # Construct the judge before the eval so OSError (missing key) and
    # ImportError (missing httpx) get a friendly message, narrowly scoped —
    # no risk of swallowing a legitimate OSError from inside the eval loop.
    try:
        judge_instance = make_judge(mode=judge)
    except (OSError, ImportError) as exc:
        console.print(f"[red]judge error:[/] {exc}")
        raise typer.Exit(code=2) from None
    stt_instance = _make_stt_from_flag(stt)
    tts_instance = _make_tts_from_flag(tts)
    try:
        report = _run_eval(
            wer_substitution_rate=wer_substitution_rate,
            false_trigger_rate=false_trigger_rate,
            judge=judge_instance,
            notes_fixture=with_notes,
            audio_fixtures=audio_fixtures,
            stt=stt_instance,
            tts=tts_instance,
        )
    except IncompleteRunError as exc:
        console.print(f"[red]incomplete run:[/] {exc}")
        raise typer.Exit(code=2) from None
    _safe_write_text(out, render_report(report))
    console.print(f"[green]wrote[/] {out}")
    if json_out is not None:
        _safe_write_text(json_out, json.dumps(report.model_dump(), indent=2))
        console.print(f"[green]wrote[/] {json_out}")
    # Echo the headline only — the per-conversation table is intentionally
    # truncated from stdout so a corpus run doesn't flood the terminal.
    # Tell the reader where to find the full table so the truncation is
    # discoverable without reading the source.
    console.print(render_report(report).split("## Per conversation")[0])
    console.print(f"[dim](full per-conversation table in {out})[/]")


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
    latency_threshold_ms: float = typer.Option(
        10.0,
        help="Allowed p95 latency increase (ms).",
        callback=_validate_compare_threshold_flag,
    ),
    wer_threshold: float = typer.Option(
        0.02,
        help="Allowed WER increase (fraction).",
        callback=_validate_compare_threshold_flag,
    ),
    faithfulness_threshold: float = typer.Option(
        0.05,
        help="Allowed faithfulness drop (fraction).",
        callback=_validate_compare_threshold_flag,
    ),
    thresholds_config: Path | None = typer.Option(
        None,
        "--thresholds-config",
        help=(
            "JSON file mapping metric names to tolerance values. "
            "Overrides the per-metric CLI flags and library defaults for "
            "any key present. Valid keys: latency_p95_ms, wer, faithfulness, "
            "barge_in, false_trigger, barge_in_latency_p95_ms, "
            "tts_first_byte_jitter_ms, endpointing, decisiveness."
        ),
    ),
) -> None:
    """Diff a fresh run against a baseline; exit non-zero if any metric regressed."""
    try:
        base = read_baseline(baseline_path)
    except FileNotFoundError:
        console.print(f"[red]baseline not found:[/] {baseline_path}", style="bold")
        raise typer.Exit(code=2) from None
    except IsADirectoryError:
        console.print(f"[red]baseline path is a directory, not a file:[/] {baseline_path}")
        raise typer.Exit(code=2) from None
    except UnicodeDecodeError as exc:
        console.print(f"[red]baseline is not valid UTF-8:[/] {baseline_path} ({exc})")
        raise typer.Exit(code=2) from None
    except OSError as exc:
        console.print(f"[red]could not read baseline {baseline_path}:[/] {exc}")
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
    # Guard --out path before running the harness so a bad path (e.g. /tmp
    # on macOS, which is a symlink to /private/tmp) fails fast, mirroring
    # the `run` subcommand's ordering.
    if out is not None:
        try:
            _assert_no_symlink_ancestor(out)
        except typer.BadParameter as exc:
            raise typer.BadParameter(str(exc)) from None
        if out.is_symlink():
            raise typer.BadParameter(f"refusing to write to symlink: {out}")
    try:
        current = _run_eval(
            wer_substitution_rate=wer_substitution_rate,
            false_trigger_rate=false_trigger_rate,
        )
    except IncompleteRunError as exc:
        console.print(f"[red]incomplete run:[/] {exc}")
        raise typer.Exit(code=2) from None
    # Start from the per-metric CLI flags, then overlay any keys present in
    # the --thresholds-config JSON file so the full 9-metric surface is
    # reachable without enumerating each one on the command line.
    threshold_overrides = (
        _load_thresholds_config(thresholds_config)
        if thresholds_config is not None
        else {}
    )
    _defaults = RegressionThresholds()
    thresholds = RegressionThresholds(
        latency_p95_ms=threshold_overrides.get("latency_p95_ms", latency_threshold_ms),
        wer=threshold_overrides.get("wer", wer_threshold),
        faithfulness=threshold_overrides.get("faithfulness", faithfulness_threshold),
        barge_in=threshold_overrides.get("barge_in", _defaults.barge_in),
        false_trigger=threshold_overrides.get("false_trigger", _defaults.false_trigger),
        barge_in_latency_p95_ms=threshold_overrides.get(
            "barge_in_latency_p95_ms", _defaults.barge_in_latency_p95_ms
        ),
        tts_first_byte_jitter_ms=threshold_overrides.get(
            "tts_first_byte_jitter_ms", _defaults.tts_first_byte_jitter_ms
        ),
        endpointing=threshold_overrides.get("endpointing", _defaults.endpointing),
        decisiveness=threshold_overrides.get("decisiveness", _defaults.decisiveness),
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
        raw_text = json_path.read_text()
    except FileNotFoundError:
        console.print(f"[red]scores file not found:[/] {json_path}")
        raise typer.Exit(code=2) from None
    except IsADirectoryError:
        console.print(f"[red]scores path is a directory, not a file:[/] {json_path}")
        raise typer.Exit(code=2) from None
    except UnicodeDecodeError as exc:
        console.print(f"[red]scores file is not valid UTF-8:[/] {json_path} ({exc})")
        raise typer.Exit(code=2) from None
    except OSError as exc:
        console.print(f"[red]could not read scores file {json_path}:[/] {exc}")
        raise typer.Exit(code=2) from None
    try:
        report = EvalReport.model_validate(json.loads(raw_text))
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


@app.command()
def calibrate(
    labels: Path = typer.Option(
        Path("evals/calibration.csv"),
        help="CSV file with human-labelled question/answer pairs.",
    ),
    out: Path = typer.Option(
        Path("evals/CALIBRATION.md"),
        help="Output Markdown report with kappa value and divergence cases.",
    ),
    judge: str = typer.Option(
        "auto",
        help=(
            "Judge backend for LLM scoring: 'auto', 'llm', or 'substring'. "
            "Using 'substring' computes kappa against the substring proxy, "
            "not a real LLM — useful for testing the harness itself."
        ),
    ),
) -> None:
    """Compute Cohen's kappa between human labels and the LLM judge.

    Reads a CSV of human-labelled question/answer pairs, runs the judge
    over the same set, and computes Cohen's kappa.  Writes a Markdown
    report to --out.

    \\b
    WARNING: The bundled evals/calibration.csv contains placeholder human
    labels.  The kappa value is meaningless until real human annotations
    replace the human_score column.
    """
    _validate_judge_flag(judge)
    try:
        from voice_eval_lab.judge.calibration import calibration_cli_main

        calibration_cli_main(csv_path=labels, out_path=out, judge_mode=judge)
        console.print(f"[green]wrote calibration report[/] {out}")
    except FileNotFoundError:
        console.print(f"[red]calibration CSV not found:[/] {labels}")
        raise typer.Exit(code=2) from None
    except ValueError as exc:
        console.print(f"[red]calibration CSV error:[/] {exc}")
        raise typer.Exit(code=2) from None
    except OSError as exc:
        console.print(f"[red]judge configuration error:[/] {exc}")
        raise typer.Exit(code=2) from None


# ---------------------------------------------------------------------------
# Notes sub-app
# ---------------------------------------------------------------------------

# Module-level singleton used by the notes CLI commands so that add / lookup /
# clear within a single shell session share state.  Re-using the same process
# (e.g. CliRunner in tests) preserves state between calls, which is the
# desired behaviour for integration tests that chain add → lookup.
_notes_store_singleton: NotesStore | None = None


def _get_notes_store() -> NotesStore:
    """Return the module-level InMemoryNotesStore, creating it on first call."""
    global _notes_store_singleton
    if _notes_store_singleton is None:
        _notes_store_singleton = InMemoryNotesStore()
    return _notes_store_singleton


@notes_app.command("add")
def notes_add(
    note_id: str = typer.Option(..., "--id", help="Unique identifier for the note."),
    text: str = typer.Option(..., "--text", help="Note text to store and embed."),
) -> None:
    """Add a note to the in-memory notes store."""
    store = _get_notes_store()

    async def _go() -> None:
        await store.add_note(note_id=note_id, text=text)

    asyncio.run(_go())
    console.print(f"[green]added note[/] {note_id!r}")


@notes_app.command("lookup")
def notes_lookup(
    query: str = typer.Option(..., "--query", help="Free-text query to search notes."),
    top_k: int = typer.Option(3, "--top-k", help="Number of hits to return."),
    fixture: Path | None = typer.Option(
        None,
        "--fixture",
        help=(
            "Optional notes fixture JSON to load before querying. "
            "Entries are merged with any notes already in the store."
        ),
    ),
) -> None:
    """Look up the most relevant notes for a query."""
    store = _get_notes_store()

    # Load fixture synchronously (before entering the event loop) to avoid
    # nested asyncio.run() calls.
    pre_loaded_fixture = _load_notes_fixture(fixture) if fixture is not None else None

    async def _go() -> None:
        if pre_loaded_fixture is not None:
            # Merge fixture notes into the singleton store.
            for rec in pre_loaded_fixture._records:
                await store.add_note(
                    note_id=rec.note_id,
                    text=rec.text,
                    embedding=rec.embedding.tolist(),
                )
        hits = await store.lookup(query, top_k=top_k)
        if not hits:
            console.print("[dim]no notes found[/]")
            return
        table = Table(title=f"Top {len(hits)} notes for {query!r}")
        table.add_column("note_id")
        table.add_column("score", justify="right")
        table.add_column("text")
        for hit in hits:
            table.add_row(hit.note_id, f"{hit.score:.4f}", hit.text)
        console.print(table)

    asyncio.run(_go())


@notes_app.command("clear")
def notes_clear() -> None:
    """Remove all notes from the in-memory notes store."""
    store = _get_notes_store()

    async def _go() -> None:
        await store.clear()

    asyncio.run(_go())
    console.print("[green]notes store cleared[/]")


# ---------------------------------------------------------------------------
# serve sub-command
# ---------------------------------------------------------------------------


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Enable uvicorn auto-reload."),
) -> None:
    """Start the FastAPI backend with uvicorn.

    Reads BACKEND_AUTH_TOKEN, BACKEND_DSN, LIVEKIT_API_KEY, and
    LIVEKIT_API_SECRET from the environment. Logs which features are
    active (auth, store mode, LiveKit).
    """
    import os

    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]uvicorn is not installed.[/] "
            "Install with: pip install 'voice-eval-lab[real]'"
        )
        raise typer.Exit(code=2) from None

    auth_enabled = bool(os.environ.get("BACKEND_AUTH_TOKEN"))
    backend_dsn = os.environ.get("BACKEND_DSN", "")
    lk_enabled = bool(
        os.environ.get("LIVEKIT_API_KEY") and os.environ.get("LIVEKIT_API_SECRET")
    )

    console.print("[bold]voice-eval serve[/] starting up")
    console.print(
        f"  auth      : {'enabled' if auth_enabled else '[yellow]DISABLED[/] (set BACKEND_AUTH_TOKEN)'}"
    )
    console.print(
        f"  store     : {'postgres (' + backend_dsn[:30] + '…)' if backend_dsn else 'in-memory'}"
    )
    console.print(
        f"  livekit   : {'enabled' if lk_enabled else '[yellow]gated[/] (set LIVEKIT_API_KEY + LIVEKIT_API_SECRET)'}"
    )

    uvicorn.run(
        "voice_eval_lab.backend.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


# ---------------------------------------------------------------------------
# pipeline sub-app
# ---------------------------------------------------------------------------


@pipeline_app.command("run")
def pipeline_run(
    turn_detector: str = typer.Option(
        "smart",
        "--turn-detector",
        help=(
            "Turn-detection mode: 'smart' (default) uses SmartTurnDetector with "
            "Pipecat SmartTurnAnalyzer when available, else energy-based silence "
            "fallback. 'none' disables turn detection (no-op stub)."
        ),
    ),
) -> None:
    """Drive the mock pipeline through Pipecat processors and print a transcript.

    Feeds a single synthetic audio chunk (representing the utterance
    "hnsw ef_search parameter") through the full STT → LLM → TTS chain and
    prints each agent Turn returned. Useful for verifying the processor
    wiring without a LiveKit room.
    """
    _validate_turn_detector_flag(turn_detector)

    from voice_eval_lab.models import Turn, TurnRole
    from voice_eval_lab.pipecat import run_pipeline
    from voice_eval_lab.pipecat.pipeline import build_pipeline
    from voice_eval_lab.pipecat.processors import AudioRawFrame
    from voice_eval_lab.pipeline import MockLLM, MockSTT, MockTTS

    pipeline = build_pipeline(
        stt=MockSTT(), llm=MockLLM(), tts=MockTTS(), turn_detector=turn_detector
    )

    # Patch the STTProcessor so it returns a canned utterance — the mock STT
    # requires a Turn with text, not raw audio bytes.
    stt_proc = pipeline.processors()[0]

    def _fake_turn(frame: AudioRawFrame) -> Turn:
        return Turn(
            role=TurnRole.USER,
            text="hnsw ef_search parameter",
            started_at_ms=0,
            ended_at_ms=200,
        )

    stt_proc._frame_to_turn = _fake_turn

    console.print("[bold]voice-eval pipeline run[/] (in-memory smoke test)")
    console.print("Audio source : synthetic 320-byte frame")
    console.print("Pipeline     : MockSTT → MockLLM → MockTTS (via Pipecat processors)\n")

    async def _go() -> None:
        async for turn in await run_pipeline(
            pipeline,
            audio_source=[b"\x00" * 320],
        ):
            console.print(f"[green]agent[/]: {turn.text}")

    asyncio.run(_go())
    console.print("\n[green]pipeline run complete[/]")


@pipeline_app.command("serve")
def pipeline_serve(
    room: str = typer.Option(..., "--room", help="LiveKit room name to connect to."),
    livekit_url: str = typer.Option(
        "",
        "--livekit-url",
        help=(
            "LiveKit server URL (e.g. wss://my-app.livekit.cloud). "
            "Falls back to LIVEKIT_URL env var."
        ),
    ),
    api_key: str = typer.Option(
        "",
        "--api-key",
        help="LiveKit API key. Falls back to LIVEKIT_API_KEY env var.",
    ),
    api_secret: str = typer.Option(
        "",
        "--api-secret",
        help="LiveKit API secret. Falls back to LIVEKIT_API_SECRET env var.",
    ),
) -> None:
    """Connect the mock Pipecat pipeline to a LiveKit room and serve it.

    Reads LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET from the
    environment (or from the explicit flags). When any credential is missing
    or when livekit-agents is not installed, logs a warning and exits cleanly
    — no exception is raised, making this safe to call in CI.
    """
    from voice_eval_lab.pipecat import make_pipecat_pipeline, serve_on_livekit

    pipeline = make_pipecat_pipeline()

    console.print(f"[bold]voice-eval pipeline serve[/] room={room!r}")

    serve_on_livekit(
        pipeline,
        room_name=room,
        livekit_url=livekit_url or None,
        api_key=api_key or None,
        api_secret=api_secret or None,
    )
    console.print("[dim]pipeline serve returned (credentials absent or SDK not installed)[/]")


# ---------------------------------------------------------------------------
# audio sub-app
# ---------------------------------------------------------------------------


@audio_app.command("populate-silence")
def audio_populate_silence(
    conv_id: str = typer.Option(..., "--conv-id", help="Conversation ID for the fixture."),
    turn: int = typer.Option(..., "--turn", help="Zero-based user-turn index."),
    duration_ms: int = typer.Option(500, "--duration-ms", help="Silence duration in ms."),
    root: Path = typer.Option(Path("evals/audio"), "--root", help="Audio fixture root directory."),
) -> None:
    """Generate a silence WAV fixture and write it to the fixture tree."""
    from voice_eval_lab.audio import FilesystemAudioStore, SilenceFixtureGenerator

    gen = SilenceFixtureGenerator()
    wav_bytes = gen.generate(duration_ms)
    store = FilesystemAudioStore(root)
    store.add_audio(conv_id, turn, wav_bytes)
    dest = root / conv_id / f"turn-{turn:02d}.wav"
    console.print(f"[green]wrote[/] {dest} ({len(wav_bytes)} bytes, {duration_ms}ms silence)")


@audio_app.command("import")
def audio_import(
    src: Path = typer.Argument(..., help="Source WAV file to import."),
    conv_id: str = typer.Option(..., "--conv-id", help="Conversation ID for the fixture."),
    turn: int = typer.Option(..., "--turn", help="Zero-based user-turn index."),
    root: Path = typer.Option(Path("evals/audio"), "--root", help="Audio fixture root directory."),
) -> None:
    """Copy a real WAV file into the audio fixture tree."""
    from voice_eval_lab.audio import FilesystemAudioStore

    if not src.exists():
        console.print(f"[red]source WAV not found:[/] {src}")
        raise typer.Exit(code=2) from None
    try:
        wav_bytes = src.read_bytes()
    except OSError as exc:
        console.print(f"[red]could not read {src}:[/] {exc}")
        raise typer.Exit(code=2) from None
    store = FilesystemAudioStore(root)
    try:
        store.add_audio(conv_id, turn, wav_bytes)
    except ValueError as exc:
        console.print(f"[red]invalid WAV file:[/] {exc}")
        raise typer.Exit(code=2) from None
    dest = root / conv_id / f"turn-{turn:02d}.wav"
    console.print(f"[green]imported[/] {src} -> {dest} ({len(wav_bytes)} bytes)")


@audio_app.command("list")
def audio_list(
    root: Path = typer.Option(Path("evals/audio"), "--root", help="Audio fixture root directory."),
) -> None:
    """List all audio fixture keys under the fixture root."""
    from voice_eval_lab.audio import FilesystemAudioStore

    store = FilesystemAudioStore(root)
    keys = store.list_keys()
    if not keys:
        console.print(f"[dim]no fixtures found under {root}[/]")
        return
    table = Table(title=f"Audio fixtures ({root})")
    table.add_column("conv_id")
    table.add_column("turn_index", justify="right")
    table.add_column("path")
    for conv_id, turn_index in keys:
        p = root / conv_id / f"turn-{turn_index:02d}.wav"
        table.add_row(conv_id, str(turn_index), str(p))
    console.print(table)
    console.print(f"[dim]{len(keys)} fixture(s)[/]")
