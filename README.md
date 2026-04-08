# voice-eval-lab

> A voice-agent eval harness — turn latency, transcription WER, response faithfulness, barge-in success, false-trigger rate (plus four diagnostic metrics) — over a deterministic reference pipeline (mock STT/LLM/TTS) so the suite runs end-to-end with no audio infra.

[![ci](https://github.com/BishBish123/voice-eval-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/BishBish123/voice-eval-lab/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What this is (and what it isn't)

The 2026 voice-agent moat isn't shipping a demo — it's *measuring*
one. The brief calls it out explicitly: voice is the hot demo, the
eval pipeline is the moat. So this repo ships **the eval pipeline
first**, with a deterministic reference Pipecat-style runner that
exercises the harness end-to-end.

What's here:

- A typed pipeline lifecycle (`VAD -> STT -> LLM -> TTS`) that any
  real Pipecat / LiveKit pipeline maps onto cleanly.
- 5 headline metrics + 4 diagnostic metrics, each with explicit
  definitions and tests:
  - **Turn latency** (p50 / p95 / p99) — gap between `vad_end` and
    `tts_first_byte`
  - **Transcription WER** — `jiwer` over (gold transcript, pipeline
    transcript)
  - **Response faithfulness** — fraction of agent replies grounded in
    gold facts
  - **Barge-in success rate** — fraction of user-interrupted turns
    the pipeline yielded inside `barge_in_yield_ms`
  - **False-trigger rate** — fraction of turns where the agent
    replied to a non-utterance
  - **Barge-in yield p95** — distribution inside the budget
  - **TTS first-byte jitter** — std-dev of first-byte across turns
  - **Endpointing accuracy** — VAD-end alignment with the gold
    utterance end
  - **LLM decisiveness** — fraction of replies that don't hedge
- Pipeline middleware: `RetryingTTS` decorator + `LatencyBudget`
  that flags turns past a wall-clock ceiling.
- A bundled 7-conversation golden set covering every metric path.
- A `voice-eval` CLI with five subcommands: `run`, `list`,
  `baseline`, `compare`, `render`.

What's *not* here yet:

- Real LiveKit / Deepgram / Cartesia adapters. They drop in behind
  the same `STT` / `LLM` / `TTS` Protocols. The repo's value is the
  harness around them.
- An LLM-judge faithfulness scorer. The current proxy is "did the
  reply quote any gold fact verbatim?". The same
  `response_faithfulness` function name + signature would be used by
  an LLM judge.
- Phoenix tracing wired in. Every `PipelineSpan` already has the
  OpenTelemetry shape — adding a Phoenix exporter is a one-file
  change.

## Headline numbers (run locally)

```bash
make install
uv run voice-eval run --wer-substitution-rate 0.1
```

Produces (with the bundled mock pipeline + golden set):

| Metric | Value |
| --- | ---: |
| Conversations | 7 |
| Turn latency p50 / p95 / p99 (ms) | 275 / 275 / 275 |
| Transcription WER (mean) | 2.81% |
| Response faithfulness (mean) | 71.43% |
| Barge-in success (mean) | 100.00% |
| False-trigger rate (mean) | 0.00% |
| Barge-in yield p95 (ms) | 29 |
| TTS first-byte jitter (ms) | 0.0 |
| Endpointing accuracy (mean) | 100.00% |
| LLM decisiveness (mean) | 57.14% |

Real numbers come from real adapters. The harness will produce the
same shape.

`evals/INTERPRETATION.md` is the cheat-sheet for what each value
means and what good / bad looks like in production.

## Architecture

```
                +------------+
   user mic --> |  Pipeline  | -- VAD -> STT -> LLM -> TTS, each
                |  runner    |    stage emits a PipelineSpan
                +-----+------+
                      v
              +-----------------+
              | ConversationRun | <- per-turn transcript + reply +
              +--------+--------+    ordered spans
                       v
              +-----------------+
              |  Eval harness   | <- 5 headline + 4 diagnostic metrics
              +--------+--------+
                       v
                 EvalReport (markdown / HTML / JSON)
                       |
              optional v
              +-----------------+
              |  Phoenix / OTel | <- spans already in the right shape;
              +-----------------+    exporter swap, not a rewrite
```

Deeper dive: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
ADRs: [`docs/adr/`](docs/adr).

## Quick start

```bash
git clone https://github.com/BishBish123/voice-eval-lab.git
cd voice-eval-lab
make install

# Run the harness on the bundled golden set with the bundled mock pipeline.
uv run voice-eval run --out evals/REPORT.md --json evals/scores.json

# Inject WER to exercise the metric.
uv run voice-eval run --wer-substitution-rate 0.2

# Force a synthetic false-trigger turn.
uv run voice-eval run --false-trigger-rate 1.0

# See the bundled golden set.
uv run voice-eval list

# Persist a baseline + diff against it on the next run.
uv run voice-eval baseline --save evals/baseline.json
uv run voice-eval compare --baseline evals/baseline.json

# Re-render the latest scores as HTML.
uv run voice-eval render --from evals/scores.json --out evals/REPORT.html --format html
```

## Plug in real adapters

Implement the `STT` / `LLM` / `TTS` Protocols in `pipeline.py`:

```python
class DeepgramSTT:
    async def transcribe(self, turn: Turn) -> tuple[str, list[PipelineSpan]]:
        # call Deepgram streaming API; emit a PipelineSpan per chunk
        ...
```

Then construct the pipeline with the real adapters:

```python
pipeline = VoicePipeline(
    stt=DeepgramSTT(...),
    llm=GroqLLM(...),
    tts=RetryingTTS(inner=CartesiaTTS(...)),  # bounded retries
    latency_budget=LatencyBudget(budget_ms=700),
)
```

The harness doesn't care which adapters you use — it scores whatever
spans the pipeline emits.

## Tests

```bash
make test       # 130+ unit tests
make check      # ruff + mypy --strict
```

What's covered:

- **Pipeline contract** — STT word substitution (global + per-turn),
  span shape, false-trigger injection, gold-fact surfacing, streaming
  LLM, RetryingTTS retry / backoff / max-attempts, LatencyBudget
  flagging, double-barge-in, determinism.
- **Metrics** — perfect / one-sub / empty-hyp / empty-ref WER,
  faithfulness with / without gold facts / case-insensitive /
  partial-substring, barge-in success vs. no-yield-span, false-trigger
  proportionality, percentile p50/p95/p99 ordering, jitter
  shift-invariance, endpointing tolerance, decisiveness hedging.
- **Baseline / compare** — round-trip serialisation + per-metric
  regression detection in both directions, custom thresholds.
- **Renderers** — markdown headline + per-conversation tables, HTML
  variant with inline CSS, no `None` leaks.
- **CLI** — `run`, `list`, `baseline`, `compare`, `render` via
  `typer.testing.CliRunner`, including the regression exit code.

## Layout

```
src/voice_eval_lab/
  models.py        Pydantic types — Turn, Conversation, PipelineSpan,
                   TurnRun, ConversationRun, ConversationScore,
                   EvalReport
  pipeline.py      Mock STT/LLM/TTS, RetryingTTS, LatencyBudget,
                   VoicePipeline runner
  baseline.py      Persist + diff scores between runs
  eval/
    metrics.py     Headline + diagnostic metric functions, markdown +
                   HTML renderers
    golden.py      7-conversation bundled golden set
  cli.py           `voice-eval` Typer entrypoint

tests/             130+ unit tests
evals/             REPORT.md + scores.json + INTERPRETATION.md
docs/              ARCHITECTURE.md + ADRs
```

## Honest limitations

- The mock pipeline is deterministic — it can't surface real-world
  failure modes (network jitter, partial transcripts, model
  hallucinations). Real adapters get a fault-injection layer; the
  bundled `FlakyTTS` is a starter.
- The faithfulness metric is substring-match. An LLM-as-judge with
  documented Cohen's-kappa calibration is the production path.
- The bundled golden set is 7 hand-written conversations. Real
  evaluation uses 30-50 with audio recordings; that requires a
  separate corpus repo.
- No Phoenix trace exporter wired yet — the `PipelineSpan` model
  already maps to OpenTelemetry shape, so adding one is small.
- This isn't the voice agent itself; it's the rig you'd score one
  with. Pair it with [Pipecat](https://github.com/pipecat-ai/pipecat)
  + [LiveKit Agents](https://github.com/livekit/agents) for the
  realtime side.

## License

MIT. See [LICENSE](LICENSE).
