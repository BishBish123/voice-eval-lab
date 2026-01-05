# voice-eval-lab

> A 5-axis eval harness for real-time voice agents — turn latency, transcription WER, response faithfulness, barge-in success, false-trigger rate — plus a reference Pipecat-style pipeline (mock STT/LLM/TTS) so the suite runs end-to-end with no audio infra.

[![ci](https://github.com/BishBish123/voice-eval-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/BishBish123/voice-eval-lab/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What this is (and what it isn't)

The 2026 voice-agent moat isn't shipping a demo — it's *measuring* one. The brief calls this out explicitly: "voice is the hot demo; the eval pipeline is the moat." So this repo ships **the eval pipeline first**, with a deterministic reference Pipecat-style runner that exercises the harness end-to-end.

What's here:

- A typed pipeline lifecycle (`VAD → STT → LLM → TTS`) that any real Pipecat / LiveKit pipeline maps onto cleanly.
- 5 metrics with explicit definitions + tests:
  - **Turn latency** (p50 / p95 / p99) — the gap between `vad_end` and `tts_first_byte`
  - **Transcription WER** — `jiwer` over (gold transcript, pipeline transcript)
  - **Response faithfulness** — fraction of agent replies grounded in gold facts
  - **Barge-in success rate** — fraction of user-interrupted turns the pipeline yielded inside `barge_in_yield_ms`
  - **False-trigger rate** — fraction of turns where the agent replied to a non-utterance
- A bundled 4-conversation golden set covering the interesting metric paths.
- A `voice-eval` CLI that runs the pipeline + writes a markdown + JSON report.

What's *not* here yet:

- Real LiveKit / Deepgram / Cartesia adapters. They drop in behind the same `STT` / `LLM` / `TTS` Protocols. The repo's value is the harness around them.
- An LLM-judge faithfulness scorer. The current proxy is "did the reply quote any gold fact verbatim?". The same `response_faithfulness` function name + signature would be used by an LLM judge.
- Phoenix tracing wired in. Every `PipelineSpan` already has the OpenTelemetry shape — adding a Phoenix exporter is a one-file change.

## Headline numbers (run locally)

```bash
make install
uv run voice-eval --wer-substitution-rate 0.1
```

Produces (with the bundled mock pipeline + golden set):

| Metric | Value |
| --- | ---: |
| Conversations | 4 |
| Turn latency p50 / p95 / p99 (ms) | 275 / 275 / 275 |
| Transcription WER (mean) | 3.8% |
| Response faithfulness (mean) | 75% |
| Barge-in success (mean) | 100% |
| False-trigger rate (mean) | 0% |

Real numbers come from real adapters. The harness will produce the same shape.

## Architecture

```
                ┌────────────┐
   user mic ──► │  Pipeline  │ — VAD → STT → LLM → TTS pipeline,
                │  runner    │   each stage emits a PipelineSpan
                └─────┬──────┘
                      ▼
              ┌─────────────────┐
              │  ConversationRun │ ← per-turn transcript + reply +
              └────────┬─────────┘   ordered spans
                       ▼
              ┌─────────────────┐
              │  Eval harness   │ ← turn latency, WER, faithfulness,
              └────────┬─────────┘   barge-in, false-trigger
                       ▼
                 EvalReport (markdown + JSON)
                       │
              optional ▼
              ┌─────────────────┐
              │  Phoenix / OTel │ ← spans already in the right shape;
              └─────────────────┘   exporter swap, not a rewrite
```

## Quick start

```bash
git clone https://github.com/BishBish123/voice-eval-lab.git
cd voice-eval-lab
make install

# Run the harness on the bundled golden set with the bundled mock pipeline.
uv run voice-eval --out evals/REPORT.md --json evals/scores.json

# Inject WER to exercise the metric.
uv run voice-eval --wer-substitution-rate 0.2

# Force a synthetic false-trigger turn.
uv run voice-eval --false-trigger-rate 1.0
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
pipeline = VoicePipeline(stt=DeepgramSTT(...), llm=GroqLLM(...), tts=CartesiaTTS(...))
```

The `eval` harness doesn't care which adapters you use — it scores whatever spans the pipeline emits.

## Tests

```bash
make test       # 20 unit tests
make check      # ruff + mypy --strict
```

What's covered:

- **Pipeline contract** — STT word substitution, span shape, false-trigger injection, gold-fact surfacing by the mock LLM.
- **Metrics** — perfect / one-substitution WER, faithfulness with and without gold facts, barge-in success vs. no-yield-span, false-trigger proportionality, percentile stats over an empty set.
- **End-to-end** — `score_run` aggregates per-conversation scores correctly.

## Layout

```
src/voice_eval_lab/
  models.py        Pydantic types — Turn, Conversation, PipelineSpan,
                   TurnRun, ConversationRun, ConversationScore, EvalReport
  pipeline.py      Mock STT/LLM/TTS + VoicePipeline runner
  eval/
    metrics.py     5 metric functions + render_report()
    golden.py      4-conversation bundled golden set
  cli.py           `voice-eval` Typer entrypoint

tests/             20 unit tests
evals/             REPORT.md + scores.json (committed by `make eval`)
```

## Honest limitations

- The mock pipeline is deterministic — it can't surface real-world failure modes (network jitter, partial transcripts, model hallucinations). A future commit will ship a fault-injection layer over the real adapters.
- The faithfulness metric is substring-match. An LLM-as-judge with documented Cohen's-kappa calibration is the production path.
- The bundled golden set is 4 hand-written conversations. Real evaluation uses 30-50 with audio recordings; that requires a separate corpus repo.
- No Phoenix trace exporter wired yet — the `PipelineSpan` model already maps to OpenTelemetry shape, so adding one is small.
- This isn't the voice agent itself; it's the rig you'd score one with. Pair it with [Pipecat](https://github.com/pipecat-ai/pipecat) + [LiveKit Agents](https://github.com/livekit/agents) for the realtime side.

## License

MIT. See [LICENSE](LICENSE).
