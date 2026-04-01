# Architecture

`voice-eval-lab` ships two things that share a vocabulary:

1. A typed pipeline lifecycle (`VAD -> STT -> LLM -> TTS`) modelled
   behind three Protocols, with a deterministic mock implementation.
2. A scoring harness that consumes whatever spans the pipeline emits
   and produces a `EvalReport` independent of the real adapters.

The boundary between the two is the `ConversationRun` model
(`src/voice_eval_lab/models.py`). Anything that produces one — the mock
pipeline, a real LiveKit pipeline, a recorded trace replayer — can be
fed straight into the harness.

## Module layout

```
src/voice_eval_lab/
  models.py        Pydantic types — Turn / Conversation / PipelineSpan /
                   TurnRun / ConversationRun / ConversationScore /
                   EvalReport
  pipeline.py      MockSTT / MockLLM / MockTTS adapters, the
                   `VoicePipeline` runner, plus the v0.2 additions:
                   FlakyTTS, RetryingTTS, LatencyBudget
  baseline.py      Persist + diff scores between runs, with per-metric
                   regression thresholds
  cli.py           `voice-eval` Typer entrypoint — run, list, baseline,
                   compare, render
  eval/
    metrics.py     Per-turn metrics (latency, WER, faithfulness,
                   barge-in, false-trigger) + diagnostic metrics
                   (barge-in latency p95, jitter, endpointing, LLM
                   decisiveness) + render_report / render_report_html
    golden.py      7 hand-rolled golden conversations
```

## Why mock-first

Real audio infra (LiveKit room + Deepgram streaming + Cartesia TTS)
costs three accounts and an hour of yak-shaving to demo. The eval
harness is the part you actually iterate on — the metrics, the gold
set, the thresholds — and a deterministic pipeline lets that loop run
in <1s on a CI runner. Real adapters drop in behind the same
Protocols whenever the rig outgrows mocks.

ADR-001 documents this choice.

## Spans = OpenTelemetry-shaped

`PipelineSpan` is a flat record with `name`, `started_at_ms`,
`ended_at_ms`, and a string-keyed `attrs` dict. That maps 1:1 to the
OpenTelemetry span model the Phoenix exporter consumes. When we wire
Phoenix in, every existing span becomes a real OTel span without
touching the metric layer.

## Streaming LLM contract

`MockLLM.reply()` returns a fully-formed string for the eval harness;
`MockLLM.stream()` yields word-chunks via `AsyncIterator[str]`. Real
voice agents need streaming so TTS first-byte fires before the LLM
finishes. The mock exposes both surfaces so tests can assert the
streaming contract without wall-clock timing.

## Retry and budget surfaces

`RetryingTTS` wraps any TTS adapter with exponential-backoff retries
that emit `tts.retry` spans. `LatencyBudget` is a post-pass middleware
that scans the run for `vad_end -> tts_first_byte` deltas above a
configurable budget and emits `latency_budget.exceeded` spans. Both
are designed for real adapters where transient failures and tail
latencies are the operational reality, not the mock pipeline.

## Baseline + compare

Eval scores are a flat JSON document (`evals/scores.json`). The
`baseline` command writes one; `compare` reads one, runs a fresh
eval, and exits non-zero if any metric regressed past its threshold.
Threshold direction is per-metric: WER / latency / jitter /
false-trigger are "lower is better"; faithfulness / barge-in /
endpointing / decisiveness are "higher is better". The CLI accepts
overrides for the three most-tuned thresholds (latency, WER,
faithfulness); `RegressionThresholds` exposes the rest in code.
