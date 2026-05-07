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

## Aggregation across conversations

Two headline metrics are computed from a *pooled* sample list (every
turn, every conversation), not the mean of per-conversation scalars:

- `aggregate_barge_in_latency_p95_ms` — global p95 over all
  `barge_in.yield` durations.
- `aggregate_tts_first_byte_jitter_ms` — population stddev (`ddof=0`)
  over all `vad_end -> tts_first_byte` latencies.

Mean-of-aggregates folds zero-signal conversations into the headline
and underweights variance that crosses conversation boundaries — both
metrics are designed to surface variance, so we keep the raw sample
list and let one statistic answer the whole-run question. When no
signal exists at all (no yield spans, fewer than 2 latency samples)
the headline reports `None` rather than `0.0`.

## Pipecat integration (planned production path)

The `src/voice_eval_lab/pipecat/` package scaffolds the real-time delivery
layer. It wraps the three Protocols behind Pipecat `FrameProcessor`
subclasses and wires them into a `Pipeline` that can run against an
in-memory source for testing or against a LiveKit room in production.

### Processor mapping

| Protocol | Pipecat processor | Frame in | Frame out |
|---|---|---|---|
| `STT.transcribe(turn)` | `STTProcessor` | `AudioRawFrame` | `TextFrame` (user transcript) |
| `LLM.reply(history, text, facts)` | `LLMProcessor` | `TextFrame` (user) | `TextFrame` (agent reply) |
| `TTS.synthesize(text)` | `TTSProcessor` | `TextFrame` (agent) | `AudioRawFrame` chunks |

Non-owned frames are forwarded downstream unchanged. Exception handling is
local to each processor — a crashing adapter logs the error and either
emits nothing or forwards the original frame, so the pipeline never stalls.

### Two drivers

**In-memory (eval / CI)** — `run_pipeline(pipeline, audio_source=[...])` feeds
pre-segmented `AudioRawFrame` chunks through the `_ShimPipeline` and yields
agent `Turn` objects. No network, no audio device. Used by `voice-eval pipeline run`.

**LiveKit (production)** — `serve_on_livekit(pipeline, room_name=...)` connects
to a real LiveKit room via `livekit-agents` and serves the pipeline. Reads
`LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET` from the environment.
Soft-fails with a logged warning when credentials or the SDK are absent.
Used by `voice-eval pipeline serve --room <name>`.

### Shim vs. real Pipecat SDK

`pipecat-ai` is an optional `[real]` dependency. When absent (base install,
CI), `processors.py` defines minimal Python shims for `FrameProcessor`,
`Frame`, `TextFrame`, `AudioRawFrame`, and `FrameDirection` that implement
the same `process_frame` / `push_frame` contract. `pipeline.py` returns a
`_ShimPipeline` instead of `pipecat.pipeline.Pipeline`. Tests run against
the shim — no SDK install required.

### Turn-detector + barge-in

`_BargeInStub` remains a no-op placeholder. The turn-detector, however,
is now a real implementation: `SmartTurnDetector`
(`src/voice_eval_lab/pipecat/turn_detector.py`) is wired into the
`STTProcessor` by default (`turn_detector="smart"` in `build_pipeline`).

#### SmartTurn integration + fallback design

`SmartTurnDetector` is the bridge between Brief's Smart-Turn-V3 reference
and the Pipecat pipeline scaffold. Its design follows a strict soft-import
pattern:

1. **Pipecat path** — On import, `turn_detector.py` attempts:
   ```python
   from pipecat.audio.turn.smart_turn.smart_turn_analyzer import SmartTurnAnalyzer
   ```
   When this succeeds, `SmartTurnDetector.__init__` instantiates the
   analyzer (passing `eou_threshold`) and all `analyze()` calls delegate to
   `SmartTurnAnalyzer.analyze(chunk)`. The result is normalised to a
   `TurnState(is_end_of_turn, confidence)` named-tuple regardless of whether
   the analyzer returns a tuple, dict, or `TurnState` directly.

2. **Energy-based fallback** — When `pipecat-ai` is absent (base install,
   CI) or `SmartTurnAnalyzer` fails to initialise, the detector uses a
   stdlib-only fallback (`struct` + simple arithmetic):
   - Per-chunk mean-squared energy of 16-bit PCM samples is computed.
   - Energy below `_ENERGY_THRESHOLD` → chunk is silent; duration
     accumulates.
   - Energy at or above threshold → chunk is active; accumulator resets.
   - Accumulated silence ≥ `min_silence_ms` (default 500 ms) →
     `TurnState(is_end_of_turn=True, confidence=0.7)`.
   - Otherwise → `TurnState(is_end_of_turn=False, confidence=0.3)`.

3. **None mode** — `build_pipeline(turn_detector="none")` wires the legacy
   `_TurnDetectorStub` (no-op pass-through). Useful for pre-segmented audio
   or testing pipelines that supply complete utterances.

The `eou_threshold` parameter is forwarded to the Pipecat analyzer when
available; in fallback mode it is stored for introspection but does not
affect the decision rule (silence duration is the only signal). The
`min_silence_ms` parameter is only used by the fallback.

This dual-path design means the same pipeline code runs correctly in CI
(no pipecat-ai) and in production (with the ONNX SmartTurnAnalyzer), with
no conditional branches at call sites.

#### Barge-in

`_BargeInStub.cancel_tts()` remains a no-op. Real implementation will
monitor incoming `AudioRawFrame` energy while TTS is playing and cancel
`TTSProcessor` above a threshold. This is the remaining wiring gap between
the scaffold and a production LiveKit room.

### Existing `VoicePipeline` role

`VoicePipeline` in `pipeline.py` remains the eval-only test rig. It
operates on text-only `Conversation` objects and drives the deterministic
mock adapters to produce the `ConversationRun` records the metrics layer
consumes. It will not be replaced — it is the controlled surface the harness
scores against. The Pipecat integration is the production path; the eval rig
is a separate, stable scoring surface.

## Baseline + compare

Eval scores are a flat JSON document (`evals/scores.json`). The
`baseline` command writes one; `compare` reads one, runs a fresh
eval, and exits non-zero if any metric regressed past its threshold.
Threshold direction is per-metric: WER / latency / jitter /
false-trigger are "lower is better"; faithfulness / barge-in /
endpointing / decisiveness are "higher is better". The CLI accepts
overrides for the three most-tuned thresholds (latency, WER,
faithfulness); `RegressionThresholds` exposes the rest in code.
