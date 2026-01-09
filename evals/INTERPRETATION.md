# Reading the eval report

Quick reference for what each metric measures and what good / bad
values look like in production. Numbers below are guidance from
public benchmarks; thresholds in the bundled `RegressionThresholds`
are conservative defaults, not law.

## Headline metrics

### Turn latency p50 / p95 / p99 (ms)

The gap between `vad_end` (the user stopped talking) and
`tts_first_byte` (the agent's audio started playing).

- **<700ms p95** is the floor for a natural conversation. Past that,
  users stop trusting the agent.
- **<400ms p95** is what the best public demos hit.
- The mock pipeline reports **275ms** because the stages are fixed
  — real numbers depend on the LLM model, network path, and TTS
  engine.
- **Full-completion latency**: the mock pipeline waits for each stage
  to complete before starting the next (STT → LLM → TTS in sequence).
  This measures the full round-trip, not streaming / time-to-first-token.
  Real streaming pipelines interleave LLM and TTS, so their p95 numbers
  will be lower than the equivalent non-streaming latency.

### Transcription WER (corpus-pooled)

Word-error rate from `jiwer` computed over the corpus-pooled
references and hypotheses (one big concatenation across every
measurable user turn in every conversation), so a long conversation
weighs proportionally more than a short one.

- **<5%** is competitive with Deepgram Nova on clean-room audio.
- **>10%** breaks downstream LLM faithfulness — the model can't
  ground if the question came in wrong.

### Response faithfulness (corpus-pooled)

Fraction of agent replies that contain at least one gold-fact
substring, computed as `sum(grounded_replies) / sum(user_turn_opportunities)`
across the corpus. Blank/whitespace-only user turns are excluded from
the denominator. Substring match is a proxy for an LLM judge — if you
swap in `LLMJudge.score(reply, gold_facts)` it should keep the same
signature and the same metric semantics.

- **>80%** target. Below 70% means the LLM is hallucinating answers
  that the gold-fact set could ground if the prompt was right.
- The bundled mock LLM produces **61.54%** because it's wired to
  surface only the first-matching gold fact, so multi-turn conversations
  in the golden set leave later turns ungrounded — see the
  per-conversation table for the breakdown.

### Barge-in success (corpus-pooled)

Of the user-interrupted turns, fraction the pipeline yielded inside
`barge_in_yield_ms` (default 100ms), computed as
`sum(yielded) / sum(interrupted_turns)` across the corpus. Binary:
did the agent stop talking when the user started?

- **100%** is the only acceptable production value — a barge-in
  failure is a UX bug, not a tuning knob.
- The mock pipeline reports 100% because it always yields.

### False-trigger rate (corpus-pooled)

Fraction of user-turn opportunities where the pipeline emitted a
false-trigger reply, computed as
`sum(false_trigger_turns) / sum(user_turn_count)` across the corpus.
The denominator is the user-turn count (not `len(turn_runs)`) so
injected synthetic false-trigger entries do not distort the rate.
Blank/whitespace-only turns are included in opportunities since they
are precisely the turns where false triggers are most likely.

- **<2%** target. Past that the agent feels jittery — it interjects
  on coughs and breath sounds.
- The bundled `--false-trigger-rate 1.0` flag forces this to 1.0 in
  one synthetic turn so the metric path is exercised.

## Diagnostic metrics (added in v0.2)

### Barge-in yield p95 (ms)

p95 of the `barge_in.yield` span duration across all interrupted
turns. Catches tail latencies that stay inside the binary budget.

- Headline value depends on the budget. With a 100ms budget,
  **<60ms p95** is the comfort zone; closer to the budget means
  you're one regression away from failing the binary metric.

### TTS first-byte jitter (ms)

Population standard deviation of first-byte latency across turns.

- **<30ms** feels steady to a listener.
- **>100ms** the audio start feels uneven turn-to-turn.

### Endpointing accuracy (corpus-pooled)

Fraction of user turns where the VAD `vad_end` span aligned with the
gold `ended_at_ms` (within `tolerance_ms`, default 50ms), computed as
`sum(aligned_turns) / sum(measured_turns_with_vad_end)` across the
corpus so long conversations weigh proportionally more.

- The mock pipeline always lines up exactly — the metric exists for
  real VAD systems where 100ms early or late is common.
- **>95%** target on a real pipeline.

### LLM decisiveness (corpus-pooled)

Fraction of agent replies that don't contain a hedging phrase ("I
don't have a confident answer", "maybe", "I'm not sure", ...), computed
as `sum(decisive_replies) / sum(user_turn_opportunities)` across the
corpus. Blank/whitespace-only user turns and false-trigger turns are
excluded from the denominator. Empty replies count as hedging.

- **>80%** in production. Hedging on questions the agent should
  answer is a calibration bug, not graceful uncertainty.
- The bundled mock LLM hits **57%** because the `agent-led-debug`
  conversation contains a turn ("idk") that forces the fallback,
  and several other conversations (`hnsw-tuning`, `empty-noise`,
  `noisy-vad`, `double-barge`) have at least one turn the mock
  can't ground, falling back to the `I don't have a confident
  answer about ...` reply.

## What to do when a metric regresses

1. Read the per-conversation table — the headline averages hide which
   conversation moved.
2. Run the eval at the previous commit (`git stash && make eval`) to
   confirm the diff is a real regression, not a refactor.
3. For latency / jitter regressions: bisect the pipeline stage by
   stage. The `PipelineSpan` records make this straightforward —
   each stage has a fixed name.
4. For WER / faithfulness / decisiveness regressions: add the failing
   conversation to a focused test, then iterate against the metric
   with the rest of the suite as a guardrail.
