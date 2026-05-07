# Voice-Eval-Lab — resume bullets

## Defensible (use these)

- **Built a voice-pipeline eval harness** measuring 10 metrics across a 25-conversation golden set: turn latency p50/p95/p99, transcription WER, response faithfulness, barge-in success, false-trigger rate, barge-in yield p95, TTS first-byte jitter, endpointing accuracy, and LLM decisiveness — all computed from a Protocol-backed mock pipeline with no real audio infra required; 626 tests collected.

  *Evidence:* `evals/REPORT.md` headline table (25 conversations); `pytest --collect-only -q` → "626 tests collected"; `src/voice_eval_lab/eval/metrics.py` implements all 10 metrics; `docs/ARCHITECTURE.md` describes the Protocol design.

- **Measured deterministic-jitter pipeline latencies on 25 conversations**: turn latency p50 278 ms / p95 478 ms / p99 629 ms; barge-in success 100% corpus-pooled; false-trigger rate 0.0%; endpointing accuracy 100%; barge-in yield p95 100 ms; TTS first-byte jitter 94.4 ms population stddev.

  *Evidence:* `evals/REPORT.md` headline table, all values committed 2026-05-06; `evals/INTERPRETATION.md` explains the mock pipeline produces fixed stage durations (deterministic, not wall-clock inference).

- **Implemented corpus-pooled aggregation** for barge-in success and WER (pooled sample list across all turns, not mean-of-means), preventing long conversations from being under-weighted — with a parallel `EvalReport` that tracks both pooled and per-conversation scores; regression thresholds enforced against a baseline via `baseline.py`.

  *Evidence:* `src/voice_eval_lab/eval/metrics.py` (corpus-pooled computation); `src/voice_eval_lab/baseline.py` (regression thresholds + diff); `docs/ARCHITECTURE.md` "Aggregation across conversations" section.

- **Designed a 7-failure-mode golden set** (happy-path Q&A, barge-in, ambiguous input, out-of-scope, long answer, rapid back-and-forth, clarifying question) covering 25 hand-curated conversations across 12 technical topics (Postgres, Kubernetes, Redis, TLS, React reconciler, Prometheus SLO, git).

  *Evidence:* `evals/REPORT.md` per-conversation table (25 rows); `evals/INTERPRETATION.md` golden-set roadmap section; per-conversation topic column.

---

## Stretch / claim with caveat (use cautiously)

- **"Pluggable real adapters for Groq, Deepgram, Cartesia, ElevenLabs, Whisper"** — adapter stubs exist in `tests/test_real_adapter_integration.py`, `tests/test_groq_role_and_phoenix.py`, `tests/test_elevenlabs_adapter.py`, `tests/test_whisper_adapter.py`. What is not defensible: measured latency numbers from those real adapters — all REPORT.md numbers come from the deterministic mock pipeline.

  *Pushback:* "What are Groq + Deepgram + Cartesia p95 latencies?" — honest answer: "The REPORT.md numbers are mock-pipeline (fixed stage durations). The adapter code exists and is tested against mock servers; real-adapter bench numbers have not been committed."

- **"OpenTelemetry-shaped spans feeding Phoenix"** — `PipelineSpan` maps 1:1 to the OTel span model (`name`, `started_at_ms`, `ended_at_ms`, `attrs`). `tests/test_groq_role_and_phoenix.py` exercises this path. Not defensible: a live Phoenix trace dashboard URL.

---

## DO NOT claim

- **"Sub-500ms p95 voice agent latency on Groq + Deepgram + Cartesia"** — the REPORT.md p95 of 478 ms is from the deterministic mock pipeline, not real adapters. `evals/INTERPRETATION.md` explicitly states: "The mock pipeline reports 275ms because the stages are fixed — real numbers depend on the LLM model, network path, and TTS engine."

  *Alternate:* "Mock-pipeline p95 478 ms on 25 conversations; architecture is built for real adapters (Groq, Deepgram, Cartesia protocols exist) but real-adapter latencies have not been benchmarked."

- **"50-conversation golden eval"** — the set has 25 conversations. `evals/INTERPRETATION.md` "Golden-set roadmap" explicitly says: "The current set has 25 (as of the v0.3 expansion)... The path to 50 is curating 25 more."

  *Alternate:* "25-conversation golden set across 7 failure-mode categories."

- **"0% WER on real audio"** — the 0.00% WER in REPORT.md is from the mock STT which returns the gold transcript verbatim. Real WER requires a real STT adapter and real audio.

  *Alternate:* "WER metric is implemented and wired; mock pipeline baseline is 0.00% (deterministic); real-adapter WER is measured separately when a real STT is plugged in."

- **"Real barge-in handling on a live call"** — the 100% barge-in success is mock pipeline behavior (always yields); tests in `test_pipeline.py` verify the mock yields, not a real VAD/audio stack.

---

## How to defend each bullet in an interview

**Bullet 1 — 10 metrics, 25 conversations, 626 tests:**
> "The 25 conversations are in `evals/REPORT.md` — each row is a named conversation (e.g., 'postgres-replication', 'mid-sentence-barge'). The 10 metrics are implemented in `src/voice_eval_lab/eval/metrics.py`. The 626-test count: `pytest --collect-only -q` on the repo. The key design point is Protocol-backed — MockSTT, MockLLM, MockTTS in `pipeline.py` implement the same Protocol as real adapters, so tests run without LiveKit or Deepgram accounts."

**Bullet 2 — p50 278ms / p95 478ms / p99 629ms:**
> "These are committed in `evals/REPORT.md`. The mock pipeline uses fixed stage durations — they're deterministic, not wall-clock inference. The INTERPRETATION doc explains this: 'The mock pipeline reports 275ms because the stages are fixed.' So these numbers validate the metric computation and the golden set, not real production latency. For real latency, the Groq + Deepgram adapters need to be plugged in and a fresh eval run."

**Bullet 3 — corpus-pooled aggregation:**
> "The barge-in and WER metrics are pooled: one big list of all turns across all 25 conversations, then p95 / pooled fraction. Not mean-of-per-conversation-means. That matters because a 10-turn conversation contributes 10 barge-in opportunities to the denominator, not a weight of 1.0. The code is in `metrics.py` under `aggregate_barge_in_success_rate`."

**Bullet 4 — 7 failure modes, 25 conversations:**
> "The `evals/INTERPRETATION.md` golden-set roadmap section names the 7 categories. The per-conversation table in REPORT.md shows examples: 'double-barge' tests consecutive barge-ins, 'oos-weather' tests out-of-scope refusal, 'clarify-oom' tests the clarifying-question path. The topics span Postgres, TLS, Kubernetes, Redis, React reconciler — not toy examples."
