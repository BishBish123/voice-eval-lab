# ADR-001: Mock-first pipeline

Status: accepted
Date: 2025-12-12

## Context

The repository ships an eval harness for real-time voice agents. Real
voice infra needs three SaaS accounts (a streaming STT, an LLM API,
a streaming TTS) and a session orchestrator (LiveKit Agents or
Pipecat). Even with all four, end-to-end tests cost wall-clock seconds
each and burn API quota — neither is acceptable for a tight inner
loop.

## Decision

The reference pipeline is a deterministic, single-process
implementation that emits the same `PipelineSpan` shape a real
pipeline would. STT, LLM, and TTS are Protocols; the mock adapters
(`MockSTT`, `MockLLM`, `MockTTS`) and any real adapter (`DeepgramSTT`,
`GroqLLM`, `CartesiaTTS`) implement the same surface.

Tests run against the mock; production uses the real adapters; the
harness scores both identically.

## Consequences

Positive:

- The full eval suite runs in under a second with no external services.
- A regression in metric logic surfaces in pytest, not as a flaky
  cloud-API integration test.
- New metrics can be developed against synthetic spans before the
  real pipeline emits them.

Negative:

- The mock pipeline cannot surface failure modes that only emerge
  under real network jitter / streaming back-pressure / partial
  transcripts. We accept this — the harness is the moat, not the
  pipeline; real adapters get a separate fault-injection layer.
- Anyone reading the README must understand that the bundled numbers
  are mock-pipeline numbers, not production. The README calls this
  out explicitly.

## Alternatives considered

- **Replay recorded fixtures.** Rejected: still requires recording
  infra and licensing for the audio. Synthetic conversations dodge
  both.
- **Skip the pipeline; just ship the metrics.** Rejected: without an
  end-to-end runner, the harness can't be exercised. The mock
  pipeline is the cheapest way to keep the metrics honest.
