# ADR-003: Binary vs. latency barge-in

Status: accepted
Date: 2026-01-22

## Context

The v0.1 harness shipped `barge_in_success_rate` as a binary metric:
of the user-interrupted turns, how many did the pipeline yield to
inside `barge_in_yield_ms` (default 100ms)? The metric correctly
distinguishes "yielded" from "didn't yield" but it hides everything
that happens *inside* the budget. A regression from 30ms p95 to 95ms
p95 — an obvious tail-latency regression — leaves the binary metric
at 100%.

## Decision

Keep the binary `barge_in_success_rate` for the headline (a 9x worse
yield latency that still hits the budget is genuinely fine for the
user) and add `barge_in_latency_p95_ms` as a diagnostic metric.

`VoicePipeline` now emits a `barge_in.yield` span whenever a user
turn is `interrupted=True`. The duration of that span is the per-turn
yield latency; the metric takes a p95 across all such spans.

## Consequences

Positive:

- The headline answer to "did barge-in work?" stays binary and simple.
- Tuning conversations now see latency tail regressions early —
  the binary metric won't catch a 90th-percentile regression that
  stays under budget.
- Reporting both gives a useful "is it fast or slow inside the
  budget?" signal that complements the headline.

Negative:

- One more metric in the report. The HTML / markdown renderers gain
  a column.
- The `barge_in_latency_p95` metric returns 0.0 when no interrupted
  turns exist, which has to be documented (in `INTERPRETATION.md`)
  to avoid being misread as "regression".

## Alternatives considered

- **Replace the binary metric with the p95.** Rejected: the binary
  metric is the right answer for "ship / no-ship" decisions. p95 is
  a tuning signal, not an acceptance test.
- **Histogram bucket counts.** Considered for v0.3. The diagnostic
  set could grow to expose explicit buckets (under-budget /
  near-budget / over-budget) without changing the headline shape.
