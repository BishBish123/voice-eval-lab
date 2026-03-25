# ADR-002: jiwer for WER

Status: accepted
Date: 2025-12-08

## Context

Transcription word-error-rate is one of the five headline metrics. We
need a calculator that:

1. Handles edit distance correctly (Levenshtein over word tokens, not
   character-level).
2. Has a stable, well-documented public API.
3. Is maintained — speech ML moves fast and an abandoned dependency is
   a pin-and-pray situation.

## Decision

Use `jiwer.wer()` directly. The metric module's `transcription_wer`
function is a thin adapter that handles empty references (returns
0.0) and skips empty user turns before delegating.

## Consequences

Positive:

- Single dependency, BSD-licensed, used by HuggingFace's evaluate
  package — well-maintained.
- The WER formula is the formula in the literature; we don't need a
  bespoke implementation that drifts from the textbook.

Negative:

- `jiwer` is one more wheel in the install. Acceptable: it's pure
  Python with no C deps.

## Alternatives considered

- **Hand-rolled Levenshtein.** Rejected: easy to write, easier to
  break. Edge cases (empty sequences, non-ASCII tokenization) are
  battle-tested in jiwer.
- **HuggingFace `evaluate.load("wer")`.** Rejected: the evaluate
  package pulls in datasets, fsspec, pyarrow, etc. — far too heavy
  for a single function call.
- **`Levenshtein` (the C package).** Rejected: word-level WER, not
  character-level edit distance. Different metric.
