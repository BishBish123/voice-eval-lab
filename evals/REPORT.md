# Voice eval report

## Headline

| Metric | Value |
| --- | ---: |
| Conversations | 25 |
| Turn latency p50 / p95 / p99 (ms) | 278 / 478 / 629 |
| Transcription WER (corpus-pooled) | 0.00% |
| Response faithfulness (corpus-pooled) | 68.00% |
| Barge-in success (corpus-pooled) | 100.00% |
| False-trigger rate (corpus-pooled) | 0.00% |
| Barge-in yield p95 (ms) | 100 |
| TTS first-byte jitter (ms) | 94.4 |
| Endpointing accuracy (corpus-pooled) | 100.00% |
| LLM decisiveness (corpus-pooled) | 62.26% |

> Faithfulness defaults to the substring proxy (no API keys required). Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` and run `voice-eval run --judge llm` to score with the LLM-as-judge — see `evals/INTERPRETATION.md#response-faithfulness-corpus-pooled`.

## Per conversation

| conv_id | topic | p95 ms | WER | faithfulness | barge-in | false-trigger | yield p95 | jitter | endpoint | decisive |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| postgres-replication | Postgres replication basics | 280 | 0.00% | 100.00% | 100.00% | 0.00% | 0 | 22.5 | 100.00% | 100.00% |
| hnsw-tuning | HNSW index tuning | 280 | 0.00% | 50.00% | 100.00% | 0.00% | 100 | 25.0 | 100.00% | 50.00% |
| prom-burn-rate | Prometheus SLO burn rate | 470 | 0.00% | 100.00% | 100.00% | 0.00% | 0 | 0.0 | 100.00% | 100.00% |
| empty-noise | False-trigger handling | 279 | 0.00% | 100.00% | 100.00% | 0.00% | 0 | 0.0 | 100.00% | 0.00% |
| agent-led-debug | Agent-led debugging session | 327 | 0.00% | 66.67% | 100.00% | 0.00% | 0 | 19.8 | 100.00% | 66.67% |
| noisy-vad | Noisy VAD endpointing | 259 | 0.00% | 50.00% | 100.00% | 0.00% | 0 | 14.5 | 100.00% | 50.00% |
| double-barge | Two consecutive barge-ins | 316 | 0.00% | 33.33% | 100.00% | 0.00% | 100 | 30.7 | 100.00% | 33.33% |
| tcp-handshake | TCP three-way handshake | 248 | 0.00% | 100.00% | 100.00% | 0.00% | 0 | 28.0 | 100.00% | 100.00% |
| s3-consistency | S3 strong read-after-write consistency | 300 | 0.00% | 100.00% | 100.00% | 0.00% | 0 | 46.0 | 100.00% | 50.00% |
| react-reconciler | React reconciler and fiber diffing | 425 | 0.00% | 100.00% | 100.00% | 0.00% | 0 | 72.5 | 100.00% | 100.00% |
| redis-eviction | Redis maxmemory eviction policies | 489 | 0.00% | 50.00% | 100.00% | 0.00% | 0 | 28.5 | 100.00% | 50.00% |
| mid-sentence-barge | Mid-sentence barge-in during explanation | 582 | 0.00% | 100.00% | 100.00% | 0.00% | 100 | 170.1 | 100.00% | 100.00% |
| early-barge-cloud | Early barge-in before agent finishes intro | 343 | 0.00% | 100.00% | 100.00% | 0.00% | 100 | 52.0 | 100.00% | 100.00% |
| triple-barge | Three consecutive barge-ins | 330 | 0.00% | 100.00% | 100.00% | 0.00% | 100 | 57.8 | 100.00% | 100.00% |
| ambig-cache | Ambiguous cache reference | 233 | 0.00% | 100.00% | 100.00% | 0.00% | 0 | 6.0 | 100.00% | 100.00% |
| ambig-deploy | Ambiguous deployment target | 306 | 0.00% | 50.00% | 100.00% | 0.00% | 0 | 47.5 | 100.00% | 50.00% |
| ambig-index | Ambiguous slow index without table name | 256 | 0.00% | 50.00% | 100.00% | 0.00% | 0 | 1.0 | 100.00% | 50.00% |
| oos-weather | Out-of-scope weather query mid-call | 244 | 0.00% | 100.00% | 100.00% | 0.00% | 0 | 0.0 | 100.00% | 0.00% |
| oos-stock-price | Out-of-scope stock price query | 273 | 0.00% | 100.00% | 100.00% | 0.00% | 0 | 0.0 | 100.00% | 0.00% |
| k8s-networking-deep | Kubernetes pod networking deep-dive | 190 | 0.00% | 100.00% | 100.00% | 0.00% | 0 | 0.0 | 100.00% | 100.00% |
| ssl-tls-handshake | TLS 1.3 handshake walkthrough | 345 | 0.00% | 100.00% | 100.00% | 0.00% | 0 | 0.0 | 100.00% | 100.00% |
| rapid-fire-git | Rapid-fire git command Q&amp;A | 392 | 0.00% | 0.00% | 100.00% | 0.00% | 0 | 53.0 | 100.00% | 0.00% |
| rapid-fire-regex | Rapid-fire regex clarifications | 444 | 0.00% | 50.00% | 100.00% | 0.00% | 0 | 73.8 | 100.00% | 50.00% |
| clarify-oom | Agent asks for OOM context before answering | 622 | 0.00% | 50.00% | 100.00% | 0.00% | 0 | 202.0 | 100.00% | 50.00% |
| clarify-latency | Agent asks which layer is slow before diagnosing | 327 | 0.00% | 50.00% | 100.00% | 0.00% | 0 | 17.5 | 100.00% | 50.00% |
