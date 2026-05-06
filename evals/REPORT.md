# Voice eval report

## Headline

| Metric | Value |
| --- | ---: |
| Conversations | 7 |
| Turn latency p50 / p95 / p99 (ms) | 275 / 275 / 275 |
| Transcription WER (corpus) | 3.19% |
| Response faithfulness (pooled) | 57.14% |
| Barge-in success (pooled) | 100.00% |
| False-trigger rate (pooled) | 0.00% |
| Barge-in yield p95 (ms) | 100 |
| TTS first-byte jitter (ms) | 0.0 |
| Endpointing accuracy (pooled) | 100.00% |
| LLM decisiveness (pooled) | 53.33% |

## Per conversation

| conv_id | topic | p95 ms | WER | faithfulness | barge-in | false-trigger | yield p95 | jitter | endpoint | decisive |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| postgres-replication | Postgres replication basics | 275 | 6.25% | 100.00% | 100.00% | 0.00% | 0 | 0.0 | 100.00% | 100.00% |
| hnsw-tuning | HNSW index tuning | 275 | 0.00% | 50.00% | 100.00% | 0.00% | 100 | 0.0 | 100.00% | 50.00% |
| prom-burn-rate | Prometheus SLO burn rate | 275 | 9.09% | 100.00% | 100.00% | 0.00% | 0 | 0.0 | 100.00% | 100.00% |
| empty-noise | False-trigger handling | 275 | 0.00% | 100.00% | 100.00% | 0.00% | 0 | 0.0 | 100.00% | 0.00% |
| agent-led-debug | Agent-led debugging session | 275 | 4.35% | 66.67% | 100.00% | 0.00% | 0 | 0.0 | 100.00% | 66.67% |
| noisy-vad | Noisy VAD endpointing | 275 | 0.00% | 33.33% | 100.00% | 0.00% | 0 | 0.0 | 100.00% | 33.33% |
| double-barge | Two consecutive barge-ins | 275 | 0.00% | 33.33% | 100.00% | 0.00% | 100 | 0.0 | 100.00% | 33.33% |
