# Voice eval report

## Headline

| Metric | Value |
| --- | ---: |
| Conversations | 4 |
| Turn latency p50 / p95 / p99 (ms) | 275 / 275 / 275 |
| Transcription WER (mean) | 3.84% |
| Response faithfulness (mean) | 75.00% |
| Barge-in success (mean) | 100.00% |
| False-trigger rate (mean) | 0.00% |

## Per conversation

| conv_id | topic | p95 ms | WER | faithfulness | barge-in | false-trigger |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| postgres-replication | Postgres replication basics | 275 | 6.25% | 100.00% | 100.00% | 0.00% |
| hnsw-tuning | HNSW index tuning | 275 | 0.00% | 0.00% | 100.00% | 0.00% |
| prom-burn-rate | Prometheus SLO burn rate | 275 | 9.09% | 100.00% | 100.00% | 0.00% |
| empty-noise | False-trigger handling | 275 | 0.00% | 100.00% | 100.00% | 0.00% |
