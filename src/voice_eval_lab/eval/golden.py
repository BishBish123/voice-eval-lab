"""Bundled golden conversation set — synthetic transcripts with known facts.

Synthetic on purpose: real-conversation corpora have licensing problems
and require audio storage we don't want to commit. The transcripts are
crafted so the metrics light up in interesting ways:

- one conversation has a barge-in turn
- one has the false-trigger flag exercised by the pipeline
- one has multi-turn history so the LLM input grows
- one has gold facts that the mock LLM is wired to surface
"""

from __future__ import annotations

from voice_eval_lab.models import Conversation, Turn, TurnRole


def default_golden_set() -> list[Conversation]:
    return [
        Conversation(
            conv_id="postgres-replication",
            topic="Postgres replication basics",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="quiz me on postgres replication",
                    started_at_ms=0,
                    ended_at_ms=1500,
                ),
                Turn(
                    role=TurnRole.AGENT,
                    text="What is the difference between physical and logical replication?",
                    started_at_ms=1700,
                    ended_at_ms=4500,
                ),
                Turn(
                    role=TurnRole.USER,
                    text="physical replicates the WAL byte by byte logical replicates by row",
                    started_at_ms=4800,
                    ended_at_ms=8500,
                ),
            ],
            gold_facts=[
                "Physical replication ships WAL bytes; logical replication ships row-level changes.",
                "Logical replication can replicate a subset of tables; physical cannot.",
            ],
        ),
        Conversation(
            conv_id="hnsw-tuning",
            topic="HNSW index tuning",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="explain ef search in hnsw",
                    started_at_ms=0,
                    ended_at_ms=1800,
                ),
                Turn(
                    role=TurnRole.USER,
                    text="wait, hold on, restart that",
                    started_at_ms=1900,
                    ended_at_ms=3200,
                    interrupted=True,
                ),
            ],
            gold_facts=[
                "ef_search controls the size of the dynamic candidate list at query time.",
                "Larger ef_search trades latency for recall.",
            ],
        ),
        Conversation(
            conv_id="prom-burn-rate",
            topic="Prometheus SLO burn rate",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="how do i compute slo burn rate from a sli histogram",
                    started_at_ms=0,
                    ended_at_ms=2400,
                ),
            ],
            gold_facts=[
                "Burn rate = error rate / allowed error rate over the window.",
                "A 14.4 burn rate over 1h burns 1% of a 30-day budget.",
            ],
        ),
        Conversation(
            conv_id="empty-noise",
            topic="False-trigger handling",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="background noise",
                    started_at_ms=0,
                    ended_at_ms=400,
                ),
            ],
            gold_facts=[],  # nothing should ground a real reply
        ),
    ]
