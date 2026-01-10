"""Bundled golden conversation set — synthetic transcripts with known facts.

Synthetic on purpose: real-conversation corpora have licensing problems
and require audio storage we don't want to commit. The transcripts are
crafted so the metrics light up in interesting ways:

- `postgres-replication` — multi-turn history so the LLM input grows;
  exercises faithfulness on a topic the mock LLM is wired for
- `hnsw-tuning` — has a barge-in turn; exercises barge_in_success_rate
  *and* the new barge_in_latency_p95 metric
- `prom-burn-rate` — single-turn factual exchange; exercises WER on a
  short utterance where small substitutions move the score a lot
- `empty-noise` — the false-trigger scenario; the agent should not
  ground on gold facts because there are none
- `agent-led-debug` — a 4-turn agent-led debug; exercises history growth
  and the `llm_decisiveness` metric since one user turn ("idk") forces
  the mock LLM into its hedging fallback
- `noisy-vad` — silence + cough + actual speech; exercises false-trigger
  semantics and the new `endpointing_accuracy` metric (the cough turn
  has tight start/end frames)
- `double-barge` — two consecutive barge-ins; exercises the
  `barge_in_latency_p95` distribution rather than the binary success bit
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
        # ------------------------------------------------------------------
        # Conversations added in the v0.2 expansion to exercise the new
        # diagnostic metrics (decisiveness, endpointing, barge-in latency).
        # ------------------------------------------------------------------
        Conversation(
            conv_id="agent-led-debug",
            topic="Agent-led debugging session",
            turns=[
                Turn(
                    role=TurnRole.AGENT,
                    text="What error are you seeing?",
                    started_at_ms=0,
                    ended_at_ms=1500,
                ),
                Turn(
                    role=TurnRole.USER,
                    text="connection reset by peer in the http2 client",
                    started_at_ms=1700,
                    ended_at_ms=4200,
                ),
                Turn(
                    role=TurnRole.AGENT,
                    text="Is that during the request or response phase?",
                    started_at_ms=4400,
                    ended_at_ms=6500,
                ),
                Turn(
                    role=TurnRole.USER,
                    text="idk",
                    started_at_ms=6700,
                    ended_at_ms=7000,
                ),
                Turn(
                    role=TurnRole.AGENT,
                    text="Try logging the goaway frame.",
                    started_at_ms=7200,
                    ended_at_ms=9100,
                ),
                Turn(
                    role=TurnRole.USER,
                    text="connection reset by peer means the server sent an RST during the http2 stream",
                    started_at_ms=9300,
                    ended_at_ms=12500,
                ),
            ],
            gold_facts=[
                "connection reset by peer means the server sent a TCP RST during the stream.",
            ],
        ),
        Conversation(
            conv_id="noisy-vad",
            topic="Noisy VAD endpointing",
            turns=[
                # Silence frame — caught by VAD, no useful text.
                Turn(
                    role=TurnRole.USER,
                    text="",
                    started_at_ms=0,
                    ended_at_ms=200,
                ),
                # A cough — short, no real content.
                Turn(
                    role=TurnRole.USER,
                    text="ahem",
                    started_at_ms=300,
                    ended_at_ms=600,
                ),
                # The actual question — this one should ground on a gold fact.
                Turn(
                    role=TurnRole.USER,
                    text="how does webrtc vad classify silence frames",
                    started_at_ms=900,
                    ended_at_ms=4100,
                ),
            ],
            gold_facts=[
                "WebRTC VAD classifies frames using a Gaussian-mixture energy model.",
                "WebRTC VAD operates on 10/20/30ms frames and emits a per-frame voiced bit.",
            ],
        ),
        Conversation(
            conv_id="double-barge",
            topic="Two consecutive barge-ins",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="give me the elevator pitch on raft consensus",
                    started_at_ms=0,
                    ended_at_ms=2700,
                ),
                # First barge-in — user cuts off the agent.
                Turn(
                    role=TurnRole.USER,
                    text="hold on actually start with leader election",
                    started_at_ms=2900,
                    ended_at_ms=5200,
                    interrupted=True,
                ),
                # Second barge-in — user cuts the agent off again.
                Turn(
                    role=TurnRole.USER,
                    text="wait no — log replication first then leader election",
                    started_at_ms=5400,
                    ended_at_ms=8300,
                    interrupted=True,
                ),
            ],
            gold_facts=[
                "Raft elects a leader by majority vote with randomized timeouts.",
                "Raft replicates a log of state-machine commands across the cluster.",
            ],
        ),
    ]
