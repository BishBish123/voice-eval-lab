"""Bundled golden conversation set — synthetic transcripts with known facts.

Synthetic on purpose: real-conversation corpora have licensing problems
and require audio storage we don't want to commit. The transcripts are
crafted so the metrics light up in interesting ways:

Original 7 conversations (v0.1 / v0.2):
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

Expansion to 25 conversations (v0.3):
Happy-path technical Q&A (4 new):
- `tcp-handshake` — networking: TCP three-way handshake details
- `s3-consistency` — cloud: S3 strong read-after-write consistency model
- `react-reconciler` — frontend: React reconciler / fiber diffing
- `redis-eviction` — databases: Redis maxmemory eviction policies

User interrupts agent (3 new):
- `mid-sentence-barge` — user barge-in mid-explanation, redirects topic
- `early-barge-cloud` — user barge-in before agent finishes intro
- `triple-barge` — three consecutive interruptions (extends double-barge)

Ambiguous user input (3 new):
- `ambig-cache` — "it" is ambiguous (Redis vs CPU cache)
- `ambig-deploy` — "push to prod" without specifying which service
- `ambig-index` — "the index is slow" without naming the DB or table

Out-of-scope (2 new):
- `oos-weather` — user asks for a weather forecast mid-technical call
- `oos-stock-price` — user asks for a stock quote

Long answer (2 new):
- `k8s-networking-deep` — long explanation of k8s pod networking
- `ssl-tls-handshake` — long TLS 1.3 handshake walkthrough

Fast back-and-forth (2 new):
- `rapid-fire-git` — rapid multi-turn git command Q&A
- `rapid-fire-regex` — rapid multi-turn regex clarifications

Clarifying question (2 new):
- `clarify-oom` — agent asks for more info on OOM context
- `clarify-latency` — agent asks which layer is slow before answering
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
        # ------------------------------------------------------------------
        # v0.3 expansion: 18 new conversations covering additional categories
        # ------------------------------------------------------------------

        # --- Happy-path technical Q&A (4 new) ---

        Conversation(
            conv_id="tcp-handshake",
            topic="TCP three-way handshake",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="walk me through the tcp three-way handshake",
                    started_at_ms=0,
                    ended_at_ms=2200,
                ),
                Turn(
                    role=TurnRole.AGENT,
                    text="The client sends SYN, the server replies SYN-ACK, then the client sends ACK. After that the connection is established.",
                    started_at_ms=2400,
                    ended_at_ms=6800,
                ),
                Turn(
                    role=TurnRole.USER,
                    text="what port does the server listen on for a new connection before accepting",
                    started_at_ms=7000,
                    ended_at_ms=9500,
                ),
            ],
            gold_facts=[
                "TCP three-way handshake: SYN, SYN-ACK, ACK.",
                "The server listens on the well-known port; after accept() the OS assigns an ephemeral port for the session.",
                "SYN cookies protect against SYN-flood attacks by encoding state in the sequence number.",
            ],
        ),
        Conversation(
            conv_id="s3-consistency",
            topic="S3 strong read-after-write consistency",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="does s3 give strong read after write consistency",
                    started_at_ms=0,
                    ended_at_ms=2600,
                ),
                Turn(
                    role=TurnRole.AGENT,
                    text="Yes. Since December 2020 S3 provides strong read-after-write consistency for all GET, PUT, LIST, and DELETE operations automatically, with no extra configuration.",
                    started_at_ms=2900,
                    ended_at_ms=8200,
                ),
                Turn(
                    role=TurnRole.USER,
                    text="does that apply to list operations too",
                    started_at_ms=8500,
                    ended_at_ms=10100,
                ),
            ],
            gold_facts=[
                "S3 provides strong read-after-write consistency for PUT, GET, LIST, and DELETE since December 2020.",
                "No extra configuration is required for S3 strong consistency.",
                "S3 LIST is strongly consistent — a newly uploaded object appears immediately.",
            ],
        ),
        Conversation(
            conv_id="react-reconciler",
            topic="React reconciler and fiber diffing",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="how does react decide what to re-render when state changes",
                    started_at_ms=0,
                    ended_at_ms=2800,
                ),
                Turn(
                    role=TurnRole.AGENT,
                    text="React builds a fiber tree and on each render computes a diff — the reconciler compares the new tree to the previous one and schedules only the minimum set of DOM mutations.",
                    started_at_ms=3100,
                    ended_at_ms=9400,
                ),
                Turn(
                    role=TurnRole.USER,
                    text="what is the key prop used for",
                    started_at_ms=9700,
                    ended_at_ms=11300,
                ),
            ],
            gold_facts=[
                "React's reconciler (Fiber) diffs the virtual DOM tree to find the minimum DOM mutations.",
                "The key prop lets React identify which list items changed, were added, or removed without comparing subtrees.",
                "React batches state updates in event handlers to reduce render passes.",
            ],
        ),
        Conversation(
            conv_id="redis-eviction",
            topic="Redis maxmemory eviction policies",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="what eviction policies does redis support when it hits maxmemory",
                    started_at_ms=0,
                    ended_at_ms=3100,
                ),
                Turn(
                    role=TurnRole.AGENT,
                    text="Redis supports eight policies: noeviction, allkeys-lru, volatile-lru, allkeys-lfu, volatile-lfu, allkeys-random, volatile-random, and volatile-ttl.",
                    started_at_ms=3400,
                    ended_at_ms=9700,
                ),
                Turn(
                    role=TurnRole.USER,
                    text="which one is best for a cache where every key should be evictable",
                    started_at_ms=10000,
                    ended_at_ms=12800,
                ),
            ],
            gold_facts=[
                "Redis maxmemory-policy options include allkeys-lru, allkeys-lfu, volatile-lru, and noeviction.",
                "allkeys-lru evicts the least-recently-used key across the entire keyspace.",
                "allkeys-lfu evicts the least-frequently-used key; better for skewed access patterns.",
            ],
        ),

        # --- User interrupts agent (3 new) ---

        Conversation(
            conv_id="mid-sentence-barge",
            topic="Mid-sentence barge-in during explanation",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="explain how oauth2 authorization code flow works",
                    started_at_ms=0,
                    ended_at_ms=2500,
                ),
                # Agent starts explaining; user cuts in.
                Turn(
                    role=TurnRole.USER,
                    text="wait, skip the redirect, just tell me what the token endpoint does",
                    started_at_ms=2700,
                    ended_at_ms=6100,
                    interrupted=True,
                ),
                Turn(
                    role=TurnRole.USER,
                    text="and what is pkce for",
                    started_at_ms=8500,
                    ended_at_ms=10200,
                ),
            ],
            gold_facts=[
                "The OAuth2 token endpoint exchanges an authorization code for an access token.",
                "PKCE (Proof Key for Code Exchange) prevents authorization code interception attacks from public clients.",
                "The token endpoint returns access_token, token_type, expires_in, and optionally refresh_token.",
            ],
        ),
        Conversation(
            conv_id="early-barge-cloud",
            topic="Early barge-in before agent finishes intro",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="tell me about cloud run versus cloud functions",
                    started_at_ms=0,
                    ended_at_ms=2200,
                ),
                # Agent barely starts; user barge-in.
                Turn(
                    role=TurnRole.USER,
                    text="actually just focus on cold starts",
                    started_at_ms=2400,
                    ended_at_ms=4800,
                    interrupted=True,
                ),
            ],
            gold_facts=[
                "Cloud Run containers can stay warm between requests; Cloud Functions are function-scoped and have more frequent cold starts.",
                "Cold starts on Cloud Run are typically 1-3 seconds; Cloud Functions can vary by runtime.",
                "Cloud Run supports minimum instances to eliminate cold starts entirely.",
            ],
        ),
        Conversation(
            conv_id="triple-barge",
            topic="Three consecutive barge-ins",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="explain the cap theorem",
                    started_at_ms=0,
                    ended_at_ms=1900,
                ),
                # First barge-in.
                Turn(
                    role=TurnRole.USER,
                    text="hold on, start with consistency",
                    started_at_ms=2100,
                    ended_at_ms=4200,
                    interrupted=True,
                ),
                # Second barge-in.
                Turn(
                    role=TurnRole.USER,
                    text="no wait, availability first",
                    started_at_ms=4400,
                    ended_at_ms=6300,
                    interrupted=True,
                ),
                # Third barge-in.
                Turn(
                    role=TurnRole.USER,
                    text="actually just give me a one-sentence summary of all three",
                    started_at_ms=6500,
                    ended_at_ms=9200,
                    interrupted=True,
                ),
            ],
            gold_facts=[
                "CAP theorem: a distributed system can provide at most two of Consistency, Availability, and Partition tolerance.",
                "Consistency means every read receives the most recent write.",
                "Availability means every request receives a response (not necessarily the latest data).",
            ],
        ),

        # --- Ambiguous user input (3 new) ---

        Conversation(
            conv_id="ambig-cache",
            topic="Ambiguous cache reference",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="our latency spiked and i think it is the cache",
                    started_at_ms=0,
                    ended_at_ms=2800,
                ),
                Turn(
                    role=TurnRole.AGENT,
                    text="Which cache do you mean — your Redis layer, a CDN, or the CPU's L2/L3?",
                    started_at_ms=3100,
                    ended_at_ms=5600,
                ),
                Turn(
                    role=TurnRole.USER,
                    text="redis, our hit rate dropped from ninety percent to sixty",
                    started_at_ms=5900,
                    ended_at_ms=8600,
                ),
            ],
            gold_facts=[
                "A Redis cache hit-rate drop from 90% to 60% significantly increases backend DB load.",
                "Common causes: key expiry mis-tuned, cache eviction under memory pressure, or a cold-start after a restart.",
            ],
        ),
        Conversation(
            conv_id="ambig-deploy",
            topic="Ambiguous deployment target",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="i want to push to prod",
                    started_at_ms=0,
                    ended_at_ms=1800,
                ),
                Turn(
                    role=TurnRole.AGENT,
                    text="Which service are you deploying — the API, the frontend, or the worker?",
                    started_at_ms=2100,
                    ended_at_ms=4200,
                ),
                Turn(
                    role=TurnRole.USER,
                    text="the api service on kubernetes",
                    started_at_ms=4500,
                    ended_at_ms=6200,
                ),
            ],
            gold_facts=[
                "kubectl rollout status deployment/<name> confirms a rolling update completed.",
                "A Kubernetes rolling update replaces pods one by one to maintain availability.",
            ],
        ),
        Conversation(
            conv_id="ambig-index",
            topic="Ambiguous slow index without table name",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="the index is really slow",
                    started_at_ms=0,
                    ended_at_ms=1600,
                ),
                Turn(
                    role=TurnRole.AGENT,
                    text="Which database and table are you working with? And do you mean query planning is slow or index scan execution is slow?",
                    started_at_ms=1900,
                    ended_at_ms=4800,
                ),
                Turn(
                    role=TurnRole.USER,
                    text="postgres, the orders table, scans take two seconds on a three million row table",
                    started_at_ms=5100,
                    ended_at_ms=8400,
                ),
            ],
            gold_facts=[
                "EXPLAIN (ANALYZE, BUFFERS) reveals whether a Postgres index scan is hitting shared buffers or reading from disk.",
                "A two-second index scan on 3M rows often indicates bloat or a missing partial index.",
                "pg_stat_user_indexes shows idx_scan count to identify unused indexes.",
            ],
        ),

        # --- Out-of-scope (2 new) ---

        Conversation(
            conv_id="oos-weather",
            topic="Out-of-scope weather query mid-call",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="by the way what is the weather in san francisco today",
                    started_at_ms=0,
                    ended_at_ms=2600,
                ),
            ],
            gold_facts=[],  # agent should decline or note it can't help with this
        ),
        Conversation(
            conv_id="oos-stock-price",
            topic="Out-of-scope stock price query",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="what is nvidia's stock price right now",
                    started_at_ms=0,
                    ended_at_ms=2100,
                ),
            ],
            gold_facts=[],  # agent should decline; no facts to ground
        ),

        # --- Long answer (2 new) ---

        Conversation(
            conv_id="k8s-networking-deep",
            topic="Kubernetes pod networking deep-dive",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="explain in detail how pod to pod networking works in kubernetes",
                    started_at_ms=0,
                    ended_at_ms=3200,
                ),
            ],
            gold_facts=[
                "Every pod gets its own IP address; pods communicate directly without NAT.",
                "CNI plugins (Calico, Flannel, Cilium) implement the network plumbing between nodes.",
                "kube-proxy programs iptables (or ipvs) rules to load-balance ClusterIP Services to pod endpoints.",
                "NetworkPolicy resources restrict ingress/egress traffic at the pod level.",
            ],
        ),
        Conversation(
            conv_id="ssl-tls-handshake",
            topic="TLS 1.3 handshake walkthrough",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="walk me through the tls 1.3 handshake from client hello to first application data",
                    started_at_ms=0,
                    ended_at_ms=3600,
                ),
            ],
            gold_facts=[
                "TLS 1.3 reduces handshake to 1-RTT (or 0-RTT for resumed sessions).",
                "ClientHello includes supported cipher suites and key_share extension with ephemeral DH parameters.",
                "The server responds with ServerHello, Certificate, CertificateVerify, and Finished in one flight.",
                "TLS 1.3 removed RSA key exchange; only (EC)DHE is supported for forward secrecy.",
            ],
        ),

        # --- Fast back-and-forth (2 new) ---

        Conversation(
            conv_id="rapid-fire-git",
            topic="Rapid-fire git command Q&A",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="how do i undo the last commit but keep the changes staged",
                    started_at_ms=0,
                    ended_at_ms=2400,
                ),
                Turn(
                    role=TurnRole.AGENT,
                    text="git reset --soft HEAD~1",
                    started_at_ms=2500,
                    ended_at_ms=3200,
                ),
                Turn(
                    role=TurnRole.USER,
                    text="and if i want to discard the changes too",
                    started_at_ms=3300,
                    ended_at_ms=5000,
                ),
                Turn(
                    role=TurnRole.AGENT,
                    text="git reset --hard HEAD~1",
                    started_at_ms=5100,
                    ended_at_ms=5900,
                ),
                Turn(
                    role=TurnRole.USER,
                    text="how do i see which files changed between two commits",
                    started_at_ms=6000,
                    ended_at_ms=7900,
                ),
                Turn(
                    role=TurnRole.AGENT,
                    text="git diff <commit1> <commit2> --name-only",
                    started_at_ms=8000,
                    ended_at_ms=9000,
                ),
                Turn(
                    role=TurnRole.USER,
                    text="great thanks",
                    started_at_ms=9100,
                    ended_at_ms=9600,
                ),
            ],
            gold_facts=[
                "git reset --soft HEAD~1 undoes the commit but leaves changes staged.",
                "git reset --hard HEAD~1 discards the commit and all working-tree changes.",
                "git diff <a> <b> --name-only lists filenames changed between two commits.",
            ],
        ),
        Conversation(
            conv_id="rapid-fire-regex",
            topic="Rapid-fire regex clarifications",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="what does the dollar sign mean in a regex",
                    started_at_ms=0,
                    ended_at_ms=2000,
                ),
                Turn(
                    role=TurnRole.AGENT,
                    text="It anchors the match to the end of the string (or line in multiline mode).",
                    started_at_ms=2100,
                    ended_at_ms=4300,
                ),
                Turn(
                    role=TurnRole.USER,
                    text="what about the question mark",
                    started_at_ms=4400,
                    ended_at_ms=5800,
                ),
                Turn(
                    role=TurnRole.AGENT,
                    text="After a token it makes that token optional — zero or one occurrence. After a quantifier like * or + it makes the quantifier lazy.",
                    started_at_ms=5900,
                    ended_at_ms=9100,
                ),
                Turn(
                    role=TurnRole.USER,
                    text="what does a lookahead do",
                    started_at_ms=9200,
                    ended_at_ms=10700,
                ),
                Turn(
                    role=TurnRole.AGENT,
                    text="A lookahead (?=...) asserts a pattern exists ahead without consuming characters.",
                    started_at_ms=10800,
                    ended_at_ms=13200,
                ),
                Turn(
                    role=TurnRole.USER,
                    text="ok got it",
                    started_at_ms=13300,
                    ended_at_ms=13900,
                ),
            ],
            gold_facts=[
                "$ in regex anchors to end of string; in multiline mode it matches end of each line.",
                "? after a token means zero-or-one (optional); after a quantifier it makes it lazy.",
                "(?=...) is a positive lookahead: asserts without consuming.",
            ],
        ),

        # --- Clarifying question (2 new) ---

        Conversation(
            conv_id="clarify-oom",
            topic="Agent asks for OOM context before answering",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="my service keeps getting killed",
                    started_at_ms=0,
                    ended_at_ms=1800,
                ),
                Turn(
                    role=TurnRole.AGENT,
                    text="Is the process being OOM-killed, or is it exiting with a non-zero code? And what runtime — JVM, Node, Python?",
                    started_at_ms=2100,
                    ended_at_ms=5200,
                ),
                Turn(
                    role=TurnRole.USER,
                    text="oom killed, it is a java service running in a container with five twelve megabytes limit",
                    started_at_ms=5500,
                    ended_at_ms=9300,
                ),
            ],
            gold_facts=[
                "The JVM's -Xmx sets the heap ceiling but non-heap (metaspace, threads, off-heap) counts against the container limit.",
                "Container OOM kills appear in dmesg and in kubectl describe pod as OOMKilled.",
                "Adding -XX:+UseContainerSupport (default since JDK 10) lets the JVM respect the cgroup memory limit.",
            ],
        ),
        Conversation(
            conv_id="clarify-latency",
            topic="Agent asks which layer is slow before diagnosing",
            turns=[
                Turn(
                    role=TurnRole.USER,
                    text="everything feels slow",
                    started_at_ms=0,
                    ended_at_ms=1500,
                ),
                Turn(
                    role=TurnRole.AGENT,
                    text="Can you tell me more — is it slow at the browser, the API response time, or the database query time? And how slow are we talking?",
                    started_at_ms=1800,
                    ended_at_ms=5000,
                ),
                Turn(
                    role=TurnRole.USER,
                    text="api calls are taking three to four seconds, the database queries look fine at under fifty milliseconds",
                    started_at_ms=5300,
                    ended_at_ms=9100,
                ),
            ],
            gold_facts=[
                "If DB queries are fast but API latency is high, the bottleneck is in application processing, a downstream service, or network I/O.",
                "Distributed tracing (e.g. OpenTelemetry) attributes time to each service span to pinpoint the bottleneck.",
                "N+1 query patterns can hide fast individual queries while producing high aggregate latency.",
            ],
        ),
    ]
