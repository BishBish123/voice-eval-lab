"""Tests for the golden conversation set properties."""

from __future__ import annotations

from voice_eval_lab.eval.golden import default_golden_set
from voice_eval_lab.models import TurnRole


class TestGoldenSetSize:
    def test_default_golden_set_returns_twenty_five(self) -> None:
        convs = default_golden_set()
        assert len(convs) == 25

    def test_all_conv_ids_are_unique(self) -> None:
        convs = default_golden_set()
        ids = [c.conv_id for c in convs]
        assert len(ids) == len(set(ids)), "duplicate conv_id found"

    def test_all_conv_ids_are_kebab_case(self) -> None:
        import re

        convs = default_golden_set()
        pattern = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
        bad = [c.conv_id for c in convs if not pattern.match(c.conv_id)]
        assert bad == [], f"non-kebab conv_ids: {bad}"


class TestGoldenSetCategoryCoverage:
    """Assert that at least one conversation per declared category exists."""

    def _ids(self) -> set[str]:
        return {c.conv_id for c in default_golden_set()}

    def test_has_happy_path_technical_qa(self) -> None:
        happy_path = {
            "postgres-replication",
            "prom-burn-rate",
            "tcp-handshake",
            "s3-consistency",
            "react-reconciler",
            "redis-eviction",
        }
        ids = self._ids()
        missing = happy_path - ids
        assert not missing, f"missing happy-path conversations: {missing}"

    def test_has_barge_in_conversations(self) -> None:
        barge_ids = {"hnsw-tuning", "double-barge", "mid-sentence-barge", "early-barge-cloud", "triple-barge"}
        ids = self._ids()
        missing = barge_ids - ids
        assert not missing, f"missing barge-in conversations: {missing}"

    def test_has_ambiguous_input_conversations(self) -> None:
        ambig_ids = {"ambig-cache", "ambig-deploy", "ambig-index"}
        ids = self._ids()
        missing = ambig_ids - ids
        assert not missing, f"missing ambiguous-input conversations: {missing}"

    def test_has_out_of_scope_conversations(self) -> None:
        oos_ids = {"oos-weather", "oos-stock-price"}
        ids = self._ids()
        missing = oos_ids - ids
        assert not missing, f"missing out-of-scope conversations: {missing}"

    def test_has_long_answer_conversations(self) -> None:
        long_ids = {"k8s-networking-deep", "ssl-tls-handshake"}
        ids = self._ids()
        missing = long_ids - ids
        assert not missing, f"missing long-answer conversations: {missing}"

    def test_has_fast_back_and_forth_conversations(self) -> None:
        fast_ids = {"rapid-fire-git", "rapid-fire-regex"}
        ids = self._ids()
        missing = fast_ids - ids
        assert not missing, f"missing fast-back-and-forth conversations: {missing}"

    def test_has_clarifying_question_conversations(self) -> None:
        clarify_ids = {"clarify-oom", "clarify-latency"}
        ids = self._ids()
        missing = clarify_ids - ids
        assert not missing, f"missing clarifying-question conversations: {missing}"

    def test_each_conversation_has_at_least_one_turn(self) -> None:
        convs = default_golden_set()
        empty = [c.conv_id for c in convs if not c.turns]
        assert not empty, f"conversations with no turns: {empty}"

    def test_each_conversation_has_at_least_one_user_turn(self) -> None:
        convs = default_golden_set()
        no_user = [c.conv_id for c in convs if not any(t.role == TurnRole.USER for t in c.turns)]
        assert not no_user, f"conversations with no user turns: {no_user}"

    def test_barge_in_conversations_have_interrupted_turns(self) -> None:
        barge_ids = {"hnsw-tuning", "double-barge", "mid-sentence-barge", "early-barge-cloud", "triple-barge"}
        convs = {c.conv_id: c for c in default_golden_set()}
        for conv_id in barge_ids:
            conv = convs[conv_id]
            has_interrupted = any(t.interrupted for t in conv.turns)
            assert has_interrupted, f"{conv_id} has no interrupted turn"
