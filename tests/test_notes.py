"""Tests for the notes RAG module.

All tests use InMemoryNotesStore only — no Docker / Postgres required.
PgVectorNotesStore is tested with mocked asyncpg so the real driver is
never imported.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voice_eval_lab.notes.factory import make_notes_store
from voice_eval_lab.notes.llm_adapter import WithNotesLLM, lookup_notes
from voice_eval_lab.notes.memory import EMBEDDING_DIM, InMemoryNotesStore, _hash_encode

# ---------------------------------------------------------------------------
# InMemoryNotesStore: basic correctness
# ---------------------------------------------------------------------------


class TestInMemoryNotesStore:
    @pytest.mark.asyncio
    async def test_empty_store_returns_no_hits(self) -> None:
        store = InMemoryNotesStore()
        hits = await store.lookup("anything", top_k=3)
        assert hits == []

    @pytest.mark.asyncio
    async def test_add_and_lookup_single_note(self) -> None:
        store = InMemoryNotesStore()
        await store.add_note("n1", "postgres vacuum tuning")
        hits = await store.lookup("postgres vacuum", top_k=3)
        assert len(hits) == 1
        assert hits[0].note_id == "n1"
        assert hits[0].text == "postgres vacuum tuning"
        # Score is a float in (-1, 1] for cosine similarity.
        assert -1.0 <= hits[0].score <= 1.0

    @pytest.mark.asyncio
    async def test_top_k_ordering(self) -> None:
        """The note most similar to the query should rank first."""
        store = InMemoryNotesStore()
        # Use pre-computed embeddings so the test is deterministic regardless
        # of whether sentence-transformers is installed.
        vec_a = _hash_encode("postgres index tuning ef_search hnsw")
        vec_b = _hash_encode("unrelated topic about cooking recipes")
        vec_c = _hash_encode("postgres performance vacuum analyse")

        await store.add_note("a", "postgres index tuning", embedding=vec_a)
        await store.add_note("b", "cooking", embedding=vec_b)
        await store.add_note("c", "postgres vacuum", embedding=vec_c)

        # Lookup using free-text — we verify the result is a valid ordered list.
        hits = await store.lookup("postgres hnsw tuning", top_k=3)
        assert len(hits) == 3
        # Scores must be in descending order.
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_top_k_truncated_when_fewer_notes(self) -> None:
        store = InMemoryNotesStore()
        await store.add_note("x", "only note")
        hits = await store.lookup("query", top_k=5)
        assert len(hits) == 1

    @pytest.mark.asyncio
    async def test_score_range(self) -> None:
        store = InMemoryNotesStore()
        await store.add_note("n1", "some text about engineering")
        await store.add_note("n2", "another note about databases")
        hits = await store.lookup("engineering databases", top_k=5)
        for h in hits:
            assert -1.0 <= h.score <= 1.0 + 1e-6  # small tolerance for fp rounding

    @pytest.mark.asyncio
    async def test_add_replaces_existing_note_id(self) -> None:
        store = InMemoryNotesStore()
        await store.add_note("n1", "original text")
        await store.add_note("n1", "updated text")
        hits = await store.lookup("updated text", top_k=1)
        assert len(hits) == 1
        assert hits[0].text == "updated text"

    @pytest.mark.asyncio
    async def test_clear_empties_store(self) -> None:
        store = InMemoryNotesStore()
        await store.add_note("n1", "some note")
        await store.clear()
        hits = await store.lookup("some", top_k=3)
        assert hits == []

    @pytest.mark.asyncio
    async def test_custom_embedding_accepted(self) -> None:
        """When an embedding is supplied it should be used without re-encoding."""
        import numpy as np

        store = InMemoryNotesStore()
        # Create a one-hot-ish vector so we can predict the similarity.
        vec = [0.0] * EMBEDDING_DIM
        vec[0] = 1.0
        await store.add_note("n1", "dummy text", embedding=vec)
        assert len(store._records) == 1
        # The stored embedding should be normalised (already unit for this vec).
        assert abs(float(np.linalg.norm(store._records[0].embedding)) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Hash encoder: determinism
# ---------------------------------------------------------------------------


class TestHashEncoder:
    def test_deterministic(self) -> None:
        a = _hash_encode("hello world")
        b = _hash_encode("hello world")
        assert a == b

    def test_different_texts_differ(self) -> None:
        a = _hash_encode("postgres")
        b = _hash_encode("cooking")
        assert a != b

    def test_output_length(self) -> None:
        vec = _hash_encode("test")
        assert len(vec) == EMBEDDING_DIM

    def test_unit_norm(self) -> None:
        import numpy as np

        vec = _hash_encode("test text")
        arr = np.array(vec)
        assert abs(float(np.linalg.norm(arr)) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# PgVectorNotesStore: mocked asyncpg
# ---------------------------------------------------------------------------


class TestPgVectorNotesStoreMocked:
    """Exercises PgVectorNotesStore logic with asyncpg and pgvector fully mocked."""

    def _make_store(self) -> Any:
        from voice_eval_lab.notes.pgvector import PgVectorNotesStore

        return PgVectorNotesStore(dsn="postgresql://mock:mock@localhost/mock")

    @pytest.mark.asyncio
    async def test_connection_setup_creates_table(self) -> None:
        store = self._make_store()
        mock_conn = AsyncMock()

        with (
            patch("voice_eval_lab.notes.pgvector._require_asyncpg") as mock_asyncpg_fn,
            patch("voice_eval_lab.notes.pgvector._require_pgvector") as mock_pgvector_fn,
        ):
            mock_asyncpg = MagicMock()
            mock_asyncpg.connect = AsyncMock(return_value=mock_conn)
            mock_asyncpg_fn.return_value = mock_asyncpg

            mock_pgvector_asyncpg = MagicMock()
            mock_pgvector_asyncpg.register_vector = AsyncMock()
            mock_pgvector_fn.return_value = mock_pgvector_asyncpg

            conn = await store._conn()

        # DDL was executed.
        assert mock_conn.execute.called
        ddl_call = mock_conn.execute.call_args_list[0]
        assert "CREATE TABLE IF NOT EXISTS notes" in ddl_call.args[0]
        assert conn is mock_conn

    @pytest.mark.asyncio
    async def test_add_note_calls_upsert(self) -> None:
        store = self._make_store()
        mock_conn = AsyncMock()
        store._connection = mock_conn  # inject pre-connected mock

        vec = [0.1] * EMBEDDING_DIM
        await store.add_note("n1", "test text", embedding=vec)

        # The INSERT ... ON CONFLICT statement was called.
        insert_call = mock_conn.execute.call_args
        sql = insert_call.args[0]
        assert "INSERT INTO notes" in sql
        assert "ON CONFLICT" in sql
        args = insert_call.args[1:]
        assert args[0] == "n1"
        assert args[1] == "test text"
        assert args[2] == vec

    @pytest.mark.asyncio
    async def test_ann_query_returns_note_hits(self) -> None:
        store = self._make_store()

        # Simulate two rows returned by the ANN query.
        row1: dict[str, Any] = {"id": "a", "text": "postgres tuning", "score": 0.92}
        row2: dict[str, Any] = {"id": "b", "text": "cooking", "score": 0.31}
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[row1, row2])
        store._connection = mock_conn

        hits = await store.lookup("postgres", top_k=2)

        assert len(hits) == 2
        assert hits[0].note_id == "a"
        assert hits[0].score == pytest.approx(0.92)
        assert hits[1].note_id == "b"

    @pytest.mark.asyncio
    async def test_clear_deletes_all_rows(self) -> None:
        store = self._make_store()
        mock_conn = AsyncMock()
        store._connection = mock_conn

        await store.clear()

        mock_conn.execute.assert_called_once_with("DELETE FROM notes")

    @pytest.mark.asyncio
    async def test_score_parsing_float(self) -> None:
        """score should be cast to float even if the driver returns Decimal."""
        from decimal import Decimal

        store = self._make_store()
        row: dict[str, Any] = {"id": "x", "text": "foo", "score": Decimal("0.85")}
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[row])
        store._connection = mock_conn

        hits = await store.lookup("foo", top_k=1)
        assert isinstance(hits[0].score, float)
        assert hits[0].score == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# make_notes_store factory
# ---------------------------------------------------------------------------


class TestMakeNotesStore:
    def test_no_dsn_returns_in_memory(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "NOTES_DSN"}
        with patch.dict(os.environ, env, clear=True):
            store = make_notes_store()
        assert isinstance(store, InMemoryNotesStore)

    def test_dsn_set_returns_pgvector(self) -> None:
        from voice_eval_lab.notes.pgvector import PgVectorNotesStore

        with patch.dict(os.environ, {"NOTES_DSN": "postgresql://u:p@h/db"}):
            store = make_notes_store()
        assert isinstance(store, PgVectorNotesStore)
        assert store.dsn == "postgresql://u:p@h/db"

    def test_satisfies_protocol(self) -> None:
        """Runtime isinstance check against the Protocol."""
        # NotesStore is a Protocol — use isinstance only for runtime_checkable.
        # Since NotesStore is NOT runtime_checkable, just confirm the attributes exist.
        store = make_notes_store()
        assert hasattr(store, "add_note")
        assert hasattr(store, "lookup")
        assert hasattr(store, "clear")


# ---------------------------------------------------------------------------
# WithNotesLLM decorator
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Minimal LLM double that returns a fixed reply."""

    async def reply(
        self,
        history: list[Any],
        last_user_text: str,
        gold_facts: list[str],
    ) -> tuple[str, list[Any]]:
        return "inner reply", []


class TestWithNotesLLM:
    @pytest.mark.asyncio
    async def test_prepends_context_when_hits_found(self) -> None:
        store = InMemoryNotesStore()
        await store.add_note("n1", "user is a postgres expert")
        llm = WithNotesLLM(inner=_FakeLLM(), store=store)
        reply, _spans = await llm.reply([], "postgres", [])
        assert reply.startswith("[Context notes]")
        assert "n1" in reply
        assert "postgres expert" in reply
        assert "inner reply" in reply

    @pytest.mark.asyncio
    async def test_pass_through_when_no_hits(self) -> None:
        store = InMemoryNotesStore()  # empty
        llm = WithNotesLLM(inner=_FakeLLM(), store=store)
        reply, _spans = await llm.reply([], "anything", [])
        assert reply == "inner reply"
        assert not reply.startswith("[Context notes]")

    @pytest.mark.asyncio
    async def test_top_k_respected(self) -> None:
        store = InMemoryNotesStore()
        for i in range(5):
            await store.add_note(f"n{i}", f"note number {i}")
        llm = WithNotesLLM(inner=_FakeLLM(), store=store, top_k=2)
        reply, _ = await llm.reply([], "note", [])
        # Only 2 note IDs should appear in the context prefix.
        prefix = reply.split("\n")[0]
        assert prefix.count("[n") == 2

    @pytest.mark.asyncio
    async def test_inner_spans_preserved(self) -> None:
        from voice_eval_lab.models import PipelineSpan

        class SpanLLM:
            async def reply(
                self, history: list[Any], last_user_text: str, gold_facts: list[str]
            ) -> tuple[str, list[PipelineSpan]]:
                span = PipelineSpan(name="llm.reply", started_at_ms=0, ended_at_ms=10)
                return "reply", [span]

        store = InMemoryNotesStore()
        await store.add_note("n1", "context")
        llm = WithNotesLLM(inner=SpanLLM(), store=store)
        _, spans = await llm.reply([], "query", [])
        assert len(spans) == 1
        assert spans[0].name == "llm.reply"


# ---------------------------------------------------------------------------
# lookup_notes helper
# ---------------------------------------------------------------------------


class TestLookupNotesHelper:
    @pytest.mark.asyncio
    async def test_returns_hits(self) -> None:
        store = InMemoryNotesStore()
        await store.add_note("k1", "postgres hnsw")
        hits = await lookup_notes("postgres", store, top_k=1)
        assert len(hits) == 1
        assert hits[0].note_id == "k1"

    @pytest.mark.asyncio
    async def test_empty_store_returns_empty(self) -> None:
        store = InMemoryNotesStore()
        hits = await lookup_notes("anything", store)
        assert hits == []


# ---------------------------------------------------------------------------
# CLI: notes add / lookup / clear
# ---------------------------------------------------------------------------


class TestNotesCLI:
    def setup_method(self) -> None:
        """Reset the module-level singleton before each test."""
        import voice_eval_lab.cli as cli_mod
        from voice_eval_lab.notes.memory import InMemoryNotesStore as _IMS

        cli_mod._notes_store_singleton = _IMS()

    def test_notes_add_succeeds(self) -> None:
        from typer.testing import CliRunner

        from voice_eval_lab.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["notes", "add", "--id", "test1", "--text", "hello world"])
        assert result.exit_code == 0, result.output
        assert "added note" in result.output

    def test_notes_lookup_with_fixture(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from voice_eval_lab.cli import app

        fixture = tmp_path / "notes.json"
        fixture.write_text(
            json.dumps([{"note_id": "pg-001", "text": "postgres vacuum tuning"}])
        )
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["notes", "lookup", "--query", "postgres", "--fixture", str(fixture)],
        )
        assert result.exit_code == 0, result.output
        assert "pg-001" in result.output

    def test_notes_lookup_no_hits(self) -> None:
        from typer.testing import CliRunner

        from voice_eval_lab.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["notes", "lookup", "--query", "anything"])
        assert result.exit_code == 0, result.output
        assert "no notes found" in result.output

    def test_notes_clear(self) -> None:
        from typer.testing import CliRunner

        from voice_eval_lab.cli import app

        runner = CliRunner()
        # Add a note first.
        runner.invoke(app, ["notes", "add", "--id", "x", "--text", "something"])
        # Clear.
        result = runner.invoke(app, ["notes", "clear"])
        assert result.exit_code == 0, result.output
        assert "cleared" in result.output
        # Lookup should now return no hits.
        result2 = runner.invoke(app, ["notes", "lookup", "--query", "something"])
        assert "no notes found" in result2.output

    def test_notes_lookup_invalid_fixture(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from voice_eval_lab.cli import app

        bad = tmp_path / "bad.json"
        bad.write_text("not json{{{")
        runner = CliRunner()
        result = runner.invoke(app, ["notes", "lookup", "--query", "q", "--fixture", str(bad)])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI: voice-eval run --with-notes
# ---------------------------------------------------------------------------


class TestRunWithNotes:
    def test_run_with_notes_fixture(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from voice_eval_lab.cli import app

        fixture = tmp_path / "notes.json"
        fixture.write_text(
            json.dumps([
                {"note_id": "n1", "text": "user is a senior postgres engineer"},
                {"note_id": "n2", "text": "avoid mentioning AWS"},
            ])
        )
        out = tmp_path / "REPORT.md"
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["run", "--out", str(out), "--with-notes", str(fixture)],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        # Report should still be a valid voice eval report.
        assert "Voice eval report" in out.read_text()

    def test_run_with_missing_notes_fixture_exits(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from voice_eval_lab.cli import app

        out = tmp_path / "REPORT.md"
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["run", "--out", str(out), "--with-notes", str(tmp_path / "nope.json")],
        )
        assert result.exit_code != 0

    def test_run_with_invalid_notes_fixture_exits(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from voice_eval_lab.cli import app

        bad = tmp_path / "bad.json"
        bad.write_text("{}")  # not a list
        out = tmp_path / "REPORT.md"
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["run", "--out", str(out), "--with-notes", str(bad)],
        )
        assert result.exit_code != 0
