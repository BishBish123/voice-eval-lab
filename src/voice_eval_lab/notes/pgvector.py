"""Postgres + pgvector notes store.

Active only when the ``NOTES_DSN`` environment variable is set.  Both
``asyncpg`` and ``pgvector`` are optional deps — import errors are caught and
re-raised with a clear message pointing at ``pip install 'voice-eval-lab[real]'``.

DDL (auto-created on first use):

    CREATE TABLE IF NOT EXISTS notes (
        id        TEXT PRIMARY KEY,
        text      TEXT,
        embedding vector(384)
    )

ANN lookup uses the pgvector ``<=>`` operator (cosine distance):

    SELECT id, text, 1 - (embedding <=> $1) AS score
    FROM   notes
    ORDER  BY embedding <=> $1
    LIMIT  $2
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from voice_eval_lab.notes.memory import EMBEDDING_DIM, _encode
from voice_eval_lab.notes.protocol import NoteHit

_DDL = f"""
CREATE TABLE IF NOT EXISTS notes (
    id        TEXT PRIMARY KEY,
    text      TEXT,
    embedding vector({EMBEDDING_DIM})
)
"""

_INSERT = """
INSERT INTO notes (id, text, embedding)
VALUES ($1, $2, $3)
ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding
"""

_ANN_QUERY = """
SELECT id, text, 1 - (embedding <=> $1) AS score
FROM   notes
ORDER  BY embedding <=> $1
LIMIT  $2
"""


def _require_asyncpg() -> Any:
    """Import asyncpg or raise ImportError with a clear install hint."""
    try:
        import asyncpg

        return asyncpg
    except ImportError as exc:
        raise ImportError(
            "asyncpg is required for PgVectorNotesStore. "
            "Install with: pip install 'voice-eval-lab[real]'"
        ) from exc


def _require_pgvector() -> Any:
    """Import pgvector.asyncpg or raise ImportError with a clear install hint."""
    try:
        import pgvector.asyncpg

        return pgvector.asyncpg
    except ImportError as exc:
        raise ImportError(
            "pgvector is required for PgVectorNotesStore. "
            "Install with: pip install 'voice-eval-lab[real]'"
        ) from exc


@dataclass
class PgVectorNotesStore:
    """Notes store backed by Postgres + the pgvector extension.

    The connection is lazy — the first call to :meth:`_conn` opens it and
    creates the table when it does not yet exist.  Call :meth:`close` when
    done to release the connection.

    Args:
        dsn: Postgres connection string.  Defaults to the ``NOTES_DSN``
            environment variable when omitted.
    """

    dsn: str = field(default_factory=lambda: os.environ.get("NOTES_DSN", ""))
    _connection: Any = field(default=None, init=False, repr=False)

    async def _conn(self) -> Any:
        """Return the (cached) asyncpg connection, creating it on first call."""
        if self._connection is None:
            asyncpg: Any = _require_asyncpg()
            pgvector_asyncpg: Any = _require_pgvector()
            conn: Any = await asyncpg.connect(self.dsn)
            await pgvector_asyncpg.register_vector(conn)
            await conn.execute(_DDL)
            self._connection = conn
        return self._connection

    async def add_note(
        self,
        note_id: str,
        text: str,
        embedding: list[float] | None = None,
    ) -> None:
        """Insert or update a note in the Postgres table."""
        vec = embedding if embedding is not None else _encode(text)
        conn: Any = await self._conn()
        await conn.execute(_INSERT, note_id, text, vec)

    async def lookup(self, query: str, top_k: int = 3) -> list[NoteHit]:
        """ANN lookup via pgvector cosine distance operator ``<=>``.

        Returns up to *top_k* :class:`~voice_eval_lab.notes.protocol.NoteHit`
        ordered by descending similarity (highest first).
        """
        vec = _encode(query)
        conn: Any = await self._conn()
        rows: Any = await conn.fetch(_ANN_QUERY, vec, top_k)
        return [
            NoteHit(note_id=row["id"], text=row["text"], score=float(row["score"]))
            for row in rows
        ]

    async def clear(self) -> None:
        """Delete all rows from the notes table."""
        conn: Any = await self._conn()
        await conn.execute("DELETE FROM notes")

    async def close(self) -> None:
        """Close the underlying asyncpg connection."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None


__all__ = ["PgVectorNotesStore"]
