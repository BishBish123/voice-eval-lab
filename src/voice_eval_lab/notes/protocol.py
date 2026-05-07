"""Protocol and result type for the notes RAG store.

Any class that satisfies :class:`NotesStore` can be used wherever a notes
backend is expected.  The two shipped implementations are
:class:`~voice_eval_lab.notes.memory.InMemoryNotesStore` (default, no
external deps) and
:class:`~voice_eval_lab.notes.pgvector.PgVectorNotesStore` (Postgres +
pgvector, gated by the ``NOTES_DSN`` env var).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class NoteHit:
    """One result from a notes-store lookup.

    Attributes:
        note_id: Opaque identifier supplied when the note was added.
        text: The raw note text.
        score: Cosine similarity in [0.0, 1.0]; higher is more relevant.
    """

    note_id: str
    text: str
    score: float


class NotesStore(Protocol):
    """Structural protocol for session-note RAG backends."""

    async def add_note(
        self,
        note_id: str,
        text: str,
        embedding: list[float] | None = None,
    ) -> None:
        """Persist *text* under *note_id*.

        Args:
            note_id: Stable identifier (caller's responsibility to keep unique).
            text: Raw note text that will be embedded and indexed.
            embedding: Pre-computed embedding.  When supplied, the backend
                SHOULD store it directly instead of re-computing.  When
                ``None``, the backend computes the embedding from *text*.
        """
        ...

    async def lookup(self, query: str, top_k: int = 3) -> list[NoteHit]:
        """Return the *top_k* notes most similar to *query*.

        Args:
            query: Free-text search string (will be embedded).
            top_k: Maximum number of hits to return.

        Returns:
            List of :class:`NoteHit`, ordered by descending similarity score.
            May be shorter than *top_k* when the store has fewer than *top_k*
            notes, or empty when the store is empty.
        """
        ...

    async def clear(self) -> None:
        """Remove all notes from the store."""
        ...


__all__ = ["NoteHit", "NotesStore"]
