"""Notes store factory.

Returns the best available backend for the current environment:

* ``NOTES_DSN`` set → :class:`~voice_eval_lab.notes.pgvector.PgVectorNotesStore`
* Otherwise        → :class:`~voice_eval_lab.notes.memory.InMemoryNotesStore`
"""

from __future__ import annotations

import os

from voice_eval_lab.notes.protocol import NotesStore


def make_notes_store() -> NotesStore:
    """Return a :class:`~voice_eval_lab.notes.protocol.NotesStore` for the
    current environment.

    When the ``NOTES_DSN`` environment variable is set the function returns a
    :class:`~voice_eval_lab.notes.pgvector.PgVectorNotesStore` configured
    with that DSN.  Otherwise an
    :class:`~voice_eval_lab.notes.memory.InMemoryNotesStore` is returned —
    no external services required.
    """
    dsn = os.environ.get("NOTES_DSN", "")
    if dsn:
        from voice_eval_lab.notes.pgvector import PgVectorNotesStore

        return PgVectorNotesStore(dsn=dsn)
    from voice_eval_lab.notes.memory import InMemoryNotesStore

    return InMemoryNotesStore()


__all__ = ["make_notes_store"]
