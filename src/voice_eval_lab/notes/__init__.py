"""Notes RAG package: in-memory and pgvector backends for session-note retrieval."""

from voice_eval_lab.notes.factory import make_notes_store
from voice_eval_lab.notes.memory import InMemoryNotesStore
from voice_eval_lab.notes.protocol import NoteHit, NotesStore

__all__ = ["InMemoryNotesStore", "NoteHit", "NotesStore", "make_notes_store"]
