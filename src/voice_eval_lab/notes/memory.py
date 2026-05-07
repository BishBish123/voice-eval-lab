"""In-memory notes store backed by numpy cosine similarity.

No external services required.  Used as the default backend and for all
unit / CI tests.

Embedding strategy (in priority order):
1. ``sentence-transformers`` model ``all-MiniLM-L6-v2`` (384-dim) if installed.
2. Hash-based fake encoder that produces a deterministic 384-dim float vector
   from the text bytes.  Results are consistent across calls but not
   semantically meaningful — sufficient for testing the plumbing.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from voice_eval_lab.notes.protocol import NoteHit

# Embedding dimension used by both the real sentence-transformer model and
# the hash encoder so callers and PgVectorNotesStore use a single constant.
EMBEDDING_DIM = 384

# Lazy singleton for the sentence-transformers model so the import and model
# load only happen once and only when the real encoder is actually needed.
_st_model: object | None = None
_st_loaded: bool = False


def _load_st_model() -> object | None:
    """Try to import and load the sentence-transformers model.

    Returns the model object on success, ``None`` if the package is not
    installed.  Subsequent calls return the cached result without re-importing.
    """
    global _st_model, _st_loaded  # noqa: PLW0603
    if _st_loaded:
        return _st_model
    _st_loaded = True
    try:
        from sentence_transformers import SentenceTransformer

        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        _st_model = None
    return _st_model


def _hash_encode(text: str) -> list[float]:
    """Deterministic 384-dim float vector derived from the text bytes.

    Uses Blake2b of the text to seed a sequence of floats.  The result is
    normalised to unit length so cosine similarity is well-defined.

    This is NOT a semantic encoder — it exists only so the store operates
    correctly (insert / lookup / rank) without requiring any ML libraries.
    """
    raw = text.encode("utf-8")
    # Generate enough bytes for 384 floats (each needs 4 bytes → 1536 bytes).
    # We chain repeated Blake2b digests over (raw + counter) until we have
    # enough material.
    floats: list[float] = []
    counter = 0
    while len(floats) < EMBEDDING_DIM:
        digest = hashlib.blake2b(raw + counter.to_bytes(4, "little"), digest_size=64).digest()
        for i in range(0, len(digest), 4):
            if len(floats) >= EMBEDDING_DIM:
                break
            (val,) = struct.unpack_from("<I", digest, i)
            floats.append(val / (2**32 - 1) * 2.0 - 1.0)
        counter += 1
    arr = np.array(floats[:EMBEDDING_DIM], dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    result: list[float] = arr.tolist()
    return result


def _encode(text: str) -> list[float]:
    """Return a 384-dim embedding for *text*.

    Uses the sentence-transformers model when available; falls back to the
    hash-based fake encoder otherwise.
    """
    model = _load_st_model()
    if model is not None:
        # sentence_transformers encode() returns an ndarray; normalise to list.
        # We use Any to avoid importing the optional sentence-transformers types.
        m: Any = model
        vec: np.ndarray = m.encode(text, normalize_embeddings=True)
        encoded: list[float] = vec.tolist()
        return encoded
    return _hash_encode(text)


@dataclass
class _NoteRecord:
    note_id: str
    text: str
    embedding: np.ndarray  # shape (EMBEDDING_DIM,), dtype float32, unit norm


@dataclass
class InMemoryNotesStore:
    """Async-compatible in-memory notes store using cosine similarity.

    Thread safety: the store is not thread-safe.  For the eval harness
    (single-threaded asyncio) this is fine; do not share across threads.
    """

    _records: list[_NoteRecord] = field(default_factory=list, repr=False)

    async def add_note(
        self,
        note_id: str,
        text: str,
        embedding: list[float] | None = None,
    ) -> None:
        """Add or replace a note.

        If a note with the same *note_id* already exists it is replaced in
        place so that repeated ``add_note`` calls are idempotent.
        """
        if embedding is not None:
            arr = np.array(embedding, dtype=np.float32)
        else:
            arr = np.array(_encode(text), dtype=np.float32)

        # Normalise defensively in case the caller supplied a non-unit vector.
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm

        # Replace existing note with the same id, or append.
        for i, rec in enumerate(self._records):
            if rec.note_id == note_id:
                self._records[i] = _NoteRecord(note_id=note_id, text=text, embedding=arr)
                return
        self._records.append(_NoteRecord(note_id=note_id, text=text, embedding=arr))

    async def lookup(self, query: str, top_k: int = 3) -> list[NoteHit]:
        """Return up to *top_k* :class:`~voice_eval_lab.notes.protocol.NoteHit`
        ordered by descending cosine similarity."""
        if not self._records:
            return []

        q_vec = np.array(_encode(query), dtype=np.float32)
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # Stack all embeddings into a matrix for a single batched dot product.
        matrix = np.stack([r.embedding for r in self._records], axis=0)  # (N, D)
        scores = matrix @ q_vec  # cosine similarity, shape (N,)

        # Sort descending; return at most top_k.
        n = min(top_k, len(self._records))
        indices = np.argsort(-scores)[:n]
        return [
            NoteHit(
                note_id=self._records[i].note_id,
                text=self._records[i].text,
                score=float(scores[i]),
            )
            for i in indices
        ]

    async def clear(self) -> None:
        """Remove all notes."""
        self._records.clear()


__all__ = ["EMBEDDING_DIM", "InMemoryNotesStore"]
