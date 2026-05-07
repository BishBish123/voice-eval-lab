"""``WithNotesLLM`` — a Protocol-decorator that wraps any :class:`LLM` with
notes-RAG context injection.

On every :meth:`reply` call the adapter:

1. Queries the notes store using the *last_user_text*.
2. Retrieves the top-3 most similar notes.
3. Prepends a ``[Context notes] …`` line to the returned reply text so
   downstream consumers (eval harness, real agents) can see which notes
   were surfaced.

The wrapped LLM's :attr:`reply` is called unconditionally; this is a
*decorator*, not a *bypass*.  When the store returns no hits the reply is
passed through unchanged.

Typical usage in the eval harness::

    from voice_eval_lab.notes import InMemoryNotesStore
    from voice_eval_lab.notes.llm_adapter import WithNotesLLM
    from voice_eval_lab.pipeline import MockLLM

    store = InMemoryNotesStore()
    await store.add_note("n1", "user is a senior engineer at Postgres")

    llm = WithNotesLLM(inner=MockLLM(), store=store)
    reply, spans = await llm.reply(history, "postgres vacuum", gold_facts)
    # reply will be prefixed with the context note
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from voice_eval_lab.models import PipelineSpan, Turn
from voice_eval_lab.notes.protocol import NoteHit, NotesStore


async def lookup_notes(
    user_text: str,
    store: NotesStore,
    top_k: int = 3,
) -> list[NoteHit]:
    """Query *store* with *user_text* and return the top-*top_k* hits.

    A thin helper so callers do not need to import :class:`NotesStore`
    directly just to drive a lookup.

    Args:
        user_text: The user's transcribed text for the current turn.
        store: Any :class:`~voice_eval_lab.notes.protocol.NotesStore`.
        top_k: Maximum number of hits to return.

    Returns:
        List of :class:`~voice_eval_lab.notes.protocol.NoteHit` ordered by
        descending similarity; may be empty.
    """
    return await store.lookup(user_text, top_k=top_k)


@dataclass
class WithNotesLLM:
    """Decorator that injects notes-RAG context into any :class:`LLM` reply.

    Satisfies the :class:`~voice_eval_lab.pipeline.LLM` Protocol so it can
    be passed anywhere a plain ``LLM`` is expected.

    Args:
        inner: The underlying LLM implementation (e.g. :class:`~voice_eval_lab.pipeline.MockLLM`).
        store: The notes store to query on each turn.
        top_k: Number of note hits to retrieve per turn.
    """

    inner: Any  # LLM Protocol — typed as Any to avoid circular imports
    store: NotesStore
    top_k: int = 3

    async def reply(
        self,
        history: list[Turn],
        last_user_text: str,
        gold_facts: list[str],
    ) -> tuple[str, list[PipelineSpan]]:
        """Query the notes store then delegate to *inner*, prepending hit context."""
        hits = await lookup_notes(last_user_text, self.store, top_k=self.top_k)

        # Delegate to the wrapped LLM.
        inner_reply, spans = await self.inner.reply(
            history, last_user_text, gold_facts
        )

        if not hits:
            return inner_reply, spans

        # Build a compact context prefix; one note per line.
        context_lines = "; ".join(f"[{h.note_id}] {h.text}" for h in hits)
        prefixed = f"[Context notes] {context_lines}\n{inner_reply}"
        return prefixed, spans


__all__ = ["WithNotesLLM", "lookup_notes"]
