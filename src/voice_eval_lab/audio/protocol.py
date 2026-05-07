"""AudioFixtureStore Protocol — structural typing surface for audio fixture stores."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AudioFixtureStore(Protocol):
    """Protocol for stores that map ``(conv_id, turn_index)`` to WAV bytes.

    All conforming implementations must be usable without inheriting from
    this class — structural subtyping via ``Protocol`` is sufficient.
    """

    def get_audio(self, conv_id: str, turn_index: int) -> bytes | None:
        """Return the WAV bytes for the given key, or ``None`` if absent."""
        ...

    def add_audio(self, conv_id: str, turn_index: int, wav_bytes: bytes) -> None:
        """Store WAV bytes for the given key.

        Implementations must validate that ``wav_bytes`` is a valid WAV
        file (starts with ``RIFF....WAVE``) and raise ``ValueError`` if not.
        """
        ...

    def list_keys(self) -> list[tuple[str, int]]:
        """Return all stored ``(conv_id, turn_index)`` pairs, sorted."""
        ...
