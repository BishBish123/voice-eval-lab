"""FilesystemAudioStore — WAV fixture store backed by the local filesystem.

Layout::

    {root}/{conv_id}/turn-{turn_index:02d}.wav

Example::

    evals/audio/postgres-replication/turn-00.wav
    evals/audio/postgres-replication/turn-01.wav
    evals/audio/hnsw-tuning/turn-00.wav
"""

from __future__ import annotations

from pathlib import Path


def _validate_wav_header(wav_bytes: bytes) -> None:
    """Raise ``ValueError`` if *wav_bytes* does not start with a valid RIFF/WAVE header.

    A valid WAV file starts with:
    - bytes 0..3: ``RIFF``
    - bytes 4..7: file size (little-endian uint32, not checked)
    - bytes 8..11: ``WAVE``
    """
    if len(wav_bytes) < 12:
        raise ValueError(
            f"wav_bytes is too short ({len(wav_bytes)} bytes) to be a valid WAV file"
        )
    if wav_bytes[:4] != b"RIFF":
        raise ValueError(
            f"wav_bytes does not start with RIFF header (got {wav_bytes[:4]!r})"
        )
    if wav_bytes[8:12] != b"WAVE":
        raise ValueError(
            f"wav_bytes bytes[8:12] are not WAVE (got {wav_bytes[8:12]!r})"
        )


class FilesystemAudioStore:
    """Filesystem-backed audio fixture store.

    Parameters
    ----------
    root:
        Root directory under which fixtures are stored.  Will be created
        automatically by :meth:`add_audio` as needed.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def _path(self, conv_id: str, turn_index: int) -> Path:
        return self._root / conv_id / f"turn-{turn_index:02d}.wav"

    def get_audio(self, conv_id: str, turn_index: int) -> bytes | None:
        """Return WAV bytes for ``(conv_id, turn_index)``, or ``None`` if missing."""
        p = self._path(conv_id, turn_index)
        if not p.exists():
            return None
        return p.read_bytes()

    def add_audio(self, conv_id: str, turn_index: int, wav_bytes: bytes) -> None:
        """Write ``wav_bytes`` to the fixture tree.

        Creates parent directories as needed.  Validates the WAV header
        before writing; raises ``ValueError`` on invalid input.
        """
        _validate_wav_header(wav_bytes)
        p = self._path(conv_id, turn_index)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(wav_bytes)

    def list_keys(self) -> list[tuple[str, int]]:
        """Return all ``(conv_id, turn_index)`` pairs found under *root*, sorted."""
        if not self._root.exists():
            return []
        keys: list[tuple[str, int]] = []
        for wav_path in sorted(self._root.rglob("turn-*.wav")):
            # Expected structure: {root}/{conv_id}/turn-{nn}.wav
            # wav_path.parent.name is the conv_id
            # wav_path.stem is "turn-NN"
            stem = wav_path.stem  # e.g. "turn-02"
            if not stem.startswith("turn-"):
                continue
            index_str = stem[len("turn-"):]
            if not index_str.isdigit():
                continue
            conv_id = wav_path.parent.name
            keys.append((conv_id, int(index_str)))
        return sorted(keys)
