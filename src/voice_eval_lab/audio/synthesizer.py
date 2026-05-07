"""SilenceFixtureGenerator — creates valid WAV silence bytes via stdlib ``wave``.

No third-party audio libraries required.  Uses only the stdlib ``wave`` and
``struct`` modules, which are available in every CPython installation.
"""

from __future__ import annotations

import io
import wave


class SilenceFixtureGenerator:
    """Generate valid WAV silence bytes for use as audio fixtures.

    Parameters
    ----------
    sample_rate:
        Audio sample rate in Hz.  Default is 16 000 (16 kHz, the standard
        for speech recognition models such as Deepgram Nova-2).
    n_channels:
        Number of audio channels.  Default is 1 (mono).
    sample_width:
        Sample width in bytes.  Default is 2 (16-bit PCM).
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_channels: int = 1,
        sample_width: int = 2,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.sample_width = sample_width

    def generate(self, duration_ms: int) -> bytes:
        """Return a valid WAV file containing ``duration_ms`` milliseconds of silence.

        The result can be fed directly to :meth:`FilesystemAudioStore.add_audio`
        or parsed by the stdlib ``wave.open`` function.

        Parameters
        ----------
        duration_ms:
            Duration in milliseconds.  Must be >= 0.

        Returns
        -------
        bytes
            Raw WAV file bytes (RIFF/WAVE header + PCM data).
        """
        if duration_ms < 0:
            raise ValueError(f"duration_ms must be >= 0, got {duration_ms!r}")
        n_frames = int(self.sample_rate * duration_ms / 1000)
        # Each frame = n_channels * sample_width bytes of zeros.
        raw_audio = b"\x00" * (n_frames * self.n_channels * self.sample_width)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.n_channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(raw_audio)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Convenience: expose the raw PCM frame count for testing
    # ------------------------------------------------------------------

    def frame_count(self, duration_ms: int) -> int:
        """Return the number of PCM frames for *duration_ms* at this sample rate."""
        return int(self.sample_rate * duration_ms / 1000)
