"""SmartTurnDetector — end-of-turn detection for the Pipecat pipeline.

Soft-imports ``pipecat-ai``'s ``SmartTurnAnalyzer`` when available and
delegates to it; otherwise falls back to an energy-based silence detector
built from stdlib only (``struct`` + ``statistics``).

Usage::

    detector = SmartTurnDetector(min_silence_ms=500, eou_threshold=0.5)
    state = detector.analyze(audio_chunk_bytes)
    if state.is_end_of_turn:
        # emit the buffered utterance
        ...

The rule-based fallback operates on raw 16-bit signed PCM (little-endian).
Any other sample format will produce incorrect energy readings but will
not raise an exception.
"""

from __future__ import annotations

import logging
import struct
import time
from typing import NamedTuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft-import Pipecat SmartTurnAnalyzer
# ---------------------------------------------------------------------------

try:
    from pipecat.audio.turn.smart_turn.smart_turn_analyzer import (  # type: ignore[import-untyped]
        SmartTurnAnalyzer as _SmartTurnAnalyzer,
    )

    _SMART_TURN_AVAILABLE = True
except ImportError:
    _SmartTurnAnalyzer = None  # type: ignore[assignment,misc]
    _SMART_TURN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class TurnState(NamedTuple):
    """Result of a single ``SmartTurnDetector.analyze`` call.

    Attributes:
        is_end_of_turn: ``True`` when the detector believes the user has
            finished speaking.
        confidence: Float in ``[0.0, 1.0]``.  Higher values indicate
            stronger evidence of end-of-turn.
    """

    is_end_of_turn: bool
    confidence: float


# ---------------------------------------------------------------------------
# Internal helpers — rule-based fallback
# ---------------------------------------------------------------------------

#: Energy threshold below which a 16-bit PCM chunk is considered silent.
#: Derived empirically: sum-of-squares / n_samples below this value
#: corresponds to peak amplitude < ~100 counts (~0.3% of full-scale).
_ENERGY_THRESHOLD: float = 10_000.0

#: Number of bytes per 16-bit PCM sample (little-endian).
_BYTES_PER_SAMPLE: int = 2


def _chunk_energy(audio_chunk: bytes) -> float:
    """Return the mean squared energy of a raw 16-bit PCM chunk.

    Returns ``0.0`` when *audio_chunk* is empty or shorter than one sample.
    """
    if len(audio_chunk) < _BYTES_PER_SAMPLE:
        return 0.0
    n_samples = len(audio_chunk) // _BYTES_PER_SAMPLE
    # Unpack all samples at once for speed.
    samples: tuple[int, ...] = struct.unpack_from(f"<{n_samples}h", audio_chunk)
    return float(sum(s * s for s in samples)) / n_samples


def _chunk_duration_ms(audio_chunk: bytes, sample_rate: int = 16_000) -> float:
    """Return the duration of *audio_chunk* in milliseconds.

    Assumes 16-bit mono PCM at *sample_rate* Hz.
    """
    if sample_rate <= 0:
        return 0.0
    n_samples = len(audio_chunk) // _BYTES_PER_SAMPLE
    return (n_samples / sample_rate) * 1_000.0


# ---------------------------------------------------------------------------
# SmartTurnDetector
# ---------------------------------------------------------------------------


class SmartTurnDetector:
    """End-of-turn detector that delegates to Pipecat's SmartTurnAnalyzer when
    available, or uses an energy-based silence fallback when it is not.

    Args:
        min_silence_ms:  Minimum consecutive silence (in ms) required before
            declaring end-of-turn in fallback mode.  Default: 500 ms.
        eou_threshold:   Confidence threshold forwarded to the Pipecat
            ``SmartTurnAnalyzer`` when it is available.  In fallback mode
            this parameter is not used for the decision but is stored for
            introspection.  Default: 0.5.
        sample_rate:     PCM sample rate used to compute chunk durations in
            fallback mode.  Default: 16 000 Hz (16 kHz mono).
    """

    def __init__(
        self,
        *,
        min_silence_ms: int = 500,
        eou_threshold: float = 0.5,
        sample_rate: int = 16_000,
    ) -> None:
        self.min_silence_ms = min_silence_ms
        self.eou_threshold = eou_threshold
        self.sample_rate = sample_rate

        # --- Pipecat SmartTurnAnalyzer path ---
        self._analyzer: object | None = None
        if _SMART_TURN_AVAILABLE and _SmartTurnAnalyzer is not None:
            try:
                self._analyzer = _SmartTurnAnalyzer(eou_threshold=eou_threshold)
                logger.debug(
                    "SmartTurnDetector: using pipecat SmartTurnAnalyzer "
                    "(eou_threshold=%.2f)",
                    eou_threshold,
                )
            except Exception:
                logger.warning(
                    "SmartTurnDetector: failed to initialise SmartTurnAnalyzer; "
                    "falling back to energy-based silence detector.",
                    exc_info=True,
                )
                self._analyzer = None

        if self._analyzer is None:
            logger.debug(
                "SmartTurnDetector: using energy-based fallback "
                "(min_silence_ms=%d, eou_threshold=%.2f)",
                min_silence_ms,
                eou_threshold,
            )

        # --- Fallback state ---
        # Accumulated silence duration (ms) across consecutive silent chunks.
        self._silence_accumulated_ms: float = 0.0
        # Wall-clock timestamp (monotonic) of the last analyze() call.
        self._last_call_time: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, audio_chunk: bytes) -> TurnState:
        """Analyse one chunk of raw PCM audio and return the current turn state.

        When the Pipecat ``SmartTurnAnalyzer`` is available, this method
        delegates to its inference pipeline.  Otherwise, the energy-based
        fallback is used:

        - If the chunk energy is below ``_ENERGY_THRESHOLD`` and the
          accumulated silence exceeds ``min_silence_ms``:
          → ``TurnState(is_end_of_turn=True, confidence=0.7)``
        - Otherwise:
          → ``TurnState(is_end_of_turn=False, confidence=0.3)``

        Args:
            audio_chunk: Raw 16-bit signed PCM bytes (any length, including
                empty).  An empty chunk is treated as silence.

        Returns:
            A :class:`TurnState` named-tuple.
        """
        if self._analyzer is not None:
            return self._analyze_via_pipecat(audio_chunk)
        return self._analyze_fallback(audio_chunk)

    def reset(self) -> None:
        """Reset the fallback silence accumulator.

        Call this after an end-of-turn has been signalled and the pipeline
        has consumed the buffered utterance, so the detector starts fresh
        for the next utterance.
        """
        self._silence_accumulated_ms = 0.0
        self._last_call_time = None

    # ------------------------------------------------------------------
    # Private: Pipecat path
    # ------------------------------------------------------------------

    def _analyze_via_pipecat(self, audio_chunk: bytes) -> TurnState:
        """Delegate to the Pipecat SmartTurnAnalyzer.

        The Pipecat API is ``analyze(audio_chunk) -> (bool, float)`` or
        returns a dict — we normalise both shapes to ``TurnState``.
        """
        assert self._analyzer is not None
        try:
            result = self._analyzer.analyze(audio_chunk)  # type: ignore[union-attr,attr-defined]
            if isinstance(result, TurnState):
                return result
            if isinstance(result, tuple) and len(result) == 2:
                is_eot, conf = result
                return TurnState(is_end_of_turn=bool(is_eot), confidence=float(conf))
            if isinstance(result, dict):
                is_eot = bool(result.get("is_end_of_turn", False))
                conf = float(result.get("confidence", 0.5))
                return TurnState(is_end_of_turn=is_eot, confidence=conf)
            # Unknown shape — fall back for this call.
            logger.warning(
                "SmartTurnDetector: unexpected SmartTurnAnalyzer result shape %r; "
                "falling back to energy-based detection for this chunk.",
                type(result).__name__,
            )
        except Exception:
            logger.warning(
                "SmartTurnDetector: SmartTurnAnalyzer.analyze() raised; "
                "falling back to energy-based detection for this chunk.",
                exc_info=True,
            )
        return self._analyze_fallback(audio_chunk)

    # ------------------------------------------------------------------
    # Private: energy-based fallback
    # ------------------------------------------------------------------

    def _analyze_fallback(self, audio_chunk: bytes) -> TurnState:
        """Energy-based silence detector (stdlib-only).

        Accumulates silence duration across calls and returns
        ``is_end_of_turn=True`` once silence exceeds ``min_silence_ms``.
        Resets the accumulator when an active (non-silent) chunk arrives.
        """
        now = time.monotonic()
        energy = _chunk_energy(audio_chunk)
        chunk_duration = _chunk_duration_ms(audio_chunk, self.sample_rate)

        if energy < _ENERGY_THRESHOLD:
            # Silent chunk — accumulate silence duration.
            self._silence_accumulated_ms += chunk_duration
        else:
            # Active speech — reset the silence accumulator.
            self._silence_accumulated_ms = 0.0

        self._last_call_time = now

        if self._silence_accumulated_ms >= self.min_silence_ms:
            return TurnState(is_end_of_turn=True, confidence=0.7)
        return TurnState(is_end_of_turn=False, confidence=0.3)


__all__ = [
    "SmartTurnDetector",
    "TurnState",
]
