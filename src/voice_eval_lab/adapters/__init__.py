"""Real adapter stubs for Groq LLM, Deepgram STT, Cartesia TTS, ElevenLabs TTS, and Whisper STT.

Each adapter reads its API key (or gate env var) at instantiation time.
When the key/var is absent the adapter falls back to the mock implementation
(with deterministic jitter).  When the key is present the first call makes a
minimal real request to validate auth and measure live latency; subsequent
failures fall back to mock with a logged warning.

Unit-testable without any API keys — the no-key (mock) path is the default.

Factories
---------
:func:`make_stt` selects the best available STT adapter:

    - Both keys set      → DeepgramSTT (cloud wins: faster + higher quality)
    - DEEPGRAM_API_KEY   → DeepgramSTT
    - WHISPER_MODEL_NAME → WhisperSTT  (local, no cloud required)
    - Neither            → MockSTT

:func:`make_tts` selects the best available TTS adapter:

    - CARTESIA_API_KEY set   → CartesiaTTS (lower latency for streaming)
    - ELEVENLABS_API_KEY set → ElevenLabsTTS
    - Both set               → CartesiaTTS (lower latency wins)
    - Neither                → MockTTS
"""

from __future__ import annotations

import os

from voice_eval_lab.adapters.cartesia import CartesiaTTS
from voice_eval_lab.adapters.deepgram import DeepgramSTT
from voice_eval_lab.adapters.elevenlabs import ElevenLabsTTS
from voice_eval_lab.adapters.groq import GroqLLM
from voice_eval_lab.adapters.whisper import WhisperSTT
from voice_eval_lab.pipeline import STT, TTS, MockSTT, MockTTS

__all__ = [
    "CartesiaTTS",
    "DeepgramSTT",
    "ElevenLabsTTS",
    "GroqLLM",
    "WhisperSTT",
    "make_stt",
    "make_tts",
]


def make_stt() -> STT:
    """Return the best available STT adapter given the current environment.

    Dispatch order
    --------------
    1. ``DEEPGRAM_API_KEY`` set (with or without ``WHISPER_MODEL_NAME``)
       → :class:`DeepgramSTT` (cloud is faster and higher quality; takes
       precedence when both are configured).
    2. ``WHISPER_MODEL_NAME`` set (and ``DEEPGRAM_API_KEY`` not set)
       → :class:`WhisperSTT` using the named model.
    3. Neither set → :class:`~voice_eval_lab.pipeline.MockSTT` (default;
       deterministic jitter, no external deps, no env vars required).

    Returns
    -------
    An object satisfying the :class:`~voice_eval_lab.pipeline.STT` Protocol.
    """
    deepgram_key = os.environ.get("DEEPGRAM_API_KEY", "")
    whisper_model = os.environ.get("WHISPER_MODEL_NAME", "")

    if deepgram_key:
        return DeepgramSTT()

    if whisper_model:
        return WhisperSTT(model_name=whisper_model)

    return MockSTT()


def make_tts() -> TTS:
    """Return the best available TTS adapter given the current environment.

    Dispatch order
    --------------
    1. ``CARTESIA_API_KEY`` set (with or without ``ELEVENLABS_API_KEY``)
       → :class:`CartesiaTTS` (lower streaming latency; takes precedence
       when both are configured).
    2. ``ELEVENLABS_API_KEY`` set (and ``CARTESIA_API_KEY`` not set)
       → :class:`ElevenLabsTTS`.
    3. Neither set → :class:`~voice_eval_lab.pipeline.MockTTS` (default;
       deterministic jitter, no external deps, no env vars required).

    Returns
    -------
    An object satisfying the :class:`~voice_eval_lab.pipeline.TTS` Protocol.
    """
    cartesia_key = os.environ.get("CARTESIA_API_KEY", "")
    elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY", "")

    if cartesia_key:
        return CartesiaTTS()

    if elevenlabs_key:
        return ElevenLabsTTS()

    return MockTTS()
