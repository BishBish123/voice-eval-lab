"""Audio fixture infrastructure for the voice-eval-lab harness.

Provides:
- ``AudioFixtureStore`` — Protocol for get/add/list operations on WAV fixtures
- ``FilesystemAudioStore`` — filesystem-backed store keyed by (conv_id, turn_index)
- ``SilenceFixtureGenerator`` — creates valid silence WAV bytes via stdlib ``wave``
"""

from voice_eval_lab.audio.filesystem import FilesystemAudioStore
from voice_eval_lab.audio.protocol import AudioFixtureStore
from voice_eval_lab.audio.synthesizer import SilenceFixtureGenerator

__all__ = [
    "AudioFixtureStore",
    "FilesystemAudioStore",
    "SilenceFixtureGenerator",
]
