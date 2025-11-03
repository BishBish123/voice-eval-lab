"""voice-eval-lab: eval harness + reference pipeline for real-time voice agents."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("voice-eval-lab")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
