"""Judge factory — returns the best available judge given the environment.

:func:`make_judge` is the primary entry point for production code.  It
checks for API keys and returns an :class:`~voice_eval_lab.judge.llm.LLMJudge`
when one is found, otherwise falls back silently to
:class:`~voice_eval_lab.judge.substring.SubstringProxyJudge`.

This means the default eval path (no env keys) always works without any
configuration.
"""

from __future__ import annotations

import logging
import os

from voice_eval_lab.judge.protocol import JudgeProtocol
from voice_eval_lab.judge.substring import SubstringProxyJudge

logger = logging.getLogger(__name__)


def make_judge(
    *,
    mode: str = "auto",
) -> JudgeProtocol:
    """Return the appropriate judge for the current environment.

    Args:
        mode: One of ``"auto"``, ``"llm"``, or ``"substring"``.

            - ``"auto"`` (default): return :class:`~voice_eval_lab.judge.llm.LLMJudge`
              if ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY`` is set,
              else :class:`~voice_eval_lab.judge.substring.SubstringProxyJudge`.
            - ``"llm"``: always return :class:`~voice_eval_lab.judge.llm.LLMJudge`;
              raises if no key is configured or httpx is missing.
            - ``"substring"``: always return
              :class:`~voice_eval_lab.judge.substring.SubstringProxyJudge`.

    Returns:
        An object satisfying :class:`~voice_eval_lab.judge.protocol.JudgeProtocol`.

    Raises:
        ValueError: If *mode* is not one of the accepted values.
        EnvironmentError: If ``mode="llm"`` and no API key is set.
        ImportError: If ``mode="llm"`` and httpx is not installed.
    """
    if mode == "substring":
        return SubstringProxyJudge()

    if mode == "llm":
        # Import here so httpx import errors surface clearly.
        from voice_eval_lab.judge.llm import LLMJudge

        return LLMJudge()  # raises EnvironmentError / ImportError if unconfigured

    if mode == "auto":
        has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY", ""))
        has_openai = bool(os.environ.get("OPENAI_API_KEY", ""))
        if has_anthropic or has_openai:
            try:
                from voice_eval_lab.judge.llm import LLMJudge

                judge = LLMJudge()
                logger.info(
                    "LLMJudge active (%s key found)",
                    "ANTHROPIC_API_KEY" if has_anthropic else "OPENAI_API_KEY",
                )
                return judge
            except ImportError:
                logger.warning(
                    "API key found but httpx is not installed; "
                    "falling back to substring judge. "
                    "Install with: pip install 'voice-eval-lab[real]'"
                )
        return SubstringProxyJudge()

    raise ValueError(
        f"Unknown judge mode: {mode!r}. Valid values: 'auto', 'llm', 'substring'."
    )


__all__ = ["make_judge"]
