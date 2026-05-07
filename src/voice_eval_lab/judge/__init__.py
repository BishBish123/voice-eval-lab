"""LLM-as-judge faithfulness scoring framework.

Public API
----------
- :class:`~voice_eval_lab.judge.protocol.JudgeProtocol` — structural protocol
  every judge must satisfy.
- :class:`~voice_eval_lab.judge.protocol.JudgeScore` — score dataclass returned
  by every judge.
- :class:`~voice_eval_lab.judge.llm.LLMJudge` — real LLM judge (requires
  ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY`` + ``[real]`` extras).
- :class:`~voice_eval_lab.judge.substring.SubstringProxyJudge` — substring-match
  fallback, zero dependencies.
- :func:`~voice_eval_lab.judge.factory.make_judge` — factory that returns the
  best available judge for the current environment.

Env-key gating
--------------
When ``ANTHROPIC_API_KEY`` (or ``OPENAI_API_KEY``) is set and ``httpx`` is
installed, :func:`make_judge` returns an :class:`LLMJudge`.  Otherwise it
silently returns :class:`SubstringProxyJudge`, keeping the default eval path
(no env keys, no network) fully functional.
"""

from voice_eval_lab.judge.factory import make_judge
from voice_eval_lab.judge.llm import LLMJudge
from voice_eval_lab.judge.protocol import JudgeProtocol, JudgeScore
from voice_eval_lab.judge.substring import SubstringProxyJudge

__all__ = [
    "JudgeProtocol",
    "JudgeScore",
    "LLMJudge",
    "SubstringProxyJudge",
    "make_judge",
]
