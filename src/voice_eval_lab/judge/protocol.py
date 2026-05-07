"""Judge protocol and JudgeScore dataclass.

Every judge implementation — substring proxy or LLM — must return a
`JudgeScore` from its async `score` method.  The protocol is kept in a
separate module so other packages can depend on just the interface without
pulling in httpx or any LLM-specific imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class JudgeScore:
    """Result of a single faithfulness judgement.

    Attributes:
        score: Float in [0.0, 1.0].  1.0 means the answer fully covers
            all key-points; 0.0 means it covers none.
        rationale: Human-readable explanation of the score.
        mode: Whether the score came from a real LLM call or the
            substring-match proxy.
    """

    score: float
    rationale: str
    mode: Literal["llm", "substring"]


@runtime_checkable
class JudgeProtocol(Protocol):
    """Structural protocol for faithfulness judges.

    Any class with an ``async score`` method matching this signature
    satisfies the protocol and can be used wherever a judge is expected.
    """

    async def score(
        self,
        question: str,
        expected_keypoints: list[str],
        answer: str,
    ) -> JudgeScore:
        """Score the *answer* against the *expected_keypoints*.

        Args:
            question: The user question that prompted the answer.
            expected_keypoints: Gold-fact key-points that the answer should cover.
            answer: The agent's reply to evaluate.

        Returns:
            A :class:`JudgeScore` with `score`, `rationale`, and `mode`.
        """
        ...


__all__ = ["JudgeProtocol", "JudgeScore"]
