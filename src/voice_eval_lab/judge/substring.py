"""Substring-match faithfulness judge — the zero-dependency fallback.

This is a refactored extraction of the existing substring logic that
lives in ``voice_eval_lab.eval.metrics._faithfulness_counts``.  The
behaviour is preserved exactly:

- Both the answer and each key-point are NFKC-normalised + lowercased.
- The turn scores 1.0 if *any* key-point appears as a substring of the
  normalised answer, 0.0 otherwise.
- When no key-points are provided the turn scores 1.0 (vacuously true).
"""

from __future__ import annotations

import unicodedata

from voice_eval_lab.judge.protocol import JudgeProtocol, JudgeScore


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


class SubstringProxyJudge:
    """Faithfulness judge using substring containment as a proxy.

    No API keys required.  Suitable for CI, local development, and as a
    fallback when no LLM key is configured.

    Implements :class:`~voice_eval_lab.judge.protocol.JudgeProtocol`.
    """

    async def score(
        self,
        question: str,
        expected_keypoints: list[str],
        answer: str,
    ) -> JudgeScore:
        """Return 1.0 if the answer contains any key-point, 0.0 otherwise.

        Args:
            question: Not used by the substring judge (included for
                protocol compatibility).
            expected_keypoints: Gold facts to look for.
            answer: The agent reply to evaluate.

        Returns:
            :class:`~voice_eval_lab.judge.protocol.JudgeScore` with
            ``mode="substring"``.
        """
        if not expected_keypoints:
            return JudgeScore(
                score=1.0,
                rationale="No key-points provided; vacuously faithful.",
                mode="substring",
            )
        normalised_answer = _normalize(answer)
        for kp in expected_keypoints:
            if _normalize(kp) in normalised_answer:
                return JudgeScore(
                    score=1.0,
                    rationale=f"Answer contains key-point substring: {kp!r}",
                    mode="substring",
                )
        return JudgeScore(
            score=0.0,
            rationale=(
                f"Answer does not contain any of the {len(expected_keypoints)} "
                "key-point(s) as a substring."
            ),
            mode="substring",
        )


# Runtime check that SubstringProxyJudge satisfies the protocol.
# Evaluated once at import time; any signature drift raises immediately.
assert isinstance(SubstringProxyJudge(), JudgeProtocol), (
    "SubstringProxyJudge does not satisfy JudgeProtocol"
)

__all__ = ["SubstringProxyJudge"]
