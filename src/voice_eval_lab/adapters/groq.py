"""Groq LLM adapter — implements the ``LLM`` Protocol.

Key:     ``GROQ_API_KEY``
Endpoint: POST https://api.groq.com/openai/v1/chat/completions
Model:   ``llama-3.3-70b-versatile`` (sub-100ms TTFT on Groq infra)

Mock path (no key / API failure):
    Delegates to ``MockLLM`` with deterministic jitter enabled.

Real path:
    Single ``httpx`` POST with a 10 s timeout.  On failure logs a warning
    and falls back to mock output so evals degrade gracefully rather than
    crashing the whole run.
"""

from __future__ import annotations

import logging
import os
import time
import warnings

try:
    import httpx as _httpx
except ImportError:  # pragma: no cover
    _httpx = None  # type: ignore[assignment]

from voice_eval_lab.models import PipelineSpan, Turn, TurnRole
from voice_eval_lab.pipeline import MockLLM

logger = logging.getLogger(__name__)

_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "llama-3.3-70b-versatile"
_TIMEOUT_S = 10.0
_GROQ_BASE_URL_ENV = "GROQ_API_BASE_URL"

# Map internal TurnRole values to the role strings the OpenAI/Groq API expects.
# TurnRole.AGENT.value == "agent", but the endpoint requires "assistant".
_OPENAI_ROLE: dict[TurnRole, str] = {
    TurnRole.USER: "user",
    TurnRole.AGENT: "assistant",
}


class GroqLLM:
    """LLM adapter backed by Groq.  Falls back to ``MockLLM`` when the key is absent.

    Real-call shape:
        POST /openai/v1/chat/completions
        body: {"model": "llama-3.3-70b-versatile", "messages": [...], "max_tokens": 256}
        header: Authorization: Bearer <GROQ_API_KEY>

    The adapter is constructed without making any network call so it is safe
    to instantiate in test environments.  Auth is validated on the first
    ``reply()`` call.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._api_key: str | None = os.environ.get("GROQ_API_KEY")
        self._mock: bool = self._api_key is None
        self._inner: MockLLM = MockLLM()
        # URL override: constructor param > env var > hard-coded default.
        self._api_url: str = (
            base_url
            or os.environ.get(_GROQ_BASE_URL_ENV, "")
            or _GROQ_API_URL
        )
        # Turn-context fields mirroring MockLLM so VoicePipeline.run can set
        # them for deterministic jitter on the mock path.
        self._conv_id: str = ""
        self._turn_index: int = -1

    async def reply(
        self,
        history: list[Turn],
        last_user_text: str,
        gold_facts: list[str],
    ) -> tuple[str, list[PipelineSpan]]:
        # Propagate turn context to inner mock for jitter.
        self._inner._conv_id = self._conv_id
        self._inner._turn_index = self._turn_index

        if self._mock:
            return await self._inner.reply(history, last_user_text, gold_facts)

        return await self._real_reply(history, last_user_text, gold_facts)

    async def _real_reply(
        self,
        history: list[Turn],
        last_user_text: str,
        gold_facts: list[str],
    ) -> tuple[str, list[PipelineSpan]]:
        if _httpx is None:  # pragma: no cover
            warnings.warn(
                "httpx is not installed; install voice-eval-lab[real] or add httpx. "
                "Falling back to mock.",
                stacklevel=2,
            )
            return await self._inner.reply(history, last_user_text, gold_facts)

        messages = [{"role": _OPENAI_ROLE[t.role], "content": t.text} for t in history]
        messages.append({"role": "user", "content": last_user_text})

        t0 = time.monotonic()
        try:
            async with _httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
                resp = await client.post(
                    self._api_url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"model": _GROQ_MODEL, "messages": messages, "max_tokens": 256},
                )
                resp.raise_for_status()
                data = resp.json()
                text: str = data["choices"][0]["message"]["content"]
                elapsed_ms = round((time.monotonic() - t0) * 1000)
        except Exception as exc:
            logger.warning(
                "GroqLLM real call failed (%s: %s); falling back to mock.", type(exc).__name__, exc
            )
            return await self._inner.reply(history, last_user_text, gold_facts)

        spans = [
            PipelineSpan(
                name="llm.reply",
                started_at_ms=0,
                ended_at_ms=elapsed_ms,
                attrs={
                    "model": _GROQ_MODEL,
                    "history_len": str(len(history)),
                    "source": "groq",
                },
            )
        ]
        return text, spans
