"""Tests for the LLM-as-judge faithfulness scoring framework.

Coverage:
- SubstringProxyJudge default behaviour (no env keys required)
- LLMJudge with mocked httpx → correct score parsing
- LLMJudge with malformed LLM response → fail-soft fallback to substring
- make_judge factory: env-present → LLMJudge, env-absent → SubstringProxyJudge
- Cohen's kappa formula: perfect agreement, random ~0, perfect disagreement
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voice_eval_lab.judge.calibration import JudgeAgreement, cohens_kappa
from voice_eval_lab.judge.factory import make_judge
from voice_eval_lab.judge.protocol import JudgeProtocol
from voice_eval_lab.judge.substring import SubstringProxyJudge

# ---------------------------------------------------------------------------
# SubstringProxyJudge
# ---------------------------------------------------------------------------


class TestSubstringProxyJudge:
    async def test_match_returns_one(self) -> None:
        judge = SubstringProxyJudge()
        result = await judge.score(
            question="what is WAL?",
            expected_keypoints=["Write-Ahead Logging"],
            answer="WAL stands for Write-Ahead Logging in Postgres.",
        )
        assert result.score == 1.0
        assert result.mode == "substring"

    async def test_no_match_returns_zero(self) -> None:
        judge = SubstringProxyJudge()
        result = await judge.score(
            question="what is WAL?",
            expected_keypoints=["Write-Ahead Logging"],
            answer="The agent doesn't know.",
        )
        assert result.score == 0.0
        assert result.mode == "substring"

    async def test_empty_keypoints_vacuously_faithful(self) -> None:
        judge = SubstringProxyJudge()
        result = await judge.score(
            question="q",
            expected_keypoints=[],
            answer="anything",
        )
        assert result.score == 1.0
        assert result.mode == "substring"

    async def test_case_insensitive(self) -> None:
        judge = SubstringProxyJudge()
        result = await judge.score(
            question="q",
            expected_keypoints=["POSTGRES"],
            answer="postgres is great",
        )
        assert result.score == 1.0

    async def test_any_keypoint_sufficient(self) -> None:
        judge = SubstringProxyJudge()
        result = await judge.score(
            question="q",
            expected_keypoints=["missing fact", "postgres"],
            answer="postgres handles replication well",
        )
        assert result.score == 1.0

    def test_satisfies_protocol(self) -> None:
        assert isinstance(SubstringProxyJudge(), JudgeProtocol)


# ---------------------------------------------------------------------------
# LLMJudge — mocked httpx
# ---------------------------------------------------------------------------


class TestLLMJudge:
    """LLMJudge tests use mocked httpx — no real LLM calls."""

    def _make_anthropic_response(self, score: float, rationale: str) -> MagicMock:
        """Build a mock httpx.Response that looks like an Anthropic messages reply."""
        body = {
            "content": [{"text": json.dumps({"score": score, "rationale": rationale})}]
        }
        resp = MagicMock()
        resp.json.return_value = body
        resp.raise_for_status = MagicMock()
        return resp

    async def test_llm_judge_parses_score(self) -> None:
        from voice_eval_lab.judge.llm import LLMJudge

        mock_resp = self._make_anthropic_response(0.8, "Covers the key point.")

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=False),
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            judge = LLMJudge(anthropic_key="test-key")
            result = await judge.score(
                question="What is WAL?",
                expected_keypoints=["Write-Ahead Logging"],
                answer="WAL is Write-Ahead Logging.",
            )

        assert result.mode == "llm"
        assert abs(result.score - 0.8) < 1e-6
        assert "Covers" in result.rationale

    async def test_llm_judge_score_clamped_to_unit_interval(self) -> None:
        from voice_eval_lab.judge.llm import LLMJudge

        # LLM returns score > 1.0 — should be clamped to 1.0
        mock_resp = self._make_anthropic_response(1.5, "Exceeds bounds.")

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=False),
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            judge = LLMJudge(anthropic_key="test-key")
            result = await judge.score("q", ["kp"], "a")

        assert result.score == 1.0

    async def test_malformed_llm_response_falls_back_to_substring(self) -> None:
        from voice_eval_lab.judge.llm import LLMJudge

        # LLM returns garbage — judge should fall back to substring
        bad_resp = MagicMock()
        bad_resp.json.return_value = {"content": [{"text": "not valid json at all!!!"}]}
        bad_resp.raise_for_status = MagicMock()

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=False),
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=bad_resp)
            mock_client_cls.return_value = mock_client

            judge = LLMJudge(anthropic_key="test-key")
            result = await judge.score(
                question="q",
                expected_keypoints=["WAL"],
                answer="WAL is a logging mechanism",  # substring match → 1.0
            )

        # Falls back to substring: "WAL" in answer → score=1.0, mode="substring"
        assert result.mode == "substring"
        assert result.score == 1.0

    async def test_malformed_response_fallback_no_match(self) -> None:
        from voice_eval_lab.judge.llm import LLMJudge

        bad_resp = MagicMock()
        bad_resp.json.return_value = {"content": [{"text": "just a plain string"}]}
        bad_resp.raise_for_status = MagicMock()

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=False),
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=bad_resp)
            mock_client_cls.return_value = mock_client

            judge = LLMJudge(anthropic_key="test-key")
            result = await judge.score(
                question="q",
                expected_keypoints=["very specific fact not in answer"],
                answer="completely unrelated reply",
            )

        assert result.mode == "substring"
        assert result.score == 0.0

    def test_no_key_raises_environment_error(self) -> None:
        from voice_eval_lab.judge.llm import LLMJudge

        with pytest.raises(EnvironmentError, match="API key"):
            LLMJudge(anthropic_key="", openai_key="")

    async def test_http_5xx_retries_then_falls_back(self) -> None:
        import httpx

        from voice_eval_lab.judge.llm import LLMJudge

        # Build a 503 response mock
        err_resp = MagicMock()
        err_resp.status_code = 503
        err_resp.reason_phrase = "Service Unavailable"
        http_error = httpx.HTTPStatusError("503", request=MagicMock(), response=err_resp)

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=False),
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=http_error)
            mock_client_cls.return_value = mock_client

            judge = LLMJudge(anthropic_key="test-key")
            result = await judge.score(
                question="q",
                expected_keypoints=["WAL"],
                answer="WAL is mentioned here",
            )

        # After retries exhausted → falls back to substring
        assert result.mode == "substring"


# ---------------------------------------------------------------------------
# make_judge factory
# ---------------------------------------------------------------------------


class TestMakeJudge:
    def test_no_keys_returns_substring_judge(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            # Remove any leaked keys from the test environment
            import os

            for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
                os.environ.pop(k, None)
            judge = make_judge(mode="auto")
        assert isinstance(judge, SubstringProxyJudge)

    def test_substring_mode_returns_substring_regardless_of_keys(self) -> None:
        with patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": "some-key"},
            clear=False,
        ):
            judge = make_judge(mode="substring")
        assert isinstance(judge, SubstringProxyJudge)

    def test_auto_with_anthropic_key_returns_llm_judge(self) -> None:
        from voice_eval_lab.judge.llm import LLMJudge

        with patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": "test-key"},
            clear=False,
        ):
            judge = make_judge(mode="auto")
        assert isinstance(judge, LLMJudge)

    def test_auto_with_openai_key_returns_llm_judge(self) -> None:
        from voice_eval_lab.judge.llm import LLMJudge

        with patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "sk-test"},
            clear=False,
        ):
            judge = make_judge(mode="auto")
        assert isinstance(judge, LLMJudge)

    def test_llm_mode_no_key_raises(self) -> None:
        import os

        with patch.dict("os.environ", {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(EnvironmentError, match="API key"):
                make_judge(mode="llm")

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown judge mode"):
            make_judge(mode="invalid")

    def test_llm_mode_with_key_returns_llm_judge(self) -> None:
        from voice_eval_lab.judge.llm import LLMJudge

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "k"}, clear=False):
            judge = make_judge(mode="llm")
        assert isinstance(judge, LLMJudge)


# ---------------------------------------------------------------------------
# Cohen's kappa formula
# ---------------------------------------------------------------------------


class TestCohensKappa:
    def test_perfect_agreement_balanced(self) -> None:
        """Perfect agreement on balanced labels → kappa = 1.0."""
        pairs = [
            JudgeAgreement("q1", human_score=1, llm_score=1),
            JudgeAgreement("q2", human_score=0, llm_score=0),
        ]
        assert cohens_kappa(pairs) == pytest.approx(1.0)

    def test_perfect_disagreement_balanced(self) -> None:
        """Perfect disagreement on balanced labels → kappa = -1.0."""
        pairs = [
            JudgeAgreement("q1", human_score=1, llm_score=0),
            JudgeAgreement("q2", human_score=0, llm_score=1),
        ]
        assert cohens_kappa(pairs) == pytest.approx(-1.0)

    def test_random_agreement_near_zero(self) -> None:
        """Chance-level agreement → kappa ≈ 0."""
        # 4 samples: human [1,1,0,0], llm [1,0,1,0]
        # p_o = 2/4 = 0.5
        # p_human_1=0.5, p_llm_1=0.5 → p_e = 0.5*0.5+0.5*0.5 = 0.5
        # kappa = (0.5 - 0.5) / (1 - 0.5) = 0
        pairs = [
            JudgeAgreement("q1", human_score=1, llm_score=1),
            JudgeAgreement("q2", human_score=1, llm_score=0),
            JudgeAgreement("q3", human_score=0, llm_score=1),
            JudgeAgreement("q4", human_score=0, llm_score=0),
        ]
        assert cohens_kappa(pairs) == pytest.approx(0.0)

    def test_empty_list_returns_zero(self) -> None:
        assert cohens_kappa([]) == 0.0

    def test_all_same_class_degenerate(self) -> None:
        """All labels identical → p_e=1.0; return 0.0 rather than divide-by-zero."""
        pairs = [
            JudgeAgreement("q1", human_score=1, llm_score=1),
            JudgeAgreement("q2", human_score=1, llm_score=1),
        ]
        assert cohens_kappa(pairs) == 0.0

    def test_partial_agreement(self) -> None:
        """3/4 agreement with skewed labels."""
        # human [1,1,1,0], llm [1,1,0,0]
        # TP=2, TN=1, FP=0, FN=1, N=4
        # p_o=3/4, p_human_1=3/4, p_llm_1=2/4
        # p_e = (3/4)*(2/4)+(1/4)*(2/4) = 6/16+2/16 = 8/16 = 0.5
        # kappa = (0.75 - 0.5)/(1-0.5) = 0.5
        pairs = [
            JudgeAgreement("q1", human_score=1, llm_score=1),
            JudgeAgreement("q2", human_score=1, llm_score=1),
            JudgeAgreement("q3", human_score=1, llm_score=0),
            JudgeAgreement("q4", human_score=0, llm_score=0),
        ]
        assert cohens_kappa(pairs) == pytest.approx(0.5)

    def test_judge_agreement_dataclass_fields(self) -> None:
        a = JudgeAgreement("id-1", human_score=1, llm_score=0)
        assert a.question_id == "id-1"
        assert a.human_score == 1
        assert a.llm_score == 0


# ---------------------------------------------------------------------------
# CLI --judge flag
# ---------------------------------------------------------------------------


class TestCLIJudgeFlag:
    def test_run_with_substring_judge(self, tmp_path: Path) -> None:
        """voice-eval run --judge substring should complete without error."""
        from typer.testing import CliRunner

        from voice_eval_lab.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app, ["run", "--judge", "substring", "--out", str(tmp_path / "report.md")]
        )
        assert result.exit_code == 0, result.output

    def test_run_with_invalid_judge_fails(self) -> None:
        """voice-eval run --judge bogus should exit non-zero."""
        from typer.testing import CliRunner

        from voice_eval_lab.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["run", "--judge", "bogus"])
        assert result.exit_code != 0

    def test_run_llm_judge_no_key_raises_clean_error(self) -> None:
        """--judge llm with no API key should exit non-zero with a clear message."""
        import os

        from typer.testing import CliRunner

        from voice_eval_lab.cli import app

        runner = CliRunner()
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
        result = runner.invoke(app, ["run", "--judge", "llm"], env=env)
        # Should fail (exit code != 0) with a clear error mentioning the keys.
        assert result.exit_code != 0
        # Typer surfaces the OSError via result.exception when the CLI hasn't caught it.
        message = (result.output or "") + str(result.exception or "")
        assert "ANTHROPIC_API_KEY" in message or "API key" in message or "key" in message.lower()
