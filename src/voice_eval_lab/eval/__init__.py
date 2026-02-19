"""Eval harness: metrics over a golden-set + reference pipeline."""

from voice_eval_lab.eval.golden import default_golden_set
from voice_eval_lab.eval.metrics import (
    ConversationScore,
    EvalReport,
    barge_in_latency_p95_ms,
    barge_in_success_rate,
    endpointing_accuracy,
    false_trigger_rate,
    llm_decisiveness,
    render_report,
    render_report_html,
    response_faithfulness,
    score_conversation,
    score_run,
    transcription_wer,
    tts_first_byte_jitter_ms,
    turn_latency_stats,
)

__all__ = [
    "ConversationScore",
    "EvalReport",
    "barge_in_latency_p95_ms",
    "barge_in_success_rate",
    "default_golden_set",
    "endpointing_accuracy",
    "false_trigger_rate",
    "llm_decisiveness",
    "render_report",
    "render_report_html",
    "response_faithfulness",
    "score_conversation",
    "score_run",
    "transcription_wer",
    "tts_first_byte_jitter_ms",
    "turn_latency_stats",
]
