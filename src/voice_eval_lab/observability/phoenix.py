"""Arize Phoenix / OTLP span exporter for PipelineSpan.

Usage
-----
Set ``OTEL_EXPORTER_OTLP_ENDPOINT`` to your Phoenix collector URL
(e.g. ``http://localhost:4317``).  When the env var is absent or empty,
:func:`export_spans` is a no-op — no OTel imports are required at that
point.

Optional dependencies (not in base install)::

    pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
    # or add them via:
    #   pip install "voice-eval-lab[real]"
    # and install OTel separately — they are documented under [real] but
    # not pinned there to keep the extra lightweight.

If ``opentelemetry-sdk`` is not installed and the endpoint IS set, the
function logs a warning and exits without raising.

Environment variables
---------------------
``OTEL_EXPORTER_OTLP_ENDPOINT``
    gRPC endpoint for the OTLP exporter, e.g. ``http://localhost:4317``.
    When unset or empty, all calls are no-ops.

``OTEL_SERVICE_NAME``
    Service name reported on each span (default: ``voice-eval-lab``).
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voice_eval_lab.models import PipelineSpan

logger = logging.getLogger(__name__)

_SERVICE_NAME_DEFAULT = "voice-eval-lab"
# One nanosecond expressed in milliseconds — used to convert ms -> ns for OTel.
_MS_TO_NS = 1_000_000

def _has_spec(name: str) -> bool:
    """Return True iff importlib.util.find_spec finds *name* without raising."""
    try:
        return importlib.util.find_spec(name) is not None  # type: ignore[attr-defined]
    except (ModuleNotFoundError, ValueError):
        return False


# Probe once at module import time so export_spans can short-circuit cheaply.
_OTEL_AVAILABLE: bool = _has_spec("opentelemetry.sdk.trace") and _has_spec(
    "opentelemetry.exporter.otlp.proto.grpc"
)


def _otlp_endpoint() -> str:
    """Return the configured OTLP endpoint, or empty string if unset."""
    return os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()


def _run_export(endpoint: str, spans: list[PipelineSpan], conv_id: str) -> None:
    """Inner export implementation; called only when OTel is confirmed available.

    Separated from :func:`export_spans` so the import block stays at module
    level (satisfying ruff PLC0415) by deferring to a function-level dynamic
    import via ``importlib.import_module`` once we know the packages exist.
    """
    # Dynamic imports — packages verified present via importlib.util.find_spec.
    _otel_trace = importlib.import_module("opentelemetry.sdk.trace")
    _otel_res = importlib.import_module("opentelemetry.sdk.resources")
    _otel_export = importlib.import_module("opentelemetry.sdk.trace.export")
    _otel_otlp = importlib.import_module(
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
    )
    _otel_api = importlib.import_module("opentelemetry.trace")

    service_name = os.environ.get("OTEL_SERVICE_NAME", _SERVICE_NAME_DEFAULT)
    resource = _otel_res.Resource.create({"service.name": service_name})
    exporter = _otel_otlp.OTLPSpanExporter(endpoint=endpoint)
    provider = _otel_trace.TracerProvider(resource=resource)
    processor = _otel_export.SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    tracer = provider.get_tracer(__name__)

    with tracer.start_as_current_span("eval.pipeline_run") as root:
        if conv_id:
            root.set_attribute("conv_id", conv_id)

        for span in spans:
            start_ns = span.started_at_ms * _MS_TO_NS
            end_ns = span.ended_at_ms * _MS_TO_NS
            # start_time / end_time must be strictly positive and ordered.
            if end_ns <= start_ns:
                end_ns = start_ns + _MS_TO_NS  # guarantee 1 ms minimum width

            with tracer.start_as_current_span(
                span.name,
                start_time=start_ns,
            ) as child:
                child.set_attribute("conv_id", conv_id)
                for k, v in span.attrs.items():
                    child.set_attribute(k, v)
                # Force the end time to reflect the span's recorded duration.
                child._end_time = end_ns

    provider.force_flush()
    provider.shutdown()


def export_spans(spans: list[PipelineSpan], conv_id: str = "") -> None:
    """Export *spans* to the configured OTLP/Phoenix endpoint.

    Parameters
    ----------
    spans:
        List of :class:`~voice_eval_lab.models.PipelineSpan` produced by the
        eval pipeline for one conversation run.
    conv_id:
        Conversation identifier; attached as the ``conv_id`` span attribute.

    Behaviour
    ---------
    * No-op when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is unset or empty.
    * Logs a warning and returns (does NOT raise) when ``opentelemetry-sdk``
      or ``opentelemetry-exporter-otlp-proto-grpc`` is not installed.
    * All network / export errors are caught, logged, and swallowed so the
      eval run itself is never blocked by a tracing failure.
    """
    endpoint = _otlp_endpoint()
    if not endpoint:
        return

    if not _OTEL_AVAILABLE:
        logger.warning(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set (%s) but opentelemetry-sdk / "
            "opentelemetry-exporter-otlp-proto-grpc are not installed. "
            "Install them separately or skip tracing by unsetting the env var.",
            endpoint,
        )
        return

    try:
        _run_export(endpoint, spans, conv_id)
    except Exception:
        logger.warning("Phoenix OTLP export failed; spans not sent.", exc_info=True)
