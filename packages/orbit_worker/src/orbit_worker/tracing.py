"""OpenTelemetry for the agent process and the Temporal worker.

Nothing is exported by default. AgentScope's TracingMiddleware puts prompts,
replies, tool arguments, and tool results into span attributes, and a console
exporter would write them to process stdout. Every exporter that is installed
goes through ``RedactingSpanExporter``.
"""

from collections.abc import Mapping, Sequence

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import Event, ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as gen_ai
from opentelemetry.trace import Link, Status
from opentelemetry.util.types import AttributeValue

from orbit_worker.secrets import redact_text

# Conversation content. These are dropped, not redacted: redaction removes
# secrets but would still export the conversation itself.
CONTENT_ATTRIBUTES = frozenset(
    {
        gen_ai.GEN_AI_INPUT_MESSAGES,
        gen_ai.GEN_AI_OUTPUT_MESSAGES,
        gen_ai.GEN_AI_SYSTEM_INSTRUCTIONS,
        gen_ai.GEN_AI_TOOL_CALL_ARGUMENTS,
        gen_ai.GEN_AI_TOOL_CALL_RESULT,
        gen_ai.GEN_AI_PROMPT,
        gen_ai.GEN_AI_COMPLETION,
    }
)

_configured = False


def configure_tracing(exporter: SpanExporter | None = None) -> TracerProvider | None:
    """Install a TracerProvider once, only when an exporter is given.

    Without one the no-op provider stays, so TracingMiddleware builds no span
    attributes at all. A second call is a no-op. Tests pass an in-memory
    exporter before the first agent reply.
    """

    global _configured
    if _configured:
        provider = otel_trace.get_tracer_provider()
        return provider if isinstance(provider, TracerProvider) else None
    if exporter is None:
        return None
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(RedactingSpanExporter(exporter)))
    otel_trace.set_tracer_provider(provider)
    _configured = True
    return provider


class RedactingSpanExporter(SpanExporter):
    """Drops content attributes and redacts every other string before export.

    Strings go through ``redact_text``, the redaction outgoing events use:
    attribute values, event attributes (``exception.message`` and
    ``exception.stacktrace``), link attributes, span names, and status text.
    """

    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._inner.export([_scrub(span) for span in spans])

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)


def _scrub(span: ReadableSpan) -> ReadableSpan:
    status = span.status
    if status.description:
        status = Status(status.status_code, redact_text(status.description))
    return ReadableSpan(
        name=redact_text(span.name),
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=_attributes(span.attributes),
        events=[
            Event(event.name, _attributes(event.attributes), event.timestamp)
            for event in span.events
        ],
        links=[Link(link.context, _attributes(link.attributes)) for link in span.links],
        kind=span.kind,
        status=status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


def _attributes(
    attributes: Mapping[str, AttributeValue] | None,
) -> dict[str, AttributeValue]:
    return {
        key: _value(value)
        for key, value in (attributes or {}).items()
        if key not in CONTENT_ATTRIBUTES
    }


def _value(value: AttributeValue) -> AttributeValue:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Sequence) and value and isinstance(value[0], str):
        return [redact_text(item) for item in value]  # type: ignore[union-attr]
    return value
