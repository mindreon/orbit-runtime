"""OpenTelemetry for the agent process and the Temporal worker."""

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor, SpanExporter

_configured = False


def configure_tracing(exporter: SpanExporter | None = None) -> TracerProvider | None:
    """Install a real TracerProvider once so AgentScope TracingMiddleware records spans.

    A second call is a no-op. Tests pass an in-memory exporter before the
    first agent reply.
    """

    global _configured
    if _configured:
        provider = otel_trace.get_tracer_provider()
        return provider if isinstance(provider, TracerProvider) else None
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter or ConsoleSpanExporter()))
    otel_trace.set_tracer_provider(provider)
    _configured = True
    return provider
