from __future__ import annotations

import contextlib
import os
import threading
from typing import Any


_LOCK = threading.Lock()
_CONFIGURED = False
_INSTRUMENTED_APPS: set[int] = set()


class _NoopSpan:
    def set_attribute(self, *_args, **_kwargs) -> None:
        return None

    def set_attributes(self, *_args, **_kwargs) -> None:
        return None

    def add_event(self, *_args, **_kwargs) -> None:
        return None

    def record_exception(self, *_args, **_kwargs) -> None:
        return None

    def set_status(self, *_args, **_kwargs) -> None:
        return None


@contextlib.contextmanager
def _noop_span_cm():
    yield _NoopSpan()


def _enabled() -> bool:
    flag = os.environ.get("AGENTSCOPE_OTEL_ENABLED", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    return bool(
        os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    )


def _import_otel() -> dict[str, Any] | None:
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.propagate import extract, inject
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.trace import Status, StatusCode
    except ImportError:
        return None

    return {
        "trace": trace,
        "OTLPSpanExporter": OTLPSpanExporter,
        "FastAPIInstrumentor": FastAPIInstrumentor,
        "extract": extract,
        "inject": inject,
        "Resource": Resource,
        "TracerProvider": TracerProvider,
        "BatchSpanProcessor": BatchSpanProcessor,
        "Status": Status,
        "StatusCode": StatusCode,
    }


def _parse_headers(raw: str | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if not raw:
        return headers
    for item in raw.split(","):
        key, sep, value = item.partition("=")
        if sep and key.strip():
            headers[key.strip()] = value.strip()
    return headers


def _resolve_endpoint() -> str | None:
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if endpoint:
        return endpoint

    base = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if base:
        return base.rstrip("/") + "/v1/traces"

    if os.environ.get("AGENTSCOPE_OTEL_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}:
        return "http://127.0.0.1:4318/v1/traces"

    return None


def configure_telemetry(service_name: str, service_version: str = "dev") -> bool:
    global _CONFIGURED

    if not _enabled():
        return False

    otel = _import_otel()
    if otel is None:
        return False

    with _LOCK:
        if _CONFIGURED:
            return True

        endpoint = _resolve_endpoint()
        if endpoint is None:
            return False

        resource = otel["Resource"].create({
            "service.name": service_name,
            "service.version": service_version,
        })
        provider = otel["TracerProvider"](resource=resource)

        headers = _parse_headers(
            os.environ.get("OTEL_EXPORTER_OTLP_TRACES_HEADERS")
            or os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")
        )
        exporter_kwargs: dict[str, Any] = {"endpoint": endpoint}
        if headers:
            exporter_kwargs["headers"] = headers

        exporter = otel["OTLPSpanExporter"](**exporter_kwargs)
        provider.add_span_processor(otel["BatchSpanProcessor"](exporter))
        otel["trace"].set_tracer_provider(provider)
        _CONFIGURED = True
        return True


def instrument_fastapi(app) -> bool:
    if not _enabled():
        return False

    otel = _import_otel()
    if otel is None:
        return False

    with _LOCK:
        key = id(app)
        if key in _INSTRUMENTED_APPS:
            return True
        otel["FastAPIInstrumentor"].instrument_app(app)
        _INSTRUMENTED_APPS.add(key)
        return True


def get_tracer(name: str):
    otel = _import_otel()
    if otel is None:
        return None
    return otel["trace"].get_tracer(name)


def _clean_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    if not attributes:
        return {}

    cleaned: dict[str, Any] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (bool, int, float, str)):
            cleaned[key] = value
            continue
        if isinstance(value, (list, tuple)):
            seq = []
            for item in value:
                if item is None:
                    continue
                if isinstance(item, (bool, int, float, str)):
                    seq.append(item)
                else:
                    seq.append(str(item))
            if seq:
                cleaned[key] = seq
            continue
        cleaned[key] = str(value)
    return cleaned


def set_span_attributes(span, attributes: dict[str, Any] | None) -> None:
    cleaned = _clean_attributes(attributes)
    if cleaned and hasattr(span, "set_attributes"):
        span.set_attributes(cleaned)


def start_span(
    name: str,
    *,
    tracer_name: str = "agentscope",
    attributes: dict[str, Any] | None = None,
    context=None,
):
    if not _enabled():
        return _noop_span_cm()

    otel = _import_otel()
    if otel is None:
        return _noop_span_cm()

    tracer = otel["trace"].get_tracer(tracer_name)

    @contextlib.contextmanager
    def _span_cm():
        with tracer.start_as_current_span(name, context=context) as span:
            set_span_attributes(span, attributes)
            yield span

    return _span_cm()


def inject_trace_context(carrier: dict[str, str] | None = None) -> dict[str, str]:
    carrier = dict(carrier or {})
    if not _enabled():
        return carrier

    otel = _import_otel()
    if otel is None:
        return carrier

    otel["inject"](carrier)
    return carrier


def extract_trace_context(carrier: dict[str, str] | None):
    if not carrier or not _enabled():
        return None

    otel = _import_otel()
    if otel is None:
        return None

    return otel["extract"](carrier)


def current_trace_ids() -> dict[str, str | None]:
    otel = _import_otel()
    if otel is None:
        return {"trace_id": None, "span_id": None}

    span = otel["trace"].get_current_span()
    if span is None:
        return {"trace_id": None, "span_id": None}

    ctx = span.get_span_context()
    if not ctx or not getattr(ctx, "is_valid", False):
        return {"trace_id": None, "span_id": None}

    return {
        "trace_id": f"{ctx.trace_id:032x}",
        "span_id": f"{ctx.span_id:016x}",
    }


def mark_span_ok(span) -> None:
    otel = _import_otel()
    if otel is None or not hasattr(span, "set_status"):
        return
    span.set_status(otel["Status"](otel["StatusCode"].OK))


def mark_span_error(span, exc: Exception) -> None:
    if hasattr(span, "record_exception"):
        span.record_exception(exc)
    set_span_attributes(span, {
        "error.type": type(exc).__name__,
        "error.message": str(exc),
    })

    otel = _import_otel()
    if otel is None or not hasattr(span, "set_status"):
        return
    span.set_status(otel["Status"](otel["StatusCode"].ERROR, str(exc)))
