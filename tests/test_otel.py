from agentscope import otel


def test_otel_noop_helpers_are_safe(monkeypatch):
    monkeypatch.delenv("AGENTSCOPE_OTEL_ENABLED", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)

    assert otel.configure_telemetry("agentscope-test") is False
    assert otel.inject_trace_context({"run_id": "abc"}) == {"run_id": "abc"}
    assert otel.extract_trace_context({"traceparent": "abc"}) is None
    assert otel.current_trace_ids() == {"trace_id": None, "span_id": None}

    with otel.start_span("agentscope.test") as span:
        otel.set_span_attributes(span, {"agentscope.run_id": "abc"})
        otel.mark_span_ok(span)
