import pytest
from agentscope.tools.cost_analyzer import compute_query_cost, compute_latencies, MODEL_PRICING
from agentscope.runner import TraceEvent


def _llm_events(prompt_tokens: int, completion_tokens: int, latency_ms: float):
    return [
        TraceEvent(event_type="llm_start", prompt_tokens=prompt_tokens),
        TraceEvent(event_type="llm_end",   completion_tokens=completion_tokens, latency_ms=latency_ms),
    ]


# --- MODEL_PRICING ---

def test_pricing_keys_present():
    assert "claude-sonnet-4-5" in MODEL_PRICING
    assert "claude-opus-4-6"   in MODEL_PRICING
    assert "gpt-4o"            in MODEL_PRICING
    assert "gpt-4o-mini"       in MODEL_PRICING


def test_pricing_has_input_output():
    for model, pricing in MODEL_PRICING.items():
        assert "input"  in pricing, f"{model} missing input price"
        assert "output" in pricing, f"{model} missing output price"


# --- compute_query_cost ---

def test_cost_exact():
    traces = _llm_events(800, 300, 1500.0)
    pricing = MODEL_PRICING["claude-sonnet-4-5"]
    expected = 800 * pricing["input"] + 300 * pricing["output"]
    assert abs(compute_query_cost(traces, "claude-sonnet-4-5") - expected) < 1e-8


def test_cost_two_calls():
    traces = _llm_events(800, 300, 1000.0) + _llm_events(400, 150, 800.0)
    pricing = MODEL_PRICING["claude-sonnet-4-5"]
    expected = 1200 * pricing["input"] + 450 * pricing["output"]
    assert abs(compute_query_cost(traces, "claude-sonnet-4-5") - expected) < 1e-8


def test_cost_unknown_model_fallback():
    traces = _llm_events(800, 300, 1000.0)
    cost_known   = compute_query_cost(traces, "claude-sonnet-4-5")
    cost_unknown = compute_query_cost(traces, "some-unknown-model")
    assert cost_known == cost_unknown


def test_cost_zero_tokens():
    traces = _llm_events(0, 0, 100.0)
    assert compute_query_cost(traces, "claude-sonnet-4-5") == 0.0


def test_cost_empty_traces():
    assert compute_query_cost([], "claude-sonnet-4-5") == 0.0


# --- compute_latencies ---

def test_latencies_basic():
    traces = _llm_events(100, 50, 1500.0) + _llm_events(100, 50, 800.0)
    lats = compute_latencies(traces)
    assert lats["p50"] > 0
    assert lats["p95"] >= lats["p50"]


def test_latencies_empty():
    lats = compute_latencies([])
    assert lats["p50"] == 0.0
    assert lats["p95"] == 0.0


def test_latencies_single():
    traces = _llm_events(100, 50, 2000.0)
    lats = compute_latencies(traces)
    assert lats["p50"] == pytest.approx(2000.0)
    assert lats["p95"] == pytest.approx(2000.0)
