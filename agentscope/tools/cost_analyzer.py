import numpy as np
from agentscope.orchestrator.state import AgentState

MODEL_PRICING = {
    "claude-haiku-4-5-20251001": {"input": 0.80 / 1e6, "output": 4.00 / 1e6},  # cheapest
    "claude-sonnet-4-5":         {"input": 3.00 / 1e6, "output": 15.00 / 1e6},
    "claude-opus-4-6":           {"input": 15.00 / 1e6, "output": 75.00 / 1e6},
    "gpt-4o":                    {"input": 5.00 / 1e6,  "output": 15.00 / 1e6},
    "gpt-4o-mini":               {"input": 0.15 / 1e6,  "output": 0.60 / 1e6},
}


def compute_query_cost(traces: list, agent_model: str) -> float:
    """
    Measures cost of the EVALUATED agent's LLM calls using its model pricing.
    Uses agent_model from AgentState — not the judge model config.
    """
    pricing = MODEL_PRICING.get(agent_model, MODEL_PRICING["claude-sonnet-4-5"])
    input_tokens  = sum(e.prompt_tokens      for e in traces if hasattr(e, "prompt_tokens"))
    output_tokens = sum(e.completion_tokens  for e in traces if hasattr(e, "completion_tokens"))
    return input_tokens * pricing["input"] + output_tokens * pricing["output"]


def compute_latencies(traces: list) -> dict:
    """Latency values are populated by AgentRunner._normalize() on llm_end and tool_end events."""
    lats = [e.latency_ms for e in traces if hasattr(e, "latency_ms") and e.latency_ms > 0]
    return {
        "p50": float(np.percentile(lats, 50)) if lats else 0.0,
        "p95": float(np.percentile(lats, 95)) if lats else 0.0,
    }


def run(state: AgentState) -> AgentState:
    # Flatten all trace events across every AgentTrace run
    all_events = []
    for trace in state["traces"]:
        all_events.extend(trace.events if hasattr(trace, "events") else [])

    trace_diag = state.get("trace_diagnostics") or {}
    if not trace_diag.get("scoreability", {}).get("cost", False):
        state["cost_results"] = {
            "scoreable":         False,
            "reason":            "trace did not capture token counts or latency observations",
            "cost_per_query":    None,
            "cost_per_success":  None,
            "p50_latency_s":     None,
            "p95_latency_s":     None,
            "qc_index":          None,
            "agent_model":       state.get("agent_model", "claude-sonnet-4-5"),
        }
        return state

    agent_model = state.get("agent_model", "claude-sonnet-4-5")

    geval_scores = state.get("geval_results", {}).get("scores", {}) or {}
    numeric_scores = [v for v in geval_scores.values() if isinstance(v, (int, float))]
    geval_score = sum(numeric_scores) / max(len(numeric_scores), 1) if numeric_scores else 0.0

    task_success = state.get("behavior_results", {}).get("plan_success", 0.0) or 0.0

    cost_per_query = compute_query_cost(all_events, agent_model)
    # Avoid division by zero — if task never succeeded, cost-per-success is infinite
    cost_per_task  = cost_per_query / task_success if task_success > 0 else float("inf")

    latencies = compute_latencies(all_events)
    qc_index  = geval_score / cost_per_query if cost_per_query > 0 else 0.0

    state["cost_results"] = {
        "scoreable":         True,
        "cost_per_query":   round(cost_per_query, 5),
        "cost_per_success": round(cost_per_task, 5) if cost_per_task != float("inf") else None,
        "p50_latency_s":    round(latencies["p50"] / 1000, 3),
        "p95_latency_s":    round(latencies["p95"] / 1000, 3),
        "qc_index":         round(qc_index, 2),
        "agent_model":      agent_model,
    }
    return state
