from __future__ import annotations

from collections import Counter


TRACE_EVENT_TYPES = [
    "llm_start",
    "llm_end",
    "tool_start",
    "tool_end",
    "retrieval",
    "handoff",
]


def _event_counts(events: list) -> dict[str, int]:
    counts = Counter(
        getattr(evt, "event_type", None)
        for evt in events
        if getattr(evt, "event_type", None)
    )
    return {name: counts.get(name, 0) for name in TRACE_EVENT_TYPES}


def summarize_trace(trace) -> dict:
    events = getattr(trace, "events", []) or []
    counts = _event_counts(events)
    issues: list[str] = []
    warnings: list[str] = []

    output = str(getattr(trace, "agent_output", "") or "")
    if output.startswith("[AgentRunner error:") or output.startswith("[AgentRunner timeout:"):
        issues.append("agent_execution_error")

    if not events:
        issues.append("missing_all_trace_events")

    if counts["llm_start"] != counts["llm_end"]:
        warnings.append("unbalanced_llm_events")

    if counts["tool_start"] != counts["tool_end"]:
        warnings.append("unbalanced_tool_events")

    if counts["llm_end"] > 0:
        completion_events = [
            evt for evt in events
            if getattr(evt, "event_type", None) == "llm_end"
        ]
        if not any(getattr(evt, "completion_tokens", 0) > 0 for evt in completion_events):
            warnings.append("missing_completion_tokens")

    status = "ok"
    if "agent_execution_error" in issues:
        status = "error"
    elif "missing_all_trace_events" in issues:
        status = "output_only"
    elif warnings:
        status = "partial"

    return {
        "run_id": getattr(trace, "run_id", ""),
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "event_counts": counts,
        "has_agent_output": bool(output),
        "has_trace_events": bool(events),
        "has_retrieval_events": counts["retrieval"] > 0,
        "has_token_counts": counts["llm_end"] > 0 and any(
            getattr(evt, "completion_tokens", 0) > 0
            for evt in events
            if getattr(evt, "event_type", None) == "llm_end"
        ),
        "has_latency_events": any(
            getattr(evt, "latency_ms", 0) > 0
            for evt in events
        ),
    }


def summarize_traces(traces: list) -> dict:
    per_trace = [summarize_trace(trace) for trace in traces]
    counts = Counter(summary["status"] for summary in per_trace)
    issues = sorted({issue for summary in per_trace for issue in summary["issues"]})
    warnings = sorted({warning for summary in per_trace for warning in summary["warnings"]})

    any_events = any(summary["has_trace_events"] for summary in per_trace)
    any_retrieval = any(summary["has_retrieval_events"] for summary in per_trace)
    any_tokens = any(summary["has_token_counts"] for summary in per_trace)
    any_latency = any(summary["has_latency_events"] for summary in per_trace)

    overall_status = "ok"
    if counts.get("error"):
        overall_status = "error"
    elif not any_events:
        overall_status = "output_only"
    elif counts.get("partial"):
        overall_status = "partial"

    return {
        "status": overall_status,
        "issues": issues,
        "warnings": warnings,
        "total_traces": len(traces),
        "status_counts": dict(counts),
        "event_coverage": {
            "has_any_events": any_events,
            "has_retrieval_events": any_retrieval,
            "has_token_counts": any_tokens,
            "has_latency_events": any_latency,
        },
        "scoreability": {
            "behavior": any_events,
            "ir": any_retrieval,
            "cost": any_tokens or any_latency,
        },
        "traces": per_trace,
    }
