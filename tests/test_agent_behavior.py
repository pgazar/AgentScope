import pytest
from agentscope.tools.agent_behavior import (
    ghost_action_rate,
    tool_selection_accuracy,
    step_budget_efficiency,
    convergence,
    step_match,
    load_permission_schema,
)
from agentscope.runner import AgentTrace, TraceEvent


def _tool_event(name: str) -> TraceEvent:
    return TraceEvent(event_type="tool_start", tool_name=name)


# --- tool_selection_accuracy ---

def test_tool_accuracy_no_reference():
    assert tool_selection_accuracy([], []) is None


def test_tool_accuracy_perfect():
    events = [_tool_event("search"), _tool_event("calculator")]
    assert tool_selection_accuracy(events, ["search", "calculator"]) == 1.0


def test_tool_accuracy_partial():
    events = [_tool_event("search"), _tool_event("unknown")]
    assert tool_selection_accuracy(events, ["search"]) == pytest.approx(0.5)


def test_tool_accuracy_no_calls():
    assert tool_selection_accuracy([], ["search"]) == 0.0


# --- step_budget_efficiency ---

def test_step_budget_under():
    assert step_budget_efficiency(5, 10) == 1.0


def test_step_budget_over():
    assert step_budget_efficiency(15, 10) == pytest.approx(10 / 15)


def test_step_budget_zero_steps():
    assert step_budget_efficiency(0, 10) == 1.0


def test_step_budget_exact():
    assert step_budget_efficiency(10, 10) == 1.0


# --- convergence ---

def test_convergence_within_budget():
    events = [_tool_event("a"), _tool_event("b")]
    assert convergence(events, max_steps=5) == 1.0


def test_convergence_exceeded():
    events = [_tool_event(f"t{i}") for i in range(6)]
    assert convergence(events, max_steps=5) == 0.0


def test_convergence_empty():
    assert convergence([], max_steps=5) == 1.0


# --- step_match ---

def test_step_match_no_reference():
    result = step_match([], [])
    assert result == {"exact": None, "precision": None, "recall": None}


def test_step_match_perfect():
    result = step_match(["a", "b"], ["a", "b"])
    assert result["exact"] is True
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)


def test_step_match_different():
    result = step_match(["a", "c"], ["a", "b"])
    assert result["exact"] is False


def test_step_match_unordered():
    result = step_match(["b", "a"], ["a", "b"], ordered=False)
    assert result["precision"] == pytest.approx(1.0)


# --- load_permission_schema ---

def test_permission_schema_loads():
    schema = load_permission_schema()
    assert "rag_retrieve" in schema
    assert "send_email" in schema
    assert schema["send_email"]["allowed"] is False
    assert schema["rag_retrieve"]["allowed"] is True


def test_ghost_action_rate_scores_per_trace():
    traces = [
        AgentTrace(
            run_id="1",
            agent_input="q1",
            agent_output="I searched the docs and found the answer.",
            events=[],
        ),
        AgentTrace(
            run_id="2",
            agent_input="q2",
            agent_output="I searched the docs and found the answer.",
            events=[TraceEvent(event_type="tool_start", tool_name="search")],
        ),
    ]
    result = ghost_action_rate(traces)
    assert result["ghost_count"] == 1
    assert result["total_checked"] == 2
    assert result["ghost_action_rate"] == pytest.approx(0.5)
