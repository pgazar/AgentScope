import pytest
from agentscope.judge.criteria import (
    SINGLE_TURN_CRITERIA,
    MULTI_TURN_CRITERIA,
    PLAN_SUCCESS_CRITERIA,
    HELPFULNESS_CRITERIA,
    SAFETY_CRITERIA,
)
from agentscope.judge.variance import ScoredMetric, measure_calibration_drift
from agentscope.tools.geval_tool import build_single_turn_metrics, _extract_responses
from agentscope.runner import AgentTrace, TraceEvent


# --- criteria keys ---

def test_single_turn_criteria_keys():
    assert "task_completion" in SINGLE_TURN_CRITERIA
    assert "faithfulness"    in SINGLE_TURN_CRITERIA
    assert "hallucination"   in SINGLE_TURN_CRITERIA
    assert "citation_acc"    in SINGLE_TURN_CRITERIA


def test_multi_turn_criteria_keys():
    assert "conversation_completeness" in MULTI_TURN_CRITERIA
    assert "turn_relevancy"            in MULTI_TURN_CRITERIA
    assert "knowledge_retention"       in MULTI_TURN_CRITERIA


def test_plan_success_criteria_keys():
    assert "plan_success"          in PLAN_SUCCESS_CRITERIA
    assert "argument_correctness"  in PLAN_SUCCESS_CRITERIA
    assert "handoff_correctness"   in PLAN_SUCCESS_CRITERIA


def test_helpfulness_and_safety_are_strings():
    assert isinstance(HELPFULNESS_CRITERIA, str)
    assert isinstance(SAFETY_CRITERIA, str)


# --- ScoredMetric ---

def test_scored_metric_fields():
    sm = ScoredMetric(
        name="test_metric",
        criteria="some criteria",
        test_cases=[],
        primary_scores=[0.8, 0.9],
        primary_model="claude-sonnet-4-5",
    )
    assert sm.name == "test_metric"
    assert sm.primary_scores == [0.8, 0.9]
    assert sm.primary_model == "claude-sonnet-4-5"


# --- calibration drift ---

def test_drift_no_baseline():
    result = measure_calibration_drift("faithfulness", [0.8, 0.9], [])
    assert result["drift_measured"] is False
    assert "reason" in result


def test_drift_with_identical_baseline():
    result = measure_calibration_drift("faithfulness", [0.8, 0.9, 0.85], [0.8, 0.9, 0.85])
    assert result["drift_measured"] is True
    assert "kl_divergence" in result
    assert result["kl_divergence"] >= 0


def test_drift_flagged_when_large():
    # Very different distributions should flag drift
    result = measure_calibration_drift("hallucination", [0.0] * 10, [1.0] * 10)
    assert result["drift_measured"] is True


# --- build_single_turn_metrics ---

def test_build_single_turn_metrics_count():
    metrics = build_single_turn_metrics("claude-sonnet-4-5")
    assert len(metrics) == 6


def test_build_single_turn_metrics_names():
    metrics = build_single_turn_metrics("claude-sonnet-4-5")
    names = {m.name for m in metrics}
    assert "task_completion" in names
    assert "safety"          in names
    assert "hallucination"   in names


# --- _extract_responses ---

def test_extract_responses_basic():
    trace = AgentTrace(
        run_id="r1",
        agent_input="What is X?",
        agent_output="X is 42.",
        events=[],
    )
    responses = _extract_responses([trace])
    assert len(responses) == 1
    assert responses[0]["input"]  == "What is X?"
    assert responses[0]["output"] == "X is 42."
    assert responses[0]["context"] == []


def test_extract_responses_with_retrieval():
    evt = TraceEvent(event_type="retrieval", retrieval_docs=["doc1", "doc2"])
    trace = AgentTrace(
        run_id="r2",
        agent_input="query",
        agent_output="answer",
        events=[evt],
    )
    responses = _extract_responses([trace])
    assert responses[0]["context"] == ["doc1", "doc2"]


def test_extract_responses_empty():
    assert _extract_responses([]) == []
