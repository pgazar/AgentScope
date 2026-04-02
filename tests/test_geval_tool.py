import os
import pytest
from unittest.mock import patch, MagicMock
from agentscope.judge.criteria import (
    SINGLE_TURN_CRITERIA,
    MULTI_TURN_CRITERIA,
    PLAN_SUCCESS_CRITERIA,
    HELPFULNESS_CRITERIA,
    SAFETY_CRITERIA,
)
from agentscope.judge.variance import ScoredMetric, measure_calibration_drift
from agentscope.tools.geval_tool import _extract_responses
from agentscope.runner import AgentTrace, TraceEvent

needs_api_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="requires ANTHROPIC_API_KEY",
)


# --- criteria keys ---

def test_single_turn_criteria_keys():
    for key in ["task_completion", "faithfulness", "hallucination", "citation_acc"]:
        assert key in SINGLE_TURN_CRITERIA
        assert isinstance(SINGLE_TURN_CRITERIA[key], str)
        assert len(SINGLE_TURN_CRITERIA[key]) > 20


def test_multi_turn_criteria_keys():
    for key in ["conversation_completeness", "turn_relevancy", "knowledge_retention"]:
        assert key in MULTI_TURN_CRITERIA


def test_plan_success_criteria_keys():
    for key in ["plan_success", "argument_correctness", "handoff_correctness"]:
        assert key in PLAN_SUCCESS_CRITERIA


def test_helpfulness_and_safety_are_strings():
    assert isinstance(HELPFULNESS_CRITERIA, str)
    assert isinstance(SAFETY_CRITERIA, str)


# --- ScoredMetric dataclass ---

def test_scored_metric_fields():
    sm = ScoredMetric(
        name="faithfulness",
        criteria="test criteria string",
        test_cases=[],
        primary_scores=[0.8, 0.9],
        primary_model="claude-sonnet-4-5",
    )
    assert sm.name == "faithfulness"
    assert sm.primary_scores == [0.8, 0.9]
    assert sm.criteria == "test criteria string"


# --- calibration drift ---

def test_drift_no_baseline():
    result = measure_calibration_drift("faithfulness", [0.8, 0.9], [])
    assert result["drift_measured"] is False
    assert "reason" in result


def test_drift_identical_distributions():
    scores = [0.8, 0.9, 0.85, 0.7, 0.95]
    result = measure_calibration_drift("faithfulness", scores, scores)
    assert result["drift_measured"] is True
    assert result["kl_divergence"] >= 0
    assert result["drift_flagged"] is False


def test_drift_very_different_distributions():
    current  = [0.9, 0.95, 0.9, 0.95, 0.9]
    baseline = [0.1, 0.05, 0.1, 0.05, 0.1]
    result = measure_calibration_drift("hallucination", current, baseline)
    assert result["drift_measured"] is True
    assert result["kl_divergence"] > 0


# --- _extract_responses ---

def test_extract_responses_basic():
    trace = AgentTrace(
        run_id="test",
        agent_input="What is 2+2?",
        agent_output="4",
        events=[],
    )
    responses = _extract_responses([trace])
    assert len(responses) == 1
    assert responses[0]["input"] == "What is 2+2?"
    assert responses[0]["output"] == "4"
    assert responses[0]["context"] == []


def test_extract_responses_with_retrieval():
    evt = TraceEvent(event_type="retrieval", retrieval_docs=["doc1", "doc2"])
    trace = AgentTrace(
        run_id="test",
        agent_input="query",
        agent_output="answer",
        events=[evt],
    )
    responses = _extract_responses([trace])
    assert responses[0]["context"] == ["doc1", "doc2"]


def test_extract_responses_empty():
    assert _extract_responses([]) == []


# --- build_single_turn_metrics (requires API key to instantiate AnthropicModel) ---

@needs_api_key
def test_build_single_turn_metrics_count():
    from agentscope.tools.geval_tool import build_single_turn_metrics
    metrics = build_single_turn_metrics("claude-sonnet-4-5")
    assert len(metrics) == 6


@needs_api_key
def test_build_single_turn_metrics_names():
    from agentscope.tools.geval_tool import build_single_turn_metrics
    metrics = build_single_turn_metrics("claude-sonnet-4-5")
    names = {m.name for m in metrics}
    assert "task_completion" in names
    assert "safety"          in names
    assert "hallucination"   in names
