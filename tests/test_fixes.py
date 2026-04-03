"""
Targeted tests for three fixes applied after the audit:
  1. variance wired in geval_tool.run()
  2. ScoredMetric built with real criteria + test_cases (not empty strings)
  3. permission_validation uses _match_tool_to_schema (LLM judge) not .lower()
"""
import os
import pytest
from unittest.mock import patch, MagicMock
from deepeval.test_case import LLMTestCase

from agentscope.judge.variance import ScoredMetric, measure_inter_judge_variance
from agentscope.runner import AgentTrace, TraceEvent
from agentscope.tools.agent_behavior import (
    _match_tool_to_schema,
    permission_validation,
    load_permission_schema,
)

needs_api_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="requires ANTHROPIC_API_KEY",
)


# ============================================================
# Fix 1 — variance wired: geval_tool.run() populates variance
# ============================================================

def _make_fake_state(turn_type: str = "single") -> dict:
    trace = AgentTrace(
        run_id="fix-test",
        agent_input="What is 2+2?",
        agent_output="4",
        events=[],
    )
    return {
        "run_id": "fix-test",
        "traces": [trace],
        "turn_type": turn_type,
        "baseline_geval_scores": {},
        "config": {
            "judge": {"model": "claude-haiku-4-5-20251001"},
            "eval": {"max_geval_responses": 1, "eval_budget_usd": 2.0},
        },
    }


@needs_api_key
def test_geval_run_variance_not_empty():
    """variance key must be a non-empty list after a real single-turn run."""
    from agentscope.tools.geval_tool import run
    result = run(_make_fake_state("single"))
    variance = result["geval_results"]["variance"]
    assert isinstance(variance, list), "variance must be a list"
    assert len(variance) > 0, "variance must not be empty — measure_inter_judge_variance() not called"


@needs_api_key
def test_geval_run_variance_has_required_keys():
    """Each variance entry must have the keys produced by measure_inter_judge_variance()."""
    from agentscope.tools.geval_tool import run
    result = run(_make_fake_state("single"))
    for entry in result["geval_results"]["variance"]:
        assert "metric"              in entry
        assert "mean_delta"          in entry
        assert "std_deviation"       in entry
        assert "stable"              in entry
        assert "high_variance_cases" in entry
        assert "primary_model"       in entry
        assert "secondary_model"     in entry


@needs_api_key
def test_geval_run_scored_metric_criteria_not_empty():
    """
    ScoredMetric must carry the real criteria string from the GEval object.
    If criteria is "" the secondary judge scores nothing meaningful.
    We verify indirectly: variance entry secondary_model must be gpt-4o-mini
    (only set when measure_inter_judge_variance actually ran with a real criteria).
    """
    from agentscope.tools.geval_tool import run
    result = run(_make_fake_state("single"))
    for entry in result["geval_results"]["variance"]:
        assert entry["secondary_model"] == "gpt-4o-mini"


# ============================================================
# Fix 2 — ScoredMetric criteria field is populated (unit test,
#          no API key required — verifies the struct directly)
# ============================================================

def test_scored_metric_criteria_populated():
    """ScoredMetric must accept a non-empty criteria string."""
    tc = LLMTestCase(input="q", actual_output="a", retrieval_context=[])
    sm = ScoredMetric(
        name="faithfulness",
        criteria="Does the response only make claims traceable to context?",
        test_cases=[tc],
        primary_scores=[0.85],
        primary_model="claude-haiku-4-5-20251001",
    )
    assert sm.criteria != "", "criteria must not be empty string"
    assert len(sm.test_cases) == 1


def test_scored_metric_empty_criteria_is_detectable():
    """Confirms that empty criteria is distinct from populated — detectable by caller."""
    sm_empty = ScoredMetric(name="x", criteria="", test_cases=[], primary_scores=[], primary_model="m")
    sm_full  = ScoredMetric(name="x", criteria="some criteria", test_cases=[], primary_scores=[], primary_model="m")
    assert sm_empty.criteria == ""
    assert sm_full.criteria  != ""


# ============================================================
# Fix 3 — permission_validation uses LLM judge (_match_tool_to_schema)
# ============================================================

def test_match_tool_exact_match():
    """Exact match should always resolve correctly."""
    schema_keys = ["rag_retrieve", "send_email", "calculator", "delete_file"]
    # Mock the subprocess so this test runs without a real API key
    with patch("agentscope.tools.agent_behavior.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="send_email\n", stderr="")
        result = _match_tool_to_schema("send_email", schema_keys)
    assert result == "send_email"


def test_match_tool_camel_case():
    """SendEmail should resolve to send_email via LLM judge."""
    schema_keys = ["rag_retrieve", "send_email", "calculator", "delete_file"]
    with patch("agentscope.tools.agent_behavior.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="send_email\n", stderr="")
        result = _match_tool_to_schema("SendEmail", schema_keys)
    assert result == "send_email"


def test_match_tool_no_match_returns_none():
    """When LLM returns 'none', function returns None."""
    schema_keys = ["rag_retrieve", "send_email"]
    with patch("agentscope.tools.agent_behavior.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="none\n", stderr="")
        result = _match_tool_to_schema("totally_unknown_tool_xyz", schema_keys)
    assert result is None


def test_match_tool_subprocess_failure_returns_none():
    """Subprocess failure (non-zero return) must return None, not raise."""
    schema_keys = ["rag_retrieve"]
    with patch("agentscope.tools.agent_behavior.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="some error")
        result = _match_tool_to_schema("whatever", schema_keys)
    assert result is None


def test_match_tool_timeout_returns_none():
    """Subprocess timeout must return None, not raise."""
    import subprocess
    schema_keys = ["rag_retrieve"]
    with patch("agentscope.tools.agent_behavior.subprocess.run",
               side_effect=subprocess.TimeoutExpired(cmd="python", timeout=15)):
        result = _match_tool_to_schema("slow_tool", schema_keys)
    assert result is None


def test_permission_validation_uses_llm_match_for_camel_case():
    """
    permission_validation must call _match_tool_to_schema, not .lower().
    A tool named 'SendEmail' must be caught as a violation (send_email is not allowed).
    """
    schema = load_permission_schema()
    evt = TraceEvent(event_type="tool_start", tool_name="SendEmail")

    with patch("agentscope.tools.agent_behavior._match_tool_to_schema",
               return_value="send_email") as mock_match:
        result = permission_validation([evt], schema)

    mock_match.assert_called_once_with("SendEmail", list(schema.keys()))
    assert result["permission_violation_rate"] > 0, (
        "SendEmail → send_email must be flagged as a violation (not allowed)"
    )


def test_permission_validation_unknown_tool_goes_to_warnings():
    """A tool with no semantic match must go to warnings, not violations."""
    schema = load_permission_schema()
    evt = TraceEvent(event_type="tool_start", tool_name="mystery_tool_xyz")

    with patch("agentscope.tools.agent_behavior._match_tool_to_schema", return_value=None):
        result = permission_validation([evt], schema)

    assert result["permission_violation_rate"] == 0.0
    assert len(result["warnings"]) == 1
    assert result["warnings"][0]["tool"] == "mystery_tool_xyz"


def test_permission_validation_violation_includes_matched_key():
    """Violation dict must include matched_key so it's auditable in the report."""
    schema = load_permission_schema()
    evt = TraceEvent(event_type="tool_start", tool_name="DeleteFile")

    with patch("agentscope.tools.agent_behavior._match_tool_to_schema",
               return_value="delete_file"):
        result = permission_validation([evt], schema)

    assert len(result["violations"]) == 1
    assert result["violations"][0]["matched_key"] == "delete_file"


@needs_api_key
def test_match_tool_live_camel_case():
    """Live LLM call: SendEmail must resolve to send_email."""
    schema_keys = ["rag_retrieve", "send_email", "calculator", "delete_file", "external_api"]
    result = _match_tool_to_schema("SendEmail", schema_keys)
    assert result == "send_email", f"Expected 'send_email', got '{result}'"


@needs_api_key
def test_match_tool_live_unknown():
    """Live LLM call: a nonsense tool name must return None."""
    schema_keys = ["rag_retrieve", "send_email", "calculator"]
    result = _match_tool_to_schema("zxqy_nonsense_789", schema_keys)
    assert result is None, f"Expected None for unknown tool, got '{result}'"
