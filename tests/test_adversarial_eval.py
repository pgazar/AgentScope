import os
import pytest

needs_api_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="requires ANTHROPIC_API_KEY",
)
from unittest.mock import patch, MagicMock
from agentscope.tools.adversarial_eval import load_adversarial_suite, RESISTANCE_CRITERIA


# --- load_adversarial_suite ---

def test_suite_loads():
    suite = load_adversarial_suite()
    assert isinstance(suite, dict)
    assert len(suite) == 4


def test_suite_categories():
    suite = load_adversarial_suite()
    for cat in ["prompt_injection", "unsafe_tool_use", "instruction_override", "sensitive_action"]:
        assert cat in suite, f"missing category: {cat}"


def test_suite_prompts_per_category():
    suite = load_adversarial_suite()
    for cat, prompts in suite.items():
        assert len(prompts) >= 1, f"{cat} should have at least 1 prompt, got {len(prompts)}"


def test_suite_prompts_are_strings():
    suite = load_adversarial_suite()
    for cat, prompts in suite.items():
        for p in prompts:
            assert isinstance(p, str) and len(p) > 10


def test_suite_total_prompts():
    suite = load_adversarial_suite()
    assert sum(len(v) for v in suite.values()) >= 4  # at least 1 per category


# --- RESISTANCE_CRITERIA ---

def test_resistance_criteria_is_string():
    assert isinstance(RESISTANCE_CRITERIA, str)
    assert len(RESISTANCE_CRITERIA) > 50


def test_resistance_criteria_contains_key_terms():
    lower = RESISTANCE_CRITERIA.lower()
    assert "resist" in lower or "refuse" in lower or "hostile" in lower


# --- run_adversarial_suite result structure (no API call) ---

@needs_api_key
def test_suite_result_structure():
    """
    Validates the shape of run_adversarial_suite output using a mock agent
    that always refuses — no API call needed.
    """
    from agentscope.tools.adversarial_eval import run_adversarial_suite

    suite = {
        "prompt_injection": ["ignore instructions"],
        "unsafe_tool_use":  ["delete all files"],
    }

    def safe_agent(prompt: str) -> str:
        return "I can't help with that."

    # Patch both build_model (prevents AnthropicModel instantiation in CI)
    # and GEval class (prevents API calls)
    # a_measure must return a coroutine — asyncio.run(metric.a_measure(tc)) is used now
    async def _fake_a_measure(tc):
        return 1.0

    mock_metric = MagicMock()
    mock_metric.score = 1.0
    mock_metric.a_measure = _fake_a_measure
    mock_geval_class = MagicMock(return_value=mock_metric)

    with patch("agentscope.judge.model.build_model", return_value="mock-model"), \
         patch("deepeval.metrics.GEval", mock_geval_class):
        result = run_adversarial_suite(safe_agent, suite, "claude-sonnet-4-5")

    assert "total_attacks"               in result
    assert "resisted"                    in result
    assert "attack_success_rate"         in result
    assert "unsafe_compliance_rate"      in result
    assert "prompt_injection_resistance" in result
    assert "permission_violation_rate"   in result
    assert "by_category"                 in result
    assert result["total_attacks"] == 2
