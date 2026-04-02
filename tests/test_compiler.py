import json
import os
import pytest
from agentscope.report.compiler import ReportCompiler, run_compiler
from agentscope.config import load_config


@pytest.fixture
def base_state():
    return {
        "run_id":          "test-compiler",
        "agent_folder":    "tests/fake_agent",
        "agent_type":      "tool_use",
        "turn_type":       "single",
        "config":          load_config().model_dump(),
        "ir_results":      None,
        "behavior_results":{"convergence": 1.0, "plan_success": 0.5},
        "geval_results":   {"scores": {"faithfulness": 0.88}},
        "cost_results":    {"cost_per_query": 0.021, "agent_model": "claude-sonnet-4-5"},
        "adversarial_results": {"attack_success_rate": 0.0},
        "synth_results":   None,
        "final_report":    None,
    }


def test_compiler_writes_file(base_state, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = ReportCompiler().run(base_state)
    path = tmp_path / "outputs" / "test-compiler_run_report.json"
    assert path.exists()


def test_compiler_report_structure(base_state, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = ReportCompiler().run(base_state)
    path = tmp_path / "outputs" / "test-compiler_run_report.json"
    with open(path) as f:
        report = json.load(f)

    assert report["run_id"]      == "test-compiler"
    assert report["agent_type"]  == "tool_use"
    assert report["turn_type"]   == "single"
    assert "eval_results"        in report
    assert "agentscope_cost"     in report


def test_compiler_eval_results_keys(base_state, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = ReportCompiler().run(base_state)
    er = result["final_report"]["eval_results"]
    assert "ir"          in er
    assert "behavior"    in er
    assert "geval"       in er
    assert "cost"        in er
    assert "adversarial" in er


def test_compiler_sets_final_report(base_state, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = ReportCompiler().run(base_state)
    assert result["final_report"] is not None


def test_run_compiler_function(base_state, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base_state["run_id"] = "test-compiler-fn"
    result = run_compiler(base_state)
    assert result["final_report"]["run_id"] == "test-compiler-fn"
