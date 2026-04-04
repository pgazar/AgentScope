from pathlib import Path

from agentscope.runner import AgentRunner


REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_AGENT = str(REPO_ROOT / "tests" / "fake_agent")


def test_validate_agent_folder_finds_entrypoint():
    result = AgentRunner.validate_agent_folder(FAKE_AGENT)
    assert result["ok"] is True
    assert result["entrypoint"].endswith("main.py")


def test_runner_subprocess_executes_fake_agent():
    runner = AgentRunner(FAKE_AGENT, agent_model="claude-sonnet-4-5", mode="subprocess", timeout_s=10)
    trace = runner.run("What is 2+2?", run_id="runner-subprocess")

    assert "42" in trace.agent_output
    assert trace.diagnostics["runner_mode"] == "subprocess"
    assert trace.diagnostics["status"] == "output_only"


def test_runner_inprocess_executes_fake_agent():
    runner = AgentRunner(FAKE_AGENT, agent_model="claude-sonnet-4-5", mode="inproc", timeout_s=10)
    trace = runner.run("What is 2+2?", run_id="runner-inproc")

    assert "42" in trace.agent_output
    assert trace.diagnostics["runner_mode"] == "inproc"
    assert trace.diagnostics["status"] == "output_only"
