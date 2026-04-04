from pathlib import Path

from agentscope.job_queue import build_run_state, enqueue_run, recover_incomplete_runs
from agentscope.run_store import create_run, load_run, load_state, save_state


REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_AGENT = str(REPO_ROOT / "tests" / "fake_agent")


def _write_config(tmp_path: Path):
    (tmp_path / "config.yaml").write_text(
        "judge:\n"
        "  backend: claude\n"
        "  model: claude-haiku-4-5-20251001\n"
        "  temperature: 0\n"
        "  max_tokens: 512\n"
        "eval:\n"
        "  k: 5\n"
        "  max_steps: 10\n"
        "  ir_green: 0.7\n"
        "  ir_orange: 0.4\n"
        "  agent_green: 0.8\n"
        "  agent_orange: 0.6\n"
        "  geval_green: 0.8\n"
        "  geval_orange: 0.6\n"
        "  hallu_green: 0.1\n"
        "  hallu_orange: 0.25\n"
        "  cost_green: 0.02\n"
        "  cost_orange: 0.05\n"
        "  latency_green: 2.0\n"
        "  latency_orange: 5.0\n"
        "  max_synth_pairs: 50\n"
        "  max_geval_responses: 1\n"
        "  eval_budget_usd: 2.0\n"
    )


def test_enqueue_run_persists_state_and_record(tmp_path, monkeypatch):
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    launched = []

    def fake_launch(run_id: str):
        launched.append(run_id)
        return 999

    monkeypatch.setattr("agentscope.job_queue.launch_worker", fake_launch)

    state = build_run_state(
        run_id="queue-test",
        agent_folder=FAKE_AGENT,
        agent_model="claude-haiku-4-5-20251001",
        eval_inputs=["What is 2+2?"],
        agent_type="tool_use",
        turn_type="single",
    )
    record = enqueue_run(state, source="test")

    assert record["run_id"] == "queue-test"
    assert load_state("queue-test")["run_id"] == "queue-test"
    assert launched == ["queue-test"]


def test_recover_incomplete_runs_relaunches_stale_run(tmp_path, monkeypatch):
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    state = {"run_id": "recover-test", "agent_folder": FAKE_AGENT}
    save_state("recover-test", state)
    create_run(
        "recover-test",
        {
            "status": "running",
            "agent_folder": FAKE_AGENT,
            "agent_type": "tool_use",
            "turn_type": "single",
            "report_path": "outputs/recover-test_run_report.json",
            "state_path": "outputs/runs/recover-test.state.json",
            "worker_pid": 123456,
        },
    )

    relaunched = []

    monkeypatch.setattr("agentscope.job_queue.worker_is_alive", lambda pid: False)

    def fake_launch(run_id: str):
        relaunched.append(run_id)
        return 777

    monkeypatch.setattr("agentscope.job_queue.launch_worker", fake_launch)

    recovered = recover_incomplete_runs()

    assert recovered == ["recover-test"]
    assert relaunched == ["recover-test"]
    assert load_run("recover-test")["stage"] == "recovered"
