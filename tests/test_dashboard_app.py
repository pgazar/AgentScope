from pathlib import Path

from agentscope.dashboard.app import run_evaluation


REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_AGENT = str(REPO_ROOT / "tests" / "fake_agent")


class DummyProgress:
    def __init__(self):
        self.calls = []

    def __call__(self, pct, desc=""):
        self.calls.append((pct, desc))


def test_dashboard_run_uses_queued_flow(monkeypatch):
    progress = DummyProgress()
    report = {
        "eval_results": {
            "ir": {},
            "behavior": {},
            "geval": {"scores": {}},
            "cost": {},
            "adversarial": {},
        }
    }

    monkeypatch.setattr("agentscope.dashboard.app.validate_run_request", lambda **kwargs: None)
    monkeypatch.setattr(
        "agentscope.dashboard.app.build_run_state",
        lambda **kwargs: {
            "run_id": "dash-run",
            "agent_folder": FAKE_AGENT,
            "agent_model": "claude-haiku-4-5-20251001",
            "eval_inputs": ["What is 2+2?"],
            "agent_type": "tool_use",
            "turn_type": "single",
            "active_tools": ["geval"],
        },
    )
    monkeypatch.setattr(
        "agentscope.dashboard.app.enqueue_run",
        lambda state, source="dashboard": {"run_id": state["run_id"], "status": "queued"},
    )

    statuses = iter([
        {"run_id": "dash-run", "status": "queued", "stage": "queued", "pct": 0.0},
        {"run_id": "dash-run", "status": "running", "stage": "running_agent", "pct": 0.2},
        {"run_id": "dash-run", "status": "completed", "stage": "done", "pct": 1.0},
    ])
    monkeypatch.setattr("agentscope.dashboard.app.load_run", lambda run_id: next(statuses))
    monkeypatch.setattr("agentscope.dashboard.app.load_report", lambda run_id: report)
    monkeypatch.setattr("agentscope.dashboard.app.time.sleep", lambda _: None)

    output = run_evaluation(
        agent_folder=FAKE_AGENT,
        agent_model="claude-haiku-4-5-20251001",
        kb_file=None,
        gt_file=None,
        eval_inputs_text="What is 2+2?",
        agent_type="tool_use",
        turn_type="single",
        progress=progress,
    )

    assert len(output) == 5
    assert progress.calls[0][1].startswith("Queued run dash-run")
    assert progress.calls[-1][1] == "Complete: dash-run"
