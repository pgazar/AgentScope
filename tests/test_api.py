import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from agentscope.api.main import app
from agentscope.run_store import create_run, load_run, load_state, report_path, save_state

client = TestClient(app, raise_server_exceptions=False)
REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_AGENT = str(REPO_ROOT / "tests" / "fake_agent")


def _stub_enqueue_run(monkeypatch):
    from agentscope.api import main as api_main

    def fake_enqueue(state, source="api"):
        save_state(state["run_id"], state)
        create_run(
            state["run_id"],
            {
                "status": "queued",
                "source": source,
                "agent_folder": state["agent_folder"],
                "agent_type": state["agent_type"],
                "turn_type": state["turn_type"],
                "report_path": report_path(state["run_id"]),
                "state_path": f"outputs/runs/{state['run_id']}.state.json",
                "active_tools": state["active_tools"],
                "stage": "queued",
                "pct": 0.0,
            },
        )
        return load_run(state["run_id"])

    monkeypatch.setattr(api_main, "enqueue_run", fake_enqueue)


def test_health_status_code():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_body():
    response = client.get("/health")
    assert response.json() == {"status": "ok"}


def test_evaluate_returns_run_id(monkeypatch):
    _stub_enqueue_run(monkeypatch)
    response = client.post("/evaluate", json={
        "agent_folder": FAKE_AGENT,
        "agent_type":   "tool_use",
        "turn_type":    "single",
        "eval_inputs":  ["What is 2+2?"],
    })
    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data
    assert "otel_trace_context" in load_state(data["run_id"])


def test_evaluate_returns_status_running(monkeypatch):
    _stub_enqueue_run(monkeypatch)
    response = client.post("/evaluate", json={
        "agent_folder": FAKE_AGENT,
        "agent_type":   "tool_use",
        "turn_type":    "single",
        "eval_inputs":  ["test"],
    })
    assert response.json()["status"] == "queued"


def test_evaluate_returns_report_path(monkeypatch):
    _stub_enqueue_run(monkeypatch)
    response = client.post("/evaluate", json={
        "agent_folder": FAKE_AGENT,
        "agent_type":   "tool_use",
        "turn_type":    "single",
        "eval_inputs":  ["test"],
    })
    data = response.json()
    assert "report_path" in data
    assert data["report_path"].startswith("outputs/")
    assert data["report_path"].endswith("_run_report.json")


def test_evaluate_default_model(monkeypatch):
    _stub_enqueue_run(monkeypatch)
    response = client.post("/evaluate", json={
        "agent_folder": FAKE_AGENT,
        "eval_inputs":  ["test"],
    })
    assert response.status_code == 200


def test_evaluate_missing_folder_returns_400():
    response = client.post("/evaluate", json={
        "agent_folder": "nonexistent/path",
        "eval_inputs":  ["test"],
    })
    assert response.status_code == 400


def test_run_status_endpoint(monkeypatch):
    _stub_enqueue_run(monkeypatch)
    response = client.post("/evaluate", json={
        "agent_folder": FAKE_AGENT,
        "eval_inputs": ["test"],
    })
    run_id = response.json()["run_id"]

    status = client.get(f"/runs/{run_id}")
    assert status.status_code == 200
    data = status.json()
    assert data["run_id"] == run_id
    assert data["status"] == "queued"
    assert load_run(run_id)["run_id"] == run_id


def test_run_report_not_ready_returns_404(monkeypatch):
    _stub_enqueue_run(monkeypatch)
    response = client.post("/evaluate", json={
        "agent_folder": FAKE_AGENT,
        "eval_inputs": ["test"],
    })
    run_id = response.json()["run_id"]

    report = client.get(f"/runs/{run_id}/report")
    assert report.status_code == 404


def test_openapi_docs_available():
    response = client.get("/docs")
    assert response.status_code == 200
