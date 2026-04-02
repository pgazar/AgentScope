import pytest
from fastapi.testclient import TestClient
from agentscope.api.main import app

client = TestClient(app, raise_server_exceptions=False)


def test_health_status_code():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_body():
    response = client.get("/health")
    assert response.json() == {"status": "ok"}


def test_evaluate_returns_run_id():
    response = client.post("/evaluate", json={
        "agent_folder": "tests/fake_agent",
        "agent_type":   "tool_use",
        "turn_type":    "single",
        "eval_inputs":  ["What is 2+2?"],
    })
    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data


def test_evaluate_returns_status_running():
    response = client.post("/evaluate", json={
        "agent_folder": "tests/fake_agent",
        "agent_type":   "tool_use",
        "turn_type":    "single",
        "eval_inputs":  ["test"],
    })
    assert response.json()["status"] == "running"


def test_evaluate_returns_report_path():
    response = client.post("/evaluate", json={
        "agent_folder": "tests/fake_agent",
        "agent_type":   "tool_use",
        "turn_type":    "single",
        "eval_inputs":  ["test"],
    })
    data = response.json()
    assert "report_path" in data
    assert data["report_path"].startswith("outputs/")
    assert data["report_path"].endswith("_run_report.json")


def test_evaluate_default_model():
    response = client.post("/evaluate", json={
        "agent_folder": "tests/fake_agent",
        "eval_inputs":  ["test"],
    })
    assert response.status_code == 200


def test_evaluate_missing_folder_still_responds():
    # FastAPI returns 200 and queues the job — failure happens async
    response = client.post("/evaluate", json={
        "agent_folder": "nonexistent/path",
        "eval_inputs":  ["test"],
    })
    assert response.status_code == 200


def test_openapi_docs_available():
    response = client.get("/docs")
    assert response.status_code == 200
