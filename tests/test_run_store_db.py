from agentscope.run_store import (
    create_run,
    list_runs,
    load_report,
    load_run,
    load_state,
    report_path,
    run_state_path,
    run_status_path,
    save_report,
    save_state,
    storage_backend,
    update_run,
)


def test_sql_backed_run_store_round_trip(tmp_path, monkeypatch):
    db_path = tmp_path / "agentscope.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.delenv("AGENTSCOPE_RUN_STORE_BACKEND", raising=False)
    monkeypatch.chdir(tmp_path)

    assert storage_backend() == "db"
    assert run_status_path("db-run") == "db://agentscope_runs/db-run"
    assert run_state_path("db-run") == "db://agentscope_run_states/db-run"

    state = {"run_id": "db-run", "agent_folder": "/tmp/agent", "otel_trace_context": {}}
    record = create_run(
        "db-run",
        {
            "status": "queued",
            "agent_folder": "/tmp/agent",
            "agent_type": "tool_use",
            "turn_type": "single",
            "report_path": report_path("db-run"),
            "state_path": run_state_path("db-run"),
            "stage": "queued",
            "pct": 0.0,
        },
    )
    save_state("db-run", state)
    update_run("db-run", status="running", stage="starting", pct=0.25)
    save_report("db-run", {"run_id": "db-run", "eval_results": {"cost": {"cost_per_query": 0.01}}})

    loaded_run = load_run("db-run")
    assert record["run_id"] == "db-run"
    assert loaded_run["status"] == "running"
    assert loaded_run["stage"] == "starting"
    assert load_state("db-run")["run_id"] == "db-run"
    assert load_report("db-run")["eval_results"]["cost"]["cost_per_query"] == 0.01

    queued = list_runs({"running"})
    assert [item["run_id"] for item in queued] == ["db-run"]
    assert (tmp_path / report_path("db-run")).exists()
