from __future__ import annotations

import os
import subprocess
import sys
import uuid

from agentscope.intake.intake_agent import IntakeAgent
from agentscope.config import load_config
from agentscope.run_store import (
    create_run,
    list_runs,
    load_report,
    load_run,
    report_path,
    run_state_path,
    save_state,
    update_run,
    utc_now,
)
from agentscope.runner import AgentRunner


def validate_run_request(
    *,
    agent_folder: str,
    eval_inputs: list[str],
    kb_path: str | None = None,
    gt_path: str | None = None,
) -> None:
    validation = AgentRunner.validate_agent_folder(agent_folder)
    if not validation["ok"]:
        raise ValueError(validation["reason"])

    if not eval_inputs:
        raise ValueError("eval_inputs must contain at least one query")

    if kb_path and not os.path.exists(kb_path):
        raise ValueError(f"kb_path not found: {kb_path}")

    if gt_path and not os.path.exists(gt_path):
        raise ValueError(f"gt_path not found: {gt_path}")


def build_run_state(
    *,
    agent_folder: str,
    agent_model: str,
    eval_inputs: list[str],
    agent_type: str,
    turn_type: str,
    kb_path: str | None = None,
    gt_path: str | None = None,
    run_id: str | None = None,
) -> dict:
    cfg = load_config()
    kb_format = "pdf" if kb_path else ("markdown" if gt_path else "none")
    answers = {
        "agent_type": agent_type,
        "turn_type": turn_type,
        "has_gt": "yes" if gt_path else "no",
        "kb_format": kb_format,
    }
    intake = IntakeAgent().run(answers)
    run_id = run_id or str(uuid.uuid4())[:8]

    return {
        "run_id": run_id,
        "agent_folder": agent_folder,
        "agent_model": agent_model,
        "eval_inputs": eval_inputs,
        "kb_path": kb_path,
        "gt_path": gt_path,
        "agent_type": intake["agent_type"],
        "turn_type": intake["turn_type"],
        "active_tools": intake["active_tools"],
        "expected_tools": [],
        "traces": [],
        "trace_diagnostics": None,
        "baseline_geval_scores": {},
        "ir_results": None,
        "behavior_results": None,
        "geval_results": None,
        "cost_results": None,
        "adversarial_results": None,
        "synth_results": None,
        "final_report": None,
        "config": cfg.model_dump(),
    }


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _worker_env() -> dict:
    env = os.environ.copy()
    repo_root = _repo_root()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([repo_root, existing]) if existing else repo_root
    env.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    return env


def worker_is_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def launch_worker(run_id: str) -> int:
    record = load_run(run_id)
    if record is None:
        raise ValueError(f"run not found: {run_id}")

    existing_pid = record.get("worker_pid")
    if worker_is_alive(existing_pid):
        return existing_pid

    proc = subprocess.Popen(
        [sys.executable, "-m", "agentscope.job_worker", run_id],
        cwd=_repo_root(),
        env=_worker_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    update_run(
        run_id,
        worker_pid=proc.pid,
        enqueued_at=record.get("enqueued_at") or utc_now(),
        state_path=record.get("state_path") or run_state_path(run_id),
    )
    return proc.pid


def enqueue_run(state: dict, source: str = "api") -> dict:
    run_id = state["run_id"]
    state_path = save_state(run_id, state)
    create_run(
        run_id,
        {
            "status": "queued",
            "source": source,
            "agent_folder": state["agent_folder"],
            "agent_type": state["agent_type"],
            "turn_type": state["turn_type"],
            "report_path": report_path(run_id),
            "state_path": state_path,
            "active_tools": state["active_tools"],
            "stage": "queued",
            "pct": 0.0,
            "enqueued_at": utc_now(),
        },
    )
    launch_worker(run_id)
    return load_run(run_id)


def recover_incomplete_runs() -> list[str]:
    recovered: list[str] = []
    for record in list_runs({"queued", "running"}):
        run_id = record["run_id"]

        if load_report(run_id) is not None:
            update_run(run_id, status="completed", stage="done", pct=1.0, report_path=report_path(run_id))
            continue

        if not os.path.exists(record.get("state_path", run_state_path(run_id))):
            update_run(
                run_id,
                status="failed",
                stage="error",
                error={"type": "MissingState", "message": "state file missing; run cannot be recovered"},
                finished_at=utc_now(),
            )
            continue

        if worker_is_alive(record.get("worker_pid")):
            continue

        update_run(run_id, status="queued", stage="recovered", pct=0.0, recovered_at=utc_now())
        launch_worker(run_id)
        recovered.append(run_id)

    return recovered
