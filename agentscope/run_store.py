from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone


_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runs_dir() -> str:
    path = os.path.join("outputs", "runs")
    os.makedirs(path, exist_ok=True)
    return path


def run_status_path(run_id: str) -> str:
    return os.path.join(_runs_dir(), f"{run_id}.json")


def run_state_path(run_id: str) -> str:
    return os.path.join(_runs_dir(), f"{run_id}.state.json")


def report_path(run_id: str) -> str:
    return os.path.join("outputs", f"{run_id}_run_report.json")


def _write_json_atomic(path: str, payload: dict) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=parent, delete=False) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def create_run(run_id: str, payload: dict) -> dict:
    record = {
        "run_id": run_id,
        "status": "queued",
        "submitted_at": utc_now(),
        "updated_at": utc_now(),
        **payload,
    }
    with _LOCK:
        _write_json_atomic(run_status_path(run_id), record)
    return record


def update_run(run_id: str, **fields) -> dict:
    path = run_status_path(run_id)
    with _LOCK:
        current = {}
        if os.path.exists(path):
            with open(path) as f:
                current = json.load(f)
        current.update(fields)
        current["run_id"] = run_id
        current["updated_at"] = utc_now()
        _write_json_atomic(path, current)
    return current


def load_run(run_id: str) -> dict | None:
    path = run_status_path(run_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_report(run_id: str) -> dict | None:
    path = report_path(run_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_state(run_id: str, state: dict) -> str:
    path = run_state_path(run_id)
    with _LOCK:
        _write_json_atomic(path, state)
    return path


def load_state(run_id: str) -> dict | None:
    path = run_state_path(run_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def list_runs(statuses: set[str] | None = None) -> list[dict]:
    records = []
    for name in os.listdir(_runs_dir()):
        if not name.endswith(".json") or name.endswith(".state.json"):
            continue
        with open(os.path.join(_runs_dir(), name)) as f:
            record = json.load(f)
        if statuses and record.get("status") not in statuses:
            continue
        records.append(record)
    return sorted(records, key=lambda r: r.get("submitted_at", ""))
