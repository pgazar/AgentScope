from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any


_LOCK = threading.Lock()
_DB_LOCK = threading.Lock()
_DB_KEY: tuple[str, str | None] | None = None
_DB_TABLES: dict[str, Any] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def storage_backend() -> str:
    mode = os.environ.get("AGENTSCOPE_RUN_STORE_BACKEND", "auto").strip().lower()
    if mode in {"db", "database", "postgres", "postgresql", "sql"}:
        return "db"
    if mode in {"file", "filesystem", "json"}:
        return "file"
    return "db" if _database_url() else "file"


def _database_url() -> str | None:
    value = os.environ.get("DATABASE_URL", "").strip()
    return value or None


def _runs_dir() -> str:
    path = os.path.join("outputs", "runs")
    os.makedirs(path, exist_ok=True)
    return path


def run_status_path(run_id: str) -> str:
    if storage_backend() == "db":
        return f"db://agentscope_runs/{run_id}"
    return os.path.join(_runs_dir(), f"{run_id}.json")


def run_state_path(run_id: str) -> str:
    if storage_backend() == "db":
        return f"db://agentscope_run_states/{run_id}"
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


def _load_json_file(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _dump_db_payload(value: Any) -> dict | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def _db_tables() -> dict[str, Any]:
    global _DB_KEY, _DB_TABLES

    backend = storage_backend()
    url = _database_url()
    key = (backend, url)

    if backend != "db" or not url:
        raise RuntimeError("database backend requested but DATABASE_URL is not configured")

    with _DB_LOCK:
        if _DB_TABLES is not None and _DB_KEY == key:
            return _DB_TABLES

        from sqlalchemy import JSON, Column, MetaData, String, Table, create_engine

        metadata = MetaData()
        tables = {
            "engine": create_engine(url, future=True, pool_pre_ping=True),
            "runs": Table(
                "agentscope_runs",
                metadata,
                Column("run_id", String(64), primary_key=True),
                Column("status", String(32), nullable=False),
                Column("submitted_at", String(64), nullable=False),
                Column("updated_at", String(64), nullable=False),
                Column("record", JSON, nullable=False),
            ),
            "states": Table(
                "agentscope_run_states",
                metadata,
                Column("run_id", String(64), primary_key=True),
                Column("updated_at", String(64), nullable=False),
                Column("state", JSON, nullable=False),
            ),
            "reports": Table(
                "agentscope_run_reports",
                metadata,
                Column("run_id", String(64), primary_key=True),
                Column("updated_at", String(64), nullable=False),
                Column("report", JSON, nullable=False),
            ),
        }
        metadata.create_all(tables["engine"])
        _DB_KEY = key
        _DB_TABLES = tables
        return tables


def _db_select_one(table, column_name: str, run_id: str) -> dict | None:
    from sqlalchemy import select

    tables = _db_tables()
    with tables["engine"].begin() as conn:
        value = conn.execute(
            select(getattr(table.c, column_name)).where(table.c.run_id == run_id)
        ).scalar_one_or_none()
    return _dump_db_payload(value)


def _db_upsert_json(table, value_column: str, run_id: str, payload: dict, **extra_fields) -> None:
    from sqlalchemy import insert, select, update

    tables = _db_tables()
    value = dict(payload)
    with tables["engine"].begin() as conn:
        existing = conn.execute(
            select(table.c.run_id).where(table.c.run_id == run_id)
        ).scalar_one_or_none()
        row = {
            "run_id": run_id,
            value_column: value,
            **extra_fields,
        }
        if existing is None:
            conn.execute(insert(table).values(**row))
        else:
            conn.execute(
                update(table)
                .where(table.c.run_id == run_id)
                .values(**row)
            )


def create_run(run_id: str, payload: dict) -> dict:
    record = {
        "run_id": run_id,
        "status": "queued",
        "submitted_at": utc_now(),
        "updated_at": utc_now(),
        **payload,
    }

    if storage_backend() == "db":
        tables = _db_tables()
        _db_upsert_json(
            tables["runs"],
            "record",
            run_id,
            record,
            status=record.get("status", "queued"),
            submitted_at=record["submitted_at"],
            updated_at=record["updated_at"],
        )
        return record

    with _LOCK:
        _write_json_atomic(run_status_path(run_id), record)
    return record


def update_run(run_id: str, **fields) -> dict:
    now = utc_now()
    if storage_backend() == "db":
        tables = _db_tables()
        current = _db_select_one(tables["runs"], "record", run_id) or {}
        current.update(fields)
        current["run_id"] = run_id
        current["updated_at"] = now
        submitted_at = current.get("submitted_at") or now
        _db_upsert_json(
            tables["runs"],
            "record",
            run_id,
            current,
            status=current.get("status", "queued"),
            submitted_at=submitted_at,
            updated_at=now,
        )
        return current

    path = run_status_path(run_id)
    with _LOCK:
        current = _load_json_file(path) or {}
        current.update(fields)
        current["run_id"] = run_id
        current["updated_at"] = now
        _write_json_atomic(path, current)
    return current


def load_run(run_id: str) -> dict | None:
    if storage_backend() == "db":
        tables = _db_tables()
        return _db_select_one(tables["runs"], "record", run_id)
    return _load_json_file(run_status_path(run_id))


def save_report(run_id: str, report: dict) -> str:
    path = report_path(run_id)
    with _LOCK:
        _write_json_atomic(path, report)

    if storage_backend() == "db":
        tables = _db_tables()
        _db_upsert_json(
            tables["reports"],
            "report",
            run_id,
            report,
            updated_at=utc_now(),
        )

    return path


def load_report(run_id: str) -> dict | None:
    if storage_backend() == "db":
        tables = _db_tables()
        report = _db_select_one(tables["reports"], "report", run_id)
        if report is not None:
            return report
    return _load_json_file(report_path(run_id))


def save_state(run_id: str, state: dict) -> str:
    path = run_state_path(run_id)
    if storage_backend() == "db":
        tables = _db_tables()
        _db_upsert_json(
            tables["states"],
            "state",
            run_id,
            state,
            updated_at=utc_now(),
        )
        return path

    with _LOCK:
        _write_json_atomic(path, state)
    return path


def load_state(run_id: str) -> dict | None:
    if storage_backend() == "db":
        tables = _db_tables()
        return _db_select_one(tables["states"], "state", run_id)
    return _load_json_file(run_state_path(run_id))


def list_runs(statuses: set[str] | None = None) -> list[dict]:
    if storage_backend() == "db":
        from sqlalchemy import select

        tables = _db_tables()
        query = select(tables["runs"].c.record).order_by(tables["runs"].c.submitted_at)
        if statuses:
            query = query.where(tables["runs"].c.status.in_(statuses))
        with tables["engine"].begin() as conn:
            rows = conn.execute(query).scalars().all()
        return [row for row in (_dump_db_payload(item) for item in rows) if row is not None]

    records = []
    for name in os.listdir(_runs_dir()):
        if not name.endswith(".json") or name.endswith(".state.json"):
            continue
        record = _load_json_file(os.path.join(_runs_dir(), name))
        if not record:
            continue
        if statuses and record.get("status") not in statuses:
            continue
        records.append(record)
    return sorted(records, key=lambda r: r.get("submitted_at", ""))
