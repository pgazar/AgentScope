from __future__ import annotations

import os
import sys

from agentscope.pipeline_runner import execute_pipeline
from agentscope.run_store import load_run, load_state, report_path, update_run, utc_now


def _progress(run_id: str, stage: str, pct: float, data: dict | None = None):
    payload = {
        "status": "running" if stage != "done" else "completed",
        "stage": stage,
        "pct": pct,
    }
    if stage == "done":
        payload["finished_at"] = utc_now()
        payload["report_path"] = report_path(run_id)
    if data:
        payload["progress_data"] = data
    update_run(run_id, **payload)


def main(run_id: str) -> int:
    record = load_run(run_id)
    if record is None:
        raise RuntimeError(f"run not found: {run_id}")

    state = load_state(run_id)
    if state is None:
        raise RuntimeError(f"state not found for run: {run_id}")

    update_run(
        run_id,
        status="running",
        stage="starting",
        pct=0.01,
        worker_pid=os.getpid(),
        started_at=record.get("started_at") or utc_now(),
    )

    try:
        result = execute_pipeline(state, lambda stage, pct, data=None: _progress(run_id, stage, pct, data))
        trace_diag = result.get("trace_diagnostics") or {}
        update_run(
            run_id,
            status="completed",
            stage="done",
            pct=1.0,
            finished_at=utc_now(),
            report_path=report_path(run_id),
            trace_status=trace_diag.get("status"),
            issues=trace_diag.get("issues", []),
            warnings=trace_diag.get("warnings", []),
        )
    except Exception as e:
        update_run(
            run_id,
            status="failed",
            stage="error",
            pct=-1.0,
            finished_at=utc_now(),
            error={"type": type(e).__name__, "message": str(e)},
        )
        raise

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m agentscope.job_worker <run_id>")
    raise SystemExit(main(sys.argv[1]))
