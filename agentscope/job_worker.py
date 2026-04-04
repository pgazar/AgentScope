from __future__ import annotations

import os
import sys

from agentscope.logging_setup import configure_logging
from agentscope.otel import (
    configure_telemetry,
    current_trace_ids,
    extract_trace_context,
    mark_span_error,
    mark_span_ok,
    start_span,
)
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
    configure_telemetry(service_name="agentscope-worker")
    log = configure_logging(run_id=run_id, component="worker")

    record = load_run(run_id)
    if record is None:
        raise RuntimeError(f"run not found: {run_id}")

    state = load_state(run_id)
    if state is None:
        raise RuntimeError(f"state not found for run: {run_id}")

    parent_context = extract_trace_context(state.get("otel_trace_context"))
    with start_span(
        "agentscope.run",
        tracer_name="agentscope.worker",
        context=parent_context,
        attributes={
            "agentscope.run_id": run_id,
            "agentscope.agent_type": state.get("agent_type"),
            "agentscope.turn_type": state.get("turn_type"),
            "agentscope.active_tools": state.get("active_tools", []),
            "agentscope.eval_inputs_count": len(state.get("eval_inputs", [])),
        },
    ) as span:
        trace_ids = current_trace_ids()
        update_run(
            run_id,
            status="running",
            stage="starting",
            pct=0.01,
            worker_pid=os.getpid(),
            started_at=record.get("started_at") or utc_now(),
            trace_id=trace_ids.get("trace_id"),
            span_id=trace_ids.get("span_id"),
        )
        log.info(
            "worker_started",
            agent_folder=state.get("agent_folder"),
            agent_type=state.get("agent_type"),
            turn_type=state.get("turn_type"),
            active_tools=state.get("active_tools", []),
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
            log.info(
                "worker_completed",
                trace_status=trace_diag.get("status"),
                issues=trace_diag.get("issues", []),
                warnings=trace_diag.get("warnings", []),
            )
            mark_span_ok(span)
        except Exception as e:
            mark_span_error(span, e)
            update_run(
                run_id,
                status="failed",
                stage="error",
                pct=-1.0,
                finished_at=utc_now(),
                error={"type": type(e).__name__, "message": str(e)},
            )
            log.error("worker_failed", error_type=type(e).__name__, error_message=str(e))
            raise

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m agentscope.job_worker <run_id>")
    raise SystemExit(main(sys.argv[1]))
