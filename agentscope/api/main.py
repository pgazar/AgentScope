import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agentscope.job_queue import build_run_state, enqueue_run, recover_incomplete_runs, validate_run_request
from agentscope.logging_setup import configure_logging
from agentscope.otel import (
    configure_telemetry,
    current_trace_ids,
    inject_trace_context,
    instrument_fastapi,
    set_span_attributes,
    start_span,
)
from agentscope.run_store import load_report, load_run, report_path

_log = configure_logging(component="api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_telemetry(service_name="agentscope-api")
    instrument_fastapi(app)
    recovered = recover_incomplete_runs()
    _log.info("api_lifespan_started", recovered_runs=recovered, recovered_count=len(recovered))
    yield


app = FastAPI(title="AgentScope API", lifespan=lifespan)


class EvalRequest(BaseModel):
    agent_folder: str
    agent_type:   str = "rag"
    turn_type:    str = "single"
    agent_model:  str = "claude-sonnet-4-5"
    kb_path:      str | None = None
    gt_path:      str | None = None
    eval_inputs:  list[str] = Field(default_factory=list)


def _validate_request(req: EvalRequest) -> None:
    try:
        validate_run_request(
            agent_folder=req.agent_folder,
            eval_inputs=req.eval_inputs,
            kb_path=req.kb_path,
            gt_path=req.gt_path,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/evaluate")
async def evaluate(req: EvalRequest):
    with start_span(
        "agentscope.api.evaluate",
        tracer_name="agentscope.api",
        attributes={
            "agentscope.agent_type": req.agent_type,
            "agentscope.turn_type": req.turn_type,
            "agentscope.eval_inputs_count": len(req.eval_inputs),
        },
    ) as span:
        _validate_request(req)
        run_id = str(uuid.uuid4())[:8]
        log = configure_logging(run_id=run_id, component="api")
        state = build_run_state(
            run_id=run_id,
            agent_folder=req.agent_folder,
            agent_model=req.agent_model,
            eval_inputs=req.eval_inputs,
            agent_type=req.agent_type,
            turn_type=req.turn_type,
            kb_path=req.kb_path,
            gt_path=req.gt_path,
        )
        state["otel_trace_context"] = inject_trace_context({"run_id": run_id})
        set_span_attributes(span, {
            "agentscope.run_id": run_id,
            **current_trace_ids(),
        })
        log.info(
            "run_enqueued",
            agent_folder=req.agent_folder,
            agent_type=req.agent_type,
            turn_type=req.turn_type,
            eval_inputs_count=len(req.eval_inputs),
        )
        record = enqueue_run(state, source="api")

        return {
            "run_id": record["run_id"],
            "status": record["status"],
            "report_path": record["report_path"],
        }


@app.get("/runs/{run_id}")
def run_status(run_id: str):
    record = load_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    configure_logging(run_id=run_id, component="api").info("run_status_requested", status=record.get("status"), stage=record.get("stage"))
    return record


@app.get("/runs/{run_id}/report")
def run_report(run_id: str):
    report = load_report(run_id)
    if report is None:
        record = load_run(run_id)
        if record and record.get("status") == "failed":
            raise HTTPException(status_code=409, detail=record.get("error", {}))
        raise HTTPException(status_code=404, detail=f"report not ready for run: {run_id}")
    configure_logging(run_id=run_id, component="api").info("run_report_requested", status="ready")
    return report


@app.get("/health")
def health():
    return {"status": "ok"}
