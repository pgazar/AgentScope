import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agentscope.job_queue import build_run_state, enqueue_run, recover_incomplete_runs, validate_run_request
from agentscope.run_store import load_report, load_run, report_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    recover_incomplete_runs()
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
    _validate_request(req)
    run_id = str(uuid.uuid4())[:8]
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
    return record


@app.get("/runs/{run_id}/report")
def run_report(run_id: str):
    report = load_report(run_id)
    if report is None:
        record = load_run(run_id)
        if record and record.get("status") == "failed":
            raise HTTPException(status_code=409, detail=record.get("error", {}))
        raise HTTPException(status_code=404, detail=f"report not ready for run: {run_id}")
    return report


@app.get("/health")
def health():
    return {"status": "ok"}
