import uuid
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="AgentScope API")


class EvalRequest(BaseModel):
    agent_folder: str
    agent_type:   str = "rag"
    turn_type:    str = "single"
    agent_model:  str = "claude-sonnet-4-5"
    kb_path:      str | None = None
    gt_path:      str | None = None
    eval_inputs:  list[str] = []


@app.post("/evaluate")
async def evaluate(req: EvalRequest, bg: BackgroundTasks):
    from agentscope.orchestrator.graph import build_graph
    from agentscope.intake.intake_agent import IntakeAgent
    from agentscope.config import load_config

    run_id = str(uuid.uuid4())[:8]
    cfg    = load_config()

    answers = {
        "agent_type": req.agent_type,
        "turn_type":  req.turn_type,
        "has_gt":     "yes" if req.gt_path else "no",
        "kb_format":  "pdf" if req.kb_path else "none",
    }
    intake = IntakeAgent().run(answers)
    graph  = build_graph(intake["active_tools"])

    state = {
        "run_id":              run_id,
        "agent_folder":        req.agent_folder,
        "agent_model":         req.agent_model,
        "eval_inputs":         req.eval_inputs,
        "kb_path":             req.kb_path,
        "gt_path":             req.gt_path,
        "agent_type":          intake["agent_type"],
        "turn_type":           intake["turn_type"],
        "active_tools":        intake["active_tools"],
        "expected_tools":      [],
        "traces":              [],
        "baseline_geval_scores": {},
        "ir_results":          None,
        "behavior_results":    None,
        "geval_results":       None,
        "cost_results":        None,
        "adversarial_results": None,
        "synth_results":       None,
        "final_report":        None,
        "config":              cfg.model_dump(),
    }

    bg.add_task(graph.invoke, state)

    return {
        "run_id":      run_id,
        "status":      "running",
        "report_path": f"outputs/{run_id}_run_report.json",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
