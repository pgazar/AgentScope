import uuid
import json
import os
import sys
import time
import tempfile
import subprocess
import logging
import gradio as gr

from agentscope.dashboard.charts import ir_chart, agent_chart, geval_chart, cost_chart, adversarial_chart
from agentscope.intake.intake_agent import IntakeAgent
from agentscope.config import load_config

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("agentscope.app")

AGENTSCOPE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER_SCRIPT   = os.path.join(AGENTSCOPE_ROOT, "agentscope", "dashboard", "runner_subprocess.py")

# Env files to pass to subprocess
ENV_FILES = [
    os.path.join(AGENTSCOPE_ROOT, ".env"),
    "/Users/pegahzargarian/projects/MCP-1/workspace/capstone-rag/.env",
]

EMPTY = (None, None, None, None, None)

STAGE_PROGRESS = {
    "starting":         (0.02, "Starting..."),
    "running_agent":    (0.08, "Running agent on queries..."),
    "agent_done":       (0.18, "Agent done — traces collected"),
    "synth_gen":        (0.25, "Generating synthetic Q&A pairs..."),
    "ir_evaluator":     (0.32, "Scoring retrieval (IR metrics)..."),
    "ir_done":          (0.38, "IR metrics done"),
    "agent_behavior":   (0.42, "Scoring agent behavior..."),
    "behavior_done":    (0.52, "Behavior metrics done"),
    "adversarial_eval": (0.56, "Running adversarial evaluation..."),
    "adversarial_done": (0.68, "Adversarial done"),
    "geval":            (0.72, "G-Eval LLM judge scoring..."),
    "geval_done":       (0.88, "G-Eval done"),
    "cost_analyzer":    (0.90, "Computing cost & latency..."),
    "cost_done":        (0.94, "Cost done"),
    "compile_report":   (0.96, "Writing report..."),
    "done":             (1.00, "Complete ✓"),
}


def run_evaluation(
    agent_folder: str,
    agent_model: str,
    kb_file,
    gt_file,
    eval_inputs_text: str,
    agent_type: str,
    turn_type: str,
    progress=gr.Progress(),
):
    eval_inputs = [l.strip() for l in (eval_inputs_text or "").strip().splitlines() if l.strip()]
    if not eval_inputs:
        raise ValueError("Evaluation inputs cannot be empty — enter at least one query.")

    cfg     = load_config()
    intake  = IntakeAgent().run({
        "agent_type": agent_type,
        "turn_type":  turn_type,
        "has_gt":     "yes" if gt_file else "no",
        "kb_format":  "pdf" if kb_file else "none",
    })

    state = {
        "run_id":              str(uuid.uuid4())[:8],
        "agent_folder":        agent_folder,
        "agent_model":         agent_model,
        "eval_inputs":         eval_inputs,
        "kb_path":             kb_file.name if kb_file else None,
        "gt_path":             gt_file.name if gt_file else None,
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

    _log.info(f"run_evaluation: folder={agent_folder}, inputs={eval_inputs}, agent_type={agent_type}")

    # Write progress file that subprocess updates
    progress_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    progress_file.write("{}"); progress_file.close()
    pf = progress_file.name

    # Build env for subprocess — inherit current env + load env files
    env = {**os.environ,
           "PYTHONPATH": AGENTSCOPE_ROOT,
           "AGENTSCOPE_STATE": json.dumps(state),
           "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE": "180",
           "AGENTSCOPE_VARIANCE": "0"}

    cmd = [sys.executable, RUNNER_SCRIPT, pf] + [f for f in ENV_FILES if os.path.exists(f)]
    proc = subprocess.Popen(cmd, env=env, cwd=AGENTSCOPE_ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    _log.info(f"subprocess PID: {proc.pid}")

    # Current chart state — updated as stages complete
    charts = {"ir": None, "behavior": None, "geval": None, "cost": None, "adversarial": None}

    yield EMPTY  # show empty panels immediately

    last_stage = ""
    while proc.poll() is None:
        time.sleep(1.5)
        try:
            with open(pf) as f:
                data = json.load(f)
        except Exception:
            continue

        stage = data.get("stage", "")
        pct, desc = STAGE_PROGRESS.get(stage, (0, stage))

        # Always update progress so spinner keeps moving even on slow stages
        progress(pct, desc=desc)

        if stage == last_stage:
            continue
        last_stage = stage
        _log.info(f"stage: {stage} ({pct*100:.0f}%)")

        # Update charts as each result arrives
        if stage == "ir_done" and data.get("ir"):
            charts["ir"] = ir_chart(data["ir"])
        if stage == "behavior_done" and data.get("behavior"):
            charts["behavior"] = agent_chart(data["behavior"])
        if stage == "adversarial_done" and data.get("adversarial"):
            charts["adversarial"] = adversarial_chart(data["adversarial"])
        if stage == "geval_done" and data.get("geval"):
            charts["geval"] = geval_chart(data["geval"])
        if stage == "cost_done" and data.get("cost"):
            charts["cost"] = cost_chart(data["cost"])

        yield (charts["ir"], charts["behavior"], charts["geval"],
               charts["cost"], charts["adversarial"])

        if stage == "error":
            _log.error(f"subprocess error: {data.get('error')}\n{data.get('traceback','')}")
            break

    # Read final state
    proc.wait()
    try:
        with open(pf) as f:
            final = json.load(f)
        if final.get("stage") == "done":
            yield (
                ir_chart(final.get("ir") or {}),
                agent_chart(final.get("behavior") or {}),
                geval_chart(final.get("geval") or {}),
                cost_chart(final.get("cost") or {}),
                adversarial_chart(final.get("adversarial") or {}),
            )
    except Exception as e:
        _log.error(f"final read failed: {e}")
    finally:
        os.unlink(pf)

    progress(1.0, desc="Done ✓")


with gr.Blocks(title="AgentScope") as demo:
    gr.Markdown("## AgentScope — Agentic Evaluation Framework")
    gr.Markdown(
        "_Agent folder must contain a `main.py` with a `run(query: str) -> str` function. "
        "Examples: `eval_targets/capstone_rag` · `tests/fake_agent`_"
    )

    with gr.Row():
        agent_folder = gr.Textbox(label="Agent folder path", value="eval_targets/capstone_rag")
        agent_model  = gr.Textbox(label="Agent model name", value="claude-haiku-4-5-20251001")

    with gr.Row():
        kb_file = gr.File(label="Knowledge base (optional)", file_types=[".pdf", ".md", ".txt"])
        gt_file = gr.File(label="Ground truth CSV (optional)", file_types=[".csv"])

    eval_inputs_text = gr.Textbox(
        label="Evaluation inputs (one query per line)",
        lines=4,
        placeholder="What is the refund policy?\nHow do I reset my password?",
        value="What was the revenue for Q3?",
    )

    with gr.Row():
        agent_type = gr.Dropdown(["rag", "tool_use", "multi_agent", "hybrid"], label="Agent type", value="rag")
        turn_type  = gr.Dropdown(["single", "multi"], label="Turn type", value="single")

    run_btn = gr.Button("Run evaluation", variant="primary")

    with gr.Row():
        ir_plot    = gr.Plot(label="1 — IR metrics")
        agent_plot = gr.Plot(label="2 — Agentic metrics")
    with gr.Row():
        geval_plot = gr.Plot(label="3 — Response quality")
        cost_plot  = gr.Plot(label="4 — Cost analysis")
    with gr.Row():
        adv_plot = gr.Plot(label="5 — Safety and robustness")

    run_btn.click(
        run_evaluation,
        inputs=[agent_folder, agent_model, kb_file, gt_file,
                eval_inputs_text, agent_type, turn_type],
        outputs=[ir_plot, agent_plot, geval_plot, cost_plot, adv_plot],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
