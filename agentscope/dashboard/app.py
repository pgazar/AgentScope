import os
import uuid
import logging
import gradio as gr

from agentscope.dashboard.charts import ir_chart, agent_chart, geval_chart, cost_chart, adversarial_chart
from agentscope.orchestrator.graph import build_graph
from agentscope.intake.intake_agent import IntakeAgent
from agentscope.config import load_config

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("agentscope.app")


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
    cfg = load_config()

    eval_inputs = [l.strip() for l in (eval_inputs_text or "").strip().splitlines() if l.strip()]
    if not eval_inputs:
        raise ValueError("Evaluation inputs cannot be empty — enter at least one query.")

    # Auto-detect GT for capstone_rag even when no file uploaded
    # Auto-detect ground_truth.csv in any agent folder
    _auto_gt = os.path.join(agent_folder, "ground_truth.csv")
    has_gt = gt_file or os.path.exists(_auto_gt)
    answers = {
        "agent_type": agent_type,
        "turn_type":  turn_type,
        "has_gt":     "yes" if has_gt else "no",
        # If GT exists, treat as having a KB so ir_evaluator activates
        "kb_format":  "pdf" if kb_file else ("markdown" if has_gt else "none"),
    }
    intake = IntakeAgent().run(answers)
    graph  = build_graph(intake["active_tools"])

    state = {
        "run_id":              str(uuid.uuid4())[:8],
        "agent_folder":        agent_folder,
        "agent_model":         agent_model,
        "eval_inputs":         eval_inputs,
        "kb_path":             kb_file.name if kb_file else None,
        "gt_path":             gt_file.name if gt_file else (_auto_gt if os.path.exists(_auto_gt) else None),
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
    progress(0.1, desc="Running agent and collecting traces...")
    result = graph.invoke(state)
    progress(1.0, desc="Complete")

    report = result["final_report"]["eval_results"]
    return (
        ir_chart(report["ir"]          or {}),
        agent_chart(report["behavior"] or {}),
        geval_chart(report["geval"]    or {}),
        cost_chart(report["cost"]      or {}),
        adversarial_chart(report["adversarial"] or {}),
    )


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
