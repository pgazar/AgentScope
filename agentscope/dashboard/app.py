import os
import time
import gradio as gr

from agentscope.dashboard.charts import (
    adversarial_chart,
    agent_chart,
    cost_chart,
    geval_chart,
    ir_chart,
)
from agentscope.job_queue import build_run_state, enqueue_run, recover_incomplete_runs, validate_run_request
from agentscope.logging_setup import configure_logging
from agentscope.otel import configure_telemetry, inject_trace_context, set_span_attributes, start_span
from agentscope.run_store import load_report, load_run

_log = configure_logging(component="dashboard")
configure_telemetry(service_name="agentscope-dashboard")


def _empty_plots():
    return (None, None, None, None, None)


def _plots_from_report(report: dict):
    eval_results = report["eval_results"]
    return (
        ir_chart(eval_results["ir"] or {}),
        agent_chart(eval_results["behavior"] or {}),
        geval_chart(eval_results["geval"] or {}),
        cost_chart(eval_results["cost"] or {}),
        adversarial_chart(eval_results["adversarial"] or {}),
    )


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
    with start_span(
        "agentscope.dashboard.evaluate",
        tracer_name="agentscope.dashboard",
        attributes={
            "agentscope.agent_type": agent_type,
            "agentscope.turn_type": turn_type,
        },
    ) as span:
        eval_inputs = [l.strip() for l in (eval_inputs_text or "").strip().splitlines() if l.strip()]
        if not eval_inputs:
            gr.Warning("Evaluation inputs cannot be empty — enter at least one query in the text box.")
            return _empty_plots()

        # Auto-detect ground_truth.csv in the agent folder (no upload needed)
        _auto_gt = os.path.join(agent_folder, "ground_truth.csv")
        kb_path = kb_file.name if kb_file else None
        gt_path = gt_file.name if gt_file else (_auto_gt if os.path.exists(_auto_gt) else None)

        try:
            validate_run_request(
                agent_folder=agent_folder,
                eval_inputs=eval_inputs,
                kb_path=kb_path,
                gt_path=gt_path,
            )
        except ValueError as e:
            gr.Warning(str(e))
            return _empty_plots()

        state = build_run_state(
            agent_folder=agent_folder,
            agent_model=agent_model,
            eval_inputs=eval_inputs,
            agent_type=agent_type,
            turn_type=turn_type,
            kb_path=kb_path,
            gt_path=gt_path,
        )
        state["otel_trace_context"] = inject_trace_context({"run_id": state["run_id"]})
        set_span_attributes(span, {
            "agentscope.run_id": state["run_id"],
            "agentscope.eval_inputs_count": len(eval_inputs),
        })

        log = configure_logging(run_id=state["run_id"], component="dashboard")
        log.info(
            "run_queued_from_dashboard",
            agent_folder=agent_folder,
            agent_type=agent_type,
            turn_type=turn_type,
            eval_inputs_count=len(eval_inputs),
        )
        record = enqueue_run(state, source="dashboard")
        run_id = record["run_id"]
        progress(0.02, desc=f"Queued run {run_id}")

        deadline = time.time() + 60 * 30
        while time.time() < deadline:
            record = load_run(run_id)
            if record is None:
                gr.Warning(f"Run {run_id} disappeared before completion.")
                return _empty_plots()

            pct = record.get("pct", 0.0)
            status = record.get("status", "queued")
            stage = record.get("stage", status)
            progress(max(0.0, pct if isinstance(pct, (int, float)) else 0.0), desc=f"{status}: {stage}")

            if status == "completed":
                log.info("dashboard_run_completed", stage=stage)
                report = load_report(run_id)
                if report is None:
                    time.sleep(0.25)
                    continue
                progress(1.0, desc=f"Complete: {run_id}")
                return _plots_from_report(report)

            if status == "failed":
                error = record.get("error", {}).get("message", "unknown error")
                log.error("dashboard_run_failed", stage=stage, error_message=error)
                gr.Warning(f"Run {run_id} failed: {error}")
                return _empty_plots()

            time.sleep(0.5)

        log.warning("dashboard_run_timed_out")
        gr.Warning(f"Run {run_id} did not finish before the dashboard timeout.")
        return _empty_plots()


with gr.Blocks(title="AgentScope") as demo:
    gr.Markdown("## AgentScope — Agentic Evaluation Framework")
    gr.Markdown(
        "_Agent folder must contain a `main.py` with a `run(query: str) -> str` function. "
        "**Docker:** agent folders are mounted at `/agents/` — enter e.g. `/agents/capstone_rag`. "       "**Local:** enter any absolute path. "       "Agent folder must contain `main.py` with `def run(query: str) -> str`. "       "Optionally add `ground_truth.csv` for IR metrics._"
    )

    with gr.Row():
        agent_folder = gr.Textbox(label="Agent folder path", value="")
        agent_model  = gr.Textbox(label="Agent model name", value="claude-haiku-4-5-20251001")

    with gr.Row():
        kb_file = gr.File(label="Knowledge base (optional)", file_types=[".pdf", ".md", ".txt"])
        gt_file = gr.File(label="Ground truth CSV (optional)", file_types=[".csv"])

    eval_inputs_text = gr.Textbox(
        label="Evaluation inputs (one query per line)",
        lines=4,
        placeholder="Single-turn: one query per line\nMulti-turn (turn_type=multi): each line is one conversation turn",
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
    recover_incomplete_runs()
    demo.launch(server_name="0.0.0.0", server_port=7860)
