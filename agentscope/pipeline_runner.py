from __future__ import annotations

import importlib


def execute_pipeline(state: dict, progress_callback=None) -> dict:
    from agentscope.config import load_config
    from agentscope.orchestrator.graph import _run_agent

    cfg = load_config()
    state["config"] = cfg.model_dump()

    def progress(stage: str, pct: float, data: dict | None = None):
        if progress_callback:
            progress_callback(stage, pct, data or {})

    def node(module: str, fn: str = "run"):
        mod = importlib.import_module(module)
        updates = getattr(mod, fn)(state)
        state.update(updates)

    progress("starting", 0.02)

    progress("running_agent", 0.08)
    state.update(_run_agent(state))
    progress("agent_done", 0.18, {"traces_count": len(state["traces"])})

    active = state["active_tools"]

    if "synth_gen" in active:
        progress("synth_gen", 0.25)
        node("agentscope.tools.synth_gen")

    if "ir_evaluator" in active:
        progress("ir_evaluator", 0.32)
        node("agentscope.tools.ir_evaluator")
        progress("ir_done", 0.38, {"ir": state.get("ir_results")})

    progress("agent_behavior", 0.42)
    node("agentscope.tools.agent_behavior")
    progress("behavior_done", 0.52, {"behavior": state.get("behavior_results")})

    progress("adversarial_eval", 0.56)
    node("agentscope.tools.adversarial_eval")
    progress("adversarial_done", 0.68, {"adversarial": state.get("adversarial_results")})

    progress("geval", 0.72)
    node("agentscope.tools.geval_tool")
    progress("geval_done", 0.88, {"geval": state.get("geval_results")})

    progress("cost_analyzer", 0.90)
    node("agentscope.tools.cost_analyzer")
    progress("cost_done", 0.94, {"cost": state.get("cost_results")})

    progress("compile_report", 0.96)
    node("agentscope.report.compiler", "run_compiler")

    report = state["final_report"]["eval_results"]
    progress("done", 1.0, {
        "trace": report.get("trace"),
        "ir": report.get("ir"),
        "behavior": report.get("behavior"),
        "geval": report.get("geval"),
        "cost": report.get("cost"),
        "adversarial": report.get("adversarial"),
    })
    return state
