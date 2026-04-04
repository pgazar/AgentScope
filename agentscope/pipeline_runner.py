from __future__ import annotations

import importlib

from agentscope.logging_setup import configure_logging
from agentscope.otel import mark_span_error, mark_span_ok, set_span_attributes, start_span


def execute_pipeline(state: dict, progress_callback=None) -> dict:
    from agentscope.config import load_config
    from agentscope.orchestrator.graph import _run_agent

    cfg = load_config()
    state["config"] = cfg.model_dump()
    log = configure_logging(run_id=state.get("run_id"), component="pipeline")

    def progress(stage: str, pct: float, data: dict | None = None):
        if progress_callback:
            progress_callback(stage, pct, data or {})

    base_attrs = {
        "agentscope.run_id": state.get("run_id"),
        "agentscope.agent_type": state.get("agent_type"),
        "agentscope.turn_type": state.get("turn_type"),
    }

    def node(module: str, fn: str = "run", stage_name: str | None = None):
        span_name = stage_name or module.split(".")[-1]
        with start_span(
            f"agentscope.stage.{span_name}",
            tracer_name="agentscope.pipeline",
            attributes={
                **base_attrs,
                "agentscope.stage": span_name,
                "agentscope.active_tools": state.get("active_tools", []),
            },
        ) as span:
            try:
                log.info("stage_started", stage=span_name)
                mod = importlib.import_module(module)
                updates = getattr(mod, fn)(state)
                state.update(updates)
                set_span_attributes(span, {
                    "agentscope.stage.status": "ok",
                    "agentscope.traces_count": len(state.get("traces", [])),
                })
                log.info("stage_completed", stage=span_name, updated_keys=sorted(list(updates.keys())))
                mark_span_ok(span)
            except Exception as e:
                mark_span_error(span, e)
                log.error("stage_failed", stage=span_name, error_type=type(e).__name__, error_message=str(e))
                raise

    progress("starting", 0.02)

    progress("running_agent", 0.08)
    with start_span(
        "agentscope.stage.run_agent",
        tracer_name="agentscope.pipeline",
        attributes={
            **base_attrs,
            "agentscope.stage": "run_agent",
            "agentscope.eval_inputs_count": len(state.get("eval_inputs", [])),
        },
    ) as span:
        try:
            log.info("stage_started", stage="run_agent", eval_inputs_count=len(state.get("eval_inputs", [])))
            state.update(_run_agent(state))
            set_span_attributes(span, {
                "agentscope.stage.status": "ok",
                "agentscope.traces_count": len(state.get("traces", [])),
            })
            log.info("stage_completed", stage="run_agent", traces_count=len(state.get("traces", [])))
            mark_span_ok(span)
        except Exception as e:
            mark_span_error(span, e)
            log.error("stage_failed", stage="run_agent", error_type=type(e).__name__, error_message=str(e))
            raise
    progress("agent_done", 0.18, {"traces_count": len(state["traces"])})

    active = state["active_tools"]

    if "synth_gen" in active:
        progress("synth_gen", 0.25)
        node("agentscope.tools.synth_gen", stage_name="synth_gen")

    if "ir_evaluator" in active:
        progress("ir_evaluator", 0.32)
        node("agentscope.tools.ir_evaluator", stage_name="ir_evaluator")
        progress("ir_done", 0.38, {"ir": state.get("ir_results")})

    progress("agent_behavior", 0.42)
    node("agentscope.tools.agent_behavior", stage_name="agent_behavior")
    progress("behavior_done", 0.52, {"behavior": state.get("behavior_results")})

    progress("adversarial_eval", 0.56)
    node("agentscope.tools.adversarial_eval", stage_name="adversarial_eval")
    progress("adversarial_done", 0.68, {"adversarial": state.get("adversarial_results")})

    progress("geval", 0.72)
    node("agentscope.tools.geval_tool", stage_name="geval")
    progress("geval_done", 0.88, {"geval": state.get("geval_results")})

    progress("cost_analyzer", 0.90)
    node("agentscope.tools.cost_analyzer", stage_name="cost_analyzer")
    progress("cost_done", 0.94, {"cost": state.get("cost_results")})

    progress("compile_report", 0.96)
    node("agentscope.report.compiler", "run_compiler", stage_name="compile_report")

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
