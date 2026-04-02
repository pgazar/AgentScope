"""
Standalone evaluation runner — called as a subprocess by the dashboard.
Writes progress updates and final results to a JSON file.
Completely isolated from Gradio's asyncio event loop.
"""
import sys
import os
import uuid
import json
import time

AGENTSCOPE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AGENTSCOPE_ROOT)

# Load env files passed as args
for env_file in sys.argv[2:]:
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    k = k.strip()
                    if k.replace("_", "").isalnum():
                        os.environ.setdefault(k, v.strip())

output_file = sys.argv[1]

def write_progress(stage: str, pct: float, data: dict = None):
    payload = {"stage": stage, "pct": pct, "ts": time.time()}
    if data:
        payload.update(data)
    with open(output_file, "w") as f:
        json.dump(payload, f)


def main():
    import importlib

    write_progress("starting", 0.02)

    from agentscope.config import load_config
    from agentscope.intake.intake_agent import IntakeAgent
    from agentscope.orchestrator.graph import _run_agent

    # State is passed via env var as JSON
    state = json.loads(os.environ["AGENTSCOPE_STATE"])
    cfg   = load_config()
    state["config"] = cfg.model_dump()

    def node(module: str, fn: str = "run"):
        mod = importlib.import_module(module)
        updates = getattr(mod, fn)(state)
        state.update(updates)

    write_progress("running_agent", 0.08)
    state.update(_run_agent(state))
    write_progress("agent_done", 0.18, {"traces_count": len(state["traces"])})

    active = state["active_tools"]

    if "synth_gen" in active:
        write_progress("synth_gen", 0.25)
        node("agentscope.tools.synth_gen")

    if "ir_evaluator" in active:
        write_progress("ir_evaluator", 0.32)
        node("agentscope.tools.ir_evaluator")
        write_progress("ir_done", 0.38, {"ir": state.get("ir_results")})

    write_progress("agent_behavior", 0.42)
    node("agentscope.tools.agent_behavior")
    write_progress("behavior_done", 0.52, {"behavior": state.get("behavior_results")})

    write_progress("adversarial_eval", 0.56)
    node("agentscope.tools.adversarial_eval")
    write_progress("adversarial_done", 0.68, {"adversarial": state.get("adversarial_results")})

    write_progress("geval", 0.72)
    node("agentscope.tools.geval_tool")
    write_progress("geval_done", 0.88, {"geval": state.get("geval_results")})

    write_progress("cost_analyzer", 0.90)
    node("agentscope.tools.cost_analyzer")
    write_progress("cost_done", 0.94, {"cost": state.get("cost_results")})

    write_progress("compile_report", 0.96)
    node("agentscope.report.compiler", "run_compiler")

    report = state["final_report"]["eval_results"]
    write_progress("done", 1.0, {
        "ir":          report.get("ir"),
        "behavior":    report.get("behavior"),
        "geval":       report.get("geval"),
        "cost":        report.get("cost"),
        "adversarial": report.get("adversarial"),
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        write_progress("error", -1.0, {"error": str(e), "traceback": traceback.format_exc()})
        sys.exit(1)
