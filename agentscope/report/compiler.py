import json
import os
import datetime

from agentscope.orchestrator.state import AgentState


class ReportCompiler:
    def run(self, state: AgentState) -> AgentState:
        run_id = state["run_id"]
        report = {
            "run_id": run_id,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "agent_folder": state["agent_folder"],
            "agent_type": state["agent_type"],
            "turn_type": state["turn_type"],
            "config": state["config"],
            "eval_results": {
                "ir":          state.get("ir_results"),
                "behavior":    state.get("behavior_results"),
                "geval":       state.get("geval_results"),
                "cost":        state.get("cost_results"),
                "adversarial": state.get("adversarial_results"),
            },
            # AgentScope's own judge token usage and Modal costs — not the evaluated agent's cost
            "agentscope_cost": {
                "note": "G-Eval judge token usage and Modal invocation costs",
                "saved_to": f"outputs/{run_id}_run_report.json",
            },
        }

        os.makedirs("outputs", exist_ok=True)
        with open(f"outputs/{run_id}_run_report.json", "w") as f:
            json.dump(report, f, indent=2)

        state["final_report"] = report
        return state


def run_compiler(state: AgentState) -> AgentState:
    """Module-level wrapper so graph.py can reference this as a plain function."""
    return ReportCompiler().run(state)
