import os
import datetime

from agentscope.orchestrator.state import AgentState
from agentscope.run_store import report_path, save_report


class ReportCompiler:
    def run(self, state: AgentState) -> AgentState:
        run_id = state["run_id"]
        report = {
            "run_id": run_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "agent_folder": state["agent_folder"],
            "agent_type": state["agent_type"],
            "turn_type": state["turn_type"],
            "config": state["config"],
            "trace_diagnostics": state.get("trace_diagnostics"),
            "eval_results": {
                "trace":       state.get("trace_diagnostics"),
                "ir":          state.get("ir_results"),
                "behavior":    state.get("behavior_results"),
                "geval":       state.get("geval_results"),
                "cost":        state.get("cost_results"),
                "adversarial": state.get("adversarial_results"),
            },
            # AgentScope's own judge token usage and Modal costs — not the evaluated agent's cost
            "agentscope_cost": {
                "note": "G-Eval judge token usage and Modal invocation costs",
                "saved_to": report_path(run_id),
            },
        }

        os.makedirs("outputs", exist_ok=True)
        save_report(run_id, report)

        state["final_report"] = report
        return state


def run_compiler(state: AgentState) -> AgentState:
    """Module-level wrapper so graph.py can reference this as a plain function."""
    return ReportCompiler().run(state)
