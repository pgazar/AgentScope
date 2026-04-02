"""
Phase 11 smoke test — runs the full AgentScope pipeline end to end
with the fake agent and writes results to smoke_output.txt.
"""
import uuid, json, os, sys, traceback

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run():
    from agentscope.config import load_config
    from agentscope.intake.intake_agent import IntakeAgent
    from agentscope.orchestrator.graph import build_graph

    cfg = load_config()
    intake = IntakeAgent().run({
        "agent_type": "tool_use",
        "turn_type":  "single",
        "has_gt":     "no",
        "kb_format":  "none",
    })

    print("active_tools:", intake["active_tools"])

    graph = build_graph(intake["active_tools"])

    run_id = str(uuid.uuid4())[:8]
    state = {
        "run_id":               run_id,
        "agent_folder":         "tests/fake_agent",
        "agent_model":          "claude-sonnet-4-5",
        "eval_inputs":          ["What is the answer?"],
        "kb_path":              None,
        "gt_path":              None,
        "agent_type":           "tool_use",
        "turn_type":            "single",
        "active_tools":         intake["active_tools"],
        "expected_tools":       [],
        "traces":               [],
        "baseline_geval_scores": {},
        "ir_results":           None,
        "behavior_results":     None,
        "geval_results":        None,
        "cost_results":         None,
        "adversarial_results":  None,
        "synth_results":        None,
        "final_report":         None,
        "config":               cfg.model_dump(),
    }

    print(f"run_id: {run_id}")
    print("invoking graph...")
    result = graph.invoke(state)

    report_path = f"outputs/{run_id}_run_report.json"
    assert os.path.exists(report_path), f"report not written: {report_path}"

    with open(report_path) as f:
        report = json.load(f)

    print("run_id:", result["run_id"])
    print("report written to:", report_path)
    print("behavior_results:", json.dumps(result.get("behavior_results", {}), indent=2, default=str))
    print("geval scores:", json.dumps(result.get("geval_results", {}).get("scores", {}), indent=2))
    print("cost_results:", json.dumps(result.get("cost_results", {}), indent=2))
    print("adversarial summary:", {
        k: v for k, v in (result.get("adversarial_results") or {}).items()
        if k != "by_category"
    })
    print("\nSMOKE TEST PASSED")

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print("\nSMOKE TEST FAILED")
        traceback.print_exc()
        sys.exit(1)
