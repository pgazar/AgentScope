from typing import TypedDict, Optional


class AgentState(TypedDict):
    run_id: str
    agent_folder: str
    agent_model: str               # model name of the EVALUATED agent (not the judge)
    eval_inputs: list[str]         # query strings to run through the target agent
    kb_path: Optional[str]
    gt_path: Optional[str]
    agent_type: str                # "rag" | "tool_use" | "multi_agent" | "hybrid"
    turn_type: str                 # "single" | "multi"
    active_tools: list[str]
    expected_tools: list[str]      # tool names the agent is expected to call
    traces: list                   # list[AgentTrace] — populated by AgentRunner
    trace_diagnostics: Optional[dict]
    baseline_geval_scores: Optional[dict]  # {metric_name: [scores]} from a prior run
    ir_results: Optional[dict]
    behavior_results: Optional[dict]
    geval_results: Optional[dict]
    cost_results: Optional[dict]
    adversarial_results: Optional[dict]
    synth_results: Optional[dict]
    final_report: Optional[dict]
    config: dict
