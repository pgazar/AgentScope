"""
LangGraph StateGraph for AgentScope evaluation pipeline.

Execution order (sequential):
    run_agent → [synth_gen →] [ir_evaluator →] agent_behavior
              → adversarial_eval → geval → cost_analyzer → compile_report → END

State contract per node:
    run_agent:       reads agent_folder, eval_inputs, agent_model   → writes traces
    synth_gen:       reads kb_path, config                          → writes gt_path, synth_results
    ir_evaluator:    reads traces, gt_path, config                  → writes ir_results
    agent_behavior:  reads traces, expected_tools, config           → writes behavior_results
    adversarial_eval:reads agent_folder, config                     → writes adversarial_results
    geval:           reads traces, config, turn_type                → writes geval_results
    cost_analyzer:   reads traces, agent_model, geval_results       → writes cost_results
    compile_report:  reads all result keys                          → writes final_report
"""

from langgraph.graph import StateGraph, END

from agentscope.orchestrator.state import AgentState
from agentscope.runner import AgentRunner


def _run_agent(state: AgentState) -> AgentState:
    runner = AgentRunner(
        agent_folder=state["agent_folder"],
        agent_model=state.get("agent_model", "unknown"),
    )
    traces = []
    for inp in state["eval_inputs"]:
        trace = runner.run(inp, state["run_id"])
        traces.append(trace)
    state["traces"] = traces
    return state


def _load_node(module_path: str, fn_name: str = "run"):
    """Lazily imports a tool's run() so missing deps only fail at execution time."""
    def node(state: AgentState) -> AgentState:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, fn_name)(state)
    node.__name__ = module_path.split(".")[-1]
    return node


def build_graph(active_tools: list[str]) -> "CompiledGraph":
    graph = StateGraph(AgentState)

    # run_agent is always the entry point
    graph.add_node("run_agent", _run_agent)

    if "synth_gen" in active_tools:
        graph.add_node("synth_gen", _load_node("agentscope.tools.synth_gen"))
    if "ir_evaluator" in active_tools:
        graph.add_node("ir_evaluator", _load_node("agentscope.tools.ir_evaluator"))

    graph.add_node("agent_behavior",  _load_node("agentscope.tools.agent_behavior"))
    graph.add_node("adversarial_eval", _load_node("agentscope.tools.adversarial_eval"))
    graph.add_node("geval",           _load_node("agentscope.tools.geval_tool"))
    graph.add_node("cost_analyzer",   _load_node("agentscope.tools.cost_analyzer"))
    graph.add_node("compile_report",  _load_node("agentscope.report.compiler", "run_compiler"))

    graph.set_entry_point("run_agent")

    # Wire run_agent → first optional node or straight to agent_behavior
    if "synth_gen" in active_tools:
        graph.add_edge("run_agent", "synth_gen")
        graph.add_edge("synth_gen", "ir_evaluator")
        graph.add_edge("ir_evaluator", "agent_behavior")
    elif "ir_evaluator" in active_tools:
        graph.add_edge("run_agent", "ir_evaluator")
        graph.add_edge("ir_evaluator", "agent_behavior")
    else:
        graph.add_edge("run_agent", "agent_behavior")

    graph.add_edge("agent_behavior",  "adversarial_eval")
    graph.add_edge("adversarial_eval", "geval")
    graph.add_edge("geval",           "cost_analyzer")
    graph.add_edge("cost_analyzer",   "compile_report")
    graph.add_edge("compile_report",  END)

    return graph.compile()
