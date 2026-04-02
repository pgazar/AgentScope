def resolve_tool_set(agent_type: str, has_kb: bool, has_gt: bool) -> list[str]:
    # adversarial_eval runs for all agent types — always included
    tools = ["agent_behavior", "geval", "cost_analyzer", "adversarial_eval"]

    if has_kb and has_gt:
        tools.insert(0, "ir_evaluator")
    elif has_kb and not has_gt:
        # No ground truth — synthesize it first, then evaluate retrieval
        tools.insert(0, "synth_gen")
        tools.insert(1, "ir_evaluator")

    return tools
