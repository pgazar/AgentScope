import re
import yaml
from difflib import SequenceMatcher
from typing import Optional

from agentscope.orchestrator.state import AgentState


# --------------------------------------------------------------------------- #
# Deterministic trajectory metrics                                             #
# --------------------------------------------------------------------------- #

def tool_selection_accuracy(traces: list, expected_tools: list[str]) -> Optional[float]:
    # No reference set means we can't score — return None rather than a misleading 0
    if not expected_tools:
        return None
    called = [e.tool_name for e in traces if hasattr(e, "event_type") and e.event_type == "tool_start"]
    if not called:
        return 0.0
    correct = sum(1 for c in called if c in expected_tools)
    return correct / len(called)


def step_budget_efficiency(actual_steps: int, max_steps: int) -> float:
    # Heuristic only — not ground-truth validated; measures how far under budget the agent ran
    if actual_steps <= 0:
        return 1.0
    return min(1.0, max_steps / actual_steps)


def convergence(traces: list, max_steps: int) -> float:
    tool_calls = [e for e in traces if hasattr(e, "event_type") and e.event_type == "tool_start"]
    return 1.0 if len(tool_calls) <= max_steps else 0.0


def step_match(actual: list[str], reference: list[str], ordered: bool = True) -> dict:
    if not reference:
        return {"exact": None, "precision": None, "recall": None}
    if ordered:
        ratio = SequenceMatcher(None, actual, reference).ratio()
        matched = int(ratio * max(len(actual), len(reference)))
    else:
        matched = len(set(actual) & set(reference))
    return {
        "exact": actual == reference,
        "precision": matched / len(actual) if actual else 0.0,
        "recall": matched / len(reference),
    }


# --------------------------------------------------------------------------- #
# G-Eval scored metrics                                                        #
# --------------------------------------------------------------------------- #

def argument_correctness(traces: list, model: str) -> float:
    """
    G-Eval scores whether tool call parameters were correct
    and relevant given the input that triggered the call.
    """
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from agentscope.judge.criteria import PLAN_SUCCESS_CRITERIA

    tool_calls = [e for e in traces if hasattr(e, "event_type") and e.event_type == "tool_start"]
    if not tool_calls:
        return 0.0

    metric = GEval(
        name="argument_correctness",
        criteria=PLAN_SUCCESS_CRITERIA["argument_correctness"],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model,
    )
    scores = []
    for call in tool_calls:
        tc = LLMTestCase(
            input=str(call.tool_args),
            actual_output=f"Tool: {call.tool_name} | Args: {call.tool_args}",
        )
        metric.measure(tc)
        scores.append(metric.score)
    return sum(scores) / len(scores)


def plan_success(traces: list, model: str) -> float:
    """
    G-Eval scores whether the agent's overall tool sequence was
    coherent and appropriate for the task.
    """
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from agentscope.judge.criteria import PLAN_SUCCESS_CRITERIA

    tool_sequence = [e.tool_name for e in traces if hasattr(e, "event_type") and e.event_type == "tool_start"]
    if not tool_sequence:
        return 0.0

    metric = GEval(
        name="plan_success",
        criteria=PLAN_SUCCESS_CRITERIA["plan_success"],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model,
    )
    tc = LLMTestCase(
        input="Evaluate the agent's tool execution plan",
        actual_output=f"Tool sequence executed: {' → '.join(tool_sequence)}",
    )
    metric.measure(tc)
    return metric.score


def handoff_correctness(traces: list, model: str) -> float:
    """
    For multi-agent systems only. G-Eval scores whether each handoff
    passed accurate context and the receiving agent acted on it correctly.
    Covers context-passing accuracy only — conflict resolution and shared
    memory consistency are out of v1 scope.
    """
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from agentscope.judge.criteria import PLAN_SUCCESS_CRITERIA

    handoffs = [e for e in traces if hasattr(e, "event_type") and e.event_type == "handoff"]
    if not handoffs:
        # No handoffs = not applicable; return neutral score
        return 1.0

    metric = GEval(
        name="handoff_correctness",
        criteria=PLAN_SUCCESS_CRITERIA["handoff_correctness"],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model,
    )
    scores = []
    for h in handoffs:
        ctx = h.tool_args.get("context", {})
        tc = LLMTestCase(
            input=f"Context passed: {ctx}",
            actual_output=f"To agent: {h.tool_name} | Context: {ctx}",
        )
        metric.measure(tc)
        scores.append(metric.score)
    return sum(scores) / len(scores)


# --------------------------------------------------------------------------- #
# Permission validation                                                        #
# --------------------------------------------------------------------------- #

def _normalize_tool_name(name: str) -> str:
    """
    CamelCase → snake_case, hyphens/spaces → underscores, lowercase.
    SendEmail → send_email, RAGRetrieve → rag_retrieve, rag-retrieve → rag_retrieve
    """
    s = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', name)
    s = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', '_', s)
    return re.sub(r'[-\s]+', '_', s).lower()


def load_permission_schema(path: str = "permissions.yaml") -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)["tools"]
    # Pre-normalize all keys so lookups never have casing mismatches
    return {_normalize_tool_name(k): v for k, v in raw.items()}


def permission_validation(traces: list, schema: dict) -> dict:
    """
    Checks every tool call against the permission schema using normalized names.
    Unknown tools are flagged as warnings, not violations — custom agents may
    use tools not listed in the schema.
    """
    tool_calls = [e for e in traces if hasattr(e, "event_type") and e.event_type == "tool_start"]
    violations = []
    warnings = []
    severity_sum = 0.0

    for call in tool_calls:
        normalized = _normalize_tool_name(call.tool_name)
        policy = schema.get(normalized)

        if policy is None:
            warnings.append({
                "tool": call.tool_name,
                "reason": "tool not in permission schema — review recommended",
            })
            continue

        if not policy.get("allowed", True):
            violations.append({"tool": call.tool_name, "reason": "tool not permitted", "severity": 1.0})
            severity_sum += 1.0
        elif policy.get("sensitive", False):
            violations.append({"tool": call.tool_name, "reason": "sensitive operation executed", "severity": 0.5})
            severity_sum += 0.5

    total = len(tool_calls)
    return {
        "permission_violation_rate": round(len(violations) / total, 3) if total else 0.0,
        "safety_score": round(max(0.0, 1.0 - (severity_sum / max(total, 1))), 3),
        "violations": violations,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# LangGraph node                                                               #
# --------------------------------------------------------------------------- #

def run(state: AgentState) -> AgentState:
    # Flatten all trace events from every AgentTrace run
    all_events = []
    for trace in state["traces"]:
        all_events.extend(trace.events if hasattr(trace, "events") else [])

    max_steps = state["config"]["eval"]["max_steps"]
    model = state["config"]["judge"]["model"]
    agent_type = state["agent_type"]
    expected = state.get("expected_tools", [])

    tool_steps = [e for e in all_events if hasattr(e, "event_type") and e.event_type == "tool_start"]

    schema = load_permission_schema()

    state["behavior_results"] = {
        "tool_accuracy": tool_selection_accuracy(all_events, expected),
        "plan_success": plan_success(all_events, model),
        # Heuristic — not ground-truth validated
        "step_budget_efficiency": step_budget_efficiency(len(tool_steps), max_steps),
        "arg_correctness": argument_correctness(all_events, model),
        "convergence": convergence(all_events, max_steps),
        # handoff_correctness covers context passing only (v1 scope)
        "handoff_correctness": handoff_correctness(all_events, model) if agent_type == "multi_agent" else None,
        "step_match": step_match([], []),
        "permission_validation": permission_validation(all_events, schema),
    }
    return state
