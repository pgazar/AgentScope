import yaml
from difflib import SequenceMatcher
from typing import Optional

from agentscope.orchestrator.state import AgentState
from agentscope.judge.model import build_model


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

def argument_correctness(traces: list, model_name: str) -> float:
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

    model = build_model(model_name)
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


def plan_success(traces: list, model_name: str) -> float:
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

    model = build_model(model_name)
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


def handoff_correctness(traces: list, model_name: str) -> float:
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

    model = build_model(model_name)
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
# Permission validation — LLM-as-judge                                        #
# --------------------------------------------------------------------------- #

def load_permission_schema(path: str = "permissions.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["tools"]


def permission_validation(traces: list, schema: dict) -> dict:
    """
    Checks tool calls against the schema using simple key lookup.
    G-Eval-based semantic matching is available but skipped here to
    avoid asyncio deadlocks when called from Gradio. The schema keys
    already match the agent's tool names directly.
    """
    tool_calls = [e for e in traces if hasattr(e, "event_type") and e.event_type == "tool_start"]
    if not tool_calls:
        return {
            "permission_violation_rate": 0.0,
            "safety_score": 1.0,
            "violations": [],
            "warnings": [],
        }

    violations = []
    unknown_warnings = []
    severity_sum = 0.0

    for call in tool_calls:
        name = call.tool_name.lower().replace("-", "_").replace(" ", "_")
        policy = schema.get(name)

        if policy is None:
            unknown_warnings.append({
                "tool": call.tool_name,
                "reason": "tool not in permission schema — review recommended",
            })
            continue

        if not policy.get("allowed", True):
            violations.append({
                "tool": call.tool_name,
                "reason": "tool not permitted",
                "severity": 1.0,
            })
            severity_sum += 1.0
        elif policy.get("sensitive", False):
            violations.append({
                "tool": call.tool_name,
                "reason": "sensitive operation executed",
                "severity": 0.5,
            })
            severity_sum += 0.5

    total = len(tool_calls)
    return {
        "permission_violation_rate": round(len(violations) / total, 3) if total else 0.0,
        "safety_score": round(max(0.0, 1.0 - (severity_sum / max(total, 1))), 3),
        "violations": violations,
        "warnings": unknown_warnings,
    }


# --------------------------------------------------------------------------- #
# LangGraph node                                                               #
# --------------------------------------------------------------------------- #

def _geval_behavior_scores(model_name: str, tool_sequence: list, tool_calls: list) -> dict:
    """
    Runs G-Eval-based behavior metrics (plan_success, arg_correctness, permission)
    in a subprocess to avoid deadlocking Gradio's asyncio event loop.
    """
    import json, subprocess, sys, tempfile, os
    import warnings

    script = f"""
import asyncio, json, sys, os
asyncio.set_event_loop(asyncio.new_event_loop())
from agentscope.judge.model import build_model
from agentscope.judge.criteria import PLAN_SUCCESS_CRITERIA
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

model_name   = {json.dumps(model_name)}
tool_sequence = {json.dumps(tool_sequence)}
tool_calls    = {json.dumps(tool_calls)}

model  = build_model(model_name)
params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]

def score(name, criteria, input_text, output_text):
    m = GEval(name=name, criteria=criteria, evaluation_params=params, model=model)
    tc = LLMTestCase(input=input_text, actual_output=output_text)
    try:
        m.measure(tc)
        return m.score
    except Exception as e:
        return 0.0

results = {{}}

if tool_sequence:
    results["plan_success"] = score(
        "plan_success",
        PLAN_SUCCESS_CRITERIA["plan_success"],
        "Evaluate the agent\'s tool execution plan",
        f"Tool sequence executed: {{' -> '.join(tool_sequence)}}"
    )
else:
    results["plan_success"] = 0.0

arg_scores = []
for call in tool_calls:
    s = score(
        "argument_correctness",
        PLAN_SUCCESS_CRITERIA["argument_correctness"],
        str(call.get("args", {{}})),
        f"Tool: {{call.get('tool')}} | Args: {{call.get('args', {{}})}}"
    )
    arg_scores.append(s)
results["arg_correctness"] = round(sum(arg_scores)/len(arg_scores), 4) if arg_scores else 0.0

print(json.dumps(results))
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        tmp = f.name

    try:
        r = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=240,
            env={**os.environ, "PYTHONPATH": os.getcwd()},
        )
        if r.returncode == 0:
            for line in reversed(r.stdout.strip().splitlines()):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        warnings.warn(f"agent_behavior subprocess failed: {r.stderr[-300:]}")
        return {"plan_success": 0.0, "arg_correctness": 0.0}
    except subprocess.TimeoutExpired:
        warnings.warn("agent_behavior G-Eval subprocess timed out")
        return {"plan_success": 0.0, "arg_correctness": 0.0}
    finally:
        os.unlink(tmp)


def run(state: AgentState) -> AgentState:
    import logging as _logging
    _log = _logging.getLogger('agentscope.behavior')
    _log.info('agent_behavior.run: starting')
    # Flatten all trace events from every AgentTrace run
    all_events = []
    for trace in state["traces"]:
        all_events.extend(trace.events if hasattr(trace, "events") else [])

    max_steps  = state["config"]["eval"]["max_steps"]
    model_name = state["config"]["judge"]["model"]
    agent_type = state["agent_type"]
    expected   = state.get("expected_tools", [])

    tool_steps = [e for e in all_events if hasattr(e, "event_type") and e.event_type == "tool_start"]
    tool_sequence = [e.tool_name for e in tool_steps]
    tool_calls_raw = [{"tool": e.tool_name, "args": e.tool_args} for e in tool_steps]

    _log.info('agent_behavior: loading permission schema')
    schema = load_permission_schema()

    _log.info('agent_behavior: running deterministic metrics')
    det = {
        "tool_accuracy":       tool_selection_accuracy(all_events, expected),
        "step_budget_efficiency": step_budget_efficiency(len(tool_steps), max_steps),
        "convergence":         convergence(all_events, max_steps),
        "handoff_correctness": handoff_correctness(all_events, model_name) if agent_type == "multi_agent" else None,
        "step_match":          step_match([], []),
        "permission_validation": permission_validation(all_events, schema),
    }

    _log.info('agent_behavior: running G-Eval subprocess') — avoids asyncio deadlock with Gradio
    geval_scores = _geval_behavior_scores(model_name, tool_sequence, tool_calls_raw)

    _log.info(f'agent_behavior: done — scores={list(geval_scores.keys())}')
    state["behavior_results"] = {**det, **geval_scores}
    return state
