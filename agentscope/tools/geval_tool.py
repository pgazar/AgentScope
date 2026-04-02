import os
import json
import warnings
import subprocess
import sys
import tempfile

from deepeval.metrics import GEval, ConversationalGEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ConversationalTestCase, Turn

from agentscope.judge.criteria import (
    SINGLE_TURN_CRITERIA,
    MULTI_TURN_CRITERIA,
    HELPFULNESS_CRITERIA,
    SAFETY_CRITERIA,
)
from agentscope.judge.variance import ScoredMetric, measure_calibration_drift
from agentscope.judge.model import build_model as _build_model
from agentscope.orchestrator.state import AgentState


def build_single_turn_metrics(model_name: str) -> list[GEval]:
    model = _build_model(model_name)
    params = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ]
    return [
        GEval(name="task_completion", criteria=SINGLE_TURN_CRITERIA["task_completion"], evaluation_params=params, model=model),
        GEval(name="faithfulness",    criteria=SINGLE_TURN_CRITERIA["faithfulness"],    evaluation_params=params, model=model),
        GEval(name="hallucination",   criteria=SINGLE_TURN_CRITERIA["hallucination"],   evaluation_params=params, model=model),
        GEval(name="citation_acc",    criteria=SINGLE_TURN_CRITERIA["citation_acc"],    evaluation_params=params, model=model),
        GEval(name="helpfulness",     criteria=HELPFULNESS_CRITERIA,                    evaluation_params=params, model=model),
        GEval(name="safety",          criteria=SAFETY_CRITERIA,                         evaluation_params=params, model=model),
    ]


def _extract_responses(traces: list) -> list[dict]:
    responses = []
    for trace in traces:
        ctx = []
        for evt in getattr(trace, "events", []):
            if hasattr(evt, "event_type") and evt.event_type == "retrieval":
                ctx.extend(evt.retrieval_docs)
        responses.append({
            "input":   trace.agent_input,
            "output":  trace.agent_output,
            "context": [str(d) for d in ctx],
        })
    return responses


def _score_in_subprocess(model_name: str, responses: list[dict]) -> dict:
    """
    Runs G-Eval scoring in a subprocess to get a clean asyncio event loop,
    isolated from Gradio's running event loop which causes deadlocks.
    """
    script = f"""
import os, json, sys, asyncio

# fresh event loop
asyncio.set_event_loop(asyncio.new_event_loop())

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from agentscope.judge.model import build_model
from agentscope.judge.criteria import SINGLE_TURN_CRITERIA, HELPFULNESS_CRITERIA, SAFETY_CRITERIA

model_name = {json.dumps(model_name)}
responses  = {json.dumps(responses)}

model  = build_model(model_name)
params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT]

criteria_map = dict(**SINGLE_TURN_CRITERIA, helpfulness=HELPFULNESS_CRITERIA, safety=SAFETY_CRITERIA)
metric_names = ["task_completion", "faithfulness", "hallucination", "citation_acc", "helpfulness", "safety"]

scores = {{}}
for name in metric_names:
    m = GEval(name=name, criteria=criteria_map[name], evaluation_params=params, model=model)
    metric_scores = []
    for r in responses:
        tc = LLMTestCase(input=r["input"], actual_output=r["output"],
                         retrieval_context=r["context"] if r["context"] else [])
        try:
            m.measure(tc)
            metric_scores.append(m.score)
        except Exception as e:
            import warnings
            warnings.warn(f"G-Eval {{name}} failed: {{e}}")
    if metric_scores:
        scores[name] = round(sum(metric_scores) / len(metric_scores), 4)

print(json.dumps(scores))
"""
    # Write script to a temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "PYTHONPATH": os.getcwd()},
        )
        if result.returncode == 0:
            # Last line of stdout is the JSON scores
            for line in reversed(result.stdout.strip().splitlines()):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        warnings.warn(f"G-Eval subprocess failed: {result.stderr[-500:]}")
        return {}
    except subprocess.TimeoutExpired:
        warnings.warn("G-Eval subprocess timed out after 300s")
        return {}
    finally:
        os.unlink(tmp_path)


def run(state: AgentState) -> AgentState:
    import logging
    log = logging.getLogger("agentscope.geval")
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        log.addHandler(logging.StreamHandler())
    log.info(f"geval starting — {len(state['traces'])} traces, turn_type={state['turn_type']}")

    model_name = state["config"]["judge"]["model"]
    turn_type  = state["turn_type"]
    responses  = _extract_responses(state["traces"])

    max_judge  = state["config"]["eval"].get("max_geval_responses", 1)
    budget_usd = state["config"]["eval"].get("eval_budget_usd", 2.00)

    if len(responses) > max_judge:
        warnings.warn(f"G-Eval: truncating {len(responses)} → {max_judge} responses.")
        responses = responses[:max_judge]

    if not responses:
        state["geval_results"] = {"scores": {}, "variance": [], "drift": [], "_judge_cost_est_usd": 0.0}
        return state

    scores: dict[str, float] = {}

    if turn_type == "single":
        log.info(f"geval: scoring {len(responses)} response(s) via subprocess...")
        scores = _score_in_subprocess(model_name, responses)
        log.info(f"geval: scores = {scores}")
    else:
        try:
            turns = [Turn(role=t.get("role", "user"), content=t.get("content", t["output"])) for t in responses]
            tc    = ConversationalTestCase(turns=turns)
            model = _build_model(model_name)
            for name, criteria in MULTI_TURN_CRITERIA.items():
                m = ConversationalGEval(name=name, criteria=criteria, model=model)
                m.measure(tc)
                scores[name] = round(m.score, 4)
        except Exception as e:
            warnings.warn(f"G-Eval multi-turn failed: {e}")

    # Calibration drift (no variance — disabled by default)
    baseline = state.get("baseline_geval_scores") or {}
    scored_metrics = [
        ScoredMetric(name=k, criteria="", test_cases=[], primary_scores=[v], primary_model=model_name)
        for k, v in scores.items()
    ]
    drift_results = [
        measure_calibration_drift(sm.name, sm.primary_scores, baseline.get(sm.name, []))
        for sm in scored_metrics
    ]

    judge_cost = len(responses) * len(scores) * (800 * 3e-6 + 300 * 15e-6)

    state["geval_results"] = {
        "scores":              scores,
        "variance":            [],
        "drift":               drift_results,
        "_judge_cost_est_usd": round(judge_cost, 4),
    }
    return state
