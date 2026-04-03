import asyncio
import warnings

from deepeval.metrics import GEval, ConversationalGEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ConversationalTestCase, Turn

from agentscope.judge.criteria import (
    SINGLE_TURN_CRITERIA,
    MULTI_TURN_CRITERIA,
    HELPFULNESS_CRITERIA,
    SAFETY_CRITERIA,
)
from agentscope.judge.variance import (
    ScoredMetric,
    measure_inter_judge_variance,
    measure_calibration_drift,
)
from agentscope.judge.model import build_model as _build_model
from agentscope.orchestrator.state import AgentState


def build_single_turn_metrics(model_name: str) -> list[GEval]:
    model  = _build_model(model_name)
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


def _measure(metric: GEval, tc: LLMTestCase) -> float:
    """Runs a_measure() in a fresh event loop to avoid conflicts with Gradio's loop."""
    try:
        return asyncio.run(metric.a_measure(tc))
    except Exception as e:
        warnings.warn(f"G-Eval {metric.name} failed: {e}")
        return 0.0


def run(state: AgentState) -> AgentState:
    import logging
    log = logging.getLogger("agentscope.geval")
    log.info(f"geval starting — {len(state['traces'])} traces, turn_type={state['turn_type']}")

    model_name = state["config"]["judge"]["model"]
    turn_type  = state["turn_type"]
    responses  = _extract_responses(state["traces"])

    max_judge  = state["config"]["eval"].get("max_geval_responses", 1)
    budget_usd = state["config"]["eval"].get("eval_budget_usd", 2.00)
    judge_cost = 0.0

    # Multi-turn needs all turns — never truncate
    if turn_type == "single" and len(responses) > max_judge:
        warnings.warn(f"G-Eval: truncating {len(responses)} → {max_judge} responses.")
        responses = responses[:max_judge]

    if not responses:
        state["geval_results"] = {
            "scores": {}, "variance": [], "drift": [], "_judge_cost_est_usd": 0.0
        }
        return state

    test_cases = [
        LLMTestCase(
            input=r["input"],
            actual_output=r["output"],
            retrieval_context=r["context"] if r["context"] else [],
        )
        for r in responses
    ]

    scores: dict[str, list[float]] = {}
    # Built inside the loop so criteria and test_cases are in scope for variance
    scored_metrics: list[ScoredMetric] = []

    if turn_type == "single":
        metrics = build_single_turn_metrics(model_name)
        for m in metrics:
            if judge_cost >= budget_usd:
                warnings.warn(f"G-Eval: budget ${budget_usd} reached, stopping.")
                break
            metric_scores = []
            for tc in test_cases:
                s = _measure(m, tc)
                metric_scores.append(s)
                judge_cost += 800 * 3e-6 + 300 * 15e-6
            if metric_scores:
                scores[m.name] = metric_scores
            log.info(f"geval {m.name}: {metric_scores}")

            # Build ScoredMetric here — m.criteria and test_cases are both available
            scored_metrics.append(ScoredMetric(
                name=m.name,
                criteria=m.criteria,       # actual criteria string from the GEval object
                test_cases=test_cases,     # same test cases used for primary scoring
                primary_scores=metric_scores,
                primary_model=model_name,
            ))
    else:
        turns = []
        for r in responses:
            turns.append(Turn(role="user",      content=r["input"]))
            turns.append(Turn(role="assistant", content=r["output"]))
        tc = ConversationalTestCase(turns=turns)
        model = _build_model(model_name)
        for name, criteria in MULTI_TURN_CRITERIA.items():
            try:
                m = ConversationalGEval(name=name, criteria=criteria, model=model)
                asyncio.run(m.a_measure(tc))
                scores[name] = [m.score]
                # Conversational metrics have no LLMTestCase list — variance skipped
                scored_metrics.append(ScoredMetric(
                    name=name,
                    criteria=criteria,
                    test_cases=[],
                    primary_scores=[m.score],
                    primary_model=model_name,
                ))
            except Exception as e:
                warnings.warn(f"G-Eval {name} failed: {e}")

    # Inter-judge variance — only for single-turn metrics that have test cases
    # Runs the same test cases through gpt-4o-mini and measures score deltas
    variance_results = []
    for sm in scored_metrics:
        if sm.test_cases:  # skip conversational metrics — no LLMTestCase list
            try:
                variance_results.append(measure_inter_judge_variance(sm))
            except Exception as e:
                warnings.warn(f"variance measurement failed for {sm.name}: {e}")

    # Calibration drift — compares current distribution against a prior run's baseline
    baseline = state.get("baseline_geval_scores") or {}
    drift_results = [
        measure_calibration_drift(sm.name, sm.primary_scores, baseline.get(sm.name, []))
        for sm in scored_metrics
    ]

    state["geval_results"] = {
        "scores":              {k: round(sum(v) / len(v), 4) for k, v in scores.items() if v},
        "variance":            variance_results,
        "drift":               drift_results,
        "_judge_cost_est_usd": round(judge_cost, 4),
    }
    log.info(f"geval done: scores={state['geval_results']['scores']}, "
             f"variance_metrics={len(variance_results)}")
    return state
