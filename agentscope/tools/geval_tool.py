import asyncio
import os
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


def _measure(metric: GEval, tc: LLMTestCase) -> tuple[float | None, str | None]:
    """Runs a_measure() in a fresh event loop to avoid conflicts with Gradio's loop."""
    try:
        asyncio.run(metric.a_measure(tc))
        return metric.score, None
    except Exception as e:
        warnings.warn(f"G-Eval {metric.name} failed: {e}")
        return None, str(e)


def run(state: AgentState) -> AgentState:
    import logging
    log = logging.getLogger("agentscope.geval")
    log.info(f"geval starting — {len(state['traces'])} traces, turn_type={state['turn_type']}")

    model_name = state["config"]["judge"]["model"]
    turn_type  = state["turn_type"]
    responses  = _extract_responses(state["traces"])
    trace_diag = state.get("trace_diagnostics") or {}
    total_responses = len(responses)

    max_judge  = state["config"]["eval"].get("max_geval_responses", 1)
    budget_usd = state["config"]["eval"].get("eval_budget_usd", 2.00)
    judge_cost = 0.0
    infra_failures: list[dict] = []
    budget_reached = False
    variance_enabled = os.environ.get("AGENTSCOPE_VARIANCE", "1").lower() not in {"0", "false", "no"}

    # Multi-turn needs all turns — never truncate
    if turn_type == "single" and len(responses) > max_judge:
        warnings.warn(f"G-Eval: truncating {len(responses)} → {max_judge} responses.")
        responses = responses[:max_judge]

    if not responses:
        state["geval_results"] = {
            "scores": {},
            "variance": [],
            "drift": [],
            "infra_failures": [],
            "coverage": {
                "total_responses": 0,
                "judged_responses": 0,
                "responses_skipped": 0,
                "budget_reached": False,
            },
            "_judge_cost_est_usd": 0.0,
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
                budget_reached = True
                break
            metric_scores = []
            successful_cases = []
            for tc in test_cases:
                s, error = _measure(m, tc)
                if s is not None:
                    metric_scores.append(s)
                    successful_cases.append(tc)
                else:
                    infra_failures.append({
                        "metric": m.name,
                        "stage": "primary_judge",
                        "error": error,
                    })
                judge_cost += 800 * 3e-6 + 300 * 15e-6
            scores[m.name] = metric_scores if metric_scores else []
            log.info(f"geval {m.name}: {metric_scores}")

            # Build ScoredMetric here — m.criteria and test_cases are both available
            scored_metrics.append(ScoredMetric(
                name=m.name,
                criteria=m.criteria,       # actual criteria string from the GEval object
                test_cases=successful_cases,
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
                scores[name] = []
                infra_failures.append({
                    "metric": name,
                    "stage": "primary_judge",
                    "error": str(e),
                })

    # Inter-judge variance — only for single-turn metrics that have test cases
    # Runs the same test cases through gpt-4o-mini and measures score deltas
    variance_results = []
    for sm in scored_metrics:
        if variance_enabled and sm.test_cases and sm.primary_scores:
            try:
                variance_results.append(measure_inter_judge_variance(sm))
            except Exception as e:
                warnings.warn(f"variance measurement failed for {sm.name}: {e}")
                infra_failures.append({
                    "metric": sm.name,
                    "stage": "variance",
                    "error": str(e),
                })

    # Calibration drift — compares current distribution against a prior run's baseline
    baseline = state.get("baseline_geval_scores") or {}
    drift_results = [
        measure_calibration_drift(sm.name, sm.primary_scores, baseline.get(sm.name, []))
        for sm in scored_metrics
    ]

    judged_responses = max((len(v) for v in scores.values()), default=0)

    state["geval_results"] = {
        "scoreable":           True,
        "trace_status":        trace_diag.get("status"),
        "scores":              {
            k: (round(sum(v) / len(v), 4) if v else None)
            for k, v in scores.items()
        },
        "variance":            variance_results,
        "drift":               drift_results,
        "infra_failures":      infra_failures,
        "coverage": {
            "total_responses": total_responses,
            "judged_responses": judged_responses,
            "responses_skipped": max(0, total_responses - judged_responses),
            "budget_reached": budget_reached,
            "variance_enabled": variance_enabled,
        },
        "_judge_cost_est_usd": round(judge_cost, 4),
    }
    log.info(f"geval done: scores={state['geval_results']['scores']}, "
             f"variance_metrics={len(variance_results)}")
    return state
