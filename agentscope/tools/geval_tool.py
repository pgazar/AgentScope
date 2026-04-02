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
from agentscope.judge.variance import ScoredMetric, measure_inter_judge_variance, measure_calibration_drift
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
    """Pulls agent input/output pairs and any retrieved context from each trace."""
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

    max_judge  = state["config"]["eval"].get("max_geval_responses", 100)
    budget_usd = state["config"]["eval"].get("eval_budget_usd", 2.00)
    judge_cost = 0.0

    if len(responses) > max_judge:
        warnings.warn(
            f"G-Eval: {len(responses)} responses exceed max_geval_responses={max_judge}. "
            f"Truncating to first {max_judge}."
        )
        responses = responses[:max_judge]

    # Build test cases once — reused by primary scoring and variance measurement
    test_cases: list[LLMTestCase] = []
    for r in responses:
        test_cases.append(LLMTestCase(
            input=r["input"],
            actual_output=r["output"],
            retrieval_context=r["context"] if r["context"] else [],
        ))

    scores: dict[str, list[float]] = {}
    scored_metrics: list[ScoredMetric] = []

    if turn_type == "single":
        metrics = build_single_turn_metrics(model_name)
        for m in metrics:
            if judge_cost >= budget_usd:
                warnings.warn(
                    f"G-Eval: eval_budget_usd={budget_usd} reached. Stopping early."
                )
                break

            metric_scores = []
            for tc in test_cases:
                try:
                    m.measure(tc)
                    metric_scores.append(m.score)
                    judge_cost += 800 * 3e-6 + 300 * 15e-6
                except Exception as e:
                    warnings.warn(f"G-Eval: {m.name} failed on a test case: {e}")

            if metric_scores:
                scores[m.name] = metric_scores

            scored_metrics.append(ScoredMetric(
                name=m.name,
                criteria=m.criteria,
                test_cases=test_cases,
                primary_scores=metric_scores,
                primary_model=model_name,
            ))

    else:
        turns = [Turn(role=t.get("role", "user"), content=t.get("content", t["output"])) for t in responses]
        tc    = ConversationalTestCase(turns=turns)
        model = _build_model(model_name)
        for name, criteria in MULTI_TURN_CRITERIA.items():
            try:
                m = ConversationalGEval(name=name, criteria=criteria, model=model)
                m.measure(tc)
                scores[name] = [m.score]
            except Exception as e:
                warnings.warn(f"G-Eval: {name} failed: {e}")
                scores[name] = []

            scored_metrics.append(ScoredMetric(
                name=name,
                criteria=criteria,
                test_cases=[],
                primary_scores=scores.get(name, []),
                primary_model=model_name,
            ))

    # Inter-judge variance disabled by default — enable by setting AGENTSCOPE_VARIANCE=1
    variance_results = []
    if os.environ.get("AGENTSCOPE_VARIANCE") == "1":
        for sm in scored_metrics:
            if sm.test_cases:
                try:
                    variance_results.append(measure_inter_judge_variance(sm))
                except Exception as e:
                    warnings.warn(f"variance measurement failed for {sm.name}: {e}")

    # Calibration drift — compare against prior run baseline if provided
    baseline     = state.get("baseline_geval_scores") or {}
    drift_results = []
    for sm in scored_metrics:
        drift_results.append(measure_calibration_drift(
            metric_name=sm.name,
            current_scores=sm.primary_scores,
            baseline_scores=baseline.get(sm.name, []),
        ))

    state["geval_results"] = {
        "scores":              {k: round(sum(v) / len(v), 4) for k, v in scores.items() if v},
        "variance":            variance_results,
        "drift":               drift_results,
        "_judge_cost_est_usd": round(judge_cost, 4),
    }
    return state
