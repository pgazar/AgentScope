from dataclasses import dataclass, field
import numpy as np
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

SECONDARY_MODEL = "gpt-4o-mini"
VARIANCE_THRESHOLD = 0.15
DRIFT_THRESHOLD = 0.10


@dataclass
class ScoredMetric:
    """
    Carries everything needed for variance measurement:
    the metric name, the criteria string used, the test cases
    evaluated, and the primary scores produced.
    """
    name: str
    criteria: str
    test_cases: list[LLMTestCase]
    primary_scores: list[float]
    primary_model: str


def measure_inter_judge_variance(scored: ScoredMetric) -> dict:
    """
    Re-runs the same test cases through a secondary judge model
    and computes per-metric score delta statistics.
    Uses scored.criteria directly — no implicit string lookup needed.
    """
    secondary_metric = GEval(
        name=f"{scored.name}_secondary",
        criteria=scored.criteria,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=SECONDARY_MODEL,
    )

    secondary_scores = []
    for tc in scored.test_cases:
        secondary_metric.measure(tc)
        secondary_scores.append(secondary_metric.score)

    deltas = [abs(p - s) for p, s in zip(scored.primary_scores, secondary_scores)]
    std = float(np.std(deltas)) if deltas else 0.0
    flagged = [i for i, d in enumerate(deltas) if d > VARIANCE_THRESHOLD]

    return {
        "metric": scored.name,
        "primary_model": scored.primary_model,
        "secondary_model": SECONDARY_MODEL,
        "mean_delta": round(float(np.mean(deltas)), 4) if deltas else 0.0,
        "std_deviation": round(std, 4),
        "high_variance_cases": flagged,
        "stable": std <= VARIANCE_THRESHOLD,
    }


def measure_calibration_drift(
    metric_name: str,
    current_scores: list[float],
    baseline_scores: list[float],
) -> dict:
    """
    Compares current score distribution against a stored baseline
    using KL divergence. Baseline is loaded from a previous run's
    JSON report if available — explicitly passed in, not assumed.
    """
    if not baseline_scores:
        return {
            "metric": metric_name,
            "drift_measured": False,
            "reason": "no baseline scores available for this metric",
        }

    bins = np.linspace(0, 1, 11)
    curr_hist, _ = np.histogram(current_scores, bins=bins, density=True)
    base_hist, _ = np.histogram(baseline_scores, bins=bins, density=True)

    # Small epsilon prevents log(0) in KL computation
    curr_hist = curr_hist + 1e-8
    base_hist = base_hist + 1e-8

    from scipy.stats import entropy
    kl = float(entropy(curr_hist, base_hist))

    return {
        "metric": metric_name,
        "drift_measured": True,
        "kl_divergence": round(kl, 4),
        "drift_flagged": kl > DRIFT_THRESHOLD,
        "threshold": DRIFT_THRESHOLD,
    }
