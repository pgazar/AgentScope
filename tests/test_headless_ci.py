from agentscope.headless_ci import evaluate_report_thresholds


def test_evaluate_report_thresholds_passes_for_matching_values():
    report = {
        "eval_results": {
            "behavior": {"convergence": 1.0, "ghost_action_rate": 0.0},
            "cost": {"cost_per_query": 0.01},
        }
    }
    failures = evaluate_report_thresholds(report, [
        ("eval_results.behavior.convergence", 0.9, "min"),
        ("eval_results.behavior.ghost_action_rate", 0.05, "max"),
        ("eval_results.cost.cost_per_query", 0.02, "max"),
    ])
    assert failures == []


def test_evaluate_report_thresholds_flags_missing_and_failing_values():
    report = {
        "eval_results": {
            "behavior": {"convergence": 0.4},
        }
    }
    failures = evaluate_report_thresholds(report, [
        ("eval_results.behavior.convergence", 0.9, "min"),
        ("eval_results.behavior.handoff_correctness", 0.8, "min"),
    ])
    assert any("convergence" in failure for failure in failures)
    assert any("handoff_correctness" in failure for failure in failures)
