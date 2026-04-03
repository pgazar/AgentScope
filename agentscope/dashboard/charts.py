import plotly.graph_objects as go
from agentscope.dashboard.colors import color_for_metric


def ir_chart(ir_results: dict) -> go.Figure:
    labels = ["Precision@k", "Recall@k", "MRR", "nDCG", "Hit Rate@k"]
    keys   = ["precision_k", "recall_k", "mrr", "ndcg", "hit_rate_k"]
    values = [ir_results.get(k, 0) for k in keys]
    colors = [color_for_metric(v, "ir") for v in values]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                              text=[f"{v:.2f}" for v in values], textposition="outside", cliponaxis=False))
    fig.update_layout(title="IR metrics", yaxis_range=[0, 1.15], height=280)
    return fig


def agent_chart(behavior_results: dict) -> go.Figure:
    labels = ["Tool acc.", "Plan success", "Step budget eff.", "Arg. correct", "Convergence", "Ghost action"]
    keys   = ["tool_accuracy", "plan_success", "step_budget_efficiency", "arg_correctness", "convergence", "ghost_action_rate"]
    values = [behavior_results.get(k, 0) for k in keys]
    colors = [
        color_for_metric(values[0], "agent"),
        color_for_metric(values[1], "agent"),
        color_for_metric(values[2], "agent"),
        color_for_metric(values[3], "agent"),
        color_for_metric(values[4], "agent"),
        color_for_metric(values[5], "hallu"),  # ghost action: lower is better (inverted)
    ]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                              text=[f"{v:.2f}" if v is not None else "N/A" for v in values], textposition="outside", cliponaxis=False))
    fig.update_layout(title="Agentic metrics", yaxis_range=[0, 1.15], height=280)
    return fig


def geval_chart(geval_results: dict) -> go.Figure:
    # geval_results has nested structure — scores live under "scores" key
    scores = geval_results.get("scores", geval_results) if isinstance(geval_results, dict) else {}
    # Strip internal keys that are not scores
    scores = {k: v for k, v in scores.items() if not k.startswith("_") and isinstance(v, (int, float))}

    label_map = {
        "task_completion": "Task complete",
        "faithfulness":    "Faithfulness",
        "hallucination":   "Hallucination",
        "citation_acc":    "Citation acc.",
        "helpfulness":     "Helpfulness",
        "safety":          "Safety",
    }
    labels = [label_map.get(k, k) for k in scores]
    values = list(scores.values())
    colors = [
        color_for_metric(v, "hallu" if k == "hallucination" else "geval")
        for k, v in scores.items()
    ]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                              text=[f"{v:.2f}" for v in values], textposition="outside", cliponaxis=False))
    fig.update_layout(title="Response quality — G-Eval", yaxis_range=[0, 1.15], height=280)
    return fig


def cost_chart(cost_results: dict) -> go.Figure:
    qc = cost_results.get("qc_index", 0) or 0
    labels = ["Cost/query", "Cost/success", "p50 lat.", "p95 lat.", "Quality-cost idx"]
    raw = [
        cost_results.get("cost_per_query",   0) or 0,
        cost_results.get("cost_per_success", 0) or 0,
        cost_results.get("p50_latency_s",    0) or 0,
        cost_results.get("p95_latency_s",    0) or 0,
        min(qc / 200, 1.0),  # normalize to 0-1 for bar height (200 = excellent)
    ]
    colors = [
        color_for_metric(raw[0], "cost_usd"),
        color_for_metric(raw[1], "cost_usd"),
        color_for_metric(raw[2], "latency_s"),
        color_for_metric(raw[3], "latency_s"),
        color_for_metric(raw[4], "agent"),   # qc_index: higher is better
    ]
    # Format text labels — show actual values even when bars are zero-height
    text = [
        f"${raw[0]:.5f}" if raw[0] else "$0.00",
        f"${raw[1]:.5f}" if raw[1] else "N/A",
        f"{raw[2]:.3f}s" if raw[2] else "0.000s",
        f"{raw[3]:.3f}s" if raw[3] else "0.000s",
        f"{qc:.1f}" if qc else "0.0",
    ]
    fig = go.Figure(go.Bar(
        x=labels, y=raw,
        marker_color=colors,
        text=text,
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        title="Input agent — cost analysis",
        height=300,
        yaxis=dict(rangemode="tozero"),
        uniformtext_minsize=10,
    )
    return fig


def adversarial_chart(adversarial_results: dict) -> go.Figure:
    labels = ["Injection resist.", "Unsafe compliance", "Attack success", "Policy violations"]
    values = [
        adversarial_results.get("prompt_injection_resistance", 0),
        adversarial_results.get("unsafe_compliance_rate",      0),
        adversarial_results.get("attack_success_rate",         0),
        adversarial_results.get("permission_violation_rate",   0),
    ]
    colors = [
        color_for_metric(values[0], "agent"),  # resistance: higher is better
        color_for_metric(values[1], "hallu"),  # compliance: inverted
        color_for_metric(values[2], "hallu"),  # attack success: inverted
        color_for_metric(values[3], "hallu"),  # violations: inverted
    ]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                              text=[f"{v:.2f}" for v in values], textposition="outside", cliponaxis=False))
    fig.update_layout(title="Safety and robustness", yaxis_range=[0, 1.15], height=280)
    return fig
