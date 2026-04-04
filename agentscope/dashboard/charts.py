from html import escape

import plotly.graph_objects as go
from agentscope.dashboard.colors import GRAY, color_for_metric


def _num(value, default=0.0):
    return value if isinstance(value, (int, float)) else default


def _fmt_metric(value, kind: str = "score") -> str:
    if value is None:
        return "N/A"
    if kind == "currency":
        return f"${value:.5f}"
    return f"{value:.2f}"


def _card_html(title: str, value: str, color: str) -> str:
    safe_title = escape(title)
    safe_value = escape(value)
    bg = color if color != GRAY else "#e5e7eb"
    fg = "#ffffff" if color != GRAY else "#374151"
    return (
        f"<div style='flex:1; min-width:150px; border-radius:12px; padding:14px 16px; "
        f"background:{bg}; color:{fg}; box-shadow:0 1px 3px rgba(0,0,0,0.08);'>"
        f"<div style='font-size:12px; opacity:0.92; margin-bottom:6px;'>{safe_title}</div>"
        f"<div style='font-size:24px; font-weight:700; line-height:1.1;'>{safe_value}</div>"
        "</div>"
    )


def summary_cards_html(report: dict | None) -> str:
    report = report or {}
    eval_results = report.get("eval_results", report)
    ir = eval_results.get("ir") or {}
    behavior = eval_results.get("behavior") or {}
    geval = eval_results.get("geval") or {}
    cost = eval_results.get("cost") or {}

    scores = geval.get("scores", {}) if isinstance(geval, dict) else {}
    composite_inputs = []
    for key, value in scores.items():
        if not isinstance(value, (int, float)):
            continue
        composite_inputs.append(1.0 - value if key == "hallucination" else value)
    composite = round(sum(composite_inputs) / len(composite_inputs), 4) if composite_inputs else None

    cards = [
        _card_html("G-Eval Composite", _fmt_metric(composite), color_for_metric(composite, "geval")),
        _card_html("Hallucination", _fmt_metric(scores.get("hallucination")), color_for_metric(scores.get("hallucination"), "hallu")),
        _card_html("Tool Accuracy", _fmt_metric(behavior.get("tool_accuracy")), color_for_metric(behavior.get("tool_accuracy"), "agent")),
        _card_html("nDCG", _fmt_metric(ir.get("ndcg")), color_for_metric(ir.get("ndcg"), "ir")),
        _card_html("Cost / Query", _fmt_metric(cost.get("cost_per_query"), "currency"), color_for_metric(cost.get("cost_per_query"), "cost_usd")),
    ]

    return (
        "<div style='display:flex; flex-wrap:wrap; gap:12px; margin:6px 0 14px 0;'>"
        + "".join(cards)
        + "</div>"
    )


def ir_chart(ir_results: dict) -> go.Figure:
    labels = ["Precision@k", "Recall@k", "MRR", "nDCG", "Hit Rate@k"]
    keys   = ["precision_k", "recall_k", "mrr", "ndcg", "hit_rate_k"]
    raw_values = [ir_results.get(k) for k in keys]
    values = [_num(v) for v in raw_values]
    colors = [color_for_metric(v, "ir") for v in values]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                              text=[f"{v:.2f}" if rv is not None else "N/A" for v, rv in zip(values, raw_values)], textposition="outside", cliponaxis=False))
    fig.update_layout(title="IR metrics", yaxis_range=[0, 1.15], height=280)
    return fig


def agent_chart(behavior_results: dict) -> go.Figure:
    labels = ["Tool acc.", "Plan success", "Step budget eff.", "Arg. correct", "Convergence", "Ghost action"]
    keys   = ["tool_accuracy", "plan_success", "step_budget_efficiency", "arg_correctness", "convergence", "ghost_action_rate"]
    raw_values = [behavior_results.get(k) for k in keys]
    values = [_num(v) for v in raw_values]
    colors = [
        color_for_metric(values[0], "agent"),
        color_for_metric(values[1], "agent"),
        color_for_metric(values[2], "agent"),
        color_for_metric(values[3], "agent"),
        color_for_metric(values[4], "agent"),
        color_for_metric(values[5], "hallu"),  # ghost action: lower is better (inverted)
    ]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                              text=[f"{v:.2f}" if rv is not None else "N/A" for v, rv in zip(values, raw_values)], textposition="outside", cliponaxis=False))
    fig.update_layout(title="Agentic metrics", yaxis_range=[0, 1.15], height=280)
    return fig


def geval_chart(geval_results: dict) -> go.Figure:
    # geval_results has nested structure — scores live under "scores" key
    scores = geval_results.get("scores", geval_results) if isinstance(geval_results, dict) else {}
    # Strip internal keys that are not scores
    score_items = [
        (k, v)
        for k, v in scores.items()
        if not k.startswith("_")
    ]

    label_map = {
        "task_completion": "Task complete",
        "faithfulness":    "Faithfulness",
        "hallucination":   "Hallucination",
        "citation_acc":    "Citation acc.",
        "helpfulness":     "Helpfulness",
        "safety":          "Safety",
    }
    labels = [label_map.get(k, k) for k, _ in score_items]
    raw_values = [v for _, v in score_items]
    values = [_num(v) for v in raw_values]
    colors = [
        color_for_metric(v, "hallu" if k == "hallucination" else "geval")
        for (k, _), v in zip(score_items, values)
    ]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                              text=[f"{v:.2f}" if rv is not None else "N/A" for v, rv in zip(values, raw_values)], textposition="outside", cliponaxis=False))
    fig.update_layout(title="Response quality — G-Eval", yaxis_range=[0, 1.15], height=280)
    return fig


def cost_chart(cost_results: dict) -> go.Figure:
    qc = _num(cost_results.get("qc_index"))
    labels = ["Cost/query", "Cost/success", "p50 lat.", "p95 lat.", "Quality-cost idx"]
    raw_input = [
        cost_results.get("cost_per_query"),
        cost_results.get("cost_per_success"),
        cost_results.get("p50_latency_s"),
        cost_results.get("p95_latency_s"),
        min(qc / 200, 1.0) if qc else 0.0,
    ]
    raw = [
        _num(raw_input[0]),
        _num(raw_input[1]),
        _num(raw_input[2]),
        _num(raw_input[3]),
        _num(raw_input[4]),
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
        f"${raw[0]:.5f}" if raw_input[0] is not None else "N/A",
        f"${raw[1]:.5f}" if raw_input[1] is not None else "N/A",
        f"{raw[2]:.3f}s" if raw_input[2] is not None else "N/A",
        f"{raw[3]:.3f}s" if raw_input[3] is not None else "N/A",
        f"{qc:.1f}" if cost_results.get("qc_index") is not None else "N/A",
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
    raw_values = [
        adversarial_results.get("prompt_injection_resistance"),
        adversarial_results.get("unsafe_compliance_rate"),
        adversarial_results.get("attack_success_rate"),
        adversarial_results.get("permission_violation_rate"),
    ]
    values = [_num(v) for v in raw_values]
    colors = [
        color_for_metric(values[0], "agent"),  # resistance: higher is better
        color_for_metric(values[1], "hallu"),  # compliance: inverted
        color_for_metric(values[2], "hallu"),  # attack success: inverted
        color_for_metric(values[3], "hallu"),  # violations: inverted
    ]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                              text=[f"{v:.2f}" if rv is not None else "N/A" for v, rv in zip(values, raw_values)], textposition="outside", cliponaxis=False))
    fig.update_layout(title="Safety and robustness", yaxis_range=[0, 1.15], height=280)
    return fig
