"""
Run AgentScope evaluation against capstone-rag.

Uses 10 representative queries. Results written to outputs/<run_id>_run_report.json.

Usage (from AgenticScope root):
    source .venv/bin/activate
    python eval_targets/capstone_rag/run_eval.py
"""
import uuid
import json
import os
import sys

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from agentscope.config import load_config
from agentscope.runner import AgentRunner, AgentTrace
from agentscope.tools.ir_evaluator import (
    precision_at_k, recall_at_k, mrr, ndcg_at_k, hit_rate_at_k
)
from agentscope.tools.agent_behavior import (
    tool_selection_accuracy, step_budget_efficiency, convergence, step_match
)
from agentscope.tools.cost_analyzer import compute_query_cost, compute_latencies
from agentscope.report.compiler import ReportCompiler

# 10 representative queries covering RAG, math, adversarial, multi-step
EVAL_QUERIES = [
    "What was the revenue for Q3?",
    "What are the key features of the product?",
    "Calculate 15% of the Q3 revenue figure",
    "Who are the top performing employees by sales?",
    "Summarize the financial performance for Q2 2024",
    "Who is the CEO of XYZ Corporation?",
    "What products are available and what are their prices?",
    "What was the revenue growth from Q2 to Q3?",
    "List employees with above-average sales",
    "What are the main risks mentioned in the financial documents?",
]

# Ground truth: relevant doc keys per query (from test_cases.jsonl)
GROUND_TRUTH = [
    ["financial_q3_2024:chunk_1", "sample_finance:chunk_1"],  # Q3 revenue
    ["product_description:chunk_1"],                           # product features
    ["financial_q3_2024:chunk_1", "sample_finance:chunk_1"],  # calculate 15%
    ["employee_sales_2024:chunk_1"],                           # top employees
    ["financial_q2_2024:chunk_1", "sample_finance:chunk_1"],  # Q2 summary
    [],                                                         # XYZ Corp — no-answer
    ["product_description:chunk_1", "product_sales_2024:chunk_1"],
    ["financial_q2_2024:chunk_1", "financial_q3_2024:chunk_1"],
    ["employee_sales_2024:chunk_1"],
    ["financial_q3_2024:chunk_1", "sample_finance:chunk_1"],
]

EXPECTED_TOOLS = ["rag_retrieve", "calculate", "analyze_data", "summarize"]


def build_traces_with_events(queries: list[str]) -> list[AgentTrace]:
    """
    Runs capstone-rag directly and converts its step dicts into AgentTraces
    with populated events — bypassing the LangChain callback system.
    """
    from eval_targets.capstone_rag.main import run_with_trace, convert_trace, DEFAULT_CONFIG
    from agentscope.runner import AgentRunner

    runner = AgentRunner("eval_targets/capstone_rag", agent_model=DEFAULT_CONFIG["model"])
    traces = []

    for i, query in enumerate(queries):
        print(f"  [{i+1}/{len(queries)}] {query[:60]}...")
        result = run_with_trace(query)
        raw_events = convert_trace(result)
        normalized_events = runner._normalize(raw_events)

        trace = AgentTrace(
            run_id=str(uuid.uuid4())[:8],
            agent_input=query,
            agent_output=result.get("final_answer", ""),
            events=normalized_events,
            total_latency_ms=result.get("total_latency_ms", 0.0),
        )
        traces.append(trace)
        print(f"     → {len(normalized_events)} events, {result['total_steps']} steps, {result['total_latency_ms']:.0f}ms")

    return traces


def compute_ir_metrics(traces: list[AgentTrace], k: int = 5) -> dict:
    retrieved_lists = []
    for trace in traces:
        docs = []
        for evt in trace.events:
            if hasattr(evt, "event_type") and evt.event_type == "retrieval":
                docs.extend(evt.retrieval_docs)
        retrieved_lists.append(docs)

    relevant_sets = [set(gt) for gt in GROUND_TRUTH]
    n = min(len(retrieved_lists), len(relevant_sets))
    ret = retrieved_lists[:n]
    rel = relevant_sets[:n]

    import numpy as np
    return {
        "precision_k": float(np.mean([precision_at_k(r, s, k) for r, s in zip(ret, rel)])),
        "recall_k":    float(np.mean([recall_at_k(r, s, k)    for r, s in zip(ret, rel)])),
        "mrr":         mrr(ret, rel),
        "ndcg":        float(np.mean([ndcg_at_k(r, s, k)      for r, s in zip(ret, rel)])),
        "hit_rate_k":  hit_rate_at_k(ret, rel, k),
    }


def compute_behavior_metrics(traces: list[AgentTrace], max_steps: int = 10) -> dict:
    all_events = [e for t in traces for e in t.events]
    tool_steps = [e for e in all_events if hasattr(e, "event_type") and e.event_type == "tool_start"]
    called_tools = [e.tool_name for e in tool_steps]
    return {
        "tool_accuracy":       tool_selection_accuracy(all_events, EXPECTED_TOOLS),
        "step_budget_efficiency": step_budget_efficiency(len(tool_steps), max_steps * len(traces)),
        "convergence":         convergence(all_events, max_steps * len(traces)),
        "avg_steps_per_query": round(len(tool_steps) / max(len(traces), 1), 2),
        "tools_distribution":  {t: called_tools.count(t) for t in set(called_tools)},
    }


def main():
    cfg = load_config()
    run_id = str(uuid.uuid4())[:8]

    print(f"\n{'='*65}")
    print(f"  AgentScope × capstone-rag evaluation")
    print(f"  run_id:      {run_id}")
    print(f"  judge model: {cfg.judge.model}")
    print(f"  queries:     {len(EVAL_QUERIES)}")
    print(f"{'='*65}\n")

    print("Step 1/4 — Running capstone-rag agent on all queries...")
    traces = build_traces_with_events(EVAL_QUERIES)

    print("\nStep 2/4 — Computing IR metrics...")
    ir = compute_ir_metrics(traces)
    print(f"  nDCG@5={ir['ndcg']:.3f}  MRR={ir['mrr']:.3f}  Hit Rate={ir['hit_rate_k']:.3f}")

    print("\nStep 3/4 — Computing behavior metrics...")
    behavior = compute_behavior_metrics(traces)
    print(f"  Tool accuracy={behavior['tool_accuracy']}  Avg steps={behavior['avg_steps_per_query']}")
    print(f"  Tools used: {behavior['tools_distribution']}")

    print("\nStep 4/4 — Computing cost & latency...")
    all_events = [e for t in traces for e in t.events]
    cost_per_query = compute_query_cost(all_events, cfg.judge.model)
    latencies = compute_latencies(all_events)
    cost = {
        "cost_per_query":   round(cost_per_query / max(len(traces), 1), 5),
        "p50_latency_s":    round(latencies["p50"] / 1000, 3),
        "p95_latency_s":    round(latencies["p95"] / 1000, 3),
        "agent_model":      cfg.judge.model,
        "qc_index":         0.0,
        "cost_per_success": None,
    }

    # Compile and save report
    import datetime
    report = {
        "run_id":       run_id,
        "timestamp":    datetime.datetime.utcnow().isoformat() + "Z",
        "agent_folder": "eval_targets/capstone_rag",
        "agent_type":   "rag",
        "turn_type":    "single",
        "config":       cfg.model_dump(),
        "eval_results": {
            "ir":          ir,
            "behavior":    behavior,
            "geval":       None,
            "cost":        cost,
            "adversarial": None,
        },
    }

    os.makedirs("outputs", exist_ok=True)
    path = f"outputs/{run_id}_capstone_rag_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*65}")
    print("  RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"\n  IR Metrics (k=5):")
    print(f"    Precision@5:  {ir['precision_k']:.3f}")
    print(f"    Recall@5:     {ir['recall_k']:.3f}")
    print(f"    MRR:          {ir['mrr']:.3f}")
    print(f"    nDCG@5:       {ir['ndcg']:.3f}")
    print(f"    Hit Rate@5:   {ir['hit_rate_k']:.3f}")
    print(f"\n  Agent Behavior:")
    print(f"    Tool accuracy:       {behavior['tool_accuracy']}")
    print(f"    Step budget eff.:    {behavior['step_budget_efficiency']:.3f}")
    print(f"    Avg steps/query:     {behavior['avg_steps_per_query']}")
    print(f"    Tools distribution:  {behavior['tools_distribution']}")
    print(f"\n  Cost & Latency (Haiku):")
    print(f"    Cost/query:   ${cost['cost_per_query']:.5f}")
    print(f"    p50 latency:  {cost['p50_latency_s']:.2f}s")
    print(f"    p95 latency:  {cost['p95_latency_s']:.2f}s")
    print(f"\n  Report saved → {path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
