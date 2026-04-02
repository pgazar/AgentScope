import csv
import numpy as np
from agentscope.orchestrator.state import AgentState


def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / k


def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr(retrieved_lists: list[list], relevant_sets: list[set]) -> float:
    rrs = []
    for ret, rel in zip(retrieved_lists, relevant_sets):
        rr = next((1 / (i + 1) for i, d in enumerate(ret) if d in rel), 0)
        rrs.append(rr)
    return float(np.mean(rrs)) if rrs else 0.0


def ndcg_at_k(retrieved: list, relevant: set, k: int) -> float:
    dcg = sum((1 / np.log2(i + 2)) for i, d in enumerate(retrieved[:k]) if d in relevant)
    idcg = sum((1 / np.log2(i + 2)) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(retrieved_lists: list[list], relevant_sets: list[set], k: int) -> float:
    hits = [1 if set(r[:k]) & rel else 0 for r, rel in zip(retrieved_lists, relevant_sets)]
    return float(np.mean(hits)) if hits else 0.0


def _parse_retrievals(traces: list) -> list[list]:
    """Extracts retrieved doc IDs from retrieval events in the trace."""
    results = []
    for trace in traces:
        # traces is a list of AgentTrace objects
        events = trace.events if hasattr(trace, "events") else []
        docs = []
        for evt in events:
            if hasattr(evt, "event_type") and evt.event_type == "retrieval":
                docs.extend(evt.retrieval_docs)
        results.append(docs)
    return results


def _load_ground_truth(gt_path: str) -> list[set]:
    """
    Loads ground truth from a CSV with columns: question, answer.
    Each answer is treated as a single relevant doc ID for that query.
    """
    relevant_sets = []
    with open(gt_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            answer = row.get("answer", "").strip()
            relevant_sets.append({answer} if answer else set())
    return relevant_sets


def run(state: AgentState) -> AgentState:
    k = state["config"]["eval"]["k"]
    retrieved_lists = _parse_retrievals(state["traces"])
    relevant_sets = _load_ground_truth(state["gt_path"])

    # Align lengths — ground truth may have more rows than traces
    n = min(len(retrieved_lists), len(relevant_sets))
    retrieved_lists = retrieved_lists[:n]
    relevant_sets = relevant_sets[:n]

    state["ir_results"] = {
        "precision_k": float(np.mean([precision_at_k(r, rel, k) for r, rel in zip(retrieved_lists, relevant_sets)])),
        "recall_k":    float(np.mean([recall_at_k(r, rel, k)    for r, rel in zip(retrieved_lists, relevant_sets)])),
        "mrr":         mrr(retrieved_lists, relevant_sets),
        "ndcg":        float(np.mean([ndcg_at_k(r, rel, k)      for r, rel in zip(retrieved_lists, relevant_sets)])),
        "hit_rate_k":  hit_rate_at_k(retrieved_lists, relevant_sets, k),
    }
    return state
