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
        events = trace.events if hasattr(trace, "events") else []
        docs = []
        for evt in events:
            if hasattr(evt, "event_type") and evt.event_type == "retrieval":
                docs.extend(evt.retrieval_docs)
        results.append(docs)
    return results


def _parse_relevant_set(row: dict) -> set:
    """
    Reads relevant doc IDs from a CSV row.

    Supports two column formats — checks for relevant_docs first:
      - relevant_docs: pipe-separated doc keys  → {"doc_a", "doc_b", "doc_c"}
        e.g. "financial_q3_2024:chunk_1|sample_finance:chunk_1"
      - answer (legacy): single doc key         → {"doc_a"}
        e.g. "financial_q3_2024:chunk_1"

    Queries with no relevant docs (empty string) return an empty set,
    which the metrics treat as "no ground truth" and score 0.0.
    """
    # Prefer relevant_docs column (multi-doc support)
    raw = row.get("relevant_docs", "").strip()
    if raw:
        return {d.strip() for d in raw.split("|") if d.strip()}

    # Fall back to legacy single-answer column
    answer = row.get("answer", "").strip()
    return {answer} if answer else set()


def _load_ground_truth(gt_path: str, queries: list[str] = None) -> list[set]:
    """
    Loads ground truth from a CSV.

    Supported formats:
      question, relevant_docs   — pipe-separated doc keys (preferred)
      question, answer          — single doc key (legacy, backward compatible)

    If queries provided, aligns rows by matching question text so the
    order matches the trace list even when the CSV has more rows.
    """
    all_rows: dict[str, set] = {}
    ordered_rows: list[tuple[str, set]] = []

    with open(gt_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q   = row.get("question", "").strip().lower()
            rel = _parse_relevant_set(row)
            all_rows[q] = rel
            ordered_rows.append((q, rel))

    if not queries:
        return [s for _, s in ordered_rows]

    result = []
    for q in queries:
        key = q.strip().lower()
        if key in all_rows:
            result.append(all_rows[key])
        else:
            # Partial match fallback for slight phrasing differences
            match = next((s for k, s in all_rows.items() if key in k or k in key), set())
            result.append(match)
    return result


def run(state: AgentState) -> AgentState:
    trace_diag = state.get("trace_diagnostics") or {}
    if not trace_diag.get("scoreability", {}).get("ir", False):
        state["ir_results"] = {
            "scoreable": False,
            "reason": "no retrieval events were captured in the trace",
            "precision_k": None,
            "recall_k": None,
            "mrr": None,
            "ndcg": None,
            "hit_rate_k": None,
        }
        return state

    k = state["config"]["eval"]["k"]
    retrieved_lists = _parse_retrievals(state["traces"])
    queries = [t.agent_input for t in state["traces"] if hasattr(t, "agent_input")]
    relevant_sets = _load_ground_truth(state["gt_path"], queries)

    # Align lengths — GT may have more rows than traces
    n = min(len(retrieved_lists), len(relevant_sets))
    retrieved_lists = retrieved_lists[:n]
    relevant_sets   = relevant_sets[:n]

    state["ir_results"] = {
        "scoreable": True,
        "precision_k": float(np.mean([precision_at_k(r, rel, k) for r, rel in zip(retrieved_lists, relevant_sets)])),
        "recall_k":    float(np.mean([recall_at_k(r, rel, k)    for r, rel in zip(retrieved_lists, relevant_sets)])),
        "mrr":         mrr(retrieved_lists, relevant_sets),
        "ndcg":        float(np.mean([ndcg_at_k(r, rel, k)      for r, rel in zip(retrieved_lists, relevant_sets)])),
        "hit_rate_k":  hit_rate_at_k(retrieved_lists, relevant_sets, k),
    }
    return state
