# Engineering Progress Report 3: Evaluation Tools — IR and Behavior
## AgentScope — Information Retrieval Evaluator and Agent Behavior Evaluator

**Author:** Pegah Zargarian  
**Program:** MS Data Science, Northeastern University  
**Report Period:** Phases 4–5 (Days 9–12)  
**Repository:** https://github.com/pgazar/AgenticScope

---

## 1. Objective

This phase implemented the two evaluation tools that operate on the normalized execution trace to measure retrieval quality and behavioral trajectory: the IR Evaluator and the Agent Behavior Evaluator. Both tools are primarily deterministic — they operate on captured trace events using well-defined mathematical formulas — with the exception of G-Eval judgment calls in the behavior evaluator for plan coherence and argument correctness, which require language understanding rather than computation.

---

## 2. IR Evaluator (`agentscope/tools/ir_evaluator.py`)

The IR Evaluator computes five standard information retrieval metrics. Each function is pure and independently testable:

```python
def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / k

def ndcg_at_k(retrieved: list, relevant: set, k: int) -> float:
    dcg  = sum((1 / np.log2(i + 2)) for i, d in enumerate(retrieved[:k]) if d in relevant)
    idcg = sum((1 / np.log2(i + 2)) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

def mrr(retrieved_lists: list[list], relevant_sets: list[set]) -> float:
    rrs = [next((1/(i+1) for i, d in enumerate(ret) if d in rel), 0)
           for ret, rel in zip(retrieved_lists, relevant_sets)]
    return float(np.mean(rrs)) if rrs else 0.0
```

### Ground Truth Loading

Two significant bugs were identified and corrected during this phase.

**Bug 1 — Positional alignment.** The initial `_load_ground_truth()` implementation aligned ground truth rows to traces by list position (row 0 → trace 0). This is incorrect when the ground truth CSV contains more rows than there are evaluation inputs, or when rows are not in the same order as the queries were submitted. The fix aligns rows by matching the `question` text to the `agent_input` field of each trace, with a partial-match fallback for minor phrasing differences:

```python
def _load_ground_truth(gt_path: str, queries: list[str] = None) -> list[set]:
    all_rows: dict[str, set] = {}
    for row in reader:
        q = row.get("question", "").strip().lower()
        all_rows[q] = _parse_relevant_set(row)
    # Align by query text, not position
    return [all_rows.get(q.strip().lower(), set()) for q in queries]
```

**Bug 2 — Single-document ground truth.** The original CSV format used a single `answer` column, limiting each query to one relevant document. Real RAG systems often have multiple relevant documents per query. The fix adds a `relevant_docs` column with pipe-separated document keys, with backward compatibility for the legacy `answer` format:

```python
def _parse_relevant_set(row: dict) -> set:
    raw = row.get("relevant_docs", "").strip()
    if raw:
        return {d.strip() for d in raw.split("|") if d.strip()}
    answer = row.get("answer", "").strip()
    return {answer} if answer else set()
```

This fix enabled correct multi-document recall computation for the sample RAG agent, whose ground truth CSV records two relevant documents per query (e.g., `financial_q3_2024:chunk_1|sample_finance:chunk_1`).

---

## 3. Agent Behavior Evaluator (`agentscope/tools/agent_behavior.py`)

The behavior evaluator computes eight metrics from the normalized trace.

### Deterministic Metrics

```python
def step_budget_efficiency(actual_steps: int, max_steps: int) -> float:
    # Heuristic only — not ground-truth validated
    return min(1.0, max_steps / actual_steps) if actual_steps > 0 else 1.0

def convergence(traces: list, max_steps: int) -> float:
    tool_calls = [e for e in traces if e.event_type == "tool_start"]
    return 1.0 if len(tool_calls) <= max_steps else 0.0

def tool_selection_accuracy(traces: list, expected: list[str]) -> Optional[float]:
    if not expected:
        return None  # no reference — return None, not 0
    called  = [e.tool_name for e in traces if e.event_type == "tool_start"]
    correct = sum(1 for c in called if c in expected)
    return correct / len(called) if called else 0.0
```

### Ghost Action Detection

Ghost action rate cross-references action-verb patterns in the final agent output against `tool_start` events in the trace:

```python
ACTION_PATTERNS = [
    r"\b(retrieved|fetched|searched|found|queried|calculated|analyzed|summarized)\b",
    r"\b(the results? (show|indicate|reveal))\b",
    r"\b(according to (the|my) (search|retrieval|data|results?))\b",
]

def ghost_action_rate(traces: list, agent_outputs: list[str]) -> dict:
    for output, trace_events in zip(agent_outputs, [traces]):
        claimed_action = bool(re.search(combined, output, re.IGNORECASE))
        has_tool_call  = any(e.event_type == "tool_start" for e in trace_events)
        if claimed_action and not has_tool_call:
            ghost_count += 1
    return {"ghost_action_rate": round(ghost_count / total, 3), ...}
```

### LLM Semantic Permission Validation

Rather than using regex normalization (`_normalize_tool_name()`), permission matching uses Claude Haiku to resolve naming variants such as `SendEmail` → `send_email`. The call runs in a subprocess to avoid asyncio deadlocks from Gradio's event loop:

```python
def _match_tool_to_schema(tool_name: str, schema_keys: list[str]) -> Optional[str]:
    prompt = (
        f"Tool name called by the agent: '{tool_name}'\n"
        f"Available schema keys: {schema_keys}\n\n"
        "Which schema key is this tool semantically equivalent to? "
        "Reply with ONLY the matching key. If none match, reply: none"
    )
    # Runs via subprocess.run([sys.executable, "-c", script], ...)
    # Returns matched schema key or None
```

Unknown tools (no semantic match found) are logged as warnings, not violations, to avoid penalizing agents for using custom tools not yet listed in the permission schema.

### G-Eval Behavior Metrics (subprocess)

Plan success and argument correctness are evaluated using G-Eval, but the G-Eval calls must run in a subprocess to prevent asyncio event loop conflicts when called from Gradio's worker thread:

```python
def _geval_behavior_scores(model_name: str, tool_sequence: list, tool_calls: list) -> dict:
    script = f"""
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())
from deepeval.metrics import GEval
# ... score plan_success and arg_correctness ...
print(json.dumps(results))
"""
    r = subprocess.run([sys.executable, tmp], capture_output=True, timeout=240, ...)
    return json.loads(r.stdout)  # {"plan_success": 0.8, "arg_correctness": 0.9}
```

---

## 4. Testing

All IR metric functions are tested with exact expected values:

```python
def test_precision_perfect():
    assert precision_at_k(["a","b","c"], {"a","b","c"}, 3) == 1.0

def test_mrr_second():
    assert mrr([["b","a"]], [{"a"}]) == pytest.approx(0.5)

def test_ndcg_perfect():
    assert ndcg_at_k(["a","b"], {"a","b"}, 2) == pytest.approx(1.0)

def test_permission_schema_loads():
    schema = load_permission_schema()
    assert schema["send_email"]["allowed"] is False
    assert schema["rag_retrieve"]["allowed"] is True
```

Behavior evaluator tests cover normalization, ghost action detection, and permission schema loading. 32 tests pass across both tools.

---

## 5. Key Design Decisions

1. **Query-text alignment for ground truth** — positional alignment silently produces wrong scores when the CSV order differs from query submission order. Text matching is the only safe approach for real evaluation sets.
2. **Multi-document relevant set support** — pipe-separated `relevant_docs` enables proper Precision@k and Recall@k computation for queries with more than one ground truth document.
3. **LLM semantic permission matching over regex** — CamelCase to snake_case regex conversion fails for abbreviations (`RAGRetrieve`), synonyms (`search_docs` → `rag_retrieve`), and stylistic variants. An LLM judge resolves all of these correctly at minimal cost (Claude Haiku, ~1 second per tool name).
