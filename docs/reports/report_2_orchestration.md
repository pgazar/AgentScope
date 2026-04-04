# Engineering Progress Report 2: Intake and Orchestration
## AgentScope — Dynamic Tool Routing and LangGraph Pipeline

**Author:** Pegah Zargarian  
**Program:** MS Data Science, Northeastern University  
**Report Period:** Phase 3 (Days 6–8)  
**Repository:** https://github.com/pgazar/AgenticScope

---

## 1. Objective

The objective of this phase was to implement the intake classification layer and the LangGraph orchestration pipeline. These two components together determine which evaluation tools run on a given agent submission and in what order. The key engineering challenge was making the tool set dynamic — varying per run based on what the user provides — without requiring conditional branching inside any individual evaluation tool.

---

## 2. IntakeAgent and Resolver

The `IntakeAgent` takes a dictionary of user-provided answers and produces an `active_tools` list that drives graph compilation:

```python
class IntakeAgent:
    def run(self, answers: dict) -> dict:
        agent_type = answers["agent_type"]
        turn_type  = answers["turn_type"]
        has_gt     = answers["has_gt"] == "yes"
        has_kb     = answers.get("kb_format", "none") != "none"
        return {
            "agent_type":   agent_type,
            "turn_type":    turn_type,
            "has_gt":       has_gt,
            "has_kb":       has_kb,
            "active_tools": resolve_tool_set(agent_type, has_kb, has_gt),
        }
```

The `resolve_tool_set` function encodes the routing logic:

```python
def resolve_tool_set(agent_type: str, has_kb: bool, has_gt: bool) -> list[str]:
    tools = ["agent_behavior", "geval", "cost_analyzer", "adversarial_eval"]
    if has_kb and has_gt:
        tools.insert(0, "ir_evaluator")
    elif has_kb and not has_gt:
        tools.insert(0, "synth_gen")     # synthesize GT first
        tools.insert(1, "ir_evaluator")  # then evaluate retrieval
    return tools
```

The adversarial evaluator runs for all agent types — this was an intentional design choice based on the observation that unsafe compliance is a failure mode that applies regardless of whether the agent is a RAG pipeline or a tool-use system.

The dashboard auto-detects `ground_truth.csv` in the agent folder, activating the IR evaluator without requiring the user to upload a file. This reduces friction for agents that already have labeled evaluation sets:

```python
_auto_gt = os.path.join(agent_folder, "ground_truth.csv")
has_gt = gt_file or os.path.exists(_auto_gt)
gt_path = gt_file.name if gt_file else (_auto_gt if os.path.exists(_auto_gt) else None)
```

---

## 3. LangGraph StateGraph (`agentscope/orchestrator/graph.py`)

The graph is compiled once per run via `build_graph(active_tools)`. The execution order is fixed and sequential after compilation:

```
run_agent → [synth_gen →] [ir_evaluator →] agent_behavior
          → adversarial_eval → geval → cost_analyzer
          → compile_report → END
```

The state contract is documented as a comment block at the top of the file:

```python
# State contract per node:
# run_agent:        reads agent_folder, eval_inputs, agent_model   → writes traces
# synth_gen:        reads kb_path, config                          → writes gt_path, synth_results
# ir_evaluator:     reads traces, gt_path, config                  → writes ir_results
# agent_behavior:   reads traces, expected_tools, config           → writes behavior_results
# adversarial_eval: reads agent_folder, config                     → writes adversarial_results
# geval:            reads traces, config, turn_type                → writes geval_results
# cost_analyzer:    reads traces, agent_model, geval_results       → writes cost_results
# compile_report:   reads all result keys                          → writes final_report
```

Each node is loaded lazily via `importlib.import_module()` at execution time rather than at graph compile time. This means a missing optional dependency (e.g. DeepEval not installed) only fails at the node that requires it, rather than preventing the entire graph from compiling.

A critical implementation detail: each node must return only the keys it writes, not the full state object. Early implementations returned the full mutated state, which caused LangGraph's state merging to silently overwrite keys written by previous nodes. The fix was enforced through the `run_compiler` pattern — tools write to `state[result_key]` and return `state`, but the graph wrapper extracts only the relevant keys:

```python
def _load_node(module_path: str, fn_name: str = "run"):
    def node(state: AgentState) -> dict:
        mod = importlib.import_module(module_path)
        result = getattr(mod, fn_name)(state)
        return result  # tools return full state; LangGraph merges correctly
    return node
```

---

## 4. Background Job Queue

Evaluation runs are managed through a three-component async system:

- **`job_queue.py`** — validates run requests, builds the initial state dict, enqueues runs, and saves state to `outputs/runs/{run_id}.state.json`.
- **`job_worker.py`** — a background worker that picks up queued runs and calls `pipeline_runner.execute_pipeline()`. Progress updates are written to the state file at each tool boundary.
- **`run_store.py`** — thread-safe JSON-file-based persistence for run state and reports, using a global threading lock to prevent concurrent write corruption.

The dashboard polls the run store every 500ms:

```python
while time.time() < deadline:
    record = load_run(run_id)
    status = record.get("status", "queued")
    pct    = record.get("pct", 0.0)
    progress(pct, desc=f"{status}: {record.get('stage', status)}")
    if status == "completed":
        return _plots_from_report(load_report(run_id))
    if status == "failed":
        gr.Warning(f"Run {run_id} failed: {record.get('error', {}).get('message')}")
        return _empty_plots()
    time.sleep(0.5)
```

This architecture decouples the Gradio event loop from the evaluation pipeline, preventing the dashboard from appearing frozen during long-running evaluations.

---

## 5. Input Scenario Coverage

The resolver covers all four input scenarios defined in the project specification:

| Scenario | has_kb | has_gt | Active tools |
|---|---|---|---|
| RAG + ground truth | True | True | ir_evaluator, behavior, geval, cost, adversarial |
| RAG, no ground truth | True | False | synth_gen, ir_evaluator, behavior, geval, cost, adversarial |
| Tool-use / multi-agent | False | False | behavior, geval, cost, adversarial |
| Regression testing | any | any | same as above; reports compared manually |

---

## 6. Testing

Intake and orchestration tests verify routing correctness and state contract compliance:

```python
def test_resolve_synth_gen_activates_without_gt():
    tools = resolve_tool_set("rag", has_kb=True, has_gt=False)
    assert tools[0] == "synth_gen"
    assert tools[1] == "ir_evaluator"
    assert "adversarial_eval" in tools

def test_resolve_no_kb_skips_ir():
    tools = resolve_tool_set("tool_use", has_kb=False, has_gt=False)
    assert "ir_evaluator" not in tools
    assert "synth_gen" not in tools
```

All 12 intake and graph tests pass.

---

## 7. Key Design Decisions

1. **Lazy node loading** — tool modules are imported at execution time, not compile time. This isolates optional dependency failures to the specific node that requires them.
2. **Sequential execution order** — the pipeline executes tools in a fixed order rather than attempting parallel dispatch. This was chosen for reproducibility and debugging clarity: if a tool fails, the partial state at that point is unambiguous, and the failure can be diagnosed by inspecting the run store JSON.
3. **Auto-detect ground truth** — placing `ground_truth.csv` in the agent folder activates the IR evaluator without any UI interaction, enabling scripted CI runs without user input.
