# Engineering Progress Report 1: Foundation Layer
## AgentScope — Configuration, State, Tracing, and Agent Runner

**Author:** Pegah Zargarian  
**Program:** MS Data Science, Northeastern University  
**Report Period:** Phase 1–2 (Days 1–5)  
**Repository:** https://github.com/pgazar/AgenticScope

---

## 1. Objective

The goal of this phase was to establish the three components that every subsequent evaluation tool depends on: a validated configuration schema, a shared state contract for the LangGraph pipeline, and a normalized trace collection mechanism capable of capturing the execution trajectory of any Python-based agentic system without modifying its source code.

All downstream evaluators — IR, behavior, G-Eval, cost, adversarial — operate on the output of this layer. A deficiency here propagates silently through the entire pipeline: if token counts are missing, the cost analyzer reports zero. If tool events are absent, the behavior evaluator scores zero. Getting this right before any evaluation tool was written was the primary design constraint of this phase.

---

## 2. Configuration System (`agentscope/config.py`)

The configuration schema is implemented using Pydantic v2, providing runtime validation, default values, and load-time failure on malformed YAML. Two nested models — `JudgeConfig` and `EvalConfig` — are combined into `AgentScopeConfig`.

```python
class EvalConfig(BaseModel):
    k: int = 5
    max_steps: int = 10
    ir_green: float = 0.70
    ir_orange: float = 0.40
    # ... threshold pairs for all 5 panels ...
    max_synth_pairs: int = 50       # cap on synthetic Q&A generation
    max_geval_responses: int = 100  # cap on G-Eval judge calls
    eval_budget_usd: float = 2.00   # hard stop for total eval spend

class AgentScopeConfig(BaseModel):
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
```

Three cost-control fields — `max_synth_pairs`, `max_geval_responses`, and `eval_budget_usd` — were added after a design review identified "silent cost creep" as a risk: evaluation pipelines that run large numbers of LLM judge calls with no enforced budget ceiling. The `eval_budget_usd` field is checked before every G-Eval batch; if the cumulative estimated cost exceeds the limit, scoring stops early and a warning is emitted.

---

## 3. AgentState TypedDict (`agentscope/orchestrator/state.py`)

The shared state is defined as a Python `TypedDict`, giving LangGraph a typed contract for the keys each node reads and writes. The full schema is declared upfront rather than discovered at runtime:

```python
class AgentState(TypedDict):
    run_id: str
    agent_folder: str
    agent_model: str          # model of the EVALUATED agent, not the judge
    eval_inputs: list[str]
    kb_path: Optional[str]
    gt_path: Optional[str]
    agent_type: str           # "rag" | "tool_use" | "multi_agent" | "hybrid"
    turn_type: str            # "single" | "multi"
    active_tools: list[str]
    expected_tools: list[str]
    traces: list              # list[AgentTrace] — populated by AgentRunner
    baseline_geval_scores: Optional[dict]
    ir_results: Optional[dict]
    behavior_results: Optional[dict]
    geval_results: Optional[dict]
    cost_results: Optional[dict]
    adversarial_results: Optional[dict]
    synth_results: Optional[dict]
    final_report: Optional[dict]
    config: dict
```

The `agent_model` field is critical for cost analysis: it records the model used by the system under evaluation, not the judge model, so the cost analyzer uses the correct pricing table.

---

## 4. Trace Collector (`agentscope/tracer.py`)

The `AgentScopeCallbackHandler` is a LangChain `BaseCallbackHandler` subclass that captures four lifecycle events. A key design decision was the `_get()` multi-key helper, which tries multiple dictionary key variants before falling back to a default:

```python
def _get(d: dict, *keys, default=None):
    """Try multiple key variants in order, return first match."""
    for k in keys:
        if k in d:
            return d[k]
    return default

def on_tool_start(self, serialized, input_str, **kwargs):
    # handles "name", "id", or "tool_name" — different LangChain versions use different keys
    tool_name = _get(serialized, "name", "id", "tool_name", default="unknown_tool")
    self._step_start[tool_name] = time.perf_counter()
    self.traces.append({"type": "tool_start", "tool": tool_name, "input": input_str})
```

This resilience was motivated by observing that LangChain 0.3.x and LangChain 1.x use different key names in the `serialized` dict passed to callbacks. Without `_get()`, any tool call from a newer LangChain version would be silently recorded as `unknown_tool`.

---

## 5. Agent Runner (`agentscope/runner.py`)

The `AgentRunner` handles three responsibilities: loading the target agent from a folder, invoking it on each evaluation input, and normalizing raw callback events into `TraceEvent` dataclasses.

```python
@dataclass
class TraceEvent:
    event_type: str   # llm_start | llm_end | tool_start | tool_end | retrieval | handoff
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    tool_output: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retrieval_docs: list = field(default_factory=list)
    latency_ms: float = 0.0
    agent_model: str = ""
    timestamp: float = field(default_factory=time.time)
```

Agent loading uses `importlib.util.spec_from_file_location()` rather than `importlib.import_module()`. This was a critical fix discovered in integration testing: `import_module` caches modules in `sys.modules` by their base name (`main`). When evaluating multiple agents in sequence, the second agent's `main.py` would silently use the first agent's cached module. `spec_from_file_location` creates a unique module spec per absolute path, preventing cache collisions.

The runner supports two integration modes. Mode A detects whether the agent's callable accepts a `callbacks` keyword argument via `inspect.signature()` and injects the handler if so. Mode B checks for an `inject_trace_events(trace)` function on the loaded module and calls it after each run, allowing custom agents to backfill trace events from their own internal result structures.

---

## 6. Trace Audit (`agentscope/trace_audit.py`)

After each trace is collected, a `summarize_traces()` call inspects event coverage and produces a `scoreability` report:

```python
def summarize_traces(traces: list) -> dict:
    # Returns:
    # status: "ok" | "partial" | "output_only" | "error"
    # scoreability: {"behavior": bool, "ir": bool, "cost": bool}
    # event_coverage: {"has_any_events", "has_retrieval_events",
    #                  "has_token_counts", "has_latency_events"}
```

A `status: "output_only"` result means the agent ran and produced output but no trace events were captured — typically because the agent uses a non-LangChain SDK and no `inject_trace_events` adapter was provided. This status is surfaced in the run record and in the dashboard, preventing operators from misinterpreting zero behavior scores as actual behavioral failures.

---

## 7. Testing

Phase 1–2 tests cover all deterministic components. Representative assertions:

```python
def test_config_loads():
    cfg = load_config()
    assert cfg.eval.k == 5
    assert cfg.eval.eval_budget_usd == 2.00

def test_runner_loads_fake_agent():
    runner = AgentRunner("tests/fake_agent", agent_model="test-model")
    trace = runner.run("What is 2+2?", run_id="test-001")
    assert trace.agent_output == "The answer to 'What is 2+2?' is 42."
    assert trace.total_latency_ms > 0
```

All 16 configuration and runner tests pass. The `test_mode_a_integration.py` suite (8 tests) verifies that the callback handler correctly captures tool names, token estimates, and latency across mock LangChain agent invocations.

---

## 8. Key Design Decisions

Three design decisions made in this phase had significant downstream impact:

1. **`spec_from_file_location` over `import_module`** — prevents sys.modules cache collisions when evaluating multiple agents sequentially.
2. **`_get()` multi-key resolution** — makes the tracer resilient to LangChain version differences without conditional version checks.
3. **Trace audit before tool dispatch** — exposes integration failures (missing trace events) to the operator before the pipeline produces misleading zero scores.
