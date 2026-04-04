# Engineering Progress Report 6: Architecture Hardening
## AgentScope — Job Queue, Subprocess Isolation, Trace Auditing, and OpenTelemetry

**Author:** Pegah Zargarian  
**Program:** MS Data Science, Northeastern University  
**Report Period:** Post-spec hardening (Days 26–35)  
**Repository:** https://github.com/pgazar/AgenticScope

---

## 1. Objective

After the core specification was implemented and validated, a second round of engineering addressed four systemic reliability concerns that emerged from real-world usage: the Gradio dashboard blocking during long evaluations, crashes in the target agent code propagating into AgentScope, silent trace failures misleading operators about why scores were zero, and the absence of production-grade observability hooks. This phase introduced the job queue, subprocess agent isolation, trace auditing, OpenTelemetry integration, and a headless CI gate.

---

## 2. Background Job Queue

**Problem.** The original `run_evaluation()` function in `app.py` called `graph.invoke(state)` synchronously in Gradio's worker thread. Evaluations taking 60–90 seconds caused the entire Gradio UI to appear frozen, and any exception in the pipeline propagated as an unhandled error to the dashboard.

**Solution.** A three-component async system decouples the dashboard from the pipeline:

```
Dashboard → job_queue.enqueue_run() → run_store.save_state()
         ← polls load_run() every 500ms

job_worker.py (background thread) → pipeline_runner.execute_pipeline()
                                  → run_store.update_run() at each stage
```

**`run_store.py`** provides thread-safe JSON-file persistence with a global `threading.Lock()`. Each run produces two files: `outputs/runs/{run_id}.state.json` (live progress) and `outputs/{run_id}_run_report.json` (final report). The state file is updated at every tool boundary, enabling the dashboard to show which specific tool is currently executing:

```python
def _progress(run_id: str, stage: str, pct: float, data: dict | None = None):
    payload = {
        "status": "running" if stage != "done" else "completed",
        "stage":  stage,
        "pct":    pct,
    }
    if stage == "done":
        payload["finished_at"]  = utc_now()
        payload["report_path"]  = report_path(run_id)
    update_run(run_id, payload)
```

**`recover_incomplete_runs()`** is called at API startup and re-queues any runs that were `running` or `queued` when the process last stopped, preventing permanent stuck-run states across server restarts.

---

## 3. Subprocess-Isolated Agent Runner

**Problem.** Agent code loaded via `importlib` runs in the same process as AgentScope. An infinite loop, uncaught exception, or memory leak in the target agent's `run()` function would crash or corrupt the AgentScope process.

**Solution.** `runner_worker.py` is a minimal subprocess entry point. The `AgentRunner` serializes the run request as JSON to stdin, the subprocess executes the agent and serializes the `AgentTrace` to stdout, and the parent process deserializes the result:

```python
# runner_worker.py — runs in subprocess
def main() -> int:
    payload = json.loads(sys.stdin.read())
    runner  = AgentRunner(
        agent_folder=payload["agent_folder"],
        agent_model=payload.get("agent_model", "unknown"),
        timeout_s=payload.get("timeout_s"),
    )
    trace = runner.run(payload["agent_input"], payload["run_id"])
    json.dump(serialize_trace(trace), sys.stdout)
    return 0
```

The parent process uses `subprocess.run()` with a configurable timeout. If the subprocess exceeds the timeout, it is killed and an error `AgentTrace` is returned with `agent_output = "[AgentRunner timeout: ...]"`. This ensures that a runaway agent cannot block the evaluation pipeline indefinitely.

---

## 4. Trace Audit (`agentscope/trace_audit.py`)

**Problem.** When an agent runs correctly but produces no trace events (e.g., a custom SDK agent without an `inject_trace_events` adapter), all evaluators that depend on trace events (behavior, cost, IR) return scores of 0.0. Without context, the operator cannot distinguish between "this agent behaves poorly" and "this agent was not properly instrumented."

**Solution.** `trace_audit.py` inspects every trace after collection and before any evaluation tool runs. It produces a `scoreability` report with four coverage flags and an overall status:

```python
def summarize_traces(traces: list) -> dict:
    return {
        "status": "ok" | "partial" | "output_only" | "error",
        "scoreability": {
            "behavior": any_events,      # tool/llm events present
            "ir":       any_retrieval,   # retrieval events present
            "cost":     any_tokens,      # token counts present
        },
        "event_coverage": {
            "has_any_events":       bool,
            "has_retrieval_events": bool,
            "has_token_counts":     bool,
            "has_latency_events":   bool,
        },
        "issues": ["missing_all_trace_events", ...],
        "warnings": ["unbalanced_llm_events", "missing_completion_tokens", ...],
    }
```

The audit result is stored in the run record under `progress_data.trace` and is visible through `GET /runs/{run_id}`. When a panel scores zero because of missing trace data, the operator can read `trace.scoreability.behavior: false` and immediately know the cause is instrumentation, not agent behavior.

---

## 5. OpenTelemetry Integration (`agentscope/otel.py`)

**Problem.** Production deployments of evaluation pipelines require distributed tracing to debug cross-service latency and failures. At the same time, most development environments do not have an OTLP backend, so importing OpenTelemetry should never fail or slow down startup.

**Solution.** `otel.py` implements a dual-mode pattern: all exports are conditional on `AGENTSCOPE_OTEL_ENABLED=1` or the presence of an `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable. When neither is set, all telemetry functions return noop objects:

```python
@contextlib.contextmanager
def _noop_span_cm():
    yield _NoopSpan()          # no-op: set_attribute, record_exception, etc.

def start_span(name: str, *, tracer_name: str = "agentscope", ...):
    if not _enabled():
        return _noop_span_cm()  # zero overhead when OTEL is off
    otel = _import_otel()
    if otel is None:            # SDK not installed
        return _noop_span_cm()
    # ... create real span ...
```

The `_import_otel()` function attempts to import the OpenTelemetry SDK at runtime. If it is not installed, all telemetry calls silently become noops without raising `ImportError`. This allows AgentScope to ship with OpenTelemetry as an optional dependency that does not affect users who do not need distributed tracing.

When enabled, every API request and every pipeline stage produces a span with structured attributes including `agentscope.run_id`, `agentscope.agent_type`, and `agentscope.eval_inputs_count`, exportable to any OTLP-compatible backend such as Jaeger, Honeycomb, or Grafana Tempo.

---

## 6. Headless CI Gate (`agentscope/headless_ci.py`)

The headless CI module provides threshold-enforced metric gates for integration into any CI/CD pipeline:

```bash
python -m agentscope.headless_ci \
  --agent-folder /agents/my_rag_agent \
  --eval-input "What was the revenue for Q3?" \
  --min-metric eval_results.ir.ndcg=0.70 \
  --min-metric eval_results.geval.scores.faithfulness=0.70 \
  --max-metric eval_results.behavior.ghost_action_rate=0.10
```

Metric paths use dot notation and resolve through the nested JSON report structure. If any configured threshold is violated, the process exits with code 1 and prints the failing metrics to stderr. Combined with the FastAPI background execution model, this enables non-blocking CI evaluations that complete in parallel with other pipeline steps.

---

## 7. New Tests Added This Phase

```python
def test_job_queue_enqueue_and_retrieve():
    state  = build_run_state(agent_folder="tests/fake_agent", ...)
    record = enqueue_run(state, source="test")
    assert record["status"] == "queued"
    loaded = load_run(record["run_id"])
    assert loaded["run_id"] == record["run_id"]

def test_trace_audit_output_only():
    trace = AgentTrace(run_id="t", agent_input="q", agent_output="a", events=[])
    summary = summarize_trace(trace)
    assert summary["status"] == "output_only"
    assert "missing_all_trace_events" in summary["issues"]
    assert summary["scoreability"]["behavior"] is False
```

19 new tests covering the job queue, runner isolation, and trace audit pass.

---

## 8. Key Design Decisions

1. **Thread-safe JSON persistence over a database** — `run_store.py` uses file-based JSON with a threading lock rather than SQLite or PostgreSQL. This keeps the AgentScope process stateless for easy deployment and eliminates a database dependency for the CI use case.
2. **Noop telemetry over conditional imports** — wrapping every OTel call in a noop context manager rather than guarding with `if otel_enabled:` keeps calling code clean and ensures the telemetry layer can be added or removed without touching any tool or pipeline code.
3. **`scoreability` before tool dispatch** — surfacing instrumentation failures before running 60-second evaluations prevents operators from waiting for a full pipeline run only to receive zero scores with no explanation.
