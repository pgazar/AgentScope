# Engineering Progress Report 5: Dashboard, API, Deployment, and Testing
## AgentScope — Gradio Frontend, FastAPI Backend, Docker, CI, and Test Suite

**Author:** Pegah Zargarian  
**Program:** MS Data Science, Northeastern University  
**Report Period:** Phases 8–11 (Days 19–25)  
**Repository:** https://github.com/pgazar/AgenticScope

---

## 1. Objective

This phase delivered the complete user-facing surface of AgentScope: a five-panel Gradio dashboard with color-coded performance visualization, a FastAPI headless API for CI integration, Docker Compose deployment, GitHub Actions CI with metric regression gates, and a comprehensive test suite covering all deterministic components. The phase concluded with a full end-to-end smoke test demonstrating the pipeline running against both a minimal fake agent and the sample agentic RAG system.

---

## 2. Color Utility and Chart Builders

The `color_for_metric()` function encodes the threshold logic used across all five dashboard panels. Inverted metrics (lower is better) use the inverse threshold direction:

```python
THRESHOLDS = {
    "ir":        {"green": 0.70, "orange": 0.40},
    "agent":     {"green": 0.80, "orange": 0.60},
    "geval":     {"green": 0.80, "orange": 0.60},
    "cost_usd":  {"green": 0.020, "orange": 0.050, "invert": True},
    "latency_s": {"green": 2.0,   "orange": 5.0,   "invert": True},
    "hallu":     {"green": 0.10,  "orange": 0.25,  "invert": True},
}

def color_for_metric(value: float, metric_type: str) -> str:
    if value is None:
        return GRAY   # "#888780" — unscored metrics render gray
    cfg = THRESHOLDS.get(metric_type, THRESHOLDS["geval"])
    if cfg.get("invert"):
        return GREEN if value <= cfg["green"] else ORANGE if value <= cfg["orange"] else RED
    return GREEN if value >= cfg["green"] else ORANGE if value >= cfg["orange"] else RED
```

Each chart builder returns a `go.Figure` with bars colored per metric. A critical bug early in this phase caused Panel 4 (Cost) to appear completely empty: when all cost values were zero (because no token counts were captured), the bars had zero height and the Plotly figure showed blank axes. The fix added `textposition="outside"` with explicit value labels on every bar, so panels display the numeric value even when the bar height is zero:

```python
fig = go.Figure(go.Bar(
    x=labels, y=values,
    marker_color=colors,
    text=[f"${v:.5f}" if v else "$0.00" for v in raw],
    textposition="outside",
    cliponaxis=False,
))
```

---

## 3. Gradio Application (`agentscope/dashboard/app.py`)

The Gradio app uses `gr.Blocks` with a `gr.Progress()` callback for live progress reporting. A key usability fix addressed a silent failure mode: Gradio placeholder text is not a default value. Early versions of the app had `eval_inputs` always submit as empty, because the placeholder ("What was the revenue for Q3?") was not stored as the initial value:

```python
# Wrong — placeholder text is not a value:
eval_inputs_text = gr.Textbox(placeholder="What was the revenue for Q3?")

# Correct — value="..." sets the actual default:
eval_inputs_text = gr.Textbox(value="What was the revenue for Q3?", lines=4)
```

Input validation now uses `validate_run_request()` before the run is enqueued, surfacing errors as `gr.Warning()` banners rather than stack traces. The dashboard binds to `0.0.0.0` rather than `127.0.0.1` so it is reachable from outside the Docker container.

---

## 4. FastAPI Service (`agentscope/api/main.py`)

The API exposes four endpoints:

```
POST /evaluate          → enqueues a run, returns run_id immediately
GET  /runs/{run_id}     → returns status, stage, pct, and full progress_data
GET  /runs/{run_id}/report → returns the complete JSON report
GET  /health            → returns {"status": "ok"}
```

The `/evaluate` endpoint returns immediately with `status: "queued"` while the pipeline runs in the background. This prevents HTTP request timeouts for long-running evaluations. OpenTelemetry instrumentation is wired through `otel.py`, which provides noop implementations when the SDK is not configured:

```python
@app.post("/evaluate")
async def evaluate(req: EvalRequest):
    with start_span("agentscope.api.evaluate", tracer_name="agentscope.api") as span:
        _validate_request(req)
        state  = build_run_state(...)
        record = enqueue_run(state, source="api")
        return {"run_id": record["run_id"], "status": record["status"], ...}
```

---

## 5. Headless CI Gate (`agentscope/headless_ci.py`)

The headless CI module provides a command-line metric gate for integration into GitHub Actions or any CI/CD system:

```bash
python -m agentscope.headless_ci \
  --agent-folder tests/fake_agent \
  --eval-input "What is 2+2?" \
  --min-metric eval_results.geval.scores.faithfulness=0.7 \
  --max-metric eval_results.behavior.ghost_action_rate=0.1
```

The module polls `/runs/{run_id}` until completion, evaluates each configured threshold against the nested JSON report using dot-notation path resolution, and exits with code 1 if any threshold is violated. This enables metric regression testing as a quality gate on every pull request.

---

## 6. Docker Compose Deployment

The `docker-compose.yml` defines two services:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    ports: ["5432:5432"]

  agentscope:
    build: .
    ports: ["7860:7860", "8000:8000"]
    volumes:
      - ./outputs:/app/outputs
      - ${AGENTS_DIR:-./tests}:/agents
    environment:
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

The `AGENTS_DIR` mount at `/agents` allows users to point the dashboard at agent folders on their local machine without copying files into the container. `host.docker.internal:host-gateway` allows the container to reach services (e.g., a local PostgreSQL database) running on the host.

---

## 7. GitHub Actions CI

The CI workflow runs all deterministic tests on every push to `main`:

```yaml
- name: Run tests
  run: python -m pytest tests/ -q --ignore=tests/test_synth_gen.py
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

Tests that require `AnthropicModel` are skipped in CI when `ANTHROPIC_API_KEY` is not set, using `pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="requires API key")`. This allows the CI to run 71–74 deterministic tests cleanly without requiring secrets on every run, while the full 119-test suite passes locally when keys are present.

---

## 8. Test Suite Overview

The test suite covers all six evaluation tools plus integration:

| File | Coverage | Tests |
|---|---|---|
| `test_ir_evaluator.py` | All 5 IR functions, GT loading, multi-doc format | 16 |
| `test_agent_behavior.py` | Deterministic metrics, permission schema | 16 |
| `test_cost_analyzer.py` | Pricing computation, latency percentiles | 10 |
| `test_adversarial_eval.py` | Suite loading, category counts, prompt format | 8 |
| `test_geval_tool.py` | Criteria keys, ScoredMetric, drift, extraction | 13 |
| `test_compiler.py` | Report structure, file writing, key coverage | 5 |
| `test_fixes.py` | Inter-judge variance, LLM permission matching | 12 |
| `test_mode_a_integration.py` | Callback handler, event capture, normalization | 8 |
| `test_job_queue.py` | Queue operations, state management | 10 |
| `test_runner.py` | Agent loading, trace collection | 8 |
| Other | API, dashboard, otel, headless CI | 13 |
| **Total** | | **119** |

---

## 9. End-to-End Smoke Test

The smoke test (`smoke_test.py`) runs the complete pipeline against the fake agent and verifies that a JSON report is produced:

```python
result = graph.invoke(state)
assert result["final_report"] is not None
assert os.path.exists(f"outputs/{run_id}_run_report.json")
print("SMOKE TEST PASSED")
```

The test passes in under 90 seconds with `max_geval_responses=1` and `eval_budget_usd=2.00`.

---

## 10. Key Design Decisions

1. **Progressive panel rendering** — panels render as each tool completes rather than all at once. This provides immediate feedback for fast tools (IR, behavior) while longer tools (adversarial, G-Eval) are still running.
2. **Value labels on zero-height bars** — prevents the misleading appearance of an empty panel when all scores are zero due to missing trace data.
3. **Headless CI metric gates** — quantitative thresholds checked against a structured JSON report provide a more precise quality gate than pass/fail tests, enabling gradual performance regression detection.
