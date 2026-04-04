# AgentScope: A Multi-Tool Agentic Evaluation Framework with LLM-as-Judge Pipeline and Performance Dashboard

**Pegah Zargarian**  
MS Data Science, Khoury College of Computer Sciences, Northeastern University  
Expected Graduation: April 2026  
Target Role: Applied AI Engineer / AI Engineer (New Grad)  
Repository: https://github.com/pgazar/AgenticScope

---

## Abstract

As agentic AI systems transition from research prototypes into production deployments, a critical tooling gap has emerged: no standardized, reproducible framework exists for evaluating the full behavioral surface of a Python-based agentic system. Existing evaluation tools assess either final response quality in isolation or require deep integration with a specific orchestration framework, and neither approach captures the trace-level failures that determine production reliability. This paper presents AgentScope, an open-source evaluation framework that accepts any Python-based agentic system as input and produces a structured, reproducible quality report across five measurement dimensions: retrieval quality, agent behavior, LLM-as-judge response quality, input agent cost efficiency, and adversarial robustness. The framework is built around a LangGraph StateGraph pipeline that dispatches six specialized evaluation tools in a deterministic sequential order, renders all results in a color-coded Gradio dashboard grounded in evaluation literature thresholds, and exposes a FastAPI endpoint for headless CI/CD integration. AgentScope is fully containerized and deployable in a single command.

---

## 1. Introduction

The deployment of agentic AI systems — autonomous pipelines in which a language model selects and invokes external tools, retrieves documents, and produces multi-step plans — has grown substantially since the introduction of tool-augmented language models [CITE: ReAct, Yao et al., 2022]. Production deployments now include RAG pipelines for enterprise knowledge retrieval, tool-using assistants that take real-world actions such as sending email or querying databases, and multi-agent orchestrators that coordinate specialized sub-agents to complete complex tasks.

Despite this growth, evaluation practice has not kept pace. A persistent and problematic assumption underlies most current evaluation workflows: that a high-quality final response implies a correct execution trajectory. This assumption fails in at least four categories of production failure that are systematically invisible to output-only evaluation, each of which is described in Section 2.

The consequences are significant. Teams deploying agentic systems face an evaluation blind spot: they can measure whether the system's final answer is correct, but not whether it arrived at that answer through a sound process, whether it would withstand adversarial inputs, or whether the quality of its responses justifies its inference cost. AgentScope is designed to close this blind spot.

The framework makes four primary contributions. First, it defines a normalized trace schema that captures the full execution trajectory of any Python-based agentic system through two integration modes — a zero-code LangChain callback handler and a lightweight adapter function — without requiring modification to the target agent's source code. Second, it implements six specialized evaluation tools that operate on this normalized trace to compute 17 core metrics spanning retrieval quality, behavioral trajectory, response quality, cost efficiency, and adversarial robustness. Third, it adds a trace health audit layer that inspects every trace before evaluation tools run and reports which panels will produce meaningful scores, preventing silent zero-scoring from misleading the operator. Fourth, it provides a production-ready deployment package including a Gradio dashboard with progressive panel rendering, a FastAPI service with a background job queue, Docker Compose orchestration, and a headless CI gate for metric regression testing.

---

## 2. Problem Statement

Four failure modes dominate production agentic systems, and none are detectable through output-only evaluation.

**The ghost action.** An agent claims in its final response to have called a tool — retrieved a document, queried a database, sent a confirmation — when no corresponding tool call appears in its execution trace. The response appears correct and complete; only a metric that inspects the trace against the final output catches the fabrication. AgentScope detects this through ghost action rate: a cross-reference of action-verb patterns in the agent's output against tool_start events in the normalized trace.

**The interrogation loop.** A multi-turn agent repeatedly requests information from the user that was already provided in an earlier turn, because context was not retained across turns. Individual responses score well in isolation. The failure only surfaces when the full conversation is evaluated end-to-end through metrics such as knowledge retention and conversation completeness.

**The confident fabricator.** A RAG agent produces a polished, coherent response built on hallucinated facts not present in any retrieved document. Output quality metrics — fluency, helpfulness, apparent completeness — rate it favorably. Only faithfulness evaluation against the retrieved context and citation accuracy checks against source documents reveal the fabrication.

**Unsafe compliance.** An agent follows malicious or policy-violating instructions: prompt injection attempts, sensitive data exfiltration requests, or tool calls that the agent is not authorized to make. The response may appear helpful in isolation. This failure is only detectable through adversarial probing and tool permission validation against a declared schema.

These failures share a common structure: they require knowledge of what the agent did, not just what it said. AgentScope addresses this by operating at the trace level rather than the response level.

---

## 3. Related Work

Existing evaluation frameworks address portions of this problem but do not provide an integrated solution.

**RAGAS** [Es et al., 2023] evaluates RAG pipelines against faithfulness, answer relevancy, and context precision, but is limited to RAG-specific metrics and does not evaluate agent behavior or adversarial robustness. **TruLens** provides LLM-as-judge feedback loops for LLM applications but requires framework-specific integration and does not compute trajectory metrics or cost efficiency. **PromptFoo** provides adversarial and regression testing for LLM prompts but does not address retrieval quality or multi-step behavioral evaluation. **LangSmith** traces LangChain/LangGraph agent calls and provides a debugging interface, but is a proprietary observability platform rather than a metric-producing evaluation framework. **DeepEval** [Tang et al., 2024] provides a rich set of LLM-as-judge metrics through the G-Eval paradigm [Liu et al., 2023] and is used as AgentScope's judge backend, but does not include retrieval evaluation, behavioral trajectory scoring, or cost analysis.

AgentScope occupies the space between these tools: it provides a unified, open-source pipeline that integrates retrieval evaluation, trajectory evaluation, LLM-as-judge response scoring, cost analysis, and adversarial robustness testing under a single reproducible framework with no vendor dependency.

---

## 4. System Architecture

### 4.1 Overview

AgentScope is composed of three layers: an ingestion and orchestration layer, an evaluation tool layer, and an output layer. The ingestion layer accepts any Python-based agentic system and produces a normalized execution trace. The evaluation layer dispatches six tools against this trace in a deterministic sequential order. The output layer aggregates results into a structured JSON report and a five-panel Gradio dashboard.

The orchestration mechanism is a LangGraph StateGraph compiled dynamically at the start of each run. The active set of evaluation tools is determined by the IntakeAgent — a lightweight classifier that resolves which tools apply based on whether the target agent is a RAG pipeline, a tool-use agent, or a multi-agent system, and whether ground truth and knowledge base inputs are provided. This per-run graph compilation is the primary architectural motivation for using LangGraph: it enables clean separation between tool routing logic and tool implementation without requiring conditional branching inside any individual tool.

It is important to note that AgentScope itself is not an autonomous agent. The evaluation pipeline executes in a predetermined sequential order after the graph is compiled; no LLM is used to decide what to do next. LLMs appear in exactly three roles: as the G-Eval judge scoring response quality, as a semantic matcher for tool permission validation, and as the DeepEval Synthesizer backend for synthetic test set generation. All orchestration decisions are deterministic.

### 4.2 Agent Runner and Trace Collector

Before any evaluation tool runs, AgentScope loads the target agent, invokes it on each evaluation input, and normalizes all captured events into a common trace schema. This component is the backbone of the framework: without a reliable trace, all downstream evaluators have nothing to measure.

Two integration modes are supported:

**Mode A — LangChain/LangGraph agents.** The `AgentScopeCallbackHandler` is injected at runtime into the target agent's callback chain. It captures `on_tool_start`, `on_tool_end`, `on_llm_start`, and `on_llm_end` events, using a multi-key resolution helper (`_get(d, *keys, default)`) that handles non-standard trace schemas from different LangChain versions without requiring source modification.

**Mode B — Custom SDK agents.** The target agent exposes a `run(query: str) -> str` entry point and an optional `inject_trace_events(trace: AgentTrace) -> None` function. After each `run()` call, AgentRunner checks for this function and calls it if present, allowing the agent to backfill tool calls, retrieval doc keys, and actual API token counts from its own internal result dict into the normalized trace. This mode is validated against a 4-tool ReAct agent that calls the Anthropic SDK directly, bypassing LangChain entirely.

Both modes produce an `AgentTrace` dataclass containing the run ID, agent input, agent output, total latency, and a list of `TraceEvent` objects. Each `TraceEvent` records the event type, tool name and arguments, prompt and completion token counts, retrieved document identifiers, latency in milliseconds, and the model name of the evaluated agent.

A **trace audit** step runs automatically after trace collection and before any evaluation tool is dispatched. It inspects event coverage — whether LLM events, tool events, retrieval events, and token counts are present — and produces a `scoreability` report that tells the operator which panels will produce meaningful scores and which will show zeros due to missing trace data.

After trace collection, the agent itself runs in a **subprocess-isolated worker** (`runner_worker.py`). This ensures that errors, memory leaks, or infinite loops in the target agent's code cannot crash the AgentScope process.

### 4.3 Orchestration Layer

The LangGraph StateGraph is compiled once per run via `build_graph(active_tools: list[str])`. The execution order is fixed and sequential:

```
run_agent → [synth_gen →] [ir_evaluator →] agent_behavior
          → adversarial_eval → geval → cost_analyzer
          → compile_report → END
```

Each node reads from and writes to a shared `AgentState` TypedDict. The state contract is documented at the top of `graph.py` for every node. No node returns the full state object; each returns only the keys it writes, which LangGraph merges into the accumulated state. This prevents silent state mutation bugs.

Runs are managed through a **background job queue** (`job_queue.py`, `job_worker.py`, `run_store.py`). When a run is submitted through the dashboard or the API, it is immediately enqueued and assigned a run ID. A background worker picks it up and executes the pipeline. The dashboard polls every 500ms for progress updates, rendering each panel as soon as its corresponding tool completes rather than waiting for all five.

### 4.4 Evaluation Tools

**Tool 1 — IR Evaluator (deterministic).** Activated when a knowledge base or ground truth CSV is present. Computes Precision@k, Recall@k, Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (nDCG@k), and Hit Rate@k against labeled relevant document identifiers. The ground truth CSV supports multi-document relevance per query using pipe-separated document keys (`financial_q3:chunk_1|sample_finance:chunk_1`), enabling proper multi-document IR evaluation rather than the single-answer approximation common in simpler implementations. Ground truth is aligned to traces by query text matching rather than positional indexing, preventing score inflation from order mismatches.

**Tool 2 — Agent Behavior Evaluator (deterministic + G-Eval).** Activated for all agent types. Computes eight trajectory-level metrics from the normalized trace: tool selection accuracy (fraction of called tools that appear in the expected tool set), step budget efficiency (heuristic: actual steps / max steps), convergence (whether the agent completed within the step budget), argument correctness (G-Eval judgment of whether tool call parameters were valid given the triggering input), plan success (G-Eval judgment of whether the tool sequence was coherent and non-redundant), ghost action rate (regex cross-reference of action-verb claims in the final output against tool_start events in the trace), handoff correctness (multi-agent only), and tool permission validation. Permission validation uses an LLM semantic matcher (Claude Haiku via subprocess) rather than string normalization, correctly resolving naming variants such as `SendEmail` → `send_email` that regex-based approaches miss. G-Eval calls from this tool run in a subprocess to avoid asyncio event loop conflicts with Gradio.

**Tool 3 — G-Eval Response Quality Tool (G-Eval + Claude Haiku, temperature=0).** Activated for all agent types. Uses DeepEval's G-Eval implementation with Claude Haiku as the judge to score six criteria: task completion, faithfulness to retrieved context, hallucination rate, citation accuracy, helpfulness, and safety. For multi-turn agents, three additional criteria apply: conversation completeness, turn relevancy, and knowledge retention. Response caps (`max_geval_responses`) and a hard budget limit (`eval_budget_usd`) prevent silent cost overruns. Inter-judge variance is measured by re-scoring every test case through a secondary judge (GPT-4o-mini) using the identical criteria string, and the mean delta and standard deviation of score differences are reported alongside each metric. Calibration drift is measured via KL divergence between the current score distribution and a stored baseline from a prior run.

**Tool 4 — Quality-Cost Analyzer (deterministic).** Instruments the evaluated agent's own LLM calls — not AgentScope's judge calls — to compute cost per query, cost per successful task, p50 and p95 latency, and a quality-cost index (G-Eval composite score divided by cost per query). Token counts are read from actual API response objects (`r.usage.input_tokens`, `r.usage.output_tokens`) for Mode B agents, not estimated from word counts. The `agent_model` field is stored in every cost result so the pricing table lookup is auditable.

**Tool 5 — Adversarial Robustness Evaluator (concurrent G-Eval).** Activated for all agent types. Runs 12 adversarial prompts across four attack categories — prompt injection, unsafe tool use, instruction override, and sensitive action — against the target agent, scoring each response using a resistance criteria string through G-Eval. All 12 prompts execute concurrently via a `ThreadPoolExecutor` (max_workers=4) with a 30-second per-prompt timeout enforced through `future.result(timeout=30)`. Timed-out prompts are conservatively counted as resisted rather than as failures, preventing infrastructure latency from penalizing the agent's safety score.

**Tool 6 — Synthetic Test Set Generator (DeepEval Synthesizer).** Activated when no ground truth CSV is available but a knowledge base is provided. Uses DeepEval's Synthesizer with Claude Haiku to generate realistic Q&A pairs from the knowledge base documents, capped at `max_synth_pairs=50` to control cost. The generated pairs are saved to `outputs/{run_id}_synthetic_gt.csv` for inspection and reuse in subsequent runs.

### 4.5 Observability

Every evaluation run emits structured JSON logs via `structlog`, capturing run ID, component name, event type, and all metric inputs and outputs. The logging layer includes an optional OpenTelemetry integration (`otel.py`) that can export spans to any OTLP-compatible backend when `AGENTSCOPE_OTEL_ENABLED=1` is set. When OpenTelemetry is not configured, all telemetry calls are no-ops — a noop span context manager is used throughout, so there is no import-time cost or runtime failure when the OpenTelemetry SDK is absent.

### 4.6 Report Compiler

The `ReportCompiler` aggregates all tool outputs into two artifacts: a structured JSON report written to `outputs/runs/{run_id}/run_report.json`, and a dashboard payload. The JSON report includes all metric scores, run metadata, YAML configuration snapshot, and a trace audit summary. AgentScope's own framework running cost (G-Eval judge token usage) is stored in `geval_results._judge_cost_est_usd` and is intentionally separated from the evaluated agent's cost metrics to keep the evaluation view focused on the system under evaluation.

---

## 5. Dashboard

The Gradio dashboard renders evaluation results across five bar graph panels using Plotly via `gr.Plot`. Every bar is colored dynamically based on the metric's value relative to literature-grounded performance thresholds: green for good performance, orange for acceptable, red for poor.

| Panel | Metrics | Green | Orange | Red |
|---|---|---|---|---|
| IR metrics | Precision@k, Recall@k, MRR, nDCG@k, Hit Rate@k | ≥ 0.70 | 0.40–0.69 | < 0.40 |
| Agentic metrics | Tool accuracy, Plan success, Step budget eff., Arg. correctness, Convergence, Ghost action rate | ≥ 0.80 | 0.60–0.79 | < 0.60 |
| Response quality | Task completion, Faithfulness, Hallucination, Citation accuracy, Helpfulness, Safety | ≥ 0.80 | 0.60–0.79 | < 0.60 |
| Cost analysis | Cost/query, Cost/success, p50 latency, p95 latency, Quality-cost index | ≤ $0.02 / ≤ 2s | $0.02–$0.05 / 2–5s | > $0.05 / > 5s |
| Safety & robustness | Injection resistance, Unsafe compliance, Attack success rate, Policy violation rate | — | — | — |

Hallucination rate and all safety/compliance metrics are inverted: green indicates low values (≤0.10 for hallucination), and the quality-cost index uses standard orientation (higher is better). All bar values are labeled with their numeric score so panels remain interpretable when bars are near-zero height.

The dashboard runs evaluations through a background job queue. When the user submits a run, it is immediately enqueued and a run ID is returned. The dashboard polls every 500ms and renders each panel as soon as its corresponding tool finishes, providing progressive feedback rather than a single blocking wait.

---

## 6. Input Scenarios

**Scenario A — RAG pipeline with ground truth CSV.** All five evaluation tools activate. The IR Evaluator benchmarks retrieval against the provided ground truth. G-Eval scores each response. The cost analyzer instruments all inference calls. All five dashboard panels populate.

**Scenario B — RAG pipeline without ground truth.** The Synthetic Test Set Generator activates first and generates Q&A pairs from the knowledge base. The IR Evaluator then runs against these synthetic pairs. The run report notes that synthetic ground truth is expected to produce slightly higher scores than human-labeled ground truth due to distribution match.

**Scenario C — Tool-use or multi-agent agent, no knowledge base.** The IR Evaluator and Synthetic Generator are skipped. The Behavior Evaluator activates fully, including handoff correctness scoring for multi-agent architectures. G-Eval evaluates task completion and response quality per trace.

**Scenario D — Regression testing.** The same agent evaluated before and after a change — a prompt edit, model swap, or tool schema update. Each run produces its own JSON report in `outputs/runs/`. Comparing the two reports surfaces metric deltas and makes the cost-quality tradeoff of any change explicit.

---

## 7. Deployment

AgentScope ships with three deployment modes governed by a single `config.yaml` manifest.

**Local development.** `docker compose up` starts the full stack: the LangGraph pipeline, PostgreSQL with pgvector, the Gradio dashboard on port 7860, and the FastAPI service on port 8000. Agent folders outside the repository are mounted at `/agents/` by setting `AGENTS_DIR` in `.env`.

**API / CI mode.** `uvicorn agentscope.api.main:app --port 8000` starts the headless FastAPI service. The `headless_ci.py` module provides a command-line metric gate: `python -m agentscope.headless_ci --agent-folder path/to/agent --eval-input "query" --min-metric eval_results.geval.scores.faithfulness=0.7`. If any configured threshold is violated, the process exits with code 1, blocking the CI pipeline.

**Cloud judge deployment.** `modal deploy modal_judge.py` deploys the G-Eval judge pipeline as a serverless Modal function. The Anthropic API key is managed as a Modal secret, never appearing in code or configuration files.

---

## 8. Metric Coverage Summary

AgentScope computes metrics across five evaluation dimensions:

**Retrieval:** Precision@k, Recall@k, MRR, nDCG@k, Hit Rate@k (5 metrics).

**Behavioral trajectory:** Tool selection accuracy, plan success, step budget efficiency, argument correctness, convergence, ghost action rate, handoff correctness, step match (exact, precision, recall), permission violation rate, safety score (11 metrics).

**Response quality:** Task completion, faithfulness, hallucination rate, citation accuracy, helpfulness, safety (6 metrics).

**Cost efficiency:** Cost per query, cost per successful task, p50 latency, p95 latency, quality-cost index (5 metrics).

**Adversarial robustness:** Attack success rate, unsafe compliance rate, prompt injection resistance, permission violation rate, per-category breakdown (4 aggregate metrics).

**Judge reliability:** Inter-judge variance (mean delta, std deviation, high-variance case flags), calibration drift via KL divergence (2 reliability metrics per G-Eval criterion).

Total: 17 core metrics + judge reliability metrics across 6 tools. Every metric is rendered in the dashboard with a reference-grounded performance color. Every run produces a reproducible JSON report. Every configuration is captured in a version-controlled YAML manifest.

---

## References

[1] Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. *International Conference on Learning Representations (ICLR) 2023*. arXiv:2210.03629.

[2] Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15217.

[3] Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). G-Eval: NLG Evaluation Using GPT-4 with Better Human Alignment. *Proceedings of EMNLP 2023*. arXiv:2303.16634.

[4] Tang, R., et al. (2024). DeepEval: An LLM Evaluation Framework. arXiv:2407.10490.

[5] Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023*. arXiv:2306.05685.

[6] Dong, Y., et al. (2024). A Survey on Evaluation of Large Language Models. *ACM Transactions on Intelligent Systems and Technology*, 15(3). arXiv:2307.03109.

[7] Zargarian, P. (2026). AgentScope: Open-Source Agentic Evaluation Framework. https://github.com/pgazar/AgenticScope.
