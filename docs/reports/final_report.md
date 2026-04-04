# AgentScope: A Multi-Tool Agentic Evaluation Framework with LLM-as-Judge Pipeline and Performance Dashboard

## Final Report — System Capabilities, Experimental Results, and Conclusions

**Pegah Zargarian**  
MS Data Science, Khoury College of Computer Sciences, Northeastern University  
Expected Graduation: April 2026  
Repository: https://github.com/pgazar/AgenticScope

---

## Abstract

This paper presents AgentScope, an open-source evaluation framework for Python-based agentic AI systems. AgentScope addresses the gap between output-only evaluation — measuring only the final response — and the trace-level assessment required to detect the failure modes that dominate production deployments of agentic systems. The framework accepts any Python agent as input through two zero-modification integration modes, dispatches six specialized evaluation tools in a LangGraph-orchestrated pipeline, and renders results across 17 metrics in a five-panel Gradio dashboard with performance-colored visualizations grounded in evaluation literature thresholds. Evaluation against a sample 4-tool ReAct RAG agent demonstrates that the system achieves near-perfect retrieval quality (nDCG@5 = 1.00, Hit Rate@5 = 1.00) while surfacing a critical behavioral failure invisible to output-only evaluation: despite correct retrieval, the agent's tool execution sequence was judged incoherent by the G-Eval behavioral assessor (plan_success = 0.20). Cost efficiency analysis reveals a quality-cost index of 235.5 at $0.00241 per query, confirming that Claude Haiku delivers high-quality responses for this retrieval domain at minimal inference cost. Adversarial evaluation identifies meaningful safety gaps: only 67% of prompt injection attempts were resisted and 33% of adversarial prompts triggered policy violations, providing actionable findings for production hardening. The complete implementation comprises 4,129 lines of Python across 29 source files, 119 automated tests, and a GitHub Actions CI pipeline with metric regression gates.

---

## 1. Introduction

Agentic AI systems — pipelines in which a language model selects and invokes external tools, retrieves documents, and produces multi-step plans — are increasingly deployed in production environments for enterprise knowledge retrieval, automated task execution, and multi-agent coordination. Despite this growth, evaluation tooling has lagged behind deployment practice.

The dominant evaluation paradigm assesses the quality of the agent's final response: is the answer correct, helpful, and safe? This approach systematically misses four failure modes that are common in production and that AgentScope was designed to detect: ghost actions (fabricated tool execution claims), interrogation loops (context not retained across multi-turn conversations), confident fabrication (hallucinated facts presented with apparent grounding), and unsafe compliance (compliance with adversarial or policy-violating instructions).

AgentScope was developed over approximately 35 days of active engineering, progressing from a project specification through implementation, integration testing, real-agent validation, and architecture hardening. The framework reached production-readiness with a complete test suite, Docker Compose deployment, FastAPI headless API, and OpenTelemetry observability hooks.

---

## 2. Related Work

Existing evaluation frameworks address portions of the agentic evaluation problem but do not provide an integrated solution. RAGAS [1] evaluates RAG-specific metrics (faithfulness, answer relevancy, context precision) but does not address agent behavior or adversarial robustness. TruLens provides LLM feedback loops for LangChain applications but requires framework-specific integration. PromptFoo covers adversarial testing for LLM prompts but not retrieval quality or trajectory scoring. LangSmith provides distributed tracing and debugging for LangChain agents but is a proprietary observability platform rather than a metric-producing framework. DeepEval [4] provides the G-Eval LLM-as-judge mechanism [3] used in AgentScope's response quality tool but does not integrate retrieval evaluation, behavioral trajectory scoring, or cost analysis.

AgentScope occupies the intersection: a unified, open-source pipeline that integrates all five evaluation dimensions under a single reproducible framework with no vendor dependency.

---

## 3. System Overview

AgentScope is composed of three layers. The ingestion layer accepts any Python-based agentic system through two integration modes and produces a normalized execution trace. The evaluation layer dispatches six tools against this trace in a deterministic sequential order managed by a LangGraph StateGraph compiled per run. The output layer aggregates results into a structured JSON report and a five-panel Gradio dashboard.

The six evaluation tools compute 17 core metrics:

- **IR Evaluator:** Precision@k, Recall@k, MRR, nDCG@k, Hit Rate@k
- **Behavior Evaluator:** Tool selection accuracy, plan success, step budget efficiency, argument correctness, convergence, ghost action rate, handoff correctness, step match
- **G-Eval Tool:** Task completion, faithfulness, hallucination rate, citation accuracy, helpfulness, safety
- **Cost Analyzer:** Cost/query, cost/successful task, p50 latency, p95 latency, quality-cost index
- **Adversarial Evaluator:** Attack success rate, unsafe compliance rate, prompt injection resistance, policy violation rate
- **Synth Generator:** n_pairs generated, coverage score (activates only when no GT is provided)

All evaluations run through a background job queue. The dashboard polls every 500ms and renders each panel progressively as its tool completes. The pipeline is fully containerized and deployable in a single `docker compose up` command.

---

## 4. Experimental Setup

**Target agent.** Evaluation was performed against a sample agentic RAG system: a 4-tool ReAct agent implementing the Reasoning + Acting paradigm [2]. The agent uses PostgreSQL with pgvector for document storage, hybrid RRF (Reciprocal Rank Fusion) retrieval combining keyword and dense vector search, Claude Haiku as the language model, and four tools: `rag_retrieve`, `calculate`, `analyze_data`, and `summarize`. The agent does not use LangChain; it calls the Anthropic SDK directly. Integration with AgentScope used Mode B (adapter function) with `inject_trace_events()` backfilling actual Anthropic API token counts and retrieval doc keys from the agent's native result structure.

**Evaluation query.** The evaluation was run on a single representative query: *"What was the revenue for Q3?"* — a query with two known relevant documents in the ground truth CSV (`financial_q3_2024:chunk_1` and `sample_finance:chunk_1`).

**Ground truth.** A 100-row ground truth CSV with pipe-separated `relevant_docs` columns, generated from the agent's test suite. The CSV includes adversarial cases (no-answer queries, ambiguous queries, division-by-zero tool calls) as well as standard factual retrieval queries.

**Configuration.** Judge model: `claude-haiku-4-5-20251001` at temperature=0. `max_geval_responses=1`, `eval_budget_usd=2.00`, `k=5`.

**Adversarial suite.** 12 prompts across 4 categories (3 per category), executed concurrently with a 30-second per-prompt timeout.

---

## 5. Results

### 5.1 Panel 1 — IR Metrics

| Metric | Score | Status |
|---|---|---|
| Precision@k (k=5) | 0.40 | Orange |
| Recall@k | 1.00 | Green |
| MRR | 1.00 | Green |
| nDCG@k | 1.00 | Green |
| Hit Rate@k | 1.00 | Green |

The Precision@k score of 0.40 indicates that 2 of the 5 retrieved documents were relevant — the agent retrieved exactly the two ground truth documents, but also retrieved three additional non-relevant documents, reducing precision. The remaining four metrics score at or near 1.00, indicating that the most relevant document was ranked first (MRR = 1.00), the ranking quality was optimal (nDCG = 1.00), and at least one relevant document appeared in the top-5 for every evaluated query (Hit Rate = 1.00). These scores confirm that the hybrid RRF retrieval pipeline is functioning correctly for the tested domain.

### 5.2 Panel 2 — Agentic Metrics

| Metric | Score | Status | Notes |
|---|---|---|---|
| Tool selection accuracy | N/A | — | No expected tool list provided |
| Plan success | 0.20 | Red | G-Eval judged the tool sequence incoherent |
| Step budget efficiency | 1.00 | Green | Agent completed within max_steps=10 |
| Argument correctness | 1.00 | Green | Tool call parameters were valid |
| Convergence | 1.00 | Green | No infinite loops detected |
| Ghost action rate | 0.00 | Green | No fabricated tool execution claims |

The most significant finding in this panel is plan_success = 0.20. Despite perfect retrieval (nDCG = 1.00), the G-Eval judge assessed the overall tool execution sequence as incoherent — the sequence of tool calls did not form a logical, non-redundant plan for completing the task. This is precisely the failure mode that output-only evaluation cannot detect: the agent retrieved the correct documents and produced a reasonable-sounding answer, but the execution trajectory was judged unsound. This finding would be invisible in any evaluation framework that does not inspect the trace.

Ghost action rate = 0.00 confirms that the agent made no false claims of tool execution — every tool call referenced in the final answer corresponded to an actual `tool_start` event in the normalized trace.

### 5.3 Panel 3 — Response Quality (G-Eval)

| Metric | Score | Status | Notes |
|---|---|---|---|
| Task completion | 0.90 | Green | Agent answered the question adequately |
| Faithfulness | 0.00 | Red | Response not traceable to retrieved context |
| Hallucination rate | 0.00 | Green (inverted) | No hallucinated facts detected |
| Citation accuracy | 0.60 | Orange | Some citations present, not all verified |
| Helpfulness | 0.90 | Green | Response was specific and actionable |
| Safety | 1.00 | Green | No harmful content |

The divergence between faithfulness (0.00) and hallucination (0.00) reveals a nuanced failure: the agent's response was not grounded in the retrieved context (faithfulness = 0.00) but also did not introduce facts that were demonstrably wrong (hallucination = 0.00). This pattern is consistent with an agent that answered from its parametric knowledge rather than the retrieved documents — a failure of attribution rather than accuracy. In a production RAG system, this distinction matters significantly: users expect answers to be grounded in the retrieved source material, not generated from training data.

### 5.4 Panel 4 — Cost Analysis

| Metric | Score | Status |
|---|---|---|
| Cost/query | $0.00241 | Green |
| Cost/successful task | $0.01203 | Green |
| p50 latency | 3.364s | Orange |
| p95 latency | 3.364s | Orange |
| Quality-cost index | 235.5 | Green |

The quality-cost index of 235.5 (computed as mean G-Eval score / cost per query = 0.567 / 0.00241) indicates that Claude Haiku delivers strong quality relative to its inference cost for this retrieval domain. The latency of 3.364s is orange (between the 2s green threshold and the 5s red threshold), reflecting the multiple LLM calls required by the ReAct agent's reasoning loop. Cost per successful task of $0.01203 is computed as cost per query / plan_success (0.00241 / 0.20), reflecting that the low plan success rate increases the effective cost per successfully completed task.

### 5.5 Panel 5 — Safety and Robustness

| Metric | Score | Status | Notes |
|---|---|---|---|
| Prompt injection resistance | 0.67 | Orange | 2/3 injection attempts resisted |
| Unsafe compliance rate | 0.17 | Orange | 2/12 prompts produced unsafe responses |
| Attack success rate | 0.17 | Orange | 2/12 attacks succeeded |
| Policy violations | 0.33 | Red | 4/12 prompts triggered policy violations |

The adversarial results identify concrete safety gaps. The policy violation rate of 0.33 (4 of 12 prompts triggered violations) is the most significant finding: the agent responded to policy-violating instructions in a third of adversarial cases. Per-category breakdown shows that `unsafe_tool_use` was fully resisted (3/3), while `instruction_override` and `sensitive_action` categories showed the weakest resistance (1/3 and 1/3 resisted, respectively). These findings provide specific, actionable targets for production hardening.

---

## 6. Discussion

### 6.1 Key Finding: The Retrieval-Behavior Divergence

The most important result from this evaluation is the divergence between IR metrics and behavioral metrics. The agent achieved perfect ranking quality (nDCG = 1.00) while receiving a behavioral coherence score of 0.20. This pattern — correct retrieval paired with incoherent execution — represents a class of agent failure that is completely invisible to output-only evaluation. It suggests that retrieval quality is a necessary but not sufficient condition for correct agent behavior: the agent must not only retrieve the right documents but also construct a sound, non-redundant tool execution plan from them.

This finding validates the core design hypothesis of AgentScope: that evaluation must operate at the trace level, not the response level.

### 6.2 Faithfulness vs. Hallucination

The simultaneous occurrence of faithfulness = 0.00 and hallucination = 0.00 illustrates that these two metrics, while related, measure different properties. Faithfulness measures whether the response is grounded in the retrieved context; hallucination measures whether the response contains demonstrably false statements. A response can be true but ungrounded — drawn from parametric knowledge rather than retrieved sources. For enterprise RAG deployments where auditability and source traceability are requirements, faithfulness = 0.00 is a critical failure even when the factual content is correct.

### 6.3 Limitations

Several limitations of this evaluation should be acknowledged.

**Single-query evaluation.** The results are based on a single evaluation query. A comprehensive evaluation would run the full 100-query ground truth set, providing statistically meaningful mean scores and enabling identification of query-type-specific failure patterns.

**Synthetic vs. human-labeled ground truth.** The ground truth CSV was constructed from the agent's test suite rather than independent human annotation. This may introduce evaluation bias: queries that the agent was designed to answer well are over-represented relative to the full distribution of real user queries.

**Plan success scoring.** The plan_success = 0.20 result reflects G-Eval's judgment of the tool sequence based on the criteria string alone. The judge receives only the sequence of tool names — not the intermediate observations or the final answer — and may penalize sequences that are rational given the observations but appear suboptimal without that context.

**Synthetic G-Eval calibration.** G-Eval scores are calibrated to the judge model (Claude Haiku). Different judge models may produce systematically different scores for the same agent behavior. Inter-judge variance measurements (mean delta vs. GPT-4o-mini) help quantify this uncertainty.

---

## 7. Conclusion

AgentScope successfully delivers a unified, open-source evaluation framework for Python-based agentic systems that addresses the four production failure modes identified in the problem statement. The framework achieves the following capabilities:

**Trace-level failure detection.** AgentScope detects ghost actions through cross-referencing final output claims against trace events, detects interrogation loops through multi-turn G-Eval conversation completeness scoring, detects confident fabrication through faithfulness evaluation against retrieved context, and detects unsafe compliance through a concurrent 12-prompt adversarial test suite with per-category resistance scoring.

**Zero-modification integration.** Two integration modes — a LangChain callback handler (Mode A) and an adapter function pattern (Mode B) — enable evaluation of any Python-based agentic system without modifying its source code. Mode B was validated against a 4-tool ReAct agent using the Anthropic SDK directly.

**Real token-cost measurement.** By reading actual API response token counts rather than estimating from word counts, AgentScope produces accurate cost-per-query and quality-cost index measurements. The sample RAG agent achieves a quality-cost index of 235.5 at $0.00241 per query with Claude Haiku.

**Actionable safety findings.** Adversarial evaluation identified specific attack categories where the sample agent is vulnerable (instruction_override: 33% resistance, sensitive_action: 33% resistance) while confirming strong resistance in others (unsafe_tool_use: 100% resistance). These findings are specific enough to guide targeted hardening.

**Production-ready deployment.** The framework ships with Docker Compose deployment, a FastAPI headless API with background job queue, GitHub Actions CI with metric regression gates, and optional OpenTelemetry observability hooks — enabling integration into existing ML deployment workflows.

### 7.1 Future Work

Several directions for future development were identified during the course of this project:

**Multi-agent evaluation.** The current implementation covers handoff correctness (was context passed accurately between agents?) but does not address conflict resolution (what happens when two sub-agents produce contradictory outputs?) or shared memory consistency (do sub-agents read the same state from a shared memory store?). These are the dominant failure modes in orchestrator-specialist multi-agent architectures.

**Streaming evaluation.** The current framework evaluates agents synchronously — the agent runs to completion before any evaluation begins. Long-running agents (e.g., agents that stream partial results) require incremental trace collection and progressive metric computation.

**Broader adversarial coverage.** The current adversarial suite of 12 prompts across 4 categories provides coverage of the most common attack vectors. Production hardening would benefit from a larger suite incorporating recent jailbreaking benchmarks and domain-specific adversarial examples.

**Regression baselining.** The KL divergence calibration drift mechanism is implemented but requires a prior run as a baseline. A curated set of reference baselines for common agent architectures would enable out-of-the-box drift detection on first evaluation.

**Modal.com cloud deployment.** The `modal_judge.py` serverless G-Eval judge is implemented but requires the `anthropic-api-key` Modal secret to be configured. Full cloud deployment would enable evaluation runs without local infrastructure.

---

## References

[1] Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *arXiv:2309.15217*.

[2] Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*. *arXiv:2210.03629*.

[3] Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). G-Eval: NLG Evaluation Using GPT-4 with Better Human Alignment. *EMNLP 2023*. *arXiv:2303.16634*.

[4] Tang, R., et al. (2024). DeepEval: An LLM Evaluation Framework. *arXiv:2407.10490*.

[5] Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023*. *arXiv:2306.05685*.

[6] Dong, Y., Ding, N., Yao, Y., Wang, P., Zhou, J., & Wang, Z. (2024). A Survey on Evaluation of Large Language Models. *ACM Transactions on Intelligent Systems and Technology*, 15(3). *arXiv:2307.03109*.

[7] Anthropic. (2024). Claude Haiku (claude-haiku-4-5-20251001): Model card and pricing. https://www.anthropic.com/claude/haiku.

[8] DeepEval. (2024). DeepEval documentation: GEval metric. https://docs.confident-ai.com/docs/metrics-llm-evals.

[9] LangGraph. (2024). LangGraph documentation: StateGraph. https://langchain-ai.github.io/langgraph/.

[10] Zargarian, P. (2026). AgentScope: Open-Source Agentic Evaluation Framework. *GitHub*. https://github.com/pgazar/AgenticScope.
