# AgentScope

An open-source Python framework for evaluating agentic AI systems — RAG pipelines, tool-use agents, single-agent or multi-agent, single-turn or multi-turn.

## What it does

Points AgentScope at a Python agent folder → runs a LangGraph evaluation pipeline → renders a Gradio dashboard with color-coded metric panels across retrieval, behavior, response quality, cost, and adversarial robustness.

## Metrics (17 total)

- **IR (retrieval):** Precision@k, Recall@k, MRR, nDCG, Hit Rate@k
- **Agent behavior:** Tool selection accuracy, plan success, path efficiency, argument correctness, convergence, step match (exact/precision/recall), handoff correctness
- **Response quality (G-Eval):** Task completion, faithfulness, hallucination rate, citation accuracy, helpfulness, safety
- **Cost:** Cost/query, cost/successful task, p50/p95 latency, quality-cost index

## Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (ReAct StateGraph) |
| LLM judge | G-Eval (DeepEval) + Claude Sonnet |
| Dashboard | Gradio + Plotly |
| API | FastAPI |
| Deployment | Docker Compose + Modal.com |

## Quick start

```bash
docker compose up
# Dashboard at localhost:7860
```
