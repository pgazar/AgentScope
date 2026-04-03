# AgentScope

> An open-source Python framework for evaluating agentic AI systems across retrieval quality, behavior, response quality, cost, and adversarial robustness.

[![CI](https://github.com/pgazar/AgenticScope/actions/workflows/ci.yml/badge.svg)](https://github.com/pgazar/AgenticScope/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What it does

Point AgentScope at any Python agent folder → it runs your agent on evaluation inputs → scores it across 17 metrics → renders a color-coded Gradio dashboard.

Output-only evaluation misses the failure modes that matter most in production. AgentScope catches them at the **trace level**:

| Failure mode | What it looks like | How AgentScope catches it |
|---|---|---|
| **Ghost action** | Agent says "I sent the email" but never called the tool | Trajectory metrics — tool calls vs. final answer |
| **Interrogation loop** | Multi-turn agent keeps asking for info the user already provided | Knowledge retention (G-Eval multi-turn) |
| **Confident fabricator** | Fluent, confident answer built on hallucinated facts | Faithfulness + hallucination rate (G-Eval) |
| **Unsafe compliance** | Agent follows a prompt injection or unsafe tool request | Adversarial evaluation suite |

---

## Dashboard

Five color-coded panels — green / orange / red based on literature-grounded thresholds:

```
Panel 1: IR metrics          Panel 2: Agentic metrics
Panel 3: Response quality    Panel 4: Cost & latency
Panel 5: Safety & robustness
```

Every bar is labeled with its value so panels are readable even when scores are zero.

---

## Metrics (17 total)

### IR / Retrieval
| Metric | Description |
|---|---|
| Precision@k | Fraction of top-k retrieved docs that are relevant |
| Recall@k | Fraction of all relevant docs retrieved in top-k |
| MRR | Mean Reciprocal Rank of first relevant doc |
| nDCG@k | Ranking quality — rewards relevant docs appearing earlier |
| Hit Rate@k | Fraction of queries with at least one relevant doc in top-k |

### Agent Behavior (Trajectory)
| Metric | Description |
|---|---|
| Tool selection accuracy | Whether the agent called the expected tools |
| Plan success | G-Eval: was the tool sequence logical and non-redundant? |
| Step budget efficiency | How far under the step budget the agent completed |
| Argument correctness | G-Eval: were tool call parameters valid? |
| Convergence | Did the agent finish within the allowed step budget? |
| Step match | Ordered/unordered comparison of actual vs. reference steps |
| Handoff correctness | Multi-agent only: was context passed accurately? |

### Response Quality (G-Eval LLM-as-judge)
| Metric | Description |
|---|---|
| Task completion | Did the agent fully complete the user's task? |
| Faithfulness | Are all claims supported by retrieved context? |
| Hallucination rate | Does the response introduce unsupported facts? |
| Citation accuracy | Are citations traceable to specific retrieved docs? |
| Helpfulness | Is the response actionable and specific? |
| Safety | Does the response avoid harmful content? |

### Cost & Efficiency
| Metric | Description |
|---|---|
| Cost per query | LLM token cost per evaluation query |
| Cost per successful task | Cost normalized by plan success rate |
| p50 / p95 latency | Median and 95th-percentile response times |
| Quality-cost index | G-Eval score ÷ cost per query |

---

## Sample results — capstone-rag ReAct agent

Evaluated against a 4-tool ReAct RAG agent (PostgreSQL + pgvector, hybrid retrieval, Claude Haiku).
Query: *"What was the revenue for Q3?"*

| Panel | Metric | Score |
|---|---|---|
| IR | nDCG@5 | **1.0** |
| IR | Hit Rate@5 | **1.0** |
| IR | MRR | **1.0** |
| Behavior | Arg. correctness | **1.0** |
| Behavior | Convergence | **1.0** |
| Behavior | Step budget eff. | **1.0** |
| G-Eval | Task completion | **0.9** |
| G-Eval | Safety | **1.0** |
| G-Eval | Hallucination | **0.1** |
| Cost | Cost/query | **$0.0037** |
| Cost | p50 latency | **5.9s** |
| Safety | Adversarial resistance | **75%** |

---

## Tech stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (ReAct StateGraph) |
| LLM judge | G-Eval (DeepEval) + Claude Haiku/Sonnet |
| Synth generation | DeepEval Synthesizer |
| Adversarial testing | Curated YAML prompt suite + G-Eval scoring |
| Judge reliability | Inter-judge variance + KL calibration drift |
| Observability | structlog (structured JSON logging) |
| Dashboard | Gradio + Plotly (5 panels) |
| API | FastAPI (`/evaluate`, `/health`) |
| Deployment | Docker Compose (local) + Modal.com (serverless) |
| Config | YAML + Pydantic |
| Testing | pytest (74 tests) + GitHub Actions CI |

---

## Quick start

### Local (recommended for development)

```bash
git clone https://github.com/pgazar/AgenticScope
cd AgenticScope

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # add ANTHROPIC_API_KEY and OPENAI_API_KEY
python -m agentscope.dashboard.app
# → http://127.0.0.1:7860
```

### Docker (full stack with pgvector)

```bash
docker compose up
# → Gradio at localhost:7860  |  FastAPI at localhost:8000
```

### Run the tests

```bash
python -m pytest tests/ -q
# 74 passed
```

---

## Evaluating your own agent

Your agent folder needs one file:

```python
# your_agent/main.py
def run(query: str) -> str:
    # call your agent here
    return answer
```

Point the dashboard at it:

| Field | Value |
|---|---|
| Agent folder path | `your_agent/` |
| Agent model name | `claude-sonnet-4-5` (or whatever your agent uses) |
| Evaluation inputs | one query per line |
| Agent type | `rag` / `tool_use` / `multi_agent` / `hybrid` |

For **LangChain / LangGraph agents**, AgentScope injects a callback handler automatically — no code changes needed.

For **custom agents** (like capstone-rag which calls the Anthropic SDK directly), add an optional `inject_trace_events(trace)` function to your `main.py` to backfill retrieval docs and token counts into the trace.

---

## Input scenarios

| Scenario | Tools activated |
|---|---|
| RAG + ground truth CSV | IR evaluator + all behavior + G-Eval + cost + adversarial |
| RAG, no ground truth | Synth gen → IR evaluator + all behavior + G-Eval + cost + adversarial |
| Tool-use / multi-agent | Behavior + G-Eval + cost + adversarial (no IR) |
| Regression testing | Run before and after a change, compare JSON reports |

---

## API (headless / CI mode)

```bash
# Start the API
uvicorn agentscope.api.main:app --port 8000

# Trigger an evaluation
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"agent_folder": "tests/fake_agent", "agent_type": "tool_use", "eval_inputs": ["What is 2+2?"]}'

# Check health
curl http://localhost:8000/health
```

---

## Project structure

```
agentscope/
├── agentscope/
│   ├── config.py          # Pydantic config + YAML loader
│   ├── runner.py          # AgentRunner — loads agent, runs inputs, normalizes trace
│   ├── tracer.py          # LangChain BaseCallbackHandler (Mode A integration)
│   ├── orchestrator/
│   │   ├── graph.py       # LangGraph StateGraph
│   │   └── state.py       # AgentState TypedDict
│   ├── tools/             # 6 evaluation tools
│   ├── judge/             # G-Eval criteria, variance, model routing
│   ├── report/            # JSON report compiler
│   ├── dashboard/         # Gradio app + Plotly charts
│   └── api/               # FastAPI endpoints
├── eval_targets/
│   └── capstone_rag/      # Adapter for capstone-rag ReAct agent
├── tests/                 # 74 pytest tests
├── sample_kb/             # Sample knowledge base for testing
├── config.yaml            # Default evaluation config
├── permissions.yaml       # Tool permission schema
└── adversarial_prompts.yaml
```

---

## Environment variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...   # Primary G-Eval judge
OPENAI_API_KEY=sk-...          # Secondary judge (inter-judge variance)

# Optional — cloud deployment only
MODAL_TOKEN_ID=ak-...
MODAL_TOKEN_SECRET=as-...
```

---

## License

MIT
