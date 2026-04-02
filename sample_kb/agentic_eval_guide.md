# Agentic AI Evaluation — Reference Guide

## What is an Agentic AI System?

An agentic AI system is an AI that takes sequences of actions to complete a goal, rather than responding in a single step. It typically has access to tools — such as search, code execution, or database queries — and uses a language model to decide which tools to call and in what order.

Agentic systems differ from standard chatbots in that they maintain state across multiple steps, make autonomous decisions about tool use, and may interact with external services or data sources during a single task.

## Types of Agentic Systems

**RAG Pipelines (Retrieval-Augmented Generation)**
A RAG pipeline retrieves relevant documents from a knowledge base before generating a response. The agent uses a retriever tool to fetch context and then grounds its answer in that context. RAG systems are evaluated on both retrieval quality and response faithfulness.

**Tool-Use Agents**
A tool-use agent has access to a set of callable tools and autonomously selects which tool to invoke based on the user's query. Common tools include calculators, search engines, code interpreters, and APIs. These agents are evaluated on tool selection accuracy and argument correctness.

**Multi-Agent Systems**
A multi-agent system consists of multiple specialized agents that collaborate to complete a task. One agent may act as an orchestrator, delegating subtasks to other agents. Evaluation includes handoff correctness — whether context is accurately passed between agents.

**Hybrid Systems**
Hybrid systems combine retrieval with tool use or multi-agent coordination. They are the most complex to evaluate because failures can originate in any component.

## Key Evaluation Dimensions

### Retrieval Quality (IR Metrics)
Retrieval quality measures how well the system finds relevant documents. Key metrics include:
- **Precision@k**: The fraction of retrieved documents in the top-k that are relevant.
- **Recall@k**: The fraction of all relevant documents that appear in the top-k results.
- **MRR (Mean Reciprocal Rank)**: The average of the reciprocal rank of the first relevant document.
- **nDCG (Normalized Discounted Cumulative Gain)**: Measures ranking quality, giving more credit for relevant documents appearing earlier.
- **Hit Rate@k**: The fraction of queries where at least one relevant document appears in the top-k.

### Agent Behavior (Trajectory Metrics)
Trajectory metrics evaluate the sequence of actions the agent takes:
- **Tool Selection Accuracy**: Whether the agent called the correct tools for the task.
- **Argument Correctness**: Whether the parameters passed to each tool call were valid and relevant.
- **Plan Success**: Whether the overall sequence of tool calls was logical and led toward task completion.
- **Step Budget Efficiency**: How efficiently the agent completed the task relative to a step budget.
- **Convergence**: Whether the agent completed the task within the allowed number of steps.

### Response Quality (G-Eval Metrics)
Response quality is assessed using an LLM-as-judge approach:
- **Task Completion**: Did the agent fully complete what the user asked?
- **Faithfulness**: Are all claims in the response supported by retrieved context?
- **Hallucination Rate**: Does the response introduce facts not present in the context?
- **Citation Accuracy**: Are citations traceable to specific retrieved documents?
- **Helpfulness**: Is the response actionable and specific?
- **Safety**: Does the response avoid harmful or dangerous content?

### Cost and Efficiency
- **Cost per Query**: The total LLM token cost for processing a single user query.
- **Cost per Successful Task**: Cost normalized by the task success rate.
- **p50/p95 Latency**: Median and 95th-percentile response times.
- **Quality-Cost Index**: Response quality divided by cost — a composite efficiency metric.

### Adversarial Robustness
Adversarial evaluation tests whether the agent resists hostile inputs:
- **Prompt Injection Resistance**: Whether the agent resists attempts to override its instructions.
- **Unsafe Compliance Rate**: Whether the agent complies with requests to perform unsafe actions.
- **Permission Violation Rate**: Whether the agent calls tools it should not have access to.

## Common Failure Modes in Agentic Systems

**Ghost Action**: The agent claims to have completed an action (e.g., "I sent the email") without actually calling the tool. Detectable only through trace-level evaluation.

**Interrogation Loop**: In multi-turn systems, the agent repeatedly asks the user for information they already provided in an earlier turn. Indicates poor knowledge retention.

**Confident Fabricator**: The agent produces fluent, confident responses that are not grounded in retrieved context. High fluency can mask hallucination.

**Tool Overuse**: The agent calls more tools than necessary to complete a task, increasing latency and cost without improving quality.

**Argument Drift**: The agent calls the right tool but passes incorrect or irrelevant parameters, causing silent failures downstream.

## Evaluation Best Practices

Ground truth is essential for reliable evaluation. Without ground truth query-answer pairs, synthetic generation using an LLM can approximate it, but synthetic datasets may not cover edge cases.

Trace-level evaluation catches failures that output-only evaluation misses. An agent that produces a correct final answer may still have taken an inefficient or unsafe path to get there.

LLM-as-judge scoring (G-Eval) provides nuanced quality assessment but introduces judge variance. Running the same test cases through a secondary judge model and measuring score agreement helps quantify how reliable the scores are.

Cost budgets prevent runaway evaluation spend. Setting a hard limit on the number of judge calls and total evaluation cost ensures the framework itself does not become expensive to operate.

## What is G-Eval?

G-Eval is an LLM-as-judge evaluation framework that uses a language model to score responses against natural-language criteria. Instead of relying on exact string matching or predefined rubrics, G-Eval allows flexible, criteria-driven evaluation that adapts to the specific requirements of each task.

G-Eval works by presenting the judge model with the evaluation criteria, the input, and the actual output, then asking it to produce a score between 0 and 1. The criteria can capture nuanced quality dimensions like faithfulness, coherence, and task completion that are difficult to measure with traditional metrics.

## Inter-Judge Variance and Calibration Drift

Different judge models may score the same response differently. Inter-judge variance measures the agreement between a primary judge (e.g., Claude Sonnet) and a secondary judge (e.g., GPT-4o-mini). High variance indicates that the score is sensitive to the choice of judge.

Calibration drift measures whether judge scores shift over time relative to a baseline run. KL divergence between the current and baseline score distributions quantifies how much the judge's calibration has changed.
