"""
AgentScope adapter for capstone-rag.

Exposes a standard run(query) -> str callable that AgentScope's AgentRunner
can load. Also patches AgentRunner post-run to inject proper trace events
(retrieval docs for IR metrics, latency for cost metrics).
"""
import sys
import os
from dotenv import load_dotenv

# Load capstone-rag's own .env so DB_PORT, ANTHROPIC_MODEL etc. are set correctly
CAPSTONE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../capstone-rag")
)
load_dotenv(os.path.join(CAPSTONE_ROOT, ".env"))

if CAPSTONE_ROOT not in sys.path:
    sys.path.insert(0, CAPSTONE_ROOT)

from src.agent.react_agent import run_agent  # noqa: E402

DEFAULT_CONFIG = {
    "retrieval_mode": os.getenv("RETRIEVAL_MODE", "hybrid"),
    "prompting":      "zero_shot",
    "model":          os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
    "temperature":    0,
    "max_steps":      int(os.getenv("AGENT_MAX_STEPS", 10)),
    "timeout_seconds": int(os.getenv("AGENT_TIMEOUT_SECONDS", 60)),
    "guarded_mode":   False,
}

# Store last result so AgentRunner can inject events after calling run()
_last_result: dict = {}


def run(query: str) -> str:
    """
    Standard AgentScope entry point.
    Stores the full capstone-rag result so inject_trace_events() can
    backfill retrieval docs and latency into the AgentTrace.
    """
    global _last_result
    result = run_agent(query, config=DEFAULT_CONFIG)
    _last_result = result
    return result.get("final_answer", "[no answer]")


def inject_trace_events(trace) -> None:
    """
    Called after AgentRunner.run() to backfill events that capstone-rag
    produces but the LangChain callback system never sees:
      - retrieval events (doc keys) → enables IR metrics
      - llm_end latency             → enables cost/latency metrics
      - token estimate              → enables cost metrics
    """
    from agentscope.runner import TraceEvent

    if not _last_result:
        return

    new_events = []
    steps = _last_result.get("steps", [])

    for step in steps:
        action = step.get("action", "")
        if action in ("final_answer", "guardrail_block"):
            continue

        new_events.append(TraceEvent(
            event_type="tool_start",
            tool_name=action,
            tool_args=step.get("args", {}),
        ))

        if action == "rag_retrieve":
            # Citations are the retrieved doc keys — needed for IR metrics
            new_events.append(TraceEvent(
                event_type="retrieval",
                retrieval_docs=step.get("citations", []),
            ))

        new_events.append(TraceEvent(
            event_type="tool_end",
            tool_output=step.get("observation", ""),
        ))

    # Estimate tokens from message count (capstone-rag doesn't expose exact counts)
    # ~800 tokens input + ~300 tokens output per step is a conservative estimate
    n_steps = max(len(steps), 1)
    latency_ms = _last_result.get("total_latency_ms", 0.0)

    new_events.insert(0, TraceEvent(
        event_type="llm_start",
        prompt_tokens=800 * n_steps,
    ))
    new_events.append(TraceEvent(
        event_type="llm_end",
        completion_tokens=300 * n_steps,
        latency_ms=latency_ms,
    ))

    trace.events = new_events
    trace.total_latency_ms = latency_ms
