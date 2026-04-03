"""
AgentScope adapter for capstone-rag.

Exposes a standard run(query) -> str callable that AgentScope's AgentRunner
can load. Also patches AgentRunner post-run to inject proper trace events
(retrieval text for faithfulness scoring, latency for cost metrics).
"""
import sys
import os
from dotenv import load_dotenv

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

_last_result: dict = {}


def run(query: str) -> str:
    global _last_result
    result = run_agent(query, config=DEFAULT_CONFIG)
    _last_result = result
    return result.get("final_answer", "[no answer]")


def inject_trace_events(trace) -> None:
    """
    Backfills trace events from capstone-rag's step dict:
    - retrieval_docs = actual observation text (not just keys) → fixes faithfulness scoring
    - token estimates → enables cost metrics
    - latency → enables latency metrics
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
            # Use observation text (full retrieved content) not just citation keys
            # This gives G-Eval actual content to verify faithfulness against
            observation = step.get("observation", "")
            citations   = step.get("citations", [])

            # Pass both the text content and the doc keys
            retrieval_content = [observation] if observation else citations
            new_events.append(TraceEvent(
                event_type="retrieval",
                retrieval_docs=retrieval_content,
            ))

        new_events.append(TraceEvent(
            event_type="tool_end",
            tool_output=step.get("observation", ""),
        ))

    n_steps    = max(len(steps), 1)
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

    trace.events      = new_events
    trace.total_latency_ms = latency_ms
