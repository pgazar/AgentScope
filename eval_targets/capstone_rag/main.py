"""
AgentScope adapter for capstone-rag.

Exposes a standard run(query) -> str callable that AgentScope's AgentRunner
can load, and a convert_trace() helper that maps capstone-rag's step dict
format into AgentScope TraceEvent objects so trajectory metrics work correctly.
"""
import sys
import os
from dotenv import load_dotenv

# Load capstone-rag's own .env so DB_PORT, ANTHROPIC_MODEL etc. are set correctly
CAPSTONE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../capstone-rag")
)
load_dotenv(os.path.join(CAPSTONE_ROOT, ".env"))

# Make capstone-rag importable
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


def run(query: str) -> str:
    """
    Standard AgentScope entry point.
    Runs the capstone-rag ReAct agent and returns the final answer string.
    """
    result = run_agent(query, config=DEFAULT_CONFIG)
    return result.get("final_answer", "[no answer]")


def run_with_trace(query: str) -> dict:
    """Returns full capstone-rag result dict for trace conversion."""
    return run_agent(query, config=DEFAULT_CONFIG)


def convert_trace(capstone_result: dict) -> list:
    """
    Converts a capstone-rag result dict into AgentScope TraceEvent-compatible
    dicts for AgentRunner._normalize().
    """
    events = []

    for step in capstone_result.get("steps", []):
        action = step.get("action", "")
        if action in ("final_answer", "guardrail_block"):
            continue

        events.append({
            "type":  "tool_start",
            "tool":  action,
            "input": step.get("args", {}),
        })

        # Emit retrieval event so IR metrics can score retrieved doc keys
        if action == "rag_retrieve":
            events.append({
                "type":       "retrieval",
                "docs":       step.get("citations", []),
                "latency_ms": 0.0,
            })

        events.append({
            "type":   "tool_end",
            "output": step.get("observation", ""),
        })

    # Wrap one llm_start/llm_end for total run latency
    latency_ms = capstone_result.get("total_latency_ms", 0.0)
    events.insert(0, {"type": "llm_start", "prompt_tokens": 0})
    events.append({"type": "llm_end", "completion_tokens": 0, "latency_ms": latency_ms})

    return events
