"""
Mode A integration test — verifies AgentScopeCallbackHandler works correctly
when injected into LangChain/LangGraph agents.

Failure modes guarded against:
- Handler not recognised by LangChain (isinstance check)
- Events silently dropped (count assertions)
- Tool name captured as empty or 'unknown_tool' (name assertion)
- latency_ms not populated on llm_end (timing assertion)
- prompt_tokens not estimated on llm_start (token assertion)
- Full AgentRunner normalisation path producing correct TraceEvent types
"""

import os
import pytest
from uuid import uuid4

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent  # noqa: deprecated in LangGraph v2

from agentscope.tracer import AgentScopeCallbackHandler
from agentscope.runner import AgentRunner, TraceEvent


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _make_agent():
    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    llm = ChatAnthropic(model="claude-haiku-4-5-20251001")
    return create_react_agent(llm, [multiply]), multiply


def _run_agent_with_handler(query: str) -> AgentScopeCallbackHandler:
    agent, _ = _make_agent()
    handler = AgentScopeCallbackHandler()
    agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"callbacks": [handler]},
    )
    return handler


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #

def test_handler_is_recognised_by_langchain():
    """LangChain must see handler as a valid BaseCallbackHandler subclass."""
    handler = AgentScopeCallbackHandler()
    assert isinstance(handler, BaseCallbackHandler), (
        "AgentScopeCallbackHandler is not a BaseCallbackHandler subclass — "
        "LangChain will silently ignore it as a callback."
    )


def test_llm_events_captured_on_plain_llm_call():
    """on_llm_start and on_llm_end must fire on a bare LLM invocation."""
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001")
    handler = AgentScopeCallbackHandler()
    llm.invoke("Say hello only.", config={"callbacks": [handler]})

    types = [e["type"] for e in handler.traces]
    assert "llm_start" in types, f"on_llm_start never fired. Got: {types}"
    assert "llm_end"   in types, f"on_llm_end never fired. Got: {types}"


def test_tool_events_captured_on_langgraph_agent():
    """on_tool_start and on_tool_end must fire when the agent calls a tool."""
    handler = _run_agent_with_handler("What is 6 multiplied by 7?")
    types = [e["type"] for e in handler.traces]

    assert "tool_start" in types, f"on_tool_start never fired. Got: {types}"
    assert "tool_end"   in types, f"on_tool_end never fired. Got: {types}"


def test_tool_name_is_not_empty_or_unknown():
    """Tool name must be populated — 'unknown_tool' means serialized key lookup failed."""
    handler = _run_agent_with_handler("What is 9 multiplied by 3?")
    tool_starts = [e for e in handler.traces if e["type"] == "tool_start"]

    assert tool_starts, "No tool_start events captured at all."
    for evt in tool_starts:
        assert evt["tool"], "tool name is empty"
        assert evt["tool"] != "unknown_tool", (
            f"tool name fell back to 'unknown_tool' — "
            f"_get() key lookup failed on serialized dict: {evt}"
        )


def test_llm_end_latency_is_populated():
    """latency_ms on llm_end must be a positive float — timing logic must be running."""
    handler = _run_agent_with_handler("What is 4 multiplied by 5?")
    llm_ends = [e for e in handler.traces if e["type"] == "llm_end"]

    assert llm_ends, "No llm_end events captured."
    for evt in llm_ends:
        assert "latency_ms" in evt, "latency_ms key missing from llm_end"
        assert evt["latency_ms"] > 0, (
            f"latency_ms is {evt['latency_ms']} — perf_counter timing not working"
        )


def test_llm_start_prompt_tokens_estimated():
    """prompt_tokens on llm_start must be > 0 — word-count estimation must run."""
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001")
    handler = AgentScopeCallbackHandler()
    llm.invoke("Explain quantum entanglement in one sentence.", config={"callbacks": [handler]})

    llm_starts = [e for e in handler.traces if e["type"] == "llm_start"]
    assert llm_starts, "No llm_start events captured."
    assert llm_starts[0]["prompt_tokens"] > 0, (
        "prompt_tokens is 0 — word-count estimation not running in on_llm_start"
    )


def test_no_events_silently_dropped_across_multi_step_run():
    """
    A multi-step agent (LLM → tool → LLM) must produce at least:
    2x llm_start, 2x llm_end, 1x tool_start, 1x tool_end.
    """
    handler = _run_agent_with_handler("What is 11 multiplied by 12?")
    types = [e["type"] for e in handler.traces]

    assert types.count("llm_start") >= 2, f"Expected >=2 llm_start, got {types.count('llm_start')}"
    assert types.count("llm_end")   >= 2, f"Expected >=2 llm_end,   got {types.count('llm_end')}"
    assert types.count("tool_start") >= 1, f"Expected >=1 tool_start, got {types.count('tool_start')}"
    assert types.count("tool_end")   >= 1, f"Expected >=1 tool_end,   got {types.count('tool_end')}"


def test_agentrunner_normalises_trace_events():
    """
    AgentRunner must produce a list of TraceEvent dataclasses with correct
    event_type fields after normalising raw callback output.
    Uses a LangGraph agent folder so Mode A path is exercised end-to-end.
    """
    import sys, os
    # Create a minimal LangGraph agent in a temp folder
    agent_dir = "/tmp/test_langchain_agent"
    os.makedirs(agent_dir, exist_ok=True)
    with open(f"{agent_dir}/main.py", "w") as f:
        f.write(
            "from langchain_anthropic import ChatAnthropic\n"
            "from langchain_core.tools import tool\n"
            "from langchain_core.messages import HumanMessage\n"
            "from langgraph.prebuilt import create_react_agent  # noqa: deprecated in LangGraph v2\n\n"
            "@tool\n"
            "def add(a: int, b: int) -> int:\n"
            "    'Add two numbers.'\n"
            "    return a + b\n\n"
            "llm = ChatAnthropic(model='claude-haiku-4-5-20251001')\n"
            "_agent = create_react_agent(llm, [add])\n\n"
            "def run(query: str, callbacks=None) -> str:\n"
            "    cfg = {'callbacks': callbacks} if callbacks else {}\n"
            "    result = _agent.invoke({'messages': [HumanMessage(content=query)]}, config=cfg)\n"
            "    return result['messages'][-1].content\n"
        )

    runner = AgentRunner(agent_dir, agent_model="claude-haiku-4-5-20251001")
    trace = runner.run("What is 3 plus 4?", run_id="mode-a-test")

    assert trace.agent_output, "agent_output is empty"
    assert isinstance(trace.total_latency_ms, float)

    event_types = [e.event_type for e in trace.events]
    assert "llm_start" in event_types, f"No llm_start in normalised events: {event_types}"
    assert "llm_end"   in event_types, f"No llm_end in normalised events: {event_types}"
    for evt in trace.events:
        assert isinstance(evt, TraceEvent), f"Expected TraceEvent, got {type(evt)}"
