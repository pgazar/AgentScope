import time
import importlib
import inspect
import sys
from dataclasses import dataclass, field

from agentscope.tracer import AgentScopeCallbackHandler


@dataclass
class TraceEvent:
    event_type: str  # llm_start | llm_end | tool_start | tool_end | retrieval | handoff
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    tool_output: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retrieval_docs: list = field(default_factory=list)
    latency_ms: float = 0.0
    agent_model: str = ""  # model used by the TARGET agent, not the judge
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentTrace:
    run_id: str
    agent_input: str
    agent_output: str
    events: list[TraceEvent] = field(default_factory=list)
    total_latency_ms: float = 0.0


class AgentRunner:
    """
    Loads the target agent from the provided folder, runs it on
    evaluation inputs, and returns a normalized AgentTrace.

    Integration Mode A (LangChain/LangGraph):
        Injects AgentScopeCallbackHandler into the agent's callback list.
        Zero source code changes required on the target agent.

    Integration Mode B (custom Python agent):
        Wraps the agent's main function with @agentscope.trace decorator.
        Requires one import added to the target agent's entry point.
    """

    def __init__(self, agent_folder: str, agent_model: str = "unknown"):
        self.agent_folder = agent_folder
        self.agent_model = agent_model  # model name of the EVALUATED agent
        self._agent_module = None  # set by _load_agent
        self._agent_fn = self._load_agent(agent_folder)

    def _load_agent(self, folder: str):
        """
        Attempts to import the agent's main callable from the folder.
        Looks for run, invoke, agent, or chat in __init__.py or main.py.
        """
        sys.path.insert(0, folder)
        for module_name in ["main", "__init__", "agent", "app"]:
            try:
                mod = importlib.import_module(module_name)
                for fn_name in ["run", "invoke", "agent", "chat"]:
                    fn = getattr(mod, fn_name, None)
                    if callable(fn):
                        self._agent_module = mod  # store for inject_trace_events
                        return fn
            except ImportError:
                continue
        raise RuntimeError(
            f"Could not load agent from {folder}. "
            "Ensure the folder contains main.py or __init__.py "
            "with a callable named 'run', 'invoke', or 'agent'."
        )

    def run(self, agent_input: str, run_id: str) -> AgentTrace:
        handler = AgentScopeCallbackHandler()
        start = time.perf_counter()
        try:
            sig = inspect.signature(self._agent_fn)
            if "callbacks" in sig.parameters:
                output = self._agent_fn(agent_input, callbacks=[handler])
            else:
                output = self._agent_fn(agent_input)
        except Exception as e:
            output = f"[AgentRunner error: {e}]"

        total_ms = (time.perf_counter() - start) * 1000
        events = self._normalize(handler.traces)
        trace = AgentTrace(
            run_id=run_id,
            agent_input=agent_input,
            agent_output=str(output),
            events=events,
            total_latency_ms=round(total_ms, 2),
        )
        # If the agent module exposes inject_trace_events(), call it to backfill
        # events that bypass the LangChain callback system (e.g. capstone-rag)
        inject_fn = getattr(self._agent_module, "inject_trace_events", None)
        if callable(inject_fn):
            inject_fn(trace)
        return trace

    def _normalize(self, raw: list[dict]) -> list[TraceEvent]:
        normalized: list[TraceEvent] = []
        pending_start: dict[str, float] = {}

        for evt in raw:
            t = evt.get("type")

            if t == "llm_start":
                pending_start["llm"] = time.time()
                normalized.append(TraceEvent(
                    event_type="llm_start",
                    prompt_tokens=evt.get("prompt_tokens", 0),
                    agent_model=self.agent_model,
                ))

            elif t == "llm_end":
                lat = (time.time() - pending_start.pop("llm", time.time())) * 1000
                normalized.append(TraceEvent(
                    event_type="llm_end",
                    completion_tokens=evt.get("completion_tokens", 0),
                    latency_ms=round(lat, 2),
                    agent_model=self.agent_model,
                ))

            elif t == "tool_start":
                tool = evt.get("tool", "")
                pending_start[tool] = time.time()
                normalized.append(TraceEvent(
                    event_type="tool_start",
                    tool_name=tool,
                    tool_args=evt.get("input", {}),
                ))

            elif t == "tool_end":
                # Match back to the most recent tool_start name
                tool = normalized[-1].tool_name if normalized else ""
                lat = (time.time() - pending_start.pop(tool, time.time())) * 1000
                normalized.append(TraceEvent(
                    event_type="tool_end",
                    tool_output=str(evt.get("output", "")),
                    latency_ms=round(lat, 2),
                ))

            elif t == "retrieval":
                normalized.append(TraceEvent(
                    event_type="retrieval",
                    retrieval_docs=evt.get("docs", []),
                    latency_ms=evt.get("latency_ms", 0.0),
                ))

            elif t == "handoff":
                normalized.append(TraceEvent(
                    event_type="handoff",
                    tool_name=evt.get("to_agent", ""),
                    tool_args={"context": evt.get("context", {})},
                ))

        return normalized
