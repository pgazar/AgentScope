import time
import functools
from typing import Any


def _get(d: dict, *keys, default=None):
    """Try multiple key variants in order, return first match."""
    for k in keys:
        if k in d:
            return d[k]
    return default


class AgentScopeCallbackHandler:
    """Mode A: zero-code integration for LangChain/LangGraph agents."""

    def __init__(self):
        self.traces: list[dict] = []
        self._step_start: dict[str, float] = {}

    def on_tool_start(self, serialized: dict, input_str: Any, **kwargs):
        tool_name = _get(serialized, "name", "id", "tool_name", default="unknown_tool")
        self._step_start[tool_name] = time.perf_counter()
        self.traces.append({
            "type": "tool_start",
            "tool": tool_name,
            "input": input_str,
            "ts": time.time(),
        })

    def on_tool_end(self, output: Any, **kwargs):
        self.traces.append({
            "type": "tool_end",
            "output": str(output),
        })

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs):
        model = _get(serialized, "name", "model_name", "model", default="unknown")
        self._step_start["llm"] = time.perf_counter()
        self.traces.append({
            "type": "llm_start",
            "model": model,
            # Word count is a rough proxy when token counts aren't available pre-call
            "prompt_tokens": sum(len(p.split()) for p in prompts),
        })

    def on_llm_end(self, response: Any, **kwargs):
        lat_ms = (time.perf_counter() - self._step_start.pop("llm", time.perf_counter())) * 1000
        raw = response.__dict__ if hasattr(response, "__dict__") else {}
        usage = _get(raw, "llm_output", "usage_metadata", default={}) or {}
        token_usage = _get(usage, "token_usage", "usage", default={}) or {}
        self.traces.append({
            "type": "llm_end",
            "completion_tokens": _get(token_usage, "completion_tokens", "output_tokens", default=0),
            "latency_ms": round(lat_ms, 2),
        })


def trace(func):
    """Mode B: decorator for custom agents without LangChain."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        handler = AgentScopeCallbackHandler()
        result = func(*args, **kwargs, _as_handler=handler)
        return result, handler.traces
    return wrapper
