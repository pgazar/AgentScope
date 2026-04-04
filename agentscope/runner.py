import time
import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field

from agentscope.otel import inject_trace_context, mark_span_error, mark_span_ok, set_span_attributes, start_span
from agentscope.tracer import AgentScopeCallbackHandler
from agentscope.trace_audit import summarize_trace


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
    diagnostics: dict = field(default_factory=dict)


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

    ENV_ALLOWLIST = {
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "PYTHONHOME",
        "PYTHONPATH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TMPDIR",
        "USER",
        "VIRTUAL_ENV",
    }
    ENV_PREFIXES = (
        "AGENTSCOPE_",
        "ANTHROPIC_",
        "AWS_",
        "AZURE_",
        "CHROMA_",
        "COHERE_",
        "DATABASE_",
        "DB_",
        "DEEPEVAL_",
        "ELASTIC_",
        "GOOGLE_",
        "HF_",
        "HUGGINGFACE_",
        "LANGCHAIN_",
        "MISTRAL_",
        "NO_PROXY",
        "OPENAI_",
        "OTEL_",
        "PG",
        "PINECONE_",
        "POSTGRES_",
        "QDRANT_",
        "REDIS_",
        "REQUESTS_CA_BUNDLE",
        "SERPAPI_",
        "TAVILY_",
        "VOYAGE_",
        "WEAVIATE_",
    )

    def __init__(
        self,
        agent_folder: str,
        agent_model: str = "unknown",
        mode: str | None = None,
        timeout_s: int | None = None,
    ):
        self.agent_folder = agent_folder
        self.agent_model = agent_model  # model name of the EVALUATED agent
        self._agent_module = None
        self._agent_fn = None
        self.mode = mode or os.environ.get("AGENTSCOPE_RUNNER_MODE", "subprocess")
        self.timeout_s = timeout_s or int(os.environ.get("AGENTSCOPE_AGENT_TIMEOUT_S", "90"))

    @classmethod
    def validate_agent_folder(cls, folder: str) -> dict:
        if not folder:
            return {"ok": False, "reason": "agent_folder is required"}
        if not os.path.isdir(folder):
            return {"ok": False, "reason": f"agent folder does not exist: {folder}"}

        for module_name in ["main", "__init__", "agent", "app"]:
            candidates = [
                os.path.join(folder, f"{module_name}.py"),
                os.path.join(folder, module_name, "__init__.py"),
            ]
            for file_path in candidates:
                if os.path.exists(file_path):
                    return {"ok": True, "entrypoint": file_path}

        return {
            "ok": False,
            "reason": (
                "expected one of main.py, __init__.py, agent.py, or app.py "
                "inside the agent folder"
            ),
        }

    def _ensure_loaded(self):
        if self._agent_fn is None:
            self._agent_fn = self._load_agent(self.agent_folder)

    def _load_agent(self, folder: str):
        """
        Loads the agent callable directly from a file path using
        importlib.util.spec_from_file_location.

        This bypasses sys.modules caching and sys.path lookup entirely,
        so evaluating multiple different agents in the same process
        (e.g. across Gradio sessions) never returns a stale cached module.

        Looks for run, invoke, agent, or chat in main.py / __init__.py.
        """
        # Ensure the agent's folder is on sys.path so its internal
        # relative imports (e.g. from src.tools import ...) resolve correctly
        if folder not in sys.path:
            sys.path.insert(0, folder)

        errors = []
        for module_name in ["main", "__init__", "agent", "app"]:
            candidates = [
                os.path.join(folder, f"{module_name}.py"),
                os.path.join(folder, module_name, "__init__.py"),
            ]
            for file_path in candidates:
                if not os.path.exists(file_path):
                    continue
                # Unique module name prevents sys.modules cache collisions
                # when the same module_name (e.g. "main") is loaded from
                # different agent folders across sessions.
                unique_name = f"_agentscope_agent_{module_name}_{abs(hash(folder))}"
                try:
                    spec = importlib.util.spec_from_file_location(unique_name, file_path)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    for fn_name in ["run", "invoke", "agent", "chat"]:
                        fn = getattr(mod, fn_name, None)
                        if callable(fn):
                            self._agent_module = mod
                            return fn
                except Exception as e:
                    errors.append(f"{file_path}: {type(e).__name__}: {e}")
                    continue

        raise RuntimeError(
            f"Could not load agent from {folder}. "
            "Ensure the folder contains main.py or __init__.py "
            "with a callable named 'run', 'invoke', or 'agent'.\n"
            + ("\nImport errors:\n" + "\n".join(errors) if errors else "")
        )

    def run(self, agent_input: str, run_id: str) -> AgentTrace:
        if self.mode == "subprocess":
            return self._run_subprocess(agent_input, run_id)
        return self._run_inprocess(agent_input, run_id)

    def _run_inprocess(self, agent_input: str, run_id: str) -> AgentTrace:
        self._ensure_loaded()
        handler = AgentScopeCallbackHandler()
        with start_span(
            "agentscope.target_agent.inproc",
            tracer_name="agentscope.runner",
            attributes={
                "agentscope.run_id": run_id,
                "agentscope.agent_folder": self.agent_folder,
                "agentscope.agent_model": self.agent_model,
            },
        ) as span:
            start = time.perf_counter()
            try:
                sig = inspect.signature(self._agent_fn)
                if "callbacks" in sig.parameters:
                    output = self._agent_fn(agent_input, callbacks=[handler])
                else:
                    output = self._agent_fn(agent_input)
                mark_span_ok(span)
            except Exception as e:
                mark_span_error(span, e)
                output = f"[AgentRunner error: {e}]"

        total_ms = (time.perf_counter() - start) * 1000
        events = self._normalize(handler.traces)
        trace = AgentTrace(
            run_id=run_id,
            agent_input=agent_input,
            agent_output=str(output),
            events=events,
            total_latency_ms=round(total_ms, 2),
            diagnostics={"runner_mode": "inproc"},
        )
        # If the agent module exposes inject_trace_events(), call it to backfill
        # events that bypass the LangChain callback system (e.g. capstone-rag)
        inject_fn = getattr(self._agent_module, "inject_trace_events", None)
        if callable(inject_fn):
            inject_fn(trace)
        trace.diagnostics.update(summarize_trace(trace))
        return trace

    def _run_subprocess(self, agent_input: str, run_id: str) -> AgentTrace:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_folder = os.path.abspath(self.agent_folder)
        payload = {
            "agent_folder": abs_folder,
            "agent_model": self.agent_model,
            "agent_input": agent_input,
            "run_id": run_id,
            "timeout_s": self.timeout_s,
            "otel_trace_context": inject_trace_context({"run_id": run_id}),
        }
        try:
            with start_span(
                "agentscope.target_agent.subprocess",
                tracer_name="agentscope.runner",
                attributes={
                    "agentscope.run_id": run_id,
                    "agentscope.agent_folder": abs_folder,
                    "agentscope.agent_model": self.agent_model,
                },
            ) as span:
                result = subprocess.run(
                    [sys.executable, "-m", "agentscope.runner_worker"],
                    input=json.dumps(payload),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    cwd=abs_folder,
                    env=self._build_subprocess_env(repo_root),
                )
                set_span_attributes(span, {"agentscope.subprocess.returncode": result.returncode})
                if result.returncode == 0:
                    mark_span_ok(span)
                else:
                    mark_span_error(span, RuntimeError(result.stderr.strip() or "target agent subprocess failed"))
        except subprocess.TimeoutExpired:
            return self._error_trace(
                run_id=run_id,
                agent_input=agent_input,
                message=f"timeout after {self.timeout_s}s",
            )
        except Exception as e:
            return self._error_trace(
                run_id=run_id,
                agent_input=agent_input,
                message=f"subprocess launch failed: {e}",
            )

        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip() or "unknown subprocess error"
            return self._error_trace(run_id=run_id, agent_input=agent_input, message=err[-400:])

        try:
            payload = json.loads(result.stdout.strip().splitlines()[-1])
            trace = deserialize_trace(payload)
        except Exception as e:
            return self._error_trace(
                run_id=run_id,
                agent_input=agent_input,
                message=f"invalid subprocess payload: {e}",
            )

        child_mode = trace.diagnostics.get("runner_mode")
        trace.diagnostics["runner_mode"] = "subprocess"
        if child_mode:
            trace.diagnostics["child_runner_mode"] = child_mode
        trace.diagnostics.update(summarize_trace(trace))
        return trace

    def _build_subprocess_env(self, repo_root: str) -> dict:
        env = {}
        passthrough = {
            key.strip()
            for key in os.environ.get("AGENTSCOPE_AGENT_ENV_PASSTHROUGH", "").split(",")
            if key.strip()
        }

        for key, value in os.environ.items():
            if (
                key in self.ENV_ALLOWLIST
                or key in passthrough
                or key.startswith(self.ENV_PREFIXES)
            ):
                env[key] = value

        py_paths = [repo_root]
        if env.get("PYTHONPATH"):
            py_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(py_paths)
        env["AGENTSCOPE_RUNNER_MODE"] = "inproc"
        return env

    def _error_trace(self, run_id: str, agent_input: str, message: str) -> AgentTrace:
        trace = AgentTrace(
            run_id=run_id,
            agent_input=agent_input,
            agent_output=f"[AgentRunner error: {message}]",
            events=[],
            total_latency_ms=0.0,
            diagnostics={"runner_mode": "subprocess"},
        )
        trace.diagnostics.update(summarize_trace(trace))
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


def serialize_trace_event(event: TraceEvent) -> dict:
    return {
        "event_type": event.event_type,
        "tool_name": event.tool_name,
        "tool_args": event.tool_args,
        "tool_output": event.tool_output,
        "prompt_tokens": event.prompt_tokens,
        "completion_tokens": event.completion_tokens,
        "retrieval_docs": event.retrieval_docs,
        "latency_ms": event.latency_ms,
        "agent_model": event.agent_model,
        "timestamp": event.timestamp,
    }


def serialize_trace(trace: AgentTrace) -> dict:
    return {
        "run_id": trace.run_id,
        "agent_input": trace.agent_input,
        "agent_output": trace.agent_output,
        "events": [serialize_trace_event(event) for event in trace.events],
        "total_latency_ms": trace.total_latency_ms,
        "diagnostics": trace.diagnostics,
    }


def deserialize_trace(payload: dict) -> AgentTrace:
    events = [TraceEvent(**event) for event in payload.get("events", [])]
    return AgentTrace(
        run_id=payload["run_id"],
        agent_input=payload["agent_input"],
        agent_output=payload["agent_output"],
        events=events,
        total_latency_ms=payload.get("total_latency_ms", 0.0),
        diagnostics=payload.get("diagnostics", {}),
    )
