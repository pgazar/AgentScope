from __future__ import annotations

import json
import sys

from agentscope.otel import configure_telemetry, extract_trace_context, mark_span_error, mark_span_ok, start_span
from agentscope.runner import AgentRunner, serialize_trace


def main() -> int:
    payload = json.loads(sys.stdin.read())
    configure_telemetry(service_name="agentscope-runner-worker")

    parent_context = extract_trace_context(payload.get("otel_trace_context"))
    with start_span(
        "agentscope.target_agent.worker",
        tracer_name="agentscope.runner_worker",
        context=parent_context,
        attributes={
            "agentscope.run_id": payload["run_id"],
            "agentscope.agent_folder": payload["agent_folder"],
            "agentscope.agent_model": payload.get("agent_model", "unknown"),
        },
    ) as span:
        runner = AgentRunner(
            agent_folder=payload["agent_folder"],
            agent_model=payload.get("agent_model", "unknown"),
            mode="inproc",
            timeout_s=payload.get("timeout_s"),
        )
        try:
            trace = runner.run(payload["agent_input"], payload["run_id"])
            mark_span_ok(span)
        except Exception as e:
            mark_span_error(span, e)
            raise

        json.dump(serialize_trace(trace), sys.stdout)
        sys.stdout.write("\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
