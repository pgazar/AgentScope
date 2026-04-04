from __future__ import annotations

import json
import sys

from agentscope.runner import AgentRunner, serialize_trace


def main() -> int:
    payload = json.loads(sys.stdin.read())
    runner = AgentRunner(
        agent_folder=payload["agent_folder"],
        agent_model=payload.get("agent_model", "unknown"),
        mode="inproc",
        timeout_s=payload.get("timeout_s"),
    )
    trace = runner.run(payload["agent_input"], payload["run_id"])
    json.dump(serialize_trace(trace), sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
