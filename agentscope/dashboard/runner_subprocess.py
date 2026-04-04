"""
Standalone evaluation runner — called as a subprocess by the dashboard.
Writes progress updates and final results to a JSON file.
Completely isolated from Gradio's asyncio event loop.
"""
import sys
import os
import uuid
import json
import time

AGENTSCOPE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AGENTSCOPE_ROOT)

# Load env files passed as args
for env_file in sys.argv[2:]:
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    k = k.strip()
                    if k.replace("_", "").isalnum():
                        os.environ.setdefault(k, v.strip())

output_file = sys.argv[1]

def write_progress(stage: str, pct: float, data: dict = None):
    payload = {"stage": stage, "pct": pct, "ts": time.time()}
    if data:
        payload.update(data)
    with open(output_file, "w") as f:
        json.dump(payload, f)


def main():
    write_progress("starting", 0.02)

    from agentscope.pipeline_runner import execute_pipeline

    # State is passed via env var as JSON
    state = json.loads(os.environ["AGENTSCOPE_STATE"])
    execute_pipeline(state, write_progress)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        write_progress("error", -1.0, {"error": str(e), "traceback": traceback.format_exc()})
        sys.exit(1)
