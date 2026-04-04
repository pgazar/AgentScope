from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


class HeadlessGateError(RuntimeError):
    pass


def _request_json(method: str, url: str, payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise HeadlessGateError(f"{method} {url} failed: {e.code} {body}") from e
    except urllib.error.URLError as e:
        raise HeadlessGateError(f"{method} {url} failed: {e}") from e


def _parse_metric_specs(items: list[str], mode: str) -> list[tuple[str, float, str]]:
    specs: list[tuple[str, float, str]] = []
    for item in items:
        path, sep, raw = item.partition("=")
        if not sep:
            raise HeadlessGateError(f"invalid metric threshold '{item}'; expected path=value")
        try:
            threshold = float(raw)
        except ValueError as e:
            raise HeadlessGateError(f"invalid numeric threshold in '{item}'") from e
        specs.append((path, threshold, mode))
    return specs


def _resolve_path(report: dict, path: str):
    current = report
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def evaluate_report_thresholds(report: dict, specs: list[tuple[str, float, str]]) -> list[str]:
    failures: list[str] = []
    for path, threshold, mode in specs:
        value = _resolve_path(report, path)
        if value is None:
            failures.append(f"{path}: missing or unscoreable")
            continue
        if not isinstance(value, (int, float)):
            failures.append(f"{path}: non-numeric value {value!r}")
            continue
        if mode == "min" and value < threshold:
            failures.append(f"{path}: {value:.4f} < minimum {threshold:.4f}")
        elif mode == "max" and value > threshold:
            failures.append(f"{path}: {value:.4f} > maximum {threshold:.4f}")
    return failures


def run_headless_gate(args: argparse.Namespace) -> tuple[dict, list[str]]:
    base_url = args.base_url.rstrip("/")
    payload = {
        "agent_folder": args.agent_folder,
        "agent_type": args.agent_type,
        "turn_type": args.turn_type,
        "agent_model": args.agent_model,
        "kb_path": args.kb_path,
        "gt_path": args.gt_path,
        "eval_inputs": args.eval_input,
    }
    result = _request_json("POST", f"{base_url}/evaluate", payload)
    run_id = result["run_id"]

    deadline = time.time() + args.timeout_s
    status = result.get("status", "queued")
    last_stage = status

    while time.time() < deadline:
        state = _request_json("GET", f"{base_url}/runs/{run_id}")
        status = state.get("status", "queued")
        last_stage = state.get("stage", status)
        if status == "completed":
            report = _request_json("GET", f"{base_url}/runs/{run_id}/report")
            break
        if status == "failed":
            error = state.get("error", {})
            raise HeadlessGateError(
                f"run {run_id} failed at stage {last_stage}: "
                f"{error.get('type', 'error')} {error.get('message', '')}".strip()
            )
        time.sleep(args.poll_interval_s)
    else:
        raise HeadlessGateError(f"run {run_id} timed out after {args.timeout_s}s at stage {last_stage}")

    specs = _parse_metric_specs(args.min_metric, "min") + _parse_metric_specs(args.max_metric, "max")
    failures = evaluate_report_thresholds(report, specs)
    return report, failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AgentScope in headless API mode and enforce metric gates.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--agent-folder", required=True)
    parser.add_argument("--agent-type", default="tool_use")
    parser.add_argument("--turn-type", default="single")
    parser.add_argument("--agent-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--kb-path")
    parser.add_argument("--gt-path")
    parser.add_argument("--eval-input", action="append", required=True, help="Repeat for each evaluation input.")
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--poll-interval-s", type=float, default=1.0)
    parser.add_argument("--report-out", default="outputs/ci_headless_run_report.json")
    parser.add_argument("--min-metric", action="append", default=[], help="Threshold of the form path=value")
    parser.add_argument("--max-metric", action="append", default=[], help="Threshold of the form path=value")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report, failures = run_headless_gate(args)

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2))

    print(f"Headless evaluation completed. Report written to {report_out}")

    if failures:
        print("Metric gate failures:", file=sys.stderr)
        for failure in failures:
            print(f" - {failure}", file=sys.stderr)
        return 1

    print("All configured metric gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
