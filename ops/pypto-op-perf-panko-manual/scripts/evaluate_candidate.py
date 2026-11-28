# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Evaluate and record an acknowledged PANKO candidate."""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import jsonio
from evaluation_session import resolve_op_file
from session_worker import write_json


SCRIPTS = Path(__file__).resolve().parent
PIPE_METRICS = ("util", "bubble", "aic_util", "aiv_util", "pred_stall", "aic_bubble", "aiv_bubble")


class CommandError(ValueError):
    def __init__(self, script, output, code):
        try:
            value = json.loads(output)
        except ValueError:
            value = {}
        self.response = value if isinstance(value, dict) else {}
        super().__init__(output.strip() or f"{script} exited {code}")


def invoke(script, *args):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *map(str, args)],
        stdout=subprocess.PIPE, text=True, check=False,
    )
    if result.returncode:
        raise CommandError(script, result.stdout, result.returncode)
    return json.loads(result.stdout)


def controller(op_dir, command, *args):
    return invoke("harness_session.py", "--op-dir", op_dir, "--", command, "--op-dir", op_dir, *args)


def checked_result(result):
    verdict = result["s"]
    metrics = result["metrics"]
    if not isinstance(verdict, int) or isinstance(verdict, bool) or verdict not in (0, 1):
        raise ValueError("Invalid frozen evaluator result")
    if not isinstance(metrics, dict):
        raise ValueError("Invalid frozen evaluator metrics")
    if metrics.get("fault_kind"):
        raise ValueError(f"Evaluation environment fault: {metrics.get('reason')}")
    if verdict:
        latency = metrics.get("p")
        if not isinstance(latency, (int, float)) or isinstance(latency, bool):
            raise ValueError("Successful evaluation requires a numeric latency")
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError("Successful evaluation requires a positive finite latency")
    return verdict, metrics


def measure(args, state, receipt):
    source = Path(resolve_op_file(args.op_dir, state["op"], state.get("op_file")))
    if hashlib.sha256(source.read_bytes()).hexdigest() != args.code_hash:
        raise ValueError("Candidate source changed after acknowledgement")
    timeout = state["config"]["eval_timeout_s"]
    binding = dict(
        run_start=state["progress"]["_start_ts"], candidate=args.code_hash, device=args.device,
        command=args.test_command, timeout=timeout, lint_root=str(args.lint_root.resolve()),
    )
    if receipt.is_file():
        result = json.loads(receipt.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("Invalid saved evaluation receipt")
        if "binding" in result:
            if result["binding"] != binding:
                raise ValueError("Saved evaluation belongs to another evaluation configuration")
            return checked_result(result)
        # A raw measurement cannot establish which command or device produced it.
    result = invoke(
        "evaluation_session.py", "--mode", "subprocess", "--op-dir", args.op_dir,
        "--op", state["op"], "--op-file", source, "--device", args.device,
        "--lint-root", args.lint_root, "--test-command", args.test_command, "--eval-timeout-s", timeout,
    )
    verdict = checked_result(result)
    if hashlib.sha256(source.read_bytes()).hexdigest() != args.code_hash:
        raise ValueError("Candidate source changed during evaluation")
    result["binding"] = binding
    write_json(receipt, result)
    return verdict


def record_result(args, state, evidence):
    run_start = state["progress"]["_start_ts"]
    saved = {}
    cached = state["eval_cache"].get(args.code_hash)
    if cached is None:
        receipt = evidence / f"{run_start}-{args.code_hash}.json"
        verdict, metrics = measure(args, state, receipt)
        saved["receipt"] = str(receipt)
    else:
        # The controller's duplicate record releases this handoff and restores its incumbent.
        verdict = cached["s"]
        metrics = {**cached.get("m", {}), "p": cached.get("p")}
    arguments = ["--parent", args.parent, "--code-hash", args.code_hash, "--s", verdict, "--run-start", run_start]
    if verdict:
        arguments.extend(["--p", metrics["p"]])
        for name in PIPE_METRICS:
            if metrics.get(name) is not None:
                arguments.extend(["--" + name.replace("_", "-"), metrics[name]])
    try:
        record = controller(args.op_dir, "record", *arguments)
    except CommandError as error:
        if error.response.get("error") != "stop_required":
            raise
        stop = controller(args.op_dir, "stop")
        return dict(stop, recorded=False, **saved)
    return dict(s=verdict, metrics=metrics, record=record, **saved)


def settle(args):
    search = args.op_dir / "optimization" / "search_state.json"
    evidence = search.parent / "measurements"
    evidence.mkdir(exist_ok=True)
    # Serialize retries until both the device result and controller response are durable.
    with (evidence / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(search.read_text(encoding="utf-8"))
        device = state.get("device_id")
        if device not in (None, "") and str(device) != str(args.device):
            raise ValueError("Device differs from the search configuration")
        stop = controller(args.op_dir, "stop")
        if stop["stop"]:
            return stop
        pending = state["progress"].get("pending_code") or {}
        acknowledged = (pending.get("action_id"), pending.get("candidate_hash"))
        if not state["config"].get("checkpoint_handoffs") or acknowledged != (args.parent, args.code_hash):
            if args.code_hash in state["eval_cache"]:
                return controller(args.op_dir, "probe", "--code-hash", args.code_hash)
            raise ValueError("Candidate does not match the acknowledged controller action")
        return record_result(args, state, evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op-dir", type=Path, required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--code-hash", required=True)
    parser.add_argument("--device", default=os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    parser.add_argument("--test-command", required=True)
    parser.add_argument("--lint-root", type=Path, required=True)
    args = parser.parse_args()
    os.environ["TILE_FWK_DEVICE_ID"] = args.device
    try:
        jsonio.emit(settle(args))
    except (OSError, KeyError, TypeError, ValueError) as error:
        jsonio.emit(dict(error=str(error)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
