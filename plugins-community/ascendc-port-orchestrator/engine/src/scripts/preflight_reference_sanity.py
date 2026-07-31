# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Phase O2.6 — reference sanity preflight.

Runs benchmark's Model.forward() alone (no kernel involved) on every case in the
benchmark .json, per-case-isolated so NPU stream poisoning on one case does NOT
contaminate the next. Answers: "how many of N cases does reference alone
complete without crash?"

Rationale (OL-87): benchmark data generators sometimes emit schema-invalid
inputs (e.g. int64 index tensors exceeding their bounding axis size). On NPU,
fancy-indexing OOB triggers sticky stream errors that cascade — making
per-case statistics unreliable without process isolation. If reference cannot
complete most cases, aog-kernel-worker's Phase D failure signatures will be
indistinguishable from real bugs vs harness-cascade artifacts, wasting iters.

Usage (runs inside npu_dev3 container):
    python3 preflight_reference_sanity.py <task_dir>
    # task_dir must contain: model.py, model.json (a copy of the benchmark .json)

Outputs:
    <task_dir>/preflight_reference.json
        {
            "n_cases": int,
            "ref_pass_count": int,
            "ref_crash_count": int,
            "per_case": [
                {"idx": int, "status": "PASS"|"CRASH",
                 "err": str (if crash, first 200 chars)}, ...
            ],
            "verdict": "CLEAN" | "WARN_EDGE_CASES" | "ABORT_HARNESS_BROKEN",
        }

Exit codes:
    0 = CLEAN (all cases pass). Proceed to Phase O3.
    1 = WARN (edge cases). Proceed but orchestrator should annotate.
    2 = ABORT (harness broken — too many refs fail). Orchestrator MUST NOT spawn
        aog-kernel-worker. Block the op.
"""
from __future__ import annotations
import logging

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile


# Thresholds (per OL-87 protocol)
WARN_RATIO = 0.90   # < 100% PASS but ≥ 90% → WARN, edge cases documented
ABORT_RATIO = 0.90  # < 90% PASS → ABORT


def _per_case_isolated_subprocess(task_dir: pathlib.Path, case_idx: int) -> dict:
    """Spawn a fresh Python subprocess per case. Ensures NPU stream poisoning
    from a prior case doesn't contaminate this one.
    """
    runner = f"""
import importlib.util, sys, traceback
import torch
import torch_npu  # noqa: F401

spec = importlib.util.spec_from_file_location('m', r'{task_dir}/model.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
groups = m.get_input_groups()
if {case_idx} >= len(groups):
    print('{{"status":"SKIP","err":"idx out of range"}}')
    sys.exit(0)
inputs = groups[{case_idx}]
device = torch.device('npu')
try:
    model = m.Model().to(device).eval()
    inp = [x.to(device) if hasattr(x, 'to') else x for x in inputs]
    with torch.no_grad():
        out = model(*inp)
    torch.npu.synchronize()
    print('{{"status":"PASS"}}')
except Exception as e:
    tb = traceback.format_exc()[-200:]
    import json as _j
    print(_j.dumps({{"status": "CRASH", "err": tb}}))
"""
    try:
        r = subprocess.run(
            [sys.executable, "-c", runner],
            cwd=str(task_dir), capture_output=True, text=True,
            timeout=120, env={**os.environ, "TORCH_NPU_STACKTRACE": "0"},
        )
        # Last non-empty line should be the JSON verdict
        out_lines = [l for l in r.stdout.splitlines() if l.strip()]
        if out_lines:
            try:
                return json.loads(out_lines[-1])
            except Exception as error:
                logging.getLogger(__name__).debug(
                    "Recoverable operation failed.", exc_info=error
                )
        return {"status": "CRASH", "err": r.stderr[-200:] or "no output"}
    except subprocess.TimeoutExpired:
        return {"status": "CRASH", "err": "subprocess timeout (120s)"}


def main():
    if len(sys.argv) != 2:
        print("usage: preflight_reference_sanity.py <task_dir>", file=sys.stderr)
        sys.exit(3)
    task_dir = pathlib.Path(sys.argv[1]).resolve()
    if not (task_dir / "model.py").exists():
        print(f"missing {task_dir}/model.py", file=sys.stderr)
        sys.exit(3)
    if not (task_dir / "model.json").exists():
        print(f"missing {task_dir}/model.json (copy benchmark .json here first)", file=sys.stderr)
        sys.exit(3)

    # Count cases by loading the module once (subprocess call is heavy, but
    # we need to isolate EXECUTION, not CASE COUNT).
    sys.path.insert(0, str(task_dir))
    spec = importlib.util.spec_from_file_location("m", task_dir / "model.py")
    m = importlib.util.module_from_spec(spec)
    # Avoid importing torch_npu here — we just want len(). model.py may call it.
    # Fall back to JSON line count if import fails.
    try:
        spec.loader.exec_module(m)
        n_cases = len(m.get_input_groups())
    except Exception:
        n_cases = sum(1 for _ in open(task_dir / "model.json") if _.strip())
    print(f"[preflight] n_cases = {n_cases}, running each in isolated subprocess...")

    results = []
    for i in range(n_cases):
        r = _per_case_isolated_subprocess(task_dir, i)
        r["idx"] = i
        results.append(r)
        print(f"  case {i:>3}/{n_cases}: {r.get('status')}"
              f" {('— ' + r['err'][:60]) if r.get('status') == 'CRASH' else ''}")

    n_pass = sum(1 for r in results if r.get("status") == "PASS")
    n_crash = sum(1 for r in results if r.get("status") == "CRASH")

    if n_pass == n_cases:
        verdict = "CLEAN"
        exit_code = 0
    elif n_pass / n_cases >= ABORT_RATIO:
        verdict = "WARN_EDGE_CASES"
        exit_code = 1
    else:
        verdict = "ABORT_HARNESS_BROKEN"
        exit_code = 2

    summary = {
        "n_cases": n_cases,
        "ref_pass_count": n_pass,
        "ref_crash_count": n_crash,
        "per_case": results,
        "verdict": verdict,
    }
    out_path = task_dir / "preflight_reference.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print()
    print("=== PREFLIGHT SUMMARY ===")
    print(f"  Reference PASS: {n_pass}/{n_cases}")
    print(f"  Reference CRASH: {n_crash}/{n_cases}")
    print(f"  Verdict: {verdict}")
    print(f"  Report: {out_path}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
