#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""通用 torch_npu.profiler 基准脚本 — 遵循 PR #103 skill (warmup=5, active=5)。

用法:
  benchmark_torch_npu_profiler.py \
      --op-name <op> \
      --cases <op>_perf_cases.jsonl \
      --candidate-fn 'candidate_fn:torch_npu.npu_apply_adam_w_v2' \
      --baseline-fn 'baseline_fn:torch_npu.npu_apply_adam_w_v2' \
      --trace-root /tmp/profiler_trace \
      --report-md <op>_torch_npu_profiler_report.md

两条路径必须**用同款** torch_npu.profiler + 同款 schedule (warmup=5,active=5)。
对于 port_a3_to_a5 场景:
- candidate = A5 端调用 (aclnn-direct 或 torch_npu)
- baseline = A3 端调用 (CANN aclnn 或 torch_npu)
- 两端通过 SSH 分别跑 + 收集 CSV → 同分母合并报告

本脚本只跑**一端** (本机 NPU), 由外层 driver 分别在 A5 / A3 host 上跑然后
合并报告。
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path

try:
    import torch
    import torch_npu  # noqa: F401
except ImportError as e:
    print(f"FATAL: torch / torch_npu not available: {e}", file=sys.stderr)
    sys.exit(2)


WARMUP = 5  # fixed per PR #103 skill — do NOT change
ACTIVE = 5
WAIT = 0


def load_cases(path: Path) -> list:
    out = []
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out


def build_inputs(case: dict, device: str) -> dict:
    """Convert a JSONL case spec → ready-to-call inputs dict."""
    tensors = {}
    attrs = {}
    for entry in case["inputs"]:
        name = entry["name"]
        t = entry["type"]
        if t == "tensor":
            dtype_str = entry["dtype"]
            shape = entry["shape"]
            dt = getattr(torch, dtype_str)
            tensors[name] = torch.randn(*shape, dtype=dt, device=device) \
                if dt.is_floating_point \
                else torch.randint(0, 100, shape, dtype=dt, device=device)
        elif t == "attr":
            attrs[name] = entry["value"]
    return {"tensors": tensors, "attrs": attrs}


def run_one_case(call_fn, inputs: dict, trace_dir: str, label: str) -> float:
    """Run one case through torch_npu.profiler with warmup=5/active=5.

    Returns: total kernel time (us) summed across the active steps,
    divided by active. (CSV column `Total Time(us)` sum / 5.)
    """
    def step():
        call_fn(**inputs["tensors"], **inputs["attrs"])

    schedule = torch_npu.profiler.schedule(
        wait=WAIT, warmup=WARMUP, active=ACTIVE, repeat=1, skip_first=0)
    handler = torch_npu.profiler.tensorboard_trace_handler(trace_dir)
    activities = [
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ]

    with torch_npu.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=handler,
    ) as prof:
        for _ in range(WAIT + WARMUP + ACTIVE):
            step()
            torch.npu.synchronize()
            prof.step()

    # Find latest *_ascend_pt/ASCEND_PROFILER_OUTPUT/op_statistic.csv
    pattern = os.path.join(trace_dir, "*_ascend_pt", "ASCEND_PROFILER_OUTPUT", "op_statistic.csv")
    csvs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not csvs:
        return float("nan")
    total_us = 0.0
    with open(csvs[0]) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            # CSV column "Total Time(us)" — handle BOM + variants
            for k in row:
                if "Total Time" in k:
                    try:
                        total_us += float(row[k])
                    except ValueError:
                        pass
                    break
    return total_us / ACTIVE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, help="JSONL cases file")
    ap.add_argument("--candidate-call", required=True,
                    help="Module:function, e.g. 'torch_npu:npu_apply_adam_w_v2'")
    ap.add_argument(
        "--baseline-call",
        required=True,
        help=(
            "Same format. For port_a3: candidate = A5-side, baseline = A3-side; "
            "both invoked here on local NPU."
        ),
    )
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--trace-root", default="/tmp/profiler_trace")
    ap.add_argument("--report-md", required=True)
    ap.add_argument("--op-name", required=True)
    args = ap.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.is_file():
        print(f"FATAL: cases file not found: {cases_path}", file=sys.stderr)
        sys.exit(2)
    cases = load_cases(cases_path)
    if not cases:
        print("FATAL: no cases", file=sys.stderr)
        sys.exit(2)

    def resolve(path: str):
        mod_name, fn_name = path.split(":", 1)
        mod = importlib.import_module(mod_name)
        return getattr(mod, fn_name)

    cand_fn = resolve(args.candidate_call)
    base_fn = resolve(args.baseline_call)

    trace_root = Path(args.trace_root)
    trace_root.mkdir(parents=True, exist_ok=True)
    results = []
    for i, case in enumerate(cases):
        case_id = case.get("case_id", i)
        # Build inputs ONCE; reuse for both paths
        inputs_c = build_inputs(case, args.device)
        inputs_b = build_inputs(case, args.device)  # fresh randn so no warmup carryover
        cand_dir = str(trace_root / f"case_{case_id:02d}" / "candidate")
        base_dir = str(trace_root / f"case_{case_id:02d}" / "baseline")
        # Clean per case
        for d in (cand_dir, base_dir):
            os.makedirs(d, exist_ok=True)
        cand_us = run_one_case(cand_fn, inputs_c, cand_dir, "candidate")
        base_us = run_one_case(base_fn, inputs_b, base_dir, "baseline")
        ratio = base_us / cand_us if cand_us > 0 else float("nan")
        # Extract shape + dtype from first tensor input
        first_tensor = next((e for e in case["inputs"] if e["type"] == "tensor"), None)
        shape = first_tensor["shape"] if first_tensor else []
        dtype = first_tensor["dtype"] if first_tensor else "?"
        results.append({
            "case_id": case_id, "shape": shape, "dtype": dtype,
            "candidate_us": cand_us, "baseline_us": base_us, "ratio": ratio,
        })
        print(f"case {case_id} shape={shape} dtype={dtype}: "
              f"cand={cand_us:.2f}us base={base_us:.2f}us ratio={ratio:.3f}")

    # Render report
    lines = [
        "# 性能评估结果",
        "",
        f"算子: `{args.op_name}`",
        f"方法: torch_npu.profiler (warmup={WARMUP}, active={ACTIVE}, wait={WAIT}, repeat=1)",
        f"采集时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 性能对比",
        "",
        "| Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比 |",
        "| ---- | ----- | ----- | ------------- | -------- | -------------- |",
    ]
    for r in results:
        lines.append(
            f"| {r['case_id']} | {r['shape']} | {r['dtype']} | "
            f"{r['candidate_us']:.2f} | {r['baseline_us']:.2f} | {r['ratio']:.3f} |"
        )
    valid_ratios = [r["ratio"] for r in results if not math.isnan(r["ratio"])]
    cand_better = sum(1 for x in valid_ratios if x > 1.0)
    base_better = sum(1 for x in valid_ratios if x < 1.0)
    avg = sum(valid_ratios) / len(valid_ratios) if valid_ratios else float("nan")
    lines += [
        "",
        "## 全量汇总",
        "",
        "| 指标 | 值 |",
        "| ---- | -- |",
        f"| 用例数 | {len(results)} |",
        f"| 平均 加速比 | {avg:.3f} |",
        f"| 自定义算子更优 (>1) | {cand_better} |",
        f"| 标杆更优 (<1) | {base_better} |",
    ]
    Path(args.report_md).write_text("\n".join(lines) + "\n")
    print(f"\nReport: {args.report_md}")


if __name__ == "__main__":
    main()
