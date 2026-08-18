#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Skill script: 单算子本地评估工具。

属 ops-knowledge-optimization-ingest skill。转调 _vendor/eval_client 和 _vendor/evaluator，
**无需**仓库根目录在 sys.path——可随 skill 文件夹整体移植。

在本机真实 NPU 上评估：先用算子工程自带的 build 步骤编译，再用 msprof 包裹 run 步骤采集，
从产物解析 score(= task_duration_us) 与 performance_metrics。

外部依赖：真实 Ascend NPU + CANN + `msprof` / `npu-smi`（先 source CANN set_env.sh）。
不再依赖任何远程评估服务。
"""
import argparse
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

eval_client = importlib.import_module("_vendor.eval_client")
Evaluator = importlib.import_module("_vendor.evaluator").KernelEvaluator

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

# Dedicated stdout channel for the final result JSON (the script's CLI contract),
# kept separate from the stderr diagnostic `logger` above.
_stdout_logger = logging.getLogger(__name__ + ".stdout")
_stdout_logger.setLevel(logging.INFO)
_stdout_logger.propagate = False
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setFormatter(logging.Formatter("%(message)s"))
_stdout_logger.addHandler(_stdout_handler)


def _path_parts(path: str) -> Tuple[str, ...]:
    return tuple(part for part in os.path.abspath(path).split(os.path.sep) if part)


def infer_op_identity(
    op_host_path: str,
    op_kernel_path: str,
    op_name: Optional[str] = None,
    op_category: Optional[str] = None,
) -> Tuple[str, str]:
    if op_name and op_category:
        return op_category, op_name

    host_parts = _path_parts(op_host_path)
    kernel_parts = _path_parts(op_kernel_path)

    if "op_host" in host_parts and "op_kernel" in kernel_parts:
        host_idx = host_parts.index("op_host")
        kernel_idx = kernel_parts.index("op_kernel")
        if host_idx >= 2 and kernel_idx >= 2:
            inferred_name = op_name or host_parts[host_idx - 1]
            inferred_category = op_category or host_parts[host_idx - 2]
            kernel_name = kernel_parts[kernel_idx - 1]
            kernel_category = kernel_parts[kernel_idx - 2]
            if inferred_name == kernel_name and inferred_category == kernel_category:
                return inferred_category, inferred_name

    if op_name and op_category:
        return op_category, op_name
    if op_name or op_category:
        missing = "op_category" if not op_category else "op_name"
        raise ValueError(f"failed to infer operator identity, please provide --{missing}")
    raise ValueError(
        "failed to infer operator identity from paths; please provide both --op_category and --op_name"
    )


def resolve_op_dir(args: argparse.Namespace) -> str:
    """算子工作副本根：显式 --op_dir，否则取 op_host 的父目录（<op_dir>/op_host）。"""
    if args.op_dir:
        return os.path.abspath(args.op_dir)
    return os.path.dirname(os.path.abspath(args.op_host))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locally evaluate a single operator via build + msprof collection"
    )
    parser.add_argument("--op_host", required=True, help="Path to op_host directory (used to infer op_dir/identity)")
    parser.add_argument("--op_kernel", required=True, help="Path to op_kernel directory (used to infer identity)")
    parser.add_argument("--cmakefilelist", default=None, help="Optional CMakeLists.txt path (compat; unused locally)")
    parser.add_argument("--op_name", default=None, help="Operator name, e.g. add_rms_norm_quant")
    parser.add_argument("--op_category", default=None, help="Operator category, e.g. norm")
    parser.add_argument(
        "--op_dir", default=None,
        help="Operator working-copy root (contains build/run entry); default = parent of --op_host",
    )
    parser.add_argument(
        "--build-cmd", dest="build_cmd", default=None,
        help="Override build command (else auto-detect build.sh/run.sh build/cmake)",
    )
    parser.add_argument(
        "--run-cmd", dest="run_cmd", default=None,
        help="Override run command wrapped by msprof (else auto-detect run.sh run/run.py/exe)",
    )
    parser.add_argument("--device", type=int, default=None, help="NPU device id (default: idlest via npu-smi)")
    parser.add_argument("--full", action="store_true", help="Collect all 7 aic-metrics groups (default: 3 key groups)")
    parser.add_argument(
        "--quick", action="store_true",
        help="Task-time only, 1 pass (performance_metrics = latency only)",
    )
    parser.add_argument("--skip-build", action="store_true", help="Skip the build step (assume already built)")
    parser.add_argument("--json_output", default=None, help="Optional path to save the evaluation result as JSON")
    parser.add_argument(
        "--kernel-match", dest="kernel_match", default=None,
        help=(
            "Kernel selector matched against the op_summary 'Op Name' column "
            "(case-insensitive, non-alphanumerics stripped, substring match). "
            "Default: the operator name inferred from --op_host/--op_name. "
            "Pass 'all' to sum every compute kernel (legacy behavior)."
        ),
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help=(
            "Baseline comparison source. Accepts two forms:\n"
            "  (1) An operator working-copy directory — it is built + profiled and compared.\n"
            "  (2) A path to a JSON file produced by a previous --json_output run — "
            "the score & performance_metrics are loaded and compared without re-evaluation."
        ),
    )
    return parser.parse_args()


def _load_baseline_result(
    baseline: str,
    op_key: str,
    op_category: str,
    op_name: str,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    """Load (from JSON) or evaluate (from an op dir) a baseline result for comparison."""
    p = Path(baseline)

    if p.is_file() and p.suffix == ".json":
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("[baseline] loaded from JSON: %s", baseline)
        return data

    if p.is_dir():
        baseline_evaluator = Evaluator(log_dir=_THIS_DIR, op_name=op_name, op_category=op_category)
        logger.info("[baseline] evaluating baseline from directory: %s", baseline)
        result = baseline_evaluator.evaluate(
            str(p), op_key,
            build_cmd=args.build_cmd, run_cmd=args.run_cmd,
            full=args.full, quick=args.quick, device=args.device,
            kernel_match=args.kernel_match,
        )
        return {
            "success": bool(result.get("success", False)),
            "op_name": op_key,
            "score": float(result.get("score", eval_client.FAIL_SCORE)),
            "duration": result.get("duration"),
            "device_id": result.get("device_id"),
            "performance_metrics": result.get("performance_metrics", {}),
            "error_info": result.get("error_info", ""),
        }

    logger.warning("[baseline] cannot resolve baseline path: %s", baseline)
    return None


def _build_comparison(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Compute a comparison summary between current and baseline results."""
    cur_score = current.get("score") or 0.0
    base_score = baseline.get("score") or 0.0

    speedup: Optional[float] = None
    improvement_pct: Optional[float] = None
    if base_score > 0 and cur_score > 0:
        speedup = base_score / cur_score
        improvement_pct = (base_score - cur_score) / base_score * 100.0

    return {
        "baseline_score": base_score,
        "current_score": cur_score,
        "score_delta": cur_score - base_score,
        "speedup": speedup,
        "improvement_pct": improvement_pct,
        "baseline_success": baseline.get("success", False),
    }


def main() -> int:
    args = parse_args()

    op_category, op_name = infer_op_identity(
        args.op_host, args.op_kernel, op_name=args.op_name, op_category=args.op_category,
    )
    op_key = f"{op_category}/{op_name}"
    op_dir = resolve_op_dir(args)

    result = eval_client.evaluate_op_local(
        op_dir, op_key,
        build_cmd=args.build_cmd, run_cmd=args.run_cmd,
        full=args.full, quick=args.quick, device=args.device,
        skip_build=args.skip_build, kernel_match=args.kernel_match,
    )

    output = {
        "success": bool(result.get("success", False)),
        "op_name": op_key,
        "score": float(result.get("score", eval_client.FAIL_SCORE)),
        "duration": result.get("duration"),
        "device_id": result.get("device_id"),
        "performance_metrics": result.get("performance_metrics", {}),
        "kernel_name": result.get("kernel_name"),
        "kernel_breakdown": result.get("kernel_breakdown", []),
        "error_info": result.get("error_info", ""),
        "op_dir": op_dir,
    }

    if args.baseline:
        baseline_data = _load_baseline_result(args.baseline, op_key, op_category, op_name, args)
        if baseline_data is not None:
            output["baseline_comparison"] = _build_comparison(output, baseline_data)

    _stdout_logger.info(json.dumps(output, ensure_ascii=False, indent=2))

    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    return 0 if output["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
