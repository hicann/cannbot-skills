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

"""cpu_truth_sweep_harness.py — sweep all archived ops, eval each against CPU.

Designed to run inside the npu_dev3 container. For each archive dir:
  1. Normalize layout: ensure {kernel/*.cpp,*.h,pybind11.cpp} structure
  2. Build via /root/AscendOpGenAgent/utils/build_ascendc.py (240s timeout)
  3. Run /root/eval_scripts/precision_eval_cpu_ref.py (180s timeout)
  4. Append one TSV line: op\tlevel\tn_total\tn_pass\trate\tstatus\tnote

If build or eval fails → record status reason, continue to next op.

Usage:
  python3 cpu_truth_sweep_harness.py \
      --archives-root /root/cpu_truth_sweep/archives \
      --out /root/cpu_truth_sweep/results.tsv \
      --build-timeout 240 --eval-timeout 180
"""
from __future__ import annotations
import logging

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

EVAL_SCRIPT = "/root/eval_scripts/precision_eval_cpu_ref.py"
CANN_EVAL_SCRIPT = "/root/eval_scripts/precision_eval_cann_vs_cpu.py"
BUILD_SCRIPT = "/root/AscendOpGenAgent/utils/build_ascendc.py"
SOC = "Ascend950PR_9589"


def detect_level(op_dir: Path) -> str:
    # Heuristic: matches pattern <num>_<TitleCase>; level inferred from manifest if present
    manifest = op_dir / "manifest.json"
    if manifest.is_file():
        try:
            import json
            data = json.loads(manifest.read_text())
            lvl = data.get("level") or data.get("source_level")
            if lvl:
                return f"L{lvl}" if not str(lvl).startswith("L") else str(lvl)
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
    return "L?"


def normalize_layout(op_dir: Path) -> bool:
    """Ensure op_dir has {kernel/*.cpp, kernel/*.h, kernel/pybind11.cpp} layout.
    Some old archives have these files at root — move them into kernel/.
    """
    kernel_dir = op_dir / "kernel"
    if kernel_dir.is_dir():
        # Has kernel/ — verify pybind11.cpp + at least one .cpp source
        if (kernel_dir / "pybind11.cpp").is_file():
            return True
        # kernel/ exists but missing pybind11 — broken
        return False

    # Old-style: files at root. Move *.cpp, *.h, pybind11.cpp into kernel/
    has_pybind = (op_dir / "pybind11.cpp").is_file()
    if not has_pybind:
        return False

    kernel_dir.mkdir(exist_ok=True)
    for f in list(op_dir.iterdir()):
        if f.is_file() and f.suffix in (".cpp", ".h", ".hpp"):
            shutil.move(str(f), str(kernel_dir / f.name))
    return (kernel_dir / "pybind11.cpp").is_file()


def run_with_timeout(cmd, cwd, timeout, env=None):
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        try:
            out, _ = proc.communicate(timeout=timeout)
            return proc.returncode, out.decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            try:
                out, _ = proc.communicate(timeout=5)
            except Exception:
                out = b""
            return -1, out.decode("utf-8", errors="replace") + "\n[TIMEOUT]"
    except Exception as e:
        return -2, f"[EXEC_ERR] {type(e).__name__}: {e}"


PASS_RE = re.compile(r"total:\s*(\d+)\s+passed:\s*(\d+)\s+failed:\s*(\d+)")


def parse_eval_summary(stdout: str):
    m = PASS_RE.search(stdout)
    if not m:
        return None
    total, passed, failed = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return total, passed, failed


def process_one(op_dir: Path, build_timeout: int, eval_timeout: int) -> dict:
    op = op_dir.name
    info = {"op": op, "level": detect_level(op_dir), "n_total": 0, "n_pass": 0, "rate": 0.0,
            "cann_total": 0, "cann_pass": 0, "cann_rate": 0.0,
            "status": "?", "note": ""}
    t0 = time.time()

    if not normalize_layout(op_dir):
        info["status"] = "LAYOUT_FAIL"
        info["note"] = "missing pybind11.cpp"
        info["elapsed"] = round(time.time() - t0, 1)
        return info

    # 1. Build
    rc, out = run_with_timeout(
        ["python3", BUILD_SCRIPT, str(op_dir.resolve()), "-v", SOC],
        cwd=str(op_dir.resolve()), timeout=build_timeout,
    )
    if rc != 0:
        info["status"] = "BUILD_FAIL" if rc > 0 else ("BUILD_TIMEOUT" if rc == -1 else "BUILD_ERR")
        # capture last 200 chars of error
        tail = out[-300:].replace("\n", " ").replace("\t", " ")
        info["note"] = tail[-200:]
        info["elapsed"] = round(time.time() - t0, 1)
        return info

    # 2. Eval
    json_path = op_dir / ".cpu_ref_result.json"
    rc2, out2 = run_with_timeout(
        ["python3", EVAL_SCRIPT, str(op_dir), "--quiet", "--json", str(json_path)],
        cwd=str(op_dir), timeout=eval_timeout,
    )
    parsed = None
    if json_path.is_file():
        try:
            import json
            data = json.loads(json_path.read_text())
            parsed = (int(data.get("n_total", 0)), int(data.get("n_passed", 0)), int(data.get("n_failed", 0)))
        except Exception:
            parsed = None
    if parsed is None:
        parsed = parse_eval_summary(out2)
    if parsed is None:
        info["status"] = "EVAL_TIMEOUT" if rc2 == -1 else ("EVAL_ERR" if rc2 != 0 else "EVAL_NOPARSE")
        tail = out2[-300:].replace("\n", " ").replace("\t", " ")
        info["note"] = tail[-200:]
        info["elapsed"] = round(time.time() - t0, 1)
        return info

    total, passed, failed = parsed
    info["n_total"] = total
    info["n_pass"] = passed
    info["rate"] = (passed / total) if total else 0.0
    info["status"] = "PASS" if failed == 0 else ("PARTIAL" if passed > 0 else "FAIL")

    # 3. CANN baseline: run Model on NPU vs Model on CPU, same MERE/MARE
    cann_json = op_dir / ".cann_vs_cpu_result.json"
    rc3, out3 = run_with_timeout(
        ["python3", CANN_EVAL_SCRIPT, str(op_dir), "--quiet", "--json", str(cann_json)],
        cwd=str(op_dir), timeout=eval_timeout,
    )
    if cann_json.is_file():
        try:
            import json
            d = json.loads(cann_json.read_text())
            info["cann_total"] = int(d.get("n_total", 0))
            info["cann_pass"] = int(d.get("n_passed", 0))
            info["cann_rate"] = float(d.get("pass_rate", 0.0))
        except Exception as e:
            info["note"] = (info["note"] + f" [cann_parse_err:{e}]")[-200:]

    info["elapsed"] = round(time.time() - t0, 1)
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archives-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--build-timeout", type=int, default=240)
    p.add_argument("--eval-timeout", type=int, default=180)
    p.add_argument("--filter", default=None, help="substring filter on op name")
    args = p.parse_args()

    root = Path(args.archives_root)
    if not root.is_dir():
        print(f"archives root not found: {root}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort: L1 first, then L2, then L3, then others. Use filename prefix only.
    ops = sorted([d for d in root.iterdir() if d.is_dir()])
    if args.filter:
        ops = [d for d in ops if args.filter in d.name]

    # Fresh TSV header
    if not out_path.exists():
        with out_path.open("w") as f:
            f.write("op\tlevel\tn_total\tn_pass\trate\tcann_total\tcann_pass\tcann_rate\tstatus\telapsed_s\tnote\n")

    print(f"[harness] {len(ops)} ops to process")
    for i, op_dir in enumerate(ops, 1):
        print(f"[{i}/{len(ops)}] {op_dir.name} ...", flush=True)
        info = process_one(op_dir, args.build_timeout, args.eval_timeout)
        line = "\t".join([
            info["op"], info["level"],
            str(info["n_total"]), str(info["n_pass"]),
            f"{info['rate']:.3f}",
            str(info["cann_total"]), str(info["cann_pass"]),
            f"{info['cann_rate']:.3f}",
            info["status"],
            str(info.get("elapsed", "?")),
            info["note"][:200],
        ])
        with out_path.open("a") as f:
            f.write(line + "\n")
        print(
            f"  → {info['status']}  ours={info['n_pass']}/{info['n_total']}  "
            f"cann={info['cann_pass']}/{info['cann_total']}  "
            f"({info.get('elapsed', '?')}s)",
            flush=True,
        )

    print(f"[harness] done — results in {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
