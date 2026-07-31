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

"""precision_eval_cann_vs_cpu.py — does CANN itself pass the MERE/MARE standard?

For each op archive, runs the SAME reference Model (model.py) twice:
  - candidate: Model(...).npu().eval()  (uses CANN through torch_npu)
  - reference: Model(...).cpu().eval()
Then compares with the same MERE/MARE thresholds the production skill uses.

This answers the user's question: "If CANN passes the new standard but our
kernels don't, CANN is still a useful reference. If CANN itself fails too,
we need to rethink — and CANN performance numbers might come from precision
shortcuts."

Usage:
  python3 precision_eval_cann_vs_cpu.py <archive_dir> [--json out.json] [--quiet]
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
try:
    import torch_npu  # noqa: F401
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

EPS = 1e-7
PRECISION_THRESHOLDS = {
    torch.float16: 2 ** -10,
    torch.bfloat16: 2 ** -7,
    torch.float32: 2 ** -13,
}


def compute_mere_mare(a, g):
    a = a.detach().to(torch.float64).cpu()
    g = g.detach().to(torch.float64).cpu()
    if a.shape != g.shape:
        raise ValueError(f"shape mismatch: a={tuple(a.shape)} g={tuple(g.shape)}")
    diff = (a - g).abs()
    rel = diff / (g.abs() + EPS)
    if rel.numel() == 0:
        return 0.0, 0.0
    return float(rel.mean().item()), float(rel.max().item())


def case_passes(actual, golden):
    dtype = golden.dtype
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
        match = bool(torch.equal(actual.cpu(), golden.cpu()))
        return {"dtype": str(dtype).replace("torch.", ""), "rule": "bit-exact",
                "passed": match, "mere": 0.0 if match else float("inf"),
                "mare": 0.0 if match else float("inf")}
    if dtype not in PRECISION_THRESHOLDS:
        return {"dtype": str(dtype).replace("torch.", ""), "rule": "unknown",
                "passed": False, "mere": float("nan"), "mare": float("nan")}
    thr = PRECISION_THRESHOLDS[dtype]
    mere, mare = compute_mere_mare(actual, golden)
    return {"dtype": str(dtype).replace("torch.", ""), "rule": "MERE<T AND MARE<10T",
            "passed": (mere < thr) and (mare < 10 * thr),
            "mere": mere, "mare": mare,
            "threshold": thr, "mare_threshold": 10 * thr}


def to_cpu(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().contiguous()
    if isinstance(v, (list, tuple)):
        return type(v)(to_cpu(x) for x in v)
    if isinstance(v, dict):
        return {k: to_cpu(x) for k, x in v.items()}
    return v


def to_npu(v):
    if isinstance(v, torch.Tensor):
        return v.detach().npu().contiguous()
    if isinstance(v, (list, tuple)):
        return type(v)(to_npu(x) for x in v)
    if isinstance(v, dict):
        return {k: to_npu(x) for k, x in v.items()}
    return v


def normalize(out):
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (list, tuple)):
        return [t for t in out if isinstance(t, torch.Tensor)]
    if hasattr(out, "_asdict"):
        return [t for t in out._asdict().values() if isinstance(t, torch.Tensor)]
    return []


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_class(mod, prefer):
    if hasattr(mod, prefer):
        return getattr(mod, prefer)
    for a in dir(mod):
        if a.startswith("Model") and isinstance(getattr(mod, a), type):
            return getattr(mod, a)
    raise AttributeError(f"no Model in {mod}")


def evaluate(archive: Path, verbose=True):
    model_py = archive / "model.py"
    if not model_py.is_file():
        raise FileNotFoundError(f"missing {model_py}")

    op_name = archive.name
    if verbose:
        print(f"=== CANN-vs-CPU eval for {op_name} ===")

    # Load model.py twice as different module names so each gets its own state
    mod_cpu = load_module(model_py, f"{op_name}_cpu_ref")
    mod_npu = load_module(model_py, f"{op_name}_npu_ref")

    cls_cpu = find_class(mod_cpu, "Model")
    cls_npu = find_class(mod_npu, "Model")

    torch.manual_seed(0)
    init = []
    if hasattr(mod_cpu, "get_init_inputs"):
        init = mod_cpu.get_init_inputs()
    get_inputs = getattr(mod_cpu, "get_input_groups", None)
    if get_inputs is None:
        raise AttributeError("no get_input_groups")
    groups = get_inputs()

    cpu_model = (cls_cpu(*copy.deepcopy(init)) if init else cls_cpu()).cpu().eval()
    npu_model = (cls_npu(*copy.deepcopy(init)) if init else cls_npu())
    if HAS_NPU:
        npu_model = npu_model.npu()
    npu_model = npu_model.eval()

    results = []
    for idx, raw in enumerate(groups):
        try:
            cpu_in = to_cpu(raw)
            npu_in = to_npu(raw) if HAS_NPU else cpu_in

            with torch.no_grad():
                ref = cpu_model(*cpu_in) if isinstance(cpu_in, (list, tuple)) else cpu_model(cpu_in)
                cand = npu_model(*npu_in) if isinstance(npu_in, (list, tuple)) else npu_model(npu_in)

            rt = normalize(ref)
            ct = normalize(cand)
            if len(rt) != len(ct):
                results.append({"case": idx, "passed": False,
                                "error": f"output count {len(rt)} vs {len(ct)}"})
                continue

            per_out = []
            ok = True
            for j, (r, c) in enumerate(zip(rt, ct)):
                cc = c.cpu().to(r.dtype) if r.dtype != c.dtype else c.cpu()
                m = case_passes(cc, r)
                m["output_idx"] = j
                m["shape"] = list(r.shape)
                per_out.append(m)
                if not m["passed"]:
                    ok = False
            results.append({"case": idx, "passed": ok, "outputs": per_out})
            if verbose:
                tag = "PASS" if ok else "FAIL"
                summ = ", ".join(f"o{m['output_idx']} M={m['mere']:.2e}/{m['mare']:.2e}"
                                  for m in per_out)
                print(f"case[{idx}]: {tag}  {summ}")
        except Exception as e:
            results.append({"case": idx, "passed": False, "error": f"{type(e).__name__}: {e}"})
            if verbose:
                traceback.print_exc()

    n_total = len(results)
    n_pass = sum(1 for r in results if r["passed"])
    summary = {
        "op": op_name,
        "reference": "CPU vs CANN(NPU) — same Model class, different device",
        "metric": "MERE/MARE per Ascend/agent-skills SKILL",
        "n_total": n_total,
        "n_passed": n_pass,
        "n_failed": n_total - n_pass,
        "pass_rate": n_pass / n_total if n_total else 0.0,
        "results": results,
    }
    if verbose:
        print(f"\n--- {op_name}: {n_pass}/{n_total} CANN passes CPU truth ---")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("archive_dir")
    p.add_argument("--json")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    s = evaluate(Path(args.archive_dir).resolve(), verbose=not args.quiet)
    if args.json:
        Path(args.json).write_text(json.dumps(s, indent=2, default=str))
    return 0 if s["n_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
