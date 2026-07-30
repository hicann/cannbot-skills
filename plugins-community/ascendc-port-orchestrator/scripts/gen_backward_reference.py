#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Self-contained backward-reference generator for the ascendc-port-orchestrator plugin.

Given a PyTorch forward spec (.py defining `forward` + `BACKWARD_SPEC`), produce the
fp64 autograd backward truth (dL/d{wrt}) the kernel is later verified against. Uses the
plugin's BUNDLED reference_provider (copied from a5_ops; the plugin is self-contained and
does NOT require an a5_ops checkout). Phase 1 of the cannbot-native backward-gen workflow.

Usage: python3 gen_backward_reference.py <forward_spec.py> [out.json]
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "reference_provider"))
import torch  # noqa: E402
from autograd_backward_reference import compute_backward_reference  # noqa: E402 (bundled)


def load_spec(path):
    spec = importlib.util.spec_from_file_location("fwd_spec", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.forward, m.BACKWARD_SPEC


def _resolve_shape(shape_def, case):
    return [case[s] if isinstance(s, str) else s for s in shape_def]


def main(spec_path, out_path=None):
    fwd, bspec = load_spec(spec_path)
    wrt = bspec["wrt"]
    inp_def = bspec["inputs"]
    cases = bspec["cases"]
    dtypes = bspec.get("dtypes", ["float32"])
    seed = bspec.get("seed", 1234)

    results = []
    for ci, case in enumerate(cases):
        for dt in dtypes:
            torch.manual_seed(seed + ci)
            inputs = {}
            for name, d in inp_def.items():
                shape = _resolve_shape(d["shape"], case)
                inputs[name] = torch.randn(*shape, dtype=torch.float64)  # fp64 oracle
            grads = compute_backward_reference(fwd, inputs, wrt)
            results.append({
                "case": case, "dtype": dt,
                "grad_shapes": {k: list(v.shape) for k, v in grads.items()},
                "grad_l2": {k: round(float(v.norm()), 6) for k, v in grads.items()},
            })
    out = {"op": Path(spec_path).stem, "wrt": wrt, "n_scoreable": len(results),
           "reference_method": "self_contained_autograd_fp64 (bundled reference_provider)",
           "results": results}
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({
        key: out[key]
        for key in ("op", "wrt", "n_scoreable", "reference_method")
    }, indent=2))
    print("sample[0]:", json.dumps(results[0], indent=2) if results else "none")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
