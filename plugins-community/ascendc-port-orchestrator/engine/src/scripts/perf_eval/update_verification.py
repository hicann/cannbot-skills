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

"""Update verification.json for the 3 re-evaluated archives.

For each op:
- Move prior perf.{ratio,method,...} into perf.retracted (audit trail)
- Insert new perf section with Event-timed A5 numbers + method declaration
- Cross-arch ratio left None (requires symmetric A3 rerun)
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OPS = [
    ("adaptive_avg_pool3d", "/tmp/pool3d_a5.json"),
    ("gather_elements_v2", "/tmp/gather_a5.json"),
    ("apply_adam_w_v2", "/tmp/adam_a5.json"),
]
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")

for op, results_path in OPS:
    vj_path = ROOT / f"output/a3_to_a5_port/src/kernels/{op}/verification.json"
    vj = json.loads(vj_path.read_text())
    a5 = json.loads(Path(results_path).read_text())

    prior_perf = vj.get("performance", {})
    valid = []
    for result in a5["results"]:
        kernel_us = result.get("kernel_us")
        if not result.get("skipped") and kernel_us == kernel_us:  # NaN filter
            valid.append(result)

    new_perf = {
        "status": "RE_EVALUATED_PENDING_A3_BASELINE"
                  if len(valid) > 0 else "RE_EVALUATED_OP_UNAVAILABLE",
        "ratio": None,
        "ratio_note": (
            "Cross-arch ratio left null: A3-side symmetric rerun pending. "
            "Prior ratio retracted under perf.retracted (P97 methodology-asymmetry)."
        ),
        "method": a5["method"],
        "a5_single_side_event_us": {
            f"case_{r['case_id']}_{r['scale_bucket']}_{r['dtype']}_numel_{r['numel']}":
            round(r["kernel_us"], 2) for r in valid
        } if valid else (
            "torch_npu does not register this op on CANN 9.0.0 install — "
            "symmetric measurement impossible without aclnn-direct path; "
            "previously measured asymmetrically (now retracted)"
        ),
        "case_source": "atk",
        "case_provenance_file": f"{op}_case_provenance.json",
        "valid_cases": len(valid),
        "total_cases": len(a5["results"]),
        "re_evaluated_at_utc": NOW,
        "retracted": {
            "ratio": prior_perf.get("ratio"),
            "status": prior_perf.get("status"),
            "method": prior_perf.get("method") or prior_perf.get("method_note", ""),
            "retraction_reason": (
                "P97 PERF_METHODOLOGY_ASYMMETRY — prior measurement used "
                "asymmetric methodology (A3-side Python dispatch vs A5-side "
                "C++ std::chrono / direct aclnn timing). Inflated ratio was "
                "an artifact of measurement-stack delta, not a real kernel "
                "speedup. P97 finalize-time gate added 2026-05-15 (commit "
                "39a86baa) blocks new archives from shipping with this pattern."
            ),
        },
    }
    vj["performance"] = new_perf

    vj_path.write_text(json.dumps(vj, indent=2) + "\n")
    print(f"updated {op}: valid_cases={len(valid)}/{len(a5['results'])} "
          f"status={new_perf['status']}")
