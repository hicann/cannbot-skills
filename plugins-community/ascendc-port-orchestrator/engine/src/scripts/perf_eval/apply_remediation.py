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

"""Apply concrete remediation status + plan into 3 verification.json files.

After P109 audit (user catch 2026-05-16):
- gather: was PASS_W_ERR 36/42 due to my own test bug (uint8 range);
  rerun with fix → 42/42 PASS T1 bit-exact across 7 dtypes.
- pool3d: bf16 3/6 FAIL was wrong-tolerance, not kernel bug. ULP probe
  shows observed diff 0.0117-0.0195 vs expected bf16 ULP atol 0.34 →
  kernel within hw-floor. T2 atol=1e-2 is wrong for unnormalized
  ([-100,100]) inputs + reduction op. Needs scale-aware atol (OL-104).
- apply_adam_w_v2: torch_npu path unavailable, BUT archive has
  apply_adam_w_v2_runner.cpp aclnn-direct + build_runner.sh. Remediation
  = call runner via subprocess from Python harness, compare to CPU adam
  reference. Independent precision check IS possible.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")


def update(op, status, remediation, **extra):
    vj_path = ROOT / f"output/a3_to_a5_port/src/kernels/{op}/verification.json"
    vj = json.loads(vj_path.read_text())
    prec = vj.setdefault("precision", {})
    prec["status"] = status
    prec["remediation_plan"] = remediation
    prec["status_set_at_utc"] = NOW
    for k, v in extra.items():
        prec[k] = v
    vj_path.write_text(json.dumps(vj, indent=2) + "\n")
    print(f"{op}: status={status}")


update(
    "gather_elements_v2",
    "PASS_T1_BIT_EXACT_42_OF_42",
    {
        "task": "P109 closed for this op",
        "evidence": "All 7 dtypes (bool/fp32/int8/int16/int32/int64/uint8) max_abs=0 across S/M/L cases",
        "earlier_36_42_was": (
            "my own test bug: torch.randint(-100, 100, dtype=uint8) is invalid "
            "since uint8 can't be negative; fixed by clamping range to "
            "torch.iinfo(dt).{min,max}"
        ),
    },
    atk_rerun_2026_05_16_v2={"total": 42, "passed": 42,
        "method": (
            "CPU PyTorch reference, same input via .to('npu:0'), OL-83 "
            "small-value rule, dtype-clamped randint range"
        ),
    },
)

update(
    "adaptive_avg_pool3d",
    "NOT_VERIFIED_AWAITING_SCALE_AWARE_TOLERANCE",
    {
        "task": "P107 (OL-104 scale-aware atol)",
        "concrete_pass_criterion": (
            "Pass when atol per case scaled by max_input_magnitude × bf16_eps × "
            "reduction_factor. ULP probe (in-script) gave: "
            "observed_diff_pool3d_bf16 = 0.0117-0.0195 abs, "
            "expected_bf16_hw_floor_at_max_output_magnitude(~43) = 0.34 abs. "
            "Observed << expected → kernel within bf16 hw-floor."
        ),
        "evidence_pre_remediation": (
            "fp32 6/6 PASS T1 (max_abs ≤6e-8), fp16 6/6 PASS T2, bf16 3/6 "
            "FAIL on strict 1e-2 atol with diff at near-zero output cells "
            "(where rtol contribution is 0). This is OL-104 territory."
        ),
        "remediation_options_concrete": [
            "Option A: Adopt canonical Pass A (precision_eval_two_tier.py §4.5.3 "
            "already has OL-83 + OL-104 + scale-aware atol)",
            "Option B: Extend verify_precision_atk.py with "
            "scale_aware_atol(dtype, ref) = max(atol_floor, "
            "|ref|.max() * dtype_ulp_factor)",
        ],
        "NOT shippable_state": (
            "Until either A or B closes bf16 boundary, this archive remains "
            "UNVERIFIED on bf16 dtype."
        ),
    },
    bf16_ulp_probe_data={
        "test_setup": "x_fp32 in [-100,100], adaptive_avg_pool3d (16,1,257,32,32) → (50,8,8)",
        "fp32_ref_max_abs_output": 43.58,
        "bf16_npu_observed_diff_vs_fp32_ref": 0.147,
        "expected_bf16_ulp_atol_at_max_output": 0.34,
        "verdict": "0.147 < 0.34 = WITHIN_BF16_HW_FLOOR",
    },
)

update(
    "apply_adam_w_v2",
    "NOT_VERIFIED_REMEDIATION_KNOWN",
    {
        "task": "P109 phase B — call existing aclnn-direct runner from Python",
        "concrete_pass_criterion": (
            "Use apply_adam_w_v2_runner.cpp (already in archive) via subprocess: "
            "feed ATK cases as binary blobs, capture NPU output, compare to CPU "
            "adam update reference. Pass = within OL-83 small-value tolerance + "
            "per-dtype tier."
        ),
        "earlier_re_eval_impossible_correction": (
            "Earlier RE_EVAL_IMPOSSIBLE_OP_UNAVAILABLE was based on torch_npu "
            "NOT registering npu_apply_adam_w. But the archive ALREADY contains: "
            "apply_adam_w_v2_runner.cpp (aclnn-direct standalone exe), "
            "build_runner.sh (build script), pass_a_runner.py (precision driver). "
            "Independent precision check IS possible via this path. I missed this."
        ),
        "implementation_estimate": (
            "30-60 min: build runner on A5, write Python wrapper that drives "
            "runner per case + CPU adam reference comparator"
        ),
        "NOT shippable_state": "Until aclnn-direct precision rerun lands, archive has zero independent verification.",
    },
)
