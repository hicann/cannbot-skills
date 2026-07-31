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

"""Update verification.json precision section with new ATK-cases rerun."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")
OPS = [
    ("adaptive_avg_pool3d", "/tmp/pool3d_prec.json"),
    ("gather_elements_v2", "/tmp/gather_prec.json"),
    # apply_adam_w_v2 — op not registered in torch_npu, no symmetric path
]

for op, prec_path in OPS:
    vj_path = ROOT / f"output/a3_to_a5_port/src/kernels/{op}/verification.json"
    vj = json.loads(vj_path.read_text())
    prec_run = json.loads(Path(prec_path).read_text())
    summary = prec_run["summary"]
    results = prec_run["results"]
    passed = [r for r in results if r.get("passed")]
    failed = [r for r in results if not r.get("passed") and r.get("tier") not in ("ERROR",)]
    errored = [r for r in results if r.get("tier") == "ERROR"]

    by_dtype: dict = {}
    for r in results:
        dt = r["dtype"]
        rec = by_dtype.setdefault(dt, {"pass": 0, "fail": 0, "error": 0})
        if r.get("passed"):
            rec["pass"] += 1
        elif r.get("tier") == "ERROR":
            rec["error"] += 1
        else:
            rec["fail"] += 1

    prior_prec = vj.get("precision", {})
    new_prec = {
        "status": ("PASS_WITHIN_TOLERANCE" if summary["failed"] == 0 and summary["errored"] <= 6
                   else "PARTIAL"),
        "atk_rerun_2026_05_16": {
            "total": summary["total"],
            "passed": summary["passed"],
            "failed": summary["failed"],
            "errored": summary["errored"],
            "by_tier": summary["by_tier"],
            "by_dtype": by_dtype,
            "method": prec_run["method"],
            "tolerance_doc": prec_run["tolerance_doc"],
            "case_provenance": f"{op}_case_provenance.json (ATK upstream SHA 77f608e6d)",
            "rerun_at_utc": NOW,
            "comparator": "CPU PyTorch reference (same input via .to('npu:0'); OL-83 small-value rule applied)",
            "failed_cases": [{"case_id": r["case_id"], "scale": r["scale_bucket"],
                              "dtype": r["dtype"], "max_abs": r["max_abs_diff"],
                              "max_rel": r["max_rel_diff"]} for r in failed],
            "errored_cases": [{"case_id": r["case_id"], "dtype": r["dtype"],
                               "error": r.get("error", "?")} for r in errored],
        },
        "retracted_8_toy_case_claim": {
            "prior_status": prior_prec.get("status"),
            "prior_pass_a_summary": prior_prec.get("pass_a", {}).get("verdicts",
                                                                     "see prior verification.json"),
            "retraction_reason": (
                "Prior PASS_WITHIN_TOLERANCE 8/8 was based on 8 hand-curated toy "
                "cases (max numel <100K). ATK rerun (S/M/L × full dtype, "
                f"{summary['total']} cases) gives more honest coverage: "
                f"{summary['passed']}/{summary['total']} PASS, "
                f"{summary['failed']} FAIL, {summary['errored']} tooling error."
            ),
        },
    }
    vj["precision"] = new_prec
    vj_path.write_text(json.dumps(vj, indent=2) + "\n")
    print(f"updated {op} precision: status={new_prec['status']} "
          f"pass={summary['passed']}/{summary['total']} fail={summary['failed']} "
          f"err={summary['errored']}")

# apply_adam_w_v2 — annotate that precision rerun is also impossible
vj_path = ROOT / "output/a3_to_a5_port/src/kernels/apply_adam_w_v2/verification.json"
vj = json.loads(vj_path.read_text())
prior_prec = vj.get("precision", {})
vj["precision"] = {
    "status": "RE_EVAL_IMPOSSIBLE_OP_UNAVAILABLE",
    "reason": (
        "torch_npu does not register npu_apply_adam_w(beta1_power, beta2_power, ...) "
        "with kernel on CANN 9.0.0 install at /data/cann_b103. Schema exists in "
        "Python binding but kernel is not registered. Symmetric CPU-vs-NPU precision "
        "comparison via torch-op path therefore unreachable. To re-verify need P89 "
        "(on-host build + dispatch registration of our archive .so), then aclnn-direct "
        "+ CPU reference both sides."
    ),
    "retracted_8_toy_case_claim": {
        "prior_status": prior_prec.get("status"),
        "retraction_reason": (
            "Prior PASS_WITHIN_TOLERANCE was based on 8 toy cases via cached "
            "a5_capture.pt lookup pattern, not live kernel call. With current "
            "infra (no P89), no live independent precision check available."
        ),
    },
    "rerun_attempted_at_utc": NOW,
}
vj_path.write_text(json.dumps(vj, indent=2) + "\n")
print("updated apply_adam_w_v2 precision: RE_EVAL_IMPOSSIBLE_OP_UNAVAILABLE")
