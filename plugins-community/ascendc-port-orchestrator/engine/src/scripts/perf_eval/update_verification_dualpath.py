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

"""Apply P101 dual-path results into verification.json + mark performance status."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")
OPS = [
    ("adaptive_avg_pool3d", "/tmp/pool3d_dual.json"),
    ("gather_elements_v2", "/tmp/gather_dual.json"),
]

for op, path in OPS:
    vj_path = ROOT / f"output/a3_to_a5_port/src/kernels/{op}/verification.json"
    vj = json.loads(vj_path.read_text())
    dual = json.loads(Path(path).read_text())
    summary = dual["summary"]

    new_perf = {
        "status": "DUAL_PATH_PORT_AS_WIRING",
        "ratio_mean_base_over_cand": summary["mean_ratio_base_over_cand"],
        "ratio_interpretation": (
            "Candidate and baseline dispatch to the SAME upstream .so (our archive "
            "build not registered — P89 pending). Ratio ~ 1.0× by construction is "
            "the EXPECTED, HONEST result, not a kernel speedup. This data point "
            "demonstrates the port-as-wiring state of the archive: it correctly "
            "exposes the upstream op through our wrapper but does NOT itself run "
            "a different kernel."
        ),
        "method": dual["method"],
        "report_format": dual["report_format"],
        "candidate_path": dual["candidate_path"],
        "baseline_path": dual["baseline_path"],
        "valid_cases": summary["valid_count"],
        "total_cases": len(dual["results"]),
        "per_case_data_file": f"output/a3_to_a5_port/docs/REEVAL_2026_05_16.md#{op}",
        "a3_side_pending": "DS will run same script on A3 NPU after current 5-op sweep; "
                          "method field must remain byte-equal for cross-arch ratio",
        "rerun_at_utc": NOW,
    }
    prior_retract = vj.get("performance", {}).get("retracted", {})
    new_perf["retracted"] = prior_retract  # preserve P98 retraction chain
    vj["performance"] = new_perf
    vj_path.write_text(json.dumps(vj, indent=2) + "\n")
    print(f"updated {op} perf: status=DUAL_PATH_PORT_AS_WIRING "
          f"ratio_mean={summary['mean_ratio_base_over_cand']:.4f} "
          f"valid={summary['valid_count']}/{len(dual['results'])}")
