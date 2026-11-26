#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""离线 ST：onboard/compare.py 判定逻辑自测（合成数据，无需板子）。

验证 classify() 的三种判定与边界，以及 compare.py 的退出码（有 BELOW_FLOOR → 1）。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[2]
COMPARE_PY = SKILL_DIR / "tests/onboard/compare.py"

spec = importlib.util.spec_from_file_location("compare", COMPARE_PY)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)

fails = []


def ck(label, cond):
    if not cond:
        fails.append(label)


# ---- 1) classify 纯函数（target_eff=0.6, tol=0.03）----
TE = 0.6
ck("跑穿下界 → BELOW_FLOOR", compare.classify(7.0, 13.6, TE) == "BELOW_FLOOR")
ck("落在 [理论,cp4] → OK", compare.classify(18.0, 13.6, TE) == "OK")
ck("效率过低 → LOW_EFF", compare.classify(30.0, 13.6, TE) == "LOW_EFF")
# 边界：measured == 理论·(1-tol) 不算跑穿（严格 <）
ck(
    "恰在容噪带边界 → 不判 BELOW_FLOOR",
    compare.classify(97.0, 100.0, TE) != "BELOW_FLOOR",
)
# 边界：eff 恰等于 target → OK
ck("eff 恰 = target → OK", compare.classify(100.0, 60.0, TE) == "OK")
# 边界：eff 略低于 target → LOW_EFF
ck("eff 略低于 target → LOW_EFF", compare.classify(100.0, 59.0, TE) == "LOW_EFF")
# 刚跨过容噪带 → BELOW_FLOOR
ck("刚跨容噪带 → BELOW_FLOOR", compare.classify(96.0, 100.0, TE) == "BELOW_FLOOR")


# ---- 2) 退出码：有 BELOW_FLOOR → 1；全 OK → 0 ----
def run_compare(csv_text):
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_text)
    try:
        r = subprocess.run(
            [sys.executable, str(COMPARE_PY), "--file", f.name],
            capture_output=True,
            text=True,
        )
        return r.returncode, r.stdout
    finally:
        Path(f.name).unlink()


# 含一个跑穿下界的 case（softmax_m_4096x1024 理论~13.6，给 7 → BELOW_FLOOR）
rc, out = run_compare(
    "# operator: softmax\n# cases: softmax_cases.json\n# source: SELFTEST\n"
    "case_id,measured_us\nsoftmax_m_4096x1024,7.0\nsoftmax_l_65536x1024,250.0\n"
)
ck("有 BELOW_FLOOR → 退出码 1", rc == 1)
ck("输出含 BELOW_FLOOR", "BELOW_FLOOR" in out)

# 全部健康 → 退出码 0
rc2, _ = run_compare(
    "# operator: softmax\n# cases: softmax_cases.json\n# source: SELFTEST\n"
    "case_id,measured_us\nsoftmax_m_4096x1024,18.0\nsoftmax_l_65536x1024,250.0\n"
)
ck("全 OK → 退出码 0", rc2 == 0)


def main():
    n = 10
    if fails:
        print(f"FAIL  ({len(fails)} 处 / {n} 项)")
        for f in fails:
            print("  -", f)
        return 1
    print(f"PASS  ({n} 项：classify 三判定+边界 / compare 退出码)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
