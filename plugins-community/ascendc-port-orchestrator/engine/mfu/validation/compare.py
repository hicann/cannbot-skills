#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""验证harness: 模型预测 MFU_max vs msprof 实测 -> η_util + 预测准确性表。
NPU 到位前用示例数据跑通格式; 拿到 msprof 后把 measured_us 换成实测即可。

用法:
  python3 compare.py            # 跑示例(展示对照表格式)
  # 真机: 解析 msprof 输出得各算子 device time, 填入 CASES 的 measured_us
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mfu_model import HW, flash_attention_ops, matmul_ops, solve_mfu, solve_matmul_tiled


def predict(case):
    """返回模型预测 (mfu_max, bottleneck, t_ideal_us)。"""
    hw = HW[case["hw"]]
    k = case["kind"]
    if k == "matmul":
        r = solve_matmul_tiled(case["M"], case["N"], case["K"], hw, case["dtype"])
        t_ideal = min(r["t_us"].values())  # t_compute (理想)
        return r["mfu_max"], r["bottleneck"], r["t_us"].get("compute", t_ideal)
    elif k == "fa":
        fwd, _ = flash_attention_ops(case["B"], case["H"], case["S"], case["D"],
                                     case["dtype"], case.get("causal", False))
        r = solve_mfu(fwd, hw)
        return r["mfu_max"], r["bottleneck"], r["t_compute_us"]
    raise ValueError(k)


def report(cases):
    print(f"{'case':32} {'hw':8} {'pred_MFU':>8} {'bottleneck':>11} "
          f"{'t_ideal_us':>10} {'meas_us':>9} {'meas_MFU':>8} {'eta':>6}")
    for c in cases:
        mfu_max, bn, t_ideal = predict(c)
        meas = c.get("measured_us")
        if meas:
            meas_mfu = round(t_ideal / meas * mfu_max, 3)   # 实测MFU = 理论compute时间/实测墙钟 × 理论MFU基
            eta = round(meas_mfu / mfu_max, 3) if mfu_max else 0
            ok = "" if meas >= t_ideal - 1e-6 else "  ⚠天花板被突破!"
        else:
            meas_mfu = eta = "-"
            ok = ""
        print(f"{c['name']:32} {c['hw']:8} {mfu_max:>8} {bn:>11} "
              f"{round(t_ideal,2):>10} {str(meas):>9} {str(meas_mfu):>8} {str(eta):>6}{ok}")
    print("\nη_util = 实测MFU / 理论MFU_max (有效利用率); <1 正常(tiling/尾块/流水损耗)。"
          "\n实测>理论(天花板被突破)=> 模型 peak/BW 口径偏低, 需修。")


# 待 msprof 填 measured_us 的 case 列表(measured_us=None 占位)
CASES = [
    {"name": "FA dense B1H8S2048D128 causal", "kind": "fa", "hw": "950PR",
     "B": 1, "H": 8, "S": 2048, "D": 128, "dtype": "fp16", "causal": True,
     "measured_us": 17.0},   # KB 实测 ceiling 0.017ms (示例: 接近理论 -> η≈1)
    {"name": "matmul 4096^3 bf16", "kind": "matmul", "hw": "950PR",
     "M": 4096, "N": 4096, "K": 4096, "dtype": "bf16", "measured_us": None},
    {"name": "matmul 4096^3 bf16", "kind": "matmul", "hw": "910C",
     "M": 4096, "N": 4096, "K": 4096, "dtype": "bf16", "measured_us": None},
]

if __name__ == "__main__":
    report(CASES)
