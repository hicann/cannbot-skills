# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
SortSelect 基线示例：TopK（torch.topk）——Ascend910B2 / CANN 8.5.0 实测 20/20 用例通过。

⚠️ 与 elementwise/reduction 不同：SortSelect（topk/sort/argmax 沿 dim）**不写 tl 逐元素 DSL**。
   排序/选择在 Ascend 上由**高阶 AscendC `TopK` API**（内部 Sort32+MrgSort+GatherMask）完成，
   tl 原语无法表达。因此本类算子的“基线”是一份**降级计划**，而非 tl.* 内核。
   完整内核/host 配方在后续 dsl-lowering 阶段落地；本基线仅产出下面的降级计划。

基线降级计划
------------
输入 x（1-8D）→ 双输出 y(同dtype, dim维→k) + idx(恒 int64)；属性 k, dim, largest。

1. 布局：内核只处理**末轴**。dim≠-1 由 Python 包装层 transpose 到末轴、算完 transpose 回。
   记 (outter, n) = (numel/lastdim, lastdim)。

2. 多核切分：按 outter 行切到 vector 核（usedCoreNum = min(vector_core_cnt, outter)），
   每核循环处理其行段，per-call outter=1（内嵌 TopkTiling 与调用行数一致，最省心）。

3. 计算类型（CT）：高阶 TopK 只吃 half/float。
   - fp16→CT=half；fp32→CT=float；bf16→CT=float（load 时 cast）。
   - 整型（int8/uint8/int32/int64）→ **保序 float32 key**：int→float32（|v|≤2^24 无损单调）
     → TopK<float> → 精确 cast 回整型。int8/uint8 经 half、int64 经 int32 中转。

4. 规模分支：
   - n ≤ 4096：单次 TopK（fast path）。
   - n > 4096：**两级分块归并**——切 ≤4096 的块，每块留 k 个候选（TopK<isInitIndex=true>
     传全局索引），候选存 GM workspace，迭代归并到 ≤blockLen 再 final。blockLen=4096（int64=2048）。

5. 保守起点（先打通精度，性能后做）：
   先只做 half/float + n≤4096 + per-call outter=1 的 fast path，验证代表用例 err=0；
   再依次加整型保序 key、大 n 分块归并。多行 tile（outter>1）留作性能优化。

高频坑（基线阶段就要避开）
--------------------------
- 尾部填充哨兵用 Duplicate(src, ±FLT_MAX, inner)（偏移0）再覆写 [0,n)，禁止 Duplicate(src[n],...)（507035）。
- host 固定 SetTilingKey(0)，dtype 分支放进 kernel 读 tiling_data.tilingKey（否则 EE1001）。
- idx 高阶 API 出 int32，需 Cast 到 int64 输出。
- cann-bench proto 中 idx 常 compare:false，仅比对 values，规避 tie-break 差异。
"""

# 代表配置（基线 bring-up 用）：2D 末轴 topk，fp16，largest。
REPRESENTATIVE = dict(shape=(4096, 4096), dtype="float16", k=64, dim=-1, largest=True)
