<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# npu_get_float_status_simt_draft —— DRAFT 参考实现说明

> **状态：DRAFT（未编译、未上板、未实测）**。ControlFlow 范式设备核路线 DSL 侧的结构参考实现（对照 AscendC 侧 `ops-nn/control/npu_*_float_status` 真实样例包）；cannbot-dsl 仓当前没有信号类算子样例，本文件填补结构空白。**降级态是唯一形态**：DSL API 面未暴露 FP 状态读取，get 恒写规范零值。

## 结构展示点

| 文件位置 | 展示内容 | 依据文档 |
|---------|---------|---------|
| `_alloc_zero_simt` / `_get_zero_simt` | 纯 SIMT kernel + `dim3` 成对启动 + 最小有效线程档 | status-kernel.md §1/§2 |
| `_clear_side_effect` | **混合形态**：SIMD 副作用段 + `vf(mode="simt", thread=N)` + `kernel[block_count]` 启动 | status-kernel.md §3、compute.md 混合编程规则 |
| 常量几何 | 退化交付契约：常量 launch、无 tiling 概念、幂等写 I3 | status-kernel.md §5 |
| 参数注释 | 副作用段 V1 对齐参数的脆弱性声明义务 | signal-model.md §3 |

## 待验证点（首例接入时逐项闭环）

| # | 待验证项 | 说明 |
|---|---------|------|
| B1 | `output[i] = 0.0` 的 SIMT GM 直写路径 | Tensor 下标写走标量流水口径（compute.md「标量计算」警告——8 元素微量场景合法） |
| B2 | 混合 kernel 内 SIMT VF 的 `thread=N` 写法与编译器校验 | compute.md「Host 启动规则」两条形态不可互换 |
| B3 | 副作用通道在 3510 的有效性 | clear→get==0 逐平台闭环实测，不信任 AscendC 平移 |
| B4 | fp16 `vdups` 值域（3~8 段） | V1 参数对齐，目标平台重新论证 |

## 与 AscendC 样例包的对应

本范式 ascendc 适配层已整包内嵌真实样例 [npu_get_float_status_package](../../../ascendc/example/npu_get_float_status_package/)（ops-nn 真实代码）；本 DRAFT 是其 DSL 对等形态的骨架示意——语义契约（[8]×fp32、幂等、降级态）两者一致，实现载体按各自语言。
