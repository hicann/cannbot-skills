<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# max_pool_grad_with_argmax —— SlidingWindow 反向 SIMT 模板

> **状态：DRAFT 参考模板（未编译、未上板、未实测）**。范式自有的 SIMT 单路线模板——算法语义蒸馏自 ops-nn 仓 `pooling/max_pool_grad_with_argmax_v3` 的 SIMT kernel（ascend950 实测，NCHW 布局，逐段对照），包组织按范式 SIMT 单路线模板书写：**自包含、无 common 依赖、无多路线分派**。与正向模板 [../max_pool_with_argmax_simt_package/](../max_pool_with_argmax_simt_package/) 构成 argmax 路由正反向配对。

## I/O 契约

| 端口 | 方向 | 形状 | dtype | 说明 |
|------|------|------|-------|------|
| x | 输入 | 4D NCHW（原输入） | fp32 / fp16 | 值不消费；y 的形状源 |
| grad | 输入 | (N, C, outH, outW) | fp32 / fp16 | 正向输出同形梯度 |
| argmax | 输入 | 同 grad | int32 / int64 | 正向 winner 索引；**与正向输出同 dtype 同口径**（grad-routing §4 契约） |
| y | 输出 | 与 x 同形 | 同 grad | 梯度回传结果 |
| ksize/strides/pads/dilations | 属性 | 长度 2 列表 | — | 与正向同源 |

## 范式展示点

| 文件 | 展示内容 | 依据文档 |
|------|---------|---------|
| `op_kernel/arch35/max_pool_grad_with_argmax_simt.h` | 输入驱动 grid-stride、UintDiv 快除（8 参数 UB 表）、反窗口区间 PStart/PEnd、argmax 命中检查、**零原子直写**、32/64 位双路径 | [../../simt-path.md](../../simt-path.md) §2；[../../../../generic/grad-routing.md](../../../../generic/grad-routing.md) §4；[../../../../generic/window-math.md](../../../../generic/window-math.md) §4 |
| `op_kernel/arch35/max_pool_grad_with_argmax_tiling_key.h` | 单维度 TPL（索引 dtype 档）——最小 TilingKey 形态 | [../../kernel-entrance.md](../../kernel-entrance.md) §3 |
| `op_host/max_pool_grad_with_argmax_tiling.cpp` | inSize 主坐标两步法切分、快除参数 UB 预算 | [../../../../generic/splitting.md](../../../../generic/splitting.md) §1/§3 |
| `tests/assets/golden.py` | 散布形态参考（与 kernel 反窗口枚举等价）、梯度守恒 | [../../../../generic/verification.md](../../../../generic/verification.md) §4 |

## 范围与不包含

覆盖 NCHW（实测 SIMT 形态如此，NHWC 反向走非 SIMT 路线）。**不包含**：重算型路由（无 mask 时由 orig_input/orig_output 重推 winner）、ceil_mode 变体、注册样板与编译 CMake。

## 首例接入验证义务

1. **编译与上板**：模板未独立编译——接入时补 def 注册/CMake 后全量编译
2. **正反向配对对拍**：与正向模板串测（前向产 mask → 本算子回传 → 数值 autograd 或 ops-nn 实测对拍）
3. **梯度守恒**：`sum(y) == sum(grad)`（mask 唯一路由）；反窗口边界用例（verification.md §3 反向行）
4. **快除正确性**：magic/shift 对 (H*W, W, sH, sW) 四组参数在大 shape 下回归（uintdiv-rules.md）
5. **线程档回归**：1024 档实测（无效扫描隐藏收益验证）
