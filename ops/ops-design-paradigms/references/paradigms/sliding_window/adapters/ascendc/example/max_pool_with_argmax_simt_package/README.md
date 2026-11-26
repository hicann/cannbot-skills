<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# max_pool_with_argmax —— SlidingWindow 正向 SIMT 模板

> **状态：DRAFT 参考模板（未编译、未上板、未实测）**。范式自有的 SIMT 单路线模板——算法语义蒸馏自 ops-nn 仓 `pooling/max_pool_with_argmax_v3` 的 SIMT kernel（ascend950 实测，逐段对照），包组织按范式 SIMT 单路线模板书写：**自包含、无 common 依赖、无多路线分派**。

## I/O 契约

| 端口 | 方向 | 形状 / 值域 | dtype | 说明 |
|------|------|-----------|-------|------|
| x | 输入 | 4D（data_format 决定 NCHW/NHWC） | fp32 / fp16 | 待池化数据 |
| y | 输出 | 窗口公式推导（floor 模式） | 同 x | 聚合值；空窗输出 −Inf |
| argmax | 输出 | 同 y | int32 / int64 | winner 线性索引（口径 h*W+w，批次内） |
| ksize/strides/pads/dilations | 属性 | 长度 2 列表 | — | 窗口参数；pads 为对称 [pH, pW] |
| data_format | 属性 | "NCHW" / "NHWC" | — | 默认 NCHW |

## 范式展示点

| 文件 | 展示内容 | 依据文档 |
|------|---------|---------|
| `op_kernel/arch35/max_pool_with_argmax_simt.h` | 输出驱动 grid-stride、直除坐标分解、窗口数学（负起点 while 推进）、−Inf 哨兵 + CycleUpdate NaN 优先、双输出 | sampling 见 [../../simt-path.md](../../simt-path.md) §1；[../../../../generic/window-math.md](../../../../generic/window-math.md) §2/§3 |
| `op_kernel/arch35/max_pool_with_argmax_tiling_key.h` | ASCENDC_TPL_ARGS_DECL 模板编程（规则 §5.5.4 强制形态）：format + 索引 dtype 档 | [../../kernel-entrance.md](../../kernel-entrance.md) §3 |
| `op_host/max_pool_with_argmax_tiling.cpp` | 属性归一化（host 侧）、窗口公式、两步法切分（PER_CORE_MIN 1024）、GET_TPL_TILING_KEY | [../../kernel-entrance.md](../../kernel-entrance.md) §2；[../../../../generic/splitting.md](../../../../generic/splitting.md) §3 |
| `op_host/max_pool_with_argmax_infershape.cpp` | 输出 shape 推导（与 tiling 侧窗口公式同源） | [../../../../generic/window-math.md](../../../../generic/window-math.md) §1 |
| `tests/assets/golden.py` | tie-break 首遇 / NaN 优先 / 空窗 −Inf 逐点对齐的参考实现 | [../../../../generic/verification.md](../../../../generic/verification.md) §1/§3 |

## 范围与不包含

覆盖 NCHW + NHWC 双布局、floor 模式、对称 pad。**不包含**（成熟包的工程形态，非 SIMT 模板内容）：ceil_mode / 自适应窗口变体、多路线 TilingKey 分派（SIMD 小核/大核）、op 定义注册样板与编译 CMake（按 ascendc 通用技能补齐）。

## 首例接入验证义务

1. **编译与上板**：模板未独立编译——接入时按仓内构建规范补 def 注册/CMake 后全量编译
2. **golden 对拍**：与 [ops-nn](https://gitcode.com/cann/ops-nn) `pooling/max_pool_with_argmax_v3` SIMT 路线实测输出对拍（同输入同属性，含 NaN/平局/空窗用例）
3. **边界用例**：verification.md §3 正向行（空窗 −Inf、负起点 dilation 推进、NaN 传播、平局首遇、大 shape int64）
4. **线程档回归**：256 档实测；寄存器压力异常时按 splitting.md §4 重新分档并回写文档
