---
type: CATLASS DSL Operator Example
title: FusedMatMul
description: Ascend 950 FusedMatMul 的矩阵乘后处理融合、调度模型、数据路径与同步语义。
tags: [catlass-dsl, operator, matmul, fusion, gelu, epilogue, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T12:29:38Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T12:29:38Z'}
sources:
  - id: guide
    resource: https://gitcode.com/cann/ops-nn/blob/39a50f12554f00809f09eaf0b8a0675477879a4e/matmul/fused_mat_mul/README.md
    title: FusedMatMul interface and fused-operation guide
  - id: entry
    resource: https://gitcode.com/cann/ops-nn/blob/39a50f12554f00809f09eaf0b8a0675477879a4e/matmul/fused_mat_mul/op_kernel/fused_mat_mul.cpp
    title: FusedMatMul entry and dispatch
  - id: tiling-key
    resource: https://gitcode.com/cann/ops-nn/blob/39a50f12554f00809f09eaf0b8a0675477879a4e/matmul/fused_mat_mul/op_kernel/arch35/fused_mat_mul_tilingkey.h
    title: matmul and epilogue tiling-key selection
  - id: public-key
    resource: https://gitcode.com/cann/ops-nn/blob/39a50f12554f00809f09eaf0b8a0675477879a4e/matmul/fused_mat_mul/op_kernel/arch35/fused_mat_mul_tiling_key_public.h
    title: fused-operation and batch-model identifiers
  - id: gelu
    resource: https://gitcode.com/cann/ops-nn/blob/39a50f12554f00809f09eaf0b8a0675477879a4e/matmul/fused_mat_mul/op_kernel/arch35/fused_mat_mul_gelu_basic_cmct.h
    title: GELU post-processing pipeline
  - id: zero-k
    resource: https://gitcode.com/cann/ops-nn/blob/39a50f12554f00809f09eaf0b8a0675477879a4e/matmul/fused_mat_mul/op_kernel/arch35/fused_mat_mul_input_k_eq_zero_copy_x3.h
    title: zero-K binary-operation semantics
operator_families: [matmul, fused-matmul]
---

# 接口与概念

## 算子算法

FusedMatMul 计算 `Y = OP(X1 @ X2 + bias, X3)`，把矩阵乘 epilogue 与后续逐元素操作合并。
`OP` 可为空、`16cast32`、add、mul、GELU(erf)、GELU(tanh) 或 ReLU；X3 只由二元 add/mul
使用。X1/X2 支持 FP16、BF16、FP32，`16cast32` 输出 FP32，其余输出组合按接口约束选择。[^guide][^public-key]

# 用法

## 分核策略与基本块切分

入口将 API level、转置、batch 模型、Matmul 调度模型、全载模型、输出模型和融合操作编码到
tiling key。无后处理、ReLU 和 cast 支持 Basic 及部分全载路径；add/mul 还覆盖 Stream-K、
`K == 0`、batch 与 Fixpipe/on-the-fly 输出；GELU 使用专用 Basic CMCT kernel。[^entry][^tiling-key]

矩阵乘仍以 `(M,N)` Cube 基本块分核并沿 K 迭代。二元操作要求 X3 tile 与输出 tile 使用同一
batch/M/N 映射；`K == 0` 不执行 MMAD，add/mul 的专用分支直接按定义处理 bias/X3，而不是
统一清零。[^entry][^zero-k]

# 代码模式

## 数据路径与存储层级

```text
X1/X2 GM -> L1 -> L0A/L0B -> MMAD -> L0C
bias GM -----------------------------> matmul epilogue
X3 GM -> UB --------------------------> add/mul
L0C -> Fixpipe/UB -> cast/ReLU/GELU/binary op -> Y GM
```

简单 cast/ReLU 可在 Matmul 输出阶段完成；add/mul 需要搬运与当前输出块对应的 X3；GELU
专用 kernel 将 Cube 结果交给 Vector 计算 erf 或 tanh 近似。融合避免把完整中间矩阵写回 GM，
但仍可能使用按 tile 的 workspace 作为 AIC/AIV 交接区。[^entry][^gelu]

## 流水排布、同步关系与数值精度

Basic 路径流水化 A/B 搬运、MMAD 与输出。需要 Vector 后处理时，AIC 生产 L0C/Fixpipe tile，
AIV 消费 tile 并写回，双方通过事件或 cross-core flag 保护 ping-pong buffer；GELU CMCT 路径
明确把矩阵乘与非线性阶段串接。[^entry][^gelu]

FP16/BF16 矩阵乘通常使用 FP32 累加，bias 与非线性在较高精度中组合后再转换为输出类型。
`16cast32` 必须保留 FP32 输出；GELU 的 erf/tanh 近似是不同的 tiling 分支，验证容差也应分别
制定。[^guide][^public-key]

# 约束

- 仅 Ascend 950 路径支持该融合接口。
- X1、X2 的 K 维及 batch 广播必须兼容；bias 和 X3 shape 必须能映射到输出 tile。
- X3 只在 add/mul 分支有效，不应在一元或空后处理分支参与地址计算。
- 融合操作字符串、dtype 组合和输出 dtype 必须与接口枚举一致。
- Stream-K、Fixpipe 和 GELU workspace 的容量及同步元数据必须与 host tiling 一致。

# 失败表现

- Matmul 正确但融合结果错：操作枚举映射、bias 顺序或 X3 tile offset 错误。
- 仅 batch add/mul 错误：X3 batch 广播 stride 没有跟随输出调度。
- GELU 在大绝对值区间偏差异常：erf/tanh 分支混用或中间精度过早降低。
- `K == 0` 的 add/mul 错误：错误复用普通清零路径，丢失 bias 或 X3 语义。
- mixed 路径偶发挂起：AIC/AIV buffer 生产、消费和释放 flag 不配对。

# 验证方法

以 `matmul + bias` 后分别执行每种 OP 的 FP32 reference，覆盖 FP16/BF16/FP32、四种转置、
batch 广播、bias 有无、X3 广播、M/N/K 尾块和 `K == 0`。分别命中 Basic、全载、Stream-K、
Fixpipe 和 GELU CMCT，并单测 erf/tanh 两种 GELU；检查融合结果和未融合 reference 的一致性。
性能结论须在空闲 Ascend 950 NPU 上另行 benchmark/profile。[^guide][^entry][^zero-k]

[^guide]: 固定提交中的融合公式、操作集合、dtype、shape 和设备支持范围。
[^entry]: 固定提交中的操作、调度、batch、全载、Stream-K 与零 K 分派。
[^tiling-key]: 固定提交中的 API level、batch、模型和输出路径 tiling 轴。
[^public-key]: 固定提交中的融合操作及 batch 模型标识。
[^gelu]: 固定提交中的 Cube/Vector GELU 后处理与同步流水。
[^zero-k]: 固定提交中的零 K add/mul 与 X3 处理语义。
