---
type: CATLASS DSL Operator Example
title: MatMul
description: Ascend 950 MatMul 的调度分派、基本块切分、Stream-K、全载与 Fixpipe 数据路径。
tags: [catlass-dsl, operator, matmul, stream-k, fixpipe, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T12:29:38Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T12:29:38Z'}
sources:
  - id: guide
    resource: https://gitcode.com/cann/ops-nn/blob/39a50f12554f00809f09eaf0b8a0675477879a4e/matmul/mat_mul_v3/README.md
    title: MatMul interface and capability guide
  - id: entry
    resource: https://gitcode.com/cann/ops-nn/blob/39a50f12554f00809f09eaf0b8a0675477879a4e/matmul/mat_mul_v3/op_kernel/arch35/mat_mul_v3.cpp
    title: MatMul kernel entry and dispatch
  - id: scheduler
    resource: https://gitcode.com/cann/ops-nn/blob/39a50f12554f00809f09eaf0b8a0675477879a4e/matmul/mat_mul_v3/op_kernel/arch35/block_scheduler_aswt.h
    title: adaptive basic-block scheduler
  - id: streamk
    resource: https://gitcode.com/cann/ops-nn/blob/39a50f12554f00809f09eaf0b8a0675477879a4e/matmul/mat_mul_v3/op_kernel/arch35/mat_mul_stream_k_kernel.h
    title: Stream-K execution path
  - id: fixpipe
    resource: https://gitcode.com/cann/ops-nn/blob/39a50f12554f00809f09eaf0b8a0675477879a4e/matmul/mat_mul_v3/op_kernel/arch35/mat_mul_fixpipe_opti.h
    title: Fixpipe and mixed-core output path
  - id: full-load
    resource: https://gitcode.com/cann/ops-nn/blob/39a50f12554f00809f09eaf0b8a0675477879a4e/matmul/mat_mul_v3/op_kernel/arch35/mat_mul_full_load.h
    title: L1 full-load kernels
operator_families: [matmul]
---

# 接口与概念

## 算子算法

MatMul 计算 `C = op(A) @ op(B) + bias`，其中 `op` 可在编译期选择转置。二维输入的逻辑形状为
`A(M,K)`、`B(K,N)`、`C(M,N)`，bias 为 `(N)` 或 `(1,N)`；入口也根据 batch 模式处理批量
矩阵。输入支持 ND，B 在 Ascend 950 路径还可使用 FRACTAL_NZ。[^guide][^entry]

# 用法

## 分核策略与基本块切分

入口以转置、batch 模式、调度模型、L1 全载对象和输出模型为编译期轴，选择 Basic、ASWT、
Stream-K、split-K、全载或 Fixpipe 路径。Basic/ASWT 将 `(M,N)` 划分为 Cube 基本块，并在每块内
沿 K 迭代；ASWT 根据 tiling 给出的块数和尾块形状动态领取任务。Stream-K 把 K 迭代而非仅
输出块分发给核，使小 M/N、大 K 的负载更均衡；多个核贡献同一输出块时通过 workspace 或
atomic 路径归并。[^entry][^scheduler][^streamk]

当 A、B 或二者可完整驻留 L1 时，全载内核先搬入可复用操作数，再遍历另一维的输出块，减少
重复 GM 读取。`K == 0` 使用独立清零分支，不进入 MMAD。[^entry][^full-load]

# 代码模式

## 数据路径与存储层级

```text
A/B GM -> L1(A1/B1) -> L0A/L0B -> MMAD -> L0C
bias GM -------------------------------> Matmul epilogue
L0C -> Fixpipe/UB cast or layout conversion -> C GM
```

普通路径由 Matmul API 管理 L1/L0 搬运和 K 循环。全载路径显式用 A1/B1 队列保存复用矩阵；
Stream-K 将部分和写入 workspace 或使用原子累加。需要行主序对齐修正或双目的输出时，Cube
把 L0C 结果送入 Fixpipe/UB，Vector 核再完成转换并写回。[^streamk][^fixpipe][^full-load]

## 流水排布、同步关系与数值精度

Basic、ASWT、全载和常规 Stream-K 主要在 AIC 上执行。Fixpipe 优化路径可采用一个 AIC 配一个
或两个 AIV：AIC 完成 MMAD 后设置 cross-core flag，AIV 消费对应输出缓冲，写回后反向释放，
从而让下一块 Cube 计算与当前块输出重叠。[^entry][^fixpipe]

FP16/BF16 输入通常在 Cube 中以 FP32 累加，再按输出 dtype 转换；FP32 输入的计算模式由 tiling
与 API level 决定。split-K 必须保证部分和的累加类型、首块初始化以及尾块写回协议一致。[^entry][^streamk]

# 约束

- A、B 的 K 维必须一致；转置标志改变逻辑维度而不是允许任意 stride。
- 非连续输入仅支持可识别的转置情形；其他 view 应先物化。
- 空张量可走专用分支，但空张量场景不支持 bias；`K == 0` 必须保留清零语义。
- FRACTAL_NZ 仅用于入口声明支持的操作数和 dtype，ND/NZ stride 不能混算。
- Stream-K、split-K 和 atomic 路径的 workspace 大小及初始化必须与 host tiling 一致。

# 失败表现

- 仅尾行或尾列错误：基本块尾尺寸或 ND/NZ 输出 stride 计算错误。
- 小 M/N、大 K 时结果偶发漂移：split-K 首块、atomic 初始化或部分和同步错误。
- 转置 case 错而普通 case 正常：逻辑 M/N/K 与物理 stride 的映射混淆。
- 双目的 Fixpipe 路径卡死：AIC/AIV flag 编号、目标核映射或释放方向不一致。
- 全载路径结果错位：驻留操作数的 L1 offset 在输出块迭代中被错误推进。

# 验证方法

用 FP32 reference 覆盖 A/B 四种转置组合、bias 有无、batch 广播、ND/NZ、M/N/K 尾块、
`K == 0` 和空张量。分别强制或构造 Basic、ASWT、Stream-K、split-K、全载及 Fixpipe 可命中的
shape，确认每个 tiling key 都被编译和执行；split-K 额外比较重复运行稳定性。性能结论须在空闲
Ascend 950 NPU 上单独 benchmark/profile，本 concept 只记录固定源码结构。[^guide][^entry]

[^guide]: 固定提交中的公式、shape、dtype、layout、空张量和非连续输入约束。
[^entry]: 固定提交中的编译期分派轴、任务类型、零 K 与各执行模型入口。
[^scheduler]: 固定提交中的 ASWT 输出基本块领取与尾块调度。
[^streamk]: 固定提交中的 Stream-K、split-K、workspace 和部分和归并流程。
[^fixpipe]: 固定提交中的 AIC/AIV 输出协同、cross-core flag 与行主序写回。
[^full-load]: 固定提交中的 A/B/AB L1 全载和操作数复用流程。
