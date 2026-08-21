---
type: CATLASS DSL Operator Example
title: QuantBatchMatmul
description: Ascend 950 QuantBatchMatmul 的量化模式分派、AIC/AIV 协同、尺度数据路径与精度约束。
tags: [catlass-dsl, operator, matmul, quantization, fp8, mx, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T12:29:38Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T12:29:38Z'}
sources:
  - id: guide
    resource: https://gitcode.com/cann/ops-nn/blob/39a50f12554f00809f09eaf0b8a0675477879a4e/matmul/quant_batch_matmul_v4/README.md
    title: QuantBatchMatmul interface and quantization guide
  - id: entry
    resource: https://gitcode.com/cann/ops-nn/blob/39a50f12554f00809f09eaf0b8a0675477879a4e/matmul/quant_batch_matmul_v4/op_kernel/arch35/quant_batch_matmul_v4.cpp
    title: QuantBatchMatmul kernel entry and mode dispatch
  - id: tiling-key
    resource: https://gitcode.com/cann/ops-nn/blob/39a50f12554f00809f09eaf0b8a0675477879a4e/matmul/quant_batch_matmul_v4/op_kernel/arch35/quant_batch_matmul_v4_tiling_key.h
    title: quantization and kernel-template selection
  - id: common
    resource: https://gitcode.com/cann/ops-nn/blob/39a50f12554f00809f09eaf0b8a0675477879a4e/matmul/quant_batch_matmul_v4/op_kernel/arch35/quant_batch_matmul_v4_reg_base_common.h
    title: register-base quantized matmul pipeline
  - id: per-group
    resource: https://gitcode.com/cann/ops-nn/blob/39a50f12554f00809f09eaf0b8a0675477879a4e/matmul/quant_batch_matmul_v4/op_kernel/arch35/quant_batch_matmul_v4_pertoken_pergroup.h
    title: per-token and per-group scale path
  - id: mx
    resource: https://gitcode.com/cann/ops-nn/blob/39a50f12554f00809f09eaf0b8a0675477879a4e/matmul/quant_batch_matmul_v4/op_kernel/arch35/quant_batch_matmul_v4_weight_quant_mx_blaze.h
    title: microscaling weight-quantized path
operator_families: [matmul, quantized-matmul]
---

# 接口与概念

## 算子算法

QuantBatchMatmul 对矩阵或批量矩阵执行量化乘法，并按所选模式应用输入 scale、weight scale、
可选 offset、bias 和输出 scale。Ascend 950 实现覆盖 per-tensor、per-channel、per-group、
per-tile、MX 和非对称 INT4 等分支，也支持 FP8/HIFLOAT8 与 FP4 权重组合。具体公式和 scale
shape 由量化模式与输入 dtype 共同确定，不能只凭同名参数互换。[^guide][^tiling-key]

# 用法

## 分核策略与基本块切分

入口先按 A/B 转置、量化模式、可选属性、weight layout 和 kernel template 选择实例。输出空间按
batch、M、N 划分，块内沿 K 迭代；per-group、per-tile 和 MX 路径还要让 K/N 基本块边界与
scale block 对齐。固定点的部分场景走 AIC-only 的 ASW 或 AL1 全载内核；需要在线反量化或
预处理的场景使用 AIC:AIV 为 1:2 的 mixed kernel。[^entry][^tiling-key]

MX 权重量化路径使用 tail-split 调度，让尾部不规则任务与常规块分开领取。非对称 INT4 路径先由
AIV 将权重转换到计算布局并写入 workspace，全核同步后再开始矩阵乘。[^entry][^mx]

# 代码模式

## 数据路径与存储层级

```text
A/B + scales/offsets GM
  -> AIV UB 解码、反量化或重排 -> L1/workspace
  -> AIC L1 -> L0A/L0B -> MMAD -> L0C
bias/output scale -> epilogue -> Y GM
```

register-base 路径让 AIV 按当前 K/N tile 搬运 scale 和量化权重，在 UB 中完成解码、缩放或布局
转换，再把可由 Cube 消费的数据交给 L1；AIC 同时计算上一 tile。per-token/per-group 路径分别
沿 M 与 K 选择 scale，MX 路径将矩阵块与共享指数块成对调度。[^common][^per-group][^mx]

## 流水排布、同步关系与数值精度

mixed kernel 用 ping-pong buffer 重叠 AIV 预处理和 AIC MMAD，并以 hard event 管理 UB 搬运，
以 cross-core flag 管理 L1 数据的生产与消费。INT4 预处理阶段结束后执行全核同步，避免 Cube
读取尚未转换的 workspace。[^entry][^common]

低比特输入先按对应 scale 还原到计算路径，矩阵累加和 bias 处理使用入口规定的高精度类型，再
转换为 FP16、BF16 或 FP32 输出。scale 的数据类型、分组范围和广播维度是数值语义的一部分；
改变分组边界会改变结果，而不只是性能。[^guide][^per-group]

# 约束

- 输入 dtype、scale dtype、bias dtype、输出 dtype 必须采用接口列出的合法组合。
- scale/offset 的 shape 必须与 per-tensor、per-channel、per-group、per-tile 或 MX 模式严格匹配。
- 不支持空张量；M、N、K 或 batch 中的零维不能落入普通内核。
- 非连续输入仅支持可识别的转置；weight ND/NZ 由独立 tiling key 选择。
- INT4 workspace、MX block 和 group size 必须与 host tiling 的对齐与容量计算一致。

# 失败表现

- 数值按固定列或 K 区间成片错误：scale block、group 索引或尾 group 边界错误。
- 仅 batch>1 错误：batch 广播规则或 scale 的 batch stride 错误。
- INT4 偶发错误或挂起：预处理 workspace 未完成全核同步，或 AIC/AIV flag 复用过早。
- MX 仅尾块错误：共享指数块与矩阵 tail-split 任务没有保持同一映射。
- ND 正常而 NZ 错误：weight layout tiling key 与实际地址换算不一致。

# 验证方法

为每种量化模式建立先反量化再 FP32 matmul 的 reference，并覆盖转置、batch 广播、bias、ND/NZ、
完整与尾 group/tile、不同 scale dtype 和输出 dtype。单独验证 INT4 非对称 offset 与预处理、MX
共享指数、FP8/FP4 极值和饱和边界；重复运行 mixed kernel 以捕获同步问题。性能结论须在空闲
Ascend 950 NPU 上另行 benchmark/profile。[^guide][^entry][^tiling-key]

[^guide]: 固定提交中的量化公式、dtype 组合、shape、layout 和接口限制。
[^entry]: 固定提交中的量化模式分派、AIC/AIV 比例、INT4 预处理与任务类型。
[^tiling-key]: 固定提交中的转置、量化类型、可选属性、weight layout 与模板选择轴。
[^common]: 固定提交中的 register-base 搬运、解码、ping-pong 和跨核同步实现。
[^per-group]: 固定提交中的 per-token/per-group scale 定位与基本块处理。
[^mx]: 固定提交中的 MX 输入配对、共享指数和 tail-split 调度。
