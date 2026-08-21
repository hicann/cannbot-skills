---
type: CATLASS DSL Operator Example
title: Flash Attention Inference
description: Ascend950 Prefill Flash Attention 的 Online Softmax、GQA、tiling、流水和验证边界。
tags: [catlass-dsl, operator, flash-attention, attention, prefill, online-softmax]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
sources:
  - id: kernel
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/flash_attention_infer/flash_attention_infer.py
    title: Flash Attention inference kernel and host validation
  - id: tiling
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/flash_attention_infer/fa_tiling.py
    title: Flash Attention tiling data implementation
  - id: guide
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/flash_attention_infer/README.md
    title: Flash Attention inference guide
operator_families: [flash-attention, attention]
arch: [c310]
---

# 接口与概念

## 算子算法

该样例实现 Prefill Flash Attention：以 Q/KV block 做 `QK^T`，用 Online Softmax
维护逐行 max 与 sum，再计算 `P @ V`，避免在 GM 物化完整注意力矩阵。数据流组合
Cube MMAD、Vector softmax/归一化和 L0C→UB、UB→L1 的片上通路。当前实现支持
GQA，并使用 Cube 相对 Vector 的 prelaunch 流水。[^kernel][^guide]

# 用法

## 分核策略与基本块切分

任务空间是 `batch × query_head × query_block`，通过
`tla.range(block_idx, total_tasks, block_num)` 做 grid-stride。每个任务固定一个
Q/head，遍历全部 KV block；GQA 用 `kv_head_idx = head_idx // group_size`。Q/KV
基本块均为 128，两个 AIV 用 `sub_block_idx()` 沿 Q 维各处理最多 64 行。[^kernel][^tiling]

kernel ABI 包含 Q/K/V/O、mask、tiling data 和 Q/KV actual sequence length：

```python
@tla.kernel
def flash_attention_infer_kernel(
    mem_q: tla.Tensor,
    mem_k: tla.Tensor,
    mem_v: tla.Tensor,
    mem_o: tla.Tensor,
    mem_mask: tla.Tensor,
    tiling_data: tla.Tensor,
    actual_q_seqlen: tla.Tensor,
    actual_kv_seqlen: tla.Tensor,
):
    ...
```

Host 将 BSND Q/K reshape 为 `[-1, HEAD_DIM]`；K 使用 ColumnMajor 适配转置，
Q/V/O 使用 RowMajor。tiling data 由独立模块打包，actual sequence length 使用以 0
开头的 batch 前缀和。[^tiling][^guide]

# 代码模式

## 数据路径与存储层级

```text
Q/K GM -> L1 -> L0A/L0B -> QK MMAD -> L0C(S fp32)
      -> FIX SPLIT_M -> UB S -> Vector Online Softmax -> UB P(fp16,zNUnAlign)
      -> L1 P + L1 V -> L0A/L0B -> PV MMAD -> L0C(Otmp fp32)
      -> FIX SPLIT_M -> UB Otmp -> Vector rescale/normalize -> GM O
```

Q 的 L1 buffer 为单份，K/V 为双 buffer，P 为三 buffer；QK/PV 的 L0C 和 UB
中间结果分别使用 ping/pong。这样 S、P、PV 均不需要落回 GM。[^kernel][^guide]

Online Softmax 每个 KV block 更新状态：

```text
m_new = max(m, rowmax(S_ij))
P_ij  = exp(S_ij - m_new)
O     = O * exp(m - m_new) + P_ij @ V_j
l     = l * exp(m - m_new) + rowsum(P_ij)
O_out = O / l
```

当前固定参数以 `HEAD_DIM=128`、`Q_BLOCK=128`、`KV_BLOCK=128` 为核心，输入输出
为 FP16，中间分数与累加使用 FP32。生成实现中，固定 block、dtype 和 prelaunch 值应
在 kernel 函数内定义；batch、head、sequence 等实际尺寸通过 tensor metadata、tiling
data 或显式 kernel 参数传入。所有编译期特化复用同一个 kernel，并由形式参数类型或
metadata 生成不同编译产物。[^kernel]

## 流水排布、同步关系与数值精度

`PRE_LAUNCH=2` 让 QK/softmax 比 PV 提前两个 KV block，循环上界额外增加两个迭代
完成流水排空。三组 mode-4 cross flag 分别交接 QK→softmax、softmax/P→PV、
PV→rescale，并为 AIV0/AIV1 单独传 `aiv_id`；同核 L1/L0/CUBE/FIX/MTE3 由普通
flag 管理 buffer 所有权。[^kernel]

Q/K/V/P 为 f16；QK score、row max/sum、PV accumulator 和 rescale 状态为 f32；
P 在写 L1 前由 f32 cast 为 f16，最终 O 在归一化后 cast 为 f16。[^kernel][^guide]

# 约束

- 当前只支持编译期常量 shape，不支持同一 artifact 的动态 shape 复用。
- mask 参数目前仅占位并要求全 0，不支持通用 0/1 mask。
- 暂不支持 PagedAttention 或分页 KV cache。
- 当前 dtype 为 FP16 输入输出、FP32 中间计算，`HEAD_DIM=128`。
- `KV_HEAD_NUM <= HEAD_NUM` 且 GQA group 映射必须整除并与 reference 一致。
- kernel 不得从模块级变量或 closure 读取 block、shape、dtype、prelaunch 等配置，也不得
  由 host 在编译前改写这些隐藏状态。
- 不得为 shape、dtype、layout、block 或 prelaunch 变体声明独立 kernel；同一 solution
  只保留一个 `@tla.kernel`。

# 失败表现

- O 保持 sentinel：block/tiling 映射、最终回写或跨流水同步未执行。
- 某些 query 行整行错误：actual sequence 前缀和或 Q block 任务映射错误。
- 数值溢出/NaN：Online Softmax 的 max rescale、sum 或 FP32 状态维护错误。
- GQA head 串数据：`kv_head_idx = head_idx // group_size` 或 K/V offset 错误。
- 修改 shape 后复用旧 artifact：编译期常量与实际 buffer 不一致。
- 不同 case 的 TLAIR 随执行顺序变化：kernel 的编译期配置来自可变全局状态。

# 验证方法

先检查 TLAIR 中 QK/PV MMAD、Vector softmax 和跨流水同步，再执行 host reference；
要求 O 不再是 sentinel 且误差低于样例阈值。case 至少覆盖 batch>1、GQA、Q/KV tail
和多个 KV block；源码审查要求 decorated kernel 除 `tla` 和 Python 内建符号外没有自由
名字。性能需要在空闲 NPU 上分别 benchmark/profile，源码流水结构本身不构成性能结论。
[^kernel][^guide]

[^kernel]: 固定提交 Flash Attention kernel、host reference、sentinel 与误差校验入口。
[^tiling]: 固定提交 tiling data 和 actual sequence metadata 的打包实现。
[^guide]: 固定提交对算法、支持特性、限制和运行方法的说明。
