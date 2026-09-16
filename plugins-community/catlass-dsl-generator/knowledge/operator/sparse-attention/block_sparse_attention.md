---
type: CATLASS DSL Operator Example
title: Block Sparse Attention (BSA)
description: Ascend 950 BSA 从零生成时强制采用 mixed Cube/Vector 流水，逐阶段对照 AscendC regular 实现，并审计批准接口与骨架 ABI。
tags: [catlass-dsl, operator, sparse-attention, block-sparse-attention, online-softmax, gqa, ascend-950, mixed-cube-vector, generation-gate, interface-contract, abi, skeleton-audit]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-09-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-09-10T00:00:00Z'}
  - {by: process:catlass-dsl-generation-gate-audit, at: '2026-09-10T06:10:21Z'}
  - {by: process:catlass-dsl-knowledge-review, at: '2026-09-11T00:00:00Z'}
sources:
  - id: guide
    resource: https://gitcode.com/cann/ops-transformer/blob/3f54e4334060281fad7b4b682393d22830928a33/attention/block_sparse_attention/README.md
    title: Block Sparse Attention algorithm and interface guide
  - id: entry
    resource: https://gitcode.com/cann/ops-transformer/blob/3f54e4334060281fad7b4b682393d22830928a33/attention/block_sparse_attention/op_kernel/block_sparse_attention_kernel_interface.cpp
    title: kernel entry and implementation selection
  - id: kernel
    resource: https://gitcode.com/cann/ops-transformer/blob/3f54e4334060281fad7b4b682393d22830928a33/attention/block_sparse_attention/op_kernel/arch35/block_sparse_attention_kernel_arch35_regular.h
    title: Ascend 950 regular BSA pipeline orchestration
  - id: qk
    resource: https://gitcode.com/cann/ops-transformer/blob/3f54e4334060281fad7b4b682393d22830928a33/attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/bsa_block_mmad_qk_arch35_ABf16_C_to_UB.hpp
    title: sparse gathered QK block matmul
  - id: pv
    resource: https://gitcode.com/cann/ops-transformer/blob/3f54e4334060281fad7b4b682393d22830928a33/attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/bsa_block_mmad_pv_arch35_ABf16_C_to_UB.hpp
    title: sparse gathered PV block matmul
  - id: softmax
    resource: https://gitcode.com/cann/ops-transformer/blob/3f54e4334060281fad7b4b682393d22830928a33/attention/block_sparse_attention/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax_arch35_reg_high_prec_S_trans.hpp
    title: high-precision BSA online softmax epilogue
  - id: rescale
    resource: https://gitcode.com/cann/ops-transformer/blob/3f54e4334060281fad7b4b682393d22830928a33/attention/block_sparse_attention/op_kernel/attn_infra/epilogue/block/bsa_block_epilogue_rescale_o_arch35_reg_high_prec.hpp
    title: high-precision BSA output rescale epilogue
operator_families: [sparse-attention, block-sparse-attention]
arch: [c310]
---

# 接口与概念

## 算子算法

BSA 对每个 batch、Q head 和 Q block 读取 `blockSparseMask` 的一行；其中非零项选择对应的
KV block。选中块按索引顺序逻辑拼接为 gathered-KV 序列，再计算
`S = scaleValue * Q @ K^T`、对 gathered-KV 轴做 softmax、以及 `O = P @ V`。输出保持 Q 的
shape 和输入 dtype；可选 LSE 为每个有效 query row 的 `log(sum(exp(S)))`。mask 的逻辑 shape
是 `[B, Nq, ceil(maxSq / blockShapeX), ceil(maxSkv / blockShapeY)]`。[^guide]

Q head 到 KV head 的映射为 `kvHead = qHead / (Nq / Nkv)`，因此 `Nq` 必须能被 `Nkv` 整除。
同一 Q block 的不同 Q head 可以使用不同稀疏模式，任务不可沿 head 合并。有效 Q/KV 长度决定
尾 Q block、尾 KV block 及 TND batch 前缀；越出有效长度的 padded query row 不参与计算。[^kernel]

路径映射如下：

- `__CCE_AICORE__ == 310`、常规 FP16/BF16 输入走 `BsaInferIntfRegular`，QK、softmax、PV、
  rescale 分别由独立 block service 串成 mixed Cube/Vector 流水；本条目只描述该路径。[^entry]
- QK 可由 tiling 选择普通或转置 MM1 表示；两者必须保持相同的 logical score、mask 索引顺序
  和 softmax 归一化轴。[^entry][^softmax]
- 入口还存在量化专用模板，但其量化 scale、数据格式与 epilogue 不属于这里的 regular 路径，
  不能直接套用本条目的精度和 workspace 假设。[^entry]

## 从零生成的强制架构门禁

当 workload 命中本条目的 c310 regular FP16/BF16 路径时，生成器**必须从第一版候选开始**采用
`Cube QK -> Vector Online Softmax -> Cube PV -> Vector Rescale/Cast` 的单个 mixed kernel。
这不是可延后尝试的性能优化项，而是该算子实现路径的架构基线：固定 AscendC regular kernel
本身就是由 QK、softmax、PV、rescale block service 组成的 AIC/AIV 流水。[^entry][^kernel]

以下规则是生成与评审的硬门禁：

1. QK 点积必须落到 Cube `mmad` 数据路径，经过 GM→L1→L0A/L0B→FP32 L0C→UB；不得用
   SIMT/SIMD 标量循环逐元素完成 `Q @ K^T`。[^qk]
2. PV 必须落到 Cube `mmad` 数据路径，P 在进入 PV 前按输入 dtype 转换并经 UB/L1→L0A，V 经
   GM/L1→L0B，结果在 FP32 L0C/UB 累加；不得用逐 key、逐输出维的标量 FMA 作为主实现。[^pv]
3. block mask 压缩/索引、Online Softmax、跨 tile max/sum 更新、旧 O 重标定、最终除法与 cast
   必须由 Vector 侧承担，并与 Cube 通过显式 cross-core flag 交接所有权。[^kernel][^softmax][^rescale]
4. 标量代码只允许处理地址、索引、循环控制、tail 条件和少量标量元数据；若源码中出现以
   `headDim` 或 gathered-KV token 为归约轴的标量 QK/PV 累加循环，该候选直接判定为架构不合格，
   不得作为 baseline、不得进入性能优化轮次。
5. 若当前 DSL 能力或编译问题暂时阻止 mixed kernel，必须把状态报告为“mixed 路径未完成/阻塞”，
   并围绕 layout、Tensor 捕获、cross-core 同步或后端编译问题继续修复；不得静默退回纯标量实现
   并把“正确可运行”当成 BSA 初版交付。

只有 workload 明确落在 c310 regular 路径之外，且 definition 或设计文档批准了 fallback，才允许
另建标量调试实现；该实现必须标记为 debug-only，不能参与与 regular AscendC/reference 的性能结论。

# 用法

## 生成前必须完成的 AscendC 对照审计

在起草 CATLASS DSL kernel 前，必须完整检查 frontmatter 固定提交中的 `entry`、`kernel`、`qk`、
`softmax`、`pv`、`rescale` 六份 AscendC 来源，并在设计记录或首轮 proposal 中产出下面的逐阶段
映射。不能只阅读 README 或 Python reference 后直接编码。

| 阶段 | 必查 AscendC 来源 | CATLASS DSL 首版必须保留的结构 |
| --- | --- | --- |
| 路径选择 | `entry` | 确认 c310 regular/quantized、MM1 转置与否、dtype/layout；只实现实际命中的实例 |
| mask→稀疏索引 | `kernel` | AIV 压缩 `sparseIdx/sparseCount`，或在受限 workload 中保留等价的块索引遍历；主计算不得退化成稠密标量扫描 |
| QK | `qk` | AIC/Cube MMAD、Q 复用、稀疏 K gather、tail M/N/K、L0C→UB layout |
| Online Softmax | `softmax` | AIV/Vector FP32 row max/sum、P dtype 转换、首块/中间块/末块状态 |
| PV | `pv` | AIC/Cube MMAD、P/V 的相同 gathered 顺序、FP32 局部 O |
| Rescale/输出 | `rescale` | AIV/Vector 旧 O rescale、最终归一化、可选 LSE、cast 与 GM 回写 |
| 流水与所有权 | `kernel` | QK→softmax→PV→rescale 的 slot/ring、cross flag、event/barrier 和排空顺序 |

对照审计必须明确列出“保留、简化、暂不支持”的 AscendC 机制。允许针对已知 workload 简化
`mask2idx` workspace、buffer stage 数或多 batch 变长逻辑，但必须说明等价条件、资源变化和回退
条件；不允许删除 QK/PV Cube 阶段，也不允许把 Vector epilogue 改写成全量标量归约。AscendC
中的 sparseIdx 映射、Q 复用、P/V gathered 顺序和跨 tile数值状态分别由固定 QK、PV、softmax、
rescale 实现直接约束。[^qk][^pv][^softmax][^rescale]

生成器在提交首版前还必须给出四项可检查证据：

- 源码中存在一个 self-contained `@tla.kernel`，同时包含 `with tla.cube()` 与
  `with tla.vector()`，并且只启动一次；
- QK 和 PV 各至少存在一条 `tla.mmad`，对应的 L1/L0/L0C/UB 或 GM 数据路径可从源码追踪；
- softmax 的 max/sum 与 rescale 使用 FP32，P 转换边界与 AscendC regular 路径一致；
- 至少用一个非连续 sparse pattern 和一个 Q/KV tail case 验证 QK/PV 使用相同的 gathered 顺序。

任一项缺失时，首版状态只能是“未完成”，不能进入以纯标量候选为起点的渐进性能优化。

## 分核策略与基本块切分

启动前由 AIV 将 uint8 block mask 压缩为 workspace 中的 `sparseIdx:int32` 和
`sparseCount:int32`，随后全核同步。主任务粒度为一个 `(batch, qHead, qBaseTile)`；task 以
`coreIdx` 为起点、按 `coreNum` 跨步分发。`qBaseTile <= 128`，`kvBaseTile` 是 128 的倍数；
每个逻辑 Q block 可以包含多个 qBaseTile，实际尾块按有效序列长度裁剪并向 16 对齐供矩阵与
向量阶段使用。[^kernel]

对一个主任务，`sparseCount` 给出选中 KV block 数，最后一个选中块决定 gathered-KV 的真实尾长；
gathered-KV 再切为 `ceil(gatheredKvLength / kvBaseTile)` 个 tile。Q 只预载一次，QK/PV 的
L1/L0 基本模板均为 `128 x 128 x 128`，实际 M/N/K 由当前 Q/KV 尾 tile 收缩。[^entry][^kernel]

实现直接读取最后一个 `sparseIdx`，所以每个有效 `(B, qHead, qBlock)` 必须至少选择一个有效
KV block；索引需按递增 KV block 顺序压缩，否则 gathered 尾长和逻辑拼接顺序会失配。该条件
也与功能 reference 中“空选择为非法”的语义一致。[^kernel][^qk][^pv]

README 仅列出 TND/BNSD，而固定 regular kernel 还包含 BSND 的 stride 分支；这是文档与实现的
覆盖差异。优化或生成代码时，只有在 host tiling/入口确实实例化 BSND 后才启用该分支，不能仅凭
kernel 内存在代码就宣称端到端接口已支持。[^guide][^kernel]

# 代码模式

## BSA 批准接口到 Kernel ABI 的硬约束

本节是 BSA 从零生成与骨架复用的专项准入规则，不是所有算子的通用编译限制。

批准的设计、definition 与 reference 共同定义 BSA 算子的接口；复用的样例或其他算子骨架
只能提供实现素材，不能反向改变接口。实现前必须建立并维护逐项映射：

```text
批准的公开参数/输出 -> wrapper 中的绑定或校验 -> kernel 形参或明确的编译期派生值
```

- 每个公开 tensor 输入和输出都必须有唯一、可解释的去向；不得漏传、换义或用无关名称承载。
- 每个 kernel tensor 形参必须反向映射到一个批准的输入、输出，或设计中明确批准的 workspace。
- 禁止为迁就旧骨架保留 dummy/placeholder tensor、重复传入同一对象、永不使用的形参、旧分支
  的 flag/metadata，以及与目标语义无关的名称。
- 公开 scalar 只有在设计允许固定特化时，才能在 wrapper 校验后变为 `Constexpr` 或 kernel
  内常量；映射表必须记录该转换，不能静默丢弃。
- `tla.compile` 的样本实参、compiled launch 实参和 kernel 形参必须在数量、顺序、结构、
  dtype 与语义上同时一致。运行时只检查到的结构一致不足以证明满足设计接口。
  编译与启动 API 见 [Kernel、编译与运行时](../../dsl/compile-and-runtime.md)。

采用骨架时，在第一次设备运行前删除所有未被批准接口使用的参数、分配、分支和元数据，
并把 kernel、形参和中间变量重命名为目标算子语义。不得为了兼容无关骨架而扩大目标 ABI。

## 数据路径与存储层级

```text
blockSparseMask uint8 GM
  -> AIV mask2idx
  -> sparseIdx int32 GM workspace + sparseCount int32 GM workspace

Q/K/V input GM
  -> Q/K/V L1 tiles -> L0A/L0B -> Cube FP32 L0C
  -> score FP32 UB (ping/pong)
  -> AIV online softmax: row max/sum FP32 UB, P cast to input dtype UB
  -> P L1 ring buffer + gathered V L1 -> L0A/L0B -> PV FP32 L0C/UB (ping/pong)
  -> FP32 running O/max/sum UB -> final divide, optional LSE, cast -> O GM
```

QK 和 PV 都通过 `sparseIdx` 把 gathered tile 区间映射回原始 KV block；跨 block 边界时拆成
多段 GM→L1 搬运，既不把完整 gathered K/V 落入 workspace，也不物化完整稠密 score。
QK service 复用已载入的 Q，按 L1/L0 tile 累加 embed 维；PV service 按相同压缩顺序消费 P 与
V，保证 softmax 列和 V 行一一对应。[^qk][^pv]

主要存储成本是：mask 压缩 workspace 约为每行 `yBlockNumAligned * 4 B` 的索引加 `4 B` 的
计数；UB 同时保留两份 score tile、两份 P tile、两份局部 O tile，以及每 query row 的
running max/sum/rescale 因子；L1 同时承载 Q/K tiles 和 `pL1BufNum` 份 P。具体字节数由
`qBaseTile`、`kvBaseTile`、`embed`、dtype 与 tiling buffer 数共同决定，修改 tile 或 stage 时
必须重新核算 UB/L1 上限与地址偏移。[^kernel]

## 流水排布、同步关系与数值精度

每个 gathered-KV tile 的依赖链是 `QK -> Online Softmax -> PV -> RescaleO`。外层循环额外执行
`PRE_LAUNCH` 次以排空流水：当前 tile 做 QK/softmax 时，延迟后的 tile 做 PV/rescale。
score 与局部 O 使用 2-slot ping/pong；P 使用 tiling 决定的 L1 ring。Cube↔Vector 分别以
`mm1ToSm`、`smToMm2`、`mm2ToRe` cross-core flag 交接所有权，槽复用还依赖 MTE/FIX/V 的
event；mask2idx 后有 `SyncAll`，kernel 结束前释放全部正反向 flag。[^kernel][^softmax][^rescale]

Online Softmax 维护跨 KV tile 的 FP32 running max 和 sum。后续 tile 产生更大 max 时，
`exp(oldMax-newMax)` 同时用于修正旧分母和旧 O；P 在进入 PV 前转换到输入 dtype，PV 结果与
running O 保持 FP32，最后一个 tile 才除以全局 sum、可选计算 LSE，并按 FP16/BF16 规则 cast
回输出。任何流水调整都必须保持 max/sum 与 O 使用同一 tile 的 rescale 因子。[^softmax][^rescale]

## 成本与观测

- 计算量近似随 `sum(selected valid KV tokens) * qRows * embed` 的两次矩阵乘增长；高密度 mask
  下 QK/PV Cube 时间首查，低密度且短序列时固定的 mask2idx、同步和尾 tile 开销占比会上升。
  [^kernel][^qk][^pv]
- 非连续稀疏索引会把 K/V 的 GM→L1 搬运拆成更多片段；若 profiler 显示 Cube 等待 MTE1、
  带宽利用率低而同密度连续块更快，首查 gathered block 的连续段数量和尾段比例。[^qk][^pv]
- `qBaseTile` 很小或大量 Q 尾块时，AIV 两个 sub-core 可能不均衡，且每 tile 固定 cross flag
  成本相对放大；表现为 Vector 等待/同步占比高、Cube 间歇空闲。[^softmax][^rescale]
- `kvSLoopNum == 1` 时流水没有稳态重叠，首尾专用分支和最终 cast/LSE 更突出；多 tile 时则要
  检查 P ring、两个 UB slot 或 rescale 依赖是否形成周期性回压。[^kernel][^rescale]

## 可验证优化候选

以下均为静态候选，不代表已经获得性能收益；每次只改变一个轴，并在同一 workload 下回退验证。

1. **先调 q/kv base tile 与 buffer 数。** 适用于 Cube 活跃率低且 profiler 呈现固定周期的
   MTE/flag 回压。修改 host tiling 的 `qBaseTile/kvBaseTile` 和 Q/K/V/P L1 buffer 数，预期
   Cube 空洞或 MTE 等待下降；代价是 UB/L1 占用、尾块浪费和寄存器压力上升。必须保持每个 task
   的唯一 O/LSE 所有权、两槽和 P ring 的生命周期以及相同 FP32 累加边界。若编译资源溢出、
   小尾块退化或 wait 时间转移而总时延不降，回退。[^kernel]
2. **合并连续 sparseIdx 的 K/V 搬运段。** 适用于选中块经常连续且 MTE1 等待高。只修改 QK/PV
   gathered-to-original 映射和 GM→L1 copy 切段，预期 copy 指令数、MTE1 stall 下降；代价是
   更复杂的边界判断和更长单次 DMA。必须保持索引顺序、最后有效 KV token、QK/PV 完全相同的
   分段映射。若非连续 pattern、尾块或乱序索引出现错误，或 copy 粒度变大却无时延收益，回退。
   [^qk][^pv]
3. **为单 KV tile 建立短流水分支。** 适用于 `kvSLoopNum == 1` 占主要 workload，且同步/epilogue
   时间超过 Cube。修改主循环与 first-and-last softmax/rescale 调度，预期 cross-core wait 和
   空转 stage 减少；代价是代码体积和分支维护成本。必须保留 QK→softmax→PV→最终归一化顺序、
   LSE 定义和 cast 模式。若多 tile 路径被扰动、数值误差扩大或短 shape 未稳定下降，回退。
   [^kernel][^softmax][^rescale]
4. **按 Q 尾长选择转置 MM1。** 适用于大量窄 Q tile 且 QK 尾块利用率低。只改变 tiling key/
   模板实例选择，预期 QK Cube 利用率提升或 FIX/UB layout 转换下降；代价是另一套 score/P layout
   和专用 softmax 路径。必须保持 logical score orientation、gathered 列顺序及 softmax 轴；若
   QK 单阶段变快但 softmax/PV 或端到端变慢，回退。[^entry][^softmax]

# 约束

- BSA 公开 wrapper、kernel ABI、编译样本和 launch 实参必须逐项符合批准接口映射；结构一致但
  语义不一致同样判定失败。
- 本条目的实现事实限定于固定提交的 c310 regular 路径；不得外推到 arch22 或量化入口。
- 对命中 c310 regular FP16/BF16 的从零生成任务，mixed Cube/Vector 是强制初始架构，不是可选
  优化候选；纯 SIMT/SIMD 标量 QK 或 PV 不得作为 baseline、交付实现或性能迭代起点。
- 开始编码前必须完成 AscendC 六源逐阶段对照；如果无法访问或理解其中任一关键实现，先报告
  取证缺口，不得仅凭 Python reference 推导硬件数据流。
- Q/K/V dtype 一致，为 FP16 或 BF16；Q/K/V head dim 一致，`Nq % Nkv == 0`。[^guide][^entry]
- `blockSparseMask` 必须提供，mask 每一有效行至少含一个非零项，且不得选择超过 actual KV
  length 的无效 block。`attentionMask` 和 paged-attention/block-table 不属于该 regular 语义。
  [^guide][^kernel]
- TND 必须提供每 batch 的 actual Q/KV length；非对齐尾长必须由 tail shape 处理，不能读取 padded
  token 参与 max/sum。[^guide][^kernel]
- blockShape 为正；接口约束中 c310 的 KV block shape 需满足 16 对齐。内部 `kvBaseTile` 仍由
  tiling 保证为 128 的倍数，两者不可混为同一参数。[^guide][^kernel]
- workspace 的 sparse index/count 大小和偏移必须与 host tiling 完全一致；修改 mask 压缩表示时
  需同步更新消费者和 workspace 计算。[^kernel]

# 失败表现

- BSA 公开接口正确但 kernel 仍包含无关 tensor 形参：接口审计失败；即使数值正确也不能准入。
- 同一 tensor 被重复塞入多个无关名称的占位参数，或形参只为满足旧骨架 arity：存在骨架残留。
- compile 和 launch 能成功，但参数无法逐项追溯到批准的 BSA 接口：只证明内部 ABI 自洽，不证明实现合规。
- 首版正确但 profiler 只有 AIV、`cube_utilization == 0`：错误采用了标量 QK/PV，违反生成架构
  门禁；不要继续做循环展开或标量 micro-tile，回退到 AscendC 阶段映射重建 mixed kernel。
- QK 使用 Cube、PV 仍为逐 key/逐维标量循环：只完成半条 mixed 路径；AIV IR/寄存器压力可能
  急剧膨胀，不能据此判断 Cube 方案无效，应继续把 PV 搬到 Cube 并保留 Vector rescale。
- 只根据 Python reference 编码、未记录 AscendC 对照：即使数值通过，也无法证明数据路径、
  精度边界和同步所有权与 c310 regular 实现一致，候选应在设计审查阶段拒绝。
- 某个 mask row 全零时越界或得到随机输出：`sparseCount == 0` 后仍读取最后一个索引。
- 非连续稀疏块结果像连续 KV：QK/PV 的 gathered offset 到原 block index 映射错误。
- GQA 中相邻 Q head 错用 K/V：`qHead -> kvHead` 分组或 layout stride 错误。
- 多 KV tile 时概率和不为 1、O 随 tile 顺序改变：running max/sum 或旧 O rescale 漏更新。
- 尾 KV block 数值异常：最后选中 block 的有效长度按 padded blockShape 计算。
- 偶发旧 score/P/O：cross flag 的 producer/consumer 或 ping/pong/ring 槽复用顺序被破坏。
- O 正确但 LSE 错位：TND/BNSD/BSND 的 LSE stride 或最终 `max + log(sum)` 回写格式错误。

# 验证方法

第一次设备运行前还要完成 BSA 静态接口审计：确认目标文件只有预期的 `@tla.kernel`，把 kernel
形参表与批准接口映射逐项比对，并统计每个 tensor 形参在函数体内的真实使用次数；使用次数
为零、只参与无效分支，或无批准来源的形参均阻断运行。wrapper 中新建的 tensor 也必须仅为
批准输出或批准 workspace。最后分别核对 compile 与 launch 的 arity、顺序和 dtype。可用
Python `ast` 提取形参与 `Name` 使用计数作为机械检查，但语义映射仍需对照设计文档审核。

先从编译日志/tiling dump 确认命中 c310 regular、layout、是否转置 MM1、q/kv base tile、L1 buffer
数和 LSE 模式，再进行数值与性能验证。功能 oracle 采用逐 Q block 收集 mask 选中 KV token、FP32
score/online-softmax 与 FP32 O 累加的 reference；P 在 PV 前按输入 dtype 量化，最终 O cast 回输入
dtype，与 kernel 的精度边界一致。[^entry][^softmax][^rescale]

从零生成时，数值测试之前先执行架构静态验收：确认同一个 kernel 中同时存在 Cube 与 Vector
region，QK/PV 均由 MMAD 承担，并逐项核对 AscendC 对照表。随后从 profiler 检查 AIC 与 AIV
均有实际活动、QK/PV 对应 Cube 工作且 single-kernel/single-launch 成立。`cube_utilization == 0`、
缺少任一 MMAD 或主归约仍位于标量循环时，即使 correctness 通过也必须失败，不进入 benchmark。

正确性矩阵至少覆盖：

- layout：TND、BNSD，以及只有端到端入口确认支持时的 BSND；
- dtype：FP16、BF16；MHA 与 `Nq/Nkv > 1` 的 GQA；
- mask：单块、多个连续块、多个非连续块、只选最后一个尾块、每行不同 pattern；
- shape：Q/KV 整块与非对齐尾块、`qBaseTile` 窄尾、单 KV tile 与多 KV tile、多个 batch 变长；
- 输出：O、可选 LSE、padded row 不被写入/保持约定值；全零 mask row 应在 host/reference 阶段拒绝；
- 变形关系：mask 全一等价同精度边界的 dense attention；对未选 KV token 修改 K/V 不改变输出；
  在保持索引顺序时拆分/合并连续 selected run 不改变结果。

性能候选使用相同设备、频率状态、shape、dtype、layout、mask pattern 和预热/采样策略比较；同时
记录端到端 kernel 时间、Cube/MTE/Vector 活跃及 cross-core wait。任何候选必须先通过完整矩阵，
再以多次采样确认变化；未 profile 的机制只能保留为候选，不能写成“已提速”。

[^guide]: 固定提交中的 BSA 数学语义、mask shape、dtype、layout 与接口约束。
[^entry]: 固定提交中的 c310 regular/quantized 路径、模板布局、基本 tile 与实现选择。
[^kernel]: 固定提交中的 mask 压缩、任务分发、GQA 映射、tail、buffer、流水和同步。
[^qk]: 固定提交中的 sparseIdx 到 K block 的映射及 QK L1/L0/FIX 数据路径。
[^pv]: 固定提交中的 sparseIdx 到 V block 的映射及 PV L1/L0/FIX 数据路径。
[^softmax]: 固定提交中的 FP32 online max/sum、P 转换、槽位和 QK→softmax→PV 同步。
[^rescale]: 固定提交中的跨 tile O 重标定、最终归一化、LSE、cast 与回写。
