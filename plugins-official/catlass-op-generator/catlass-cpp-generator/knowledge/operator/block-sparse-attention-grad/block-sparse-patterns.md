---
type: "operator"
title: "Block Sparse Attention Grad 模式"
description: "块稀疏注意力梯度的二值 block mask 语义、稀疏任务切分、反向计算链及 FP32 workspace 精度边界。"
tags: ["catlass-cpp", "block-sparse-attention-grad", "bsag", "pr1109"]
status: "stable"
generated: {"by": "process:catlass-cpp-pr1109-extraction", "at": "2026-09-23T00:00:00Z"}
verified: [{"by": "process:pr1109-source-audit", "at": "2026-09-23T00:00:00Z"}]
sources: [{"id": "bsag-guide", "resource": "git:cann/ops-transformer@adc5fd30003da4c73bc478cf42c6f031b491b973:attention/block_sparse_attention_grad/README.md", "title": "Block Sparse Attention Grad algorithm and interface guide", "kind": "repository"}, {"id": "bsag-entry", "resource": "git:cann/ops-transformer@adc5fd30003da4c73bc478cf42c6f031b491b973:attention/block_sparse_attention_grad/op_kernel/block_sparse_attention_grad.cpp", "title": "kernel entry and Arc35 tiling-key dispatch", "kind": "repository"}, {"id": "bsag-tiling", "resource": "git:cann/ops-transformer@adc5fd30003da4c73bc478cf42c6f031b491b973:attention/block_sparse_attention_grad/op_host/arch35/block_sparse_attention_grad_tiling_arch35.cpp", "title": "Arc35 capability checks tiling and workspace layout", "kind": "repository"}, {"id": "bsag-pipeline", "resource": "git:cann/ops-transformer@adc5fd30003da4c73bc478cf42c6f031b491b973:attention/block_sparse_attention_grad/op_kernel/arc35/block_sparse_attention_grad_arch35.h", "title": "Arc35 AIC and AIV pipeline orchestration", "kind": "repository"}, {"id": "bsag-scheduler", "resource": "git:cann/ops-transformer@adc5fd30003da4c73bc478cf42c6f031b491b973:attention/block_sparse_attention_grad/op_kernel/arc35/addr_compute.h", "title": "Arc35 sparse task generator and core assignment", "kind": "repository"}, {"id": "bsag-cube", "resource": "git:cann/ops-transformer@adc5fd30003da4c73bc478cf42c6f031b491b973:attention/block_sparse_attention_grad/op_kernel/arc35/cube_op_bsag_arch35.h", "title": "Arc35 five-MMAD Cube implementation", "kind": "repository"}, {"id": "bsag-vector", "resource": "git:cann/ops-transformer@adc5fd30003da4c73bc478cf42c6f031b491b973:attention/block_sparse_attention_grad/op_kernel/arc35/vec_op_bsag_arch35.h", "title": "Arc35 Softmax gradient and workspace vector implementation", "kind": "repository"}, {"id": "bsa-guide", "resource": "git:cann/ops-transformer@3f54e4334060281fad7b4b682393d22830928a33:attention/block_sparse_attention/README.md", "title": "Block Sparse Attention forward algorithm and interface guide", "kind": "repository"}, {"id": "select-guide", "resource": "git:cann/ops-transformer@90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e:attention/bsa_select_block_mask/README.md", "title": "BSA Select Block Mask algorithm and interface guide", "kind": "repository"}, {"id": "sink-guide", "resource": "git:cann/ops-transformer@2b67400cf8284717cb38f1d4ecdc69df9191943f:attention/flash_attn/README.md", "title": "FlashAttn forward sink definition in the softmax denominator", "kind": "repository"}]
operator_families: ["block-sparse-attention-grad"]
architectures: ["ascend950"]
---

# 接口与概念

块稀疏注意力用一个二值 block mask 决定哪些 `blockShapeX × blockShapeY` 的 Q/KV 块参与计算：
mask 非零即选中该 KV 块，零则整块跳过。它不提供 token 级 attention mask 路径，稀疏粒度就是
块粒度。[^bsag-guide][^bsa-guide]

本 concept 是 PR1109 固定来源的结构化入口，只记录算法、分核、数据路径、流水和精度边界等设计
输入。所有实现结论必须以本工程冻结的 contract、CPU 标杆和实际运行证据为准，本文不声称任何
精度、性能或 sanitizer 验收已通过。

## 算子算法

### 稀疏掩码 ABI

- mask 的逻辑 shape 为 `[B, Nq, ceil(maxSq / blockShapeX), ceil(maxSkv / blockShapeY)]`，
  元素为 UINT8/BOOL 二值。[^bsa-guide]
- 每个有效 mask 行至少含一个非零项；不得选择超过 actual KV length 的无效块；空选择非法。
  [^bsa-guide][^bsag-guide]
- Q head 到 KV head 的映射为 `kvHead = qHead / (Nq / Nkv)`，因此 `Nq % Nkv == 0`；同一 Q block
  的不同 Q head 可以使用不同稀疏模式，任务不可沿 head 合并。[^bsa-guide][^bsag-guide]
- 存在实现与文档的轴语义漂移：README 声明 mask shape 为 `[B, Nq, QBlocks, KVBlocks]`，而固定
  源码的线性 offset 为 `kvBlock * QBlocks + qBlock`，等价于读取 `[B, Nq, KVBlocks, QBlocks]`。
  全 1 或对称 mask 会掩盖该差异；非对称 pattern 在确认期望 ABI 前必须双向构造测试。
  [^bsag-guide][^bsag-entry]

### 前向 BSA

对每个 `(batch, qHead, qBlock)` 读取 mask 一行，非零项选中的 KV 块按索引顺序逻辑拼接为
gathered-KV 序列，再计算：[^bsa-guide]

```text
S   = scaleValue * Q @ K^T          # K 为 gathered-KV
P   = softmax(S, axis=gathered-KV)
O   = P @ V
LSE = log(sum_j exp(S_j))           # 可选，每有效 query 行
```

输出保持 Q 的 shape 与输入 dtype；越出有效长度的 padded query row 不参与计算。可选 LSE 是每
个有效 query row 的 `log(sum(exp(S)))`。[^bsa-guide]

BSA 前处理（select block mask）先做块级 mean pooling，再按 `score = scale * Qcmp @ Kcmp^T`
逐 Q block 对 K blocks 做 softmax，最后全局选择 `round(sparsity * Xblocks * Yblocks)` 个最大值
输出 INT8 二值 mask；部分压缩时每块均值分母取自 actual block length。[^select-guide]

### 反向 BSAG

反向接收 `dO/Q/K/V/O/LSE` 与二值 block mask，重新计算被选稀疏块的 score 和概率，不保存完整
前向概率。对一个有效 Q/KV tile，计算链为：[^bsag-guide][^bsag-pipeline][^bsag-vector]

```text
D  = rowsum(dO * O)
S  = Q @ K^T
P  = exp(scale * S - LSE)
dP = dO @ V^T
dS = P * (dP - D)
dV = P^T @ dO
dQ = scale * (dS @ K)
dK = scale * (dS^T @ Q)
```

`D` 先对全部有效 query 行计算一次并写入 workspace；主循环随后对每个有效稀疏 tile 发起
`QK`、`dO @ V^T`、`dV`、`dQ`、`dK` 共五次 MMAD。oracle 必须接收同一次前向产生的 O 和 LSE，
并允许“P/dS 低精度化后再 MMAD”的舍入边界。[^bsag-guide][^bsag-cube]

带 attention sink 的变体中，sink 是 logit 为 `sinks[h]`、value 恒为 0 的虚拟 key：它进入
softmax 分母因而进入 LSE，但不进入 O；其梯度是逐 query head 的标量归约
`dsinks[h] = -sum exp(sinks[h] - LSE) * D`。`sinks` 为 null 时 `dsinks` 必须精确为 0。
[^sink-guide][^bsag-guide]

## 分核策略与基本块切分

反向路径的固定分核约定如下，前向与 select-mask 也采用同类的“块级任务”思想：[^bsag-tiling]
[^bsag-scheduler][^bsag-vector]

- host 选择所有可用 AIC，设置 `baseM = min(blockShapeX, 128)`、
  `baseN = min(blockShapeY, 128)`、`singleM = align_up(2048, baseM)`；在 blockShape 为 64 倍数
  的约束下 `singleM` 实际为 2048。尾部 M/N 取真实长度，计算尺寸再向 16 对齐。
- 设备侧按 `S1 -> S2 -> query-head -> batch` 遍历，S1 先以 `singleM` 分组，S2 以 `baseN` 前进。
- 对 `(b, qHead, singleM-group, kv-base-tile)`，先扫描该组是否含至少一个有效 sparse block；
  非空组按有效组序号对 AIC 数取模分配。获得该组的 AIC 再以 `baseM` 遍历 S1，仅生成 mask 有效
  的 tile。每个 AIC 对应的两个 AIV 复现完全相同的任务序列，并分别处理 tile 的上、下半行。
- 同一 `singleM`/KV tile 内，第一个有效 Q tile 把 K/V 搬入 L1，此后有效 Q tile 复用 K/V；
  `dK/dV` 在专用 FP32 L0C 中跨这些 Q tile 累加，到组内最后一个有效 tile 才写 GM workspace；
  `dQ` 每个 tile 直接写 workspace。因此 `singleM` 同时是任务所有权和 K/V 复用边界，不能只按
  MMAD tile 数理解分核。
- 主计算量近似为 `MMAD FLOPs ~= 10 * D * sum_{t in V}(M_t * N_t)`，其中 `V` 为 mask 有效且经
  `baseM/baseN` 展开的实际 tile 集；并行组数等于非空 `(b, qHead, singleM-group, kv-base-tile)`
  的个数。
- 固定成本与稀疏率弱相关：全量清零 `dQ/dK/dV` FP32 workspace、全量读取 `dO/O` 计算 `D`、
  全量读取三个梯度 workspace 做 scale/cast/回写，以及每个 AIC/AIV 从头扫描 mask 重建任务序列。
- task 分配按非空 `singleM` group 轮转、不按组内有效 base tile 数加权，mask 分布偏斜时可能产生
  尾核不均衡；改变任务粒度会同时改变 K/V 复用和 dK/dV 的 L0 累加边界。

select-mask 前处理使用 mixed AIC 1:AIV 2：K pooling 在有效 AIV 上均分，Q pooling 在同一 AIC
对应的两个 AIV 间二分；一个 batch/head 的 score 完成后所有 AIV 协作 Radix TopK。[^select-guide]

## 数据路径与存储层级

反向主路径的数据流与存储层级：[^bsag-pipeline][^bsag-cube][^bsag-vector]

```text
dO/O GM -> AIV UB SoftmaxGradFront -> D FP32 GM workspace
block mask GM -> 每个 AIC/AIV 的地址模块 -> 有效 (M,N) task
Q/K GM -> AIC L1/L0A/L0B -> QK FP32 L0C -> S FP32 UB ping/pong
dO/V GM -> AIC L1/L0A/L0B -> dP FP32 L0C -> dP FP32 UB ping/pong
S/LSE/D/dP UB -> AIV P,dS FP32 -> cast FP16/BF16 NZ -> L1 ping/pong
P/dO L1 -> AIC dV FP32 L0C -> FP32 GM workspace
dS/K L1 -> AIC dQ FP32 L0C -> FP32 GM workspace
dS/Q L1 -> AIC dK FP32 L0C -> FP32 GM workspace
三个 FP32 workspace -> AIV UB scale/cast -> dQ/dK/dV GM
```

用户 workspace 依次放 `D`、`dQ`、`dK`、`dV`，全部 FP32；固定 layout 的元素数为
`D: B*Nq*Sq*8`、`dQ: B*Nq*Sq*D`、`dK/dV: B*Nkv*Sk*D`，TND 用 total token 数替换 `B*S`。
这里的 8 是 `SoftmaxGradFront` 每 query 行输出的重复槽位，不是 head dim。[^bsag-tiling]
[^bsag-vector]

顶层固定申请 247 KiB UB 和 512 KiB L1。`baseM=baseN=128` 时，UB 中四个半 M 的 FP32 `S/dP`
ping-pong 占 128 KiB、P/dS 低精度临时区占 32 KiB、LSE/D 行缓冲占 8 KiB；L1 中 P/dS 四块占
128 KiB、Q/K/V/dO 八块占 256 KiB。Cube 另申请 64 KiB L0A、64 KiB L0B、256 KiB L0C，后者分成
S/dP ping-pong 与 dK/dV 专用累加区。[^bsag-pipeline][^bsag-cube][^bsag-vector]

源码中 V 的 L1 tensor 按 `baseN*D` 取得，但 offset 两次按 `baseM*D` 前移：`baseM < baseN` 时
相邻 V/dO 缓冲会静态重叠，`baseM > baseN` 时留下空洞。这是固定提交可见的 offset 风险，不是
实测故障结论；任何非对称 block shape 都必须先做地址范围与数值专项验证。[^bsag-cube]

前向 BSA 的存储路径按 gathered 顺序搬运：QK 与 PV 都通过 block 索引把 gathered tile 区间映射
回原始 KV block，跨 block 边界时拆成多段 GM→L1 搬运，既不把完整 gathered K/V 落入 workspace，
也不物化完整稠密 score；Q 只预载一次，QK/PV 的 L1/L0 基本模板为 `128 x 128 x 128`，实际
M/N/K 由当前 Q/KV 尾 tile 收缩。[^bsa-guide]

## 流水排布、同步关系与数值精度

反向流水按 task ping-pong：AIV 先并行清零三个梯度 workspace，再计算全量 `D = rowsum(dO*O)`；
经 `PipeBarrier`、`SyncAll` 后用 phase flag 释放等待中的 AIC。主循环中 AIC 产生当前 task 的 QK
和 `dO @ V^T` 结果并通过 flag 通知两个 AIV；AIV 各处理一半 M，生成 P/dS 到 L1，再通知 AIC；
AIC 滞后一项消费上一 task 的 P/dS，依次发射 dV、dQ、dK。末尾 AIC 排空 MM345 并再次用 phase
flag 通知 AIV，AIV 全核同步后执行最终 post。[^bsag-pipeline]

设计实现候选时必须显式给出下面四类映射，并以固定来源逐项核对：

- **任务映射**：`(b, qHead, singleM-group, kv-base-tile)` → AIC/AIV，两个 AIV 的上下半行分工；
- **存储生命周期**：GM / workspace / L1 / L0A / L0B / L0C / UB 的分配、复用与释放点，尤其
  K/V 复用边界和 dK/dV 的 L0C 累加区间；
- **同步协议**：阶段 flag（phase flag 释放 AIC、QK/dP 生产通知、P/dS 生产通知）与 task/KV
  ping-pong 是否成组变化；
- **精度边界**：见下。

数值精度边界：[^bsag-cube][^bsag-vector]

- Q/K/V/dO/O 输入为 FP16 或 BF16；QK 与 dP 的 MMAD 累加和 Fixpipe 输出为 FP32。
- P 由 FP32 score、FP32 scale 和 FP32 LSE 重算，dS 也在 FP32 中形成，但 P/dS 在进入后三次
  MMAD 前会以 RINT cast 成输入 dtype。
- dQ/dK/dV 以 FP32 MMAD 累加并原子加到 FP32 workspace；post 对 dQ/dK 乘 softmax scale、对三者
  RINT cast 成输入 dtype，dV 不乘 scale。
- dQ/dK/dV 的 workspace 写回显式启用 FP32 atomic add，host 同时拒绝 deterministic 模式。不同
  运行的低位变化属于实现允许的现象，但数值偏差仍必须落在已定义容差内，不能用“非确定性”掩盖
  系统性错误。

前向/select-mask 的精度边界类似：Q/K 输入保留 FP16/BF16；pool 输出供 Cube 使用，QK 与 softmax
max/sum 为 FP32；归一化 score cast FP16 后参与 Radix 排序，最终只回写 UINT8/INT8 mask。
[^select-guide][^bsa-guide]

# 用法

先通过 compact query 定位本 concept，再完整读取；不得把知识内容当作当前工程已验收的证据。

设计新算子时按以下顺序消费本 concept：

1. 固定 mask 轴语义（Q/KV 轴、layout、是否 K-major bitmap）与无效块、空行的处理约定。
2. 由 blockShape、head dim、dtype 和 layout 决定 tiling key / 模板实例，再推导 `baseM/baseN`、
   `singleM`、尾块 16 对齐和 K/V 复用边界。
3. 按“数据路径与存储层级”核算 UB/L1/L0/workspace 上限与地址偏移，任何非对称 block shape 必须
   做地址重叠专项检查。
4. 按“流水排布”建立 stage/slot/flag/event 表，并单独核对启动、稳态、skip 路径和排空。
5. 按“验证方法”建立正确性矩阵，再谈性能候选。

# 代码模式

以下为与语言无关的伪代码骨架，用于表达算法与数据流，不绑定具体编程模型。

```text
for each (b, qHead, singleM_group, kv_base_tile):
    if group has no valid sparse block: continue
    for each q_tile in group (baseM rows):
        if mask(b, qHead, q_tile, kv_base_tile) == 0: continue
        load K/V to L1 if first valid q_tile in group
        S  = mmad(Q_tile, K^T)            # FP32 accumulate
        P  = exp(scale * S - LSE)         # FP32 -> cast input dtype
        dP = mmad(dO_tile, V^T)           # FP32 accumulate
        dS = P * (dP - D)                 # FP32 -> cast input dtype
        dV_acc += mmad(P^T, dO_tile)      # FP32 L0C accumulate
        dQ_acc  = mmad(dS, K)             # FP32 L0C
        dK_acc += mmad(dS^T, Q_tile)      # FP32 L0C
        atomic_add(dQ_workspace, dQ_acc)
    atomic_add(dK_workspace, dK_acc)
    atomic_add(dV_workspace, dV_acc)
// post
dQ = cast(scale * dQ_workspace); dK = cast(scale * dK_workspace); dV = cast(dV_workspace)
```

C++ 形式实现时，矩阵乘应落到 Cube MMAD 数据路径（GM→L1→L0A/L0B→FP32 L0C），softmax 重算、
P/dS 生成、scale/cast 和 workspace 归约应由 Vector 侧承担；标量代码只处理地址、索引、循环控制
和 tail 条件，不得用逐元素标量循环代替 QK/PV 主计算。[^bsag-cube][^bsag-vector][^bsa-guide]

# 约束

- 固定来源只覆盖 `__CCE_AICORE__ == 310`、DAV_3510 注册的 Ascend 950 Arc35 路径，不能外推到
  其他架构；本 concept 的 `architectures` 因此只声明 `ascend950`。[^bsag-entry][^bsag-tiling]
- Arc35 host tiling 只接受 FP16/BF16、`headDim = 128`、blockShapeX/Y 为 64 的倍数，且 Q/K/V 的
  batch、序列长度和 head dim 相同；Q heads 必须等于 `qGroup * kvHeads`。[^bsag-guide][^bsag-tiling]
- Q/KV layout 必须相同且只允许 BSND、BNSD、TND；TND 必须提供两组 actual-seqlen，非 TND 忽略。
  attention mask、非零 mask type、滑窗 pre/next token 均不在当前路径内，确定性模式也不支持。
  [^bsag-guide][^bsag-tiling]
- TND kernel 把 actual-seqlen 数组解释为累计结束位置：第 0 项直接作为长度，后续长度为
  `x[i] - x[i-1]`，前一累计值又直接用于 GM 基址。这与 README “每个 batch 实际长度数组”的文字
  存在漂移；调用和 reference 必须使用累计边界，直至源码或文档统一。[^bsag-guide][^bsag-entry]
- workspace 的 sparse index/count（前向）与 `D/dQ/dK/dV`（反向）大小和偏移必须与 host tiling
  完全一致；修改压缩表示时必须同步更新消费者和 workspace 计算。[^bsag-tiling][^bsa-guide]
- workspace 大小使用 64-bit 元素数，但 AIV 的均分结构和循环入口使用 32-bit
  `data_size/start_idx/len`；接近或超过 `2^32` 元素的梯度区必须做 host 上界检查和专项验证。
  [^bsag-tiling][^bsag-vector]
- 每个有效 `(B, qHead, qBlock)` 必须至少选择一个有效 KV block；索引需按递增 KV block 顺序压缩，
  否则 gathered 尾长和逻辑拼接顺序会失配。[^bsa-guide][^bsag-scheduler]
- 不得把知识内容中的候选优化写成已验证收益；每条优化必须先在完整正确性矩阵下通过再进入
  benchmark。[^bsag-guide]

# 失败表现

- 非对称 sparse pattern 像沿 block 对角线转置：优先核对 README shape 与 `IsValidBlock` 的
  KV-major 线性读取漂移。[^bsag-guide][^bsag-entry]
- TND 第二个 batch 起整体错位：actual-seqlen 传成逐 batch 长度而非累计结束位置。
  [^bsag-guide][^bsag-entry]
- `blockShapeX != blockShapeY` 时出现局部随机误差或旧 tile：优先检查 V/dO L1 buffer 的
  `baseM/baseN` offset 重叠，再检查尾部 16 对齐。[^bsag-cube]
- dV 正确但 dQ/dK 成比例偏差：post 的 softmax scale 缩放遗漏、重复或位置改变。
  [^bsag-vector]
- 只有 GQA/MQA 的 dK/dV 错：检查 `qHead/qGroup` 的 KV-head 映射和跨 query-head 的 FP32 atomic
  累加。[^bsag-guide][^bsag-scheduler]
- 高稀疏 workload 加速不随有效块数下降：先看全量 workspace zero/post、dO/O 前处理和每核重复
  mask 扫描占比，而不是直接归因 MMAD。[^bsag-pipeline][^bsag-vector]
- 偶发挂起或读取上一 task：核对 phase flag、QK/dP 生产 flag、P/dS 生产 flag，以及 task
  ping-pong 与 KV ping-pong 是否成组变化。[^bsag-pipeline]
- 某个 mask row 全零时越界或得到随机输出：`sparseCount == 0` 后仍读取最后一个索引。
  [^bsa-guide]
- 非连续稀疏块结果像连续 KV：QK/PV 的 gathered offset 到原 block index 映射错误。
  [^bsa-guide]
- 多 KV tile 时概率和不为 1、O 随 tile 顺序改变：running max/sum 或旧 O rescale 漏更新。
  [^bsa-guide]
- 超大 tensor 仅尾段未初始化或回写缺失：检查 64-bit workspace size 到 32-bit AIV 均分字段的
  截断。[^bsag-tiling][^bsag-vector]

# 验证方法

先记录 tiling key，确认实际命中的 dtype/layout 组合与预期一致，再用 CPU/FP32 oracle 按有效
sparse tile 计算算法公式。反向 oracle 必须接收同一次前向产生的 O 和 LSE；除最终 dQ/dK/dV 外，
至少抽查 `D`、重算 P、dS，才能区分前向中间量、SoftmaxGrad 和后三次 MMAD 的误差。[^bsag-guide]
[^bsag-cube]

正确性矩阵至少覆盖：

- dtype × layout：FP16/BF16 × BSND/BNSD/TND（前向另含 mask 前处理的 TND/BNSD）；
- head 结构：MHA、GQA、MQA，含 `q_heads != kv_heads`；
- mask：全 1、单 block、多个连续块、多个非连续块、只选最后一个尾块、每行不同 pattern、
  QBlocks 与 KVBlocks 不相等；
- shape：`64x64`、`128x128`，以及专门暴露 L1 offset 的 `64x128`、`128x64`；对齐/非对齐 S 尾块、
  TND 多 batch 累计边界、零长度 batch、Q/KV 每 batch 长度不同但总 token 数满足 host 约束；
- 输出：O、可选 LSE、padded row 不被写入/保持约定值；全零 mask row 应在 host/reference 阶段
  拒绝；
- 变形关系：mask 全一等价同精度边界的 dense attention；对未选 KV token 修改 K/V 不改变输出；
  保持索引顺序时拆分/合并连续 selected run 不改变结果。

反向还需覆盖重复运行的容差分布、deterministic=true 的 host 拒绝，以及接近 32-bit 元素边界的
静态上界检查。固定提交的硬件测试源码只有一个 FP16/BNSD/MHA、`128x128` 序列、`64x64` block、
全 1 mask 的 launch smoke，O/LSE 被置零且不与 golden 比较，不能作为数值正确性证据；它会同时
隐藏 mask 转置和非对称 L1 offset 风险。[^bsag-guide][^bsag-tiling]

性能验证必须使用相同设备、频率状态、shape、dtype、layout、mask pattern 和预热/采样策略比较，
同时记录端到端 kernel 时间与各流水活跃度；未 profile 的机制只能保留为候选，不能写成“已提速”。
[^bsag-guide][^bsag-cube]

[^bsag-guide]: 固定提交中的反向公式、输入布局、mask ABI、dtype 与公开约束。
[^bsag-entry]: 固定提交中的 AICORE 310 条件、mixed-core 属性、`1000..1005` 模板分发与 TND 累计长度语义。
[^bsag-tiling]: 固定提交中的 Arc35 capability、base tile、core 数、workspace 大小/布局与 tiling key。
[^bsag-pipeline]: 固定提交中的前处理、主 ping-pong 流水、跨核 flag、全核同步和 post 阶段。
[^bsag-scheduler]: 固定提交中的稀疏扫描、`singleM` 分组、轮转分核、KV 复用与任务运行时信息。
[^bsag-cube]: 固定提交中的 L1/L0 分区、五次 MMAD、跨 Q tile 累加和 FP32 atomic 写回。
[^bsag-vector]: 固定提交中的 workspace 清零、SoftmaxGradFront、P/dS、缩放 cast 与 AIV 均分。
[^bsa-guide]: 固定提交中的前向 BSA 数学语义、gathered-KV、mask shape、dtype 与接口约束。
[^select-guide]: 固定提交中的 select block mask 池化、QK、两遍 Online Softmax、Radix TopK 与 mask 回写。
[^sink-guide]: 固定提交 flash-attention 前向文档中 sink 进入 softmax 分母并参与 max 的定义。
