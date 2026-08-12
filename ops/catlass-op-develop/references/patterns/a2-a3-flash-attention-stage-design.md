# A2/A3 FlashAttention Stage 设计经验

本文沉淀 FlashAttention（FA-2）算子在 A2/A3 上的 stage 设计经验。适用于同时包含 Cube/Vector 协作、KV 分块、online softmax 状态递推、跨 stage workspace 或性能流水调优的 Catlass FA 算子。参考实现：catlass `examples/23_flash_attention_infer/`（`FAInferKernel`）。

## 何时读取

命中 FlashAttention 路由后，如果出现以下任一情况，设计/实现前必须读取本文：
- 算子含多个 stage（QK^T → softmax → PV），需 AIC/AIV 异核协作；
- KV 序列长、需要分块 + online softmax 状态（max/sum/O_acc）跨块递推；
- 涉及 CrossCoreFlag、workspace slot 复用、尾块处理；
- 性能优化涉及 Cube/Vector 流水重叠。

## 1. 四段流水与状态递推（FA-2）

每个 KV block 的四段流水（数据依赖方向严格）：

```
C1 (AIC)  S_j = Q·K_j^T · scale        → L0C → Fixpipe → UB
V1 (AIV)  m_j = max(m, rowmax(S_j))
          P_j = exp(S_j − m_j)
          l_j = l · exp(m − m_j) + rowsum(P_j)
C2 (AIC)  O_j = P_j · V_j               → L0C → Fixpipe → UB
V2 (AIV)  O_acc = O_acc · exp(m − m_j) + O_j
末块      O = O_acc / l
```

**禁止**改变 C1→V1→C2→V2 的顺序，或把 online 状态（m/l/O_acc）当作简单 buffer 复用以覆盖跨块依赖。跨 KV block 递推的状态必须完整、正确地累积。

## 2. GM workspace 与 CrossCoreFlag 协议

A2/A3 上 Cube→Vector 跨 stage 中间量默认经 GM workspace 中转。每条跨 stage 边满足：

| 规则 | 说明 |
|------|------|
| 稳定 slot | 同一个逻辑块在所有 stage 使用同一 workspace slot 公式（3 级轮转：s/p/oTemp/oUpdate）|
| stage 末尾 set | producer 完整写完该 slot 后再 set ready flag |
| stage 入口 wait | consumer 进入内层循环前 wait，不在 row/tile 循环内反复握手 |
| slot 不混用 | GM slot 与 UB ping/pong 概念分离 |
| modeId 一致 | `CrossCoreSetFlag/WaitFlag` 两侧 modeId 必须一致（A2 modeId 0/1/2；950PR 另有 4）|
| 清零明确 | O_acc 初值 0、m 初值 -inf 的初始化路径明确 |

## 3. Online softmax 状态管理（m / l / O_acc）

- `m`（running max）、`l`（running sum）、`O_acc`（输出累加）是跨 KV block 递推状态，必须按 block 更新：
  - rescale 因子 `exp(m_old − m_new)` 对 `O_acc` 和 `l` 都要乘；
  - 每个 block 先 rescale 旧状态，再加新 block 贡献（`O_acc = O_acc·exp(m_old−m_new) + P_j·V_j`）。
- 尾块（`Skv % 128 != 0`）0 填充后，softmax 语义不受影响（0 填充位置的 exp 贡献为 0），但必须保证 0 填充后的数值不引入 NaN/Inf。

## 4. AIC/AIV 双特化与流水重叠

catlass `FAInferKernel` 用 `operator()<AIC>` / `operator()<AIV>` 双特化：
- AIC 特化负责两个 Mmad（QK^T 与 PV）及 Fixpipe；
- AIV 特化负责 softmax/rescale；
- 两者用 CrossCoreFlag 同步。经验：A2 上正确重叠后 `cube_util ~ 96%`、AIC/AIV 高度并行（长上下文 1159μs vs 参考 1166μs）。
- 若 Cube/Vector 互等明显，优先检查 flag 是否放在内层循环、workspace slot 是否跨 stage 冲突，而不是调 TileShape。

## 5. TileShape 与分块

- 首个可跑选型参考 `L1 = L0 = GemmShape<128, 128, 128>`；Bq（query 分块）、Bk（KV 分块）由 UB/L1 预算推导。
- `Bk=128`（blockSize）与 block_table 的 KV 块粒度一致；改 Bk 需同步改 host 块重排。
- 小 `Sq*H`（少于核数）时先评估核调度/分块策略，不直接调 TileShape。

## 6. BNSD 接口转换的 stage 归属

BNSD→内部布局转换（Q→BSND、K/V→块格式、O→BNSD）属于 **host 侧 stage**，不进 kernel：
- host 转换与 kernel 四段流水解耦，kernel 只消费一种内部布局；
- 转换循环的 shape 校验（B/H/S/D、`Skv%128`）在 host 完成，避免 kernel 分支。

## 7. buffer 预算（FA 跨核流水最易踩 UB/L1 越界，写 kernel 前先纸面算）

### 硬件上限（A2 / dav-2201 / 910B3）

| Region | 上限 | FAInferKernel 里装什么 |
|---|---|---|
| UB | 192 KB | AIV：online softmax 的 m/l（每行标量）、S（QK 输出 fp32，ping-pong）、P（softmax 输出 fp16，ping-pong）、O_acc rescale、mask（causal）|
| L1 | 512 KB | AIC：Q/K/V/P tile（`L1TileShape 128×128×128`，K 维 = D ≤ 128），preload/ping-pong 操作数 |
| L0A | 64 KB | Q tile（PV 阶段复用为 P）|
| L0B | 64 KB | K tile（PV 阶段复用为 V）|
| L0C | 256 KB | QK 的 fp32 累加器（PV 写 `(M,D)` 子区）|

> 任一 region 超上限的表现都是 **runtime 设备错误**（编译干净）。FA 最常踩的是 **UB**：S/P ping-pong + m/l + mask 同时在 UB 里，rowNum/stackSeqTile 一放大就超 192KB。

### GM workspace（4 段独立 buffer，指针透传，禁 `GetUserWorkspace`）

| 段 | 用途 | 尺寸（`blockDim = aicCoreNum`）|
|---|---|---|
| `s` | QK 输出 S（AIC→AIV）| `blockDim × 131072 × 12` B |
| `p` | online softmax 输出 P | `blockDim × 131072 × 6` B |
| `oTemp` | PV 输出 O_block | `blockDim × 131072 × 12` B |
| `oUpdate` | rescale O 更新量 | `blockDim × 131072 × 12` B |

### 预算 tally 模板（生成算子时在 DESIGN.md 填这张表）

```
| Buffer        | shape           | dtype | depth | bytes |
|---------------|-----------------|-------|-------|-------|
| S_ub (QK out) | (rowNum, N_qk)  | fp32  | 2     |  ...  |
| P_ub (softmax)| (rowNum, N_qk)  | fp16  | 2     |  ...  |
| m / l (状态)  | (rowNum, 1)     | fp32  | -     |  ...  |
| O_acc (rescale)| (rowNum, D)    | fp32  | 1     |  ...  |
| mask_ub       | (128, 1024)     | fp16  | 1     |  ...  |
| TOTAL (UB)    |                 |       |       |  ≤192K?|
```

预算紧时的压缩顺序：① 减 `stackSeqTile`（KV 块栈数 blockStackNum）；② 减 `rowNum`（Q 块行 qSBlockSize/qNBlockSize）；③ aliasing 生命周期不重叠的 UB region。**不要**靠"vf/cast 会被编译器折叠掉"来省 UB（可能不折叠，实测占 UB）。

## 8. Checklist

- [ ] 四段流水顺序严格 C1→V1→C2→V2，不重排
- [ ] online 状态（m/l/O_acc）跨块正确 rescale/累积
- [ ] GM workspace slot 公式跨 stage 稳定，flag 只在 stage 入口/末尾
- [ ] CrossCoreFlag Set/Wait modeId 一致
- [ ] 尾块 0 填充且不引入 NaN/Inf
- [ ] A2 上 `PAGED=true` + 恒等 block_table
- [ ] BNSD 接口转换在 host，kernel 无运行时布局分支
- [ ] 性能报告区分 Cube/Vector 互等、调度欠利用、尾块 3 类瓶颈
