# FlashAttention — BNSD QKV 类实现注意事项

> **导航**：设计路由见 design skill [kernels/flash-attention.md](../../../catlass-op-design/references/kernels/flash-attention.md)。本文聚焦实现期规则：kernel 设计经验、BNSD 接口、PAGED/block_table 方案、host 布局转换、AIC/AIV 协作、精度/性能归档。

---

## 0. FA kernel 设计经验总结（★最高优先级——先读本文再写 kernel）

### 0.1 FA kernel 手搓的四大门槛（历史坑，已全部破解并沉淀为手搓手册）

标准 FA-2（Q·Kᵀ→softmax→·V）的 kernel 在 Ascend A2 上是 **AIC（Cube）+ AIV（Vector）跨核协作**的混合 kernel。早期（无手搓手册时）用 catlass Block 级组件从零拼装，以下四项时序是**黑盒**、盲拼必死锁（现已全部破解，见下方现状）：

| 黑盒项 | 为什么手工对不齐 | 错了的后果 |
|---|---|---|
| **CrossCoreFlag Set/Wait modeId 配对** | AIC set flag 和 AIV wait flag 的 modeId（A2: 0/1/2）必须精确配对，但哪些 stage 用哪个 modeId 没有文档，只能从已验证 kernel 逆向 | **静默死锁 507015**——AIC 已 set，AIV 卡在 `CrossCoreWaitFlag` 不返回，无报错无日志 |
| **HardEvent flag 预置** | kernel 入口必须预置 `M_MTE1×8 + FIX_M×2 + MTE1_MTE2×8` 共 18 个 `SetFlag`，对应 kernel 出口 18 个 `WaitFlag`。漏一个或顺序错 → 流水死锁 | 同上 |
| **online softmax m/l UB 布局** | m（running max）和 l（running sum）在 UB 中的位置、与 S/P buffer 的 aliasing 关系是 EpilogueOnlineSoftmax 内部实现，外部不可见 | 精度静默错误（S 被置零或 rescale 错） |
| **preLaunch 流水时序** | preLaunch=2 意味着 AIC 比 AIV 领先 2 个 KV 块，flag 的 set/wait 要匹配这个偏移 | 流水断流 → 死锁或性能崩溃 |

**现状（已破解）**：上表四项黑盒已全部从已验证 kernel 源码**逆向推导为显式规则**，沉淀进 [fa-kernel-handcraft.md](fa-kernel-handcraft.md)（§3 同步编排、§5 事件预置、§6-§7 组件流水与 softmax 内部、§12.5.3 正确同步架构）。**手搓是默认路径**——FA 族任意形态（含标准 FA）均可凭手册从零写出，并经两轮纯 skill 实证（FFA 全家族 1.68× vs 标杆、复合变体一次成型）。
FA 族任意形态（含标准 FA）默认按手搓手册从零写出——§1–9 即 FAInfer 级内核的全量逆向知识，§12.13 为 kfc 直调终态配方；**只用本 skill 文件，不读任何外部内核源**。

### 0.2 FA kernel 设计经验（从 FAInferKernel 提炼——FA 变体开发的基础知识）

以下是从已验证 FA 内核逆向提炼的**核心设计知识**——是手搓任何 FA 族 kernel（标准 FA 或 FFA 等变体）的设计原则，实现层细则见 [fa-kernel-handcraft.md](fa-kernel-handcraft.md)。**MLA 例外**：有整经验证内核，优先采用（见 [fa-mla-paged.md](fa-mla-paged.md)，手搓慢一个数量级）。

#### 0.2.1 AIC/AIV 双核分工模型

FA kernel 在 A2 上分两个逻辑核运行：
```
AIC（Cube 核）：
  - 负责 GEMM（矩阵乘）：Q·Kᵀ（stage C1）和 P·V（stage C2）
  - 使用 L0C 累加 → Fixpipe → 写 GM workspace
  - 每个 KV 块做两轮 GEMM（QK 和 PV），通过 CrossCoreFlag 通知 AIV

AIV（Vector 核）：
  - 负责逐元素运算：online softmax（max/sum/exp/P）和 O rescale
  - 从 GM workspace 读 AIC 的 GEMM 结果
  - 维护跨 KV 块递推的 softmax 状态（m, l, O_acc）
  - 通过 CrossCoreFlag 通知 AIC
```

#### 0.2.2 在线 softmax 状态管理（核心算法——所有 FA 变体共用）

FA-2 的核心是不物化完整 S×S 得分矩阵，改用**在线 softmax**逐块递推：
```
初始化：m = -inf, l = 0, O_acc = 0

对每个 KV 块 j：
  S_j = scale · Q · K_jᵀ                         # AIC GEMM
  m_new = max(m, rowmax(S_j))                     # AIV
  P_j = exp(S_j - m_new)                          # AIV
  l_new = l · exp(m - m_new) + rowsum(P_j)        # AIV（rescale + accumulate）
  O_acc = O_acc · exp(m - m_new) + P_j · V_j       # AIV rescale + AIC GEMM
  m = m_new, l = l_new

最终：O = O_acc / l                                 # AIV 除法
```

**关键设计要点**：
- `m`（running max）和 `l`（running sum）跨 KV 块递推，**不可中断或重置**
- rescale 因子 `exp(m_old - m_new)` 对 `O_acc` 和 `l` 都要乘
- 除法（÷l）**后移到所有 KV 块处理完之后**（避免每块做除法）

#### 0.2.3 跨核同步协议（A2 平台）

AIC↔AIV 用 `Arch::CrossCoreSetFlag/WaitFlag` 握手：
```
AIC 产出 S_j → CrossCoreSetFlag<PIPE_FIX>(qkReady)     → AIV 等 qkReady
AIV 产出 P_j → CrossCoreSetFlag<PIPE_MTE3>(softmaxReady) → AIC 等 softmaxReady
AIC 产出 O_j → CrossCoreSetFlag<PIPE_FIX>(pvReady)     → AIV 等 pvReady
```
- **modeId 配对**：每个 flag ID（qkReady=1, softmaxReady=2, pvReady=3）在 AIC Set 和 AIV Wait 两侧 modeId 必须一致。A2 支持 modeId 0/1/2。
- **硬件同步基址**：kernel 首参 `hardwareSyncAddr`，由 host `aclrtGetHardwareSyncAddr` 获取，kernel 内 `SetSyncBaseAddr(hwSync)` 设置。

#### 0.2.4 workspace 槽位设计

FA 流水需要 4 段 GM workspace，用独立 buffer 指针透传（**禁 `GetUserWorkspace`/`SetSysWorkspaceForce`**——直调路径会丢入参返回 kfc 地址致 MTE 越界）：

| 段 | 用途 | A2 尺寸（blockDim = aicCoreNum） |
|---|---|---|
| s | QK 输出 S（AIC→AIV）| blockDim × 131072 × 12 B |
| p | softmax 输出 P（AIV→AIC）| blockDim × 131072 × 6 B |
| oTemp | PV 输出 O_block（AIC→AIV）| blockDim × 131072 × 12 B |
| oUpdate | rescale O 更新量 | blockDim × 131072 × 12 B |

**3 级轮转**（preLaunch=2）：每核用 `coreIdx × WORKSPACE_BLOCK_SIZE_DB × (preLaunch+1)` 定址，3 个 slot 循环使用，实现 AIC/AIV 流水重叠。`WORKSPACE_BLOCK_SIZE_DB = 131072`。

#### 0.2.5 Tile / 分块策略

| 参数 | 取值 | 约束 |
|---|---|---|
| L1TileShape | `GemmShape<128, 128, 128>` | K 维必须 = D（head_dim）；D ≤ 128 |
| Q 分块（qSBlockTile） | 128 | 固定（`GetQSBlockTile`） |
| N 分块（qNBlockTile） | 由 `GetQNBlockTile(qSeqlen, groupSize)` 推导 | 保证 vec 核负载均衡 |
| KV 分块（blockSize） | 128 | 与 PAGED 块大小一致 |
| preLaunch | 2 | AIC 领先 AIV 2 个 KV 块 |

#### 0.2.6 硬件事件预置（HardEvent flag pattern）

kernel 入口必须预置以下 flag（否则流水死锁）：
```
SetFlag<M_MTE1>(EVENT_ID0~7)   × 8
SetFlag<FIX_M>(EVENT_ID0~1)    × 2
SetFlag<MTE1_MTE2>(EVENT_ID0~7) × 8
```
kernel 出口对应 `WaitFlag` 各 8/2/8 个。**这些 flag 是 in-pipe 流水同步的基础**——AIC 的 M（Mmad）→ MTE1（L0→UB）→ MTE2（UB 操作）三段流水依赖这些 flag 推进。

#### 0.2.7 PAGED 方案（A2 平台）

A2 上 `PAGED_CACHE_FLAG=false`（真非 Paged）触发 aicore exception（507015），**必须 `PAGED=true`** + 恒等 block_table：
- `block_table[g] = g`（连续 KV 等价非 Paged）
- 每 batch `maxNumBlocksPerBatch = ⌈maxSkv/128⌉` 槽
- K 按 `LayoutK=ColumnMajor` 在 kernel 内消费，**host 不预转置 K**

#### 0.2.8 Causal mask 结构性跳块（自研掩码张量 = 固定 [1024,1024] fp16 压缩下三角）

下三角 causal（`maskType=MASK_CAUSUAL`）**不逐位算 mask**，而是**结构性跳过**：
- 内核按 query 块 `qSBlockIdx` 推导 `noSkipKvS = (qSBlockIdx+1) × curQSBlockTile + diffS`（diffS=Skv-Sq）
- 对角块以上的整 KV 块**完全跳过**（不算 QK、不算 softmax、不算 PV）
- 仅对角块调用带 mask 的 softmax epilogue（`mask[r][c]=1 当 c>r`）
- 效果：causal 比 full attention **省约一半计算**（上三角整体跳过）

### 0.3 FA kernel 开发策略（按算子形态分流）

| 算子形态 | kernel 策略 | 原因 |
|---|---|---|
| **标准 FA / FA-Paged / GQA / Causal**（CrossCoreFlag 组件路径） | **首选 [fa-paged-handcraft-recipe.md](fa-paged-handcraft-recipe.md) 逐行配方**（四文件端到端：kernel Step 1-7 + mask Step 8 + 三件套 Step 9-11，NO_MASK 闭卷实证 kv 至 20万 全 PASS）；机制解读回手册 §1–9 | 逐行级配方比骨架级手册少一层"再发明"；mask/causal 的结构跳块与 QKTail/PVTail 只有 recipe 有完整步骤 |
| 标准形态（kfc 单 TU 直调）/ fa-sink（S1 模板）/ varlen / q8 复合 | 手搓：**fa-sink → [fa-sink-handcraft-recipe.md](fa-sink-handcraft-recipe.md)**（S1 模板逐步骤，sink=SoftmaxFlashV2 isUpdate 播种）；**varlen → [fa-varlen-handcraft-recipe.md](fa-varlen-handcraft-recipe.md)**（varlen kernel 逐步骤，TND 直读+批界截断）；任意序列长/复合 → §12.13 kfc 配方 + §12.12/§12.14 叠加 | **skill 自包含**；§12.13 实测 geomean 0.693 vs 标杆 sink 算子；§12.11 S1 模板 0.80-0.92x |
| **其他 FA 变体**（双 KV、FFA、FlashSoftmax 分片等） | 按 [fa-kernel-handcraft.md](fa-kernel-handcraft.md) 手搓单 kernel：套用骨架模式（任务切分/逐块流水/事件预置/host 模板），按变体映射法增改 GEMM 数/flag 数/workspace 段/softmax 相位特性；FFA 直接用 §12.5.14 终态配方 | 手册含全部同步规则与 12 个实锤坑，手搓能力已经两轮纯 skill 实证；变体数学没有现成 kernel |
| MLA（latent 压缩 + 分页 KV） | **手搓**（[fa-mla-handcraft-recipe.md](fa-mla-handcraft-recipe.md) 端到端配方：kernel §1-8 + tiling §11 + 槽位/host §12；fa-mla-paged.md §8 为设计依据） | recipe 为端到端逐行配方，闭卷实证一次跑通；§8.9-8.12 五大性能支柱解释为什么快 |
| FFA / FIA（双 KV TensorList + sharedPrefix，DeepSeek 形态） | **生成走 §12.5.14 终态配方**；算子结构契约见 [ffa-handcraft-recipe.md](ffa-handcraft-recipe.md)（sink-in-epilogue/FD/LSE/nZ/量化入参族语义） | §12.5.14 实证几何 2.19×；kernel 与 paged recipe 同构，差异已有清单 |

---

## 1. 公开接口必须是 BNSD，布局转换放 host

> **命名**：FA 算子统一命名 `FA-<layout>[-<mode>]`（如 `FA-BNSD` / `FA-BNSD-Causal` / `FA-TND`），见 design 路由「命名约定」。

FlashAttention 算子的**公开接口**取严格 BNSD：`Q=[B,H,Sq,D]`、`K/V=[B,H,Sk,D]`、`O=[B,H,Sq,D]`，全部连续 fp16。kernel 内部消费一种布局，**host 负责转换**，kernel 只认一种内部布局、不做运行时分支。

标准 FA（FAInferKernel）的内部布局：
- Q 取 **BSND** `[B,Sq,H,D]`（`strideQO = H*D`）
- K/V 取**块格式** `[numBlocks, blockSize, H, D]`（`blockSize=128`）
- O 出 BSND，host 转回 BNSD

host 转换要点：
```text
Q:   BNSD [B,H,Sq,D] → BSND [B,Sq,H,D]      （三重循环转置）
K/V: BNSD [B,H,Sk,D] → 块 [numBlocks,128,H,D]（按 128 块重排，尾块 0 填充）
O:   BSND [B,Sq,H,D] → BNSD [B,H,Sq,D]
```
**bin 读入尺寸与 device 尺寸分离**：BNSD K 的 bin 是 `B*H*Sk*D*2`，kernel 的块格式是 `numBlocks*128*H*D*2`。

---

## 2. KV 非 128 对齐的尾块处理

`numBlocksPerBatch = ceil(Skv / 128)`。`Skv % 128 != 0` 时尾块 0 填充：
- 块 buffer 先 memset 0，再按有效长度填充
- gen_data 与 host 都要用同一套 block 映射
- golden 用原始 BNSD K/V 计算，不受块格式影响

---

## 3. 精度验证：dump O + aclnn 标杆对比

- kernel 算出 O 后，**必须 dump 输出**到 `o.bin`，供外部对比
- 精度双口径：
  1. 内部 golden（CPU numpy/torch fp32 累加 softmax attention）逐元素对比
  2. **标杆接口对比**：`torch_npu.npu_fusion_attention`（封装 `aclnnFlashAttentionScore`），与算子 O 逐元素比
- 判据：`atol=0.02, rtol=0.1`，通过率 ≥ 99.9% 且 `max_abs < 0.05`

---

## 4. 性能归档

### 度量口径（device time，不是 wall time）
- **device kernel 时间**用 cannbot `ops-profiling`（msprof）的 `task_time(us)`，**取 min of N**
- 对比标杆：`npu_fusion_attention` 的 `FlashAttentionScore` kernel（同 msprof 口径）
- **不要用 host 侧 wall-clock**比 kernel 性能（含 launch/sync/布局转换开销）

### FA skill Step 4 覆盖矩阵系统性枚举的实测基线（910B3 device 1）

| 算子 | 小 shape (Sq≤256) | 大 shape (Sq≥1024) | 总均比 |
|---|---|---|---|
| FA-BNSD（全注意力）| 1.49×（catlass 快） | 0.87×（npu 快） | 1.16× |
| FA-BNSD-Causal | 1.68× | 0.98× | 1.23× |
| FA-TND（varlen）| 1.72× | 0.88× | 1.35× |

> 小 shape catlass 快：npu aclnn FA 的框架调度开销在小 shape 占主导，kernel 直调无此开销。大 shape npu 快：aclnn FA 的大 shape 优化更激进。Sq=512 基本持平。

### 性能字段（归档到 `docs/perf/round_NNN/`）
| 字段 | 说明 |
|------|------|
| baseline | `aclnnFlashAttentionScore` / `npu_fusion_attention` 同 shape 实测（msprof device time）|
| Task Duration | custom kernel vs baseline，同 shape、同 msprof 口径 |
| cube_util | AIC 利用率（A2 经验 ~96%，AIC/AIV 高度重叠）|
| dominant pipeline | Cube / MTE / Vector / 同步等待中的主瓶颈 |
| workspace peak / profiler path | 原始 msprof trace 路径 |

---

## 4.5 手搓 FA kernel

- **标准 FA / FA-Paged / GQA / Causal（组件路径）→ 直接按 [fa-paged-handcraft-recipe.md](fa-paged-handcraft-recipe.md) 逐行写**（含 mask 路径与三件套，勿跳过 Step 6 的 3 条修复与 Step 8 的结构跳块公式）。
- **fa-sink（S1 模板）→ 按 [fa-sink-handcraft-recipe.md](fa-sink-handcraft-recipe.md)**（sink 用 SoftmaxFlashV2 isUpdate 播种法，勿手写 max/add 链）。
- **fa-varlen（TND）→ 按 [fa-varlen-handcraft-recipe.md](fa-varlen-handcraft-recipe.md)**（seqlen 累加和 kernel 内差分；TND 直读不转块；批界用 s2End 截断）。
- **FFA 双 KV → §12.5.14 生成 + [ffa-handcraft-recipe.md](ffa-handcraft-recipe.md) 语义对照**。
- **MLA → 按 [fa-mla-handcraft-recipe.md](fa-mla-handcraft-recipe.md) 逐行写**（分流必须 `__DAV_CUBE__/__DAV_VEC__`、主循环必须 prologue+延迟 PV——两种锁步/`__mix__` 变体均实测死锁）。
- **其他 FA 变体 / kfc 直调路径 → 先读 [fa-kernel-handcraft.md](fa-kernel-handcraft.md)**：4 个文件结构、任务切分算法、AIC/AIV 双入口与主循环、跨核同步编排、HardEvent 预置、BlockMmad 流水、online softmax/rescale 内部实现、host 直调模板、变体映射法、自查表。它是 §0.2 的实现层补充，§12.14-C 有全家族覆盖路由表。

---

## 5. 已知问题与踩坑（实战总结，生成前必读）

| # | 现象 | 根因 | 对策 | 来源 |
|---|------|------|------|------|
| T1 | AIC 已 set flag，AIV 卡在 softmax 内 `CrossCoreWaitFlag`/MTE2 wait 不返回（507015） | **从零拼装标准 FAKernel，flag 时序未对齐** | 标准 FA 复用 FAInferKernel（§0.1）；FA 变体用高阶 API（§0.3）| 实战复现 |
| T2 | A2 上 `PAGED_CACHE_FLAG=false` 跑出 aicore exception | 组件库该路径在 A2 未验证 | 固定 `PAGED=true` + 恒等 block_table（§0.2.7）| A2 平台复现 |
| T3 | 报某组件"未定义"/"只有声明无特化体" | 不同 catlass 版本组件名不同；某些 DispatchPolicy 只有 `struct X{}` 声明、无 `block_epilogue_*.hpp` 特化体 | 生成前 grep 核验真实组件名 + 是否有特化体（design §Step2）| catlass 版本差异实测 |
| T4 | 编译报 `MmadAtlasA2FAIQK 未定义` / `AscendC 未声明`（~20 errors） | 用了系统自带 catlass（canndev / CANN opp，**FAQK 新版 API**），与 FAInferKernel 要的 **FAIQK 旧版 API** 不兼容（6→9 参 BlockMmad、无 Tail） | 用 gitcode 的 catlass：`git clone https://gitcode.com/cann/catlass.git`（HEAD 兼容 FAIQK）；**不**用 canndev 自带 catlass | 部署实证（canndev catlass 不兼容）|
| T5 | gen_data 与算子/对比脚本的参数顺序不一致 | 参数顺序不统一 | 统一参数顺序，三方对齐 | 测试工程实证 |
| T6 | `o.bin` reshape 失败 | data 目录残留旧 shape 数据 | 每次 gen 前清 data 目录 | 测试工程实证 |
| T7 | 首次 launch 偶发卡死 | NPU 首次初始化竞态 | 重试复现即排除 | NPU 经验 |
| T8 | workspace 越界（MTE DDR） | 直调路径用了 `GetUserWorkspace` | 独立 GM buffer 指针透传（§0.2.4）| 实战实证 |

---

## 6. 强制检查表（FA0–FA13）

| # | 检查项 |
|---|--------|
| FA0 | **kernel 来源正确**：FA 族形态默认**按 fa-kernel-handcraft.md 手搓单 kernel**（骨架模式 + 变体映射法，过 §11 自查表 K1–K10）；标准 FA 复用整经验证内核仅当语义完全一致且零定制（快捷方式，非能力依赖）；**MLA 按 fa-mla-paged.md §8 手搓（达标架构，五大性能支柱缺一不可；§8.13 已闭卷实证）** |
| FA1 | 公开接口为严格 BNSD（Q=[B,H,Sq,D]，K/V=[B,H,Sk,D]），布局转换在 host |
| FA2 | 冻结同语义 baseline（`aclnnFlashAttentionScore`），语义/dtype/layout/scale 一致 |
| FA3 | A2 上固定 `PAGED=true` + 恒等 block_table（标准 FA）；FA 变体按算子需求设计 KV 布局 |
| FA4 | KV 非 128 对齐时尾块 0 填充；bin 读入尺寸与 device 尺寸分离 |
| FA5 | workspace 用独立 GM buffer 指针透传，禁 `GetUserWorkspace` |
| FA6 | AIC↔AIV 同步（CrossCoreFlag 或 Event）modeId/ID 一致，无死锁 |
| FA7 | 标准 FA 组件选型为 BlockMmadQK/PV + EpilogueOnlineSoftmax/RescaleO；FA 变体按 §0.2 原则选高阶 API |
| FA8 | shape 覆盖矩阵含 B/Sq/Sk 组合、非 128 对齐尾块、MHA/GQA、D=64/128、数值边界 |
| FA9 | 精度双口径（内部 golden + aclnn 标杆），mixed tolerance，`max_abs`/`matched_ratio` 记录 |
| FA10 | 性能报告含 baseline、Task Duration、launch count、cube_util、profiler path |
| FA11 | 多 stage/AIC-AIV 场景已读取并执行 `a2-a3-flash-attention-stage-design.md` checklist |
| FA12 | **mask 模式正确**：掩码按内核 `MaskType` 选型（`NO_MASK`/`MASK_SPEC`/`MASK_CAUSUAL`）；下三角 causal 用 `MASK_CAUSUAL`，勿自实现 causal。mask 张量语义/契约见 design skill `kernels/flash-attention-npu-reference.md` |
| FA13 | **actualQseqlen/Kvseqlen 是 per-batch 原始长度（非累加）**；dense BNSD 传 `[Sq]×B`/`[Skv]×B` |

---

## 7. mask 模式（MaskType）

| 值 | 名称 | 含义 |
|---|------|------|
| 0 | `NO_MASK` | 全注意力，无 mask |
| 1 | `MASK_SPEC` | 用 mask 张量逐位屏蔽 |
| 2 | `MASK_CAUSUAL` | **下三角 causal**：结构性跳过 + 对角块读 mask |

- 下三角 causal（npu `sparse_mode` 2/3）用 `MASK_CAUSUAL`(=2)：内核按 query 块 `qSBlockIdx` 结构性跳过对角块以上整 KV 块（§0.2.8），仅对角块调用带 mask 的 softmax。
- mask 语义：`ApplyMask = score += mask·(-3e38)` → mask≠0 屏蔽、0 保留（与 npu `atten_mask` 的 `1=不参与` 一致）。
- 标杆对比：`npu_fusion_attention(input_layout="BNSD", sparse_mode=2, atten_mask=triu(ones,bool,1))`（封装 `aclnnFlashAttentionScore`），与算子 O 逐元素比。

## 8. seqlen 契约：per-batch（非累加）

`actualQseqlen/actualKvseqlen` 是**每 batch 原始长度**（int64 数组，长度=B），内核 `qBOffset += qSeqlen·strideQO` 自行累加偏移。
- dense BNSD：`[Sq]×B` / `[Skv]×B`
- varlen TND：传每 batch 原始 Sq/Skv（非累加和）
