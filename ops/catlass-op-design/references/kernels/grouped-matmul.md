# Kernel 路由：Grouped Matmul 类算子

> 本指南是 Grouped Matmul（含 MoE 融合）类算子的设计路由入口。结构对标 [matmul.md](./matmul.md)，叠加分组特有的 tiling、调度与融合 epilogue 风险评估。架构背景见 [architecture/](../architecture/)。

---

## 场景定义

Grouped Matmul 处理多组独立的矩阵乘法 `C_i = A_i × B_i`（i = 1..G），各组 shape 不同、各自独立，常见于 MoE（Mixture of Experts）：token 按 expert 分组，每组与各自的专家权重相乘。

与普通 matmul 的本质差异：**问题规模从单个 `(M,N,K)` 变成「一组 GroupList + 各组 offset」**，分核调度要在「组」与「组内 tile」两个层级上展开。

---

## 前置

**必读文档**（按顺序）：

| # | 文档 | 关键内容 |
|---|------|---------|
| 1 | [architecture/00-hardware-arch.md](../architecture/00-hardware-arch.md) | NpuArch 映射、ArchTag、资源约束 |
| 2 | [architecture/02-block-layer.md](../architecture/02-block-layer.md) | ★ DispatchPolicy 决策树、BlockEpilogue 组装 |
| 3 | [architecture/03-kernel-layer.md](../architecture/03-kernel-layer.md) | ★ Kernel 选型、Device 调用、Params 结构 |

**必看 example**：先打开下表中形态最近的样例，对照其 `README.md` 与 host/kernel 拆分。

| Example | 场景 |
|---------|------|
| `02_grouped_matmul_slice_m` | M 方向分组（各组 M_i 不同，N/K 一致）——MoE 最常见 |
| `05_grouped_matmul_slice_k` | K 方向分组 |
| `08_grouped_matmul` | 通用 Grouped Matmul |
| `07_grouped_matmul_slice_m_per_token_dequant_moe` | ★ MoE + per-token 反量化（融合 epilogue 参考基准） |
| `10_*_per_token_dequant_multistage_workspace` | ★ 多级轮转 workspace + AIC/AIV 协同骨架（融合算子主参考） |

---

## Step 1: 确认分组形态

```
分组维度？
├── M 方向（各组 token 数不同）  → slice_m（02 / 07）  ← MoE 主流
├── K 方向                       → slice_k（05）
└── 通用                        → 08

组大小来源？
├── 运行期动态（GroupList 由上游算子产出，host 不可知）→ 走 device 侧 GroupList 寻址
└── 编译期/host 期已知                                  → host tiling 直接算 offset

调度粒度 vs 核数？（★ 设计期必判，见 Step 4.5）
├── 大 M（调度块数 ≥ 核数）        → 常规 row-block / 全网格 round-robin
└── 小 M（调度块数 < 核数）        → 需小-M 专用调度变体，否则核空转（Step 4.5）

有融合后处理？（dequant / SwiGLU / 量化 / norm）
├── 有 → 继续读「Step 4: 融合 epilogue 风险评估」（★ 高危区）
└── 无 → 纯分组 matmul，Step 2-3 选型后即可
```

---

## Step 2: 提取关键决策参数

| 参数 | 说明 | 影响 catlass 模板 |
|------|------|-----------------|
| 目标芯片 | ascend910b → A2(2201) | ArchTag + **平台能力核对（见 Step 5）** |
| 组数 G / GroupList | 静态 or 动态 | Tiling 策略、是否需 device 侧组寻址 |
| 各组 dtype | half / bf16 / int8(A8W8) | AType/BType/CType |
| 是否量化 | scale / per-token scale | Kernel（QuantMatmulMultiStageWorkspace 系）|
| 融合后处理 | dequant / SwiGLU / 量化 / 无 | BlockEpilogue 槽位 + **Step 4 风险评估** |
| per-group N/K | 是否一致 | layout 与 offset 计算 |

---

## Step 3: 选择组件

### DispatchPolicy

```
Grouped Matmul                       → MmadAtlasA2PreloadAsync
Grouped + 融合 epilogue（性能优先）  → MmadAtlasA2PreloadAsyncWithCallback
一次性功能验证基线                   → Pingpong（仅基线，勿用于交付）
```

> `PreloadAsyncWithCallback` 是所有「融合 matmul+epilogue 流水」算子的性能最优 DispatchPolicy（预取+异步+回调，AIC/AIV 细粒度重叠），与多级轮转 workspace Kernel 搭配。详见 [mmad-epilogue-selection.md](../mmad-epilogue-selection.md) §2。

### Kernel

| 条件 | Kernel | 参考 example |
|------|--------|-------------|
| 纯分组 matmul | grouped matmul kernel（slice_m/k 对应类型） | 02 / 05 / 08 |
| 分组 + per-token dequant | per-token dequant grouped kernel | 07 |
| 分组 + 量化 / 多级 workspace 融合 | `QuantMatmulMultiStageWorkspace` 系（多级轮转） | 10 / 12 |

### Tiling（分组特有）

- **逐组 offset**：`A_i` 在 GM 上的起址 = `Σ_{j<i} M_j × K`（slice_m）；`B_i`、`C_i` 同理按组累加。host tiling 或 device GroupList 二选一计算。
- **两级分核**：先把 task 摊到「组 × 组内 tile」，再交给 BlockScheduler。注意各组 M_i 不齐时尾块的核空转。
- **Workspace 来源**：catlass hand-launch 直调路径 **指针透传** `GM_ADDR userWs = workspace;`（**禁** `GetUserWorkspace`，见 develop skill rules Δ4）。量级由对应 Kernel 的 `GetWorkspaceSize` + 融合中间 buffer 决定。

---

## Step 4: ★ 融合 epilogue 的「跨 N-block 依赖」风险评估（高危，设计阶段必做）

> 这是分组融合算子最容易在**设计阶段漏判、开发期才暴露、导致回退**的一类坑。融合 epilogue 默认「按输出 tile 跨核轮转」，**每趟只能看到一个 `L1TileShape::N` 宽的 N-block**。一旦后处理需要「跨 N-block 的信息」，per-block 轮转就会出错。

### 强制检查点：尾段后处理是否含「跨 tile 边界」的操作？

逐项过一遍尾段后处理，判断它对 N 维的依赖类型：

| 后处理类型 | 对 N 维的依赖 | 跨 N-block 安全？ |
|-----------|--------------|:---:|
| dequant（逐元素 × 列广播 scale） | 逐元素 | ✅ 安全 |
| 激活 GELU/SILU/RELU | 逐元素 | ✅ 安全 |
| **SwiGLU / GeGLU / ReGLU** | **配对列** `f(C[:, :H]) · C[:, H:]`（相距 H=N/2） | ❌ 跨 N-block（见 develop Δ10） |
| **per-token / per-row 动态量化** | **行归约** `rowmax(\|S\|)` 跨整行 N | ❌ 跨 N-block（行归约陷阱） |
| softmax | 行归约（row max + row sum） | ❌ 跨 N-block |
| layernorm / rmsnorm | 行归约（mean / var） | ❌ 跨 N-block |
| top-k | 行/全局归约 | ❌ 跨 N-block |

只要命中 ❌ 行，**必须显式评估**：归约/配对维度（通常是 N）是否会被 `L1TileShape::N` 或 epilogue tile 切分？切分时单趟 epilogue 能否看到完整的归约维度 / 配对列？

> ⚠️ 经典误判：`N ≤ L1TileShape::N`（单 N-block）的用例会**碰巧通过**——此时「局部 == 全局」纯属巧合，掩盖 bug。**必须用 `N > L1TileShape::N` 的实网 shape 验证。**（典型表现：per-token 量化在单 N-block 用例全过、大 N 整片错；SwiGLU 在 `N ≤ L1.N` 全过、大 N 时大比例元素错。）

### 两类失效模式与标准解法

**模式 A — 配对列依赖（SwiGLU 类）**：按输出形状 `[M, H]` 调度，每个输出块同时产出左/右两个 N-tile（列 `c` 与 `c+H`）供 epilogue 门控。详见 develop skill rules **Δ10** 与 [mmad-epilogue-selection.md](../mmad-epilogue-selection.md) §4.2。

**模式 B — 行归约依赖（per-token 量化 / softmax / norm 类）**：把 AIV epilogue 调度从「按全 tile 跨核轮转」改为 **「按 M-row-block 切分 + 同一 row-block 的全部 N-block 由同一核顺序处理」**，使整行归约变成**核内局部归约**（规避跨核原子；A2 无 fp32 HBM 原子 max）。每个 row-block 走**两趟**：

- **Pass-1**：遍历该 row-block 全部 N-block，算出中间结果 S tile(fp32) →（a）落到该 row-block 专属 S workspace `[rowBlockM, N]`；（b）在核内 UB 常驻的 `rowReduce[rowBlockM]` 上跨 N-block 累积（核内 `ReduceMax`/`Max`）。
- **Pass-2**：用整行同一归约结果（如 `Q_scale = rowMaxAbs/127`）写出，并读回 S workspace 逐 N-block 用**整行同一 scale** 做最终运算（量化）写出。

代价：新增 per-core S workspace（单核 `rowBlockM × N × fp32`，以典型 N 量级估算约数 MB/核、全局数十 MB，占 HBM 比例极小，合理）。实现细节见 develop skill [patterns/grouped-matmul.md](../../../catlass-op-develop/references/patterns/grouped-matmul.md)。

---

## Step 4.5: ★ 小-M 核空转评估（row-block 调度必做，设计期漏判 = 后期返工）

> 这是 row-block 调度（Step 4 模式 B）在**小规模**下最容易在**设计阶段漏判**、对标补测小 M 才暴露、且**必须回设计换调度**的坑。row-block 调度天然按 `CeilDiv(M_i, rowBlockM)` 摊核；小 M 时调度块数 < 核数 ⇒ 部分核空转，时延远差于铺满全核的竞品。

### 强制检查点：调度块数是否可能 < 核数？

`totalRowBlocks = Σ CeilDiv(M_i, rowBlockM)`（`rowBlockM = L1TileShape::M`，典型 128）。若目标场景存在 `totalRowBlocks < coreNum`（如 M=1024、rowBlockM=128 ⇒ 8 个块、20 核 ⇒ 12 核空转），**必须在 DESIGN 显式规划小-M 专用调度路径**，不能只按大 M 设计。

### ⚠️ 陷阱：不能靠「缩小 `rowBlockM` 铺满核」

`rowBlockM` **同时是核调度单元和 BlockMmad 的 M-tile**，两角色耦合不可分。缩小它铺满核会同时缩小 matmul 的 M-tile ⇒ block 数翻倍 ⇒ 每 block 重复串流 B 权重 ⇒ **cube 饿死、反而更慢**（实测有 +29% 负结果）。调度粒度 == 计算块粒度时，「缩粒度铺核」与「保 cube 效率」互斥。

### 标准解法：小-M 专用调度变体（切 N 不切 M）

- **M-tile 恒 128**（B 零膨胀、cube 不饿死），改在 **N 方向** split，全网格 `(rowBlock × nBlock)` round-robin 铺满全核（对齐竞品小-M 专用 kernel 变体、example 10 骨架）。
- host tiling 按 `schedMode = (totalRowBlocks < coreNum && nBlocks > 1) ? SMALL_M : LARGE_M` 分流，**复合 TilingKey**（`key = schedMode × K + dtypeKey`），SMALL_M 走**独立 Kernel 文件**。
- **与行归约（模式 B）冲突的解法**：N-split 后整行 N-block 散落多核、A2 无 fp32 HBM 原子 max ⇒ 用**两阶段原子-free 归约**（全程 AIV 域，AIC 只产 C-region）：各核写私有 partial-max 槽（按 nBlock 索引，slot 宽 = `nBlocks × subblockNum`）→ AIV 域 `CrossCoreBarrier` → reducer 跨 lane `Max` 合整行 → 写 scale → barrier → 整行同一 scale 量化。
- **大-M 零回退**：SMALL_M 是独立 Kernel 文件、大 shape 恒选 LARGE_M 物理不进入，既有 row-block 路径一字不改（**结构性隔离**，非 env 守护）。
- workspace：S workspace 改按 `totalRowBlocks` 全驻留 + 新增 partial-max 私有槽区（host launch 前须 `aclrtMemset` 清 partial-max 区）。

**DESIGN 记录要求**：若目标 shape 覆盖小 M，DESIGN 须含「小-M 专用调度路径」节（判据阈值 + TilingKey 分流 + N-split 全网格调度 + 与行归约冲突的原子-free 归约方案 + 大-M 零回退论证 + workspace 变化）。实现细节见 develop [patterns/grouped-matmul.md](../../../catlass-op-develop/references/patterns/grouped-matmul.md) §7。

---

## Step 5: ★ 目标平台能力核对（锁定 example 后必做）

> example 能跑 ≠ 你的目标平台能跑。catlass examples 混合多代际平台（`43_*`～`57_*` 中 `ascend950_*` 是 950 专属，但**并非所有 950 专属 example 都带前缀**）。按「形态最像」挑 example，容易把高代际硬件能力误当通用能力。

锁定参考 example 后，列出它依赖的关键硬件通路/特性，逐项核对目标 ArchTag 是否支持：

| 能力 / 通路 | A2 (2201) | Ascend950 (3510) | 影响 |
|------------|:---:|:---:|------|
| L0C → UB 的 CV（Cube-Vector）直通路径 | ❌ | ✅ | A2 上 Cube 结果**只能经 HBM workspace** 中转给 Vector，**不能** UB ping/pong 驻留交换 |
| 随路 dequant（int4 QuantTileCopy 等） | 视特性 | 视特性 | 逐项查 dispatch_policy / tile 头文件 |

**A2 的硬约束**：双 C region / 多 stage 的 AIC→AIV 同步**只能走 example 10 的「HBM workspace + CrossCoreFlag」骨架**，不能照搬 `example 65` 的 UB ping/pong（它依赖 950 的 L0C→UB CV 通路）。

**要求**：在 DESIGN §1.3「参考 example 锁定理由」中**强制注明**：该 example 原生目标平台 + 本算子目标平台 + 二者能力差异及对应调整。

---

## Step 6: 分支实例化条件

| 条件 | 影响 | 取值集合 |
|------|------|---------|
| 输入 dtype | AType/BType/CType | half / bf16 / int8 |
| 量化 | Kernel + Epilogue 槽位 | 有 / 无 |
| slice 维度 | Tiling / Kernel | slice_m / slice_k |
| Swizzle 方向 | BlockScheduler | M>=N → <3,0>; M<N → <3,1> |

按 [matmul.md](./matmul.md) Step 4 同法列出合法组合表。

---

## Step 7: 输出设计章节

按 [design-document.md](../design-document.md) 模板填写，**额外**包含分组特有项：

1. 参考 Example + **平台能力差异说明（Step 5 强制项）**
2. 组件选型表（DispatchPolicy 选 Async 系）
3. **分组 Tiling 方案**：offset 计算来源（host / device GroupList）、两级分核
4. BlockEpilogue 槽位清单（如有）
5. **★ 融合后处理「跨 N-block 风险评估」栏（Step 4）**：逐后处理标注依赖类型 + 解法
6. 适配方案、分支条件
7. Workspace 来源（指针透传 + 融合中间 buffer 量级）
8. **★ 性能对标基准**：融合算子在 `catlass/examples/` 常无同形态样例，须在设计阶段锁定对标口径——**首选同语义 aclnn/竞品算子**（须功能等价：同量化粒度、同激活、同 dtype/layout）。**对标竞品时须先核对其格式/约束硬条件**（如权重 `FRACTAL_NZ`、N/K 轴长上限）：若与竞品同条件比时延，本算子应**对齐到相同格式**（如权重 NZ），而非用不同格式跑完再折算。**权重格式（ND/NZ）、量化粒度等若用户需求未写明，必须向上游追问，禁止臆测**；改格式属设计变更，须在 DESIGN 显式记录。次选「最近 example 拆解相加」，再次 Cube/MTE 理论上限。详见 develop [patterns/grouped-matmul.md](../../../catlass-op-develop/references/patterns/grouped-matmul.md) §6。

---

## 常见陷阱

| 陷阱 | 表现 | 正确做法 |
|------|------|---------|
| 把行归约后处理当逐元素 | 单 N-block 用例全过、大 N 整片错 | Step 4 模式 B：row-block 调度 + 两趟 epilogue |
| SwiGLU 用 per-block 轮转 workspace | N≤L1.N 过、大 N 大比例元素错 | Step 4 模式 A / Δ10：按 `[M,H]` 调度产左右两路 |
| 测试只覆盖 N≤L1TileShape::N | 巧合性通过，掩盖跨 N-block bug | 必须含 `N > L1.N` 的实网 shape |
| 照搬 950 example 的 UB ping/pong | A2 无 L0C→UB CV，运行异常 | A2 走 HBM workspace + CrossCoreFlag（example 10） |
| GetUserWorkspace 取 workspace | hand-launch 直调下 MTE DDR 越界 | 指针透传 `userWs = workspace`（develop Δ4） |
| 各组 M_i 不齐忽略尾块 | 核空转 / 负载不均 | tiling 时评估尾块分核 |
| 只按大 M 设计 row-block 调度 | 小 M 时调度块数<核数、多核空转、对标大幅不达标 | Step 4.5：设计期评估小-M，规划专用调度变体 |
| 小 M 靠缩 `rowBlockM` 铺核 | block 数翻倍、B 重复串流、cube 饿死、反而更慢 | Step 4.5：切 N 不切 M 的小-M 专用变体 |

---

> 设计依据补充：catlass 官方 `catlass/docs/zh/2_Design/01_kernel_design/04_matmul_summary.md` 的 grouped 部分；对应 example 的 README.md。开发实现注意事项见 develop skill [patterns/grouped-matmul.md](../../../catlass-op-develop/references/patterns/grouped-matmul.md)。
