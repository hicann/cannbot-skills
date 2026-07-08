# Grouped Matmul — 分组矩阵乘（含 MoE 融合）实现注意事项

> **导航**：设计选型见 design skill [kernels/grouped-matmul.md](../../../catlass-op-design/references/kernels/grouped-matmul.md)；Device 调用基础见 [architecture/02-device-calling.md](../architecture/02-device-calling.md)。
> 参考 example：`02_grouped_matmul_slice_m`、`07_*_per_token_dequant_moe`、`10_*_per_token_dequant_multistage_workspace`。

Grouped Matmul 处理多组独立 `C_i = A_i × B_i`。本文聚焦**实现期**易踩的非显然坑——尤其是含融合 epilogue（dequant / SwiGLU / per-token 量化）时的同步、调度与调试。结构性选型请先读设计侧文档。

---

## 1. 骨架要点

| 项目 | 写法 |
|------|------|
| DispatchPolicy | `MmadAtlasA2PreloadAsync`（纯分组）/ `MmadAtlasA2PreloadAsyncWithCallback`（融合，性能优先） |
| Kernel | 分组 kernel（slice_m/k）/ `QuantMatmulMultiStageWorkspace` 系（量化融合） |
| 逐组 offset | `A_i` 起址 = `Σ_{j<i} M_j × K`（slice_m），`B_i`/`C_i` 同理；host tiling 或 device GroupList 计算 |
| Workspace | **指针透传** `GM_ADDR userWs = workspace;`（hand-launch 直调，**禁** `GetUserWorkspace`，见 [rules.md](../rules.md) Δ4） |

---

## 2. ★ 跨 PIPE 同步栅栏（高频坑，编译不报错）

> 融合算子在自定义 Tile / Epilogue 里手动编排多 PIPE 时，**`PipeBarrier<PIPE_V>` 只保证 V-V 排序，不跨 PIPE**。凡跨 PIPE 的数据依赖，必须配齐对应方向的 `SetFlag/WaitFlag`，否则会读到「半成品」中间结果——且**编译期完全不报错**。

### 跨 PIPE 同步事件对照表

| 数据依赖 | 需要的同步 | 典型场景 |
|---------|-----------|---------|
| V 算完 → MTE3 存出 | `SetFlag/WaitFlag<PIPE_V, PIPE_MTE3>` | SwiGLU/激活算完的结果存进 workspace |
| MTE2 搬入 → V 计算 | `SetFlag/WaitFlag<PIPE_MTE2, PIPE_V>` | 从 GM 搬入数据后做矢量运算 |
| MTE3 存出 → MTE2 复用同 buffer | `SetFlag/WaitFlag<PIPE_MTE3, PIPE_MTE2>` | 双 buffer 轮转，存完才能覆盖 |
| V-V 之间 | `PipeBarrier<PIPE_V>` | 仅同 PIPE 排序，**不能**替代上面任何一条 |

### 典型事故模式（V→MTE3 缺失）

**现象**：量化输出几乎全部饱和到 int8 极值，但同一算子产出的量化 scale 数值精确，编译无任何报错。

**根因**：Pass-1 把中间结果 tile 存到 workspace 的存出指令（PIPE_MTE3），在矢量后处理（PIPE_V）写的中间 UB buffer **退役之前**就读了它——二者间只有 `PipeBarrier<PIPE_V>`，**缺 V→MTE3 同步**。MTE3 把半成品中间结果存进了 workspace，Pass-2 读回量化自然全错。

**解法**：存储中间结果前加 `SetFlag/WaitFlag<PIPE_V, PIPE_MTE3>`；并把上一次存储的守护 `WaitFlag<PIPE_MTE3, PIPE_V>` 上提到条带循环顶部（保证覆盖前上一轮已存完）。

> 💡 **「部分正确」的迷惑性**：双输出里一个对（scale）一个错（量化结果），往往是**某条数据路径的同步缺失**，而非整体逻辑错——应**按数据路径**而非按输出排查。scale 对，是因为它走的是另一条已正确同步的核内累积路径。

---

## 3. ★ 跨 N-block 行归约：两趟 epilogue 写法

> 设计背景见 design [kernels/grouped-matmul.md](../../../catlass-op-design/references/kernels/grouped-matmul.md) Step 4 模式 B。per-token 动态量化等**行归约**后处理，per-block 轮转 workspace 只能看到单个 N-block，必须改 row-block 调度 + 两趟。

**调度**：按 M-row-block 切分，**同一 row-block 的全部 N-block 由同一核顺序处理**，使整行归约成为核内局部归约（A2 无 fp32 HBM 原子 max，必须核内做）。

```
for each M-row-block 分给本核:
  # Pass-1：遍历该 row-block 的全部 N-block
  for nb in N-blocks:
      S_tile = matmul + dequant + SwiGLU(...)        # fp32 中间结果
      copy S_tile -> S_workspace[rowBlockM, nb区段]   # (a) 落盘，★需 V→MTE3 同步
      ReduceMax(|S_tile|) -> 累积到 UB 常驻 rowMaxAbs[rowBlockM]  # (b) 核内跨 N-block Max

  # Pass-2：整行同一 scale
  Q_scale = rowMaxAbs / 127                           # 写出
  for nb in N-blocks:
      read S_workspace[rowBlockM, nb区段]              # ★需 MTE3→MTE2 守护
      Q = round(S / Q_scale)                          # 整行同一 Q_scale 量化
      copy Q -> 输出
```

**关键点**：
- `rowMaxAbs` 必须 **UB 常驻、跨 N-block 累积**（不是每 N-block 各自算）。
- Pass-1 落盘 S 与 Pass-2 读回之间靠 S workspace 中转——这正是 §2 V→MTE3 / MTE3→MTE2 同步的来源。
- **S workspace 按 per-core 复用**：单核 `rowBlockM × N × fp32`（以典型 N 量级估算约数 MB/核、全局数十 MB）。量级写进 host tiling 的 workspaceSize。

---

## 4. ★ 中间结果 dump 比对（融合算子调试方法论）

> 融合算子输出错误时，**别在最终输出层猜逻辑**。优先 dump 关键中间 workspace（如落盘的中间结果 S）与 golden 逐元素比对，快速二分定位是「计算错」还是「搬运/同步错」。

做法：在自定义 Tile/Epilogue 加一个**标准可选 dump 钩子**（如 `-DCATLASS_DUMP_<中间量>` 编译开关），把 device 端中间 workspace dump 出来，用脚本与 golden 逐元素 diff。

**实战收益**：上面 §2 的事故，正是 dump 中间结果后由逐元素 diff 出现大偏差，锁定「存储环节出错」（而非量化逻辑）；补上同步后中间结果回到位精确（diff 近 0）。没有 dump 的话，会在量化逻辑里反复空转。

建议把 `CATLASS_DUMP_*` 做成自定义 Tile/Epilogue 的标准可选钩子，交付时关闭。

---

## 5. 平台能力（A2）实现约束

- A2 (2201) **无 L0C→UB 的 CV 直通路径**：Cube 结果只能经 **HBM workspace** 中转给 Vector，不能照搬 `example 65` 的 UB ping/pong（那是 950 专属）。
- A2 双 C region / 多 stage 的 AIC→AIV 同步**走 example 10 的「HBM workspace + CrossCoreFlag」骨架**。
- 锁定参考 example 前先核对其原生平台（详见 design Step 5）。

---

## 6. ★ 性能对标基准（融合算子无同形态 example 时）

> Reviewer 默认口径是「与 catlass **同形态 example** 基线差距 <30%」。但**融合 grouped matmul**（GMM+dequant+SwiGLU+量化等）在 `catlass/examples/` 里**没有完全同形态的样例**——硬套单一 matmul example 会低估真实工作量（漏掉 dequant/激活/量化/行归约），让「达标」失去意义。此时按下表降级选取基准。

### 对标基准优先级

| 优先级 | 基准 | 适用 | 口径 |
|:---:|------|------|------|
| 1（首选） | **同语义 aclnn / 竞品算子**实测 Task Duration | 存在功能等价的官方/竞品融合算子 | catlass 实现 ≤ 竞品 × 1.0~1.3（视融合深度） |
| 2 | **最近 example 拆解相加**：分组 matmul example + epilogue 部分各自 Task Duration 估算求和 | 无竞品但能拆出可跑的子算子 | 融合实现应**优于**朴素串接之和（体现融合收益） |
| 3 | **Cube 理论上限 / MTE 带宽上限** | 无任何可跑对照 | 实测 vs 理论上限的达成率（与平台无关，最硬） |
| 4 | **同算子历史最优**（PRE/POST 调优） | 迭代调优阶段 | 每轮只要求不回退 |

### 同语义对标的操作要点

- **确认功能等价**：竞品算子的数学公式、I/O dtype/layout、量化粒度（per-token / per-channel）须与本算子一致，否则不可直接比时延。例如 A8W8 GMM+SwiGLU+per-token 量化，须找同为 A8W8、同 SwiGLU、同 per-token 的竞品。
- **同 shape 同 groupList**：用竞品 example 里的 `M/K/N/E` 与 groupList 分布跑出竞品基线，再用**完全相同的 shape**跑 catlass 实现，避免 shape 不一致导致的伪对比。
- **★ 对齐竞品的格式/约束硬条件**：竞品常对权重格式、dtype、轴长有硬约束（如某 A8W8 GMM 强制权重 `FRACTAL_NZ`、N≤上限、K<上限）。对标前**先核对这些约束**——若要与竞品同条件比时延，本算子实现需对齐到相同格式/约束（如权重也用 NZ），而**不是**用不同格式跑完再去折算差异（折算会引入扯皮且不可比）。
- **⚠️ 格式/dtype 未写明时必须向上游追问**：权重 ND 还是 NZ、量化粒度、轴长上限等是**需求项**，不是开发期可自行假定的。用户/设计未写明时，按 design 流程向上游追问确认，**禁止**臆测一种格式实现后再倒推对标口径。改格式（ND↔NZ）属算子**设计变更**，须回到 DESIGN.md 由 Architect 决策，不在开发期擅自切换。
- **对标产物归档**：把竞品基线与 catlass 实测一并写入 `docs/perf/round_NNN/`，注明竞品来源（算子名 + 版本/commit）、shape、groupList、**双方权重格式**，便于 Reviewer 复核。

> 💡 选基准本身就是设计结论：DESIGN 里若已锁定「对标某 aclnn 算子」，开发期直接复用其 example 跑基线；若设计未指定，按上表优先级降级，并在 PLAN.md 记录所选基准与理由。

---

## 7. ★ 小 M 核空转：`rowBlockM` 的双重角色陷阱与专用调度变体

> **教训来源**：row-block 调度的 per-token 量化 GMM，大 shape 达标、唯独**小 M**（M 小到 row-block 数 < 核数）严重不达标（实测比值 1.7×+）。这是行归约调度（§3）在小规模下的**必然副作用**，设计期极易漏判。

### 7.1 根因：`rowBlockM` 同时是「核调度单元」和「BlockMmad 的 M-tile」

§3 的 row-block 调度里，`rowBlockM`（= `L1TileShape::M`，典型 128）身兼两职：

1. **核调度单元**：`totalRowBlocks = Σ CeilDiv(M_i, rowBlockM)` 个 row-block 按 round-robin 摊到各核。
2. **BlockMmad 的 M 方向 tile**：每个 row-block 就是一次 matmul 的 M 分块。

当 `totalRowBlocks < 核数`（如 M=1024、rowBlockM=128 ⇒ 仅 8 个 row-block、20 核 ⇒ **12 核空转 `cube_time=0`**）。Task Duration 由最慢核决定，空转核不干活，等价于用 8 核跑，时延自然远差于铺满 20 核的竞品。

### 7.2 ⚠️ 反直觉陷阱：「缩小 `rowBlockM` 铺满核」会更慢（已证伪）

直觉解法是把 `rowBlockM` 从 128 缩到 32，使 `CeilDiv(M, rowBlockM) ≥ 核数`、铺满所有核。**实测反而更慢（+29%）**，因为两个角色**耦合不可分**：缩小 `rowBlockM` 同时缩小了 BlockMmad 的 M-tile ⇒ block 数翻数倍 ⇒ **每个 block 都要重新从 GM 串流主导的 B 权重**（B 加载 ×N 膨胀）⇒ MTE2/MTE1/scalar 全线上涨、`aic_cube_ratio` 从 0.76 崩到 0.38（**cube 被喂不饱、饿死**）。多用 12 个核补不回每核数倍的访存/开销膨胀。

> **结论：调度粒度 == 计算块粒度时，「缩粒度铺满核」与「保 cube/访存效率」互斥。** 小 M 核空转是 **design_issue（需换调度/分块），不是 Step 6 能调优掉的**。

### 7.3 正解：小-M 专用调度变体（N-split 全网格 + 原子-free 跨核行归约）

对齐竞品做法（竞品小-M 常切到专用 kernel 变体）。要点：

- **切 N 不切 M**：M-tile 恒 128（B 每列只加载一次、**零膨胀、cube 不饿死**），改为在 **N 方向** split，让全网格 `(rowBlock × nBlock)` round-robin 铺满所有核（对齐 example 10 骨架）。
- **判据（充要）**：host tiling 期 `schedMode = (totalRowBlocks < coreNum && nBlocks > 1) ? SMALL_M : LARGE_M`；用**复合 TilingKey**（如 `key = schedMode * K + dtypeKey`）分流，SMALL_M 走独立新 Kernel 文件。
- **与行归约（§3）的冲突及解法**：N-split 后整行的 N-block 散落多核，而 per-token max 是行归约、**A2 无 fp32 HBM 原子 max**。用**两阶段原子-free 归约**（全程限定 **AIV 域**，AIC 只产 C-region 不参与握手）：
  - 阶段 A：各核算 S 落全局 S workspace + 写**私有 partial-max 槽**（互不重叠、免原子）；
  - AIV 域 `CrossCoreBarrier` → 阶段 B：reducer 跨 lane `Max` 合并整行 max → 写 Q_scale → barrier → 各核用整行同一 scale 量化。
  - **partial-max 槽按 nBlock 索引（不是 coreIdx）**，slot 宽度 = `nBlocks × subblockNum`（AIV 双 subblock 各占槽，免竞争）。
- **workspace 变化**：S workspace 从 §3 的 per-core 复用改为**按 `totalRowBlocks` 全驻留**（跨 barrier 供别核读回，寻址式 `r_global × L1.M × nHalf`）；新增 partial-max 私有槽区。★ **partial-max 区 host 端 launch 前必须 `aclrtMemset` 清零**（`aclrtMalloc` 不保证零页；reducer 对未写 lane 行做 Max 依赖中性元 0，脏页会致该行量化精度错——只清 partial-max 区，C/S 由算法写满不必清）。
- **大-M 零回退**：SMALL_M 是独立 Kernel 文件，大 shape 恒选 LARGE_M 物理不进入；既有 row-block 路径**一字不改**（BlockEpilogue 只**增**公有方法，不动既有槽位/`UB_STAGES`）——用**结构性隔离**保证零回退，避免 round_004 那种改同一 kernel 参数需 env 守护的脆弱做法。

### 7.4 `CrossCoreBarrier` 是核类型域内栅栏（易错）

`catlass/include/catlass/arch/cross_core_sync.hpp` 里 barrier flag 由 `g_coreType` **编译期**选取（AIV 与 AIC 用不同 flag、**不互通**）。所以两阶段归约链**必须全程限定 AIV 域**触发 barrier，AIC 侧不能参与同一 barrier 握手，否则半侧挂死。参考纯 AIV 域先例（`#ifdef __DAV_VEC__` 包住整段归约）。

### 7.5 ★ 复用同一 HardEvent 的 SetFlag/WaitFlag 计数净平衡（新增方法：小-M partial-max 导出踩坑）

> **教训来源**：小-M N-split 的 partial-max 导出函数复用了主 Pass 的 `MTE3_V` 事件队列，导致一类**只在低-K / 小规模偶发、被误判成时序竞态**的确定性量化错——实际连错三次根因才定位。

**坑**：小-M 专用 kernel 里，partial-max 导出函数（写私有槽的那步）**复用了主计算 Pass 的 per-strip HardEvent**（如 `eventQMTE3VList` 的 `MTE3_V`）做 SetFlag/WaitFlag。主 Pass 的 strip 循环末尾会遗留一个**悬挂 `SetFlag<MTE3_V>`**（+1，本应由「下一 strip」顶部 WaitFlag 消费，但末 strip 无下一迭代）。导出函数进入时计数已 +1 → 它的每个 `WaitFlag` 都释放在**上一次陈旧 set** 上 → 每次导出存储晚一拍被等待 → 顶部 strip 的 partial-max 存储要等到**下一个 cell** 的主 Pass，而那时 `ResetRowMax`（PIPE_V Duplicate）已覆盖单一内驻的 rowMaxAbs → 顶 strip 导出被丢弃 → reducer 少计高行 max → 量化 scale 偏小 → int8 输出错（但 scale 之外一切正确，极具迷惑性）。低-K 时 AIC 喂 cell 快、丢弃几乎必然；高-K 给晚到存储足够余量常能赶上 → **伪装成「K 相关时序竞态」**。

**修复**：不是加大栅栏，是让复用事件的整条链**计数净平衡**——导出函数 strip 循环**前**加一个 `WaitFlag<MTE3_V>` 消费上游遗留的悬挂 token、循环**后**补一个 `SetFlag<MTE3_V>` 重新武装下一 cell 主 Pass strip[0] 期望的 +1。两条标量指令，架构零改动。

**两个决定性诊断手段（凡「小规模/低-K 偶发、大规模消失」的正确性问题必先做）**：

1. **`PipeBarrier<PIPE_ALL>` 二分法**：在疑似点前加一次**全流水 drain**。若仍 FAIL ⇒ 缺陷能扛过完整 drain，**不是时序竞态**，必是确定性 accounting（事件计数 / 地址偏移 / buffer 覆盖）问题——别再往「加同步」方向空转。若 PASS 才是真跨 PIPE 同步缺失（回 §2 配对向 SetFlag/WaitFlag）。
2. **byte-identical shape 翻转即确定性信号**：完全相同的 shape 仅换 group-list 就 PASS↔FAIL，**不是竞态证据**（竞态应随机、跨 rep 抖动），而是「跨 cell/跨块的确定性链在不同边界上净平衡与否」的差异——指向 accounting bug。用中间量 device dump（如 partial-max 逐 lane 探针 `MISSING_LANES`）逐 lane/逐行比对 golden，直接看「哪条 lane/哪个 strip 的数据丢了」。

> **强制核对**：任何「复用另一循环/另一 Pass 的 HardEvent」的写法，必须把整条链的 `SetFlag`/`WaitFlag` 计数在**跨迭代、跨 cell 边界**上核平（尤其末迭代遗留的悬挂 token 会被下一复用者错配一拍）。新开一个独立 event 队列往往比复用更安全。

<!-- 上述 §7.3/§7.4/§7.5 的 file:line、事件名会随 catlass 版本与本算子实现漂移，用时以当前 header 为准 -->

---

## 强制规则

| 规则 | 说明 |
|------|------|
| 跨 PIPE 同步 | 凡跨 PIPE 数据依赖必配 `SetFlag/WaitFlag`；`PipeBarrier<PIPE_V>` 不跨 PIPE（§2） |
| 行归约调度 | per-token 量化等行归约用 row-block 调度 + 两趟 epilogue，禁 per-block 轮转直接归约（§3） |
| Δ4 | workspace 指针透传 `userWs = workspace`，禁 `GetUserWorkspace`/`SetSysWorkspaceForce` |
| Δ10 | SwiGLU 等配对列门控按 `[M,H]` 调度，见 [rules.md](../rules.md) Δ10 |
| 测试覆盖 | 必须含 `N > L1TileShape::N` 的实网 shape（暴露跨 N-block bug）+ **小 M shape（`totalRowBlocks < 核数`）暴露核空转**（§7） |
| 性能对标 | 融合算子无同形态 example 时，按 §6 优先级选基准（首选同语义 aclnn/竞品），禁无基准空跑 |
| 小 M 调度 | row-block 调度的 GMM 小 M 核空转是 design_issue，**禁**缩 `rowBlockM` 铺核（cube 饿死，§7.2）；须小-M 专用变体（N-split + AIV 域原子-free 归约，§7.3）|
| partial-max 清零 | 小-M N-split 的 partial-max workspace 区 host 端 launch 前必须 `aclrtMemset` 清零（§7.3）|
| 复用 HardEvent 计数平衡 | 复用另一循环/Pass 的 HardEvent 做 SetFlag/WaitFlag 时，须核对整条链跨迭代/跨 cell 的 set/wait 计数净平衡（末迭代悬挂 token 会被下一复用者错配一拍）；宁可新开独立 event 队列（§7.5）|
| 偶发正确性诊断 | 「小规模/低-K 偶发、大规模消失」的正确性问题：先用 `PipeBarrier<PIPE_ALL>` 二分「时序竞态 vs 确定性 accounting」（drain 后仍 FAIL ⇒ 非竞态）；byte-identical shape 换 group-list 翻转 PASS/FAIL ⇒ 确定性 accounting，非竞态（§7.5）|
