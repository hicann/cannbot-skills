---
name: tune-frontend
description: PyPTO 算子开箱性能调优技能。主要关注代码级的调优、前端写法不同导致的性能差异，包括 loop 写法优化、TileShape 设置优化、数据操作优化等。当用户需要进行算子初始开发性能优化、开箱性能调优时使用此技能。触发词：开箱性能调优、代码级优化、loop 优化、TileShape 设置、前端优化。
---

# PyPTO 算子开箱性能调优

## 概述

开箱性能调优主要关注代码级的调优、前端写法不同导致的性能差异。在算子初始编写过程中直接得到较好的开箱性能。

调优前用 `pypto-docs-search` 查询官方接口、硬件约束，以及形状和布局相近的算子实现。搜索 TileShape 等配置时，遵循[受约束搜索](../SKILL.md#受约束搜索)。

## ⛔ 调优三阶段流程（编排器强制执行）

开箱调优必须严格按以下三阶段顺序执行，编排器会逐阶段核查输出制品：

```
阶段A: 全局分析 (F-1~F-10, F-16~F-21) ──→ 阶段B: 局部分析 (F-11~F-15) ──→ 阶段C: 逐项优化
  │                               │                               │
  │ 产出:                         │ 产出:                         │ 产出:
  │ · A1 Loop结构分析表           │ · B1 数据操作分析表            │ · 逐项优化记录
  │ · A2 常量依赖关系图           │                               │
  │ · A3 Reshape全局分析表        │                               │
  │ · A4 分段优化议程表          │                               │
  │                               │                               │
  └──── 编排器核查 ───────────────┘└──── 编排器核查 ───────────────┘└─ 按分析结果逐项优化
       制品完整性                     制品完整性                     每项验证精度+性能
```

**⛔ 禁止：未完成阶段A+阶段B的分析就直接进入阶段C逐项优化。**
**⛔ 禁止：凭直觉选择优化点跳过分析环节。**

## 阶段A: 全局分析（对应优化点 F-1~F-10, F-16~F-21）

**目标**：从算子整体结构出发，理解循环组织、常量依赖、Reshape 分布、基本块(TileShape)配置。对应阶段C中的"全局性能优化"（F-1~F-10, F-16~F-21）。

### A1. Loop 结构分析（对应优化点 F-1~F-3, F-5~F-8）

从 JIT 入口跟随计算函数，由外到内枚举 `pypto.loop`、`pypto.loop_unroll` 和 Python `for`。每个循环在 A1 表中占一行。

| 循环名与代码行 | 父循环及层级 | 类型 | 轴性质、迭代次数、切块 | 主要操作及依赖 | 读写与搬运 | 审查结论及 A2/A4 行号 |
|---------------|-------------|------|----------------------|---------------|-----------|-------------------|

1. **层次与粒度**。记录每层轴的语义、静态或动态、每次迭代 shape 和计算量。检查重复遍历同一轴、可合并的独立循环、过长的最内层循环（如超过 100 次）和过小的计算块。外层动态轴可切块，最内层小任务评估批量化或 `loop_unroll`。静态轴可评估 Python `for`，参与 `view` offset 的 n_kv/group 等语义循环按 F-18 保留。
2. **依赖与搬运**。按读写链标注各操作依赖哪些循环变量，并核对 A3 的 F-21 候选。循环内 `assemble` 或切片写入的缓冲先按循环可变处理。检查每次迭代是否重复装载相同输入，以及 `reshape`/`concat`、基本块切分或逐迭代 `assemble` 是否产生额外 GM 搬运。有泳道图时核对 COPY 或等待事件。仅在数据独立、shape 兼容时评估共享数据外提或批量执行，收集逐迭代结果时用独立目标缓冲。
3. **边界与尾块**。按代表用例核对循环起止、步长、切块及 `view`/`assemble` 的 offset 与实际轴长。检查整块的 TileShape 与适用的 Cube 或搬运对齐要求，验证不足一块时的 `valid_shape`、尾块分支和 unroll 回退能恰好覆盖有效数据。
4. **常量与参数（A2）**。在依赖表中记录硬编码常量和 `runtime_options` / `pass_options` 的定义、值、引用位置、影响的循环或 tile 站点，以及改动后须同步检查的参数。以 KV 切块为例，调整 `S2_TILE` 后同步核对循环次数、`view` 的 offset 和 `valid_shape`、相关 matmul 的轴长及其 tile 配置。
5. **逐站选 TileShape（A4）**。对循环方案涉及的 `view`、Vector 和 Cube 操作，按主块和尾块分别记录生效 tile、候选及依据。Vector 用 `set_vec_tile_shapes`，按实际 shape、dtype、UB 预算和搬运对齐选值，优先保留整条归约轴并按预算切行。reduce/cast 或 view/reshape 前后按实际 shape 和 dtype 分段核对，检查能否让相邻 op 直接消费同一切分以减少 GM 回写。每个 matmul 用 `set_cube_tile_shapes`，按实际 M/K/N 及适用的对齐和 L0/L1 约束选值。大 K/N 轴重复载入时评估增大对应 L1 tile，并核对并行任务数及计算利用率。阶段 C 逐用例验证精度和耗时。

每行记录保留、切块、合并或外提的依据，并引用对应的 A2 和 A4 记录。共享输入的批量扩展按 F-19 Mode B 核对 `for range` 的完整迭代范围或 `loop_unroll` 的单个 unroll 块。

### A2. 常量与参数依赖分析（对应优化点 F-11）

扫描算子中所有硬编码常量（如 BLOCK_SIZE、s_tile、g_tile 等），以及 `jit` 装饰器中的 `runtime_options`、`pass_options`：

| # | 常量名 | 值 | 代码行 | 被引用位置(行号) | 引用的TileShape/基本块 | 影响范围 |
|---|--------|-----|-------|-----------------|---------------------|---------|
| 1 | s_tile | 512 | L146 | L282,L292,L298,L304,L308,L318 | vec_tile(s_tile, kv_size) | 循环次数、view shape、vec_tile |

**关键检查项**：
- [ ] 常量值变更后，所有引用该常量的 `set_vec_tile_shapes` / `set_cube_tile_shapes` 是否需要同步调整？
- [ ] `runtime_options` / `pass_options` 的每个参数是否有明确的作用说明？

### A3. Reshape 与依赖深度全局分析（对应优化点 F-4 / F-21）

对算子中**每一个** `pypto.reshape` 调用，以及循环体内每个操作链的循环变量依赖深度，逐个分析：

**分析方法**：
1. 使用 `grep -nE "pypto\.(reshape|squeeze|unsqueeze)" <算子文件>` 获取所有 reshape/squeeze/unsqueeze 调用位置
2. 逐行分析每个操作的输入来源、目标 shape、是否在 loop 内
3. 对循环体内每个操作链，统计其依赖的循环变量（跨全部外层循环），按依赖深度分类（0=完全静态，少于全部=部分依赖，全部=完全依赖）

| # | 代码行 | 输入 Tensor | 源 Shape | 目标 Shape | 在loop内? | 输入类型(原始/中间/输出) | inplace? | 冗余? | 问题与建议 |
|---|--------|------------|---------|-----------|----------|----------------------|----------|------|-----------|
| 1 | L183 | input_ln_weight | [4096] | [1,4096] | 否 | 原始输入 | ✅ | 否 | 无问题 |
| 2 | L246 | k_embed | [8,128] | [8,128] | 否 | 中间结果 | ❌ | **✅冗余** | 源shape==目标shape，删除 |

**依赖深度分析表**（循环体内操作链）：

| # | 操作链 | 代码行 | 依赖循环变量 | 依赖数量 | 嵌套层数 | 分类 | 外提方式 |
|---|--------|-------|-------------|---------|---------|------|---------|
| 1 | `view(s2_range, ...)` | L147 | 无 | 0 | 3 | 完全静态 | 外提至所有循环前 |
| 2 | `s1+offset`→`full`→`le` | L145-148 | s1 | 1 | 3 | 部分依赖 | 独立 `loop(S1)` 预计算 → GM buffer |
| 3 | `si_row`→`scatter_`→`gt` | L139-142 | b,n2,s1 | 3 | 3 | 完全依赖 | 保留在循环体内 |

**关键检查项**：
- [ ] 源 Shape == 目标 Shape 的冗余 reshape？（应删除）
- [ ] 原始输入的 reshape 是否在 loop 外？（应外提）
- [ ] 原始输入的 reshape 是否使用了 `inplace=True`？（必须用）
- [ ] 原始输入的 squeeze/unsqueeze 是否在 loop 内？（⚠️ squeeze/unsqueeze 不支持 inplace，应替换为等价的 `pypto.reshape(..., inplace=True)` 外提）
- [ ] 中间结果或输出 tensor 是否使用了 `inplace=True`？（禁止用）
- [ ] loop 内的 reshape 是否依赖循环变量？（不依赖则外提）
- [ ] reshape 前后是否有对应的 `set_vec_tile_shapes` 变更？
- [ ] 循环体内是否存在依赖部分（非全部）循环变量的操作链？（F-21 依赖深度外提，详见 references/loop-hoisting.md）

### A4. 分段 TileShape 优化议程（对应优化点 F-9, F-10, F-16, F-17, F-20）

TileShape 决定 GM 与 UB/L1 之间的搬运粒度及 Cube 的计算切分。块过小会增加载入和调度次数，块过大可能压缩并行度和双缓冲空间。按算子类型、用例 shape 和 dtype 筛选候选，再用阶段 C 的 ITER 比较精度与耗时。A4 只读审查现有实现并产出议程。

1. **生成清单和议程**。用 [list_tile_boundaries.py](scripts/list_tile_boundaries.py) 分析当前算子实现，传 `--out _debug/tile_boundaries.txt --emit-agenda _debug/a4_agenda.md`。报告列出 JIT 计算路径中的分段、F-17 变化点、F-21 候选、view/assemble/reshape 和 Cube↔Vector 边界，另标注需在 trace 时核对的 tile 参数。核对内联或赋值传递的 `view` 是否实际作为 matmul 的 A/B，变量重赋值后以该处有效的绑定为准。
2. **逐 op 核对**。按源码与代表用例填写审查表。追踪实际执行分支的 `set_vec_tile_shapes` / `set_cube_tile_shapes` 来源，带入完整块和尾块 shape。

   | op 与行号 | 输入、输出 shape 和 dtype | 生效 vec/cube tile 及行号 | 上下游 tile、基本块拼接及搬运证据 | 判定与建议 |
   |-----------|--------------------------|--------------------------|-----------------------------------|-----------|

   - **Vector**。vec tile 各维度不超过实际 tensor 对应维度，输入、输出和中间量能放入 UB。检查归约轴是否被切分，非归约段优先用满尾轴。[基本块优化 §3](references/basic-block-optimization.md#3-vector-tileshape-设置规范)
   - **Cube**。按每个 matmul 的 M/K/N 检查 L0/L1 容量、对齐和整除约束。L1 各轴以实际计算轴长为上限，检查大 K/N 轴的 L1 tile 过小是否导致重复载入。不同 shape 分别给出配置，并检查并行任务数和双缓冲空间。[基本块优化 §2](references/basic-block-optimization.md#2-cube-tileshape-设置规范)
   - **分段**。在 reduce 或 cast 改变 shape/dtype 的每个 F-17 站点重算 tile。对每个 view/assemble/reshape 核对生效 vec tile、源/目标 shape 及重设后的下游 tile，按 F-20 Rule 4 选整 shape 或按 UB 预算切行。同一路径上设置值、shape、dtype 和消费者约束一致时，评估合并重复的 `set_*_tile_shapes`。阶段 C 按站点实测记录采纳或回退。
   - **基本块**。沿数据依赖比较生产者输出与消费者输入的 tile 切分，记录一对一、一对多、多对一或多对多拼接。优先检查热循环内扇出重复载入、汇聚导致的 GM 落地与流水等待。用泳道图核对实际搬运，有静态依据的预测标为假设。

3. **完成议程**。逐站点填写脚本行 ID、源码行号、结构依据、优先级、证据与建议首步。N/A 逐行写明结构事实，实测只关闭该次验证的规则。议程完成时无 `PENDING`，行数与边界报告一致。S3→S4 交接时，两份工件的 SHA-256 均对应当前实现。实现变化后重新生成。[S3 退出门控](../tune-orchestrator/references/phase-handoff.md)

**适用性 = 结构判定，二元且机械**（由清单 op 类统计直接得出）：

| 优化点 | 适用条件（结构性） |
|--------|-------------------|
| F-9 / F-16 | 段含 matmul / ≥2 个不同 M-N-K 的 matmul |
| F-10 | 段含 vec ops |
| F-17 | V 段内 shape 或 dtype 变化（如 reduce 或 cast） |
| F-20 R1 | matmul 下游接 reduce |
| F-20 R2 | 并发 matmul 共享依赖 |
| F-20 R3 | matmul 下游纯 vector（至下一 matmul/结束） |
| F-20 R4 | 段含 view/assemble/reshape |
| ⛔ view→matmul alignment | view 结果作为 matmul 的 A/B 矩阵操作数输入（view feed matmul） |
| F-4 / F-15 / F-19 | reshape 站点 / 尾轴1 broadcast 二元运算 / 可批量化内层循环 |

**view→matmul 对齐**。使用报告中该站点的 A/B 角色、`a_trans`/`b_trans`、view 前 vec tile 和 matmul 前 Cube tile。对齐以 matmul 对应的 L1 轴为准，变更时同步设置并加入 trace-time assert。报告标有多消费者警告时核对 Vector 侧 UB 空间，按 §3 多消费者规则调整 view 的 vec tile。[操作数映射与多消费者规则](references/basic-block-optimization.md#3-vector-tileshape-设置规范)

**优先级 = ⭐ 基线 + 证据调制**：
- 证据两级：**实测**（MEMORY.md 记录的 util%、A/B 实测、既往 ITER 结果）与**假设**（shape/UB 推理）。
- 实测证据可上调/下调同级 ⭐ 优化点的执行顺序（如 compute-bound 且有 util 实测 → 可降低 vec 侧调优点排序）。
- 假设级证据**不得**将某优化点排到同级 ⭐ 且有实测证据的优化点之后，更**不得**据此判为不适用。
- 每条证据标注来源：「实测：MEMORY.md cycle-N」或「假设」。

**⛔ Task-count ratio（每个 tile 设置点的量化证据，A4 必算，标注「假设」）**：对**每个** `set_vec_tile_shapes` / `set_cube_tile_shapes` 站点（不限 view——所有 vector/cube 操作均适用），计算 current（ambient/入口 tile）vs proposed（§2/§3/Rule 4 公式值）任务数比：
- vec 装载任务数：`tasks = ceil(rows / tile_row) × ceil(cols / tile_col)`
- cube 计算切分数：`tasks = ceil(M / mL0) × ceil(K / kL0) × ceil(N / nL0)`
- `ratio = tasks_current / tasks_proposed`，写入 A4 议程证据列。
**排序调制**：ratio ≥ 10× → 该站点在 P2 内提升排序（tile 优化预期收益大，先做）；ratio < 2× → 该站点后置（低优先级，最后重审）。此 ratio 是 P2 内部排序的量化依据（替代泛化的"典型收益"区间）。

**建议首步**。记录覆盖该实现全部用例的 tile 公式、fallback 常量和回退时要尝试的变体。按 §2/§3 核对容量和对齐，普通计算与 Rule 4 站点分别使用各自的 UB 预算。阶段 C 按 §1b 实现分档和 trace-time assert，并逐用例验证。[基本块优化](references/basic-block-optimization.md)

**约束取证**。代码注释、设计文档或 `accumulated_context` 声称某 tile 值受限时，查 `MEMORY.md` 的对应 cycle，并在议程记录证据。PASS 确认约束，DISPROVED / SUPERSEDED / INERT 解除约束，缺少记录时标为仅有文字依据。

### A5. 全局分析输出要求

阶段A完成后，**必须产出**：
1. **Loop结构分析表**（按 A1 模板填写）
2. **常量依赖关系图**（按 A2 模板填写）
3. **Reshape全局分析表**（按 A3 模板填写，每个 reshape 一行）
4. **分段 TileShape 优化议程表**（按 A4 模板填写，每个段/边界一行：适用优化点+结构依据+优先级+证据+建议首步）
5. **基于分析的优化建议清单**（标注优先级 P0/P1/P2）

编排器核查：以上 5 项制品齐全 → 允许进入阶段B。

---

## 阶段B: 局部分析（对应优化点 F-11~F-15）

**目标**：对算子中的数据操作逐行审查，识别局部优化机会（NZ格式、Transpose融合、冗余搬运消除、尾轴 broadcast 合轴）。对应阶段C中的"局部性能优化"（F-11~F-15，其中 F-11 常量分析已在阶段A2完成）。

### B1. 数据操作分析（对应优化点 F-12~F-15）

对算子中**所有涉及数据格式和搬运的操作**，逐行分析：

**分析方法**：
1. 使用 `grep -n "transpose\|concat\|assemble" <算子文件>` 获取相关调用
2. 检查权重矩阵的 shape 和访问模式（F-12）
3. 检查是否有 transpose + matmul 模式（F-13）
4. 检查 concat 是否可替换为 assemble（F-14）
5. 扫描所有 tensor shape，查找尾轴为 1 的 broadcast 二元运算（F-15）

| # | 检查类型 | 优化点 | 代码行 | 具体操作 | 适用? | 问题与建议 |
|---|---------|--------|--------|---------|------|-----------|
| 1 | 输入矩阵格式 | F-12 | - | 权重矩阵 shape 检查 | ❌/✅ | Shape 较大时可尝试 NZ 格式 |
| 2 | Transpose+Matmul | F-13 | - | transpose 后紧跟 matmul | ❌/✅ | 可通过 a_trans/b_trans 融合 |
| 3 | 冗余搬运 | F-14 | - | concat → assemble 替换 | ❌/✅ | 可替换 concat 为 assemble |
| 4 | 尾轴 broadcast | F-15 | - | `[M,1]*[M,N]` 等尾轴1 broadcast | ❌/✅ | 可添加 `combine_axis=True` 内联 brcb |

**关键检查项**：
- [ ] 权重矩阵 Shape 较大（>1024）？（F-12：可尝试 NZ 格式）
- [ ] 有 transpose 后紧跟 matmul 的模式？（F-13：可用 a_trans/b_trans 融合）
- [ ] 有 concat 数据搬运操作？（F-14：可替换为 assemble）
- [ ] 存在尾轴为 1 的 tensor 参与 broadcast 二元运算？（F-15：可尝试 `combine_axis=True`）

### B2. 局部分析输出要求

阶段B完成后，**必须产出**：
1. **数据操作分析表**（按 B1 模板填写，覆盖 F-12~F-15）
2. **最终优化点排序清单**（基于 A+B 分析结果，按优先级排序，引用 optimization_catalog.md 编号）

编排器核查：以上 2 项制品齐全 → 允许进入阶段C（逐项优化）。

---

## 阶段C: 逐项优化

⛔ **进入门控**：回复中必须已包含以下产出物（缺失则回退到阶段A/B补充）：
- A1 Loop结构分析表（含循环名/代码行/类型/轴性质/循环次数/最内层?）
- A3 Reshape全局分析表（含每个reshape的源shape/目标shape/inplace?/冗余?）
- A4 TileShape审查表（含每个operation的前置TileShape/合理?/拼接模式）
- B1 数据操作分析表（F-12~F-15逐项判定）
- **优化点清单：对照 [shared/optimization_catalog.md](../shared/optimization_catalog.md) 的 F-1~F-21 全表逐项标记**，每项标记为 ✅已尝试 / ❌已失败 / ❌不适用(须注明原因) / ⏳待尝试。**禁止仅列出"自己觉得有用"的项，必须覆盖全表。** 清单中存在 ⏳待尝试 项时禁止退出本阶段。

**⚠️ 约束交叉验证（防止过时代码注释误导优化决策）**：

当优化点的**任何标记**（✅已尝试 / ❌已失败 / ❌不适用）的原因涉及"代码注释/设计文档中的约束阻止或限制优化"时（例如注释标注某参数为 "precision fix"、"required"、"不得修改"，或声称某 tile 参数被 L1/L0 预算"锁死"），**不得直接采信注释**，必须执行以下验证：

1. 在 `MEMORY.md` 中搜索该约束的关键词（如 `kL1`、`split-K`、`precision fix`、`cycle`、`L1 锁死`、`L1 budget`）
2. 查看对应的调试 cycle 记录：该约束是被**实际验证（PASS）**还是被标记为 **DISPROVED / SUPERSEDED / INERT**
3. 判定规则：
   - 若调试 cycle 标记为 **DISPROVED / SUPERSEDED / INERT**：该约束**不成立**，优化点改为 ⏳待尝试，在优化时移除该约束并重新测量
   - 若调试 cycle 标记为 **PASS / 验证有效**：保留原标记，原因中附注 "MEMORY.md cycle-N 已验证"
   - 若 MEMORY.md 中**未找到**对应验证记录：保留原标记，但原因中必须附注 "⚠️ 未在 MEMORY.md 中找到验证记录，约束来源仅为代码注释"
4. 此规则同样适用于 `accumulated_context` 中前次 S4 返回的"约束与发现"——前次返回可能继承了过时注释的误判
5. **⛔ 特别警示**：当标记为 ✅已尝试 但失败原因包含"L1 预算不足"/"L1 锁死"/"L0B 超限"等资源约束，且该资源约束依赖于另一个被代码注释声明的参数（如 `kL1=full-K` 占满 L1 导致 nL1 无法增大），**必须对被依赖参数执行上述交叉验证**——若被依赖参数的约束已 DISPROVED，则资源约束自动解除，该优化点改为 ⏳待尝试

> **典型场景**：impl 注释写 `kL1=full-K (precision fix: eliminates split-K accumulation)`，但 MEMORY.md 调试 cycle-2 记录显示 `DISPROVEN — NPU cube L0C hardware accumulator does full-K fp32 reduction regardless of kL1 tile size`。此时 kL1=full-K **不构成精度约束**，可自由减小 kL1（如 512）以释放 L1 预算，使 F-20 Rule 1 的 nL0=nL1=64 配置从"L1 锁死"变为可行。**禁止仅因注释声称"precision fix"就将 F-20 标记为 ✅已尝试/❌不适用而跳过。**

**⛔ F-19 标记证据要求（防止 F-19 被静默跳过）**：F-19 Mode A / Mode B 的任何标记（✅已尝试 / ❌已失败 / ❌不适用 / ⏳待尝试）必须引用 **A1 表格行号**作为证据（如"依据：A1表第N行，LOOP_X 静态/动态分类与外提评估"）。未引用 A1 行号的 F-19 标记**无效**，视为未评估——F-19 Mode A 与 Mode B 均有有效标记前，阶段C 不得退出。**Mode A 的 ✅已尝试 另须附加证据**：对当前结构重新评估 M_current<8 的结论（按 §1.2.4 重新评估当前结构）。"已应用（基线）"本身**不是**合法终止标记。

**执行方式**：按优化点排序清单，由编排器的迭代循环（ITER_START → ITER_MODIFY → ITER_VERIFY → ITER_MEASURE → ITER_RECORD → ITER_JUDGE）逐项执行。

**⛔ ITER_VERIFY 测量范围（tile/unroll 类 ITER）**：通过 `run_cases.py --cases <affected> --perf` 逐 case 测 E2E（精度集 = INIT 确定的 `tuning_case_set`，2-3 case 覆盖结构极端值；确定机制见 tune-orchestrator/SKILL.md「covering set + runner 创建」），台账按 case 分列。ITER_MODIFY 后先过「分支验证」（§5）。采纳/回退/分歧处理按 §1b「采纳决策策略」判定（分歧→分支细化，禁止整体回退）。

**分支→case 映射**：ITER_MODIFY 改变了哪些编译期分支参数（如 S2、G、Dk）→ 从 `tuning_case_set` 中选出命中受影响分支取值的 case；未被精度集覆盖的等价类 → `--compile-only` 补漏（见 INIT 编译期补漏清单）。详见 §5「分支验证」。

**具体命令模式**：
- 分支验证（ITER_MODIFY 后、ITER_VERIFY 前）：`python run_cases.py --cases <affected> --compile-only`（tile 切分变化 → 精度模式：`--cases <affected>` 无 flag）
- ITER_VERIFY（E2E 对比 S3 基线表）：`python run_cases.py --cases <affected> --perf` → 输出 `[case N] E2E XXXus` 行，逐 case 对比 S3 基线
- 全量回归（采纳后 / 阶段退出）：`python run_cases.py --cases all`（精度模式）

**每项优化前必须确认**：
1. ⛔ 该优化点来源于阶段A或阶段B的分析结论——ITER_START 时必须在回复中写出"选择 [F-X]，依据：A4表第N行，[分析结论]"。无法写出"依据：XX表第N行" → 该表未完成 → 回退补充
2. 修改只涉及一个参数
3. 修改后该参数的依赖项是否需要同步调整（参考 A2 常量依赖关系图）
4. 如果本项优化涉及代码结构变更（loop合并/拆分、reshape移动、循环切块），必须在修改后重新检查受影响的 A1/A3/A4 议程行，更新过期的分析结论

**以下各章节为具体的优化操作指南，供 ITER_MODIFY 阶段参考。**

---

## 💡 关键启发

### 1. loop_unroll 的本质
- **目的**：增加并行度，让多个循环任务并行执行
- **unroll_list 数值含义**：代表并行块的大小（如[8,4,2,1]表示优先并行 8 个块）
- **关键**：数值代表并行块的大小，不是简单的循环展开次数
- **F-19 扩展（Mode A）**：适用于**任意循环**（`pypto.loop` / `for range` / `pypto.loop_unroll`，自含 unroll——`pypto.loop` 时自行插入 unroll_list 与 range 循环）：若循环体内 matmul 的 M_current < 8 且迭代数据独立，必须批量化——将 for-range 逐元素循环替换为批量 tensor 操作（view 整批 + transpose + 批量 matmul），将 M 从 M_current 增大到 unroll × M_current。unroll=1 时跳过 transpose，使用 reshape 替代。若 Mode A 已应用，须按 §1.2.4 Step 3 重新评估当前结构
- **F-19 扩展（Mode B）**：独立于 F-8，适用于任何 `for range` 或 `pypto.loop_unroll` 循环（不含 `pypto.loop`）。若循环体内操作的 tile 可扩展到覆盖循环迭代单元（`for range` → 全部迭代；`pypto.loop_unroll` → unroll 块大小），且操作使用共享/可广播输入（如共享权重、cos/sin、gamma、常量），可通过 reshape 将循环维度纳入 tile，批量执行操作并外提到循环外。不限于 matmul M 轴——elementwise、reduction 等任意操作类型均可适用

### 2. 优化循环结构的思路
- **合并循环** - 减少嵌套层级，增加单层迭代次数
- **外层动态轴** → 使用切块（静态切分）
- **内层动态轴** → 使用loop_unroll（动态展开）
- **黄金组合** → 外层切块 + 内层 unroll = 最优并行度

### 3. 性能优化三要素
1. **任务粒度** - 每个任务的计算量（越大越好）
2. **并行度** - 可以并行执行的任务数（越多越好）
3. **调度开销** - 任务切换的时间成本（越小越好）

**优化目标**：在保证任务粒度的前提下，最大化并行度，最小化调度开销。

### 4. Vector 操作批量化（GM assemble + 批量执行）

**触发场景**：循环体内存在逐迭代执行的 vector 操作（elementwise / reduction / softmax / cast / where 等），操作对象是更大逻辑张量的 per-iteration 小切片——vector 操作被循环拆成 N 次小任务，任务粒度小、调度次数多。

**判定方法**：若各次迭代的输出 shape 一致、迭代数据独立（无跨迭代依赖）、且下游操作行独立（dim=-1 reduction / elementwise / matmul M 轴可扩展），则 vector 操作可跨迭代批量化。

**批量化模式（assemble-to-GM）**：
1. 循环内计算 per-iteration 结果
2. 用**独立 GM tensor** 收集结果：`Y = pypto.tensor([U*M, N], dtype)`，循环内 `pypto.assemble(result_bu, [bu*M, 0], Y)`
3. 循环后对完整 `Y` 执行**一次**批量 vector 操作（softmax / reduction / elementwise / matmul）

```python
# 批量化前：vector 操作在 range 循环内逐迭代执行（U 次小任务）
for bu in range(unroll):
    x = pypto.view(all_rows, [G, N], [bu * G, 0])
    m = pypto.where(not_mask_bu, neg_inf, x)      # per-iter [G, N]
    y = softmax_like(m)                            # per-iter [G, N]
    ...

# 批量化后：assemble 到 GM tensor，vector 操作一次执行
Y = pypto.tensor([unroll * G, N], pypto.DT_FP32)
for bu in range(unroll):
    x = pypto.view(all_rows, [G, N], [bu * G, 0])
    m = pypto.where(not_mask_bu, neg_inf, x)      # per-iter [G, N]
    pypto.assemble(m, [bu * G, 0], Y)             # 收集到 GM
y = softmax_like(Y)                                # 一次执行 [unroll*G, N]
```

**⛔ 多个 range 循环不是性能问题**：目标是将 vector 操作批量化，**不是消除 range 循环**。多个 `for range` 用于 per-iteration 计算、assemble 收集、结果拆分等是**可接受的**——性能代价在重复的小 tensor vector 操作（任务粒度小、调度次数多），不在 range 循环本身。

**⛔ constraint 13 规避（separate buffer 模式）**：`view(X) → pypto.assemble(result, offset, X)`（assemble 回被 view 的同一 tensor）构成 view+assemble 环，禁止。规避方法：用**独立 `pypto.tensor` 作为 assemble 目标**——`Y = pypto.tensor(...); pypto.assemble(result, offset, Y)`，`view(X)` 与 `assemble(..., Y)` 无环。典型场景：批量化 matmul 的输出 scores 不能回写（matmul 结果 tensor），须 assemble 到新的 GM tensor 再继续批量 vector 操作。

**⛔ 与 A-D3 的边界**：本模式的 `pypto.tensor + assemble` 是 same-scope 数据收集（将 per-iteration 结果拼成批量 tensor 供下游批量消费），**不是** A-D3 禁止的"跨 scope 中转"（CV 通路断点）。A-D3 反模式仅针对打断 CV 通路的中间 assemble，不适用于本模式。

**收益来源**：
- vector 操作从 N 次小任务 → 1 次批量任务（任务粒度增大、调度开销减少）
- 下游 matmul M 轴从 M 增大到 unroll × M（Cube 利用率提升）
- tile 可按批量 tensor 整体设置（配合 F-17 分段 vec tile）

### 5. Tile 值必须按 case 分档（if/elif/else 条件分支 + fallback；禁止字面常量）

**kernel `: int` 标量参数在 trace 期是具体 Python 值**（host 传入，非 SymbolicScalar）。**SymbolicScalar（`pypto.loop`/`pypto.loop_unroll` 循环变量）禁止用于 tile**。每个 tile 设置点按 **§1b 四步流程**执行（详见 references/basic-block-optimization.md §1b；推导方法见 §2/§3）：

1. **识别分块目标 tensor 的 shape 及其中的变量**（如 `[G, S2]` → `G`、`S2`）。
2. **确认变量非 dynamic**（判别法：可用于 `pypto.view(tensor, [var, ...], ...)` 的 **shape** 参数 = plain int——view shape 只接受 `List[int]`；loop counter 是 SymbolicScalar，只出现在 view **offsets**，不可用于分支条件）。
3. **每个 tile 设置点使用自己的 tile 变量**——不同 `set_*_tile_shapes` 调用之间禁止共享同一变量，避免调优一处影响另一处。
4. **从最后一个维度开始逐维建条件分支**（靠前的维度可能依赖靠后的维度，如 row 预算 = UB // (col × dtype_bytes)）。**条件中可使用任意函数/表达式**（`S2 >= 512`、`topK >= 64` 等）——lint 只检查 tile 调用参数，不检查条件。枚举本实现所辖 case 集合中该参数的全部取值放置分支边界（推荐 2 的幂：16/32/64/…/8192），每分支 tile 参数为 **int 字面量 Assign**（设计期预计算，非运行期表达式），每条分支链以 **fallback 常量**（else 分支，对其余所有 case 安全）兜底。trace 期 assert（硬件硬限 L0B/UB）可选推荐。

**不可**将 `min()`/`max()`/`//` 等 BinOp/Call 表达式直接作为 `set_*_tile_shapes` 参数（lint 在 AST 层面拒绝 BinOp/Call 节点）；func 参数（`S2`/`Dk`/`G`）也不可直接作为 tile 参数（lint 无法从 AST 解析为 int 字面量）。

**💡 新增 tile 调用成本可忽略**：增加更多 `set_vec_tile_shapes`/`set_cube_tile_shapes` 调用对性能影响极小——只要 tile 值正确，按站点/按 shape 段插入调用是安全的（F-17 分段、F-20 Rule 3/4/5 均依赖插入新调用，不要因调用数量犹豫）。

**放置规则**：所有 if/elif/else tile 条件块统一放置在 **kernel 函数体开头（所有 loop 之前）**；`set_*_tile_shapes` 调用保留在各自 tile 站点（loop 内、shape 切换点等），引用预计算的 tile 变量。原因：tile 条件变量均为 kernel int 参数（步骤 2 已验证非 dynamic），函数入口即可用；集中放置 = 单一审计点，避免 loop 内 Assign 的作用域问题；trace 期条件只求值一次，放置位置不影响性能。禁止把条件块散落进 loop 体内计算。

所有 `set_vec_tile_shapes` / `set_cube_tile_shapes` 的取值必须使用上述 if/elif/else 分支模式，禁止单一 tuning case 的字面常量硬编码——包括优化 ITER 中调出的值（只能修改 if/elif/else 分支参数：分支内 int 字面量值、分支条件边界，禁止用计算结果字面替换整个分支结构）。

**分支验证（tile/unroll 类 ITER_MODIFY 之后、ITER_VERIFY 之前，强制）**：tile 取值或 unroll_list 变更影响**编译期分支**——验证范围是**全部分支**，不是 tuning case 所在的那一个。使用 `run_cases.py`（INIT 创建，见 tune-orchestrator/SKILL.md「covering set + runner 创建」）：
1. 识别本次修改影响的分支参数（如 `S2`、`G`、`S1`、`Dk`）。
2. 从 `tuning_case_set` 精度集中为该参数**每个受影响分支取值**选 ≥1 个 case；未被精度集覆盖的等价类 → `--compile-only` 补漏。
3. 逐 case 验证：`run_cases.py --cases <affected> --compile-only`（纯编译路径变化）或 `run_cases.py --cases <affected>`（精度模式，tile 切分变化）。
4. 任一 case 编译失败/精度回退 → 本次 ITER_MODIFY 无效，回退并记录失败 case + 失败分支（单 case 精度通过 ≠ 全分支编译正确——分支是编译期展开的，未测 case 一样会按自己的分支编译）。

```python
# ✅ if/elif/else 分支（2 的幂边界点，int 字面量 Assign per branch）
# ⚠️ 条件块放置在 kernel 函数体开头（所有 loop 之前）；下方 set 调用 = tile 站点引用预计算变量
# vec tile: row 维 UB 填充公式 → 预计算 per Dk range
# (view/assemble/reshape 站点 — op 本身不占 UB，用全部 UB_size（§一），见 Rule 4 Step 1b)
if Dk >= 512:
    vec_row = 192       # 192*1024//(512*2) = 192
elif Dk >= 256:
    vec_row = 384
elif Dk >= 128:
    vec_row = 768
else:
    vec_row = 768        # fallback
if Dk >= 128:
    vec_col = 128        # vec_col ≤ Dk: threshold = tile value
elif Dk >= 64:
    vec_col = 64
elif Dk >= 32:
    vec_col = 32
else:
    vec_col = 16         # CUBE_ALIGN 对齐下限（§一）
pypto.set_vec_tile_shapes(vec_row, vec_col)  # both resolve to int ✅

# cube tile: M/G/K/N 各轴 if/elif/else
if G >= 128:
    cube_m = 128
elif G >= 64:
    cube_m = 64
elif G >= 32:
    cube_m = 32
else:
    cube_m = 16          # CUBE_ALIGN 对齐下限（§一），G < 16 时编译器 padding
if Dk >= 128:
    cube_k = 128
elif Dk >= 64:
    cube_k = 64
else:
    cube_k = 32
if S2 >= 1024:
    cube_n = 1024
elif S2 >= 512:
    cube_n = 512
else:
    cube_n = 256
pypto.set_cube_tile_shapes([cube_m, cube_m], [cube_k, cube_k], [cube_n, cube_n])

# ❌ BinOp/Call/func-param 直接作 tile 参数：
# pypto.set_vec_tile_shapes(192 * 1024 // (Dk * 2), Dk)     # BinOp + func param
# cube_m = min(G, 16)                                        # Assign RHS = Call → rejects
# pypto.set_cube_tile_shapes([cube_m, cube_m], [min(Dk, 64), min(Dk, 128)], ...)  # Call

# ❌ 字面常量（仅对单一 tuning case 正确，其他 case 失配）
CUBE_M_TILE = 16; CUBE_K_TILE = 32
pypto.set_vec_tile_shapes(CUBE_M_TILE, CUBE_K_TILE)
```

**为何强制**：字面常量只对 tuning case 正确——其他 case class（如 MLA G=128、Dk=576）代入失配 tile 产生严重 padding 或越界；if/elif/else 分支让同一份代码在全部 case 上自适应。lint 检查 `set_*_tile_shapes` 调用的 AST 参数节点（`_resolve_to_const_int` 接受 `ast.Constant(int)` 和解析到 `ast.Constant` 的 `ast.Name` Assign；拒绝 BinOp/Call/func-param）。per-case 自适应通过 if/elif/else 分支实现（分支条件不在 lint 检查范围），每个分支内 tile 参数为 int 字面量 Assign → lint 通过。不可绕过 lint——按上述 if/elif/else 模式写。

## 调优方向（阶段C 参考指南）

以下各章节为阶段C逐项优化时的具体操作指南。每个章节对应 optimization_catalog.md 中的优化点编号。

**阶段C 优化顺序规则**：
1. **先全局后局部**：P0（F-1~F-4, F-16, F-17, F-21）→ P1（F-5~F-8, F-18, F-19 结构级 M 轴批量化）→ P2（F-9~F-10, F-20）→ P3（F-11~F-15）。**F-19（P1）先于 tile 基线（P2）**——F-19 变更 M 轴后，P2 按 §2/§3 以 M_new 直接推导 tile 值（无需既有分支可更新）；P2 内部排序按 A4 议程的 task-count ratio（见 §A4）量化证据执行
2. **每次只执行一个优化点**，按 ITER 循环执行
3. **优先执行阶段A/B分析中发现的问题**，而非盲目按编号顺序
4. **⛔ 采纳后结构台账（结构变更的统一处理机制）**：每采纳一个优化点（未回退），在**结构变更台账**中标记 `结构级`/`配置级`：
   - **配置级**（tile/cache/pass option 变更）：重新运行 `list_tile_boundaries.py` 重新生成 A4 议程（阶段内立即重审，保留原机制）
   - **结构级**（循环消除/创建/合并/类型变更/外提改变了最内层循环）：仅在台账登记，**不在阶段内重跑分析**——阶段退出时由编排器检查台账（PHASE_SUMMARY_F 门控）：若本轮存在结构级采纳（或首轮存在入口基线未重审的结构标记），编排器清除全部 DO-NOT-RETRY 标记并**重跑 FRONTEND**（结构变更后循环体已改变，旧失败结论可能失效）
   - **台账内容（阶段退出交付物）**：(a) F-1~F-21 全表标记完整性（✅/❌/❌/⏳）；(b) 已采纳未回退优化点清单，每项标注 `结构级`/`配置级`；(c) 首轮追加：入口基线既有结构标记清单（如 F-19 batched G、已消除的循环等）并标注 `audited` 状态（本轮已重审的标记置 audited，后续轮次不再触发重跑）；(d) Rule 4 view 站点三元组记录（脚本标记的每个站点：Step 1a/1b 公式 set 值 + ITER 前后 µs 实测 + 采纳/回退——缺任一项 = Rule 4 未完成，编排器门控将拒绝本摘要）
5. **⛔ tile 采纳决策**（ITER_VERIFY 后每个 tile 设置点必循）：按 references/basic-block-optimization.md §1b「采纳决策策略」执行——通过 `run_cases.py --cases <affected> --perf` 逐 case 实测精度集（非仅 tuning case），台账按 case 分列；分歧→分支细化（§1b），非整体回退。tile/unroll 类 ITER_MODIFY 后、ITER_VERIFY 前必须先过「分支验证」（§5，`run_cases.py --cases <affected> --compile-only`）。

### 全局性能优化（P0: F-1~F-4, F-16, F-17, F-21；P1: F-5~F-8, F-18, F-19；P2: F-9~F-10, F-20——本节为各优化点操作指南，执行顺序以「阶段C 优化顺序规则」为准）

#### 1. Loop 写法优化（F-1~F-3, F-5~F-8）

**增加 root function 的大小，减少它们的个数**
由于不同 root function 之间的子图不能合并，而子图合并是 PyPTO 优化性能的关键手段。

##### 1.1 静态轴使用 Python for 循环

`pypto.loop` 方法会按当前轴循环展开成不同的 root function。因此静态轴上的循环应使用 Python 的 for 循环。

```python
# ✅ 推荐：静态轴使用 Python for
for i in range(batch_size):
    result[i] = process(data[i])

# ❌ 避免：静态轴使用 PyPTO loop
for i in pypto.loop(batch_size, name="LOOP_1", idx_name="i"):
    result[i] = process(data[i])
```

##### 1.2 减少循环次数，增加并行度
###### 1.2.1 如果外层的动态轴范围很大，使用切块处理（高优先级）

算子的循环轴的 dim 数值范围往往较广，往往需要对其进行静态切分，否则循环次数太大。

示例：
```python
# 推荐：动态轴使用 loop 切块，对b_loop轴按128进行切分，提高并行度
bsz, h = x.shape
b = 128
b_loop = (bsz + b - 1) // b
for b_idx in pypto.loop(b_loop, name="LOOP_1", idx_name="b_idx"):
    b_valid = (bsz - b_idx * b).min(b)
    x_view = pypto.view(x, [b, h], [b_idx * b, 0], valid_shape=[b_valid, h])
    # Matmul
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    y = pypto.matmul(x_view, W)
```

###### 1.2.2 如果内层的动态轴范围很大，调整切块大小或使用loop_unroll展开，增加并行度
   ```python
   for idx in pypto.loop(A.shape[0] // 64, unroll_list=[128, 64, 8, 1], name="A", idx_name='b'):
       offset = idx * s2_tile
   ```

   **参数说明：**
   - `unroll_list`: 展开因子列表，按优先级从高到低排列，数值代表并行块大小
    - `[64, 32, 16, 8, 4, 2, 1]`: 默认起始（每迭代维度 < CUBE_ALIGN（§一））
   - `[128, 64, 16, 8, 1]`: 适用于更大循环（须实测编译时间）
   - `name`: 循环名称（用于调试）
   - `idx_name`: 循环索引变量名

    **⚠️ 重要原则：**
     - **loop_unroll 必须放在最内层循环！不要在外层循环使用 unroll_list**
     - 数值代表**并行块大小**，不是简单的展开次数
     - 目的是**增加并行度**，让多个任务可以并行执行
     - **unroll_list 的最大值，不要超过循环次数**
     - 改多值 `unroll_list` 后若精度回归，立即回退该 loop
      - **⛔ 编译爆炸防护**：设定 `unroll_list` 前先计算每迭代被批量化维度大小（如 matmul M_current = G；非 matmul 循环则检查每迭代 tensor 大小是否已填满 UB）。每迭代维度 ≥ CUBE_ALIGN（cube tile 最小对齐（§一））时，完整多变体列表会产生过大中间 tensor（如 64×128=8192）导致 JIT 编译超时。规则：每迭代维度 < CUBE_ALIGN → 完整列表（如 `[64,32,16,8,4,2,1]`）；≥ CUBE_ALIGN 且大 loop_count（如 prefill 类）→ `[large_unroll, 1]`（2 变体，具体 large_unroll 须实测）；≥ CUBE_ALIGN 且小 loop_count（如 decode 类）→ `[1]`。非 matmul 循环 tensor 已填满 UB → `[1]`。详见 F-8 编译爆炸防护（optimization_catalog.md）
     - **⛔ 全 case 普遍性检查 + fallback**: 设定 unroll_list 前枚举**全部 case**的 loop_count（S1/G 等——view shape 中的 int 参数，可用于 if 条件），找 MAX(loop_count)。用 if/elif/else 分支（2 的幂边界点）为不同 loop_count 选择不同 unroll_list：大 loop_count → 小 unroll（限 root function 数 < 512）；小 loop_count → `[1]`。unroll_list 是 `pypto.loop` kwarg（lint 不检查），if/elif/else + list Assign 自由使用。详见 optimization_catalog.md F-8「全 case 普遍性检查 + fallback」。**循环消除策略见 optimization_catalog.md F-8「循环消除策略」**: 全部 case loop_count 有界小、或 MAX(loop_count) 编译测试通过时允许全批量消除；否则保留 loop_unroll + unroll_list。⛔ for-range 全展开始终禁止。

###### 1.2.3 切块优化策略：外层切块 + 内层unroll

**核心思路**：对循环轴切块，减少循环次数，增大任务粒度，然后在最内层使用unroll增加并行度。

**适用场景**：当最内层循环次数过少（<8）无法有效unroll时，应考虑对外层轴切块。

**🔥 典型案例：Flash Attention（最常见错误）**

**❌ 错误实现（任务粒度过小）**：
```python
for q_idx in pypto.loop(SEQ_LEN_Q, name="LOOP_Q"):  # 64 次
    q_vec = pypto.view(query, [1, HEAD_DIM], [q_idx, 0])  # ❌ M 轴 = 1
    for kv_idx in pypto.loop(num_kv_blocks, name="LOOP_KV"):  # 1 次
        scores = pypto.matmul(q_vec, k_block, ...)  # ❌ [1, 128]
```

**问题**：
- Matmul M 轴只有 1，浪费 Cube 计算能力
- 循环 64 次，调度开销极大
- 每个任务计算量太小，任务粒度过细

**✅ 正确实现（Query 切块）**：
```python
Q_BLOCK_SIZE = 16
num_q_blocks = SEQ_LEN_Q // Q_BLOCK_SIZE  # 4 次

for q_block_idx in pypto.loop(num_q_blocks, name="LOOP_Q"):  # 4 次
    q_start = q_block_idx * Q_BLOCK_SIZE
    cur_q_size = pypto.min(Q_BLOCK_SIZE, SEQ_LEN_Q - q_start)

    # ✅ 批量获取 16 个 query
    q_block = pypto.view(query, [Q_BLOCK_SIZE, HEAD_DIM], [q_start, 0],
                        valid_shape=[cur_q_size, HEAD_DIM])

    for kv_block_idx in pypto.loop(num_kv_blocks, unroll_list=[4,2,1], name="LOOP_KV"):  # ✅ 可 unroll
        # ✅ Matmul M 轴 = 16
        scores = pypto.matmul(q_block, k_block, ...)  # ✅ [16, 128]
```

**收益来源**：
- ✅ 任务数: 64 → 4（减少 16 倍）
- ✅ Matmul M 轴: 1 → 16（计算量增大 16 倍）

**⚠️ 切块大小建议**:
- 从较大值开始尝试（如 64, 32, 16），但切开的值不应该超过shape中该维度的大小
- 平衡任务粒度和内存占用
- 调整中间 tensor 的 shape

**🔥 特例：调大 online softmax 的序列分块 tile 跨过 flash 临界点要同步删 flash 操作。** 调大 online softmax 的分块 tile（如 `S2_TILE`，取值非固定、属可调）时，一旦放大到 softmax 不再是 flash/online（单块覆盖全序列、bn=1），必须**同步删除** running-max 状态机（mi/li/oi 累加器、`is_loop_begin` 分支、跨块 rescale），改为普通 full softmax。

###### 1.2.4 内层批量化（F-19，必做）——含 `pypto.loop` 自动 unroll 化

**核心思路**：若某循环体内 matmul 的 M 轴 M_current < 8（通常为 1）且各次迭代数据独立，则**必须**将循环维度批量化进 M 轴——把逐元素循环替换为批量 tensor 操作。F-19 Mode A **自含 unroll 机制**：目标循环是 `pypto.loop` 时，由本优化点**自行插入 `unroll_list` 与 `for range(unroll)` 中间循环**（即内联执行 F-8 变换），不依赖 F-8 先行应用。

**适用场景**：每个任务的 matmul M < 8，Cube 利用率低；或入口基线已含部分批量化但仍有外层循环维度可并入 M 轴，进一步 M = unroll × M_current。

**触发条件（必做判定）**：
1. 循环类型不限：`for range(unroll)`（F-8 已应用时）、`pypto.loop_unroll`、或 **`pypto.loop`（由本优化点自行转换）**
2. 循环体内存在 matmul，其 M_current < 8（通常为 1）
3. 循环维度可通过 view 重构批量化进 matmul 的 M 轴（M_new = unroll × M_current），且各次迭代数据独立（无跨迭代依赖）
4. M_current < 8 → **必须执行 F-19**；`M_current × 循环次数` 可提升 Cube 利用率时优先选大 unroll

**不适用情况**：
- 循环维度无法映射到 matmul M 轴（如映射到 N/K/Vector 维度）
- 迭代间存在数据依赖（后次迭代依赖前次迭代结果）
- M_current ≥ 8（已达标，无需批量化）

**⛔ 渐进变换（每步精度门控）**——⛔ 禁止一次性大爆炸变换（一次性完成全部变换后精度失败无法定位根因）。**执行纪律**：每步完成后先跑精度测试（**Steps 1-4 期间不测性能**），PASS 才进入下一步；**若某操作精度失败，仅回退该操作**（保留在 for range 内）；已批量化的操作保留：

**Step 1（结构改造，无批量化）**：确保目标循环为 `pypto.loop_unroll` 形态 + 添加 `for range(unroll)` 中间循环
- 若是 `pypto.loop`：**自行改写** `pypto.loop(0, N, 1, name="L")` → `pypto.loop_unroll(0, N, 1, name="L", unroll_list=[U1, U2, ..., 1])`
- 循环体内**保持原有逐迭代操作不变**（每迭代做与改造前相同的工作，建立代码结构）
```python
for idx, unroll in pypto.loop_unroll(0, t, 1, name="LOOP",
                                      unroll_list=[4, 2, 1]):
    for bu in range(unroll):
        x = pypto.view(src, [1, N, D], [bu, 0, 0])
        y = pypto.matmul(x, weight)  # M=1，Cube 利用率低（暂不批量化）
        z = pypto.cast(y, pypto.DT_BF16)
        pypto.assemble(z, [idx + bu, 0, 0], out)
```
- ⛔ 精度门控：FAIL → `loop_unroll`/`for range` API 本身问题，与批量化逻辑无关

**Step 2（外提，仍逐迭代）**：将可外提的操作移出 `for range`
- view 操作：若 view 的输入可批量重构（连续 → 单一批量 view；非连续 → assemble 收集后批量 view），将 view 移出 `for range`，为 Step 3 批量化做准备
- 静态操作：仅依赖部分循环变量的操作链（如 causal 阈值 [S1,S2] 只依赖 s1）——按 F-21 外提为一次性预计算，避免批量化后逐 unroll 块重复计算放大开销
- 外提后按 F-20 Rule 4 Step 1 修正相关 view 的 vec tile（UB 装载公式 `rows = UB_budget // (col × dtype_bytes)`）
- ⛔ 精度门控：FAIL → 外提/重构 bug

**Step 3（逐操作批量化分析）**：逐行分析 Step 1 添加的 `for range` 循环体内的每个操作。对每个操作：

(a) 判定是否可批量化：
  - 输入可重构为批量 tensor：连续 → 单一 `view([unroll×M_current, ...])`；非连续 → 通过 `assemble` 将逐迭代输入收集到独立 GM buffer，再从 GM buffer 批量 `view`
  - 操作支持批量执行（matmul M 轴、elementwise broadcast 等）
  - 无阻碍统一的逐迭代控制流

(b) 可批量化：替换为批量操作（M = unroll × M_current）+ 更新 tile 匹配新 shape + 移出 `for range`。若批量输出被逐迭代操作消费：`assemble` 到 GM buffer 供逐迭代 `view`。

(c) 不可批量化：保留在 `for range` 内。若不可批量化操作前后存在可批量化操作：在不可批量化操作周围创建独立 `for range` 循环；前后的可批量化操作移出，通过 GM-assemble 管线连接（批量输出 → GM buffer → 逐迭代 `view`；逐迭代输出 → GM buffer → 批量 `view`）。

(d) 条件批量化：若某操作在条件的一个分支可批量化（如连续布局）但在另一个分支不可（如非连续布局）：仅在不可批量化的分支保留 `for range`；仅对 gather/assembly 步骤分支；可批量化操作共享（不复制）。避免增加条件上下文和代码重复。

(e) 每个操作变更后：运行精度测试。若精度失败：仅回退该操作（保留在 `for range` 内）；已批量化的操作保留。

⛔ **禁止 `for range` 直接 view `loop_unroll` 体内的 local tensor（scope hazard：编译通过但读错数据，大 max_diff）**——须用独立 GM buffer 作为 `assemble` 目标。
⛔ `view(X) → assemble(..., X)` 是环（constraint 13）——须用独立 `pypto.tensor` 作为 `assemble` 目标（详见 关键启发 §4）。
⛔ 禁止一次性大爆炸变换——逐操作处理。

**Step 4（M_new 确认）**：确认 M_new = unroll × M_current，供 P2 tile 优化按 §2/§3 推导 tile 值
- M_new = unroll × M_current 代入 row 维 UB 预算除法（view/assemble 站点用全部 UB_size（§一），普通计算 op 用 UB_size // 3）/ cube 三重积
- 精度复验 + 普遍性检查（M_new 随 case class 变化，逐 case 代入）

**Step 5（性能测量——仅此步测性能）**：全部精度 PASS 后才测性能（3 次取中位）

**收益来源**：
- ✅ 消除 Python for-range 循环开销
- ✅ matmul M 从 M_current 增大到 unroll × M_current（Cube 利用率提升）
- ✅ 减少 matmul 调用次数 → 减少 cube→vector 与 vector→cube 交接次数 → **降低 AIV↔AIC 交接延迟**（每次交接 = GM 读 roundtrip + 调度等待，批量化后交接次数按 M 增大倍数缩减）
- ✅ 单次 assemble 替代多次 per-element assemble
- ✅ unroll=1 时跳过 transpose，使用 reshape（零开销）
- ✅ 静态前缀外提消除批量化后的重复计算

**unroll_list 选择**（同 F-8，含编译爆炸防护）:
- `[64, 32, 16, 8, 4, 2, 1]`: 默认起始（每迭代维度 < CUBE_ALIGN（§一））
- `[32, 16, 8, 4, 2, 1]`: 循环次数不被 64 整除时
- `[16, 8, 4, 2, 1]`: 较小循环次数
- `[1]`: 最小（无批量化，仅 reshape 无 transpose）
- ⛔ **编译爆炸防护 + 全 case 普遍性（同 F-8）**：当 M_current ≥ CUBE_ALIGN（§一）时缩减 unroll_list 变体数（变体选择规则 + 全 case 普遍性检查 + fallback if/elif/else + 循环消除策略见 F-8 §1.2.2 及 optimization_catalog.md F-8）。F-19 特有：M_new = unroll × M_current 随 case 变化——大 loop_count 用小 unroll → M_new 较小 → tile 也不同（P2 tile 分支须覆盖全部 case）；large_unroll 具体值须在目标 case 上实测。
- 规则: 必须在最内层 loop_unroll；最大值不超过循环次数；1 必须在列表中

**⚠️ 约束**：
- 每次迭代必须数据独立（无跨迭代依赖）
- 批量 tensor 须满足 UB/L1 内存约束
- transpose 开销须被 matmul M 增大收益覆盖（unroll=1 时跳过 transpose）
- 所有 tile 须更新以匹配新 tensor shape
- 不同 unroll 因子可能需要不同 tile（用条件分支）
- 改多值 `unroll_list` 后若精度回归，立即回退该 loop
- M_current > 1 批量化时，view 重构须保证内存连续性或使用 GM-assemble 管线收集非连续数据（per-iter 结果 assemble 到独立 GM buffer → `for range` 从 GM view 消费）
- 本变换改变最内层循环体（结构级变更）→ 采纳后须在**结构变更台账**中标记 `结构级`（由编排器在阶段交接后触发 FRONTEND 重跑，见编排器 PHASE_SUMMARY_F 门控）

###### 1.2.5 Mode B: 循环维度 tile 扩展 + 操作外提（独立于 F-8）

**核心思路**：任何 `for range` 或 `pypto.loop_unroll` 循环体内，若操作的 tile 可扩展到覆盖循环的迭代单元，且操作使用共享/可广播输入，则可通过 reshape 将循环维度纳入 tile，批量执行操作并**外提到循环外**。**不依赖 F-8**，可独立检查和应用。

**适用循环类型**:
- `for range(N)` — 扩展 tile 覆盖全部 N 次迭代（同一编译 scope）
- `pypto.loop_unroll(N, unroll_list=[U])` — 扩展 tile 覆盖 unroll 块大小 U（task scope 内）
- `pypto.loop(N)` — ❌ **不适用**（每次迭代为独立 root function，无法跨迭代扩展 tile）

**触发条件（全部满足）**:
1. 循环体内存在操作（任意类型：matmul / elementwise / reduction 等），且该操作的 tile **未覆盖**当前循环的迭代单元
2. 操作至少有一个输入为**共享输入**（不依赖循环变量 — 如共享权重 W、cos/sin、gamma、常量等）
3. 循环变量仅通过 `pypto.view` 的**规律步长 offset** 选择数据切片（不转换数据）
4. 源数据可通过 reshape 将循环维度纳入新维度（规律 stride 布局）
5. 共享输入可**广播**到循环维度（或通过 reshape 匹配）
 6. 扩展后的 tile 不超出内存约束（UB_size / L1_size（§一））

**不适用情况**:
- 操作使用**逐迭代不同权重**且无可用的批量操作（如逐 head 不同 weight 的 matmul，且 `transposed_batchmatmul` 不支持当前 dtype → 保留循环）
- 数据切片为**不规则 stride**（无法通过 reshape 视为单个 tensor）
- 扩展 tile 超出内存约束
- 循环维度**是归约轴**（对 reduction 操作，扩展会改变语义）

**变换步骤**:

**Step 1**: Reshape 源数据，将循环维度纳入新维度
```python
# for range(N) → 覆盖全部迭代
src_3d = pypto.reshape(src, [dim0, N, stride_dim])
# pypto.loop_unroll(N, unroll_list=[U]) → 覆盖 unroll 块
src_3d = pypto.reshape(src, [dim0, U, stride_dim])
```

**Step 2**: View 所有迭代的数据切片为单个 tensor
```python
slice_all = pypto.view(src_3d, [dim0, N_or_U, slice_dim], [0, 0, offset])
```

**Step 3**: 扩展操作的 tile 以覆盖循环维度
```python
pypto.set_vec_tile_shapes(dim0, N_or_U * slice_dim)  # elementwise（概念描述——实际按 §1b 四步流程分档，放置见 §1b）
# 或 pypto.set_cube_tile_shapes(..., 含扩展维度)  # matmul
```

**Step 4**: 执行操作一次（共享输入自动广播到循环维度）

**Step 5**: Reshape 结果回 2D（如需要）并 assemble 到输出

**Step 6**: 将操作从循环体内**外提**到循环外

**代码示例 — elementwise 批量外提（RoPE 模式）**:
```python
# BEFORE: N_HEADS 次小 elementwise 操作在循环内（共享 cos/sin）
for n in range(N_HEADS):
    q_r_n = pypto.view(qr, [tile_bs, Dr], [0, n * stride + D])
    x1 = pypto.view(q_r_n, [tile_bs, Dr // 2], [0, 0])
    x2 = pypto.view(q_r_n, [tile_bs, Dr // 2], [0, Dr // 2])
    neg_x2 = pypto.mul(x2, -1.0)           # 共享常量
    x1_mat = pypto.add(x1, 0.0)            # 共享常量
    rot = pypto.concat([neg_x2, x1_mat], -1)
    y = pypto.add(pypto.mul(q_r_n, cos_fp32),    # 共享 cos
                  pypto.mul(rot, sin_fp32))       # 共享 sin
    pypto.assemble(y, [bs_offset, n * Dr], out)

# AFTER: 1 次批量操作外提到循环外
qr_3d = pypto.reshape(qr, [tile_bs, N_HEADS, stride])
q_r_all = pypto.view(qr_3d, [tile_bs, N_HEADS, Dr], [0, 0, D])
# N_HEADS * Dr is a BinOp → rejects as tile arg; use if/elif/else:
if N_HEADS * Dr >= 1024:
    nh_dr = 1024
elif N_HEADS * Dr >= 512:
    nh_dr = 512
else:
    nh_dr = 256
pypto.set_vec_tile_shapes(tile_bs, nh_dr)   # tile_bs must also be int literal/Assign
x1_all = pypto.view(q_r_all, [tile_bs, N_HEADS, Dr // 2], [0, 0, 0])
x2_all = pypto.view(q_r_all, [tile_bs, N_HEADS, Dr // 2], [0, 0, Dr // 2])
neg_x2_all = pypto.mul(x2_all, -1.0)       # 广播
x1_all_mat = pypto.add(x1_all, 0.0)        # 广播
rot_all = pypto.concat([neg_x2_all, x1_all_mat], -1)
y_all = pypto.add(pypto.mul(q_r_all, cos_fp32),    # cos 广播
                  pypto.mul(rot_all, sin_fp32))     # sin 广播
y_2d = pypto.reshape(y_all, [tile_bs, N_HEADS * Dr])
pypto.assemble(y_2d, [bs_offset, 0], out)

# 循环内仅保留不可批量化的操作（逐 head 不同 weight 的 matmul）
for n in range(N_HEADS):
    q_c_n = pypto.view(qr, [tile_bs, D_HEAD], [0, n * stride])
    w_uk_n = pypto.view(w_uk, [D_HEAD, HCKV], [n * D_HEAD, 0])
    q_n = pypto.matmul(q_c_n, w_uk_n)  # 不同 weight per head → 不可批量
    pypto.assemble(q_n, [bs_offset, n * HCKV], query)
```

**代码示例 — 共享权重 matmul 批量外提**:
```python
# BEFORE: N 次小 matmul（M=1，共享权重 W）
for i in range(N):
    x = pypto.view(src, [1, K], [i, 0])
    y = pypto.matmul(x, W)   # M=1, W 共享
    pypto.assemble(y, [i, 0], out)

# AFTER: 1 次批量 matmul 外提（M=N）
x_all = pypto.view(src, [N, K], [0, 0])
pypto.set_cube_tile_shapes([16, 16], [128, 128], [128, 128])
y_all = pypto.matmul(x_all, W)   # M=N, W 共享
pypto.assemble(y_all, [0, 0], out)
```

**收益来源**:
- ✅ 消除 N 次小操作的调度开销（root function 数量减少）
- ✅ 操作 tile 从覆盖 1 个元素扩展到覆盖 N 个元素（tile 利用率提升）
- ✅ 共享输入只需加载一次（而非每迭代重复加载）
- ✅ 单次 assemble 替代 N 次 per-element assemble

**⚠️ 约束**:
- 每次迭代必须**数据独立**（无跨迭代依赖）
- 共享输入须支持广播到循环维度（否则需先 reshape/expand）
- 批量 tensor 须满足 UB/L1 内存约束
- 须更新所有 `set_vec_tile_shapes` / `set_cube_tile_shapes` 以匹配新 tensor shape
- 外提后循环内剩余操作须保持正确数据依赖
- `pypto.loop`（不含 unroll）**不可用** — 每次迭代为独立 root function

##### 1.3 尽可能合并 loop

检查算子代码是否有可以合并的 loop 块：

```python
# ❌ 不推荐：两个独立的 loop
bsz = x1.shape[0]
for b_idx in pypto.loop(bsz, name="LOOP_1", idx_name="b_idx"):
    out_1 = Operation1(x1[b_idx, :], y)
for b_idx in pypto.loop(bsz, name="LOOP_2", idx_name="b_idx"):
    out_2 = Operation2(x2[b_idx, :], y)

# ✅ 推荐：合并 loop
for b_idx in pypto.loop(bsz, name="LOOP_1", idx_name="b_idx"):
    out_1 = Operation1(x1[b_idx, :], y)
    out_2 = Operation2(x2[b_idx, :], y)
```

#### 2. Reshape 全局优化

> **⛔ 执行 F-4 Reshape 优化时，必须加载 [Reshape 全局优化](references/reshape-global-optimization.md) 获取完整操作指南。**

#### 2.1 操作链依赖深度外提（F-21）

> **⛔ 执行 F-21 依赖深度外提时，必须加载 [操作链依赖深度外提](references/loop-hoisting.md) 获取完整操作指南。**

#### 3. 基本块优化

> **⛔ 执行 F-9/F-10 TileShape 优化时，必须加载 [基本块优化](references/basic-block-optimization.md) 获取完整规范。**

### 局部性能优化（P3: F-11~F-15）

#### 局部§1. 输入矩阵格式优化（对应优化点 F-12）

检查输入矩阵、尤其是 Shape 较大的权重矩阵是否可以提前以 NZ 格式存储。

**NZ 格式的数据搬运到 L1 的带宽更高。**

**调优方法**：
- 算子入口处对权重矩阵调用 `tensor.to_format(pypto.NZ)` 转换格式
- 要求权重矩阵在算子调用前已按 NZ 格式存储在 HBM 中
- NZ 格式的数据搬运到 L1 的带宽更高，适合只读一次的大权重矩阵

#### 局部§2. Transpose 优化（对应优化点 F-13）

矩阵乘前后有 transpose 时，可以尝试更换左右矩阵并使用左右矩阵转置的配置。

当 M 轴较大、N 轴较小时，使得左右矩阵有更大的尾轴，提升搬运带宽。

**⚠️ 重要原则**
- `transpose + matmul` 的结构，可以通过 matmul 的 `a_trans` 及 `b_trans` 参数进行配置，完成 op 融合。好处是，matmul 运算时，可以随路 transpose

**代码示例**：

```python
# ❌ 不推荐：先 transpose 再 matmul
b_t = pypto.transpose(b, [1, 0])
y = pypto.matmul(a, b_t)

# ✅ 推荐：matmul 参数直接带转置
y = pypto.matmul(a, b, b_trans=True)

# 当 M 轴大于 N 轴时，交换左右矩阵
# A: [M, K], B: [N, K]
# y = matmul(A, B, b_trans=True)  →  Matmul shape: [M, N]
# 等价于 y^T = matmul(B, A, a_trans=True)  →  Matmul shape: [N, M]
```

#### 局部§3. 冗余搬运优化（对应优化点 F-14）

检查是否有不合理数据操作导致的冗余搬运：

**优化方法**：
- 更换 concat 为 assemble：当目标仅是将数据拼接到已有 tensor 的指定位置时，`pypto.assemble` 比 `pypto.concat` 开销更低
- 识别 concat 拼接后不再修改的场景，替换为 assemble 直接写入

**判断依据**：
- `concat`：需要分配新内存 + 数据搬运，开销高
- `assemble`：直接写入目标位置，零额外内存分配

#### 局部§4. 尾轴 Broadcast 合轴优化（对应优化点 F-15）

##### 局部§4.1 问题诊断

当算子中存在形如 `[M, 1] * [M, N]` 的尾轴 1 broadcast 二元运算时，默认编译策略会先将 `[M, 1]` 展开（broadcast）为 `[M, N]` 再执行计算，多一次数据搬运。

**典型来源**：online softmax 中的 `sum_update`/`max_update`（`pypto.sum(keepdim=True)` / `pypto.amax(keepdim=True)` 产生），形如 `[g_tile, 1]`。

**诊断步骤**：
1. 扫描算子所有 tensor shape，标记 shape 尾轴为 1 的 tensor
2. 追踪该 tensor 的参与的所有二元运算（mul/add/sub/div），检查另一侧 tensor 尾轴是否 >1
3. 确认尾轴 1 的 tensor 是否由前序 reduce 操作（`sum`/`amax` 等 + `keepdim=True`）产生（保证 GM 连续）

##### 局部§4.2 优化方法

在 JIT 函数体首行添加：

```python
pypto.experimental.set_operation_options(combine_axis=True)
```

编译器会将 `[32,1] + [32,128]` 优化为：`[32,1]` 通过 brcb 指令扩展到 `[32,8]`，再做 `[32,8] + [32,128]`，省去完整 broadcast 的数据搬运。

##### 局部§4.3 约束条件

- 尾轴 broadcast 输入尾轴**必须连续**，否则功能失效
- `pypto.sum(keepdim=True)` / `pypto.amax(keepdim=True)` 输出保证连续，符合条件
- 若前序是 COPY_IN，需在前端保证 GM 连续
- 设置是**局部**的，只影响当前 jit/loop 作用域

##### 局部§4.4 收益预期

- ✅ **Vector 密集算子**（纯 softmax、layer norm、激活函数）：预期有明显收益
- ✅ **尾轴 1 broadcast 位于内层热循环**且执行次数多：预期有收益
- ⚠️ **Cube 密集型算子**（大 matmul 占 >80% 时间）：收益有限，vector 操作非瓶颈

##### 局部§4.5 案例

详见 [尾轴 Broadcast 合轴优化案例](cases/combine-axis-broadcast.md)

### 多 Matmul 差异化 Cube TileShape（对应优化点 F-16）

> **适用场景**：算子含 2 个及以上 Matmul（如 attention 类的 QK^T + PV 模式）。

#### 诊断方法

逐个分析每个 Matmul 的 M/N/K 轴大小，填写下表：

| # | Matmul | M 轴 | N 轴 | K 轴 | 当前 cube tile | 问题 |
|---|--------|------|------|------|---------------|------|
| 1 | C1 (QK^T) | 128 | s2_tile | dn+dr | [128,128],[128,128],[128,128] | L1 太小，K 轴数据重复搬运 |
| 2 | C2 (PV) | 128 | dn | s2_tile | [128,128],[128,128],[128,128] | K_L1 太小，重复载入 |

#### 优化原则

**每个 Matmul 的 cube tile 应根据其 M/N/K 特征独立配置**：

1. **K 轴较大（>128）的 Matmul**：L1 调大（如 256），让 A/B 矩阵驻留 L1，减少重复载入
2. **K 轴与另一个 Matmul 的 N 轴相同的 Matmul**：K_L1 可调大复用前序加载的数据
3. **禁忌**：所有 Matmul 使用完全相同的 cube tile 配置（除非 M/N/K 完全一致）

**公式化要求（⛔）**：每个 matmul 的 M/K/N 是**不同的 case 标量函数** → 每个 matmul 必须独立执行 references/basic-block-optimization.md §2「Cube Tile 公式推导方法」得到自己的分档分支实例（§1b 四步流程，普遍性检查逐 matmul 进行）；禁止所有 matmul 共用一组分支或字面常量（除非 M/K/N 函数完全一致）。下方配置示例为量级示意，实际值以分档推导结果为准。

#### 配置示例

```python
# C1 (QK^T): M=128, N=s2_tile(512), K=576 → L1 调大让 K 驻留
pypto.set_cube_tile_shapes([128, 128], [256, 256], [128, 128])

# C2 (PV): M=128, N=512, K=s2_tile(512) → K_L1 调大复用
pypto.set_cube_tile_shapes([128, 128], [128, 128], [256, 256])
```

### V 段内多 shape 分段 vec tile（对应优化点 F-17）

> **适用场景**：同一 V 段内处理的 tensor shape 发生变化（如 online softmax 中 [M,N]→[M,1]→[M,N] 交替）。

#### 诊断方法

逐行扫描 V 段内每个 operation 的输入/输出 tensor shape，标记所有 shape 变化点：

| # | operation | 输入 shape | 输出 shape | 当前 vec tile | shape 变化? | 问题 |
|---|-----------|-----------|-----------|-------------|-----------|------|
| 1 | amax(sij) | [128, 512] | [128, 1] | [32, 512] | ✅ | [128,1] ops 用 [32,512] tile 大量 padding |
| 2 | sub(sij, max) | [128, 512] | [128, 512] | [32, 512] | 否 | 无问题 |
| 3 | mul(oi_update, e_old) | [128, 512] | [128, 512] | [32, 512] | 否 | 无问题 |

#### 优化原则

1. **shape 变化时必须重设 vec tile**：每次 tensor shape 从 [M,N] 变为 [M,1] 或反向变化时，必须重新设置匹配的 vec tile。**⛔ 任务数爆炸机制（重设的性能依据）**：[M,1]→[M,N] 反向变化时若继承小列 tile（col=1），[M,N] tensor 被切成 **N 个子任务**（而非 1 个）——任务数爆炸、调度开销大；[M,N]→[M,1] 方向继承大列 tile 只浪费 UB 不爆炸，但仍应重设以最小化 UB 分配。**正确重设 = 每个变化点 1 任务**（与 Rule 4/A4 task-count ratio 同一任务数判据）
2. **归约类 [M,1] ops**：用大行数小列数 tile（如 [128, 128]），匹配 reduction 输出 shape
3. **elementwise [M,N] ops**：用匹配 N 轴的 tile（如 [32, 512]），用满尾轴带宽
4. **⚠️ 嵌套表达式必须展平后才能分段**：若 V 段内使用嵌套表达式（如 `pypto.add(pypto.mul(a,b), pypto.mul(c,d))`），编译器将整个嵌套视为一个不可分割的 op，无法在中间插入 `set_vec_tile_shapes`。必须先展平为独立中间变量，才能在 shape 切换点插入 vec tile 分段。**展平是分段的必要前置条件，不展平则分段无法生效**。

#### 展平方法

```python
# ❌ 嵌套表达式：编译器视为一个整体，无法在中间切tile
# [128,1] ops 和 [128,512] ops 混在嵌套中，编译器用开头设的tile统一处理
sum_new = pypto.add(pypto.mul(e_old, sum_update),     # [128,1]
                    pypto.mul(e_local, sum_local))     # [128,1]
oi_new = pypto.add(pypto.mul(oi_update, e_old),       # [128,512] ← 用[128,128]处理，严重padding
                    pypto.mul(q1, e_local))            # [128,512]

# ✅ 展平为独立变量：可在shape切换点插入set_vec_tile_shapes
pypto.set_vec_tile_shapes(128, 128)        # [128,1] ops
t1 = pypto.sub(max_update, max_new)         # [128,1]
t2 = pypto.exp(t1)                          # [128,1]
t6 = pypto.mul(t2, sum_update)              # [128,1]
t3 = pypto.sub(m_local, max_new)            # [128,1]

pypto.set_vec_tile_shapes(32, 512)         # ← 切换到匹配[128,512]的tile
t4 = pypto.exp(t3)                          # [128,1]
t5 = pypto.mul(t4, sum_local)               # [128,1]
oi_last = pypto.mul(oi_update, t2)          # [128,512] ← tile匹配
oi_flash = pypto.mul(q1, t4)                # [128,512] ← tile匹配
oi_tmp = pypto.add(oi_last, oi_flash)       # [128,512] ← tile匹配
```

**展平的核心价值**：不是为了VF融合，而是为了在 [M,1] ops 和 [M,N] ops 之间插入 vec tile 切换。不展平时编译器用统一tile处理所有ops，导致 [M,N] ops 用 [M,1] 的tile产生严重padding。适用于任何 V 段内存在归约输出 [M,1] 和 elementwise 输出 [M,N] 交替的场景。

#### 配置示例

```python
# V2 段：先做 [128,1] 归约类操作
pypto.set_vec_tile_shapes(128, 128)    # 匹配 [128,1] 输出
max_new = pypto.maximum(max_update, tilda_mij_reduce)
t1 = pypto.sub(max_update, max_new)
t2 = pypto.exp(t1)

# 切换到 [128,512] elementwise 操作
pypto.set_vec_tile_shapes(32, 512)     # 匹配 [128,512] 输出
oi_last = pypto.mul(oi_update, t2)
oi_flash = pypto.mul(q1, t4)
oi_tmp = pypto.add(oi_last, oi_flash)
```

### 语义维度的静态循环保留（对应优化点 F-18）

> **适用场景**：算子有 n_kv/group 等语义维度的静态循环（即使迭代次数=1）。

#### F-5 的例外原则

F-5 说"静态轴改 Python for"，但以下情况的静态循环应**保留为 `pypto.loop`**：

| 保留条件 | 原因 | 示例 |
|---------|------|------|
| 循环变量在 tensor shape 或 view offset 中被使用 | 编译器需要识别这些维度的语义边界 | `cur_offset = ... + n_kv_idx * group + group_idx * cur_group_tile` |
| 循环代表模型语义维度（n_kv, group, head） | 影响 root function 划分和数据流分析 | `for n_kv_idx in pypto.loop(0, n_kv_sym, 1, ...)` |

#### 判断标准

```
该循环变量是否在 tensor shape 或 view offset 中被使用？
├─ 是 → 保留 pypto.loop（即使迭代1次）
└─ 否 → 可改为 Python for（按 F-5 处理）
```

#### 代码示例

```python
# ✅ 保留语义循环（即使 n_kv=1, group_loop=1）
for n_kv_idx in pypto.loop(0, n_kv_sym, 1, name="LOOP_n_kv"):
    for group_idx in pypto.loop(0, g_loop_sym, 1, name="LOOP_group"):
        # 循环变量参与 offset 计算 → 保留
        cur_offset = batch_idx * s1_n2_gsym + slc_idx * nq + n_kv_idx * group + group_idx * cur_group_tile
```

### 邻居算子感知的 TileShape 优化（对应优化点 F-20）

> **适用场景**：matmul 后接 reduce / matmul 后接 vector(无reduce) / 多个并发 matmul / 存在 view/assemble/reshape 链。
> **前置条件**：F-9（baseline cube tile）和 F-10（baseline vector tile）已完成。
> **Buffer 容量约束**：cube tile 受 L1/L0A/L0B/L0C 限制，vec tile 受 UB 限制（容量见 [shared/npu-memory-arch.md](../shared/npu-memory-arch.md) §一）。

#### 诊断（脚本自动）

`list_tile_boundaries.py` 自动枚举所有 matmul 邻居链，每行附规则标签 + 公式建议。cube→vector 边界：R1（下游首个 op 为 reduce）/ R3（下游纯 vector）。vector→cube 边界：R4（view feeds matmul，附 view→matmul 对齐核对 + A/B 操作数 a_trans/b_trans 映射）。每行包含 view 前 active vec tile 与 view shape 供对比。

#### Rule 1: Matmul→Reduce — 最大化 n，减小 k

- **触发**：matmul 输出在下一个 matmul 或计算结束前经过 reduce（sum/amax/amin/mean）。
- **策略**：最大化 nL0/nL1（减少 N 轴切分 → 减少 partial reduce 汇聚开销），减小 kL0 腾出 L0B 空间。
- **变体**：kL0 ∈ {64,128,256} × nL0 ∈ {128,256}，受 L0B/L0C/L1 约束（§一）。
- **回退**：沿 L0B 容量边界修改 F-9 分支参数（nL0 cap 减半 ↔ kL0 cap 放大），逐档实测；每 site ≤2-3 档。

#### Rule 2: 并发 Matmul — 核数匹配

- **触发**：多个 matmul 共享相同依赖（同一 ready tensor / 同一 vec/cube 操作输出），可并发执行。
- **公式**：`kernel_count = Σ ceil(M_i/mL1_i) × ceil(N_i/nL1_i)`（L1 tile 决定核分派粒度）。
- **Cube 核数**：cube_units（§二）

| kernel_count vs cube_units | 策略 |
|---------------------------|------|
| `< cube_units` | 减小 mL1 优先，然后 nL1 |
| `≤ 2×cube_units` | 合理，微调 |
| `> 2×cube_units` | 增大 nL1 优先，然后 mL1 |

- **⚠️ L0/L1 耦合**：`L0<=L1 && L1%L0==0`（compile-time）；mL0 ≥ 64（S-14：优先 128，禁 CUBE_ALIGN）；`nL1%nL0==0`；L1 ≤ L1_size（§一）。详见 [merge-optimization.md](../tune-swimlane/references/merge-optimization.md) §4.2。

#### Rule 3: Matmul→Vector(无reduce) — vec tile 跟随 cube tile

- **触发**：matmul 输出只经过 vector 操作（cast/mul/add/concat），无 reduce，到下一个 matmul 或结束。
- **策略**：`vec_tile = [cube_mL0, cube_nL0]`；UB 溢出时**递归减 M**（`m/2→m/4→...→1`，保持 `cube_mL0%vec_m==0`），M=1 仍溢出则**减 N**（`n/2→...→1`）。
- **为什么递归（非一步 `[m/2,n]`）**：L0C > UB 的平台（§一）cube 满载 L0C 的 tile 作为单 tensor 即超 UB，带 intermediates 常需多次减行；L0C < UB 的平台通常 0-1 次 fit。
- **UB 预算**：普通计算 op = UB_size // 3（§一）；view/assemble/reshape 站点 = UB_size（§一，op 本身不占 UB）。
- **S-14 caveat**：Mix合图场景 vec tile 两维须 CUBE_ALIGN 对齐（§一）（`vec_m∈{1,16,32,...}`，1 为 floor 例外）；非 Mix 场景 vec_m 可取 8/4/2。

```python
# Example pattern (values are case-derived per §1b):
pypto.set_cube_tile_shapes([mL0, mL1], [kL0, kL1], [nL0, nL1])
c = pypto.matmul(x, w, pypto.DT_FP32)
# vec follows cube; halve M until vec_m×vec_n×dtype ≤ UB_effective
pypto.set_vec_tile_shapes(vec_m, vec_n)  # cube_mL0%vec_m==0 ✅
y = pypto.cast(c, pypto.DT_BF16)
```

#### Rule 4: View/Assemble/Reshape — 显式 set（consumer GM→UB 装载粒度）

- **触发**：kernel 中存在任意 `view`/`assemble`/`reshape`。
- **判据**：consumer GM→UB 装载任务数（非 "view 是否零搬移"，非 "ambient 是否覆盖"）。ambient 覆盖 view shape 只保证数据完整，**不保证装载任务数最优**（如 ambient `(4,4096)` 对 view `[4096,128]` 截断为 `(4,128)` → 1024 任务 vs Step 1b `(768,128)` → 6 任务）。
- vec tile 与 cube tile 是**两条独立机制**——"consumer 是 matmul 所以 vec tile 不适用"的推理**不成立**（matmul 输入侧 view 仍需 vec tile 设置）。
- **与 Rule 3 优先级**：view/assemble 直接消费 matmul 输出（CV 衔接）时 Rule 3 优先；Rule 4 适用于所有 view/assemble/reshape 站点（含 matmul 输入侧 view）。

**Step 1：整shape tile**（每站点强制）：
- **1a**：`whole_rows × col × dtype_bytes ≤ UB_size` → `set_vec_tile_shapes(whole_rows, col)`
- **1b**：whole > UB → `rows = UB_size // (col × dtype_bytes)`；col 保持整shape（不切分）
- **模式**：`set → view → reset`。不 reset 时残留 tile 污染下游站点（小 shape view 继承大 row tile → 截断 → 任务膨胀）。
- **UB 预算**：view/assemble/reshape op 不占 UB → 用全部 UB_size（§一）。

⛔ **门控**：每站点须有 ITER 三元组——(a) Step 1a/1b 公式 set 值 + (b) ITER 前后实测 µs + (c) 采纳/回退。任何**无实测数字**支撑的"无需设置/已覆盖/不适用/继承即可"结论（无论措辞）一律**无效**。历史规避措辞（非穷举）："ambient 已覆盖"、"邻居传播"、"整shape已试"、"cube tile applies"、"ambient sufficient/covers"、"vec tile 对 matmul 输入不适用"。判据是实测数字而非措辞。

**Guard**：reshape 尾轴须切分时 → 真实搬移（COPY_OUT→reshape→COPY_IN）→ 结构化改造：F-4 / I-4 / 对齐尾轴。

**链内无 COPY 条件 + Step 2 回退详情**：见 [catalog](../shared/optimization_catalog.md) [F-20] Rule 4。

```python
# Example pattern: set → view → reset (if/elif/else, §1b four-step flow)
if Dk >= 128:
    v_row, v_col = 768, 128     # = UB_size // (col × dtype_bytes), precomputed
elif Dk >= 64:
    v_row, v_col = 768, 64
else:
    v_row, v_col = 768, 32
pypto.set_vec_tile_shapes(v_row, v_col)
k_row = pypto.view(input, [S2, Dk], [offset, 0])
pypto.set_vec_tile_shapes(ambient_row, ambient_col)  # ⛔ reset

# ❌ pypto.set_vec_tile_shapes(192 * 1024 // (Dk * 2), Dk)  # BinOp
```

#### view→matmul alignment（⛔ 强制，须 trace-time assert）

view 的 vec_tile 各维度必须 == matmul 对应 L1 tile 维度（A 按 a_trans、B 按 b_trans 映射）。不匹配时在未测试 case 上产生非确定性精度错误。完整映射表、原因与代码示例见 [basic-block-optimization.md](references/basic-block-optimization.md) §3。

#### S-14 Mix合图 兼容性检查清单

> ⚠️ 若算子可能在 A5 平台（DAV_3510）上使用 S-14 Mix合图，修改任何 tile 后必须逐项检查。详见 [tune-swimlane/references/merge-optimization.md](../tune-swimlane/references/merge-optimization.md) §4.2。

| # | 检查项 | 适用规则 | 约束 | 来源 |
|---|--------|---------|------|------|
| 1 | nL0 是 vec tile n 的整数倍（或通过 Rule 3 同步 vec tile） | Rule 1, 2 | §4.2.5b IsDimMultiple | merge-optimization.md:931 |
| 2 | mL0 ≥ 64（优先 128），禁止 CUBE_ALIGN | Rule 2 | 破坏 CV 通路 | merge-optimization.md:952, 1060 |
| 3 | nL0 满足 Check 1 同样约束 | Rule 2 | §4.2.5b | merge-optimization.md:931 |
| 4 | vec tile 两维均 CUBE_ALIGN 对齐（§一） | Rule 3 | §4.2.5a CheckUBTileShape | merge-optimization.md:930 |
| 5 | UB ≤ UB_size（§一）— 独立于 L0C 是否溢出。溢出则递归减 M（m/2→m/4→...→1，M=1 仍溢出减 N；L0C > UB 的平台常需多次，非一步 `[m/2,n]`） | Rule 3 | §4.2.3 | merge-optimization.md:915 |
| 6 | UB→L1 内轴 32B 对齐（若 vector 回喂 cube） | Rule 3 | §4.2.5c | merge-optimization.md:932 |
| 7 | 若 S-14 可能使用（A5），优先标准值（~128）避免 SWIMLANE 阶段重置 | All | 避免非标准值重置 | merge-optimization.md:1056 |

#### 执行顺序

Rule 1 & 2 → Rule 3 → Rule 4 → S-14 兼容性检查（详见 [catalog](../shared/optimization_catalog.md) [F-20]）。

## 调优检查清单

**⛔ 必须按以下清单逐项执行。每项标记为 ✅已尝试 或 ❌已失败（附原因），禁止跳过。完整优化点信息参考 [shared/optimization_catalog.md](../shared/optimization_catalog.md)。**

**🔥 P0 - 全局优化（F-1~F-4, F-16, F-17, F-21）**：
- [ ] [F-1] 是否检查了所有 Matmul 的 M/N/K 轴是否充分利用硬件（M 轴 < 8 是常见问题）
- [ ] [F-2] 是否检查了循环体内部计算量是否太小（应 unroll 或增大切块）
- [ ] [F-3] 是否检查了循环总次数（>100 应切块减少循环次数）
- [ ] [F-4] 是否对所有 reshape/squeeze/unsqueeze 逐行分析（原始输入 reshape 应外提+inplace，冗余应删除）
- [ ] [F-21] 是否对循环体内依赖深度可外提的操作链逐行分析（部分依赖操作链按依赖深度外提，详见 references/loop-hoisting.md）
- [ ] [F-16] 是否对每个 Matmul 独立配置了 cube tile（含2个及以上 Matmul 时禁止全用相同配置；每个 matmul 独立公式实例，§2 推导）
- [ ] [F-17] 是否逐行扫描 V 段内每个 operation 的 shape 变化点并在变化时重设 vec tile（重设值按 §1b 四步流程分档；每站点独立 tile 变量，新增调用成本可忽略）

**P1 - 循环优化（F-5~F-8, F-18, F-19）**：
- [ ] [F-5] 是否将静态轴改为 Python for（⚠️ F-18 例外：语义维度循环须保留 pypto.loop）
- [ ] [F-6] 是否检查了可合并的独立 loop
- [ ] [F-7] 是否对外层动态轴范围大的循环尝试了切块
- [ ] [F-8] 是否对内层动态轴尝试了 unroll_list
- [ ] [F-18] 是否检查了 n_kv/group 等语义维度循环是否被错误改为 Python for（应保留 pypto.loop）
- [ ] [F-19 Mode A] 是否对**所有循环**（`pypto.loop` / `for range` / `pypto.loop_unroll`）检查了循环体内 matmul 的 M_current < 8 且迭代数据独立（必做，不可标记"不适用"跳过）；目标循环为 `pypto.loop` 时须自行插入 unroll_list 与 range 循环（F-19 自含 unroll，不依赖 F-8）；若 Mode A 已应用，须按 §1.2.4 Step 3 重新评估当前结构。**⛔ 执行 = 渐进变换（§1.2.4）：逐操作批量化分析，每步精度门控（Steps 1-4 不测性能），某操作精度失败 → 仅回退该操作；禁止一次性大爆炸变换**
- [ ] [F-19 Mode B] 是否检查了所有 `for range` / `pypto.loop_unroll` 循环（不限 F-8）中操作的 tile 可扩展到覆盖循环迭代单元且使用共享/可广播输入（必做，不可标记"不适用"跳过；`pypto.loop` 不适用）

**P2 - TileShape 优化（F-9~F-10, F-20）**：
- [ ] [F-9] 是否为每个 Matmul 设置了 `set_cube_tile_shapes`（⛔ 值必须分档——if/elif/else 分支，§1b 四步流程：每站点独立 tile 变量、从最后一维起条件分支（case 标量 G/S2/Dk/Dv 作条件，条件中任意函数）、枚举全部 case 取值放置边界（推荐 2 的幂）、每分支 int 字面量 + fallback 常量兜底；trace 期 assert 可选；按 §2 推导方法逐 matmul 推导，禁止单一 case 字面常量或照抄推荐配置字面值；Decode M=1 用 K 轴三维配置）
- [ ] [F-10] 是否为每个 Vector 计算设置了 `set_vec_tile_shapes`？⛔ 值必须分档（if/elif/else 分支，§1b 四步流程 + §3 推导方法），禁止单 case 字面值硬编码；优先用满尾轴，reshape 前后重设。**row 维分支必须体现 UB 预算除法**：row 取「不超过 M」与「UB_budget 除以（col × dtype_bytes）」中较小值、下限 1（每分支 int 字面量 + fallback 兜底）；**UB_budget 按 op 类型区分**（⛔ 用错预算 = 未完成）：view/assemble/reshape 站点（Rule 4）= **全部 UB_size**（§一——op 本身不占 UB，仅控制 consumer 装载粒度）；普通计算 op（cast/mul/add/reduce）= **UB_size // 3**（§一——input+output+intermediates 共享 UB）；dtype 随 cast 变化重算；trace 期 assert（`row*col*dtype_bytes ≤ UB_budget`）可选推荐；⛔ row 维无 UB 预算除法（或整shape≤预算证明）= 未完成
- [ ] [F-20 Rule 1] 是否检查了 matmul→reduce 链并尝试最大化 nL0/nL1、减小 kL0（搜索 k∈{64,128,256}×n∈{128,256}，约束 L0B≤L0B_size/L0C/L1（§一））
- [ ] [F-20 Rule 2] 是否对并发 matmul 用 **mL1/nL1** 计算了 kernel_count（`Σ ceil(M_i/mL1_i)×ceil(N_i/nL1_i)`）并与 cube_units（§二）比较调整 mL1/nL1（L0 仍须满足 mL0≥64、`L0<=L1 && L1%L0==0`、L0B/L0C 预算；减小 mL1 下限=mL0，非 64）
- [ ] [F-20 Rule 3] 是否对 matmul→vector(无reduce) 链设置 vec tile 跟随 cube tile（`[mL0, nL0]`→ UB 溢出时**递归减 M** `m/2→m/4→...` 下限 1，M=1 仍溢出则**减 N**；非一步 `[m/2,n]`——L0C > UB 的平台（950PR，§一）带 intermediates 常需多次减行）
- [ ] [F-20 Rule 4] 是否对所有 view/assemble/reshape 站点**逐站点显式设置** vec_tile（Step 1a/1b 公式取值 + set→view→reset）？⛔ 无效判定为**措辞无关模式**：任何**无 ITER 前后实测 µs 数字**支撑的"无需设置/已覆盖/不适用/继承即可"结论（无论措辞）一律无效 = 该站点未评估——`list_tile_boundaries.py` 标记的每个 view 站点在 PHASE_SUMMARY_F 台账中必须有三元组记录（(a) Step 1a/1b 公式 set 值 + (b) ITER 前后 µs 实测 + (c) 采纳/回退），缺任一项 = Rule 4 未完成；且全部站点 set+ITER **在 P2 完成**（F-19 已在 P1 先行，按 M_new 后 shape 设置）；Step 1 设置失效（真实搬移）时才走 Step 2 传播回退并复检
- [ ] [F-20 S-14] 是否检查了所有修改的 tile 满足 S-14 Mix合图兼容性（衔接轴整数倍、CUBE_ALIGN 对齐（§一）、UB ≤ UB_size（§一）、UB→L1 32B对齐）

**P3 - 局部优化（F-11~F-15）**：
- [ ] [F-11] 是否检查了常量配置（BLOCK_SIZE 等；常量值同样以 case 标量公式表达，禁止单 case 字面值）
- [ ] [F-12] 是否检查了权重矩阵格式（大 Shape 可尝试 NZ）
- [ ] [F-13] 是否检查了 transpose+matmul 模式（可用 a_trans/b_trans 融合）
- [ ] [F-14] 是否检查了 concat/assemble 搬运（concat 可替换为 assemble）
- [ ] [F-15] 是否检查了尾轴为 1 的 broadcast 二元运算（可添加 combine_axis=True）

## 参考资料

- [性能调优文档](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/tutorials/debug/performance.md)
- [GDR 算子案例](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/tutorials/debug/performance_case_GDR.md)
- [Matmul 高性能编程](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/tutorials/debug/matmul_performance_guide.md)
- [典型案例库](cases/README.md)
