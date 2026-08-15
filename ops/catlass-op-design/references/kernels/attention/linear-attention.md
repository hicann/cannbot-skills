# Kernel 路由：Linear Attention 类算子

> 本指南覆盖 GDN / gated delta rule / KDA / retention / RWKV 等线性 Attention 家族算子的 catlass 设计入口。它不替代普通 Matmul、Grouped Matmul 路由；只有需求命中线性 Attention 或状态递推特征时才启用。

---

## 场景定义

线性 Attention 类算子的共同特点是：计算图中同时存在矩阵乘、逐元素门控、前缀/分块状态递推、跨 stage 中间量复用，且常见 I/O 维度不只是 `(M,N,K)`，而是 `(B, T, H, K/V)`、`BT/chunk`、`HK/HV/GQA` 等模型维度。

触发信号包括：`linear attention`、`flash linear attention`、`GDN`、`gated delta rule`、`delta rule`、`KDA`、`Kimi Delta Attention`、`retention`、`RWKV`、`chunk_gdn`、`cumsum/scan inside attention`、`state recurrence`。

### 子场景路由

命中 Attention / State Recurrence 大类后，再按子场景加载更窄的 reference，避免把所有同族算子的细节堆在同一个入口文档里：

| 子场景 | 读取 |
|---|---|
| GDN / KDA / KDA dAv backward stage | [gdn-kda.md](gdn-kda.md) |
| retention / RWKV / 其他 state recurrence | 先执行本文 Step 0-6；若新增专用 reference，应放在 `references/kernels/attention/` 下并从本表路由 |
| 仅泛化 Linear Attention 设计判断 | 本文 Step 0-6 |

---

## Step 0: 自动锁定 primary reference

命中本路由时，先读取 [../open-source-linear-attention-map.md](../open-source-linear-attention-map.md) 并执行其中的 Reference Source Policy。不要要求用户在 prompt 中重复提供本机参考路径；如果用户没有显式给本地实现参考路径，必须使用远程开源仓作为 primary reference。需要源码细节时尝试 clone 到当前工作区；clone 失败不得阻塞设计，改用仓内开源规范摘要、远程搜索路径和 curated reference 继续。

primary reference 选择结果必须是以下二选一：

- `USER_LOCAL`：用户 prompt 显式给出了本地实现参考路径，按该路径对齐。
- `OPEN_SOURCE`：用户 prompt 未给本地实现参考路径，按远程开源仓对齐；clone 可用时对齐源码，clone 不可用时对齐仓内开源规范摘要。

仓内 GDN/KDA 泛化用例和既有 Catlass 经验只能作为 curated reference。它们不是用户显式本地实现参考路径，不能覆盖 `OPEN_SOURCE` 的 primary reference。用户在 prompt 中给出的 baseline / 评测 / 性能对比路径也不是 implementation reference，只能记录为 evaluation_baseline。用户 contract 优先规则集中定义在 [../open-source-linear-attention-map.md#user-contract-priority](../open-source-linear-attention-map.md#user-contract-priority)，本文件只引用该规则。

DESIGN.md 的 baseline/reference 章节必须写明 `reference_source`、已读取的参考来源、具体文件路径、commit 或当前仓库状态，以及哪些语义来自 primary reference、哪些只是 Catlass 工程经验。

---

## Step 1: 先判 full-flow 还是 stage operator

```
目标是什么？
├── full-flow fused op
│   ├── 必须冻结同语义 full-flow baseline
│   └── GDN forward 首选 baseline: flash_chunk_gated_delta_rule_fwd
└── stage operator
    ├── 必须说明上下游 stage 输入/输出契约
    └── golden 可用 stage-aware reference，但前提是上游中间量先被 full-flow 独立验证
```

**禁止**把某个局部 helper、临时 NumPy 片段或只覆盖单 stage 的脚本当作 full-flow baseline。baseline 的语义、dtype、layout、chunk 规则和状态输出必须与交付目标一致。

**禁止**把 full-flow baseline 当作实现 source-of-truth。baseline 只用于评测指标、shape、报告字段和 baseline_status；实现 pipeline 仍来自 `reference_source` 指定的 primary reference 与用户数学 contract。

---

## Step 2: 先画 dependency graph，再切 stage

设计文档必须先列出依赖图，再按依赖图切 stage。不要按公式书写顺序机械切分。

如果目标运行在 A2/A3，且存在多 stage、Cube/Vector 协作、CrossCoreFlag、L1 resident、`V=256`、GQA/GVA、partial/varlen 或 stage operator，必须进一步读取 develop 侧参考：

```text
catlass-op-develop/references/patterns/a2-a3-linear-attention-stage-design.md
```

设计文档要把其中 checklist 映射到本算子：stage/window 调度、GM workspace slot、flag 协议、L1 resident/scratch 分区、L0/UB double buffer、`V=256` split accumulation、HK/HV cache scope。

每个节点至少记录：

| 字段 | 说明 |
|------|------|
| 输入/输出 | GM tensor、workspace tensor、常驻状态 |
| 依赖类型 | matmul、elementwise、scan/cumsum、state update、transpose/layout |
| 归属核型 | AIC/Cube、AIV/Vector、混合协同 |
| 中间量生命周期 | 只在核内、跨 stage、跨 chunk、跨 head |
| 同步边界 | stage 入口 wait、stage 末尾 set、是否跨 core |

随后把节点分成以下类别：

| 类别 | 设计动作 |
|------|----------|
| L1/L0 resident candidate | 评估是否适合 Cube tile 内复用，避免重复 GM 读 |
| Catlass L0-first matmul / epilogue candidate | 尝试映射到 BlockMmad、BlockEpilogue 或自定义 Tile |
| Vector/UB stage | 用 UB 预算、double buffer 和 API 文档约束设计 |
| GM orchestration stage | 明确 workspace layout、slot、flag 和清零需求 |
| dependency-based non-L0 exemption | 说明为什么不能塞进单个 catlass epilogue |
| fallback/helper | 标记为非交付主路径或后续替换对象 |

### Step 2.1: 公共 Catlass 组件缺口的通用处理

线性 Attention 的非 GEMM 逻辑不一定能由公开 BlockEpilogue 直接表达。若遇到 gate、decay、causal mask、validRows、prefix/scan、state recurrence、双输入 finalize、layout 转换等公共组件缺口，不要把“公共 epilogue 缺失”直接判为设计阻塞；先按以下顺序处理：

1. 把 GEMM 节点映射到 Catlass `Kernel` / `BlockMmad` / scheduler。
2. 把非 GEMM 节点封装为 Catlass-style 自定义 Block/Tile，写入 DESIGN.md §1.6 契约。
3. 对跨 chunk、跨 head、跨 stage 依赖写 `dependency-based non-L0 exemption`，明确 GM workspace、flag、同步和清零。
4. 如果只能先交付正确性 MVP，PLAN.md 标记 `Catlass-stage MVP` 和性能风险，但仍需完整生成编译、精度、性能脚本和报告。

自定义 Block/Tile 的合规边界：

- 允许组件内部使用 Ascend C Vector API、固定 tile 内循环、DataCopy、mask/writeback 表达逐元素逻辑。
- 禁止组件内部手写 GEMM。
- 禁止 host 侧完成真实计算。
- 禁止 device kernel 入口为空。
- 禁止 op_kernel 顶层散写大段标量循环替代组件。

DESIGN.md §1.6 至少使用以下格式记录自定义 Block/Tile 契约：

| 字段 | 要求 |
|---|---|
| 组件名 | `Block*` 或 `Tile*` 类型名，必须和落盘头文件一致 |
| 头文件路径 | 计划落盘路径，例如 `operators/{operator_name}/op_kernel/custom/<component>.h` |
| 输入/输出 | GM/workspace/UB tensor 名称、shape、layout、生命周期 |
| dtype/cast | 输入、累加、输出 dtype，以及 cast/round mode |
| tile 尺寸 | 每次处理的行/列/chunk/head 粒度，与 TilingKey 的关系 |
| UB/L1 预算 | input/compute/output buffer 字节数，事件 ID 和 double buffer 数 |
| API 依据 | 使用的 Ascend C Vector/DataCopy API 或 catlass 槽位 header 路径 |
| 同步边界 | 入口 wait、出口 set、是否跨 pipe/core/stage |
| 合规边界 | 明确不手写 GEMM、不做 host 真实计算、不留空 device kernel |

---

## Step 3: 核查内存层级术语

设计中统一使用以下术语：

| 层级 | 用法 |
|------|------|
| GM / workspace | 跨 stage、跨核、跨 launch 中间量。A2/A3 上 Cube 与 Vector 跨域中转默认走这里。 |
| L2 | 硬件 cache，不是可手工分配的 workspace；只能描述为期望驻留/命中，不要写成显式 buffer。 |
| L1 | Cube 侧 tile 驻留和复用空间，适合复用 Q/K/V/WY 等块状输入。 |
| L0A/L0B/L0C | Cube 矩阵乘局部 tile 和累加结果。 |
| UB | Vector 计算、归约、转置、gate/beta/exp(g) 等短生命周期数据。 |

A2/A3 设计中，除非当前 catlass 版本和目标芯片文档明确证明支持，不要声称存在可用的物理 L0C -> UB 直通路径。Cube -> Vector 的跨 stage 中间量应按 GM workspace + 同步 flag 设计。

---

## Step 4: Shape 覆盖按算法维度设计，不按随手枚举

线性 Attention 的测试 shape 需要覆盖影响分支、tiling 和数值稳定性的维度组合。推荐至少 30 例，但数量不是核心，覆盖矩阵才是核心。

必覆盖类别：

| 类别 | 构造要点 |
|------|----------|
| TilingKey 分支 | 每个 dtype、chunk size、V/K 特化、调度模式至少一组 |
| `BT` / chunk 边界 | 1 个 chunk、2 个 chunk、多 chunk、尾 chunk、不整除边界 |
| `K` / `V` | 常见 128/256；`V=256` 要覆盖 K 维 split accumulation。`V=64` 为可选覆盖，仅当用户需求、TilingKey 或 primary reference 支持时必测 |
| `HK/HV/GQA` | `HK == HV`、`HV > HK`、`HV/HK` 分组共享，覆盖 KKT 按 HK 复用 |
| batch/head | 单 batch/head、多 batch/head、`B*chunk` 小于/接近/大于核数 |
| 数值边界 | zero gate、high beta、exp(g) 饱和风险、近零输出、状态初值边界 |
| 对标 shape | 与 full-flow baseline 完全同 shape、同 layout、同 dtype |

shape 的来源应写进 DESIGN/PLAN：来自用户指定实网规模、baseline 支持范围、catlass tile 整数倍、边界类别或 TilingKey 覆盖。禁止只把一串 shape 硬编码到脚本而不说明设计依据。

---

## Step 5: 精度与性能基准在设计阶段冻结

### 精度

- 标准优先使用 `ops-precision-standard`，线性 Attention 浮点输出按 mixed tolerance：`abs(actual - golden) <= atol + rtol * abs(golden)`，再看 `matched_ratio` 与 `max_abs`。
- golden 必须来自独立 baseline 或数学定义。stage operator 可用 stage-aware reference，但必须说明上游中间量如何被 full-flow 或独立 reference 验证。
- 报告中必须列出每个 case 的 shape、dtype、标准、matched_ratio、max_abs、结论。
- 报告字段按 mixed tolerance 口径固定：`case_name`、shape、dtype、输出名、atol、rtol、matched_ratio、max_abs、pass/fail。evaluation baseline 仅作为辅助对比状态记录，不改变 custom 对 independent golden 的主判定。
- 用户 contract 优先规则见 [../open-source-linear-attention-map.md#user-contract-priority](../open-source-linear-attention-map.md#user-contract-priority)。

### 性能

性能基准优先级：

| 优先级 | 基准 |
|:---:|------|
| 1 | 同语义 full-flow baseline 实测 Task Duration |
| 2 | 同 stage 官方/竞品算子实测 |
| 3 | 可运行子算子拆解相加 |
| 4 | Cube/MTE/Vector 理论上限和历史最优 |

性能报告至少包含：custom/baseline Task Duration、speedup、kernel launch count、主导流水、workspace peak、profiler 输出路径。若 full-flow baseline 多 launch，必须同时报告 launch count，避免只比较单 kernel 时长。

---

## Step 6: 输出设计章节补充

命中本路由时，DESIGN.md 除 catlass 通用选型表外，必须额外包含：

1. full-flow vs stage operator 判定和 evaluation baseline 路径
2. dependency graph 与 stage 切分表
3. GM workspace layout、slot 稳定性、flag 同步边界
4. A2/A3 stage feasibility：window 调度、L1 resident、L0/UB double buffer、`V=256` split accumulation、GQA/HK cache
5. shape 覆盖矩阵和每类 shape 的设计依据
6. mixed tolerance 精度标准与 golden 来源
7. 性能对标口径、baseline 版本/commit、同 shape 约束
8. 明确哪些 stage 是 catlass candidate，哪些是 dependency-based non-L0 exemption
9. 公共 Catlass 组件缺口的处理方案：哪些非 GEMM 节点封装为自定义 Block/Tile，哪些 stage 化，哪些暂列 MVP 风险

---

## GDN / KDA 子场景

GDN、KDA、KDA `dAv` backward stage 的接口、shape 覆盖、varlen/partial chunk、精度报告和性能复用规则见 [gdn-kda.md](gdn-kda.md)。本文只保留 Attention 大类设计入口；命中这些子场景时必须进一步读取该专项 reference。

---

## 常见陷阱

| 陷阱 | 后果 | 正确做法 |
|------|------|----------|
| 未区分 full-flow / stage | baseline 语义错，精度数据不可比 | 先冻结目标层级和同语义 baseline |
| 按公式顺序切 stage | 中间量生命周期和同步边界错 | 先 dependency graph，再 stage 切分 |
| 把 L2 写成显式 workspace | 设计不可实现 | L2 只作为 cache 命中假设，显式中转写 GM workspace |
| 只测一个代表 shape | 漏掉 BT/HV/V/K/小规模分支缺陷 | 用覆盖矩阵生成 case |
| stage golden 直接当最终标准 | 上游中间量误差被隐藏 | stage-aware reference 必须有独立上游验证依据 |
| 每次依赖用户手写本机路径 | 参考来源漂移，漏读同族实现 | 未给本地路径时自动使用 [../open-source-linear-attention-map.md](../open-source-linear-attention-map.md) 的远程 URL；clone 失败则用仓内规范摘要继续 |
| 未给本地路径却自动使用开发机本地实现 | 设计不可复现，且可能偏离开源语义 | 只有用户显式给路径才用 USER_LOCAL；否则必须是 OPEN_SOURCE |
| varlen 用有效行数替代物理 chunk 尺寸 | 可能进入不规则尾块慢路径或改变 mask 契约 | 先用 BT 物理块 + validRows 掩码建立正确/性能基线，再单独验证尾块优化 |
| 公共 epilogue 缺少 gate/mask/finalize 就停在 architect | 无法完成算子流程 | 将非 GEMM 节点封装为 Catlass-style 自定义 Block/Tile，并记录合规边界 |
| 从参考实现继承不同数学 contract | 精度报告看似通过但语义不对 | 遵守 [../open-source-linear-attention-map.md#user-contract-priority](../open-source-linear-attention-map.md#user-contract-priority)，差异写入文档和 golden |
