# ArgReduce 范式场景模型

> **层级**：通用知识层（generic/），语言中立。
>
> 来源：ops-math argmax 家族工业实现（arg_max_with_value / arg_max_v2 / arg_min 系，arch35）与实验版实现（A2/A3）的设计蒸馏：两套实现的第一级判定逻辑同源于 (A1, R, A0) 三段轴折叠。TilingKey/TilingData 的具体实现在适配层 [../adapters/ascendc/overview.md](../adapters/ascendc/overview.md)。

## 1 准入条件

本范式覆盖的算子需满足：

1. **跨元素选择语义**——输出元素由输入沿选择轴（R 轴）做"选出胜者"运算得到，胜者的**位置**构成必要输出、**值**构成可选输出
2. **胜负规则可二元判定**——任意两个 (值, 位置) 二元组的先后可判定，且满足结合性（见 [selection-semantics.md](selection-semantics.md) §2）
3. **输出 shape 由 A 轴决定**——选择后 R 轴消失（keep_dims 场景 R 轴变 1，仍按消失处理）

典型场景：argmax / argmin / arg_first / arg_last / find_index（单输出 indices）、argmax_with_value / argmin_with_value（双输出）、arg_topk（位置+排序序）、含选择的复合计算（选择前/后有 elementwise 变换）。仅输出值、不输出位置的 max/min 归约走 Reduction 范式。

## 2 坐标系与术语（语言中立）

| 术语 | 白话解释 | 举例 |
|------|---------|------|
| **R 轴** | 选择轴（dimension）——胜者从这一串元素里产生 | 对 [2,100,4000] 取 dim=1，R 就是中间那 100 个元素 |
| **A1 轴段** | 选择轴**前面**所有维度合起来的大小（外层，等于要做多少次选择） | 上面的例子里 A1=2（2 行，每行选一次） |
| **A0 轴段** | 选择轴**后面**所有维度合起来的大小（内层，决定相邻两场"比赛"隔多远） | 上面的例子里 A0=4000（每行 4000 列，列与列之间隔 4000 个元素） |
| **aTotal** | A1 × A0，输出总元素数（一共要比多少场） | 2 × 4000 = 8000 个索引 |
| **(值, 位置) 二元组** | 参与比较的最小单元：成绩单（值）+ 号码牌（位置），两者必须一起比较、一起搬运 | 见 [selection-semantics.md](selection-semantics.md) §1 |
| **AR / ARA / COPY / EMPTY** | 四种数据形态（§3）——**决定代码怎么写的第一件事**：选手是排成一排（连续）还是隔着站（跨步） | AR: [4,60000] dim=1；ARA: [2,100,4000] dim=1 |
| **RA 特化** | ARA 的一种向量化写法（当 A1==1 时，即只有一场比赛且选手跨步排列） | [50,100000] dim=0 |
| **keep_dims** | 输出 shape 里把 R 轴保留成长度 1。只影响输出形状，不影响计算 | keep_dims=true 时 [2,4000] → [2,1,4000] |
| **indice_dtype** | 位置的输出类型（int32 或 int64），不影响计算，只影响输出格式 | int32 |

**轴折叠（一步完成）**：任意 N 维输入 × dimension → 三段：

```
A1 = ∏(dims[0 .. dim-1])    R = dims[dim]    A0 = ∏(dims[dim+1 .. N-1])
内存排布恒为 A1 × R × A0（行主序）
```

负 dimension 先归一化（dim += rank）。对比 Reduction 范式的 A R A R 交替 pattern 坐标系：argreduce 只沿单一选择轴选择，折叠到三段即完备，不需要多段交替坐标系。

## 3 场景分类

按"R 轴候选元素在内存中的排布方式"分四种主场景（判定顺序固定，短路优先）：

| 场景 | 判定条件（按序） | 数据排布 | 典型例 |
|------|----------------|---------|--------|
| **EMPTY** | shape 任一维含 0（折叠**前**短路） | — | [0,5]、[4,0] |
| **COPY** | R == 1 | 每场"比赛"只有 1 个参赛者：位置恒 0，值透传 | [4,1] dim=1 |
| **AR** | A0 == 1（含 dim 为末维） | 每个输出的 R 个候选在内存中**连续**（stride=1） | [4,60000] dim=1 |
| **ARA** | A0 > 1 | 每个输出的 R 个候选**跨步**（stride=A0） | [2,100,4000] dim=1 |

**判定伪码**：

```
if 任一 dim size == 0:        → EMPTY（短路，跳过折叠）
A1, R, A0 = 折叠(shape, dim)
if R == 1:                    → COPY（values 透传，indices 全 0）
elif A0 == 1:                 → AR（连续，整块处理）
else:                         → ARA（跨步，重排或逐元素处理）
```

### 3.1 为什么轴位置排布是第一分类标准

这是 ArgReduce 与 Reduction 范式结构差异的根源。**位置跟踪放大了排布差异的实现代价**：

| 场景 | Reduction（纯值归约）的代价 | ArgReduce（值+位置选择）的代价 |
|------|---------------------------|------------------------------|
| AR | 换一个 API 参数 | **小增量**：值+位置 API（如 `calIndex` 类参数）或位置流随行同步生成 |
| ARA | 换一个 Pattern 方向的 API | **结构性差异**：多数语言的 Pattern 归约指令**不返回位置**，须改用向量比较+掩码选择逐元素手法，并引入新的对齐/精度约束 |

因此在 ArgReduce 中，AR/ARA 从 Reduction 的"模板内运行时分支"升级为**主场景分类**（选择键的一部分）；而多核切分结构（是否跨核借 R）降为正交维度（见 [splitting.md](splitting.md) §3）。

### 3.2 RA 特化（A1 == 1 的 ARA）

ARA 且 A1==1（选择轴是第 0 维）时，数据呈 `[R, A0]` 布局：A0 天然连续、可铺到向量通道（lane），R 逐行比较。这是 ARA 的向量化实现特化——**同一主场景的另一条实现路径**，是否提供由语言适配层决定（AscendC 适配层提供，见 [../adapters/ascendc/mode-mapping.md](../adapters/ascendc/mode-mapping.md)）。

## 4 数据流骨架

计算图按三段式划分：

```
┌─ 选择前段（preSelectPhase，可选）────────────────────────┐
│  输入 → 搬入 → PreElewise（如谓词计算/类型转换）          │
└──────────────────────────┬───────────────────────────────┘
                           ↓
┌─ 选择段（selectPhase）───────────────────────────────────┐
│  值流 + 位置流 → 二元组选择（块内逐段/跨块合并）          │
│  → (winValue, winPosition)                               │
└──────────────────────────┬───────────────────────────────┘
                           ↓
┌─ 选择后段（postSelectPhase，可选）───────────────────────┐
│  位置类型转换（存储类型 → indice_dtype）→ 搬出            │
│  withValue 算子：值同步搬出                               │
└──────────────────────────────────────────────────────────┘
```

- **位置流**：不是从内存载入，而是**在片上生成**——等差数列（起始 = 块偏移，步长 1），随值流同步参与比较。生成与变换规则见 [selection-semantics.md](selection-semantics.md) §5
- **winValue / winPosition 是桥接节点**；选择前段与选择后段独立做存活节点分析
- **跨块/跨核合并**：每级合并同时合并值与位置，且必须施加平局规则与索引基址偏移（局部位置 + 块/核偏移 = 全局位置），见 [splitting.md](splitting.md) §4

## 5 编译期特化

- **场景与性能特化路径进选择键**（由 tiling 现算，落地为 TilingKey）；ARA 内部还可细分载入子模式
- **算子族参数不进选择键**：dtype 由框架按输入 dtype 注入编译实例；isMin（max/min 方向）、withValue（是否输出值）、indice_dtype（位置输出类型）由模板参数或编译期常量承载——一套实现支撑 argmax/argmin/双输出四算子
- 选择宽 k（变体，见 [variants.md](variants.md)）为算子级常量，不进选择键

## 6 能力边界

1. **位置精度（等宽选型）**：位置通道元素与值通道元素**等宽**（共享选择掩码的物理前提，见 [selection-semantics.md](selection-semantics.md) §5），块内偏移上界 = 位置元素整数容量，全局位置 = 块基址 + 块内偏移拼接（int32 输出上限 2^31）。设计必须显式核对三项（选型依据 / 块内偏移 / 拼接方案），**未声明视为设计缺陷**
2. **平局语义必须算子级单一结论**：first-occurrence 或 last-occurrence，不得缺省。前后向配套算子（如 argmax 的梯度）必须消费同一约定，否则产生静默错误
3. **NaN 语义必须算子级单一结论**：NaN 传播（恒胜）或按值处理，与参考实现对齐（PyTorch max/min 族：NaN 传播恒胜）
4. **空 R 语义必须算子级单一结论**：R==0 且 A 非空时输出约定（常见：indices 全 0 / 报错），范式不预设
5. **确定性**：固定切分 + 平局规则由 position 决定（与合并顺序无关）→ 同设备同输入多次执行 bitwise 可复现
6. 本范式覆盖单轴选择；多轴依次选择按"逐轴独立选择"组合处理（每次沿一根 R 轴）

## 7 依赖谓词

| 谓词 | 本文档哪里用到它 |
|------|------|
| `INDEX_TRACKING_REDUCE` | AR 场景手法 A（一条指令同时给值和位置） |
| `VECTOR_COMPARE_SELECT` | 主规则落地（批量比较 + 掩码选择；AR 手法 B / ARA 主手法） |
| `INDEX_ARITH_OP` | 位置流生成与偏移（造号码牌） |
| `INDEX_STORAGE_TYPE` | 位置流存储类型与整数上界（等宽原则） |
| `VECTOR_LANE_WIDTH` | 一条向量能装多少元素（决定 R 很小时能否一次比完） |
| `ONCHIP_CAPACITY` / `DATA_ALIGNMENT` | 片上容量与对齐（UB 预算） |
| `MULTICORE_BALANCE` | 多核工作量分配 |
| `CROSS_CORE_SYNC` / `WORKSPACE_MODEL` | GroupReduce 两阶段（栅栏 + 公共内存） |

> 谓词的白话解释与"适配层要回答什么"见 [capability-contract.md](capability-contract.md) §2。

谓词定义见 [capability-contract.md](capability-contract.md)；各语言真值见对应适配层 capability-mapping。
