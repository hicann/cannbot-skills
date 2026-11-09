# ArgReduce 选择语义（所有 arg 类算子的"比赛规则"）

> **层级**：通用知识层（generic/），语言中立。
>
> **一句话**：argreduce 算子本质上是在办一场比赛——每个元素带着"成绩"（value）和"号码牌"（position）参赛，按一套固定规则决出胜者，最后公布胜者的号码牌（可能连成绩一起）。
>
> 本文档定义这套"比赛规则"的通用写法。场景模型（[scene.md](scene.md)）解决"数据怎么搬进芯片"，本文档解决"怎么比"；泛化算子（arg_first / find_index / top-k）的语义推导也以本文档为唯一依据。

## 1 参赛者：每个元素 = (成绩, 号码牌)

```
参赛者 e = (value, position)
    value    : 成绩——用来比大小的那个数（原始输入值，或它的谓词/变换结果）
    position : 号码牌——这个元素沿归约轴排第几位（从 0 开始数）
```

**关键点：号码牌不是从内存读出来的**，而是"站在第几位"这个事实本身——芯片上自己生成（0,1,2,... 序列），不占内存带宽（§5）。

输出的两种形态：

| 输出 | 内容 | 对应算子族 |
|------|------|-----------|
| indices（必要） | 胜者的号码牌，转成 int32/int64 输出 | argmax / argmin |
| values（可选） | 胜者的成绩原样输出 | *_with_value 双输出 |

## 2 "两两决斗"规则与一个必须成立的性质

**两两决斗** `select(a, b)`——任取两个参赛者，按同一套规则决出胜者：

```
select(a, b):
    主规则判定 a 胜 → 返回 a
    主规则判定 b 胜 → 返回 b
    平局         → 比号码牌：号码小的赢（first-occurrence）
```

标准路径一次只选 1 个胜者（k=1）；选前 N 名的扩展见 [variants.md](variants.md) §2。

**必须成立的性质（整个范式正确性的根基）**：

> **不管怎么分组，冠军都不变。**
> 先分小组决出小组冠军、再让小组冠军决斗，与所有人直接大乱斗——结果必须相同。这个性质叫**可结合性**。

为什么它是根基：芯片上数据必须切块（缓存放不下）甚至切到多个核。切块/切核的正确性完全依赖这条性质：

1. 切块：每块先决出局部冠军，再跨块合并 ⟹ 结果正确
2. 切多核（GroupReduce）：每核决出局部冠军，再跨核合并 ⟹ 结果正确
3. **平局必须由号码牌裁决**（而不是"谁先比谁赢"）——这正是可结合性的构造方法：
   first-occurrence 等价于"按 (成绩, -号码牌) 的字典序取最大"，字典序比较对分组不敏感

推论链：

1. R 轴切块（UB 放不下）→ 块内各选局部胜者 → 跨块 `select` 合并 ⟹ 正确
2. R 轴切到多核（GroupReduce）→ 每核局部胜者 → 跨核 `select` 合并 ⟹ 正确
3. **平局规则必须由 position 裁决**（而非合并顺序裁决）——这是 select 可结合的充要构造：first-occurrence 等价于"对 (value′, -position) 做字典序取最大"，字典序极值对分组不敏感

**验证义务**：新算子的主规则必须能论证"分组不改变冠军"。如果规则依赖合并顺序（如"谁先被比较谁赢"），切分一变结果就变——这类规则不在本范式标准路径内，必须报告为语义缺口。

## 3 主规则实例注册表

| 实例名 | 主规则 | 平局规则 | 宽度 k | 算子 |
|--------|--------|---------|--------|------|
| `extremum-max-first` | value 降序（大者胜） | position 升序（first-occurrence） | 1 | argmax / argmax_with_value |
| `extremum-min-first` | value 升序（小者胜） | position 升序 | 1 | argmin / argmin_with_value |
| `extremum-max-last` | value 降序 | position 降序（last-occurrence） | 1 | 平局语义为 last 的变体 |
| `predicate-first` | 命中(true) 者胜 | position 升序 | 1 | arg_first / find_index（变体） |
| `predicate-last` | 命中者胜 | position 降序 | 1 | arg_last（变体） |
| `top-k` | value 降序 | position 升序 | N | arg_topk / topk（变体） |

**标准路径** = `extremum-*`（k=1），覆盖 argmax/argmin 家族；`predicate-*` 与 `top-k` 为变体（挂载方式见 [variants.md](variants.md)，不进场景选择键）。

**统一视角**：全部实例 = **沿 R 轴的字典序选择**——(主键, 次键) 二元字典序取极值，主键为选择键（value 或谓词），次键恒为 position（方向由平局规则决定）。新算子进来先在本表登记实例，再推导实现。

## 4 平局规则与 NaN 语义

### 4.1 平局规则（必须算子级显式声明）

| 规则 | 判定写法（max 实例，argmax 与 torch 对齐） | 语义 |
|------|------------------------------------------|------|
| **first-occurrence** | 主规则用 `>=`（`a.value >= b.value → 取 a`） | 相等时保留**前**者 |
| **last-occurrence** | 主规则用 `>`（严格大于才替换） | 相等时保留**后**者 |

- min 实例对称：first-occurrence 用 `<=`，last-occurrence 用 `<`
- **平局规则与切分顺序无关**（§2 推论 3），跨块/跨核/多副本执行结果一致
- **配套消费义务**：反向算子（如 argmax 的梯度 scatter）必须把更新写到与本约定一致的同一 position；本范式知识同时供组合范式消费（ArgMaxGrad = 本范式 × ScatterUpdate）

### 4.2 NaN 语义（必须算子级显式声明）

标准约定（与 PyTorch max/min 族一致）：**NaN 传播恒胜**——

```
is_nan(x) := (x != x)          // 值不等自身即 NaN
NaN 与任何值比较 → NaN 胜（max 与 min 实例均如此）
NaN 与 NaN 平局 → 按平局规则裁决
```

实现手法：主规则比较结果与 `is_nan` 检测结果做**掩码或**（OR）后再选择——NaN 检测在两个操作数上都做，任一为 NaN 时该操作数胜。语言落地见适配层 [../adapters/ascendc/highlevel-api.md](../adapters/ascendc/highlevel-api.md)。

### 4.3 其他特殊值

±inf、±0 等**不特判**，由主规则自然处理（与参考实现一致）；仅 NaN 因"与任何值比较均为 false"破坏主规则传递性，必须显式处理。

## 5 位置流的生成与变换

位置流在片上生成，不占用 GM 带宽；其全部变换规则：

| 阶段 | 变换 | 规则 |
|------|------|------|
| 块内生成 | 等差数列 | `pos[i] = blockOffset + i`，步长 1；由 `INDEX_ARITH_OP` 原语构造（Duplicate + 掩码加法） |
| 跨块合并 | 基址偏移 | 局部胜者 position + chunkOffset（该块在 R 轴上的起点） |
| 跨核合并 | 核基址偏移 | 局部胜者 position + coreOffset（该核分到的 R 段起点）；合并前或合并后施加均可，一次施加 |
| 复合场景 | 窗口/ROI 偏移 | 组合范式消费点：MaxPoolWithArgmax 的"窗口内位置 → 全局位置"= 本规则的实例化 |
| 输出转换 | 类型转换 | 物理存储类型（`INDEX_STORAGE_TYPE`）→ indice_dtype（int32/int64） |

**精度约束（等宽选型原理，按此推导而非死记结论）**：

1. **位置通道必须能与值流共享比较/选择掩码**——在逐 lane 掩码的向量架构中，这要求**位置元素与值元素等宽**（同 lane 数、同掩码粒度）：值 2 字节（fp16/bf16）→ 位置 2 字节整数；值 4 字节（fp32/int32）→ 位置 4 字节整数
2. **块内偏移上界** = 位置元素宽度的整数容量（2 字节 → 65536；4 字节 → 2^32）；切块大小通常受片上预算约束远小于此上界，**必须显式核对**（块内偏移 ≤ 切块上界）
3. **全局位置** = 块基址（宽整数，`INDEX_ARITH_OP` 拼接指令一次施加）+ 块内偏移 → 全域上限为输出整数类型容量（int32 输出 → 2^31），**不受位置元素宽度限制**
4. **历史口径标注**：位置用浮点类型存储（float32 + 2^24 上界）是旧代平台指令能力限制下的权宜口径（彼时选择指令不支持整数目的），**仅在适配层明确声明该代平台能力时适用**，禁止跨代际沿用

**验证义务**：设计必须声明位置元素选型（含等宽依据）、块内偏移上界核查、块基址拼接方案；依赖浮点位置通道时必须同时声明平台依据。

## 6 输出形态与 shape

| 属性 | 规则 |
|------|------|
| indices shape | A1 × A0（keep_dims 时 A1 × 1 × A0） |
| values shape | 同 indices；value dtype = 输入 dtype（或算子声明的输出 dtype） |
| indice_dtype | int32 / int64，算子规格声明；影响输出 buffer 规划与 R 上界（int32 要求 position < 2^31） |
| A1 或 A0 为空（EMPTY） | 输出元素数为 0；EMPTY_R（R==0 且 A 非空）按算子级声明的空约定输出（scene §6.4） |

## 7 与 Reduction 范式的语义关系

- **Reduction**：值合并算子 `merge(v, v′) → v″`（max/sum/…），输出纯值
- **ArgReduce**：二元组合并算子 `select((v,p), (v′,p′)) → (v″,p″)` = 值合并 ⊕ 位置平局裁决
- arg 版 = 值版 + 位置跟踪：`argmax(x) = argmax(max-semantics, first-tie)`；因此 ArgReduce 的值通道约束（精度、比较）与 Reduction 的 max/min 语义共享，位置通道是本范式的增量
- **组合消费边界**：Scatter / Pooling / ROI 等上下文由对应范式提供；本范式对外输出三个可复用组件——平局约定（§4.1）、select 代数（§2）、位置算术（§5）
