# ArgReduce 变体挂载

> **层级**：通用知识层（generic/），语言中立。
>
> 变体 = 主规则实例注册表（[selection-semantics.md](selection-semantics.md) §3）中非 `extremum` 标准路径的语义。**挂载原则**（对齐 Reduction 范式 Tuple Reduce 先例）：变体不进场景选择键、不改切分/多核/精度模型，只替换或扩展**选择段内部**的比较与输出逻辑。

## 1 变体清单与共性

| 变体 | 替换点 | 复用部分（100% 不变） |
|------|--------|---------------------|
| 谓词首达（predicate-first / predicate-last） | 主规则：值比较 → 命中判定 | 场景判定、切分、GroupReduce、位置流、精度约束 |
| top-k（宽度 k = N，含 arg_topk） | select：二选一 → 有序列表归并；输出 ×k | 场景判定、切分结构、位置流、平局规则 |
| 多输出独立选择（Tuple Arg） | 外层 for 循环 N 次独立选择 | 全部（含 select 本身） |

**with_value 双输出**（值 + 位置）不是独立变体：选择段与位置流完全不变，只是把胜者的值随位置一并写出。

- values 输出 = 胜者元素原值；NaN 恒胜时 values 输出 NaN，indices 指向首个 NaN——两路同源同一次选择
- values dtype 跟随输入 x，indices 为 int32/int64；单遍选择产出两路，buffer/workspace 预算不乘 2
- 分类不变：输出含位置即属 ArgReduce，即使首输出是 values（见 [../patterns.md](../patterns.md) 的分类判定表）

## 2 谓词首达（arg_first / arg_last / find_index）

**语义**：沿 R 轴返回第一个/最后一个满足谓词 P 的元素位置；无命中时的输出约定必须算子级声明（常见：返回哨兵值如 -1 / R）。

```
predicate-first ≡ extremum-max-first 的主键替换：
  value′ = P(value) ? 1 : 0      // 谓词命中映射为可比较键
  select((value′, pos))，value′ 大者胜、平局 position 小者胜
find_index(target) = predicate-first 的实例化：P(v) := (v == target)
arg_last = predicate-last：平局规则换为 position 降序
```

**实现增量**（相对标准路径）：仅 preSelectPhase 增加谓词计算（或比较掩码直接作为主键），select 结构不变。谓词命中掩码与 `VECTOR_COMPARE_SELECT` 同源。

**可结合性**：命中键为布尔量、平局由 position 裁决 → select 可结合成立，跨块/跨核结构直接复用。

## 3 top-k（宽度 k = N，含 arg_topk）

**语义**：沿 R 轴选出前 k 个 (值, 位置)，按 (value 降序, position 升序) 排列输出。

**arg_topk 与 topk**：位置优先的 arg_topk 属 ArgReduce（top-k 变体）；带值+位置排序的 topk 可叠加 SortSelect 负责排序/宽度语义，但 ArgReduce 必含。

**select 推广**：`select_k` 从"两个二元组取一"推广为"两个长度 ≤ k 的有序列表归并取前 k"：

```
merge_k(A, B)：双指针归并，(value, -position) 字典序小者先出，截断到 k
```

- **可结合性保持**：归并满足结合律 → 切块/切核结构与 k=1 完全一致（每核产出局部 top-k，聚合时 merge_k）
- **资源增量**：值流/位置流 buffer × k；输出 indices shape 追加 k 维
- **精度注意**：k 通道的 position 并行存储，`INDEX_STORAGE_TYPE` 约束逐通道不变
- k=1 退化为标准路径（top-1 = argmax）

## 4 多输出独立选择（Tuple Arg）

N 次相互独立的选择（如同时输出 argmax 与 argmin 位置），对齐 Reduction Tuple Reduce 模型：

- `N_SELECTS` 为算子级常量，**不进选择键、不进 TilingData**
- 入口分发结构不变，每路内包 `for (processIdx < N_SELECTS)` 循环调用 Process
- 存活节点按"一条流"规则取互斥分支峰值 max；**禁止在 buffer 预算或 workspace 公式中乘以 N**
- GroupReduce 变体：循环末尾追加栅栏，兜住跨过程 WAR

## 5 新变体登记规则

新语义进本范式时按序执行：

1. 在 [selection-semantics.md](selection-semantics.md) §3 实例注册表登记（实例名 / 主规则 / 平局 / 宽度）
2. 论证 select 可结合性（不满足则报告语义缺口，不进本范式）
3. 在本文档登记替换点与复用部分；仅当替换点超出"选择段内部"（如改变数据排布或切分结构）时，才允许动 scene/splitting 层并同步修订
4. 场景判定、切分、精度知识**默认全部复用**，不复制粘贴
