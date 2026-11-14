# IndexGather 范式入口（语言中立）

> 本文件是 IndexGather 范式的**统一入口**：场景判定 + 语言路由 + 消费规则。具体知识分布在以下层次：
>
> - **通用知识层** `generic/` — 语言无关的算法、模型、方法论
> - **公共知识层** `../common/` — 跨范式共享的芯片级事实
> - **语言适配层** `adapters/<lang>/` — 强耦合具体开发语言的实现知识（当前提供 `ascendc`）
>
> 范式路由（`routes.yaml` → 本文件）与语言路由（本文件 → 适配层）分离：范式路由由 `op.paradigms` 决定，语言路由由开发语言决定。

---

## 场景判定

**前置约束**：目标芯片需支持 SIMT 编程能力；不支持时本范式不适用。

满足以下条件的算子属于 IndexGather 范式（**index 张量驱动的读收集**：gather / gather_nd / index_select / take / embedding 及其变体）：

1. **index 驱动读收集语义**——输出元素由按 index 指定位置从输入读取得到（`out[i] = in[<index 变换后的地址>]`），输出连续、无写冲突
2. **解码结构可分型**——A 型单坐标整维切片 / B 型坐标元组 / C 型逐元素对齐（batch_dims 为附加结构轴不参与分型）

任意 IndexGather 算子 = 四个正交维度的组合（解码结构 × 越界语义 × 规模路径 × tiling 预处理），逐维判定：

```
给定: input/indices 语义（spec.yaml 公式与 shape 规则）

Step 1 — 解码结构分型（本质 = 索引表示）:
  ├─ A 型 单坐标整维切片: 每个索引值为单一 dim 坐标，收集整切片
  ├─ B 型 坐标元组: 每个索引为 K 维坐标元组（K = indices.shape[-1]，合法范围由 spec/需求确定）
  └─ C 型 逐元素对齐: index 与 output 同 shape，dim 单维坐标替换，支持广播
  附加结构轴: batch_dims（A/B 型可携带，前缀相等约束；C 型天然涵盖）

Step 2 — 越界处理（两步模型，优先级 ①声明→②对标→③默认 none + err）:
  ① 负索引处理: none / +dim_size / %dim_size
  ② 越界处理: err / fill（gather）/ skip（scatter）/ %dim_size / clamp / undefined
  常见预设: error / neg-wrap / mod-wrap / clamp / fill / skip / undefined

Step 3 — 规模路径: use64Bit = (inputNum > INT32_MAX) || (outSize > INT32_MAX) || (indexNum > INT32_MAX) || (taskNum > INT32_MAX)；taskNum=outSize，indexDtype 不参与

Step 4 — tiling 预处理: dim 归一化 → 广播校验 → 合轴

未落入三型的 gather 族变体（embedding / take / masked_select 等）：
  按 generic/semantics.md 变体推导映射；无法映射时登记新 type。
```

语言中立的三型语义、变体推导见 [generic/semantics.md](generic/semantics.md)；两步越界模型见 [generic/decode-bounds.md](generic/decode-bounds.md)；坐标解码模型与合轴算法见 [generic/decode-model.md](generic/decode-model.md)。

> **K 范围原则**：B 型不设范式固定上限，`K` 的合法范围由 `spec`/需求确定；范式只保留语义约束 `0 <= K <= rank(params) - batch_dims`。`K=0` 是否支持同样由需求决定：支持时走 copy 特化，不支持时由接口层拒绝。

---

## 语言路由

按开发语言选择适配层。语言由调用方显式提供，或读取 `spec.yaml` 的 `op.language`；缺失或未知时先报告输入缺口，不选择语言适配层。

| 开发语言 | 加载内容 | 入口 |
|------|------|------|
| `ascendc` | 通用知识层 + AscendC 适配层 | [adapters/ascendc/index.md](adapters/ascendc/index.md) |
| `dsl`（CANNBotDSL） | 通用知识层 + DSL 适配层 | [adapters/dsl/index.md](adapters/dsl/index.md)（含语言规则与能力谓词 [adapters/dsl/language-rules.md](adapters/dsl/language-rules.md)、能力映射 [adapters/dsl/capability-mapping.md](adapters/dsl/capability-mapping.md)、设计约束 [adapters/dsl/design-spec.md](adapters/dsl/design-spec.md)） |
| 其他语言 | 通用知识层；适配层待确认 | 报告当前没有对应的语言适配层，不读取其他语言的 API 或模板 |

**互斥规则**：一次任务只加载命中语言的适配层，其他语言的适配层不加载、不参考其 API 与代码模板。

---

## 通用/公共知识层索引

语言无关的算法与模型，按需展开（完整检索表见 [generic/index.md](generic/index.md)）：

| 文档 | 覆盖 |
|------|------|
| [generic/semantics.md](generic/semantics.md) | 三型语义（A/B/C）/ 越界两步模型概览 / 变体推导 |
| [generic/decode-bounds.md](generic/decode-bounds.md) | 越界两步模型（Step 1 负索引处理 + Step 2 越界处理）、优先级、竞品证据、防漏判原则、易混淆语义注记 |
| [generic/decode-model.md](generic/decode-model.md) | 坐标解码模型（两套步长分离）、快除中转原则、MergeAxis 合轴算法与不变量 |
| [generic/capability-contract.md](generic/capability-contract.md) | 能力谓词定义：SIMT/VF、DCache/UB、线程/寄存器、rank/index dtype、快除接口、mulhi、字节级搬移 |
| [common/simt-hardware-model.md](../common/simt-hardware-model.md) | 芯片级硬件模型：存储层级图、VF 调度执行模型、DCache 一致性与 volatile 三分法（含复合单 kernel 判定）、DCache 预留规则、SIMT 线程代价、超发核屏障、Bank 冲突 |
| [generic/thread-layout.md](generic/thread-layout.md) | SIMT 线程排布模型：一维 grid-stride 原则、核数两步法算法、PER_CORE_MIN 下限、空 tensor 早退与核数、线程数选型 |

---

## 消费规则

1. **按需加载**：Ascend C 先读 ascendc 适配层 index；DSL 先读 [adapters/dsl/index.md](adapters/dsl/index.md)（语言规则与能力谓词在 [adapters/dsl/language-rules.md](adapters/dsl/language-rules.md)，能力映射在 [adapters/dsl/capability-mapping.md](adapters/dsl/capability-mapping.md)，设计约束在 [adapters/dsl/design-spec.md](adapters/dsl/design-spec.md)）。仅展开当前问题需要的叶子文档。
2. **叶子文档限额**：单个子问题最多展开 5 个叶子文档，仍未闭环则拆分为更小的子问题
3. **能力真值**：Ascend C 读取 [adapters/ascendc/capability-mapping.md](adapters/ascendc/capability-mapping.md)；DSL 读取 [adapters/dsl/capability-mapping.md](adapters/dsl/capability-mapping.md)。未知项保留待验证，不得写成已验证事实。
4. **编码前检视参考实现**：Ascend C 可从 `adapters/ascendc/example/` 选择（读收集基例：gather/index_select）；DSL 从 `adapters/dsl/example/` 选择（读收集基例：custom_index_select）。样例仅用于核实语言用法，不提供当前算子的算法或资源默认值。
5. **范式优先级**：本范式的专用知识优先于 agent 通用知识；同一设计点结论冲突时，以本范式指向的必读文件为准
