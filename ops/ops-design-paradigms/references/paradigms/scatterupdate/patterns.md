# ScatterUpdate 范式入口（语言中立）

> 本文件是 ScatterUpdate 范式的**统一入口**：场景判定 + 语言路由 + 消费规则。具体知识分布在以下层次：
>
> - **通用知识层** `generic/` — 语言无关的算法、模型、方法论
> - **公共知识层** `../common/` — 跨范式共享的芯片级事实
> - **语言适配层** `adapters/<lang>/` — 强耦合具体开发语言的实现知识（当前提供 `ascendc`）
>
> 范式路由（`routes.yaml` → 本文件）与语言路由（本文件 → 适配层）分离：范式路由由 `op.paradigms` 决定，语言路由由开发语言决定。本范式承载写散射的 **assign 覆盖写分型**（`out[idx] = src`，纯搬运 bit-exact，无算术无归约）；reduce 归约写分型（`out[idx] op= src`，读-改-写）不在本知识库范围。

---

## 场景判定

**前置约束**：目标芯片需支持 SIMT 编程能力；不支持时本范式不适用。

满足以下条件的算子属于 ScatterUpdate 范式（**index 张量驱动的写散射·覆盖写**：scatter(reduce=none) / ScatterElements(reduce=none) / ScatterNdUpdate / tensor_scatter_nd_update / scatter_nd_update / put_(accumulate=False) 及其变体）：

1. **index 驱动写散射语义**——输出由输入副本 + 按 index 指定位置的**覆盖写**得到
2. **写语义为 assign 覆盖**——`out[idx] = src`，纯赋值无算术（NaN/Inf/±0 bit-exact 原样透传，无类型提升无舍入）；reduce 归约写（`out[idx] op= src`）不属本范式
3. **index 形态可归类**——dim 形态（index 覆盖全部维 / index 覆盖单维两个并列子场景）或 nd 坐标元组（覆盖前 K 维）；展平 1D 为覆盖全部维的 rank=1 极限退化

覆盖写算子 = 四个正交维度的组合（index 形态 × 越界 × 确定性 × src 表示），逐维判定：

```
给定: var/index(或 indices)/src(或 updates) 语义（spec.yaml 公式与 shape 规则）

Step 1 — 写语义划界: out[idx] = src（纯赋值覆盖）→ 本范式；
  归约写 out[idx] op= src（op ∈ {add, mul, max, min, div, mean}）→ 不属于本范式

Step 2 — index 形态（决定 Phase1 冲突 key 合成；Phase1×Phase3 正交）:
  ├─ dim 形态（单维 + dim），按 index 覆盖维分两个并列子场景:
  │    ├─ index 覆盖全部维（逐元素；示例 torch scatter(reduce=none) / ScatterElements）
  │    └─ index 覆盖单维（沿 dim 轴共享）
  │    展平 1D 为「覆盖全部维」的极限退化
  └─ nd 坐标元组（覆盖前 K 维；示例 scatter_nd_update / tensor_scatter_nd_update）
  边界（划界，不属本范式整数坐标轴）: 布尔掩码索引（→ mask_predicate）/
  高级索引组合（→ 动态高级索引复合）/ 分段组号索引（→ DynamicShape segment）

Step 3 — 越界形态: error / skip（哨兵，assign 常见对标形态）/ neg-norm（负索引归一化）
  按优先级 ①声明→②对标→③默认 error

Step 4 — 确定性: 默认 nondet 直写（多写者任意赢 = 覆盖语义本身正确，
  对齐单条 store 原子防撕裂、全 dtype 无白名单）；
  高重复（性能）或（张量源）确定性请求 → det 排序段尾覆盖（last-write-wins）；
  标量源（值同）仅高重复走 det 排序（dedup，纯性能——值同天然 bit 确定）
```

语言中立的语义模型、变体推导见 [generic/semantics.md](generic/semantics.md)；冲突域统一视角（key/b/s）与排序量原则见 [generic/conflict-domain.md](generic/conflict-domain.md)。

---

## 语言路由

按开发语言选择适配层。语言由调用方显式提供，或读取 `spec.yaml` 的 `op.language`；缺失或未知时先报告输入缺口，不选择语言适配层。

| 开发语言 | 加载内容 | 入口 |
|------|------|------|
| `ascendc` | 通用知识层 + AscendC 适配层 | [adapters/ascendc/index.md](adapters/ascendc/index.md) |
| `dsl`（CANNBotDSL） | 通用知识层 + DSL 通用方法 | 当前无 ScatterUpdate DSL 专用适配层；读取 `references/paradigms/general-methodology-dsl.md`，DSL API 和设备能力由调用方提供 |
| 其他语言 | 通用知识层；适配层待确认 | 报告当前没有对应的语言适配层，不读取其他语言的 API 或代码模板 |

**互斥规则**：一次任务只加载命中语言的适配层，其他语言的适配层不加载、不参考其 API 与代码模板。

---

## 通用/公共知识层索引

语言无关的算法与模型，按需展开（完整检索表见 [generic/index.md](generic/index.md)）：

| 文档 | 覆盖 |
|------|------|
| [generic/semantics.md](generic/semantics.md) | 覆盖写语义、src 表示（张量源/标量源）、index 形态、覆盖写变体推导 |
| [generic/bounds.md](generic/bounds.md) | 越界三形态决策方法论（优先级/二维独立/竞品证据）、哨兵 key 机制契约、双模板一致性约束 |
| [generic/conflict-domain.md](generic/conflict-domain.md) | 冲突域统一视角（key/b/s 三轴）、排序量原则、key 位宽自适应、空 tensor 早退原则 |
| [generic/capability-contract.md](generic/capability-contract.md) | 能力谓词定义：SIMT/VF、DCache/UB、线程/寄存器、rank/index dtype、快除、mulhi、字节搬移，以及 L2 volatile、跨核同步、对齐单 store 原子、稳定排序、UB 对齐、workspace 模型 |
| [common/simt-hardware-model.md](../common/simt-hardware-model.md) | 芯片级硬件模型：存储层级图、VF 调度执行模型、DCache 一致性与 volatile 三分法（含复合单 kernel 判定）、DCache 预留规则、SIMT 线程代价、超发核屏障参与原理、Bank 冲突 |
| [generic/thread-layout.md](generic/thread-layout.md) | SIMT 线程排布模型：det 三阶段共用核排布、nondet 单阶段两步法、(b,k,s) 任务一维化、超发核 guard、空 tensor 早退与核数、线程数选型 |

---

## 消费规则

1. **按需加载**：Ascend C 先读适配层 index；DSL 读取通用层与 `general-methodology-dsl.md`。仅展开当前问题需要的叶子文档。
2. **叶子文档限额**：单个子问题最多展开 5 个叶子文档，仍未闭环则拆分为更小的子问题
3. **能力真值**：Ascend C 读取 [adapters/ascendc/capability-mapping.md](adapters/ascendc/capability-mapping.md)；DSL 无专用映射时，能力谓词须从调用方提供的 DSL API 与设备证据确认，未知项保留待验证。
4. **读收集方法论**：快除与合轴方法论与读收集范式共享（[common/uintdiv-rules.md](../common/uintdiv-rules.md)），越界决策方法论同源但 scatter 写侧自建承载（[generic/bounds.md](generic/bounds.md)）
5. **写侧划界**：reduce 归约写不在本知识库范围；本范式仅承载 assign 覆盖写。
6. **编码前检视参考实现**：Ascend C 可从 `adapters/ascendc/example/` 选择（assign × nd 双模板基例）；DSL 从调用方提供的样例中选择
7. **范式优先级**：本范式的专用知识优先于 agent 通用知识；同一设计点结论冲突时，以本范式指向的必读文件为准
8. **确定性分治**（本范式核心规则）：默认 nondet 直写——多写者任意赢即覆盖语义正确（"丢失"其他写者正是覆盖语义），仅需对齐单条 store 原子防撕裂（全 dtype 无白名单）；高重复（同地址并发写慢一个数量级，排序段尾单写者把 M 次冲突写降为 unique(key) 次，纯性能）或（张量源）确定性请求（last-write-wins 固定顺序）才走 det 排序；标量源（值同）无确定性问题，仅高重复走排序（dedup）
