# Broadcast 范式入口（语言中立）

> 本文件是 Broadcast 范式的**统一入口**：场景判定 + 语言路由 + 消费规则。具体知识分布在两个层次：
>
> - **通用知识层** `generic/` — 语言无关的算法、模型、方法论
> - **语言适配层** `adapters/<lang>/` — 强耦合具体开发语言的实现知识（当前提供 `ascendc`、`dsl`）
>
> 范式路由（`routes.yaml` → 本文件）与语言路由（本文件 → 适配层）分离：范式路由由 `op.paradigms` 决定，语言路由由开发语言决定。

---

## 场景判定

满足以下条件的算子属于 Broadcast 范式：

1. **element-wise 计算语义**——输出元素由输入元素的逐点函数计算得到
2. **多输入间存在 broadcast**——输入 shape 不一致，需广播对齐后计算
3. **计算公式不含 Reduce**——无跨元素归约（归约类走 Reduction 范式）

典型场景：输入输出 shape 不一致需广播对齐的 element-wise 算子。语言中立的场景模型、坐标系理论与数据流骨架见 [generic/scene.md](generic/scene.md)。

---

## 语言路由

按开发语言选择适配层。语言由调用方显式提供，或读取 `spec.yaml` 的 `op.language`；缺失或未知时先报告输入缺口，不选择语言适配层。

| 开发语言 | 加载内容 | 入口 |
|------|------|------|
| `ascendc` | 通用知识层 + AscendC 适配层 | [adapters/ascendc/index.md](adapters/ascendc/index.md) |
| `dsl`（CANNBotDSL） | 通用知识层 + CANNBotDSL 适配层 | [adapters/dsl/index.md](adapters/dsl/index.md) |
| 其他语言 | 通用知识层；适配层待确认 | 报告当前没有对应的语言适配层，不读取其他语言的 API 或模板 |

**互斥规则**：一次任务只加载命中语言的适配层，其他语言的适配层不加载、不参考其 API 与代码模板。

---

## 通用知识层索引

语言无关的算法与模型，按需展开（完整检索表见 [generic/index.md](generic/index.md)）：

| 文档 | 覆盖 |
|------|------|
| [generic/scene.md](generic/scene.md) | 场景模型、坐标系理论、数据流骨架、能力边界 |
| [generic/capability-contract.md](generic/capability-contract.md) | 能力谓词定义（通用层算法引用的语言/硬件能力参数） |
| [generic/shape-preprocessing.md](generic/shape-preprocessing.md) | 补 1 / 去 1 / 归一 / 合轴四步算法与兼容校验 |
| [generic/splitting.md](generic/splitting.md) | 片上内存单切分与多核均衡切分算法 |
| [generic/liveness-fusion.md](generic/liveness-fusion.md) | L/P 存活节点模型、持有法则、融合决策方法论 |
| [generic/sync-model.md](generic/sync-model.md) | RAW/WAR 依赖识别与同步点推导 |
| [generic/compute-flow.md](generic/compute-flow.md) | 计算流伪码规范（P 分析到编码的桥梁） |
| [generic/verification.md](generic/verification.md) | golden/opt 交叉验证方法论与 harness |

---

## 消费规则

1. **按需加载**：先读对应语言适配层的 index，再展开当前问题需要的叶子文档。
2. **叶子文档限额**：单个子问题最多展开 5 个叶子文档，仍未闭环则拆分为更小的子问题
3. **能力真值**：通用层算法中的能力谓词，以命中语言适配层的 capability-mapping 填值为准（ascendc: [adapters/ascendc/capability-mapping.md](adapters/ascendc/capability-mapping.md)；dsl: [adapters/dsl/capability-mapping.md](adapters/dsl/capability-mapping.md)，其中标注"待验证"的谓词须在产出中声明假设并给出验证义务）
4. **实现依据**：Ascend C 编码前检视 `adapters/ascendc/example/` 中的参考实现，选型参考 `references/reference-ops/open_source_operator_table.md`。DSL 实现依据对应适配层的能力映射与实现约束。参考实现中的具体算法与参数须按当前算子规格重新核定。
5. **设计依据**：根据规格读取通用算法与对应语言约束，给出分支、数据结构、资源预算、同步和验证结论；输出格式由调用方指定。Ascend C 设计检查见 [design-spec.md](adapters/ascendc/design-spec.md)，DSL 使用 `references/paradigms/general-methodology-dsl.md`。
6. **范式优先级**：本范式的专用知识优先于 agent 通用知识；同一设计点结论冲突时，以本范式指向的必读文件为准
