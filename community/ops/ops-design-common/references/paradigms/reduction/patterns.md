# Reduction 范式入口（语言中立）

> 本文件是 Reduction 范式的**统一入口**：场景判定 + 语言路由 + 消费规则。具体知识分布在两个层次：
>
> - **通用知识层** `generic/` — 语言无关的算法、模型、方法论
> - **语言适配层** `adapters/<lang>/` — 强耦合具体开发语言的实现知识（当前提供 `ascendc`）
>
> 范式路由（`routes.yaml` → 本文件）与语言路由（本文件 → 适配层）分离：范式路由由 `op.paradigms` 决定，语言路由由开发语言决定。

---

## 场景判定

满足以下条件的算子属于 Reduction 范式：

1. **跨元素归约语义**——输出元素由输入沿归约轴（R 轴）折叠计算得到（sum/mean/max/min/prod/any/all 及其复合），归约结果沿非归约轴（A 轴）保留
2. **归约轴可标记**——输入张量的每一根轴可按算子语义标记为 A 轴（preserve）或 R 轴（reduce）
3. **输出 shape 由 A 轴决定**——归约后 R 轴消失（keep_dims 场景 R 轴变 1，仍按消失处理）

典型场景：单/多轴 reduce、全轴归约（All Reduce）、含归约的复合计算（归约前/后有 elementwise 变换）、多输出多次独立归约（Tuple Reduce）、空 tensor（shape 含 0）。语言中立的场景模型、模板分类与数据流骨架见 [generic/scene.md](generic/scene.md)。

---

## 语言路由

按开发语言选择适配层。语言由调用方显式提供，或读取 `spec.yaml` 的 `op.language`；缺失或未知时先报告输入缺口，不选择语言适配层。

| 开发语言 | 加载内容 | 入口 |
|------|------|------|
| `ascendc` | 通用知识层 + AscendC 适配层 | [adapters/ascendc/index.md](adapters/ascendc/index.md) |
| `dsl`（CANNBotDSL） | 通用知识层 + DSL 通用方法 | 当前无 Reduction DSL 专用适配层；读取 `references/paradigms/general-methodology-dsl.md`，DSL API 和设备能力由调用方提供 |
| 其他语言 | 通用知识层；适配层待确认 | 报告当前没有对应的语言适配层，不读取其他语言的 API 或模板 |

**互斥规则**：一次任务只加载命中语言的适配层，其他语言的适配层不加载、不参考其 API 与代码模板。

---

## 通用知识层索引

语言无关的算法与模型，按需展开（完整检索表见 [generic/index.md](generic/index.md)）：

| 文档 | 覆盖 |
|------|------|
| [generic/scene.md](generic/scene.md) | 场景模型、A/R 轴与 pattern 坐标系、Base/Group/Empty 三模板分类、三段式数据流骨架、能力边界 |
| [generic/capability-contract.md](generic/capability-contract.md) | 能力谓词定义（通用层算法引用的语言/硬件能力参数） |
| [generic/axis-preprocessing.md](generic/axis-preprocessing.md) | axes 归一化 6 步、空 tensor 短路判定、四步合轴算法、pattern 形态与 MAX_PATTERN_RANK 推导 |
| [generic/tree-reduction.md](generic/tree-reduction.md) | 二分缓存树：精度动机、非 2 幂主尾配对、多级缓存写入层级与吸收规则 |
| [generic/liveness-fusion.md](generic/liveness-fusion.md) | L/P 存活节点模型、公平基线、互斥分支流规则、融合决策方法论 |
| [generic/splitting.md](generic/splitting.md) | UB 双切分三步算法、多核均衡切分、Group 2D 分核、Empty 切分约束、UB 预算不等式 |
| [generic/sync-model.md](generic/sync-model.md) | RAW/WAR 依赖识别、三层 WAR、循环间反向同步、跨核栅栏模型 |
| [generic/tuple-reduce.md](generic/tuple-reduce.md) | Tuple Reduce 场景模型：N 次独立归约的循环复用、存活节点一条流规则 |

---

## 消费规则

1. **按需加载**：Ascend C 先读适配层 index；DSL 读取通用层与 `general-methodology-dsl.md`。仅展开当前问题需要的叶子文档。
2. **叶子文档限额**：单个子问题最多展开 5 个叶子文档，仍未闭环则拆分为更小的子问题
3. **能力真值**：Ascend C 读取 [capability-mapping.md](adapters/ascendc/capability-mapping.md)；DSL 没有专用映射时，能力谓词必须从调用方提供的 DSL API 与设备证据确认，未知项保留待验证。
4. **编码前检视参考实现**：Ascend C 可从 `adapters/ascendc/example/` 选择；DSL 从调用方提供的 CANNBotDSL 样例中选择。样例仅用于核实语言用法，不提供当前算子的算法或资源默认值。
5. **设计依据**：根据规格读取通用算法与对应语言约束，给出分支、数据结构、资源预算、同步和验证结论；输出格式由调用方指定。Ascend C 设计检查见 [design-spec.md](adapters/ascendc/design-spec.md)，DSL 使用 `references/paradigms/general-methodology-dsl.md`。
6. **范式优先级**：本范式的专用知识优先于 agent 通用知识；同一设计点结论冲突时，以本范式指向的必读文件为准
