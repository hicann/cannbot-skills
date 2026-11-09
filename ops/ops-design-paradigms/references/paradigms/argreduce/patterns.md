# ArgReduce 范式入口（语言中立）

> 本文件是 ArgReduce 范式的**统一入口**：场景判定 + 语言路由 + 消费规则。具体知识分布在两个层次：
>
> - **通用知识层** `generic/` — 语言无关的算法、模型、方法论
> - **语言适配层** `adapters/<lang>/` — 强耦合具体开发语言的实现知识（当前提供 `ascendc`）
>
> 范式路由（`routes.yaml` → 本文件）与语言路由（本文件 → 适配层）分离：范式路由由 `op.paradigms` 决定，语言路由由开发语言决定。

---

## 场景支持

ArgReduce范式支持的场景判定：

1. **跨元素选择语义**——输出元素由输入沿选择轴（R 轴）做"选出胜者"运算得到：胜者的**位置**（必要）与**值**（可选）构成输出，选择结果沿非选择轴（A 轴）保留
2. **胜负规则可二元判定**——任意两个 (值, 位置) 二元组可判定先后（主规则比较 + 平局规则），且该判定满足结合性（分组判定不改变最终胜者，见 [generic/selection-semantics.md](generic/selection-semantics.md) §2）
3. **输出 shape 由 A 轴决定**——选择后 R 轴消失（keep_dims 场景 R 轴变 1，仍按消失处理）

典型场景：argmax / argmin / arg_first / arg_topk / find_index（含 with_value 双输出变体）、含选择的复合计算（选择前/后有 elementwise 变换）、空 tensor。谓词首达与 top-k 位置输出见 [generic/variants.md](generic/variants.md)。

**与 Reduction 范式的分界**：输出**只含值**的归约（sum/max/min/prod）走 Reduction 范式；输出**含位置**（索引）的选择归约走本范式。值+位置双输出（如 torch.max/min(x, dim) / arg_max_with_value / arg_min_with_value）属本范式。分类判定表：

| 输出形态 | 归类 | 例 |
|---|---|---|
| 只含值，无位置 | Reduction | reduce_max / reduce_min / sum(dim) |
| 只含位置 | ArgReduce | argmax / argmin / arg_first / arg_last / find_index |
| 值 + 位置双输出 | **ArgReduce**（with_value） | torch.max/min(x, dim) / arg_max_with_value / arg_min_with_value |
| 位置 + 排序序 | **ArgReduce**（可叠加 SortSelect） | arg_topk / topk(k) / argsort |

> **位置输出必含 ArgReduce**：凡输出含位置（索引）的算子，ArgReduce 必须出现在 paradigms 中；排序/宽度语义可叠加 SortSelect，但不得因选了 SortSelect 而漏掉 ArgReduce。

> **上游注册表口径**：`ops-spec-gen` 的 ArgReduce 定义现为"沿维度选择并返回位置（可同时返回值）"，with_value 双输出与 arg_first/arg_topk/find_index 等位置输出均属此类；旧口径曾把它限定为"返回索引而非值"且要求每个输出 int32/int64。若所用注册表仍是旧口径，设计节点应报告"spec 分类与范式边界冲突"并回到 spec 阶段修正；§消费规则 的"不复选"针对正确分类后的路由，不覆盖分类缺陷。

---

## 语言路由

按开发语言选择适配层。语言由调用方显式提供，或读取 `spec.yaml` 的 `op.language`；缺失或未知时先报告输入缺口，不选择语言适配层。

| 开发语言 | 加载内容 | 入口 |
|------|------|------|
| `ascendc` | 通用知识层 + AscendC 适配层 | [adapters/ascendc/index.md](adapters/ascendc/index.md) |
| `dsl`（CANNBotDSL） | 通用知识层 + DSL 适配层（真值含待验证标注，收敛规则见其 capability-mapping §3） | [adapters/dsl/index.md](adapters/dsl/index.md) |
| 其他语言 | 通用知识层；适配层待确认 | 报告当前没有对应的语言适配层，不读取其他语言的 API 或模板 |

**互斥规则**：一次任务只加载命中语言的适配层，其他语言的适配层不加载、不参考其 API 与代码模板。

**冲突裁决**：当用户显式索要**非命中语言**的适配层资料（如 DSL 任务索要 AscendC 的 API 用法/模板）时——报告互斥规则并**拒绝提供**该语言的 API 与代码模板细节；用户需要的语言中立结论（场景判定、选择语义、切分模型等 generic 层知识）照常提供，并指引用户为命中语言提供 API/设备证据以确认能力谓词。用户若坚持需要另一语言的完整实现，应将其作为**独立的 ascendc/dsl 任务**重新发起（切换语言路由），而不是在当前任务内混用两种适配层。

---

## 通用知识层索引

语言无关的算法与模型，按需展开（完整检索表见 [generic/index.md](generic/index.md)）：

| 文档 | 覆盖 |
|------|------|
| [generic/scene.md](generic/scene.md) | 场景模型、A1/R/A0 轴折叠坐标系、EMPTY/COPY/AR/ARA 四场景分类、三段式数据流骨架、能力边界 |
| [generic/selection-semantics.md](generic/selection-semantics.md) | 选择语义代数：(值, 位置) 二元组、主规则与平局规则、NaN 语义、可结合性不变量、主规则实例注册表 |
| [generic/capability-contract.md](generic/capability-contract.md) | 能力谓词定义（通用层算法引用的语言/硬件能力参数） |
| [generic/splitting.md](generic/splitting.md) | UB 三通道预算（值流+位置流+输出/暂存）、FullLoad/Split 判定、多核切分、GroupReduce 触发与 workspace 模型、跨块索引偏移 |
| [generic/sync-model.md](generic/sync-model.md) | 同步模型：RAW/WAR 依赖、核内同步点、跨核同步（仅 GroupReduce）、AscendC/DSL 原语分工 |
| [generic/variants.md](generic/variants.md) | 变体挂载：谓词首达（arg_first / find_index）、top-k、多输出 |

---

## 消费规则

1. **按需加载**：Ascend C 先读适配层 index；DSL 读取通用层与 `general-methodology-dsl.md`。仅展开当前问题需要的叶子文档。
2. **叶子文档限额**：单个子问题最多展开 5 个叶子文档，仍未闭环则拆分为更小的子问题
3. **能力真值**：Ascend C 读取 [capability-mapping.md](adapters/ascendc/capability-mapping.md)；DSL 没有专用映射时，能力谓词必须从调用方提供的 DSL API 与设备证据确认，未知项保留待验证。
4. **编码前检视参考实现**：Ascend C 可从 `adapters/ascendc/example/` 选择；DSL 从调用方提供的 CANNBotDSL 样例中选择。样例仅用于核实语言用法，不提供当前算子的算法或资源默认值。
5. **设计依据**：根据规格读取通用算法与对应语言约束，给出场景判定、选择语义结论（主规则/平局/NaN 必须逐项落到设计文档）、数据结构、资源预算、同步和验证结论；输出格式由调用方指定。Ascend C 设计检查见 [design-spec.md](adapters/ascendc/design-spec.md)，DSL 使用 `references/paradigms/general-methodology-dsl.md`。
6. **范式优先级**：本范式的专用知识优先于 agent 通用知识；同一设计点结论冲突时，以本范式指向的必读文件为准
