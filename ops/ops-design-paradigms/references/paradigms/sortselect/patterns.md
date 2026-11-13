# SortSelect 范式入口（语言中立）

本文件承载场景判定、语言路由和消费规则。通用知识在 `generic/`，具体语言实现知识与 Sort 参考实现在 `adapters/<lang>/`。

## 场景判定

沿指定轴重排元素或选择顺序统计量，包括 Sort、Argsort、TopK、KthValue。

本范式使用一套计算流程：**布局规范化 → 完整排序 → 按名次区间写回 → 布局还原**。Sort、Argsort、TopK、KthValue 只在调用层配置方向、输出区间、字段和最终 shape，排序主体与资源规则统一。

命中语言适配层的选路流程属于设计、实现与验收约束。必须覆盖目标支持域内的全部适用模板，不得以兜底路线已覆盖正确性或降低开发复杂度为由省略。

生成输出配置时，还须保留所选模板的批处理粒度：对原本 B>1 合批的路径，最终写回工作域为 B×L，L=1 也不能默认退化为外层逐行提交小任务。适配范围包含恢复、截取和搬出，不能只保留排序原语就视为完整复用。具体约束见 [统一输出映射](generic/output-mapping.md#保持批处理粒度) 及命中语言的输出适配。

非尾轴输入经布局规范化后复用连续尾轴排序模板，输出再恢复原布局。物理转置条件与具体步骤见 [布局预处理](generic/shape-preprocessing.md)。

先读 [统一输出映射](generic/output-mapping.md) 确定生成参数，再读命中语言的排序模板与 API。算法描述和语言模板定义待实例化的执行方案：填入目标规格、平台资源和输出参数，按条件展开执行路径；处理域、并行分工、数据通道和生产消费关系都是该方案的组成部分。输出适配沿模板标出的接口完成，并将新增缓冲、运行保留和同步依赖带回选路及容量计算。

example 是执行方案的实现对照。按 [实现导航](source-map.md) 中已选语言的文件和函数，将模板节点对应到具体计算体及其依赖，再展开目标接口的差异。生成工程可自行组织类、函数和字段；接口语义以目标规格为准，执行方式按命中模板落实。描述与源码不一致时，定位具体差异及适用域，按语言适配层的替换约束处理。

## 设计细节的阅读路径

1. 公共语义和布局：[scene](generic/scene.md) → [输出映射](generic/output-mapping.md) → [shape 预处理](generic/shape-preprocessing.md)。
2. 排序主体：[计算流](generic/compute-flow.md)给出完整算法伪码与索引关系，[切分](generic/splitting.md)给出工作域、核分配和尾批推导。
3. 选择语言后，用 [AscendC 全局选路](adapters/ascendc/tiling.md) 或 [DSL 全局选路](adapters/dsl/tiling.md)确定候选，再读其 `templates/<算法>/` 下的 Tiling、Buffer、Kernel 三份文档。模板文档直接承载切分公式、参数、寻址、计算阶段、同步和可复算例子。
4. 输出适配、比较辅助及共用局部 Radix 按需补读。伪代码须展开影响性能的处理粒度、并行分工、访存和同步，不能只用“恢复”“发布”等阶段名留给生成者重新选机制。算法描述用于推导执行方案，example 用于逐节点核对计算体、具体 API 和版本适用域；已有计算体覆盖的节点沿用其执行方式，目标差异在模板的适配位置展开。

## 语言路由

语言由调用方显式提供，或读取 `spec.yaml` 的 `op.language`。缺失或未知时先报告输入缺口，不选择语言适配层。

| op.language | 加载内容 | 入口 |
|---|---|---|
| ascendc | 通用知识层 + AscendC 适配层 | [adapters/ascendc/index.md](adapters/ascendc/index.md) |
| dsl | 通用知识层 + CANNBotDSL 适配层 | [adapters/dsl/index.md](adapters/dsl/index.md) |
| 缺失或未知 | 报告输入缺口 | 不选择其他语言的 API/模板 |

## 通用知识层索引

按 [generic/index.md](generic/index.md) 的 purpose/read_when/keywords 检索表选择 scene、capability-contract、shape-preprocessing、splitting、liveness-fusion、sync-model、compute-flow、output-mapping。基本算法、资源模型和同步依赖在公共层，API 真值、平台常数与代码实例在 adapter。

## 消费规则

1. 一次任务只加载命中语言的适配层与公共层。
   设计与验收还必须读取该语言适配层的选路规则，按其中的路线覆盖要求逐项核对；命中入口文件不等于已落实算法路线。
2. 能力支持范围与资源限制以命中语言的 capability-mapping 为准；未知能力保留待确认。
3. 源码只支持其接口和 caller 条件内的事实；已有实现名称不能证明可直接复用或泛化。
4. 范式确定算法与约束，example 提供 API、布局和同步表达参考；独立交付时所需模板与公共原语随算子组织，不在运行期依赖 example。
   AscendC 的设计、编码与验收还须核对 [实现复用与替换约束](adapters/ascendc/design-spec.md#实现复用与替换约束)：逐路线记录实际排序原语与同步协议，缺少验证证据时保留缺口，不能仅保留 schedule 名称而整体替换实现。
5. 设计输出格式由调用方指定；AscendC 实现约束见 [design-spec.md](adapters/ascendc/design-spec.md)，DSL 映射见 [implementation-guide.md](adapters/dsl/implementation-guide.md)。

## 内容维护

范式只沉淀可复用的算法与实现约束；实现策略说明所需能力、资源和接口约束，不记录软件版本号、具体平台型号或本机环境信息。个案验证结果、性能测量口径、耗时与加速比留在库外报告，不写入范式，也不以报告作为理解算法的必读依赖。
