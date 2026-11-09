---
name: ops-design-paradigms
description: 当需要设计或修订 Broadcast、Reduction、ArgReduce、Contraction 等算子范式方案，或查询对应算法、资源规划、语言实现约束和故障经验时使用；提供 AscendC regbase 与 CANNBotDSL 的设计依据、适用条件和验证要点。
---

# 算子通用设计知识与范式指导

根据算子语义和开发语言提供范式知识，支持方案设计、实现选择与问题定位。内容分为语言中立的算法模型和语言专用的实现约束，按当前问题读取。

## 输入与输出

- 输入：算子规格或等价的数学语义与范式信息、明确的开发语言（`ascendc` 或 `dsl`）、目标平台，以及待解决的设计或实现问题。文件路径由调用方提供。
- 缺少语言或范式信息时，返回具体缺项；确认后再选择专用参考。
- 输出：适用的算法与实现建议、约束及其依据、资源与同步分析、尚待验证的事实。产物格式和保存位置由调用方确定。
- 路由类问题的回答必须逐条覆盖三项：① 判定依据（命中的规则，并说明标签匹配按**集合包含**判断、**顺序与额外标签不影响**）；② 资料路径（实际读取的文件与加载顺序，含按需加载条件）；③ 入口纪律（方案语言无关、按步骤推进并按需加载；芯片规格等前置信息缺失时向用户补齐，无法提供则标注"待提供"，不猜测）。
  ④ 当问题同时包含设计方法诉求（如"未注册范式的 Tiling 设计怎么做"）时，除①-③外还必须按所用方法论**逐项概述设计要素**，不能只做算子定制化展开：走通用方法论兜底时须覆盖多核切分策略（负载均衡、数据局部性、尾块处理）、UB 切分与 chunk 规划（按目标芯片 UB 容量预算）、buffer 规划（inQueue/outQueue/中间 buffer、double buffer、总和校验）、分支场景覆盖（dtype/shape/对齐/边界/架构差异）、TilingKey 与 Tiling 字段设计，并显式说明命中专用范式时优先专用范式知识、通用方法论仅作为通用基线。

## 范式与语言选择

1. 先读取 [知识索引](references/index.md)，按当前问题选择入口。
2. 范式标签判定（只看标签集合，**与顺序、额外标签无关**）：
   - 记 `P = op.paradigms`。若 `Contraction ∈ P` 且（`Reduction ∈ P` 或 `Broadcast ∈ P`）：
     - 先核对实际公式：含注意力分数计算、行归一化与加权值聚合时，读取
       [FA 组合范式](references/paradigms/fused_composite/fa/patterns.md)，不把三个范式的资料逐项叠加；
     - 标签命中但公式不匹配时，作为**其余组合**按范式索引逐项读取。
   - 不满足上述集合条件时，统一按 [范式索引表](references/paradigms/routes.yaml) 逐项查找。
3. 多个范式覆盖不同设计维度时分别按需读取；未注册范式使用已有的对应语言通用方法，缺少对应方法时报告资料缺项。
4. 每个范式的 `patterns.md` 根据调用方提供的语言选择适配层。只读取通用层和对应语言的适配层；其明确引用的配套运行时或同步章节按需读取。缺少适配层时保留通用算法，语言 API 与设备能力依据调用方提供的资料确认，不回退到其他语言的实现。

## 参考选择

| 当前问题 | 入口 |
|---|---|
| Broadcast、Reduction、Contraction、IndexGather、ScatterUpdate 方案与资源规划 | [范式索引表](references/paradigms/routes.yaml) |
| Ascend C regbase 开发方法 | [开发指导](references/knowledge/regbase_development_guide.md) |
| Ascend C API、实现模式、精度风险或开发经验 | [知识分类索引](references/knowledge/index.md) |
| 缺少专用范式或语言适配层 | [Ascend C 通用方法](references/paradigms/general-methodology.md)、[DSL 通用方法](references/paradigms/general-methodology-dsl.md) |
| 选择现有 Ascend C 实现作为参考 | [开源算子表](references/reference-ops/open_source_operator_table.md) |

`knowledge/` 与 `reference-ops/` 是 Ascend C regbase 资料，DSL 问题仅使用相关通用层和 DSL 适配资料。

## 使用约束

- 进入 `api/`、`patterns/`、`pitfalls/` 或 `dev-experience/` 时先读目录索引，按 `purpose`、`read_when`、`keywords` 选取叶子文档。单个子问题最多展开 5 份，仍未解决则拆分问题。
- 能力谓词以对应语言的 `capability-mapping.md` 为依据；标注“待验证”的项须作为假设和验证义务输出。
- 实现参考用于核对 API 与组织方式；名称或类型匹配不能证明可以直接复制或融合，算法与参数需依据当前规格判断。
- 明确资料对应的平台。兼容性经验不能直接作为 `ascend950 / regbase` 的已验证结论。
- 设计输出应能追溯到规格和参考资料；缺少 API、资源或运行证据时报告具体缺口。
