# 算子范式知识索引

按具体设计问题、知识类型和开发语言选择参考。本模块提供算法、实现约束与验证依据。

## 检索规则

1. 先应用 Skill 入口的组合范式规则；未命中组合时，统一从 [范式索引表](paradigms/routes.yaml) 查找 `op.paradigms`，不通过扫描目录推断注册状态。
2. 范式入口按调用方明确提供的 `op.language` 选择适配层。缺失或未知时返回输入缺项，不默认选择 Ascend C。
3. 读取语言中立的通用层和一个对应语言的适配层。DSL 范式没有专用适配层时，使用 [DSL 通用方法](paradigms/general-methodology-dsl.md) 并核实所需 API 和设备事实。
4. 进入知识分类目录后先读该目录的 `index.md`；单个子问题最多展开 5 份叶子文档。

## 按问题选择

| 问题 | 参考 |
|---|---|
| Broadcast 算法、形状归一化与切分 | [Broadcast 入口](paradigms/broadcast/patterns.md) |
| Reduction 归约轴、缓存树与资源规划 | [Reduction 入口](paradigms/reduction/patterns.md) |
| Contraction、Reduction、Broadcast 组合 | [FA 组合范式](paradigms/FA/patterns.md) |
| Ascend C regbase 开发方法 | [开发指导](knowledge/regbase_development_guide.md) |
| API 约束与签名 | [API 索引](knowledge/api/index.md) |
| Kernel、VF 与 Reg-MicroAPI 组织 | [实现模式](knowledge/patterns/index.md) |
| 精度、数值稳定性和错误定位 | [常见问题](knowledge/pitfalls/index.md) |
| 构建、验证与上下文恢复 | [开发经验](knowledge/dev-experience/index.md) |
| 选择现有 Ascend C 实现 | [开源算子表](reference-ops/open_source_operator_table.md) |

## 平台与语言范围

- `ascend950 / regbase`：根据问题读取 [知识分类索引](knowledge/index.md) 下的 API、模式和经验。
- `ascend910b / membase`：只在明确的兼容性问题中参考对应资料，核对适用平台。
- `dsl / CANNBotDSL`：从范式入口选择 DSL 适配层。`knowledge/` 和 `reference-ops/` 属于 Ascend C regbase，不作为 DSL 默认参考。

算子表只用于筛选可检查的实现；能否复用必须结合具体代码、规格和平台约束判断。
