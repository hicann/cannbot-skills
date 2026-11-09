# ArgReduce 范式 CANNBotDSL 语言适配层

> **层级**：CANNBotDSL 语言适配层（adapters/dsl/）。语言中立的算法与模型见 [../../generic/index.md](../../generic/index.md)；范式入口与语言路由见 [../../patterns.md](../../patterns.md)。

## 1 定位

本目录承载 ArgReduce 范式中**强耦合 CANNBotDSL（下文简称 DSL）**的知识：能力谓词真值、通用层模型到 DSL 三层结构（公开入口 / Host launcher / Kernel）的落地映射、实现约束。开发语言为 `dsl` 时，与通用知识层组合消费：

```
通用算法/模型（generic/）× DSL 实现细节（本目录）→ CANNBotDSL 设计与代码产出
```

**与 AscendC 适配层的关系**：`adapters/ascendc/` 的 8 模式体系（mode-mapping 决策树、TilingKey 机制）是 **AscendC 工程形态的落地知识**，DSL 任务不加载、不参考；DSL 的模式/切分决策直接消费通用层 [../../generic/splitting.md](../../generic/splitting.md)（FullLoad/Split/多核切分/GroupReduce 正交维度）。

## 2 阅读路径

| 任务阶段 | 入口 | 内容 |
|------|------|------|
| 能力映射 | [capability-mapping.md](capability-mapping.md) | 全部谓词的 DSL 真值（类目事实 / 待验证标注）、收敛规则、语言级约束 |
| 实现指引 | [implementation-guide.md](implementation-guide.md) | 场景判定/选择语义/位置流/切分/GroupReduce 的 DSL 落地映射与验证义务 |
| 通用知识 | [../../generic/index.md](../../generic/index.md) | 场景模型、选择语义代数、切分与资源、变体（语言中立，全部复用） |

## 3 消费规则

1. 语言路由命中本适配层后，**只读本适配层 + 通用层**，不加载 AscendC 适配层（互斥规则见 [../../patterns.md](../../patterns.md)）
2. **真值纪律**：能力谓词以 [capability-mapping.md](capability-mapping.md) 为准——"类目存在"不等于"能力已验证"；引用待验证谓词的结论必须声明假设并给出验证义务，**禁止编造 DSL API 签名/参数/行为**
3. 单个子问题最多展开 5 个叶子文档
4. 设计产出必须包含：场景判定（通用层）、语义三声明（平局/NaN/空 R）、待验证假设清单与验证义务
