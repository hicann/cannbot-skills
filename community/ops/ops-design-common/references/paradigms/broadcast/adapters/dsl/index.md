# Broadcast 范式 CANNBotDSL 语言适配层

> **层级**：CANNBotDSL 语言适配层（adapters/dsl/）。语言中立的算法与模型见 [../../generic/index.md](../../generic/index.md)；范式入口与语言路由见 [../../patterns.md](../../patterns.md)。

## 1 定位

本目录承载 Broadcast 范式中**强耦合 CANNBotDSL（下文简称 DSL）**的知识：DSL 实体映射、能力谓词真值、代码组织与实现约束。开发语言为 `dsl` 时，与通用知识层组合消费：

```
通用算法/模型（generic/）× DSL 实现细节（本目录）→ CANNBotDSL 设计与代码产出
```

## 2 能力映射

本语言对通用层能力谓词的真值与 DSL 实体见 [capability-mapping.md](capability-mapping.md)。通用层算法中引用的谓词（对齐粒度、随路广播能力、融合链长等）一律以该文件填值为准；标注**待验证**的谓词不得当作已验证真值使用，须在产出中显式声明假设。

## 3 阅读路径

| 任务阶段 | 入口 | 内容 |
|------|------|------|
| 方案设计 | [../../generic/index.md](../../generic/index.md)（通用层算法）→ [capability-mapping.md](capability-mapping.md)（谓词真值） | 场景模型 / shape 归一化 / 切分 / 存活节点等语言中立算法，叠加 DSL 能力真值与落地差异 |
| DSL 代码组织与实现 | [implementation-guide.md](implementation-guide.md) | 通用层数据流骨架到 DSL 入口/Host/Kernel 三层的映射、Buffer/Channel 规划、尾块与同步 |

## 4 补充输入

设计需要公开 DSL API 资料、目标设备能力和已有实现参考时，由调用方提供相应资料。只依据实际提供的证据选择 API、资源和同步写法；缺少证据时列出待确认事实。

## 5 消费规则

1. 语言路由命中本适配层后，**只读本适配层 + 通用层**，不加载 ascendc 适配层、不参考其 API 与代码模板
2. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分子问题
3. 能力谓词以 [capability-mapping.md](capability-mapping.md) 为准；**待验证**谓词支撑的设计结论必须在产出中声明假设并给出验证义务，禁止写成已验证事实
4. **设计文档生成**：使用 `references/paradigms/general-methodology-dsl.md` 通用方法与通用层知识，并按 [capability-mapping.md](capability-mapping.md) §4 和 [implementation-guide.md](implementation-guide.md) 补充 DSL 实现约束。
