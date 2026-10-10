# Broadcast 范式通用知识层索引

> **层级**：通用知识层（generic/），语言中立。与语言适配层（`../adapters/<lang>/`）组合消费：先经 [../patterns.md](../patterns.md) 的语言路由确定适配层，再按下表按需展开。

## 检索表

| 文档 | purpose | read_when | keywords |
|------|---------|-----------|----------|
| [scene.md](scene.md) | 场景模型：准入条件、坐标系理论、数据流骨架、编译期特化、能力边界 | 判断算子是否属于本范式；建立全局心智模型 | 广播、准入、effective_shape、maximumBroShape、数据流、RANK |
| [capability-contract.md](capability-contract.md) | 能力谓词定义：通用层算法引用的语言/硬件能力参数 | 编写新语言适配层；理解算法中的能力依赖 | 谓词、能力、对齐粒度、链长、融合 |
| [shape-preprocessing.md](shape-preprocessing.md) | shape 归一化：补 1 / 去 1 / 归一 / 合轴四步算法与 broadcast 兼容校验 | Host 侧预处理设计；shape 归一化推导 | 补 1、去 1、合轴、坐标系、兼容校验 |
| [splitting.md](splitting.md) | 片上内存单切分与多核均衡切分算法 | 切分策略设计；tiling 推导 | 切分轴、尾块、内轴、外轴、多核、tile、perBufElems |
| [liveness-fusion.md](liveness-fusion.md) | L/P 存活节点模型、持有法则、融合决策方法论 | buffer 数量（P 值）推导；融合取舍分析 | 存活节点、L、P、持有法则、融合、峰值、buffer 复用 |
| [sync-model.md](sync-model.md) | RAW/WAR 依赖识别与同步点推导模型 | 同步设计；从持有法则推导同步点 | RAW、WAR、同步、执行单元、流水线 |
| [compute-flow.md](compute-flow.md) | 计算流伪码规范：P 分析到编码之间的桥梁 | 复杂算子需要写计算流伪码时 | 伪码、计算流、驻留计数、dtype 分路径 |
| [verification.md](verification.md) | golden/opt 交叉验证方法论与验证 harness | 验证计算流 / 切分设计正确性 | golden、opt、交叉验证、随机 case、harness |

## 使用规则

1. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分为更小的子问题
2. 通用层结论中的能力依赖一律引用 [capability-contract.md](capability-contract.md) 的谓词；具体真值以命中语言适配层的 capability-mapping 为准
3. 通用层不含任何具体语言 API；文中出现 API 名处均为 AscendC 示例标注，其他语言按各自适配层映射
4. 具体语言的实现（代码模板、参考源码、生成规范）在适配层，不在本文档族内
