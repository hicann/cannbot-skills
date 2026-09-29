# Reduction 范式通用知识层索引

> **层级**：通用知识层（generic/），语言中立。与语言适配层（`../adapters/<lang>/`）组合消费：先经 [../patterns.md](../patterns.md) 的语言路由确定适配层，再按下表按需展开。

## 检索表

| 文档 | purpose | read_when | keywords |
|------|---------|-----------|----------|
| [scene.md](scene.md) | 场景模型：准入条件、A/R 轴与 pattern 坐标系、Base/Group/Empty 三模板分类、三段式数据流骨架、能力边界 | 判断算子是否属于本范式；建立全局心智模型 | 归约、A 轴、R 轴、pattern、tail-A/tail-R、全 R、空 tensor、Base、Group、Empty、Phase A/B |
| [capability-contract.md](capability-contract.md) | 能力谓词定义：通用层算法引用的语言/硬件能力参数 | 编写新语言适配层；理解算法中的能力依赖 | 谓词、能力、对齐粒度、树缓存容量、归约指令、融合链长 |
| [axis-preprocessing.md](axis-preprocessing.md) | 归约轴归一化、空 tensor 短路判定、四步合轴算法、pattern 形态与 MAX_PATTERN_RANK 推导 | Host 侧预处理设计；合轴推导 | axes 归一化、去 1、合轴、补 leading A、补 R 增广、EMPTY_A、EMPTY_R、pattern 上界 |
| [tree-reduction.md](tree-reduction.md) | 二分缓存树：精度动机、非 2 幂主尾配对、多级缓存写入层级与吸收规则 | 理解跨 chunk 累加的精度机制；归约循环设计 | 二分树、主尾配对、多级缓存、写入层级、吸收、树根、累加精度 |
| [liveness-fusion.md](liveness-fusion.md) | L/P 双层存活节点模型、公平基线、互斥分支流规则、融合决策方法论 | buffer 数量（P 值）推导；VF/链融合取舍分析 | 存活节点、L、P、持有法则、公平基线、互斥分支、融合、链长、峰值、buffer 复用 |
| [splitting.md](splitting.md) | UB 双切分三步算法、A 方向多核均衡切分、Group 2D 分核、Empty 切分约束、UB 预算不等式与 buffer 大小公式 | 切分策略设计；tiling 推导；UB 预算验证 | 双切分、切分轴、切分因子、valid/padded、大小核、2D 分核、workspace、预算不等式 |
| [sync-model.md](sync-model.md) | RAW/WAR 依赖识别、三层 WAR 检查、循环间反向同步、跨核栅栏模型 | 同步设计；从持有法则推导同步点 | RAW、WAR、三层 WAR、首轮/末轮、事件对、跨核栅栏、两阶段 |
| [tuple-reduce.md](tuple-reduce.md) | Tuple Reduce 场景模型：N 次独立归约的循环复用、存活节点一条流规则 | 多输出/多次独立归约算子设计 | Tuple Reduce、N_REDUCES、循环复用、禁止乘 N、互斥分支 |

## 使用规则

1. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分为更小的子问题
2. 通用层结论中的能力依赖一律引用 [capability-contract.md](capability-contract.md) 的谓词；具体真值以命中语言适配层的 capability-mapping 为准
3. 通用层不含任何具体语言 API；文中出现 API 名处均为 AscendC 示例标注，其他语言按各自适配层映射
4. 具体语言的实现（代码模板、参考源码、生成规范）在适配层，不在本文档族内
