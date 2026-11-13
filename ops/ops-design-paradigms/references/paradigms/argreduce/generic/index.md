# ArgReduce 范式通用知识层索引

> **层级**：通用知识层（generic/），语言中立。与语言适配层（`../adapters/<lang>/`）组合消费：先经 [../patterns.md](../patterns.md) 的语言路由确定适配层，再按下表按需展开。

## 检索表

| 文档 | purpose | read_when | keywords |
|------|---------|-----------|----------|
| [scene.md](scene.md) | 场景模型：准入条件、A1/R/A0 轴折叠坐标系、EMPTY/COPY/AR/ARA 四场景分类、三段式数据流骨架、能力边界 | 判断算子是否属于本范式；场景判定；建立全局心智模型 | 选择、argmax、argmin、A1、R、A0、轴折叠、连续、跨步、AR、ARA、RA、COPY、EMPTY、keep_dims、indice |
| [selection-semantics.md](selection-semantics.md) | 选择语义代数：(值, 位置) 二元组、主规则与平局规则、NaN 语义、可结合性不变量、主规则实例注册表、位置流生成与变换 | 确定胜负规则；平局/NaN 语义结论；跨块跨核合并正确性论证；衍生/泛化算子语义推导 | 二元组、主规则、平局、first-occurrence、last-occurrence、NaN、可结合性、结合律、位置流、索引偏移、实例注册 |
| [capability-contract.md](capability-contract.md) | 能力谓词定义：通用层算法引用的语言/硬件能力参数 | 编写新语言适配层；理解算法中的能力依赖 | 谓词、能力、索引跟踪、向量比较、位置存储、通道宽度、栅栏、workspace |
| [splitting.md](splitting.md) | UB 三通道预算（值流+位置流+输出/暂存）、FullLoad/Split 判定、A1/A0 多核切分、GroupReduce 触发与 workspace 模型、跨块索引偏移与位置精度 | 切分策略设计；tiling 推导；GroupReduce 触发判定；workspace 布局 | 三通道、值流、位置流、全载、分载、切块、切核、GroupReduce、workspace、基址偏移、位置精度 |
| [sync-model.md](sync-model.md) | 同步模型：RAW/WAR 依赖的白话解释、核内同步点、跨核同步（仅 GroupReduce）、AscendC/DSL 同步原语分工 | 同步设计；判断哪里必须等；GroupReduce 两阶段衔接 | 同步、RAW、WAR、持有法则、双缓冲、全核栅栏、SyncAll、RingBuffer、事件对 |
| [variants.md](variants.md) | 变体挂载：谓词首达（arg_first / find_index）、top-k 宽度扩展、多输出独立选择 | 泛化算子（非 extremum 主规则）语义与设计；多输出算子 | 谓词、首达、find_index、top-k、宽度 k、列表归并、多输出 |

## 使用规则

1. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分为更小的子问题
2. 通用层结论中的能力依赖一律引用 [capability-contract.md](capability-contract.md) 的谓词；具体真值以命中语言适配层的 capability-mapping 为准
3. 通用层不含任何具体语言 API；文中出现 API 名处均为 AscendC 示例标注，其他语言按各自适配层映射
4. 具体语言的实现（代码模板、参考源码、生成规范）在适配层，不在本文档族内
