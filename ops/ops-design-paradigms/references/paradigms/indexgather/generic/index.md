# IndexGather 范式通用知识层索引

> **层级**：通用知识层（generic/）与公共知识层（common/），语言中立。与语言适配层（`../adapters/<lang>/`）组合消费：先经 [../patterns.md](../patterns.md) 的语言路由确定适配层，再按下表按需展开。

## 检索表

| 文档 | purpose | read_when | keywords |
|------|---------|-----------|----------|
| [semantics.md](semantics.md) | 三型语义（A 单坐标整维切片 / B 坐标元组 / C 逐元素对齐）、术语映射、变体推导（embedding / take / masked_select 等映射到维度组合） | 判断算子是否属于本范式；确定分型；推导变体归属 | 读收集、gather、A 型、B 型、C 型、坐标元组、整维切片、逐元素对齐、batch_dims、变体 |
| [decode-bounds.md](decode-bounds.md) | 越界两步模型：Step 1 负索引处理（none/+dim/%dim）、Step 2 越界处理（err/fill/skip/%dim/clamp/undefined）、优先级（声明→对标→默认 none+err）、竞品证据（torch/ONNX/NumPy/JAX）、防漏判原则（原值域变换窄化最后）、易混淆语义注记（padding_idx/bool index/回绕两义） | 越界策略决策；竞品对标；区分性测试用例设计 | 越界、两步、负索引处理、+dim、%dim、err、fill、skip、clamp、undefined、padding_idx、回绕 |
| [decode-model.md](decode-model.md) | 坐标解码模型：两套步长分离（index 空间 × input 空间）、快除中转原则、字节级搬移、MergeAxis 合轴算法（切段规则/不可合轴判定/不变量） | 解码设计；合轴推导；广播正确性分析 | 坐标解码、步长分离、合轴、MergeAxis、切段、广播、快除、magic/shift |
| [capability-contract.md](capability-contract.md) | 能力谓词定义：SIMT/VF、DCache/UB、线程/寄存器、rank/index dtype、快除接口、mulhi、字节级搬移 | 编写或核对语言适配层；确认设计中的能力依赖 | 能力谓词、SIMT、VF、DCache、UB、线程数、寄存器、rank、index dtype、快除、mulhi、字节搬移 |
| [common/simt-hardware-model.md](../../common/simt-hardware-model.md) | 芯片级硬件模型：存储层级图、VF 调度执行模型、DCache 一致性与 volatile 三分法（含复合单 kernel 判定）、DCache 预留规则、SIMT 线程代价（UB 让出/寄存器分档）、超发核屏障、Bank 冲突 | 理解硬件约束对算法设计的影响；volatile/DCache 决策；UB 预算推导；线程数选型；多核编排 | 存储层级、VF、调度、DCache、volatile、L2、UB、预留、SIMT、线程数、寄存器、超发核、Bank |
| [thread-layout.md](thread-layout.md) | SIMT 线程排布模型：一维 grid-stride 原则、核数两步法算法与公式、PER_CORE_MIN 下限设计理由、空 tensor 早退与核数关系、线程数选型策略 | Host 侧 tiling 核数推导；线程排布设计；PER_CORE_MIN 选值 | 一维 grid-stride、核数两步法、blockFactor、PER_CORE_MIN、一元素一线程、线程数、空输出 |

## 使用规则

1. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分为更小的子问题
2. 通用层不含任何具体语言 API；文中出现 API 名处均为 AscendC 示例标注，其他语言按各自适配层映射
3. 具体语言的实现（代码模板、参考源码、生成规范）在语言适配层（AscendC：`../adapters/ascendc/`；DSL：`../adapters/dsl/`），不在本文档族内。**芯片级硬件事实（[common/simt-hardware-model.md](../../common/simt-hardware-model.md)）在公共层**——各适配层仅做 API 映射，不重复承载
4. 快除（UintDiv）方法论的真值源在 [../../common/uintdiv-rules.md](../../common/uintdiv-rules.md)
