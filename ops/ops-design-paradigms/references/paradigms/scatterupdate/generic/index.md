# ScatterUpdate 范式通用知识层索引

> **层级**：通用知识层（generic/）与公共知识层（common/），语言中立。与语言适配层（`../adapters/<lang>/`）组合消费：先经 [../patterns.md](../patterns.md) 的语言路由确定适配层，再按下表按需展开。

## 检索表

| 文档 | purpose | read_when | keywords |
|------|---------|-----------|----------|
| [semantics.md](semantics.md) | 场景模型：覆盖写语义、src 表示（张量源/标量源）、index 形态（dim 两子场景 + nd + 展平 1D）、覆盖写变体推导 | 判断算子是否属于本范式；确定形态组合；推导变体归属 | 写散射、assign、覆盖写、张量源、标量源、dim、nd、坐标元组、变体 |
| [bounds.md](bounds.md) | 越界三形态决策方法论：优先级（声明→对标→默认 error）、二维独立决策、竞品证据、哨兵 key 机制契约、双模板一致性约束 | 越界形态决策；skip 哨兵设计；双模板越界语义一致性核验 | 越界、error、skip、neg-norm、负索引、哨兵、SENTINEL、双模板一致性 |
| [conflict-domain.md](conflict-domain.md) | 冲突域统一视角：key/b/s 三轴合成、排序量原则（= 索引个数，不含 b/s）、key 位宽自适应、空 tensor 早退原则 | Phase1 冲突 key 设计；排序量推导；key 位宽选型 | 冲突域、冲突 key、排序量、batch、slice、key 位宽、空 tensor 早退 |
| [capability-contract.md](capability-contract.md) | 能力谓词定义：SIMT/VF、DCache/UB、线程/寄存器、rank/index dtype、快除、mulhi、字节搬移，以及 L2 volatile、跨核同步、对齐单 store 原子、稳定排序、UB 对齐、workspace 模型 | 编写或核对语言适配层；确认设计中的能力依赖；判定 det/nondet 路径的硬件前提 | 能力谓词、SIMT、VF、DCache、UB、线程数、寄存器、rank、index dtype、快除、mulhi、字节搬移、L2 volatile、跨核同步、单 store 原子、稳定排序、workspace |
| [common/simt-hardware-model.md](../../common/simt-hardware-model.md) | 芯片级硬件模型：存储层级图、VF 调度执行模型、DCache 一致性与 volatile 三分法（含复合单 kernel 判定）、DCache 预留规则、SIMT 线程代价（UB 让出/寄存器分档）、超发核屏障参与原理、Bank 冲突 | 理解硬件约束对算法编排的影响；volatile/DCache 决策；UB 预算推导；线程数选型；多核编排 | 存储层级、VF、调度、DCache、volatile、L2、UB、预留、SIMT、线程数、寄存器、超发核、active_cores、Bank |
| [thread-layout.md](thread-layout.md) | SIMT 线程排布模型：det 三阶段共用核排布（Phase1/2/3 各自核数策略）、nondet 单阶段两步法、(b,k,s) 任务一维化公式、超发核 guard、空 tensor 早退与核数关系、线程数选型 | Host 侧 tiling 核数推导（det/nondet）；Phase3 任务分解；blockDim 计算 | 线程排布、核数、两步法、三阶段、blockDim、(b,k,s) 一维化、超发核、grid-stride |

## 使用规则

1. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分为更小的子问题
2. 通用层不含任何具体语言 API；文中出现 API 名处均为 AscendC 示例标注，其他语言按各自适配层映射
3. 具体语言的实现（代码模板、参考源码、生成规范）在语言适配层（AscendC：`../adapters/ascendc/`），不在本文档族内。**芯片级硬件事实（[common/simt-hardware-model.md](../../common/simt-hardware-model.md)）在公共层**——各适配层仅做 API 映射，不重复承载
4. 快除（UintDiv）与合轴方法论的真值源在 [../../common/uintdiv-rules.md](../../common/uintdiv-rules.md)
