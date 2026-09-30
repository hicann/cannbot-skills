


# Sparse Attention
- [A2/A3 注意力类算子的分段流水与向量归约实测事实](device-pipeline-and-reduce-facts.md)
- [blockX/blockY 泛化约束](blockxy-generalization.md)
- [CATLASS 组件映射](catlass-components.md)
- [Golden 与验证](golden-and-validation.md)
- [多行 Vector epilogue](vector-epilogue-design.md)
- [性能假设与验证](performance-hypotheses.md)
- [执行与完整调用测量](execution-and-measurement.md)
- [稀疏任务与四阶段流水](scheduling-and-pipeline.md)
- [范围与使用边界](scope-and-evidence.md)
- [语义与十二分派](semantics-and-dispatch.md)
- [宿主暂存区尺寸与分批驻留模式](host-workspace-and-batching-patterns.md)
- [优化变换的准入与判定纪律](optimization-admission-rules.md)
- [BSA Arch22 任务边界与槽复用同步模式](task-boundary-sync-patterns.md)
- [BSA Arch22 高吞吐流水结构规则](throughput-structure-rules.md)
本 family 收录 DAV_2201（Arch22）的非量化 block_sparse_attention 知识。先确认输入域与平台范围，再按技术问题读取条目。
条目按适用条件、技术机制、失败表现与验证方法组织。
知识不替代当前实现的验收，也不扩展到 Ascend950、量化或 Linear Attention。
