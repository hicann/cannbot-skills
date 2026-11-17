# 按算子选择 CCU 接入与输入模型

只读本次算子一行的接入卡；先校验来源，MATCH且范围适用时直接复用。表中是已核对的起点，不限定新算法必须采用Mesh；Parallel/Sequence/OmniPipe、MS具体实现或不同executor只补查自身差异，不套用Sole。

| 算子目录 / preview --op | 接入卡 | 知识卡 ID | HCCL_ALGO DSL前缀 | HVM -o / 测试二进制 |
|---|---|---|---|---|
| all_gather | [AllGather Sole](operator-contracts.md) + [方法契约](framework-contracts.md) | all-gather-sole、all-gather-framework、all-gather-routing | allgather | allgather / all_gather_test |
| all_reduce | [AllReduce Sole](operators/all_reduce.md) | all-reduce-sole | allreduce | allreduce / all_reduce_test |
| reduce_scatter | [ReduceScatter Sole](operators/reduce_scatter.md) | reduce-scatter-sole | reducescatter | reduce_scatter / reduce_scatter_test |
| broadcast | [Broadcast Sole](operators/broadcast.md) | broadcast-sole | broadcast | broadcast / broadcast_test |
| reduce | [Reduce Sole](operators/reduce.md) | reduce-sole | reduce | reduce / reduce_test |
| scatter | [Scatter Sole](operators/scatter.md) | scatter-sole | scatter | scatter / scatter_test |

后续角色可运行`check_knowledge.py --hccl-root "$HCCL_ROOT" --hcomm-root "$HCOMM_ROOT" --op <算子目录名> --summary`；默认只选择该算子及公共卡。显式`--cards`时须列全部实际依赖。新五卡各自记录当前HEAD/文件哈希，不更新AllGather旧卡掩盖漂移；输入模型另按 [测试预检](test-input-preflight.md) 校验已安装hccl_test的公共来源和本算子文件。

公共约束：先推导新算法的片归属、rank映射、字节范围与同步活性，再按本算子真实executor填字段。root是用户rank，子域rank需要显式映射；reduceType/dtype/root等注册期参数改变时核对缓存key。launch槽与缓存metadata分开。测试工具输入可表达、目标实现支持、路径可达、真实Checker通过是独立结论，不互相替代。
