# ReduceScatter CCU 接入卡

来源卡 `reduce-scatter-sole`，逐字节范围及reviewed_heads见 [基线](../knowledge-baseline.json)。本页覆盖当前 **InsV2ReduceScatterSoleExecutor / Sched Mesh**；仅列其他现成注册以便导航，不外推MS实现、NHR、Parallel/Sequence/OmniPipe或新算法的数据流。MATCH后复用，变化用`check_knowledge.py --op reduce_scatter --summary --explain`补查。

## 接口与数据单位

API recvCount=C 是每rank输出元素数；每rank输入为N×C个元素。rank j输出所有输入的第j块归约，不能把输入总元素数当recvCount。

| 接入点 | 已核对契约 |
|---|---|
| 调用方 | `src/ops/reduce_scatter/algorithm/executor/ins_v2_reduce_scatter_sole_executor.{h,cc}`；CalcRes和OrchestrateLoop均用`(param,userRank,algHierarchyInfo.infos[0])`构造template，资源阶段直接调CalcRes |
| 本轮字段 | inputSliceStride=dataSize_=整次每rank输出bytes（C×w）；outputSliceStride=0；in/outBuffBaseOff=本轮已处理输出元素数×w；sliceSize=tailSize=本轮输出count×w。template再加局部rank×inputSliceStride，不用本轮slice替换整次跨度。 |
| loop与scratch | CalcScratchMultiple=templateRankSize_；scratchRepeatStride=N×本轮normalSliceSize。CCU分支先从hcclBuff.size扣TMP_MEM_RESERVE_SIZE（当前1MiB，在executor文件内定义），再除scratch倍数并向下按HCCL_MIN_SLICE_ALIGN对齐，与UB_MAX_DATA_SIZE取小后除dtype宽度得到maxCountPerLoop。先保证缓冲区足够，避免无符号减法下溢。executor按每rank输出count分轮；supportSymmetricMemory分支强制loopTimes=1，预测多loop须排除该分支。 |
| 资源 | 所选template请求1个CCU kernel、slaveThreadNum=0、notifyNumOnMainThread=0，GetThreadNum=1；CCU handles来自resCtx.ccuKernels。channel消费按局部peer顺序跳过自身，不能用无序容器重排 |
| 重放入口 | loopTimes=1、engine=CCU且非OFFLOAD才保存FastLaunch上下文；重放默认构造template，不能依赖带参构造遗留的成员；实际threads和submitInfos均须与执行器消费一致 |
| 成本与构建 | template提供静态CalcCostCoeff；新算法成本有依据且非空。`.cc`按本算子的`template/ccu/CMakeLists.txt`和`kernel/CMakeLists.txt`接入，executor注册留在真实入口，不套AICPU device登记 |

定位template：`src/ops/reduce_scatter/algorithm/template/ccu/ccu_temp_reduce_scatter_mesh_1D_mem2mem.{h,cc}`；kernel在其`kernel/ccu_kernel_reduce_scatter_mesh1d_mem2mem.{h,cc}`。

## launch、缓存与算法边界

16槽：input, output, token, scratch, currentRankSliceInputOffset, currentRankSliceOutputOffset, inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize, scratchRepeatStride, encodedRepeat, goSize[0..3]。encodedRepeat=UINT64_MAX-repeatNum。缓存尾槽16/17/18为基址偏移；FastLaunch修补0/1/3，只发16槽。

kernel读取每个peer的本rank目标块到scratch，做本地归约后写本rank输出；没有AllGather阶段。template在sliceSize和tailSize均零时直接返回；正常路径的PreSync/PostSync不能因为某rank零片而失配。不能机械反转AllGather的同步。

槽顺序以本卡对应的KernelRun/BuildTaskArgs和LoadArgs为准；注册期kernelArg不是uint64启动槽。token、root、dtype、reduceType、topology和派生字段改变时核对实际注册/缓存key与重新生成路径；本页不宣称框架已保证所有缓存失效条件。按 [同步契约](../api-and-synchronization.md#设计期同步契约检查) 检查每channel配额、Event/Notify及复用活性，不能以样本注册上限替代资源证明。

## 注册与选路

现成Sched名为`CcuSchedReduceScatterSoleMesh`；MS名为`CcuMSReduceScatterSoleMesh`。

Sched Mesh maxTopoLevelNum=2、maxRank当前64、Mesh/MeshClos；禁PROD、inplace、INT8/INT64/UINT64/FP64；拒绝level2UbRtp；MeshClos按PcieMix/全mesh或mesh=clos且rank≤4筛选。MS Mesh maxTopoLevelNum=1，MeshClos条件不同，需读自身注册块。

`HCCL_ALGO`用解析器DSL，例如当前Sched模式`reducescatter:sole{mesh}`再由ComposeAlgoName组成注册名；HCCL目录名`reduce_scatter`、DSL前缀`reducescatter`与HVM算子参数不是通用同名规则。hard属性、成本、实际模式和算法命中分别核验；显式优先级/tuner会影响cost_table的soft check，不能只靠opCustomCheck=false证明永不默认竞争。

## 测试输入与验收

先用 [按算子输入模型](../test-input-preflight.md) 的`preview_cases.py --op reduce_scatter`，按工具真实count/bytes制定预算。对每个输出rank核对块归属和全部rank贡献；至少有多rank不相同输入，避免错误块也得到同值。外层loop边界按recvCount验证；换地址重放同时检查input、output、scratch。

本卡所列API新入口对零count直接成功、正常单rank在Selector前走SingleRankProc；因此零数据/单rank用例不能用正常算法名命中断言证明CCU kernel覆盖，其他legacy/兼容分支按实际路径另核对。正常功能验收仍需目标算法命中与Checker证据；输入预检、静态声明及源码MATCH都不是功能PASS。
