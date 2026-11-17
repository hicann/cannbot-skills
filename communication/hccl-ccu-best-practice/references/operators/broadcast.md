# Broadcast CCU 接入卡

来源卡 `broadcast-sole`，逐字节范围及reviewed_heads见 [基线](../knowledge-baseline.json)。本页覆盖当前 **InsV2BroadcastSoleExecutor / Sched Mesh**；仅列其他现成注册以便导航，不外推MS实现、NHR、Parallel/Sequence/OmniPipe或新算法的数据流。MATCH后复用，变化用`check_knowledge.py --op broadcast --summary --explain`补查。

## 接口与数据单位

HcclBroadcast只有一个buf：root提供源数据，所有rank在同一buf接收。count是整个广播向量元素数，不是每个内部slice的元素数；无reduceType。

| 接入点 | 已核对契约 |
|---|---|
| 调用方 | `src/ops/broadcast/algorithm/executor/ins_v2_broadcast_sole_executor.{h,cc}`；CalcRes和OrchestrateLoop均用`(param,userRank,algHierarchyInfo.infos[0])`构造template，资源阶段直接调CalcRes |
| 本轮字段 | executor将inputPtr/outputPtr都设为param.inputPtr，inBuffType/outBuffType都为INPUT；两个SliceStride为0；基址偏移都是本轮字节偏移。FastLaunch上下文中的两基址与该别名关系需保持一致。 |
| loop与scratch | 该Mesh mem2mem类继承CcuAlgTemplateBase::CalcScratchMultiple=0；普通loop受UB_MAX_DATA_SIZE限制；supportSymmetricMemory分支会强制单loop并enableRemoteMemAccess。 |
| 资源 | 所选template请求1个CCU kernel、slaveThreadNum=0、notifyNumOnMainThread=0，GetThreadNum=1；CCU handles来自resCtx.ccuKernels。channel消费按局部peer顺序跳过自身，不能用无序容器重排 |
| 重放入口 | loopTimes=1、engine=CCU且非OFFLOAD才保存FastLaunch上下文；重放默认构造template，不能依赖带参构造遗留的成员；实际threads和submitInfos均须与执行器消费一致 |
| 成本与构建 | template提供静态CalcCostCoeff；新算法成本有依据且非空。`.cc`按本算子的`template/ccu/CMakeLists.txt`和`kernel/CMakeLists.txt`接入，executor注册留在真实入口，不套AICPU device登记 |

定位template：`src/ops/broadcast/algorithm/template/ccu/ccu_temp_broadcast_mesh_1D_mem2mem.{h,cc}`；kernel在其`kernel/ccu_kernel_broadcast_mesh1d_mem2mem.{h,cc}`。

## launch、缓存与算法边界

11槽：input, output, token, currentRankSliceInputOffset, currentRankSliceOutputOffset, inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize, allgatherOffset, encodedRepeat。缓存尾槽11/12为in/out偏移；FastLaunch修补0/1，只发11槽。

构造时把param.root（用户rank）映射为subCommRootId_，注册kernelArg.rootId；root不是运行期LoadArg槽。当前实现是Scatter后AllGather；normalSliceSize=本轮bytes/N，末片吸收余量，不能假定内部每片dtype对齐。每轮两阶段PostSync及零片Event补齐均需保留。

槽顺序以本卡对应的KernelRun/BuildTaskArgs和LoadArgs为准；注册期kernelArg不是uint64启动槽。token、root、dtype、reduceType、topology和派生字段改变时核对实际注册/缓存key与重新生成路径；本页不宣称框架已保证所有缓存失效条件。按 [同步契约](../api-and-synchronization.md#设计期同步契约检查) 检查每channel配额、Event/Notify及复用活性，不能以样本注册上限替代资源证明。

## 注册与选路

现成Sched名为`CcuSchedBroadcastSoleMesh`；MS名为`CcuMSBroadcastSoleMesh`。

Sched Mesh maxTopoLevelNum=2、maxRank当前64、Mesh/MeshClos且要求全mesh；MS Mesh maxTopoLevelNum=1，亦检查MeshClos全连接。Broadcast不归约，不继承AllReduce的dtype/PROD黑名单；仍核对目标API、引擎和拓扑。

`HCCL_ALGO`用解析器DSL，例如当前Sched模式`broadcast:sole{mesh}`再由ComposeAlgoName组成注册名；HCCL目录名`broadcast`、DSL前缀`broadcast`与HVM算子参数不是通用同名规则。hard属性、成本、实际模式和算法命中分别核验；显式优先级/tuner会影响cost_table的soft check，不能只靠opCustomCheck=false证明永不默认竞争。

## 测试输入与验收

先用 [按算子输入模型](../test-input-preflight.md) 的`preview_cases.py --op broadcast`，按工具真实count/bytes制定预算。root=0和至少一个非零root；各非root初值与root不同，检查所有rank输出均为root数据；包含root位于末rank及内部非均匀/零片场景。不同root必须使用与注册/缓存key一致的kernel，不仅修改运行期地址。

本卡所列API新入口对零count直接成功、正常单rank在Selector前走SingleRankProc；因此零数据/单rank用例不能用正常算法名命中断言证明CCU kernel覆盖，其他legacy/兼容分支按实际路径另核对。正常功能验收仍需目标算法命中与Checker证据；输入预检、静态声明及源码MATCH都不是功能PASS。
