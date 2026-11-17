# AllReduce CCU 接入卡

来源卡 `all-reduce-sole`，逐字节范围及reviewed_heads见 [基线](../knowledge-baseline.json)。本页覆盖当前 **InsV2AllReduceSoleExecutor / Sched Mesh**；仅列其他现成注册以便导航，不外推MS实现、NHR、Parallel/Sequence/OmniPipe或新算法的数据流。MATCH后复用，变化用`check_knowledge.py --op all_reduce --summary --explain`补查。

## 接口与数据单位

每rank输入 C 个元素，所有rank输出同位置的归约结果；count 是整个向量元素数，不能再按rank把API count除一次。

| 接入点 | 已核对契约 |
|---|---|
| 调用方 | `src/ops/all_reduce/algorithm/executor/ins_v2_all_reduce_sole_executor.{h,cc}`；CalcRes和OrchestrateLoop均用`(param,userRank,algHierarchyInfo.infos[0])`构造template，资源阶段直接调CalcRes |
| 本轮字段 | inputSliceStride=outputSliceStride=0；in/outBuffBaseOff=已处理元素数×dtypeBytes；sliceSize=tailSize=本轮count×dtypeBytes。 |
| loop与scratch | CalcScratchMultiple 返回 templateRankSize_；普通loop上限同时受scratch预算和UB_MAX_DATA_SIZE限制，不能直接假定256MiB可用。supportSymmetricMemory分支会强制loopTimes=1；预测多loop须明确没有走此分支。 |
| 资源 | 所选template请求1个CCU kernel、slaveThreadNum=0、notifyNumOnMainThread=0，GetThreadNum=1；CCU handles来自resCtx.ccuKernels。channel消费按局部peer顺序跳过自身，不能用无序容器重排 |
| 重放入口 | loopTimes=1、engine=CCU且非OFFLOAD才保存FastLaunch上下文；重放默认构造template，不能依赖带参构造遗留的成员；实际threads和submitInfos均须与执行器消费一致 |
| 成本与构建 | template提供静态CalcCostCoeff；新算法成本有依据且非空。`.cc`按本算子的`template/ccu/CMakeLists.txt`和`kernel/CMakeLists.txt`接入，executor注册留在真实入口，不套AICPU device登记 |

定位template：`src/ops/all_reduce/algorithm/template/ccu/ccu_temp_all_reduce_mesh_1D_mem2mem.{h,cc}`；kernel在其`kernel/ccu_kernel_all_reduce_mesh1d_mem2mem.{h,cc}`。

## launch、缓存与算法边界

15槽：input, output, token, scratch, currentRankSliceInputOffset, currentRankSliceOutputOffset, normalSliceSize, lastSliceSize, mySliceSize, sliceOffset, isInputOutputEqual, goSize[0..3]。cachedArgs末尾15/16/17为in/out/scratch基址偏移，不送入kernel。FastLaunch修补槽0/1/3，发15槽。

CalcSlice按dtype宽度将本轮向量划给局部rank：非末rank取 floor(bytes/(N×w))×w，末rank吸收余量。kernel先读取各rank对应片归约，再向所有输出广播该片；mySliceSize=0仍要走PreSync/PostSync。小于N个元素可触发内部零片，但非整除CLI字节本身不证明它。

槽顺序以本卡对应的KernelRun/BuildTaskArgs和LoadArgs为准；注册期kernelArg不是uint64启动槽。token、root、dtype、reduceType、topology和派生字段改变时核对实际注册/缓存key与重新生成路径；本页不宣称框架已保证所有缓存失效条件。按 [同步契约](../api-and-synchronization.md#设计期同步契约检查) 检查每channel配额、Event/Notify及复用活性，不能以样本注册上限替代资源证明。

## 注册与选路

现成Sched名为`CcuSchedAllReduceSoleMesh`；MS名为`CcuMSAllReduceSoleMesh`。

Sched Mesh maxTopoLevelNum=2、maxRank=CCU_SCHED_MAX_RANK_SIZE（当前64）、Mesh/MeshClos；禁PROD、inplace、INT8/INT64/UINT64/FP64。MeshClos纯UB还要求mesh/clos数量相等且rank≤4；PcieMix要求全mesh连接。MS Mesh另有maxTopoLevelNum=1及不同MeshClos限制，不沿用Sched范围。

`HCCL_ALGO`用解析器DSL，例如当前Sched模式`allreduce:sole{mesh}`再由ComposeAlgoName组成注册名；HCCL目录名`all_reduce`、DSL前缀`allreduce`与HVM算子参数不是通用同名规则。hard属性、成本、实际模式和算法命中分别核验；显式优先级/tuner会影响cost_table的soft check，不能只靠opCustomCheck=false证明永不默认竞争。

## 测试输入与验收

先用 [按算子输入模型](../test-input-preflight.md) 的`preview_cases.py --op all_reduce`，按工具真实count/bytes制定预算。sum及需求指定的min/max分别验证；浮点按Checker精度，整数避免溢出掩盖贡献遗漏。验证每个输出含每个rank恰好一次贡献；内部非均匀分片、count<N、多loop、地址变化重放分别证明。

本卡所列API新入口对零count直接成功、正常单rank在Selector前走SingleRankProc；因此零数据/单rank用例不能用正常算法名命中断言证明CCU kernel覆盖，其他legacy/兼容分支按实际路径另核对。正常功能验收仍需目标算法命中与Checker证据；输入预检、静态声明及源码MATCH都不是功能PASS。
