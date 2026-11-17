# Reduce CCU 接入卡

来源卡 `reduce-sole`，逐字节范围及reviewed_heads见 [基线](../knowledge-baseline.json)。本页覆盖当前 **ReduceSoleExecutor / Sched Mesh**；仅列其他现成注册以便导航，不外推MS实现、NHR、Parallel/Sequence/OmniPipe或新算法的数据流。MATCH后复用，变化用`check_knowledge.py --op reduce --summary --explain`补查。

## 接口与数据单位

每rank输入C个元素，只root输出C个元素归约结果；非root输出不是结果断言对象。文件名是reduce_sole_executor，不是ins_v2_reduce_sole_executor；不要因名称不同转用多阶段Sequence契约。

| 接入点 | 已核对契约 |
|---|---|
| 调用方 | `src/ops/reduce/algorithm/executor/reduce_sole_executor.{h,cc}`；CalcRes和OrchestrateLoop均用`(param,userRank,algHierarchyInfo.infos[0])`构造template，资源阶段直接调CalcRes |
| 本轮字段 | inputSliceStride=0，outputSliceStride=dataSize_=整次输入向量bytes；in/outBuffBaseOff=本轮已处理元素数×w；sliceSize=tailSize=本轮count×w。当前mem2mem KernelRun不消费input/outputSliceStride，不把该outputStride解释成N份输出。 |
| loop与scratch | 此Mesh mem2mem继承scratch倍数0。普通Sole loop按UB_MAX_DATA_SIZE/dtype宽度计算count上限；没有借用Broadcast/AllReduce对称内存强制单loop的证明。 |
| 资源 | 所选template请求1个CCU kernel、slaveThreadNum=0、notifyNumOnMainThread=0，GetThreadNum=1；CCU handles来自resCtx.ccuKernels。channel消费按局部peer顺序跳过自身，不能用无序容器重排 |
| 重放入口 | loopTimes=1、engine=CCU且非OFFLOAD才保存FastLaunch上下文；重放默认构造template，不能依赖带参构造遗留的成员；实际threads和submitInfos均须与执行器消费一致 |
| 成本与构建 | template提供静态CalcCostCoeff；新算法成本有依据且非空。`.cc`按本算子的`template/ccu/CMakeLists.txt`和`kernel/CMakeLists.txt`接入，executor注册留在真实入口，不套AICPU device登记 |

定位template：`src/ops/reduce/algorithm/template/ccu/ccu_temp_reduce_mesh_1D_mem2mem.{h,cc}`；kernel在其`kernel/ccu_kernel_reduce_mesh1d_mem2mem.{h,cc}`。

## launch、缓存与算法边界

动态槽数 N+12：前9槽=input, output, token, isInputOutputEqual, inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize, encodedRepeat；随后N-1个chunkSize；最后goSize[0..3]。缓存布局是[槽数, launch前缀..., inBaseOff, outBaseOff]，FastLaunch从args+1发args[0]个槽，不能套固定15/16槽或把计数头发给LoadArg。 声明使用 [8rank启动契约样本](../../assets/launch-contract.reduce.example.json) 的`cache_launch_offset=1`；`updates.index`分别为物理缓存槽1/2，`loads.index`仍为启动槽0/1。其他rank按N+12重新填写。

param.root映射为mySubCommRoot_并进入kernelArg.rootId。root先做需要的本地拷贝，再用ReadReduce依次累积peer的chunk；非root也参加PreSync/PostSync。以N-1切分元素，余数前面的chunk多1个元素。N=1不能直接调用这个template（会除以N-1），API单rank路径另外验证。当前kernel单Event mask按1<<rank形成，局部rank编号达到16（N≥17）时必须按16位配额重新证明或限制，注册上限64本身不证明可用。

槽顺序以本卡对应的KernelRun/BuildTaskArgs和LoadArgs为准；注册期kernelArg不是uint64启动槽。token、root、dtype、reduceType、topology和派生字段改变时核对实际注册/缓存key与重新生成路径；本页不宣称框架已保证所有缓存失效条件。按 [同步契约](../api-and-synchronization.md#设计期同步契约检查) 检查每channel配额、Event/Notify及复用活性，不能以样本注册上限替代资源证明。

## 注册与选路

现成Sched名为`CcuSchedReduceSoleMesh`；MS名为`CcuMSReduceSoleMesh`。

Sched Mesh maxTopoLevelNum=1、maxRank当前64、Mesh/MeshClos、全mesh；禁PROD、inplace、INT8/INT64/UINT64/FP64。MS Mesh亦有maxTopoLevelNum=1和自身全mesh约束，只覆盖注册信息，MS实现另查。

`HCCL_ALGO`用解析器DSL，例如当前Sched模式`reduce:sole{mesh}`再由ComposeAlgoName组成注册名；HCCL目录名`reduce`、DSL前缀`reduce`与HVM算子参数不是通用同名规则。hard属性、成本、实际模式和算法命中分别核验；显式优先级/tuner会影响cost_table的soft check，不能只靠opCustomCheck=false证明永不默认竞争。

## 测试输入与验收

先用 [按算子输入模型](../test-input-preflight.md) 的`preview_cases.py --op reduce`，按工具真实count/bytes制定预算。sum/min/max按需求分别核对root结果，root=0、非零及末rank；非root输出只做所需的边界保护检查。覆盖N-1不整除元素数、少元素零chunk与root本地贡献；PROD不可用该样本证明支持。

本卡所列API新入口对零count直接成功、正常单rank在Selector前走SingleRankProc；因此零数据/单rank用例不能用正常算法名命中断言证明CCU kernel覆盖，其他legacy/兼容分支按实际路径另核对。正常功能验收仍需目标算法命中与Checker证据；输入预检、静态声明及源码MATCH都不是功能PASS。
