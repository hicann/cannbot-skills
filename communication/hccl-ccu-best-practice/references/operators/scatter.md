# Scatter CCU 接入卡

来源卡 `scatter-sole`，逐字节范围及reviewed_heads见 [基线](../knowledge-baseline.json)。本页覆盖当前 **InsV2ScatterSoleExecutor / Sched Mesh**；仅列其他现成注册以便导航，不外推MS实现、NHR、Parallel/Sequence/OmniPipe或新算法的数据流。MATCH后复用，变化用`check_knowledge.py --op scatter --summary --explain`补查。

## 接口与数据单位

root输入N×C个元素，各rank接收C个元素；API recvCount=C。只有root输入有语义，root自身也要接收自己的块；不进行归约或AllGather。

| 接入点 | 已核对契约 |
|---|---|
| 调用方 | `src/ops/scatter/algorithm/executor/ins_v2_scatter_sole_executor.{h,cc}`；CalcRes和OrchestrateLoop均用`(param,userRank,algHierarchyInfo.infos[0])`构造template，资源阶段直接调CalcRes |
| 本轮字段 | inputSliceStride=dataSize_=整次每rank接收bytes；outputSliceStride=0；in/outBuffBaseOff=已处理接收元素数×w；sliceSize=tailSize=本轮recvCount×w。kernel槽3虽名currentRankSliceInputOffset，host传的是inputSliceStride，由kernel再按rank累加。 |
| loop与scratch | CalcScratchMultiple=0；普通loop按UB_MAX_DATA_SIZE和dtype计算每rank接收count上限。没有通用的对称内存强制单loop结论。 |
| 资源 | 所选template请求1个CCU kernel、slaveThreadNum=0、notifyNumOnMainThread=0，GetThreadNum=1；CCU handles来自resCtx.ccuKernels。channel消费按局部peer顺序跳过自身，不能用无序容器重排 |
| 重放入口 | loopTimes=1、engine=CCU且非OFFLOAD才保存FastLaunch上下文；重放默认构造template，不能依赖带参构造遗留的成员；实际threads和submitInfos均须与执行器消费一致 |
| 成本与构建 | template提供静态CalcCostCoeff；新算法成本有依据且非空。`.cc`按本算子的`template/ccu/CMakeLists.txt`和`kernel/CMakeLists.txt`接入，executor注册留在真实入口，不套AICPU device登记 |

定位template：`src/ops/scatter/algorithm/template/ccu/ccu_temp_scatter_mesh1d.{h,cc}`；kernel在其`kernel/ccu_kernel_scatter_mesh1d.{h,cc}`。

## launch、缓存与算法边界

15槽：input, output, token, inputSliceStride, outputSliceStride, inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize, encodedRepeat, isInputOutputEqual, goSize[0..3]。缓存0/1保存的是in/out基址偏移；首次FastLaunch把当前基址加进去原地写回再发15槽。再次重放若未恢复相对偏移可能累加绝对地址，这是当前样本风险，不能复制为正确范式或宣称换地址重放已安全。

param.root映射为subCommRootId_并进入kernelArg.rootId。root向各peer输出写其目标块，自留块通过GroupCopy处理，非root参加前后同步。rootSliceSize按root是否末rank选择；isInputOutputEqual比较的是root自身的片地址。零片用EventRecord补齐；template双零slice/tail直接返回。

槽顺序以本卡对应的KernelRun/BuildTaskArgs和LoadArgs为准；注册期kernelArg不是uint64启动槽。token、root、dtype、reduceType、topology和派生字段改变时核对实际注册/缓存key与重新生成路径；本页不宣称框架已保证所有缓存失效条件。按 [同步契约](../api-and-synchronization.md#设计期同步契约检查) 检查每channel配额、Event/Notify及复用活性，不能以样本注册上限替代资源证明。

## 注册与选路

现成Sched名为`CcuSchedScatterSoleMesh`。

Sched Mesh maxTopoLevelNum=1、maxRank当前64、Mesh/MeshClos且全mesh、禁inplace；没有Sole MS Mesh注册，不能因其他算子有MS就自动宣称支持。NHR和OmniPipe是额外范围，只在目标使用时补查。

`HCCL_ALGO`用解析器DSL，例如当前Sched模式`scatter:sole{mesh}`再由ComposeAlgoName组成注册名；HCCL目录名`scatter`、DSL前缀`scatter`与HVM算子参数不是通用同名规则。hard属性、成本、实际模式和算法命中分别核验；显式优先级/tuner会影响cost_table的soft check，不能只靠opCustomCheck=false证明永不默认竞争。

## 测试输入与验收

先用 [按算子输入模型](../test-input-preflight.md) 的`preview_cases.py --op scatter`，按工具真实count/bytes制定预算。root=0、非零及末rank；root的各块使用可区分内容，逐rank确认取的是自己的块。验证root本地拷贝与远端写均完成；连续至少两次FastLaunch、每次切换地址时核对缓存来源及地址，重复调用次数不等于地址覆盖。

本卡所列API新入口对零count直接成功、正常单rank在Selector前走SingleRankProc；因此零数据/单rank用例不能用正常算法名命中断言证明CCU kernel覆盖，其他legacy/兼容分支按实际路径另核对。正常功能验收仍需目标算法命中与Checker证据；输入预检、静态声明及源码MATCH都不是功能PASS。
