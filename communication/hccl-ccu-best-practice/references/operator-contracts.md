# CCU 算子接入卡

来源范围和内容哈希见 [knowledge-baseline.json](knowledge-baseline.json)，由 `check_knowledge.py` 校验。本页当前覆盖 **AllGather / InsV2AllGatherSoleExecutor / CCU 分支**，不外推到Parallel、legacy或不同版本。其他五个算子按 [算子索引](operator-index.md) 读取自己的接入卡，不套本页；仍可复用MATCH的公共同步卡。

## AllGather Sole 输入与缓存

方法调用、默认行为、成本注册和建链容器语义见 [框架调用契约](framework-contracts.md)，不要从每个样本重新提炼接口清单。

定位：`HCCL/src/ops/all_gather/algorithm/executor/ins_v2_all_gather_sole_executor.cc` 的 `CalcRes`、`Orchestrate`、`OrchestrateLoop`、`FastLaunchSaveCtx`。

| 契约 | 已核对结论 |
|---|---|
| template 构造与资源 | `param, userRank, algHierarchyInfo.infos[0]`；CalcRes 直接调用 template::CalcRes；CCU kernel handles 来自 `resCtx.ccuKernels` |
| buffer | INPUT→OUTPUT，指针为用户基址；inBuffBaseOff/outBuffBaseOff 为本轮已经处理的字节，template 各加一次；不能再把 output rank 槽距离算进基址 |
| 数据单位 | `count` 是本轮每 rank 元素数；`sliceSize=tailSize=currDataCount*dtypeSize`；`outputSliceStride=dataSize_=整次每 rank bytes`；`inputSliceStride=0` |
| 输出落位 | 本端第 j 个 rank 片位于 `output_base + 本轮字节偏移 + j*outputSliceStride`，不能用本轮 sliceSize 替代整次 stride |
| repeat | repeatNum=1，input/outputRepeatStride=0；executor 外层 loop 与 template repeat 是不同覆盖点 |
| 外层 loop | 无 scratch 时上界为 `UB_MAX_DATA_SIZE`（当前 alg_param.h 为 256 MiB/每 rank），loopTimes=ceil(dataCount/maxCountPerLoop)；有 scratch 时还受预算约束；对称内存分支会改为一次 loop |
| FastLaunch | `loopTimes==1 && engine==CCU && opMode!=OFFLOAD` 才保存；参数和地址更新仍按本次 ABI/key 证明 |
| 单rank断言 | AllGatherOutPlaceCommon 的 SingleRankProc 在Selector前返回；首次 N=1 无正常选路日志，不能以 --expect-algo 声称kernel命中；见验证参考的逐用例可观测性要求 |
| 零数据 | out-place `HcclAllGather` 与 graph-mode 入口对 sendCount=0 提前返回；即使假设到 executor，普通非对称分支 loopTimes=0 不调用 KernelRun。需要直接 kernel 测试才能证明其零片防御分支被执行 |
| channels | CCU 不走 AICPU 的 RestoreChannelMap/SetchannelsPerRank 分支；从 CcuKernelInfo.channels 传递到注册回调，顺序由 template 与 kernel 约定 |

## 选路接线

`alg_parse.{h,cc}` 负责算法枚举、大小写映射、DSL 和 `ComposeAlgoName`；新增算法先查其存在性，再追加缺项。经当前源码验证的例子：`allgather:sole{ring}` 组成 `CcuSchedAllGatherSoleRing`，不是把注册全名直接写成 HCCL_ALGO。

`op_common/selector/cost_table.cc` 中 `needSoftCheck=!(tunerLoaded || hcclAlgoPriority>0)`：仅以 `opCustomCheck=false` 实现“只显式”时，要核对 tuner 状态，不能声称所有路径都禁止默认竞争。hard topology/rank/dtype 条件仍须正确注册。独立的 legacy selector 不会自动获得新增 DSL 路径；CCU 模式、成本入口和真实算法命中必须记录。不要根据 normal/UBX 文件名推断实际 topology，读运行日志中的分层/引擎。

这些是选路方法与风险，非所有任务必须采用的设计。默认选路是否允许变化仍由用户需求确定。

## 按需参考

- 需要Ring蓝本时再读 [可选历史样本](ring-example.md)，并校验其独立来源范围。
- 设计测试矩阵或执行验证时再读 [工具与内存约束](validation.md#工具与内存的已知约束)。
