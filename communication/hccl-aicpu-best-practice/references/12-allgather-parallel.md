# AllGather Parallel 接入契约

用于任何拟复用 `InsV2AllGatherParallelExecutor` 的 AICPU template，不限定 Ring/Mesh/NHR。
来源：HCCL `978d01b68c477572962b5be67c4ea9929ca9e2d7` 的
`src/ops/all_gather/algorithm/executor/ins_v2_all_gather_parallel_executor.{cc,h}` 与
`src/ops/op_common/selector/cost_model.h`，2026-09-24 核对。来源校验：

```bash
python3 "$SKILL_DIR/scripts/check_sources.py" --repo "$HCCL_REPO" \
  --hcomm-repo "$HCOMM_REPO" --op all_gather --contract all-gather-parallel
```

指纹对应已提交基准，包含本文指出的 `algName` 缺口；修复后的工作区会报告 CHANGED，
按差异核对并记录任务证据即可，不能为消除提示自动重写知识基线。
这里描述 AICPU 的 `Orchestrate → OrchestrateLoop` 路径；CCU 快速 launch 和其他 executor 不在范围内。
字段在 OFFLOAD/对称内存下仍有赋值不等于已验证这些模式的完整算法支持。

## 1. 四次调用的布局

真源：`GenTemplateAlgParamsIntra0/Inter0/Inter1/Intra1`。记：

- `D = dataSize_ = param.DataDes.count * dataTypeSize_`，为单 rank 完整贡献字节数。
- `N0/N1 = rankSizeLevel0_/rankSizeLevel1_`；`r0/r1` 为该层逻辑 rank 索引。
- `b0 = processedCount * dataTypeSize_`，`b1 = b0 + currCountPart0 * dataTypeSize_`。
- 第 0/1 份数据的 `count=currCountPart0/1`，`sliceSize=tailSize=count*dataTypeSize_`。

| 方法后缀 / template | 输入指针/类型 → 输出指针/类型 | in base / out base | RPT | ISS / OSS | IRS / ORS |
|---|---|---|---|---|---|
| Intra0 / template0 | user input / INPUT → user output / OUTPUT | b0 / r1*N0*D+b0 | 1 | 0 / D | 0 / 0 |
| Inter0 / template1 | user output / OUTPUT → user output / OUTPUT | b0 / b0 | N0 | N0*D / N0*D | D / D |
| Inter1 / template1 | user input / INPUT → user output / OUTPUT | b1 / r0*D+b1 | 1 | 0 / N0*D | 0 / 0 |
| Intra1 / template0 | user output / OUTPUT → user output / OUTPUT | b1 / b1 | N1 | D / D | N0*D / N0*D |

`inputSize/outputSize` 字段仍分别取 `param.inputSize/outputSize`，不能据此把中间输入指针误认成 user input。
四阶段 scratch 均为 `resCtx.cclMem`、HCCL_BUFFER，Intra 使用 `intraScratchOffset`，Inter 使用
`interScratchOffset`；不要在模板中覆盖 executor 的 base offset。

调用顺序是 `(Intra0, Inter1) → executor 同步 → (Inter0, Intra1)`。
四个 stride 跨的是完整贡献 `D`，即使最后一轮 `S < D` 也不改为 `S`。
Inter0 和 Intra1 都必须处理 repeat。其 input/output 指针相同，需论证转发与覆盖顺序。

地址量化建议：选 `N0=2,N1=3,D=1024,b0=64,b1=192`：
Inter0 中层索引 `i=2,rpt=1` 的输出偏移为 `64+2*2048+1024=5184`；
Intra1 中 `i=1,rpt=2` 为 `192+1024+2*2048=5312`。
两者应落在总输出的 6 个完整 rank 块中。再加入尾片、非零 scratch base 和输入地址检查，
不能只用两个输出例子宣称布局正确。

## 2. 资源申请与消费

真源：`CalcRes`、`PrepareResForTemplate`、`Orchestrate`。

| 位置 | 实际契约 |
|---|---|
| CalcRes | 分别调用两个模板的 CalcRes；executor 主线程通知数为 2；从线程数为两模板 slaveThreadNum 之和再加 2 |
| 模板主线程通知 | executor 在每份模板的 notifyNumOnMainThread 之外再加 1，用于阶段同步 |
| channel | AICPU 路径要求每个模板 channels 外层非空，取各自 channels[0] 组成两层资源 |
| PrepareResForTemplate | 重新构造模板后读 GetRes；按 slaveThreadNum+1 切出两份线程，按 notifyNumOnMainThread 选 executor 专用通知索引 |
| KernelRun | 使用分给该模板的线程和 channel map；executor 的 Pre/Post 同步与模板内部同步分开 |

因此 GetRes 必须在新构造对象上给出与 CalcRes 一致的线程/通知预算，不能依赖此前对同一对象调用过 CalcRes。
本版本 Orchestrate 的 AICPU_TS 分支只对 **inter** 调用 SetchannelsPerRank，再读两个模板 GetRes；
intra 的线程数若依赖 channel 状态，不能假设已走同一初始化。相关返回码在此路径未传播，不能靠默认报错止住执行。
Sole 的初始化顺序、N=1 空 channel 与默认构造区别见[AllGather 接入卡](../../hccl-aicpu-best-practice/references/allgather-integration.md#3-执行期对象初始化)。
单线程模板可以没有内部 Pre/Post 同步，但不等于整个 Parallel executor 没有同步。
支持单 rank/空 channel 时，须额外检查此 executor 的拒绝分支，不能从 Sole 成功外推。

## 3. 显式启用与成本参数传递

基准版本 `CalcCostCoeff` 中有 **4** 处直接 `CalcCostCoeffParam{...}`，只传前 8 个字段，
第 9 个 `algName` 及后续 `comm/topoInfo` 均默认 nullptr。模板若用这些字段判断显式启用或拓扑，可能被过滤而回退到其他算法。
这是可编译的契约缺口。接入前对照实际字段定义，修复时同时核对 lambda 捕获和所有受影响消费者。

首次构建前运行本 Skill 的只读工具：

```bash
python3 "$SKILL_DIR/scripts/check_cost_forwarding.py" \
  --executor "$HCCL_REPO/src/ops/all_gather/algorithm/executor/ins_v2_all_gather_parallel_executor.cc" \
  --cost-header "$HCCL_REPO/src/ops/op_common/selector/cost_model.h" --expected-calls 4
```

旧命令默认检查 algName；需要通信域配置/拓扑时，按[参数矩阵](../../hccl-aicpu-best-practice/references/allgather-integration.md#4-成本参数传递矩阵)
显式列出 `--require-field algName=algName --require-field comm=comm --require-field topoInfo=topoInfo` 中本次依赖项。
不机械要求全部字段。PASS 仅表示初始化点使用指定表达式；FAIL 定位省略/空指针；
UNVERIFIED 表示写法或调用数量变化，定向读对应函数。不能通过降低 expected-calls 掩盖缺失调用。
注册/映射/DSL 的落点与组合复用[AllGather 接入卡](../../hccl-aicpu-best-practice/references/allgather-integration.md)，
选择器开关是否存在取决于目标源码，不能从历史命令照搬 `HCCL_USE_NEW_SELECTOR=1`。

## 4. 多 loop 的预算入口

真源：`OrchestrateLoop`、`GetParallelDataSplit`。令分流比例 `f0+f1=1`，
四次 scratch 查询结果分别为 `mIntra0/mInter0/mIntra1/mInter1`：

```text
M0 = max(ceil(f0*mIntra0), ceil(f1*mIntra1*N1))
M1 = max(ceil(f1*mInter0), ceil(f0*mInter1*N0))
M = M0 + M1
当 M > 0：B = min(floor(maxTmpMemSize_/A/M)*A, UB_MAX_DATA_SIZE)
当 M = 0：B = maxTmpMemSize_
maxCountPerLoop = min(B, UB_MAX_DATA_SIZE) / dataTypeSize_
intraScratchOffset = 0；interScratchOffset = M0*B
```

`A=HCCL_MIN_SLICE_ALIGN`，整数除法与源码保持一致。四次 scratch 查询在该版本均使用
`(INPUT, OUTPUT)`，即使实际后两阶段输入是 OUTPUT；模板若按 buffer 类型改变倍数，必须另核此预算。
`supportSymmetricMemory` 分支直接令 `maxCountPerLoop=dataCount_`，不适合用来证明多 loop。
普通分支还会对两个分流片在非末轮做 `AICPU_ALIGN_SIZE` 对齐；需确保进度非零且两条目标路径均被覆盖。

以上得到 executor count 阈值后，还须核对所用测试程序的 bytes→count 换算。
使用 [hccl-vm 的 loop-size 与日志断言](../../hccl-test-tool-hvm/references/test-plan.md#多-loop-参数推导)
一次生成边界大小；repeat 和多 loop 分别取证。小缓冲区只在目标源码允许的范围内选取。
