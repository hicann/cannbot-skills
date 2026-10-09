# 03 · 资源模型：CalcRes / GetRes / CalcScratchMultiple / thread & notify

## 1. 两个阶段，两次调用

```
① 资源准备期（每个 algTag 一次）    executor::CalcRes → template::CalcRes(comm, param, topoInfo, resourceRequest)
                                    产出：要几个 thread、几个 notify、几条 channel
                                    ↓ 框架据此真正建链、建流、建 notify
② 执行期（每轮 loop 一次）          executor::OrchestrateLoop → template::CalcScratchMultiple()  ← 先算 loop 次数
                                                              → template::KernelRun(...)         ← 再逐轮编排
```

`CalcRes` 拿到的是 `HcclChannelDesc`（描述），`KernelRun` 拿到的是 `ChannelInfo`（已建好的链路）。两者一一对应，顺序一致。

## 2. CalcScratchMultiple —— 最容易写错的一个数

```cpp
u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
```

**语义**：本 template 在一轮 loop 里，往 `hcclBuff` 写入的字节数是"单份数据量"的几倍。
executor 用它反推每轮能搬多少：

```
scratchBoundDataSize = hcclBuff.size / multiple  向下对齐到 HCCL_MIN_SLICE_ALIGN(128)
maxDataSizePerLoop   = min(transportBoundDataSize, scratchBoundDataSize)
loopTimes            = ceil(dataCount / (maxDataSizePerLoop / dataTypeSize))
```

**报小 → hcclBuff 越界；报大 → 多切 loop 白掉性能。**

存量取值参考：

| template | 返回值 | 理由 |
|---|---|---|
| `InsTempAllReduceMesh1DOneShot` | `templateRankSize_` | 每个 rank 把整份数据写进我 scratch 的独立一段 |
| `InsTempAllReduceMesh1DTwoShot` | `2` | 理论 1 份够，但非均衡切分下会不足，取 2 留余量 |
| `InsTempAllReduceNHR` | `1` | 原地 ReduceScatter + AllGather，scratch 只存一份 |
| `InsTempReduceScatterMesh1D` | `templateRankSize_` | 同 one-shot |
| `InsTempAllGatherMesh1D` | `OPBASE ? templateRankSize_ : 0` | 图模式直连对端 output，不过 scratch |
| 完全不用 scratch | `0` | 只受传输上限约束 |

不能凭算法形态（如 Ring）推断 `inBuffType` / `outBuffType` 可直接 `(void)` 掉。仅在已核对绑定的 executor 是单段、`outBuffType` 恒为 `OUTPUT`，且两种 buffer 类型不改变本次 scratch 预算时，才可在 `CalcScratchMultiple` 中忽略它们。一个 template 被多个 executor 注册，或经 `REGISTER_EXECUTOR_BY_TWO_TEMPS` / 多级 executor 复用时，先逐段读 executor 对 `buffInfo` 的赋值；输出落位见 [五参数说明 §1.1](09-template-data-params.md#11-输出落位先看-executor)。

`KernelRun` 中的 scratch→out 回搬也须按实际输出落位分流。`outBuffType == HCCL_BUFFER` 不能直接推出“跳过拷贝”：只有证明数据已在目标槽位，才可跳过；若源和目标错位，仍需搬移并说明地址依据。

### 2.1 与 `repeatNum` 的乘法关系 —— 不要自己乘

`CalcScratchMultiple()` **永远按"单次 repeat"报**，哪怕你明知自己会被分层 executor 重复调用。
放大是 executor 的活：它把两层 template 的倍数合成一个总预算，再据此算 loop 次数。

```cpp
// src/ops/all_gather/algorithm/executor/ins_v2_all_gather_sequence_executor.cc:204
u32 intraMultiplier = intraTempAlg.CalcScratchMultiple(BufferType::OUTPUT, BufferType::OUTPUT);
u32 interMultiplier = interTempAlg.CalcScratchMultiple(BufferType::INPUT,  BufferType::OUTPUT);
u32 templateScratchMultiplier = std::max(interMultiplier, intraMultiplier * rankSizeLevel1_);
//                                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                                        intra 会被 repeat rankSizeLevel1_ 次，占用同比放大
u64 maxCountPerLoop = hcclBuff.size / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN * ... ;
```

ReduceScatter 那边是同一个式子的另一半写法（`ins_v2_reduce_scatter_sequence_executor.cc:195`）：

```cpp
templateScratchMultiplier = std::max(templateScratchMultiplierInter * rankSizeLevel1_, templateScratchMultiplierIntra);
// 并把倍数编进 stride 交回 template（:248）
tempAlgParamsInter.inputRepeatStride = templateScratchMultiplierInter * dataCount_ * dataTypeSize_;
```

两种写法都假设 template 报的是**单次**用量。

| 你写的 template | `CalcScratchMultiple` 该返回 | 实际 hcclBuff 占用 |
|---|---|---|
| 单层用，one-shot | `templateRankSize_` | 同左 |
| 会被 2 级 executor 当 intra 用，one-shot | 仍是 `templateRankSize_` | executor 乘上 `rankSizeLevel1_` |

**自己乘 `repeatNum` 的后果**：倍数双重放大 → `maxCountPerLoop` 减半 → loop 次数翻倍，
带宽白掉一半，而且编译和 Checker 都不会报错（只是变慢），只能靠 review 抓。

反过来，`KernelRun` 里往 scratch 落地址时**必须**乘 `rpt`——那是地址，不是预算。
跨度须按各 buffer 的实际布局推导，见 `09-template-data-params.md` §7；`S*N` 与输入 stride 只是特定蓝本的取法。

## 3. thread / notify 模型

**约定**：`threads[0]` 是主流，`threads[1..n-1]` 是从流。
下面的常见 Mesh 方案把本地 `LocalCopy`/`LocalReduce` 挂主流、跨 rank 的 send/recv 挂从流。
自定义算法按实际数据依赖选择线程分配，不要求从线程与远端一一对应，也不预设线程总数等于 rankSize。

一次完整的主从协作：

```
主流: ─── LocalCopy ──┬─ record(notifyIdxMainToSub) ──────────────┬─ wait(notifyIdxSubToMain) ─ LocalReduce ─
                      │  PreSyncInterThreads                      │  PostSyncInterThreads
从流i:                └─ wait ─ SendRecvBatchWrite ─ record ──────┘
```

### CalcRes 里怎么填

Mesh 类（thread 数 = rankSize）：

```cpp
u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
resourceRequest.slaveThreadNum        = threadNum - 1;
resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1); // 每个从流 1 个 notify
resourceRequest.notifyNumOnMainThread = threadNum - 1;                        // 主流为每个从流各留 1 个
```

有 executor 直接消费 `GetRes` 时（不限多 jetty），应把线程/通知计算抽到 `GetRes()`，供 `CalcRes` 与执行期复用。多 jetty / 多 channel 类（thread 数 = `channelsPerRank_`）例如：
并在 `CalcRes` 里先 `channelsPerRank_ = CalcChannelsPerRank(channels);` 再 `GetRes(resourceRequest);`：

```cpp
HcclResult XXX::GetRes(AlgResourceRequest& resourceRequest) const {
    u32 threadNum = channelsPerRank_;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 i = 0; i < threadNum - 1; i++) { resourceRequest.notifyNumPerThread.push_back(1); }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}
u64 XXX::GetThreadNum() const { return channelsPerRank_; }
```

### 按消费者核对，而非按算法形态猜接口

当前 `src/ops/all_gather/algorithm/executor/ins_v2_all_gather_parallel_executor.cc` 的 `PrepareResForTemplate` 在执行期直接调用 intra/inter 模板的 `GetRes`，按 `slaveThreadNum + 1` 切线程，并读取 `notifyNumOnMainThread`。因此即使 Ring 固定使用两条线程，也必须实现有效 `GetRes`，不能仅在 `CalcRes` 填资源后继承基类空实现。

- `CalcRes` 负责通道请求等申请；其线程/notify 结果与同一模板状态下的 `GetRes` 必须一致，优先共享同一计算实现。
- `GetRes` 必须能够在执行期构造的模板对象上工作；检查 rank/channel 状态是否已初始化，不能依赖只发生在另一申请期对象上的副作用。
- `GetThreadNum` 按所有实际调用点和基类默认值判断是否覆写：executor 或模板用它裁剪/计数，且默认值与真实线程数不符时必须实现。该 Parallel 函数使用 `GetRes`，不能将补 `GetThreadNum` 当成修复 `GetRes` 的替代品。
- 模板接收的线程集合可能大于实际使用数量时，按已核验的资源契约截取，不直接把 `threads.size()` 当作算法线程数。

### GetNotifyIdx* 的含义

- `GetNotifyIdxMainToSub(v)`：主流向第 i 个从流 record 时，用**从流自己的第 `v[i]` 个 notify**。
  从流每人只有 1 个 notify（`notifyNumPerThread = 1`），所以恒为全 0：
  ```cpp
  notifyIdxMainToSub.assign(threadNum_ - 1, 0);
  ```
- `GetNotifyIdxSubToMain(v)`：第 i 个从流向主流 record 时，用**主流的第 `v[i]` 个 notify**。
  主流有 `threadNum-1` 个 notify，每个从流占一个，所以是 0,1,2,...：
  ```cpp
  for (u32 i = 0; i < threadNum_ - 1; ++i) { notifyIdxSubToMain.push_back(i); }
  ```

**自洽性检查**（漏了会挂死或踩别人的 notify）：
`notifyIdxMainToSub.size() == notifyIdxSubToMain.size() == slaveThreadNum`
`max(notifyIdxSubToMain) < notifyNumOnMainThread`
`max(notifyIdxMainToSub) < notifyNumPerThread[i]`

## 4. CalcRes 的完整写法

最简 Mesh：

```cpp
HcclResult XXX::CalcRes(HcclComm comm, const OpParam& param,
                        const TopoInfoWithNetLayerDetails* topoInfo,
                        AlgResourceRequest& resourceRequest)
{
    // ① thread / notify
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    // ② channel（每一层 push 一组）
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}
```

NHR + 拓扑分支 + 多 jetty（照抄 `ins_temp_all_reduce_nhr.cc`）：

```cpp
std::vector<HcclChannelDesc> level1Channels;
if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
    std::vector<HcclChannelDesc> descs;
    CHK_RET(CalcChannelRequestNhrMultiJetty(comm, param, topoInfo, subCommRanks_, descs));
    for (auto& ch : descs) {                       // 只保留 UB_CTP 协议的链路
        if (ch.channelProtocol == COMM_PROTOCOL_UB_CTP) { level1Channels.push_back(ch); }
    }
} else {
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level1Channels));
}
resourceRequest.channels.push_back(level1Channels);
channelsPerRank_ = CalcChannelsPerRank(level1Channels);
CHK_PRT_RET(channelsPerRank_ > MAX_JETTY_NUM, HCCL_ERROR("..."), HCCL_E_INTERNAL);
GetRes(resourceRequest);
```

`topoInfo`（`TopoInfoWithNetLayerDetails`）常用字段：`userRank`、`userRankSize`、
`level0Topo`（`Level0Shape::MESH_1D` / `MESH_1D_CLOS` / …）、`level0PcieMix`。

## 5. CalcCostCoeff —— 参与算法选路

**static，不是 virtual**；executor 通过模板参数 `InsAlgTemplate::CalcCostCoeff(...)` 静态调用。
不实现就继承基类的空实现，返回 `{}`。当前新 selector 会在 HCCL_ALGO 过滤前排除空成本候选，因此显式选中也不能绕过这一条件。候选启用与性能标定须分开声明；默认行为不变的接入约束见[HCCL_ALGO 显式选路](executor-selector.md#4-hccl_algo-显式选路保留默认选择)。

**入参结构体以 `src/ops/op_common/selector/cost_model.h` 的 `CalcCostCoeffParam` 为准**——
它没有 `n` / `needLocalCopy` 这些字段（数据量走 `dataRatio`，是否要 LocalCopy 由 buffer 类型推断），
`netType` 是 `CommTopo` 不是 `AlgNetType`。返回的 `CostModelParam` 有 **A/B/C/D 四个**成员。

```cpp
std::vector<CostModelParam> XXX::CalcCostCoeff(CalcCostCoeffParam param)
{
    if (param.rankSize > 8) { return {}; }   // 不适用就返回空，等于弃权
    int portNum = (param.portNum.size() == 1) ? param.portNum[0] : (param.portNum[0] + param.portNum[1]);
    int kernelNum = 1;                       // 延迟项按 kernel 数
    int taskNum = CostModelManager::CalcTransTaskNum(param.rankSize)
                  + CostModelManager::CalcSyncTaskNum(param.rankSize) * 2;   // 下发项按 task 数
    float A = 0.0f, B = 0.0f, C = 0.0f, D = 0.0f;   // A=传输 B=本地计算 C=延迟 D=下发
    float n = param.dataRatio * param.rankSize;     // executor 统一按 two-shot 的单片大小传入，
                                                    // one-shot 需在此乘回 rankSize
    CostModelManager::Global()->CalcMeshParam(n, param.netType, portNum, param.rankSize, A, param.isPod);
    // NHR 用 CalcNHRParams(n, param.netType, portNum, param.rankSize, A, param.isPod)
    float B1 = 0.0f, B2 = 0.0f;
    if (param.inputBuffer != param.scratchBuffer) {   // 输入不在 scratch 上才有一次 LocalCopy
        CostModelManager::Global()->CalcLocalCopyParams(n, EngineType::AICPU, B1);
    }
    CostModelManager::Global()->CalcLocalReduceParams(n, EngineType::AICPU, B2);
    B = B1 + (param.rankSize - 1) * B2;
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::AICPU, C);
    CostModelManager::Global()->CalcLaunchParams(taskNum, EngineType::AICPU, D);
    return { {A, B, C, D} };   // 四个都要填，漏一个会被 -Werror=missing-field-initializers 拦下
}
```

若算法有 rankSize 限制，在开头直接 `if (param.rankSize > 8) { return {}; }` —— 返回空表示"我不适用"。

### TemplateProp / AlgoType

在头文件类体开头声明算法属性：

```cpp
static constexpr TemplateProp props = {.algoType = AlgoType::NHR};
```

- `TemplateProp` **当前只有 `algoType` 一个成员**（`src/ops/op_common/algorithm/template/common_alg_template_base.h`）。
  历史版本的 `isNhr` 已被删除，再这么写会**编译失败**（designated initializer 指向不存在的成员）。
- 枚举值定义在 `src/common/alg_parse.h`，别写死在脑子里，用时现查：
  `MESH / MESH_2DIE / MESH_ONESHOT / MESH_TWOSHOT / MESH_CHUNK / MESH_CHUNK_TWOSHOT / MESH_CONCUR /
  MESH_CONCURRENT / MESH_MULTILINK / MESH_SINGLE_CHANNEL / NHR / NHR_MULTILINK / NHR_AICPU_REDUCE / UNKNOWN`。
  同文件里还有 `MESH_ALGO_TYPES` / `NHR_ALGO_TYPES` 两个集合，按家族分类时用它们。
- **不声明不会编译失败**：基类给了 `props = {}`（`algoType = UNKNOWN`）。而且当前仓内 `props`
  **还没有任何消费者**——`grep -rn props src/` 只能搜到声明。它是给 cost model 预留的属性。
- 实践口径：实现了 `CalcCostCoeff` 的 template 基本都声明（仓内 19 个声明者里 18 个如此），
  纯被 executor 静态绑定、不参与选路的可以不写。`check_template.py` 就是按这个口径给 WARN 的。
- 实时核对谁声明了什么：`python3 "$SKILL_DIR/scripts/list_templates.py" --repo "$HCCL_REPO"`，
  尾部方括号里就是各 template 的 `algoType`。
