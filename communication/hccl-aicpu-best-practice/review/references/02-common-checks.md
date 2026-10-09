# 02 · 机械层（W01–W07）与通用语义检查（C01–C16）

与拓扑无关，四族都要过。括号里是脚本的默认严重度。
**机械层排在最前**：一个没接进 CMake、或者缺纯虚实现的文件，语义评得再细也没有意义。

---

## 0. 机械层 W01–W07

### W01 两处 CMake 登记（BLOCKER）

AICPU template 同时被编进 host 侧 `libhccl.so` 和 device 侧 `libscatter_aicpu_kernel.so`：

1. `src/ops/{算子}/algorithm/template/aicpu/CMakeLists.txt` 的 `set(src_list ...)`（漏了：**根本不编译**）；
2. `src/scatter_aicpu_kernel.cmake` 的 `add_library(scatter_aicpu_kernel SHARED ...)` 块
   （漏了：**host 编得过，运行时 AICPU 侧找不到符号**）。

★ 第 2 处是本仓最高频的接线 bug。带版本守卫的（`if(NOT HCCL_CANN_COMPAT_850)` / `list(APPEND src_list ...)`）
也算登记，脚本按文件名子串匹配，不区分在哪个块里——但**评审时要顺手看一眼守卫条件对不对**。

### W02 必须实现的方法（BLOCKER）

`Describe` / `GetNotifyIdxMainToSub` / `GetNotifyIdxSubToMain`（纯虚，不实现编译不过）、
`CalcRes` / `KernelRun` / `CalcScratchMultiple`（基类默认实现直接 `HCCL_ERROR` 返回失败，
编得过但一跑就错）。变体子类（`class A : public InsTempXxx`）由父类提供，脚本会跳过。

### W03 props / AlgoType（BLOCKER）

`TemplateProp` **只有 `algoType` 一个成员**，写法固定：

```cpp
static constexpr TemplateProp props = {.algoType = AlgoType::MESH_ONESHOT};
```

- 用了历史上的 `isNhr` → 指向不存在的成员，**编译失败**；
- `AlgoType::X` 不在 `src/common/alg_parse.h` 的枚举里 → 编译失败（脚本实时解析该头文件，不写死清单）；
- 类名像 NHR 却声明 `MESH_*`（或反过来）→ MINOR，通常是复制粘贴漏改。

### W04 命名 / guard / 版权 / namespace（BLOCKER|MAJOR|MINOR）

`.h` 缺 include guard 是 BLOCKER；缺 CANN Open Software License 版权头、缺 `namespace ops_hccl` 是 MAJOR；
文件名 / 类名前缀、guard 与文件名不一致是 MINOR（仓内本就不统一，跟同目录邻居保持一致即可）。

### W05 构造函数（MAJOR|MINOR）

executor 硬编码 `make_shared<T>(param, rankId, subCommRanks)`，
签名必须是 `(const OpParam&, const u32, const std::vector<std::vector<u32>>&)`，对不上是 MAJOR。
走 FastLaunch 的 executor 还要 `T() = default`；DPU / Intra / Inter / OmniPipe 这类不走
FastLaunch 的可以没有，所以缺默认构造只报 MINOR。

### W06 / W07 注册（MINOR|MAJOR|BLOCKER）

- **W06**：没有任何 `REGISTER_EXEC_V2(...)` 引用这个类 → 它不会被实例化，等于没写。
  新代码可能只是 executor 还没接，故 MINOR，但评审要问一句"谁来用它"。
- **W07**：DPU 侧展开的 template 必须在 `.cc` 末尾写 `REGISTER_TEMPLATE_V2("类名", 类名);`
  （`dpu/kernel_launch.cc` 按字符串反查），缺了是 MAJOR；非 DPU 写了通常没必要（MINOR）；
  用 V1 的 `REGISTER_TEMPLATE(` 是 BLOCKER。

---

## 1. scratch 倍数：C01（MAJOR）/ C02（MAJOR|MINOR）

`CalcScratchMultiple(inBuffType, outBuffType)` 的语义：**这一轮 loop 往 `hcclBuff` 写多少倍单份数据量**。
executor 拿它反推每轮能搬多少 → 直接决定 loop 次数。

**C01：里面出现 `repeatNum` 就报 MAJOR。**（后果是确定性的性能退化而非算错，
按 SKILL.md §3 的定义够不上 BLOCKER，但必须要作者答复。） 倍数按**单次 repeat** 报，放大是 executor 的活：

```cpp
// ins_v2_all_gather_sequence_executor.cc
templateScratchMultiplier = std::max(interMultiplier, intraMultiplier * rankSizeLevel1_);
```

自己再乘一遍 → `maxCountPerLoop` 减半 → loop 次数翻倍，带宽白掉一半。
**编译过、ST 也过，只是变慢**——这就是为什么它必须靠 review 抓。

**C02：** 没有实现（基类默认实现直接返回失败）；或变体子类继承了父类的倍数——
后者只在"变体没改写入量"时成立，改了收发模式 / 加了暂存段就不成立了，脚本按 MINOR 提醒人工确认。

各族的期望值见 `references/03..06`。判断口径：
**AllReduce 这类 in/out 等大的算子有唯一正解（MAJOR）；AllGather / ReduceScatter / Scatter
这类 in/out 不等大的，N 倍往往是对的（MINOR，人工确认）。**

## 2. 两级地址与输出落位：C03 / C04 / C16（MINOR）

```
外层 repeat rpt ∈ 0..RPT-1     一个 rank 有 RPT 块互不相邻的数据，块间距 IRS / ORS
内层 alg rank r ∈ 0..N-1       一块之内第 r 个 rank 的数据在 ISS*r / OSS*r
```

`repeatNum` / 五个 stride 全由 executor 填、template 只读；单层算法恒为 `1 / 0 / 0 / S / S`。

- **C03**：没有以 `repeatNum` 为上界的 `for` / `while` 循环 → 这个 template 一旦被 2/3 级 executor 当构件复用，
  只会处理第 0 块。存量 52 个里 25 个如此（只服务单层，不算错），所以是 MINOR；
  **新写的默认要写上**（`RPT == 1` 时零代价退化）。
- **C04**：写了 repeat 循环又落 scratch 地址，却没出现 `inBuffType`。
  scratch 跨度默认自己算 `S*N`；只有 `inBuffType == HCCL_BUFFER`（intra 接在 inter 之后、
  输入已在 CCL buffer）时才改用 `inputRepeatStride` / `inputSliceStride`。
  蓝本：`ins_temp_all_gather_mesh_1D.cc:152-170`。
- **C16**：常规 AICPU template 的 `LocalCopy` 明确从 scratch `DataSlice` 搬到 output `DataSlice`，
  同时引用两侧 base offset，却在该方法中没有读取 `buffInfo.outBuffType` → MINOR，提示逐段核对输出落位。`C05` 已用于空数据早退，
  因而沿用稳定编号新增 `C16`。DPU/OmniPipe 的 step 布局不适用这一文本判据。
  脚本只能发现遗漏；即使有分流，仍需人工核对在位/错位的地址依据和分支旁注释。

## 3. 同步配对：C07（MAJOR|MINOR）

```
主流: ─ LocalCopy ─┬─ record ─────────────────┬─ wait ─ LocalReduce ─
                   │ PreSyncInterThreads      │ PostSyncInterThreads
从流i:             └─ wait ─ SendRecvBatch* ─ record ┘
```

一边是 0、另一边非 0 → **MAJOR**（主流会在从流写完前去读 scratch，读到半成品，现象是偶发结果错）。
两边都非 0 但次数不等 → MINOR，人工核对每一处：存量里有故意不对称的用法
（`ins_temp_all_gather_nhr.cc:325` 用 `PreSyncInterThreads` 做 post-copy 线程之间的单向握手）。

## 4. 资源三字段：C08（MAJOR）

`CalcRes`（或 `GetRes`）里这三个要一起填，缺一个就可能 notify 索引越界或死等：

```cpp
resourceRequest.slaveThreadNum        = threadNum - 1;
resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
resourceRequest.notifyNumOnMainThread = threadNum - 1;
```

`slaveThreadNum = 0`（单线程实现）时不要求填后两个，脚本会跳过。
另外要人工核一遍：**`GetNotifyIdxMainToSub` / `GetNotifyIdxSubToMain` 吐出的下标个数
要和这里申请的数量对得上**——这一条脚本判不了。

## 5. 边界早退：C05 / C06（MINOR）

```cpp
if (count_ == 0) { HCCL_WARNING("..."); return HCCL_SUCCESS; }
if (templateRankSize_ == 1 || subCommRanks_[0].size() == 1) { /* 只做 LocalCopy */ return HCCL_SUCCESS; }
```

存量里大多数没写（靠 executor 保证），所以是 MINOR；**新写的要补**。

## 6. 64-bit 与 PROD：C09（MINOR）/ C10（MAJOR）

硬件归约不支持 INT64 / UINT64 / FP64 / `HCCL_REDUCE_PROD`。仓内**两条兜底路线**：

- **a）template 内部**：`needAicpuReduce_ = 64-bit || PROD`，跑完搬运再软归约（见 mesh one-shot）；
- **b）选路层**换到专门的 `*_aicpu_reduce_*` template（`ins_temp_all_reduce_aicpu_reduce_nhr.cc` 这一支）。

C09 只在两条都看不到时报，**是个提问而不是判决**：请作者说清走哪条。两条都没有 = INT64/PROD 直接算错。

C10 管顺序，**恒为**：

```cpp
HcommBatchModeEnd(param.algTag);      // 结束批下发
HcommBatchModeStart(param.algTag);    // 重开一批
HcommThreadJoin(t, CUSTOM_TIMEOUT);   // 等所有流真正落地
```

顺序错 = 在从流还没落盘时就去读 scratch。

## 7. channel 取用：C11（MINOR）/ C15（MINOR）

```cpp
CHK_PRT_RET(channels.count(peerRank) == 0,
            HCCL_ERROR("[XXX] remoteRank[%u] is not in channels.", peerRank), HCCL_E_INTERNAL);
const ChannelInfo& ch = channels.at(peerRank).at(channelIdx);
```

- **C11**：`.at()` 前没有 `.count()`。AICPU 上抛异常就是直接崩。全仓 29 处欠账，故 MINOR。
- **C15**：`channels` 的 key 是**通信域 userRank**（`subCommRanks_[0][idx]` / `rankList_.at(idx)`），
  **不是算法内序号 0..N-1**。脚本只能看出"这个 key 不像是从 `subCommRanks_` / `rankList_` 取的"，
  必须人工确认——这是本仓高频 bug，且现象是"随机 rank 数下才错"。

## 8. 返回值：C12（MINOR）

`SendRecvBatchWrite* / SendRecvBatchRead* / LocalCopy / LocalReduce / Pre|PostSyncInterThreads /
HcommThreadJoin / HcommBatchMode*` 都返回状态，必须 `CHK_RET` 或 `CHK_PRT_RET` 接住。
不接 = 失败被静默吞掉，现象是结果错但日志干净。

## 9. Describe()：C13（MINOR）

一行自述，出现在日志与 DFX 里，**必须带 `templateRankSize_`**，否则排障时分不清是哪一层的实例。

## 10. 读写模式：C14（MINOR）

跨 rank 边只有一个方向：**写模式底层只用 `txSlicesList_`，读模式只用 `rxSlicesList_`**。
两套原语同时出现却没有 `IsPcieProtocol(channels)` / `isDmaRead_` 分支，
说明要么抄漏了、要么填了不生效的参数。

## 11. 人工才能看的（脚本不报，评审必看）

- 切片偏移：`CalcSlice` 的累加是否覆盖 `dataSize * N`，尾片归谁，`RoundUp` 用的是向上还是向下取整；
- 错峰顺序：`(myRank_ + queIdx) % N` 是不是真的让每个 rank 打不同的目标；
- `GetNotifyIdx*` 的下标序列与 `CalcRes` 申请量是否一一对应；
- `CalcCostCoeff` 的适用区间（`return {}` 的条件）与 template 实际能力是否一致；
- 日志里的 rank 到底是 userRank 还是算法内序号（写错会把排障带沟里）。


## 11. 支持性状态与模式完整性（人工必查，相关改动时）

- 对新增“不支持”条件，记录字段初始化/赋值 → Selector 消费 → Executor/Template 消费的位置。
  核对拦截时字段已经有效，且所有可达入口都被约束；不以 attrs 声明代替控制流证据。
- 从目标仓读取完整 OpMode 等枚举，逐态核对可达性、scratch/remote buffer 和执行分支。
  不可达须有调用链依据；不把未测试的模式当作不存在。
- 全域协议判定须证明当前拓扑同质；放宽拓扑时重新核对按链路读写选择。
- 守卫错误路径检查资源清理与通信下发时点；零数据/单 rank 早退按语义核对。
- 证据表至少记录条件、赋值位置、选路时有效性、执行消费/拒绝位置、测试或未覆盖范围。
  这些性质当前不由正则脚本证明；人工检查未完成应报告未完成。

来源：用户提供的 2026-09-22 Broadcast Ring 报告（对称内存时序与 OpMode 三态问题）。
该案例是检查方法的依据，其他算子的字段和调用顺序以其目标版本源码为准。
