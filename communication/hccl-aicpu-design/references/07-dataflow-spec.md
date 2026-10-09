# 07 · 数据流规格（dataflow spec）—— 本 skill 的输入契约

写一个 template = 先把**数据怎么流动**说清楚，再机械翻译成 C++。
模式 C 的输入产物是一份 dataflow spec markdown；模板见根目录 `dataflow-spec-template.md`（自带填写说明），
填好的范例见 `assets/dataflow-spec.example.md`（逆向自 `ins_temp_all_reduce_mesh_1D_one_shot.cc`，可与源码逐行对照）。

**一个 spec 文件对应一个 template**（一对 `.h`/`.cc`）。分层算法（Broadcast = Scatter_intra + AllGather_inter）
写多份 spec，各自在头部的「所属层级 / 兄弟 template」字段里交代组合关系。

---

## 1. 地址记法

一条数据流边 = `<源> -> <目标>`。源和目标都用下面五个句柄之一：

| 记法 | 指向 | 翻译成 C++ |
|---|---|---|
| `IN[off, len]` | 本 rank 的 userIn | `DataSlice(buffInfo.inputPtr, buffInfo.inBuffBaseOff + off, len, cnt)` |
| `OUT[off, len]` | 本 rank 的 userOut | `DataSlice(buffInfo.outputPtr, buffInfo.outBuffBaseOff + off, len, cnt)` |
| `SCR[off, len]` | 本 rank 的 hcclBuff（scratch） | `DataSlice(buffInfo.hcclBuff.addr, buffInfo.hcclBuffBaseOff + off, len, cnt)` |
| `SCR@p[off, len]` | **rank p 的** hcclBuff | `DataSlice(channels.at(p)[ch].remoteCclMem.addr, buffInfo.hcclBuffBaseOff + off, len, cnt)` |
| `OUT@p` / `IN@p` | 图模式下对端的 user buffer | `channels.at(p)[ch].remoteOutputGraphMode` / `remoteInputGraphMode` |

实际地址 = `addr_ + offset_`。`len` 是字节数；带归约的原语底层用 `count_`，所以 `cnt = len / dataTypeSize`。

### 预定义符号

| 符号 | 含义 | C++ |
|---|---|---|
| `S` | 本轮 loop 的字节数 | `tempAlgParams.sliceSize` |
| `C` | 本轮 loop 的元素个数 | `tempAlgParams.count` |
| `N` | 本 template 视角的 rank 总数 | `templateRankSize_` |
| `R` | 我的 rank | `myRank_` |
| `p` | 对端 rank（通信域 userRank） | `subCommRanks_[0][idx]` |
| `NCH` | 每个远端的 channel 数 | `channelsPerRank_` |
| `DT` | 单元素字节数 | `DATATYPE_SIZE_TABLE[dataType_]` |
| `RPT` | 本次 template 调用内的逻辑组数 | `tempAlgParams.repeatNum` |
| `IRS` / `ORS` | 相邻 repeat 块的间距（in / out） | `tempAlgParams.inputRepeatStride` / `outputRepeatStride` |
| `ISS` / `OSS` | 同一 repeat 内相邻 alg rank 块的间距（in / out） | `tempAlgParams.inputSliceStride` / `outputSliceStride` |

自定义符号在 spec 的「切片」章节里显式定义（如 `SS = (C / N) * DT`），之后可直接引用。

> **`p` 是通信域 userRank，不是算法内序号。** `channels` 的 key 就是它。
> 算法内序号（`0..N-1`）另用 `i` / `r` / `algRank` 表示，两者混用是本仓高频 bug。

五参数由 executor 描述本阶段布局，template 只读。以目标版本绑定 executor 的实际字段赋值为准，不得按“单层/多层”猜取值；下面的公式用于独立量化核对。

### 1.5 repeat 与 stride：两个寻址维度

规则切片中，`rpt` 是逻辑组索引，`i` 是算法 rank 索引（不是 userRank 或通信 step）：

```text
IN  相对 buffInfo.inputPtr 的偏移  = inBuffBaseOff  + rpt*IRS + i*ISS
OUT 相对 buffInfo.outputPtr 的偏移 = outBuffBaseOff + rpt*ORS + i*OSS
```

上述公式包含 base offset；写成 `IN[off,len]` / `OUT[off,len]` 时只填后两项，避免基址偏移重复相加。
四个 stride 均为字节，切片长度 `S` 不一定等于完整 rank 块大小 `D`。
例如普通 AllGather 可以是 `ISS=0, OSS=D`；Parallel 的某阶段可为 `ISS=OSS=D*N0, IRS=ORS=D`。
不能要求输入输出 stride 相等，也不能要求 RepeatStride 大于 SliceStride。

1. 起草 spec 第 5 章时记录目标 executor 的函数、分支和五参数赋值；输入可以是上一阶段的 OUTPUT。
2. 分别推导 input/output/scratch 布局及 rank 映射。scratch 的 `S*N` 仅适用于相应紧凑布局，
   是否复用输入 stride 必须对照实际 buffer 别名与生产者布局。scratch 预算按目标 executor 的消费方式单独推导。
3. repeat 循环处理本次调用的所有逻辑组；不要替换成 Ring step 循环。`repeatNum=0` 不能默认为一次。
   特殊 template 按实际消费契约处理；仅支持单次处理时在 spec 写明依据。
4. 第 5 章加入 `layout-check` JSON 量化样例，并运行 `check_layout.py`。
   期望偏移从物理布局独立推导，实际公式从拟实现/已实现代码摘取；不能用同一公式生成两边。
   Ring 必测非零 `ISS` 与非零算法 rank，使用不同 rank/repeat 内容，避免 rank 0 和单层测试掩盖漏项。

`check_spec.py` 检查字段与量化样例是否存在；`check_layout.py` 校验有限样例的算术与范围。
它们不验证完整 C++ 控制流、跨 rank 数据归属、通信同步或归约语义，最终仍须完整构建和双轨 Checker 验收。

---

## 2. 动作与原语

### 2.1 跨 rank 传输：**每条边只写一个方向**

这是 wrapper 的实际语义，不是简写约定：

- **写模式** `SendRecv*Write*`：底层 `DoSendRecvBatchTx` 只用 `txSlicesList_` 搬运，
  `rxSlicesList_` 完全不参与——rx 侧只贡献一个 channel 做 notify 握手。
- **读模式** `SendRecv*Read*`：底层 `DoSendRecvBatchRx` 只用 `rxSlicesList_`，tx 侧只做握手。

所以 spec 里**写模式只描述"我推给谁"，读模式只描述"我从谁拉"**，对称的另一半由框架保证。
写第二行反而会诱导填入不生效的参数。

| spec 写法 | 原语 | 约束 |
|---|---|---|
| `SendRecvBatchWrite   A -> SCR@p[...]` | `SendRecvBatchWrite` | 目标**必须**带 `@p` |
| `SendRecvBatchWriteReduce A reduce-> SCR@p[...]` | `SendRecvBatchWriteReduce` | 目标必须带 `@p`；对端做归约 |
| `SendRecvBatchRead    SCR@p[...] -> B` | `SendRecvBatchRead` | 源**必须**带 `@p` |
| `SendRecvBatchReadReduce SCR@p[...] reduce-> B` | `SendRecvBatchReadReduce` | 源必须带 `@p`；本地做归约 |
| `SendBatchWrite  A -> SCR@p[...]` | `SendBatchWrite` | 单向发 |
| `RecvBatchRead   SCR@p[...] -> B` | `RecvBatchRead` | 单向收 |

传输长度取**源**的 `len`；源和目标的 `len` 必须写成相等，否则是 spec 错误。

### 2.2 本地操作

| spec 写法 | 原语 |
|---|---|
| `LocalCopy    A -> B` | `LocalCopy(thread, A, B)` |
| `LocalReduce  A into B` | `LocalReduce(thread, A, B, dataType_, reduceOp_)`，语义 `B = reduce(B, A)` |
| `LocalCopySlices  [A1,A2,...] -> [B1,B2,...]` | `LocalCopySlices` |

`into` 和 `->` 的区别是刻意的：`into` 表示目标是累加位置（读-改-写），`->` 表示纯覆盖。

### 2.3 同步

| spec 写法 | 翻译 |
|---|---|
| `sync main -> sub` | `GetNotifyIdxMainToSub(...)` + `PreSyncInterThreads(T0, subThreads, ...)` |
| `sync sub -> main` | `GetNotifyIdxSubToMain(...)` + `PostSyncInterThreads(T0, subThreads, ...)` |
| `barrier(all threads)` | `HcommBatchModeEnd(algTag)` → `HcommBatchModeStart(algTag)` → 逐个 `HcommThreadJoin(t, CUSTOM_TIMEOUT)` |

`sync main -> sub` 和 `sync sub -> main` 必须成对出现，且都包在 `if N > 1` 的语义下（翻译时自动加）。
第 6 章的数据流写在闭合代码围栏内，同步动作独占一行；条件放在外层，允许缩进与行末注释。
正文说明和注释中的同步记号不计入动作次数；单线程无需伪造同步点。
`barrier(all threads)` 只在 64-bit / PROD 软归约前用，三步顺序不能改。

### 2.4 流

| 记法 | 含义 |
|---|---|
| `T0` | 主流。本地 copy/reduce 一律挂它 |
| `T[i]` | 第 i 个从流（`i` 从 1 起），Mesh 下"第 i 个从流对第 i 个远端" |
| `T[ch]` | 按 channel 下标取流（多 jetty，`ch ∈ 0..NCH-1`） |
| `all` | 所有流，只用于 `barrier` |

---

## 3. 控制流

```
repeat for rpt in 0..RPT-1:         # 外层 repeat（见 §1.5），单层算法 RPT=1 时退化
for i in 1..N-1:                    # 闭区间，含两端
for r in 0..N-1 where r != R:       # 带过滤条件
for k in steps(reduce_scatter):     # NHR：遍历 step 列表
batch for k in 0..M-1:              # 累进同一次批量下发（一个 SendRecvBatch* 调用）
if <谓词>: ... / else: ...
return                              # 提前返回 HCCL_SUCCESS
```

**预定义谓词**（不要自造，避免翻译时二义）：

| 谓词 | C++ |
|---|---|
| `C == 0` | `tempAlgParams.count == 0` |
| `S == 0` | `tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize == 0` |
| `N == 1` | `templateRankSize_ == 1` 或 `subCommRanks_[0].size() == 1` |
| `needAicpuReduce` | dataType ∈ {INT64, UINT64, FP64} 或 reduceOp == PROD |
| `isPcie` | `IsPcieProtocol(templateResource.channels)` |
| `graphMode` | `opMode_ == OpMode::OFFLOAD` / `tempAlgParams.enableRemoteMemAccess` |
| `symMem` | `tempAlgParams.supportSymmetricMemory` |
| `inFromScratch` | `tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER`（输入已在 CCL buffer；其布局仍须核对生产者） |

`batch for` 与普通 `for` 的区别：`batch for` 体内的传输**攒成一个** `SendRecvBatch*` 调用
（`srcSlices` / `dstSlices` 各 push M 个 pair），普通 `for` 是 M 次独立调用。NHR 的每个 step 内部用 `batch for`。

---

## 4. 从 spec 到代码的映射

| spec 章节 | 生成什么 |
|---|---|
| 元信息 | 文件名、类名、header guard、`Describe()` |
| 适用条件 | `CalcCostCoeff` 开头的早退（`if (param.rankSize > 8) return {};`）；写给 selector 的约束说明 |
| 资源 | `CalcRes()`：thread 数、notify 数、channel 申请函数；必要时 `GetRes()` + `GetThreadNum()` |
| Buffer 布局 | `CalcScratchMultiple()` 返回值 —— **必须等于布局图里 scratch 占用的份数** |
| 切片 | `CalcSlice()` / `SplitData()` 私有方法；`RPT/IRS/ORS/ISS/OSS` 的取用方式 |
| 数据流 | `KernelRun()` 及其拆出的 `RunXxx()` 私有方法 |
| 边界条件 | `KernelRun` 开头的早退与校验分支 |
| Cost model | `CalcCostCoeff()` 的 A/B/C/D 计算 |
| 验证 | hccl-vm 用例的引擎/算子/数据类型/通信域选择 |

**一致性检查**（静态脚本只覆盖其中可解析部分；布局与数据归属仍需量化对照及功能验证）：

1. 数据流里出现过 `SCR` / `SCR@p` → scratch 倍数不能是 0
2. scratch 倍数 = Buffer 布局图里 scratch 被划成的份数
3. `sync main -> sub` 与 `sync sub -> main` 成对
4. 写模式的目标必须带 `@p`；读模式的源必须带 `@p`
5. 同一条边的源和目标 `len` 相等
6. 用到 `T[i]`（i ≥ 1）→ 资源章节的 thread 数必须 > 1
7. 切片章节声明的 `RPT != 1` → 数据流必须有 `repeat for`；反之数据流写了 `repeat for` /
   引用了 `RPT`/`IRS`/`ORS` → 切片章节必须把这几个符号声明齐
8. scratch 倍数表达式里**不许出现 `RPT`**（放大由 executor 做，见 §1.5）
9. 全文不得残留 `TBD`

---

## 5. 规格形成与实现前置条件

1. 根据上游需求、已确认约束和目标版本源码填写 spec；无证据的项写成 `TBD: <具体问题>`，不猜。
2. scratch 倍数、尾块、同步和归约等技术项优先用源码与布局公式消解，并记录依据。
3. 运行 **`python3 "$SKILL_DIR/scripts/check_spec.py" <spec.md>`** 和布局量化检查；无法由证据消解的需求语义或范围取舍作为上游输入缺口返回。
4. 语义 TBD 清零且 spec 检查通过后，才可作为代码实现输入；实现能力负责骨架、`KernelRun` 与接线检查。
5. 完整编译和 hccl-vm 双轨验收由验证能力执行，不由本参考文档分配角色或裁决任务状态。

**影响实现语义的 TBD 未解决时，不进入对应代码实现。** spec 里一个 offset 写错，代码全对也是错的。
