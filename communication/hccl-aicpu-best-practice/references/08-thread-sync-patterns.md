# Thread 主从流与同步顺序

> **真源归属**：hccl-aicpu-best-practice skill（自入口 skill 迁入，长期保留）。事实来源：`src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc`（Pre/PostSyncInterThreads 实现）、`src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc`（launch 级同步）、`src/ops/op_common/inc/alg_param.h`（notify 常量）、各算子 `algorithm/template/aicpu/*.cc` 真实实现。
>
> **覆盖度**（2026-09 全量扫描 53 个 AICPU 模板）：§4 深入分析 8 个代表模板；其余模板按同步签名归类（标准型 25 个 / 无 sync 型 16 个 / 多 sync 变体型约 10 个，见 §4 家族表与 §5 变体）。标注"模式同§4"的家族未逐一深查代码，使用前建议 grep 目标文件核对。

## 1. 三级同步架构

一次 AICPU 集合通信任务的完整同步链路分三级：

```text
Host stream ──aclrt notify──▶ 主线程(threads[0]) ──thread notify──▶ 从线程(threads[1..N-1])
                                  │                                        │
                                  └────────channel notify(ACK/DATA_SIGNAL)──▶ 对端 rank 线程
```

| 级别 | 接口 | 位置 | 触发方 |
|------|------|------|--------|
| ① launch 级（stream ↔ 主线程） | `HcommAclrtNotifyWaitOnThread(threads[0], notifyIds[0])` 开始等待；`HcommAclrtNotifyRecordOnThread(threads[0], notifyIds[1])` 结束回执 | kernel_launch.cc | 框架，template 不感知 |
| ② 线程间（主 ↔ 从） | `PreSyncInterThreads` / `PostSyncInterThreads`（底层 `HcommThreadNotifyRecord/WaitOnThread`） | template KernelRun | template |
| ③ 通道级（本线程 ↔ 对端） | `HcommChannelNotifyRecord/WaitOnThread`，索引约定 `NOTIFY_IDX_ACK=0` / `NOTIFY_IDX_DATA_SIGNAL=1` | wrapper 内置 | wrapper |

AICPU_TS（导出线程）场景下 ① 改用 `HcommThreadNotifyWaitOnThread(threads[0], notifyNumOnMainThread)` / `HcommThreadNotifyRecordOnThread(threads[0], opThread, 0)`，仍由框架处理。

## 2. 资源配比公式（CalcRes 必须与 notify 索引一致）

```cpp
u32 threadNum = <template 决定>;                        // 见 §4 各算子取值
resourceRequest.slaveThreadNum = threadNum - 1;
resourceRequest.notifyNumPerThread.assign(slaveThreadNum, 1);  // 每个从线程 1 个 notify
resourceRequest.notifyNumOnMainThread = threadNum - 1;         // 主线程 N-1 个 notify
```

### notify 索引方向约定（与 wrapper 实现严格对应）

| 方向 | record 方 | wait 方 | 索引 |
|------|----------|---------|------|
| 主→从（PreSync） | 主线程向**每个从线程** record | 各从线程 wait 自己的 | `notifyIdxMainToSub` 全 0（每从线程用自己的 notify[0]） |
| 从→主（PostSync） | 每个从线程向**主线程** record | 主线程 wait 全部 | `notifyIdxSubToMain` = 0..N-2（主线程专属索引，每从线程一个） |

```cpp
// 通用实现（各算子一致）
void GetNotifyIdxMainToSub(std::vector<u32>& idx) { idx.assign(threadNum - 1, 0); }
void GetNotifyIdxSubToMain(std::vector<u32>& idx) { for (u32 i = 0; i < threadNum - 1; i++) idx.push_back(i); }
```

### Pre/PostSync 内部顺序（重要）

**PreSync 先下发全部 record，再下发全部 wait；PostSync 先下发主线程的全部 wait，再下发从线程的全部 record。** 这是 wrapper 的任务下发顺序；设备执行时 wait 必须等对应 record 满足，不能将 host 下发顺序理解为同一线程上的同步阻塞。

## 3. 标准 KernelRun 同步时序

```text
KernelRun
  ├─ 主线程 LocalCopy（in→out / in→cclBuff）        ← 主线程本地准备，无需同步
  ├─ [threadNum>1] PreSyncInterThreads              ← 主→从：放行从线程
  ├─ Run<Op>：各线程并行执行通信
  │    主线程: 本地拷贝或首个 peer 的通信（按算子而定，见 §4）
  │    从线程[i]: wrapper 调用（内部自带 channel notify ACK/DATA 协议）
  ├─ [threadNum>1] PostSyncInterThreads             ← 从→主：汇合
  ├─ [needAicpuReduce_] BatchModeEnd/Start + HcommThreadJoin(全线程)   ← 64bit/PROD 插入点
  └─ 主线程 LocalReduce / PostCopy                   ← 归约收尾必须在主线程（确定性顺序）
```

**PostSync 之后主线程的数据依赖才安全**：从线程搬运的数据在 PostSync 汇合后才能被主线程的 LocalReduce 消费；64bit/PROD 因 AICPU 归约读取 cclBuff 更晚，还须 ThreadJoin 等写任务真正落盘。

## 4. 各算子线程分配对照（真实代码）

### 4.1 深入分析的代表模板（8 个）

| 算子模板 | threadNum | 线程分配 | 主线程(threads[0])职责 | 参考文件 |
|---------|-----------|---------|----------------------|---------|
| AR Mesh1D OneShot | rankSize | **相对偏移**：threads[queIdx] ↔ peer (myRank+queIdx)%N | LocalCopy + PostLocalReduce | all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_one_shot.cc |
| AR Mesh1D TwoShot | rankSize | **绝对索引**：threads[remoteIdx] ↔ 数据片 remoteIdx（自片=LocalCopy） | ReduceData：片 1..N-1 顺序 LocalReduce 到片 0（确定性归约） | all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_two_shot.cc |
| AR NHR | channelsPerRank_ | **按通道**：threads[channelIdx]，所有 step 共用 | PreCopy → sync → RS+AG per channel → sync → PostCopy | all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_nhr.cc |
| AG Mesh1D | (rankSize-1)×maxChannelsPerRank | **(peer,channel) 轮转**：queIdx 从 **0** 起，主线程也参与通信 | LocalDataCopy（in→out + in→cclBuff），且承担首个 (peer,channel) | all_gather/algorithm/template/aicpu/ins_temp_all_gather_mesh_1D.cc |
| RS Mesh1D | threadNum（= 申请数） | **相对偏移 + 多通道**：queIdx 从 **1** 起，per (peer,channel) 递增 | LocalCopy/LocalReduce（cclBuff 内多片归约） | reduce_scatter/algorithm/template/aicpu/ins_temp_reduce_scatter_mesh_1D.cc |
| Reduce Mesh1D | rankSize | **角色分工**：root 的从线程 RecvWrite；非 root 的主线程 SendBatchWrite 到 root | root: 本地数据准备、接收汇合与顺序 LocalReduce；非 root: 发送 | reduce/algorithm/template/aicpu/reduce_mesh_1D.cc |
| Broadcast Mesh1D TwoShot | rankSize | **两段各自分配**：Scatter 段 threads[i]↔commRanks[i]（root 只发/非 root 只收）；AllGather 段 threads[i]↔commRanks[i] 双向 | root 片 LocalCopy | broadcast/algorithm/template/aicpu/ins_temp_broadcast_mesh_1D_two_shot.cc |
| Barrier Mesh1D | rankSize-1（单卡为 1） | threads[threadIdx] ↔ (myAlgRank+1+threadIdx)%N | 只走 notify 同步对（空 slice 的 SendRecvWrite），不搬数据 | barrier/algorithm/template/aicpu/ins_temp_barrier_mesh_1D.cc |

### 4.2 其余模板家族归类（按同步签名，未逐一深查）

| 家族 | 成员（同步签名） | 模式归属 |
|------|----------------|---------|
| intra/inter 阶段模板 | AR 的 rs_mesh_1D_intra / ag_mesh_1D_intra，broadcast 的 scatter/ag intra，reduce 的 gather_intra（均 Pre:1 Post:1，slave=threadNum-1） | 模式同 §4.1 RS/AG Mesh（queIdx 从 1 起，相对偏移）；作为 3 级 sequence executor 的阶段 template 被 Orchestrate 串联 |
| 其它算子的 Mesh/NHR 标准型 | broadcast_nhr、scatter_mesh_1D、scatter_nhr、reduce_two_shot、reduce_scatter_meshchunk、AG/RS omnipipe_mesh_1D、all_gather_v/reduce_scatter_v_mesh_1D、AG omnipipe_nhr、order_preserved_group/level1、dpu_alltoall_mesh（均 Pre:1-2 Post:1-2） | 模式同 §4.1 + §5.1 多段各自 Pre/Post；omnipipe 的 queIdx 从 0 起（主线程参与） |
| **无 Pre/Post 型**（16 个） | 各 `*_dpu` / `*_dpu_inter`（slave=0）、`*_nhr_dpu`、barrier_nhr_aicpu、`*_Z_axis_detour`、`aicpu_reduce_nhr` 家族、reduce_nhr | **不使用线程间 Pre/Post 同步**：DPU 系单线程执行（slaveThreadNum=0），同步走 HcommThreadSynchronize/SendRequest/WaitResponse（见 `02` §9.1）；aicpu_reduce 系用 BatchMode+ThreadJoin 节奏（§5.3）；Z_axis_detour 继承基类结构。使用前须 grep 目标文件确认 |
| **多 sync 变体型** | AG nhr（Pre:2 Post:1）、RS nhr（Pre:1 Post:2）、AR mesh_chunk（Pre:3 Post:3）、A2A mesh（Pre:5 Post:5）、UBX A2A（Pre:2 Post:2，分组） | 见 §5.4-§5.8 |

> RS nhr 的 Pre:1 Post:2 为假性不对称：第二个 Post 在 rankSize≤1 单卡短路分支内，主路径仍是标准 Pre/Post 对。

三种分配语义的区别：
- **相对偏移** `(myRank+queIdx)%N`：各 rank 的同号线程负载对称（每线程都是一收一发），one_shot/RS 用。
- **绝对索引** `threads[remoteIdx]`：线程号=数据片号，与 rank 无关，主线程恰好处理自己的片，two_shot/Reduce 用——注意此时从线程号在不同 rank 上指向不同方向的对端。
- **按通道/轮转**：线程数与 rankSize 脱钩，AG/NHR 多通道场景用——`SetchannelsPerRank()` 先统计 `channelsPerRank_`/`maxChannelsPerRank_` 再定 GetThreadNum。

## 5. 同步变体

### 5.1 多阶段算法：每段独立 Pre/PostSync

TwoShot（RS→AG 两段）、Broadcast（Scatter→AllGather 两段）在**每个阶段内部**各自做 Pre → 通信 → Post，阶段之间天然串行（主线程顺序调用，PostSync 保证上一段全部完成）：

```cpp
RunReduceScatter(): PreSync → ScatterData → PostSync → [needAicpuReduce_: Join] → ReduceData
RunAllGather():     PreSync → GatherData  → PostSync
```

### 5.2 NHR（线程=通道）：一次 sync 包住全部 step

```cpp
PreCopy(主线程) → PreSync → for channelIdx: { RunReduceScatter(threads[channelIdx]); RunAllGather(threads[channelIdx]); }
→ PostSync → PostCopy(主线程)
```

Pre/PostSync 的 subThreads 只取 `threads.begin()+1 .. begin()+threadNum_`（threadNum_ = channelsPerRank_，可能小于 threads.size()）。

### 5.3 64bit/PROD（needAicpuReduce_）插入点

PostSync 之后、主线程读 cclBuff 归约之前：

```cpp
HcommBatchModeEnd(param.algTag);   // 结束当前批量段（刷掉已下发任务）
HcommBatchModeStart(param.algTag); // 开新段
for (thread : threads) HcommThreadJoin(thread, CUSTOM_TIMEOUT);   // 等 device 侧真正写完
```

### 5.4 线程对间 sync（AG NHR）：任意线程对，不止主从

AG NHR 为通信与收尾拷贝重叠，把 PostLocalCopy 下发到**专用搬运线程** `threads[channelsPerRank_+channelIdx]`，由通信线程在**最后一步前**对其做 producer→consumer 依赖同步（`ins_temp_all_gather_nhr.cc:325`）：

```cpp
constexpr u32 POST_COPY_NOTIFY_IDX = 1;   // 专用 notify 索引，与主从索引(0)区分
PreSyncInterThreads(threads[channelIdx], {threads[postCopyThreadIdx]}, {POST_COPY_NOTIFY_IDX});
PostLocalCopy(threads[postCopyThreadIdx], channelIdx);   // 与剩余 step 的读并行
```

要点：Pre/PostSyncInterThreads 本身支持**任意线程对**（参数即 main/sub），主从只是最常见用法；线程间 notify 索引空间需按线程规划，避免冲突。

### 5.5 chunk 循环内嵌 sync（AR MeshChunk / RS MeshChunk）

数据按 chunk 分多轮搬运时，**每个 chunk 轮内独立 Pre/Post**（AR mesh_chunk Pre:3 Post:3：两轮通信 chunk + 一轮收尾），形如：

```cpp
for (chunk : chunks) { PreSync → 该 chunk 的 Scatter/Gather → PostSync }
```

### 5.6 轮内错峰 sync（A2A Mesh1D）

A2A 的多轮 peer 通信中，第 1 轮**中途**插入一组 Post+Pre，把主线程从"等待"中放出，用 0 号流穿插本卡数据拷贝（`ins_temp_all_to_all_v_mesh_1D.cc:191-213`）：

```cpp
PreSync → 第1轮通信下发 → PostSync + PreSync（错峰）→ LocalCopyForMyRank(threads[0]) → 其余轮通信 → PostSync
```

线程分配也特殊：`queIdx = myRankCclBuffIdx * channelsPerRank_ + 1`——起始索引与本 rank 的 cclBuff 槽位挂钩；另有 `PreSyncInterThreadsPerRank` 封装按 rank 分组同步。

### 5.7 线程分组 sync（UBX A2A）

线程按角色分两组（板内 board / 全互联 fullMesh），**各组独立 Pre/Post**（`ins_temp_ubx_all_to_all_v_mesh_1D.cc`）：

```cpp
subThreadsBoard_ = threads[1 .. 1+maxRankNumPerBoard_);     // 板内组
subThreadsFullMesh_ = threads[1+maxRankNumPerBoard_ .. );   // 全互联组
PreSync(threads[0], subThreadsBoard_) → 板内阶段 → PostSync(...) → PreSync(threads[0], subThreadsFullMesh_) → 全互联阶段 → PostSync(...)
```

threadNum = maxPathNum_ + maxRankNumPerBoard_（与 rankSize 无关），并设 fullMeshThreadId 专用线程承担本地/全互联拷贝。

### 5.8 线程复用回绕（order_preserved 系列）

线程数小于通信轮数时，queIdx 递增到 threadNum_ 后**回绕到 1 循环复用**（`ins_temp_reduce_scatter_order_preserved_group.cc:295-298`）：

```cpp
queIdx++;
if (queIdx >= threadNum_) { queIdx = 1; }   // ring 多轮复用同一批从线程
```

复用安全的依据：每轮之间有 §5.1 的分轮 Pre/Post 汇合，同一线程的任务在 device 侧天然按下发序执行。

## 6. 新增算法的同步设计规则

1. 标准主从模式用 Pre/Post 包裹存在依赖的并行通信；特殊线程对或多阶段模式按实际依赖验证，不能只按源码调用次数判断配对。单线程无需主从同步。
2. 一个线程可按队列顺序执行多个通信 step（如 §5.2 NHR）；跨线程共享数据或复用 buffer 前，按生产者/消费者依赖补同步，不能将每 step 主从汇合当作通用要求。
3. **notify 索引遵循 §2 约定**（MainToSub 全 0 / SubToMain 递增），且 GetNotifyIdx* 的长度必须等于 CalcRes 的 slaveThreadNum——长度不一致时 Pre/PostSync 直接报 `HCCL_E_INTERNAL`。引入线程对间 sync（§5.4）时为专用用途分配独立索引并在 CalcRes 中计入 notifyNumPerThread。
4. **最终归约/PostCopy 放主线程**并按固定片序执行（确定性计算要求，two_shot 的 ReduceData 是范本）；追求重叠时可用 §5.4 的专用搬运线程模式。
5. **数据依赖决定 sync 位置**：主线程要消费从线程写入的数据（cclBuff/userOut）前必须有 PostSync；64bit/PROD 额外要求 ThreadJoin。
6. **threadNum 必须与 CalcRes 申请一致**：KernelRun 开头校验 `threadNum_ != 预期` 直接报错（参考 one_shot/two_shot 的 CHK_PRT_RET）。
7. 主线程在 PreSync **之前**可自由做本地准备（LocalCopy 无需同步）；PostSync **之后**可自由做收尾。
8. **优先套用已知模式**：先对照 §4/§5 找同拓扑同数据量档位的已有模板，套其 sync 结构再改搬运逻辑；DPU/单线程场景不引入 Pre/Post。
