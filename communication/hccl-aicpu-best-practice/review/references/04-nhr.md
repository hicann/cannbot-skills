# 04 · NHR 族评审（N01–N05）

**语义**：Nonuniform Hierarchical Ring 的 halving-doubling 变体，非 2 的幂也能用。
ReduceScatter 阶段 log 级步数收敛，AllGather 阶段 log 级扩散，每步和一个"距离 2^k"的对端换若干片。
跨节点场景主力。蓝本：`src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_nhr.cc`。

---

## 1. scratch 倍数：N01（MAJOR|MINOR）

| 算子 | 期望 | 为什么 |
|---|---|---|
| AllReduce | **1** | 原地跑在 hcclBuff 上，只存一份 |
| AllGather / Scatter / ReduceScatter / gather 类 | `templateRankSize_` 常见 | in/out 不等大，scratch 要放下 N 份 |

所以只有 `all_reduce` 按 MAJOR 报，其余 MINOR 人工确认。图模式（`opMode_ != OPBASE`）常返回 0。

## 2. 进出场拷贝：N02（MINOR）

NHR 原地跑在 scratch 上，因此需要一对：

```cpp
CHK_RET(PreCopy(p, res.threads));    // input → hcclBuff
... RunReduceScatter / RunAllGather ...
CHK_RET(PostCopy(p, res.threads));   // hcclBuff → output
```

少了 `PreCopy` → 拿旧数据算；少了 `PostCopy` → 结果留在 scratch 里没回给用户。

## 3. step 表：N03（MAJOR|MINOR）

**step 数必须是 `rankSize` 的函数**（`GetNHRStepNum(rankSize)`），写死成常数就只在一种规格上对——
脚本看到 `for (step < 3)` 这类字面量上界直接报 MAJOR。

每步"和谁换哪几片"应该由一张 step 表描述，而不是散在主循环里：

```cpp
std::vector<NHRStepInfo> stepInfoList;
CHK_RET(GetReduceScatterStepInfoList(stepInfoList));
for (auto& s : stepInfoList) { ... }
```

**人工必看**：`s.txSliceIdxs` / `s.rxSliceIdxs` 的下标推导，以及尾片
（`tx == templateRankSize_ - 1` 时用 `dataOffsetTail_` / `dataSplitTail_`）的分支。

## 4. 多 jetty：N04（MAJOR）

一个远端有多条 channel 时，线程数由 channel 数决定，**必须同时覆写两个方法**：

```cpp
HcclResult XXX::GetRes(AlgResourceRequest& req) const;  // 按 channelsPerRank_ 申请
u64 XXX::GetThreadNum() const { return channelsPerRank_; }
```

并在 `CalcRes` 里先 `channelsPerRank_ = CalcChannelsPerRank(channels);` 再 `GetRes(req);`。
少一个 → 申请到 `rankSize` 个线程却要跑 `channelsPerRank_` 个 step，或反过来。
变体子类可以由父类提供这两个方法（脚本会沿继承链找）。

数据的二次切分用 `CalcDataSplitByPortGroup(...)`，`dataSplit_[ch] / dataOffset_[ch]` 从它的输出取。

## 5. channel 校验与读模式：N05（INFO）

- NHR 每步的收发对端**不是同一个**：`from = rankList_.at(s.fromRank)`、`to = rankList_.at(s.toRank)`，
  两个都要 `channels.count()` 过一遍。脚本只能数 `.count(` 的出现次数，所以按 INFO 提醒。
- PCIe 链路走**读模式**：`isDmaRead_ = IsPcieProtocol(res.channels);`，
  真时用 `SendRecvReadReduce` / `SendRecvBatchRead`，假时用 `SendRecvBatchWriteReduce` / `SendRecvBatchWrite`。
  两条路径的地址方向是**反的**，人工要各读一遍（对应通用规则 C14）。
