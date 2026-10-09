# 03 · Mesh 族评审（M01–M05）

**语义**：level0 全连接，任意两 rank 直连，一步到位。仓内存量最多的一族。
变体：one-shot / two-shot / mesh-chunk / omnipipe / Z 轴绕行 / DPU / intra / inter。

评审前先打开同族蓝本对照：
`src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_one_shot.cc`（one-shot）、
`..._two_shot.cc`（two-shot）、`src/ops/all_gather/algorithm/template/aicpu/ins_temp_all_gather_mesh_1D.cc`（带 repeat/stride 的 gather 形态）。

---

## 1. one-shot：M01（MAJOR|MINOR）/ M05（MAJOR）

**形态**：一步把整份数据推给所有对端的 scratch，然后各自本地把 N 份归约掉。

| 项 | 期望 |
|---|---|
| `CalcScratchMultiple` | `templateRankSize_`——每个 rank 在我 scratch 里独占一段，共 N 段 |
| thread 数 | `templateRankSize_` |
| 收尾 | **必须有 `LocalReduce`**（归约类算子），否则输出里只有自己那份 |

- **M01**：倍数不是 N。报小 → `hcclBuff` 越界；报大 → 白切 loop。
  `all_reduce` 按 MAJOR，其余算子（in/out 不等大）按 MINOR 人工确认。
- **M05**：`all_reduce / reduce / reduce_scatter` 的 one-shot 变体没有收尾 `LocalReduce`。

## 2. two-shot：M02（MAJOR|MINOR）

**形态**：ReduceScatter（每 rank 只归约自己那片）+ AllGather（互相广播结果）。

| 项 | 期望 |
|---|---|
| `CalcScratchMultiple` | AllReduce 场景 **1 或 2**（理论 1 份够，非均衡切分时不足，蓝本取 2 并注明理由） |
| thread 数 | `templateRankSize_` |
| 两阶段 | 各自 `PreSync` / `PostSync`，共两对 |

`reduce_mesh_1D_two_shot.cc` 返回的是 `enableRemoteMemAccess_ ? 0 : templateRankSize_`——
Reduce 的落点不同，N 倍是对的。所以非 `all_reduce` 只报 MINOR。

**人工必看**：切片用 `RoundUp(count, sliceNum)` 向上取整 + 尾片截断，
`sliceInfoList_` 的最后一片是 `count_ - offsetCount`，两阶段必须用同一张表。

## 3. 错峰：M03（MINOR）

```cpp
for (u32 queIdx = 1; queIdx < threadNum_; queIdx++) {
    u32 nextRank = (myRank_ + queIdx) % templateRankSize_;   // ★ 错峰
```

所有 rank 同时打同一个目标会把链路打爆。脚本只能看出"从流循环里没有 `(myRank_ + i) % N` 这个形状"，
**顺序对不对要人工看**。

## 4. 线程模型：M04（MINOR）

`slaveThreadNum` 必须从 `templateRankSize_` 或 `channelsPerRank_` 推导，写死数字在别的卡数规格上就错。
多 jetty（一个远端多条 channel）时线程数 = `channelsPerRank_`，且必须覆写 `GetRes()` + `GetThreadNum()`（同 N04）。

M04 还兼管一条形态检查：**mesh 里出现 `stepInfoList` / `GetNHRStep` 通常是从 NHR 蓝本抄来没删干净**。

## 5. 变体的额外看点（脚本判不了）

| 变体 | 看什么 |
|---|---|
| mesh-chunk | 分块粒度与 `CalcScratchMultiple` 是否自洽；块尾的处理 |
| omnipipe | `stepSliceInfo` 给的 `repeatNum` 与地址推导；与另一层 template 的 scratch 预算是否重复计算 |
| Z 轴绕行 | 绕行路径上的中转 rank 是否也建了 channel；绕行只改路由不改语义 |
| intra / inter | 它是被分层 executor 当构件用的 → **必须写 repeat 循环**（C03 在这里升级为硬要求） |
| DPU | `.cc` 末尾要有 `REGISTER_TEMPLATE_V2("类名", 类名);`（规则 W07） |
