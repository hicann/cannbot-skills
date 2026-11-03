# ScatterUpdate Tiling 预处理：校验 / SortLib / 元信息

> dim 归一化 / 分型校验清单 / SortLib tiling / use64Bit / 多核切分 / workspace 基础四段 + SUBSET 段5布局。设计文档 §3 的真值源。

---

## 1 dim 归一化与静态校验

```cpp
// dim ∈ [-R, R)，负值 +R 归一化（仅 dim 形态；nd 形态无 dim）
int64_t dimNorm = (dimRaw < 0) ? dimRaw + static_cast<int64_t>(rank) : dimRaw;
OP_CHECK_IF(dimNorm < 0 || dimNorm >= static_cast<int64_t>(rank),
    OP_LOGE(context, "dim %ld out of range [0, %zu)", dimNorm, rank),
    return ge::GRAPH_FAILED);
```

校验清单（infershape 与 tiling 双端，动态维由 tiling 兜底；逐条标注适用形态）：

- 【dim 形态】`rank(var) == rank(index) == rank(src)`，`rank ∈ [1, MAX_RANK]`（nd 形态无此约束）
- 【dim 形态·index 覆盖全部维】前段子集：`index.size(d) <= src.size(d)`（所有 d，不广播）——覆盖单维子场景 src 为 flat 沿 axis，无此约束
- 【dim 形态】`index.size(d) <= var.size(d)`（d != dim）
- 【nd 形态】`K = indices.shape[-1] ∈ [1, rank(var)]`
- 【nd 形态】`updates.shape == indices.shape[:-1] + var.shape[K:]`（逐维相等）
- 【全型】var/updates（或 var/src）dtype 一致；index/indices dtype ∈ {int32, int64}
- 【src 表示】**标量源子场景**：满秩 src.shape 校验（前段子集 / updates.shape 拼接）不适用，N 仅由 index/indices 决定；标量值**来源二选一**——① 标量属性：tiling `GetAttr` 读入并按字节位模式打包进 TilingData 固定字段（禁数值转换）；② 标量张量输入（rank-0 / shape=[1]）：**存在 src 张量输入**，仅校验 shape ∈ {(), (1,)}，tiling 不读值（无值依赖），kernel 运行时按字节读 GM

> ⚠️ **index 取值越界检查依赖运行时数据**，无法在 tiling/infershape 侧完成——由 Phase1 kernel 运行时按 [bounds](../../generic/bounds.md) 形态处理（__trap / 哨兵）。
>
> ⚠️ **host 侧禁止读取/校验 index/indices 的具体值**（tiling 阶段不消费输入张量数据）。

## 2 元信息计算

```cpp
// var 各维尺寸与行主序步长（两形态共用）
int64_t varAcc = 1;
for (int32_t d = rank - 1; d >= 0; --d) {
    varDims[d] = varShape.GetDim(d);
    varStrides[d] = varAcc;
    varAcc *= varDims[d];
}
totalVarElems = varAcc;

// 【dim 形态】index 各维步长（坐标反解用）+ N = index.numel()
// 【nd 形态】K = indices.shape[-1]；N = prod(indices.shape[:-1])；sliceSize = prod(var.shape[K:])
```

key dtype 自适应与 use64Bit / useInt32Index 判定：

```cpp
// KeyT 按 totalVarElems 自适应（linearize.md §3）
// uint32 坐标计算域（det/nondet 统一）：五项元素计数均 ≤ INT32_MAX 才启用
int64_t varNum   = var.GetShapeSize();
int64_t srcNum   = srcOrUpdates.GetShapeSize();     // 标量源填 1
int64_t outNum   = varOut.GetShapeSize();
int64_t indexNum = indexOrIndices.GetShapeSize();   // nd 形态为 N×K
int64_t taskNum  = ...;                             // N / b×N×s / N×slice
bool useInt64 = (varNum > INT32_MAX) || (srcNum > INT32_MAX) ||
                (outNum > INT32_MAX) || (indexNum > INT32_MAX) ||
                (taskNum > INT32_MAX);
bool useInt32Index = !useInt64;
// taskNum：全部维/展平=N；单维=batchSize×N×sliceSize；nd=N×sliceSize
```

## 3 SortLib tiling（Phase2，det 路径）

```cpp
// 排序对象 = linearIdx（keySize 字节），置换 = int32
bool isInt32Safe = SortLib::IsInt32Safe(totalIndexNum);      // N ≤ 2^30 用 uint32 计数
auto sortR = SortLib::SortTilingCompute(
    totalIndexNum, coreNum, ubSize - STATIC_UB_ESTIMATE,
    keySize, 4 /*perm int32*/, isInt32Safe, keyValueType);
needCoreNum = (sortR.coreNumNeed > 0) ? sortR.coreNumNeed : 1;
// numTileData / tileCount / activeCores / tmpUbSize / isSingleCore 直传 TilingData
```

- 排序不是瓶颈（radix_sort_simd 直测 74us@185K，占大 shape 总耗时 ~2%）——**不要在排序上投入优化**。
- SortLib UB 预算须扣除静态参数：`ubSize - STATIC_UB_ESTIMATE`。

## 4 多核切分

det 三阶段排布策略与公式真值源在 [../../generic/thread-layout.md](../../generic/thread-layout.md) §1。本文仅载 AscendC 侧落地。

```cpp
// det 路径：三阶段共用核数，Phase1/Phase3 全核 Grid-Stride，Phase2 SortLib 内建多核
int64_t phase1Cores = (totalIndexNum + THREAD_NUM - 1) / THREAD_NUM;
// taskNum = N_eff：全部维/展平=N；单维=batchSize×N×sliceSize；nd=N×sliceSize
int64_t taskNum = ...;
int64_t phase3Cores = (taskNum + THREAD_NUM - 1) / THREAD_NUM;
int64_t phaseCores = std::max(phase1Cores, phase3Cores);
phaseCores = std::min(phaseCores, coreNum);
if (phaseCores < 1) { phaseCores = 1; }      // 空 tensor 兜底
int64_t blockDim = std::max(phaseCores, static_cast<int64_t>(needCoreNum));
if (blockDim < 1) { blockDim = 1; }
context->SetBlockDim(blockDim);
context->SetScheduleMode(1);                  // 排序 SyncAll 依赖

// nondet 路径：核数两步法，用户 workspace = 0；TilingData 不携带 needCoreNum
// blockDim = min(coreNum, max(1, ceil(N / 1024)))，SetBlockDim 下发，kernel 用 gridDim 自解析
```

## 5 workspace 布局（det 基础四段 + SUBSET 段5，128B 对齐）

```
usrWs 基址:
[0]                          段1 sortWorkspace     大小 = sortR.workspaceBytes
[AlignUp(seg1End, 128)]      段2 linearIdxBuffer   N × keySize（Phase1 出 / Phase2 入）
[AlignUp(seg2End, 128)]      段3 sortedLinearIdx   N × keySize（Phase2 出 / Phase3 入）
[AlignUp(seg3End, 128)]      段4 permutation       N × 4（Phase2 出 / Phase3 入）
[AlignUp(seg4End, 128)]      段5 srcPosBuffer      N × 8（仅 SUBSET；SAME/单维/nd 不分配，wsSrcPosOff=0）
```

- 段1 由 SortLib 内部管理（histogram / tile data / double buffer）。
- **nondet 路径用户 ws = 0**（仅系统 workspace）；系统 workspace 无条件分配（`GetWorkspaceSizes(1)` 必须调用）。
- 覆盖单维 / nd / SAME 形态无 srcPosBuffer（src/updates 直接 flat 访问 `updates[perm·sliceSize + s]`）；**仅 index 覆盖全部维且 `index.size(d) < src.size(d)` 的 assign SUBSET 场景**启用 srcPos 双输出时，追加段5并写 `wsSrcPosOff`（**srcPos 双输出加速**见 [linearize](linearize.md) §4）。

## 6 DCache 预留

DCache 预留的硬件级规则真值源在 [common/simt-hardware-model.md](../../../common/simt-hardware-model.md) §4。本文仅载本范式 AscendC 侧落地（SortLib 依赖 DCache 预留）。

- `context->SetLocalMemorySize(ubSize - dcache - STATIC_UB_ESTIMATE)`，剩余不足 32KB 时 `OP_CHECK_IF` 报错兜底。
- Phase1/Phase3 的 SIMT VF UB 参数缓冲（dims/strides/magic/shift）由 kernel 内 `LocalMemAllocator` 动态分配，不占此预留。
