# Reduction 范式切分算法（UB 双切分 + 多核切分 + UB 预算）

> **层级**：通用知识层（generic/），语言中立。
>
> 来源：adapters/ascendc/templates/binary-base/tiling.md §1/§2/§5（原 reduction-binary-base-tiling.md）、adapters/ascendc/templates/binary-group/tiling.md §2/§5、adapters/ascendc/templates/empty/tiling.md §1/§2/§5。C++ 实现（`ComputeAUbFactor()` / `ComputeRUbFactor()` 等）在适配层。

## 1 记号与输入

| 记号 | 含义 |
|------|------|
| `axisShape[]` / `axisNum` | 合轴后的 pattern（A 起头、严格交替），来自 [axis-preprocessing.md](axis-preprocessing.md) |
| `isTailR / isTailA` | axisNum 偶数 → tail R；奇数 → tail A |
| 片上容量 | 单核算可用的片上 buffer 总容量（AscendC: UB 大小） |
| `P_pre` / `P_pre_ext`（固定 1）/ `P_post` | 三段式 buffer 份数，来自 [liveness-fusion.md](liveness-fusion.md) §2 |
| `ALIGN_GRANULARITY` | 片上数据对齐粒度（能力谓词）；`bsElem = ALIGN_GRANULARITY / sizeof(D_T)`（元素数取整用） |
| `CACHELINE_SIZE` | cache line 宽度（能力谓词），决定爬坡上界 |
| `COMPUTE_PRECISION` | 归约统一计算精度（能力谓词）；`maxDtypeSize = max(sizeof(D_T), 精度宽度)`（字节容量用） |
| `TREE_CACHE_CAPACITY` | 树缓存容量（能力谓词，固定值） |
| `maxCores` | 可用核数 |

⛔ **dtype 派生参数适用边界**：`bsElem`（元素数取整）按输入 dtype 算；`maxDtypeSize`（字节容量）按输入 dtype 与计算精度取宽者算——**禁止混用**。片上 buffer 统一按 `maxDtypeSize` 分配容量（跨 dtype 复用）。

## 2 UB 双切分（Base / Group Phase 1 通用）

### 2.1 切分原理

A 轴与 R 轴**各有**一个片上切分点，互相独立：

- **R 必须切**：片上必须做归约，必须把 R 搬到片上
- **A 必须切**：tail-A 下 R 在外部存储非连续排布，A 切入片上才能保证搬运连续性
- 统一 AR/RA 用同一套双切分框架

被切分的轴记为切分轴（`aSplitIdx` / `rSplitIdx`），切分因子 `aUbFactor` / `rUbFactor`。**轴归属**：切分点**左侧**轴 + 切分轴的外层部分（`.o`）→ 外层 for 循环；切分轴的内层部分（`.i`）+ **右侧**所有轴 → 驻留片上。整轴全载是切多 chunk 的特例（factor = axisSize，`.o = 1`）。

### 2.2 三步算法（顺序固定）

**Step 1：定 A 切分（联合爬坡）**

A 轴与 R 轴一起从最内轴向外爬坡，目标上界为 `cachelineTmp`（一个 cache line 的元素数，`FloorAlign(CACHELINE_SIZE / sizeof(D_T), bsElem)`）。爬坡中 R 轴参与累乘（估算片上 A+R 总元素量）但不进 `innerAProdAlign`（只统计 A 轴）。尾轴（无论 A/R）一律 CeilAlign 到对齐粒度（burst 尾轴方向），非尾轴按原值累乘。终止于 `product × axisSize > cachelineTmp`。

停止轴 `idx` 的判定：

| `idx` 情况 | `aSplitIdx` | `aUbFactor` | 说明 |
|---|---|---|---|
| `idx < 0`（所有轴全驻） | 0 | `axisShape[0]` | 全部装入仍未超上界，无轴可切 |
| `idx` 是 A 轴 | `idx` | `min(cachelineTmp / product, axisShape[idx])` | 直接切该 A 轴 |
| `idx` 是 R 轴 | `idx − 1` | 1 | R 轴不能作 A 切分点，左移到其左侧第一根 A（交替排列必存在），每 chunk 取 1 |

`innerAProdAlign`（padded）只统计切分轴**右侧**的 A 轴，tail-A 时最内 A 轴 CeilAlign。定义 `aUnit = aUbFactor × innerAProdAlign`（片上 A 维元素数）。tail-A 下 aUnit 一定对齐；tail-R 下可能非对齐。**不需要 `aUbFactorAlign`**：三种情形（aSplitIdx==LastA / !=LastA / tail-R）均直接使用 `aUbFactor`。

**Step 2：反解 R 切分（UB 预算反解）**

1. **先算 `postBufSize`**（归约后段单份 buffer 字节）：`CeilAlign(aUnit × maxDtypeSize, ALIGN_GRANULARITY)`（1D，必须对齐）
2. **反解 `r_i_max`**：可用片上容量先扣固定树缓存，再减归约后段独占字节，剩余除以"每 R 元素的归约前段字节开销"：

```
ubAvailable = 片上容量 − TREE_CACHE_CAPACITY
aOnlyBytes  = P_post × postBufSize
bytesPerRElem = (P_pre + P_pre_ext) × aUnit × maxDtypeSize
r_i_max = (ubAvailable − aOnlyBytes) / bytesPerRElem      // 向下取整
```

> `aOnlyBytes` 用 `postBufSize`（1D 须对齐）；`bytesPerRElem` 用 `aUnit × maxDtypeSize`（2D，burst 尾轴方向已由 padded 字段保证对齐，非尾轴方向不对齐，无需额外 CeilAlign）。

3. **R 端爬坡**：从最内 R 轴向外逐根吸入 `innerRProdAlign`（tail-R 时尾轴先 CeilAlign 再比较），终止于超预算
4. **定 `rUbFactor` 与 padded 版**：`rUbFactor = min(r_i_max / innerRProdAlign, axisShape[rSplitIdx])`；
   - tail-R 且 rSplitIdx 是尾轴（burst 尾轴方向）：切多 chunk 需 FloorAlign（对齐后为 0 → 切分失败）；整轴需 CeilAlign 并校验不超预算，放大超预算则退回切多 chunk
   - 其他情况不需对齐，`rUbFactorAlign = rUbFactor`
   - **Align > valid 条件**：仅 tail-R 且切尾轴且尾轴全载且非对齐时 `rUbFactorAlign > rUbFactor`，其余相等

**Step 3：所有 R 全驻后扩 A**

若 R 切分轴整轴全载且已无更外 R 轴（所有 R 全驻片上），回头扩 A 填满剩余空间：

1. **解 `aUnitMax`**：R 端固定占用代回预算不等式，提出公因式：

```
aUnitMax ≤ ubAvailable / (maxDtypeSize × ((P_pre + P_pre_ext) × rPaddedElems + P_post))
其中 rPaddedElems = rUbFactorAlign × innerRProdAlign
```

   并取 `min(aUnitMax, ∏所有 A 轴 size)`
2. **钳制树缓存**：所有 R 全驻即树缓存只有 1 层，`aUnitMax ≤ TREE_CACHE_CAPACITY / 精度宽度`（不钳制会树缓存溢出）
3. tail-A 时在 min(∏A) 钳制**之后** FloorAlign（∏A 可能非对齐）；重做 A 端爬坡（只爬 A 轴），切分点可能外移，重算 `innerAProdAlign`
4. `aUnitMax ≤ 当前 aUnit` 时扩 A 无效，保持 Step 1 结果不变

### 2.3 valid / padded 双字段语义

| 字段对 | valid（原值）用途 | padded（对齐值）用途 |
|---|---|---|
| `rUbFactor` / `rUbFactorAlign` | 外部存储 stride、valid 元素数、partial chunk 判定 | 片上行 stride、归约指令 srcShape、片上占用预算 |
| `innerAProd` / `innerAProdAlign` | （含尾轴 A 的有效连乘） | 片上 A 维预算、aUnit 构成 |

## 3 多核切分（Base：A 方向 fused 分核）

外层 A loop 全部 fuse 成一根线性计数，多核按此瓜分：

```
aSplitChunkCnt = CeilDiv(axisShape[aSplitIdx], aUbFactor)
outerAProd     = ∏(切分轴左侧的外层 A 轴)
aLoopCntTotal  = outerAProd × aSplitChunkCnt
```

线性编码 row-major（切分 chunk 在最内）；解码顺序相反：先取 chunk，再逐轴分解。

**大小核均衡**（`MULTICORE_BALANCE` 谓词）：

```
aSmallCoreLoopCnt = aLoopCntTotal / maxCores        // floor
aBigCoreCnt       = aLoopCntTotal % maxCores
aBigCoreLoopCnt   = aSmallCoreLoopCnt + (aBigCoreCnt > 0 ? 1 : 0)
usedCoreNum       = (aSmallCoreLoopCnt > 0) ? maxCores : aBigCoreCnt
```

前 `aBigCoreCnt` 核多担 1 份；blockIdx ≥ usedCoreNum 空闲；最大负载差 ≤ 1。

**R 外层循环计数**：`rLoopCntTotal = ∏(切分轴左侧外层 R 轴) × CeilDiv(axisShape[rSplitIdx], rUbFactor)`。

**约束**：使用核数为 0 时仍按 `BLOCK_DISPATCH_MIN` 发起（≥1），kernel 侧短路；发起核数严禁超过物理核数。

## 4 Group 2D 分核

**触发判定**（`ShouldUseGroup`）：A 方向外层总数 `aOuter = aLoopCntTotal`，R 方向 `rOuter = rLoopCntTotal`：

```
if aOuter > maxCores / 2    → Base   # A 够分核
else if rOuter <= 1          → Base   # R 无并行度
else                         → Group  # A 用不满核，借 R 补并行度
```

> 阈值取 `maxCores / 2` 的原因：A×R 2D 分核时 A 切片不可再拆（片上切分后的最小调度单元）。若 `aOuter > maxCores/2`，即使 R 组数为 2 也会超物理核数，跨核栅栏会挂死。

**2D 分核**（`ComputeGroupSplit`）：

```
totalOuter = aOuter × rOuter
perCoreNum = CeilDiv(totalOuter, maxCores)
numBlocks  = CeilDiv(totalOuter, perCoreNum)
# 对齐到 aOuter —— 保 2D 网格整齐、负载均匀
if CeilAlign(numBlocks, aOuter) <= maxCores: numBlocks = CeilAlign(...)
else:                                         numBlocks = FloorAlign(...)
usedCoreNum = numBlocks
rGroupCnt   = numBlocks / aOuter     # 整除
```

**`aPerCore = 1` 恒成立**：numBlocks 对齐到 aOuter 的倍数 → 每核只处理 1 个 A chunk。

**Group workspace**（`WORKSPACE_MODEL` 谓词）：布局 `[rGroupCnt, aTotal]`、按计算精度 dense 行优先；大小 = `rGroupCnt × aTotal × 精度宽度`。总 workspace = 系统部分 + 用户部分。

**R 方向分组分配**（kernel 侧）：R 方向**大小核式均匀分配**（每组 ≥ 1 chunk、无空组，`rGroupCnt ≤ rOuter` 恒成立），⛔ 禁止 CeilDiv 截断式分配。

**调度模式**：Group 使用跨核栅栏（`CROSS_CORE_FENCE`），host 侧须按语言要求声明调度模式（等全部核空闲才启动）。

## 5 Empty 切分

**EMPTY_A**：无切分。使用核数为 0，按 `BLOCK_DISPATCH_MIN` 发起，所有核早退。

**EMPTY_R**：按 **a 元素数**切（aTotal = 合所有 A 轴后的线性元素数）：

```
maxDtypeSize = max(sizeof(D_T), 精度宽度)
四约束：
  a) 单 buf 下界（保证搬运效率）     → min_a_per_core = CeilDiv(下界字节数, maxDtypeSize)
  b) 优先多核                       → CeilDiv(aTotal, maxCores)
  c) UB 上限                        → maxBufSize = min(片上容量 / P_post, SINGLE_BUF_MAX)
                                       → max_ub_factor = maxBufSize / maxDtypeSize
  d) aTotal 兜底                    → aTotal
aUbFactor = min(max(min_a_per_core, CeilDiv(aTotal, maxCores)), max_ub_factor)
aUbFactor = min(aUbFactor, aTotal)
```

之后按 `aLoopCntTotal = CeilDiv(aTotal, aUbFactor)` 做大小核均衡（同 §3）。`postBufSize = CeilAlign(max(aUbFactor × maxDtypeSize, ALIGN_GRANULARITY), ALIGN_GRANULARITY)`（单份至少一个对齐粒度）。

## 6 UB 预算不等式与 buffer 大小公式

**buffer 大小**：

```
aUnit        = aUbFactor × innerAProdAlign
postBufSize  = CeilAlign(aUnit × maxDtypeSize, ALIGN_GRANULARITY)          # 1D，对齐
preBufSize   = aUnit × rUbFactorAlign × innerRProdAlign × maxDtypeSize     # 2D dense
树缓存        = TREE_CACHE_CAPACITY（固定）
```

**UB 预算不等式**（padded 尺寸）：

```
(P_pre + P_pre_ext) × preBufSize + P_post × postBufSize ≤ ubAvailable
其中 ubAvailable = 片上容量 − TREE_CACHE_CAPACITY
```

**树缓存硬约束**：

```
cacheCount × CeilAlign(aUnit, 最小对齐元素数) × 精度宽度 ≤ TREE_CACHE_CAPACITY
cacheCount = log2(FindNearestPower2(rLoopCntTotal)) + 1
```

计算精度下恒满足（aUnit ≤ 爬坡上界 → 层数有限）；窄 dtype 需极端 rLoopCntTotal 才违反、实际不可达，不设校验。R 全驻（cacheCount=1）由 §2.2 Step 3 的 aUnitMax 钳制保证。

## 7 切分总流程（伪码）

```
TilingFunc(ctx):
    # 0) 平台信息（全走接口，逐项非零校验）
    # ★ 空 tensor 短路（必须在合轴四步之前）
    if 空 tensor: return HandleEmptyTensor()      # §5

    # 合轴四步（顺序不可换）
    DropSizeOneAxes(); FuseAxis(); PadLeadingOneA(); PadRIfPureA()
    校验 2 ≤ axisNum ≤ MAX_PATTERN_RANK

    # 1) UB 切分三步
    ComputeAUbFactor()        # Step 1: 定 A 切分
    ComputeRUbFactor()        # Step 2: 反解 R 切分
    ExpandAIfRFullyLoaded()   # Step 3: R 全驻后扩 A

    # 2) 多核切分
    ComputeFusedALoopSplit()  # A 方向 fused 分核 + 大小核均衡
    ComputeRLoopCnt()

    # 3) Group 判定（条件性）
    if ShouldUseGroup():
        ComputeGroupSplit()   # A×R 2D 分核 → usedCoreNum, rGroupCnt
        声明调度模式

    # 4) 收尾
    ComputeUbSizes(); FillAndLogTilingData()
    设 TilingKey / BlockDim / Workspace
```
