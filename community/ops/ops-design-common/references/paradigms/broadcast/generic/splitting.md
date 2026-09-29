# Broadcast 范式切分算法（片上内存单切分 + 多核均衡切分）

> **层级**：通用知识层（generic/），语言中立。
>
> 来源：adapters/ascendc/tiling.md §1/§2/§3/§6（原 broadcast-standard-tiling.md）。C++ 实现（FindSplitAxis / MultiCoreSplit）、平台信息获取与注册流程在适配层。

## 1 记号与输入

| 记号 | 含义 |
|------|------|
| `maximumBroShape` | 切分坐标系（见 [shape-preprocessing.md](shape-preprocessing.md)） |
| 片上容量 | 单核算可用的片上 buffer 总容量（AscendC: UB 大小） |
| `P` | 物理存活节点数（见 [liveness-fusion.md](liveness-fusion.md)），决定 buffer 槽位数 |
| `ALIGN_GRANULARITY` | 片上 buffer 对齐粒度（能力谓词） |
| `COMPUTE_PRECISION` | 范式统一计算精度（能力谓词），perBufElems 按其字节宽算 |
| `maxCores` | 可用核数 |

## 2 片上内存单切分

### 2.1 切分原理

以 effective_shape 为坐标系选择**单根切分轴**。

被切分的轴 a = ubOuter × ubFactor。包含 ubFactor 在内的所有内侧轴为「内轴」，一次加载进片上；ubOuter 至最外侧轴为「外轴」，是片上外的 for 循环。

```
内轴 = 轴 k (贡献 ubFactor) + 轴 k+1..n-1 (全量)
外轴 = 轴 0..k-1 (全量循环) × ubOuter (轴 k 的分段循环)

片上内存占用 = ubFactor × ∏_{j>k} d_j
必须满足: ubFactor × inner ≤ perBufElems
```

**通用约束**：
- 每份 buffer 大小需按 `ALIGN_GRANULARITY` 对齐
- 所有 buffer 总和不能超过实际片上容量
- 必须考虑空 tensor 场景：**使用核数为 0 时仍按 1 核发起**，kernel 侧通过 usedCoreNum=0 短路处理
- perBufElems 必须满足内轴需求：`perBufElems ≥ effective_shape[ubSplitIdx]` 对应的内轴连乘

### 2.2 选轴算法

从最内轴向外扫描，累积内侧轴连乘 inner。当 `d_k × inner > perBufElems` 时，该轴无法全量放入——在此切分。

```
inner = 1
for k from n-1 down to 0:
    if d_k × inner > perBufElems:
        split at k
        ubFactor = perBufElems / inner
        ubOuter = ceil(d_k / ubFactor)
        break
    inner *= d_k
```

永远至少有一根可切轴。

**尾块**：d_k 不能整除 ubFactor 时，最后一轮 ubTail = d_k % ubFactor（整除时为 ubFactor）。段大小查询：`ubBlockLength = (ubBlockIdx == ubOuter - 1) ? ubTail : ubFactor`。

### 2.3 特殊场景

- 末尾轴装不下：`d_{n-1} > perBufElems` → ubSplitIdx = n-1, ubFactor = perBufElems
- 全量装得下：`total ≤ perBufElems` → ubSplitIdx = 0, ubFactor = d_0, ubOuter = 1
- 空 tensor：`numel == 0` 直接返回，不走 kernel（但仍按 1 核发起）

### 2.4 buffer 预算公式

- `perBufBytes = floor(片上容量 / P)` 按 `ALIGN_GRANULARITY` 向下对齐
- `perBufElems = perBufBytes / sizeof(COMPUTE_PRECISION)`——**永远按统一计算精度算**，不随输入 dtype 变化（原因：低精度输入搬入后立即转换到计算精度，真正填满 buffer 的是计算精度数据；即使输入本身就是计算精度，也按同一公式保持 TilingData 统一）

## 3 多核均衡切分

以片上切分产出的 tile 为单元，将 `totalTiles = ubOuter × Π_{j<k} d_j` 平摊到多核。

**通用约束**：
- 发起核数严禁为 0（空 tensor 场景见 §2.3）
- 发起核数严禁超过实际物理核数
- 使用核间均衡策略，字段约定：

| 字段 | 含义 |
|------|------|
| totalTiles | 用于切多核的总 tile 块数 |
| coreNum | 总核数 |
| usedCoreNum | 实际使用的核数 |
| mainTiles | 主核分配的 tile 块数（尾核分配的 tile 块数 = mainTiles − 1） |
| mainCoreNum | 主核数（尾核数 = usedCoreNum − mainCoreNum） |

**切分公式**：

```
usedCoreNum = (totalTiles < maxCores) ? totalTiles : maxCores
mainTiles   = ceil(totalTiles / usedCoreNum)
mainCoreNum = totalTiles - (mainTiles - 1) × usedCoreNum
```

前 `mainCoreNum` 个核各处理 `mainTiles` 块，其余核各处理 `mainTiles−1` 块。

**核间 tile 区间解码**（kernel 侧）：

```
GetCoreRange(coreId, mainTiles, mainCoreNum) → [start, end):
    if coreId < mainCoreNum:
        start = coreId × mainTiles
        end   = start + mainTiles
    else:
        start = mainCoreNum × mainTiles + (coreId − mainCoreNum) × (mainTiles − 1)
        end   = start + mainTiles − 1
```

## 4 坐标解码与偏移计算（通用算法）

- **flat → 坐标**：`ubBlockIdx = flat % ubOuter`（split 轴外层偏移），`outer = flat / ubOuter`（split 轴以外各维的 flat 索引），从 ubSplitIdx−1 向外逐维取模解码；`coord[ubSplitIdx] = ubBlockIdx × ubFactor`
- **坐标 → GM 偏移**：`offset = Σ coord[d] × strides[d]`，**返回元素个数**（非字节）；B 轴 stride=0 实现随路广播，padding 轴 stride=0 不贡献偏移
- **搬运量计算**：split 轴为广播轴（该张量 split 轴大小 == 1）时 split 段只搬 1 个元素（由搬运单元随路展开）；否则搬 ubBlockLength 个元素；总量 = split 段 × 内轴连乘

## 5 切分策略微调

不涉及。当前单切分以 effective_shape 为坐标系，功能全覆盖。部分场景非最优（多数入参远小于 effective_shape 时 padding 占比高），预留多切分扩展点。

## 6 产出校验 golden

**对范式开发者的要求**：
1. 给一个 case，枚举所有切分可能性，作为 golden；需要"人"保证正确性
2. 删除 golden，agent 读取本文档可以产出所有的切分可能；产出不了则不断改进范式质量，直到成功
3. golden 随本文档上库

**golden 样例（Adam: fp32, P=5, 片上容量=262144B, 对齐粒度=32B, coreNum=24）**：

输入 shape 全同 `(256, 128, 64)`，归一化后 `maximumBroShape=(256,128,64)`, rank=3 → 落入低档 RANK=4。

| 参数 | 值 | 计算 |
|------|-----|------|
| perBufBytes | 52416 | `(262144 / 5) & ~31` |
| perBufElems | 13104 | `52416 / 4`（按 4 字节计算精度） |
| split.ubSplitIdx | 1 → 修正为 0 | d=2(64): 64×1=64 ≤ 13104 ✓; d=1(128): 128×64=8192 ≤ 13104 ✓; d=0(256): 256×8192 > 13104 ✗ → split at 0 |

修正：split.ubSplitIdx=0（d=0 无法全量放入），ubFactor=13104/8192=1（取整后=1），ubOuter=ceil(256/1)=256。

| 参数 | 值 |
|------|-----|
| split.ubSplitIdx | 0 |
| split.ubFactor | 1 |
| split.ubOuter | 256 |
| split.ubTail | 1 |
| totalTiles | 256 |
| usedCoreNum | 24 |
| mainTiles | 11 (ceil(256/24)=11) |
| mainCoreNum | 16 (256−(11−1)×24=16) |

前 16 个核各处理 11 个 tile，后 8 个核各处理 10 个 tile。

## 7 适配层落地

| 事项 | 位置 |
|------|------|
| C++ 实现（FindSplitAxis / MultiCoreSplit / GetCoreRange 等） | adapters/ascendc/tiling.md、kernel-helpers.md |
| 平台信息获取（核数/片上容量，零值守卫） | adapters/ascendc/tiling-preprocess.md §1、tiling.md §8 |
| 超过搬运维数上限（`TRANSFER_MAX_DIMS`）的逐段搬运分叉 | 各语言适配层 kernel 文档（AscendC: NDDMA outer iters） |
| TilingKey / TilingData 写入与维测日志 | 各语言适配层 tiling 文档 |
