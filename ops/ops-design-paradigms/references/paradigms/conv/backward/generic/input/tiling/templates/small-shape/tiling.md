# small-shape Tiling 模板（小任务量 M/N 均衡）

small-shape 对应 SmallShape。小任务量下优先改善核间均衡，联合确定任务 M/N 和计算块。共用计算见 [公共 Tiling](../../tiling.md)。

本模板也接收 Transpose 的统一问题。每次 L1 候选检查计入可选 bias，最终 baseN/singleCoreCin 确定后按 [公共 bias 预算](../../tiling.md#51-可选-bias-的候选预算与实际分配) 复核 L1/BT 实际范围，再发布字段。

## 1 选择条件

通过公共规格及适配层检查后，按一份 L0C 估算任务量：

`estimatedTasks=ceilDiv(mValue*nValue*accBytes,L0CBytes)*N*Di*G`。

当 `kValue<=32768`、`estimatedTasks<maxCores*5//2`，且 `estimatedTasks` 不能被 `maxCores` 整除时，进入本模板。K 上限是当前求解策略的入口条件，最终配置仍须通过公共准入检查。估计值用于入口判断，最终任务数由搜索结果确定；与 L1 驻留初筛同时成立时，优先选择本模板。

## 2 基本块与均衡搜索

`InitBaseMNK` 使用核体积为 1 的 `(256,256,64)` 或其余情况的 `(512,128,32)`，两路输入初始双 Buffer，部分和初始单 Buffer；按公共 `AdjustBaseMNK` 调整后进入搜索。

在初始候选上枚举 M/N 分段数。`nValue<=n0` 时 `maxM=512`，否则为 1024；`mValue<=m0` 时 `maxN=512`，否则为 1024。

| 方向 | 分段数范围，含两端 | 计算块候选 |
|---|---|---|
| 空间 M | `spaceParts=ceilDiv(Hi*Wi,maxM)…ceilDiv(Hi*Wi,64)` | `calcM=alignUp(ceilDiv(Hi*Wi,spaceParts),m0)` |
| 通道 N | `channelParts=ceilDiv(nValue,maxN)…ceilDiv(nValue,64)` | `calcN=alignUp(ceilDiv(nValue,channelParts),n0)` |

任务空间长度 `taskM` 初始为 `calcM`。整行候选 `wholeRowM=max(calcM//Wi,1)*Wi`；若它与 `calcM` 对应的空间段数相同，则取 `taskM=wholeRowM`、`calcM=alignUp(wholeRowM,m0)`。候选须满足单份部分和容量 `calcM*calcN*accBytes<=L0CBytes`。

评价量如下，`驻留行数(taskM)` 按 [公共 L1 容量](../../tiling.md#5-l1-驻留容量) 计算：

| 评价量 | 计算公式 |
|---|---|
| 任务数 | `tasks=N*Di*G*ceilDiv(Hi*Wi,taskM)*ceilDiv(nValue,calcN)` |
| 最忙核轮数 | `rounds=ceilDiv(tasks,maxCores)` |
| 单任务搬运估计 | `loadBytes=weightBytes*calcN*Hk*Wk*Cout + dedyBytes*驻留行数(taskM)*(Wo*s_w)*Cout` |

`AdjustSingleCoreAndL0Info` 以当前 `baseM/baseN` 为初始最优候选，空间分段数在外层、通道分段数在内层，均按递增顺序比较：

1. `rounds` 更小时接受。
2. `rounds` 相同时，仅当任务数不少于当前最优值、且 `loadBytes` 不大于当前最优值时接受。

接受候选时同步记录 `baseM=calcM`、`baseN=calcN`、`singleCoreM=taskM` 及三个评价量。搜索完成后执行 `AdjustBaseMNK()`，再取 `singleCoreCin=baseN`、`singleCoreCout=CoutG`、`singleCoreDin=1`。

搬运估计中的 `Cout` 是总输出通道数，用于同一输入的候选比较。任务 M 与计算块 M 可以不同。

## 3 L1 驻留与任务恢复

`InitL1Params` 在均衡搜索前建立初始 L1 参数；搜索完成后，`CalStepK` 使用公共分块驻留规则，在最终 baseM/baseN/baseK 上选择 stepKa/stepKb。

若 `IsL1ParamsValid` 失败，执行公共 `LegalProtection`。成功后调用与 InnerProduct 相同的 `SetSingleCoreInfo` 恢复任务，保留 `templateId=small-shape`。核体积为 1 时，N/M 调整阈值为 256/256，其余为 128/512；容量回调不会重新进入另一个 Tiling 模板。

## 4 Host 求解顺序

```text
InitBaseMNK(l0)
InitL1Params(l1,l0)
AdjustSingleCoreAndL0Info(core,l0)
CalStepK(l1,l0)
if not IsL1ParamsValid(l1,l0):
    LegalProtection(l1,l0)
    if not IsL1ParamsValid(l1,l0): return false
    SetSingleCoreInfo(core,l0)
SetTilingCondition(core,l1,l0)
if hasBiasFlag_ and not UpdateIsBiasFullLoad(l1,l0): return false
SetTilingData(core,l1,l0)
```

`AdjustSingleCoreAndL0Info` 对应第 2 节的候选搜索，`SetSingleCoreInfo` 的恢复公式见 [公共任务恢复](../../tiling.md#8-setsinglecoreinfo公共任务恢复)。最终 `singleCoreDin=1、singleCoreCout=coutG、enableFullLoad=0`；其他字段使用搜索和容量调整结果。

主计算共用 [compute-flow](../../../compute-flow.md)，任务起点使用 singleCore*，核内循环使用 base*。短任务不生成不存在的下一段预取；每核的 Buffer 独立。

## 5 检查要点

任务按 dedx 空间段互斥划分，两核并行时的 DAG 形态：

```text
core 0: dedx [0,taskM)        --> one full reduction --> F
core 1: dedx [taskM,2*taskM)  --> one full reduction --> F

Shared read-only: dedy, weight
Per core: L1A / L1B / L0A / L0B / L0C
```

- [ ] 任务起点与计算块 M 的区别（任务 M 可小于 baseM，对齐跨度按任务收缩）
- [ ] 条件重叠的优先序（与 l1-resident 同时命中时选本模板）、候选同代价裁决
- [ ] 容量回调后不切换模板；一核多轮任务的尾块
- [ ] 公共数值及生命周期检查见 [verification](../../../verification.md)
