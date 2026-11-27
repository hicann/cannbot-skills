# small-shape Tiling 模板（小任务量 M/N 均衡）

small-shape 优先改善核间均衡，并在条件成立时复用完整 K 权重驻留。它与 l1-resident 共用驻留计算，不另设模板。字段及容量公式见 [公共 Tiling](../../tiling.md)。

本模板接收 DX/Transpose 的统一问题；所有候选计入可选 bias，最终 baseN/singleCoreCin 确定后按 [bias 预算](../../tiling.md#51-可选-bias-的候选预算与实际分配) 复核 L1/BT。

## 1 选择条件

先按形状建立候选输出块面积上限 blockSizeLimit_，单位为部分和元素数，不是 L0C 容量：

| 条件 | 面积上限 |
|---|---|
| 通常情况 | `256*256` |
| 对齐 N<256、对齐 M>=256 | `min(256*256,min(对齐M,1024)*对齐N)` |
| 对齐 M<256、对齐 N>=256 | `min(256*256,对齐M*min(对齐N,1024))` |

对齐 M/N 分别为 alignUp(mValue,m0)、alignUp(nValue,n0)。预计任务数：

`expectedMaxCnt=ceilDiv(mValue*nValue,blockSizeLimit_)*N*Di*real_g`。

real_g 是计算组数，原始分组时等于 G。当完整窗口 K 不超过当前策略上限（32768）、expectedMaxCnt<thresholdCnt，且不能被可用核数整除时选择本模板。thresholdCnt 按目标策略绑定；可用 `floor(2.5*maxCores)`，或更保守的 `floor(1.5*maxCores)`。这些是均衡策略参数，不能由 L0C 字节数替代；所有候选仍独立检查真实 L0/L1 容量。

## 2 基本块与均衡搜索

先形成一个合法起点。非全载使用 standard 的 InitBaseMNK；全载使用 [l1-resident 起点](../l1-resident/tiling.md#2-基本块起点)，其范围和份数按同一目标配置调整。

候选仍按空间分段数、通道分段数递增枚举。通常 maxM/maxN 为 1024；另一方向只有一个分形时，相应上限为 512。全载候选还用 `L1Bytes//(kValue*weightBytes)` 限制 maxN。

| 方向 | 分段数范围，含两端 | 计算块候选 |
|---|---|---|
| M | `ceilDiv(Hi*Wi,maxM)…ceilDiv(Hi*Wi,64)` | `calcM=alignUp(ceilDiv(Hi*Wi,spaceParts),m0)` |
| N | `ceilDiv(nValue,maxN)…ceilDiv(nValue,64)` | `calcN=alignUp(ceilDiv(nValue,channelParts),n0)` |

taskM 初始为 calcM。若 `wholeRowM=max(calcM//Wi,1)*Wi` 与 calcM 对应的空间段数相同，候选取整行 taskM=wholeRowM、calcM=alignUp(taskM,m0)。面积须不超过 blockSizeLimit_，L0C 容量与各路操作数容量另行检查。

全载时，候选还要求 `kValue*calcN*weightBytes + dedyBytes*驻留行数(taskM)*Wo*s_w*Cout <= L1Bytes`。这只是候选初筛；最终驻留份数、bias 和真实地址使用公共检查。

评价量如下，驻留行数使用公共 CalFmapH：

| 量 | 公式 |
|---|---|
| 任务数 | `tasks=N*Di*real_g*ceilDiv(Hi*Wi,taskM)*ceilDiv(nValue,calcN)` |
| 最忙核轮数 | `rounds=ceilDiv(tasks,maxCores)` |
| 普通搬运估计 | `weightBytes*calcN*Hk*Wk*Cout + dedyBytes*驻留行数(taskM)*Wo*s_w*Cout` |
| 全载搬运估计 | `weightBytes*kValue*calcN + dedyBytes*驻留行数(taskM)*Wo*s_w*Cout` |

rounds 更小时接受；相同时，仅接受任务数不少于当前值且搬运估计不大的候选。Cout 在估计中使用总通道口径。保留最优 baseM/baseN 后执行 AdjustBaseMNK，再以最终块写 `singleCoreM=min(baseM,Hi*Wi)、singleCoreCin=baseN、singleCoreDin=1`，避免使用调整前的过期 taskM。

## 3 驻留、收益复核与回退

均衡与全载可以组合：先按 l1-resident 的几何、复用收益和容量初筛尝试完整 K 驻留；不满足时直接使用普通驻留。两种驻留分别检查资源、搬运与生命周期约束。

| 当前策略 | 复用规则 |
|---|---|
| 分块驻留 | 使用 InnerProduct 的 InitBaseMNK、CalStepK、SetSingleCoreInfo 和容量回调 |
| 完整 K 驻留 | 使用 FullLoad 的对应计算；完整 B1 的复用仍由 Kernel 检查份数、归约范围与权重身份 |

CalcAndValidateTiling 顺序为：选起点、建立 L1 参数、M/N 均衡、选择 step、容量回调，必要时重新恢复任务。基本块、L1 和任务参数作为一套结果返回。

全载合法后再检查 IsEnableFullLoad 的收益条件。已有实现对单份 B1 且 `kValue<=64` 的配置回退，避免很短 K 下单份驻留的串行开销；这一阈值是优化经验，目标验证后可调整。全载候选无法形成合法计划或收益不足时，关闭 enableFullLoadTiling，重建 L0/L1/任务并重新求解，不能只清 flag 沿用旧参数。

## 4 Host 求解顺序

下列伪码组合上述优化思想；各阶段名称沿用对应求解函数。

```text
enableFullLoadTiling = CheckFullLoadCapable()
ok, l0, l1, core = CalcAndValidateTiling()
if enableFullLoadTiling and (not ok or not IsEnableFullLoad(l1)):
    enableFullLoadTiling = false
    ok, l0, l1, core = CalcAndValidateTiling()  # 新建三份参数后重算
if not ok: return false
SetTilingCondition(core,l1,l0)
if hasBiasFlag_ and not UpdateIsBiasFullLoad(l1,l0): return false
if not CheckVectorCoreNum(): return false
SetTilingData(core,l1,l0)
```

SetTilingData 发布最终 enableFullLoadTiling；templateId 保持 small-shape。单次求解内的容量回调使用当前驻留策略，完整重算后才能从全载回退到分块驻留。

主计算共用 [compute-flow](../../../compute-flow.md)。任务起点用 singleCore*，块内循环用 base*；全载命中后沿用既有 L1B 生命周期，短任务不生成不存在的下一段预取。

## 5 检查要点

任务按 dedx 空间段互斥划分，两核并行时的 DAG 形态：

```text
核 0: dedx [0,taskM)       --> 一次完整归约 --> F
核 1: dedx [taskM,2*taskM) --> 一次完整归约 --> F

共享只读: dedy、weight
各核独立: L1A / L1B / L0A / L0B / L0C
```

- [ ] 任务起点与计算块 M 的区别（任务 M 可小于 baseM，对齐跨度按任务收缩）
- [ ] 条件重叠的优先序（与 l1-resident 同时命中时选本模板）、候选同代价裁决
- [ ] blockSizeLimit_ 与真实 L0C 容量分别检查，最终任务 M 使用 AdjustBaseMNK 后的 baseM
- [ ] 全载成功、容量失败及收益不足三种结果；回退重新求解 L0/L1/任务，发布值与 Kernel 保留条件一致
- [ ] 容量回调后不切换模板；一核多轮任务的尾块
- [ ] 公共数值及生命周期检查见 [verification](../../../verification.md)
