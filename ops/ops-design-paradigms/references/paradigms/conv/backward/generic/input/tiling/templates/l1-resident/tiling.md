# l1-resident Tiling 模板（完整 K 权重驻留）

l1-resident 对应 FullLoad。当前 dedx 通道块的完整权重归约范围驻留 L1B，跨空间块复用；L1A 按窗口更新，L0B 每轮准备当前 K。共用计算见 [公共 Tiling](../../tiling.md)。

本模板也接收 Transpose 的统一问题。每次 L1 候选检查计入可选 bias，最终 baseN/singleCoreCin 确定后按 [公共 bias 预算](../../tiling.md#51-可选-bias-的候选预算与实际分配) 复核 L1/BT 实际范围，再发布字段。

## 1 选择条件

通过公共规格及适配层检查后，同时满足：

- 几何条件：`Dk=1、mValue>nValue、Wi<=512`。
- 收益条件：`mValue*nValue < (mValue+nValue)*512` 或 `kValue<=128`，分别用于判断减少搬入量或掩盖写回的收益。
- 容量初筛：`kValue*min(nPreferred,nValue)*weightBytes*2 <= L1Bytes`，为优选通道块预留权重容量。

`nPreferred` 在核体积为 1 时取 256，其余取 128。此值用于容量初筛；第 2 节按完整 K 长度选择实际起点。

初筛使用公共约定中的对齐总长度。选定基本块后，再核算两路完整 L1 预算；与小任务量条件同时成立时，选择入口先进入小任务量均衡模板。

## 2 基本块起点

完整 K 越长，N 起点越小，为 L1B 驻留留出容量：

| `kValue` | `baseN` 起点 |
|---|---|
| 大于 1024 | 64 |
| 513～1024 | 128 |
| 257～512 | 256 |
| 不大于 256 | 512 |

两路操作数取 `al0Pbuffer=bl0Pbuffer=2`。当 `kValue<=128` 且（`nValue>=2*kValue` 或 `nValue>=128`）时，取 `cl0Pbuffer=2`，其余取 1。

| 字段 | 起点计算 |
|---|---|
| `baseM` | `min(L0CBytes//(baseN*accBytes*cl0Pbuffer),L0ABytes//(C0*dedyBytes*al0Pbuffer))` |
| `baseK` | `L0BBytes//(baseN*weightBytes*bl0Pbuffer)` |

M 同时受部分和容量、最小 K 下的操作数容量限制；K 由权重操作数容量反推。Host 的 `InitBaseMNK` 计算上述初值后，执行公共 `AdjustBaseMNK()`。

## 3 与驻留配合的任务划分

`SetSingleCoreInfo` 先计算 `channelParts=ceilDiv(nValue,baseN)`，再取 `baseN=alignUp(nValue//channelParts,n0)`。任务通道字段为 `singleCoreCin=baseN`、`singleCoreCout=CoutG`，空间长度初始为 `singleCoreM=baseM`。

**整行任务数能被核数整除时**，即 `(N*Di*G*channelParts*Hi) % maxCores=0`，按公因子直接划分：

| 量 | 计算公式 |
|---|---|
| 剩余并行因子 | `remainingFactor=maxCores//gcd(N*G*channelParts,maxCores)` |
| 深度分段因子 | `depthFactor=gcd(Di,remainingFactor)` |
| 任务深度数 | `singleCoreDin=Di//depthFactor` |
| 任务空间长度 | `singleCoreM=(Hi*depthFactor//remainingFactor)*Wi` |

`gcd` 表示最大公因数。

**其余情况**取 `singleCoreDin=1`，按递增分段数 `parts=1…ceilDiv(Hi*Wi,baseM)` 搜索整行候选 `candidateM=ceilDiv(Hi,parts)*Wi`。评价量为：

| 量 | 计算公式 |
|---|---|
| 任务数 | `tasks(taskM)=N*Di*G*channelParts*ceilDiv(Hi*Wi,taskM)` |
| 完整轮次的任务槽位 | `slots=alignUp(tasks,maxCores)` |
| 任务代价 | `cost(taskM)=ceilDiv(tasks(taskM),maxCores)*(1+ceilDiv(taskM,baseM))` |

候选先满足 `tasks*5>=slots*4`，即槽位占用至少 80%；再比较“最忙核轮数 ×（1 + 每任务 M 计算块数）”。以 `taskM=baseM` 的代价为起点，严格下降才替换 `singleCoreM`，无收益时保留原划分。

若最终 `singleCoreM<baseM`，取 `baseM=alignUp(singleCoreM,m0)`。划分后统一执行 `AdjustBaseMNK()`。任务可以包含多个空间计算块或深度；坐标与尾块按共同计算流程恢复，任务切换检查权重身份。

## 4 完整 K 驻留

权重覆盖当前组、当前 dedx 通道块的完整 K 范围，跨对应空间块复用；dedy 按窗口分段装载。初始取 `al1Pbuffer=2、bl1Pbuffer=1`。

权重驻留长度由 `stepKb=ceilDiv(kValue,baseK)` 确定，预留 `stepKb*baseK` 个归约位置。`LadderMatchStepKWithFullLoad()` 从 `stepKa=stepKb` 开始逐次减 1，取首个同时满足 L1 合法性和 `stepKb % stepKa=0` 的值；到 1 时停止并检查结果。

容量回调中的 `LadderMatchStepKWithFullLoad()` 使用相同规则，但保留回调后的 buffer 份数。最终要求 step 为正、两路容量合法、窗口配对成立、`stepKa` 整除 `stepKb`，且 `stepKb*baseK>=kValue`。

有效 K 与预留长度分别记录。例如 `kValue=320、baseK=240` 时，`stepKb=2`，预留 480 个位置，实际计算仍只覆盖 320 个位置。

容量检查使用公共 `IsL1ParamsValid`。容量不足时按 `LegalProtection` 回调；其中重选 step 的实现绑定本模板的完整 K 搜索，保留调整后的 Buffer 份数。回调成功后重新划分任务，再更新完整 K 的 step。

## 5 Host 求解顺序

```text
InitBaseMNK(l0)
InitL1Params(l1,l0)
SetSingleCoreInfo(core,l0)
CalStepK(l1,l0)
if not IsL1ParamsValid(l1,l0):
    LegalProtection(l1,l0)
    if not IsL1ParamsValid(l1,l0): return false
    SetSingleCoreInfo(core,l0)
    LadderMatchStepKWithFullLoad(l1,l0)
SetTilingCondition(core,l1,l0)
if hasBiasFlag_ and not UpdateIsBiasFullLoad(l1,l0): return false
SetTilingData(core,l1,l0)
```

最终 `enableFullLoad=1`，完整驻留仍由 Kernel 实际范围与份数判定。组、dedx 通道或权重地址变化时，按 [L1B 生命周期](../../../compute-flow.md#84-三个模板的差异) 结束复用。每个输出块的 L0C 独立初始化。

## 6 检查要点

完整 K 驻留的 DAG 形态（一次 L1B 装载跨任务复用）：

```text
core 0: one full L1B load (L1B_R)
  |
  +--> task 0: L0B[0,kA) -> L0B[kA,2kA) -> dedx
  |
  +--> task 1: L0B[0,kA) -> L0B[kA,2kA) -> dedx
  |
  +--> task 2: L0B[0,kA) -> L0B[kA,2kA) -> dedx
  |
  +--> task N: L0B[0,kA) -> L0B[kA,2kA) -> dedx
  |
  v
last MTE1 read completes -> release L1B
```

- [ ] 有效 K 项、含通道补零的计算 K、驻留预留 K 三种长度分别记录，容量按预留、贡献按有效
- [ ] 首次/末次消费者、权重身份切换、组边界；容量回调后的驻留条件
- [ ] L1B 预留长度与末段实际父跨度分别使用，槽基址与预留容量保持原分配
- [ ] 精度与实际复用量分别按 [verification](../../../verification.md) 检查
