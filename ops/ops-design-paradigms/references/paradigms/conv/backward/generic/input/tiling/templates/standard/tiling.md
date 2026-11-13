# standard Tiling 模板（经验选块与任务调整）

standard 对应 InnerProduct 的基本计算路径，先选 L0 基本块，再划分输出任务和 L1 驻留。字段及共用公式见 [公共 Tiling](../../tiling.md)。

本模板也接收 Transpose 的统一问题。每次 L1 候选检查计入可选 bias，最终 baseN/singleCoreCin 确定后按 [公共 bias 预算](../../tiling.md#51-可选-bias-的候选预算与实际分配) 复核 L1/BT 实际范围，再发布字段。

## 1 选择与优选起点

在小任务量和 L1 驻留模板未命中、且当前执行配置合法时选择 standard。核体积为 1 的起点为 `(256,256,64)`；其余为 `(512,128,32)`。初始 `al0Pbuffer=bl0Pbuffer=2、cl0Pbuffer=1`。

核体积为 1 时选较均衡 M/N，增加计算访存比；其余用较大 M 分摊权重转换成本。`InitBaseMNK` 设置起点，`AdjustBaseMNK` 按形状、L0 容量、窗口配对和部分和份数调整。经验起点不代表最终最优值。

## 2 输出任务划分

`SetSingleCoreInfo` 使用 [公共任务恢复公式](../../tiling.md#8-setsinglecoreinfo公共任务恢复)：先估算 M/N 段数，再调整通道均分和整行空间长度。核体积为 1 时 N/M 阈值为 256/256，其余为 128/512。任务 M 小于计算 M 时收缩 baseM，最后重算任务计数和尾块。

### 2.1 一维退化问题的 M 均衡

`ApplyBalancedMSplit` 在已有方案上进一步调整任务 M，当前场景同时满足：

| 条件 | 取值 |
|---|---|
| 输入与通道 | batch=1；weight、dedy、dedx 均为 BF16、NCDHW；groups=1（原始分组寻址） |
| 严格一维几何 | di=dout=dk=strideD=dilationD=1；hi=ho=hk=strideH=dilationH=1；D/H 两侧反向边界均为 0；wi>0 |
| 计算配置 | 原始分组寻址（groupConvMode=0）、GM→L1 权重搬入（loadB1Condition=0）、普通完整核窗口；L0B 为转置加逆序 |
| 已有计划 | standard 已完成求解，singleCoreM、singleCoreCin、baseM 均为正；沿用本范式的运行时配置 |
| 能力 | 适配层确认当前平台与 SDK 可沿用该 M 划分及行宽寻址，见 [能力判定](../../../../../adapters/ascendc/buffer-and-types.md#13-能力判定) |

此时 M 沿宽度展开。能力尚未确认时保留已有 standard 计划；这项均衡调整不改变计算语义或块内布局。

令 `parallelWork=N*G*ceilDiv(nValue,singleCoreCin)`。某个任务长度 `taskM` 的代价为：

`cost(taskM)=ceilDiv(parallelWork*ceilDiv(Wi,taskM),maxCores)*ceilDiv(taskM,baseM)`。

该值表示“最忙核轮数 × 每任务 M 计算块数”。`调整一维任务M()` 从当前 `singleCoreM` 的代价出发，依次枚举分段数 `parts=2…min(ceilDiv(Wi,baseM),maxCores)`，候选为 `candidateM=alignUp(ceilDiv(Wi,parts),m0)`。

仅保留 `baseM<=candidateM<Wi` 的候选，代价严格下降才更新 `singleCoreM`。计算块和驻留方案保持不变；未找到更优候选时保留原任务划分。

## 3 L1 驻留与容量回调

`InitL1Params` 从 `al1Pbuffer=bl1Pbuffer=2` 开始；`CalStepK` 比较等分容量和阶梯候选，按 [公共驻留规则](../../tiling.md#6-calstepk分块驻留的候选) 确定 stepKa/stepKb。

`IsL1ParamsValid` 同时检查窗口配对、两路预算和搬运范围。失败时执行 `LegalProtection`：先缩 K，再缩 M/N，再按份数顺序调整。容量回调成功后重新执行 `SetSingleCoreInfo`，保留当前模板，复核最终资源和任务。

## 4 Host 求解顺序

```text
InitBaseMNK(l0)
if not InitL1Params(l1,l0): return false
SetSingleCoreInfo(core,l0)
CalStepK(l1,l0)
if not IsL1ParamsValid(l1,l0):
    LegalProtection(l1,l0)
    if not IsL1ParamsValid(l1,l0): return false
    SetSingleCoreInfo(core,l0)
SetTilingCondition(core,l1,l0)
ApplyBalancedMSplit(core,l0)
if hasBiasFlag_ and not UpdateIsBiasFullLoad(l1,l0): return false
SetTilingData(core,l1,l0)
```

以上函数表示本工程需实现的求解步骤，计算分别见第 1～3 节和 [公共 Tiling](../../tiling.md)。`SetTilingCondition` 在最终 baseN 上选择 loadB2Condition。最后的 M 均衡仅改变 M，发布时仍复核最终计划；`enableFullLoad=0`。

任务解码、尾块和槽生命周期使用 [共同计算流程](../../../compute-flow.md)。每个任务完成本区域全部 dedy 通道与有效核深度归约，任务之间按 dedx 区域分工。

## 5 检查要点

- [ ] 形状/容量截断与经验起点的关系：较小形状下最终优选块由有效范围决定，经验起点只引导搜索
- [ ] 整行任务与空间尾部：任务 M 与计算块 M 的区别、尾块按剩余量收缩
- [ ] M 均衡条件（一维退化问题）与容量临界点
- [ ] 枚举任务与归约贡献，确认输出覆盖与组隔离
- [ ] 数值和实际路径按 [验证方案](../../../verification.md) 验收
