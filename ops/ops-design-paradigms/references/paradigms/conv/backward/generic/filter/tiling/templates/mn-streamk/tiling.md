# MN_StreamK Tiling：输出任务与尾轮 K 切分

本模板按 M/N、核深度和组划分 dedw 输出任务；主轮完成各任务的完整归约，尾轮可再切 K，交给 AIV 做确定性归约。公共选块和容量公式见 [Tiling](../../tiling.md)，主轮数据流见 [compute-flow](../../../compute-flow.md)，尾轮执行见 [Kernel §7.3](../../../../../adapters/ascendc/filter/kernel-template.md#73-尾轮-streamk-与确定性归约)。

## 1 准入与模板关系

模板按 **DLoad(8) → Winograd(9) → MN_StreamK(10)** 尝试。前序模板不满足 IsCapable 才继续；参数错误或已选模板求解失败不作为普通回退。

| 条件 | 当前约定 |
|---|---|
| 语义 | x、dedy、filter_size 对应同一正向卷积；普通权重梯度使用 output_padding=0 |
| 类型 | x/dedy 同为 FP16、BF16 或 FP32，dedw 为 FP32；FP32 的 hf32Flag 按 enable_hf32 及目标能力绑定 |
| 格式 | x/dedy 配对为 NCDHW 或 NDHWC；dedw 使用已绑定的 NCDHW、NDHWC、DHWCN 分支 |
| 分组 | 当前范式采用原始分组；数学通道、执行组跨度和补齐通道分别记录 |
| 能力 | 搬运、Load3D、MMAD、FP32 原子写回及 AIC/AIV 通知可用；MIX 配比、blockDim 和实际参与单元一致 |
| 资源 | L0/L1/UB、字段范围、物理地址与输出覆盖全部合法 |

逻辑选块保持 16 对齐；物理 channelSize/k0 随输入类型取 32/sizeof(T)，FP32 为 8。不能把 FP32 的物理通道块当作 16，也不能把 hf32 当作存储类型。

## 2 在线求解主线

```text
+----------------------------------------------------------+
| InitSplitWOI                                             |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
| MultiCoreSplitMN                                         |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
| InitBaseBlock / SetStepK4SplitMN / UpdateSingleCoreInfo  |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
| AdjustSmallCaseBaseBlock                                 |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
| DoStreamKTiling                                          |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
| DoLibApiTiling / PostTiling                              |
+----------------------------------------------------------+
```

InitSplitWOI 按普通或扩展 pad 字段选择整宽阈值；可切宽时从宽片候选开始。InitBaseBlock 按目标块长均分 M/N，N 对 3×3 窗口优先保留完整窗口关系；GetBaseK 根据两路 L0 容量推导 K。具体起点、驻留和容量回调公式统一在公共 Tiling §4–§8。

UpdateSingleCoreInfo 令 singleCoreM=baseM，singleCoreN 至少覆盖一个完整 `hk*wk*16` 逻辑通道窗口；baseN 可把任务再分为多个 L0 子块。普通窗口、拆核窗口和切宽是本模板内部的搬运分支。

## 3 小任务量调整

`totalCnt=ceilDiv(mValue,singleCoreM)*ceilDiv(nValue,singleCoreN)*dk*realGroup`。AdjustSmallCaseBaseBlock 的提前返回条件为：

- groupEnlarge 为真；当前原始分组路径不触发此项。
- totalCnt≥平台核数。
- 平台核数能整除 totalCnt，且 batch*dout 或 ho 至少有 `平台核数/totalCnt` 个可切单位。

其余情况先搜索 M 划分，再按更新后的任务数考虑 N：

| 调整 | 搜索和接受条件 |
|---|---|
| M | mCnt 从大到小；候选 M 向上对齐 16。新任务数不超过核数且整除核数；batch*dout 或候选 baseK/wo 能提供后续切分；通过 L0C 与 L1 检查 |
| N | 更新后 totalCnt 在核数一半与全核数之间，且 baseN≥baseM；nCnt 从大到小，singleCoreN 按完整 `hk*wk*16` 窗口对齐，再检查容量 |

M 搜索可能增大 baseM，**256 是初始选块目标，不是所有最终块的上限**。修改 M/N 后重算 baseK、L0C 份数、stepKa/Kb、Bound 与 singleCore*。

## 4 尾轮 StreamK 选择

令 `coreNum=平台计算核数`，尾块数为 `totalCnt%coreNum`。仅尾块非零、不超过 coreNum/2，且确定性输出格式可用时尝试切 K；否则 streamkType=0。

格式检查使用 3D 格式名：x/dedy 为 NCDHW 或 NDHWC；dedw 为 NCDHW，或在 `dk=di=dout=1` 的退化情形下使用 NDHWC/DHWCN。其他输出组合保留主轮实现并回退 streamkType=0。

| streamkType | 切分范围 | 发布值 |
|---|---|---|
| 0 / NO_STREAMK_CALC | 无跨核 K 切分 | singleCoreBatchDout=batch*dout，singleCoreK=ho*wo |
| 1 / STREAMK_BATCHDOUT | batch×dout | singleCoreBatchDout=ceilDiv(batch*dout,coreStreamK) |
| 2 / STREAMK_HWOUT | 只切 H，W 完整 | singleCoreK=ceilDiv(ho,coreStreamK)*wo，singleCoreBatchDout=batch*dout |

初始 `coreStreamK=coreNum/尾块数`。实际分核数由向上取整后的范围重算；只分出一个核时不启用。方向选择按源码的 IsSplitBatchDoutBetter：

```cpp
B = batch * dout;
b = ceilDiv(B, coreStreamK);
bdDim = ceilDiv(B, b);
hwK = ceilDiv(ho, coreStreamK) * wo;
hwDim = ceilDiv(singleCoreK, hwK);
if (bdDim < hwDim) {
    return !isSplitKernelHW && bdDim > 1 &&
           bdMte2Count < hwMte2Count;  // 严格小于，次数相同仍选 HWout
}
if (B % b > 0 && double(B % b) / b < double(singleCoreK % hwK) / hwK) {
    return false;
}
return true;                           // true 选 BatchDout
```

此处尾长比例比较也适用于 bdDim>hwDim，不能简写成“仅分核数相等时比较”。MTE2 估计为：

`batchDoutPerCore*ceilDiv(wo,splitWo)*hoSegs*(ceilDiv(kIterPerSeg,stepKa)+ceilDiv(kIterPerSeg,stepKb))`。

其中 `reduceSegmentsH=min(ceilDiv(65536,wo),singleCoreHo)`、`hoSegs=ceilDiv(singleCoreHo,reduceSegmentsH)`、`kIterPerSeg=ceilDiv(reduceSegmentsH*wo,baseK)`；65536 是累加分段阈值，不是 buffer 容量。HWout 候选先按 hwK 收缩 baseK 和 step，再代入比较。

HWout 选定后重查 L1；不合法时，仅在 stepKa、stepKb 都大于 1 时同步递减。仍无合法组合则恢复原 singleCoreK/baseK/stepKa/stepKb、streamkType=0、coreStreamK=0。singleShapeK 是 Host 临时量，最终通过 singleCoreK 发布。

## 5 主轮、尾轮与启动核数

coreNum 是共享 TilingData 中的调度基数；usedCoreNum、coreStreamK、coreBindDirection 是 **Host 中间量**，不作为同名字段序列化。

| 项目 | 计算 |
|---|---|
| usedCoreNum / blockDim | `min(coreNum, streamkType==0 ? totalCnt : coreStreamK*totalCnt)` |
| 主轮数 calRound | type=0：ceilDiv(totalCnt,coreNum)；type≠0：totalCnt/coreNum |
| 主轮轮转 | basicBlockIdx 从 block_idx 起，每轮加 coreNum |
| 尾轮分核数 | `deterAddCoreNum=ceilDiv(batch*dout,singleCoreBatchDout)*ceilDiv(ho*wo,singleCoreK)` |

即使 blockDim 小于 coreNum，Kernel 仍用 coreNum 计算主轮和尾轮坐标。尾轮把同一输出块分给 deterAddCoreNum 个连续调度位置；每位只归约自己的 batch×dout 或 H 段，不能再次执行完整 K。

## 6 确定性归约与 workspace

主轮的多个精度段在同一核按顺序写回。尾轮多核部分和先写 workspace，AIV 按固定二叉树次序归约，再写对应 dedw 块。其协议与 Winograd 的尾轮交织写回分别实现：

```text
AIV                                  AIC
ClearWorkspace (Cube 区)
    |
VecNotifyCube ---------------------> CubeWaitVector
                                         |
                                  计算本核 K 段
                                         |
                              GetTensorC(workspace,1,true)
                              已清零槽内做原子累加
                                         |
VecWaitCube <----------------------- CubeNotifyVector
    |
DeterministicReduceKInUb
    |
DeterministicAddFunc：1,2,4,... 树形归约
    |
UBRearrange2Gm -> dedw
    |
下一子块清零前完成上一子块的消费
```

用户 workspace 预算为 `2*coreNum*平台L0CBytes`，分别容纳 Cube 局部结果和 Vector 中间归约。每核槽跨度为 `CUBE_WORKSPACE=平台L0CBytes/sizeof(float)` 个 FP32 元素；设备 Vector 区起点使用 `GetBlockNum()*CUBE_WORKSPACE`，不能直接替换成共享 coreNum。绑定目标 MIX 核类型后检查实际分区、各槽最大地址和 Host 预算，地址按尾轮输出块、组内核号及子块推导，不套用 Winograd 的 `(tailBlockId,kGroupIdx)` 公式。

AIV 清零后经 MODE0 汇合，再发 MODE2 通知 AIC；AIC 结果就绪后也先汇合再通知 AIV。无有效计算的尾核保留通知配对。VecUb2Gm 写中间归约区，最终 dedw 写回由 UBRearrange2Gm 完成。具体事件、槽偏移和无任务核规则见适配层。

## 7 发布与初始化

| 项目 | 约定 |
|---|---|
| key | `GET_TPL_TILING_KEY(TPL_STREAM_K,isSplitKernelHW,0,0,0)`；TPL_MN_STREAM_K 为保留编号，不由当前 Host 发出 |
| 字段 | coreNum、streamkType、singleCore*、base*、step*、Bound、Pbuffer、splitWo、hf32Flag 等共享字段 |
| 初始化 | AIV 先清零整个 FP32 dedw，通知 AIC；初始化参与者与 MIX 配比、启动核数相匹配 |
| 主计算 | type=0 时 AIV 不做主轮数值工作；type≠0 时 AIC/AIV 继续同一次 Process，AIV 承担尾轮清零与归约 |
| workspace | type=0 用户区为 0；type≠0 按 §6；系统区与用户区分别记录 |
| 调度 | 使用 MODE0 汇合时按 SDK 配置 SetScheduleMode(1) 及匹配的 MIX 实例 |

在线求解顺序为 `InitSplitWOI → MultiCoreSplitMN → AdjustSmallCaseBaseBlock → DoStreamKTiling → DoLibApiTiling/PostTiling`。所有回调完成后再校验并发布；不读取调优数据仓库。

## 8 产出校验

- [ ] 普通/拆核窗口、切宽与三态 StreamK 的字段来自同一最终计划。
- [ ] 小任务量调整的提前返回条件、M/N 搜索和 L0/L1 回调一致。
- [ ] FP32 存储、hf32 计算模式、逻辑 16 对齐与物理 channelSize 分开记录。
- [ ] coreNum 与启动 blockDim 区分，Host 临时量未误写为共享字段。
- [ ] BatchDout/HWout 分段并集覆盖全部有效 K，交集为空；回退恢复全部关联字段。
- [ ] Cube 清零、原子写局部和、Vector 树形归约与最终写回顺序完整。
- [ ] 无贡献、尾核、非整窗及不同输出格式均检查实际搬运与归约范围。
- [ ] 规则复算、Host 求解和设备正确性分别验收，不把文档推演作为设备验证。
