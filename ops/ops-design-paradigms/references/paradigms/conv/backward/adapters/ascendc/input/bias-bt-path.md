# bias 通路落地（BT）

bias 为 FP32、长度 Cin，沿输出通道广播。公共 [资源](../buffer-and-types.md)、[Tiling](../../../generic/input/tiling/tiling.md) 和 [同步](sync-implementation.md) 已定义容量及事件；本文件给出 bias 装载序列、MMAD 初值参数与生命周期规则；仅转置卷积使用。

## 1 能力、对象与寻址

目标 SDK 须支持 GM→L1 的 FP32 DataCopyPad、L1→BT 的 DataCopy、带 BT 的 Mmad，以及相应队列依赖。BTBytes 由目标能力记录取得；平台查询不提供该容量时，根据同一 SDK 的目标能力定义绑定并验证。

```cpp
TQue<TPosition::A1, 1> biasL1Que_;
TQue<TPosition::C2, 1> biasBTQue_;
```

两队列各一份，布局为通道连续的 FP32。`biasBytes=4`；实际分配长度为 `alignUp((isBiasFullLoad ? singleCoreCin : baseN)*biasBytes,64)`，分别检查 L1 和 BT。当前任务尾通道只搬有效值。

| 模式 | GM 元素起点 | 有效通道数 | 当前 MMAD 的 BT 元素偏移 |
|---|---|---|---|
| isBiasFullLoad=1 | `groupIdx*cinG+channelStart` | `t.channelUse` | `cinStart-channelStart` |
| isBiasFullLoad=0 | `groupIdx*cinG+cinStart` | `baseUseN_` | 0 |

`offsetBias_` 是上述 GM 起点。`CalcBiasFullLoadFlag` 比较本次与前次起点，决定是否重载；生成实现还绑定 bias 基址和有效通道范围。group 或通道段改变后重新检查，M/深度变化不改变通道身份。

## 2 CopyIn MTE2：FullLoadBias / LoadBiasToBT

设本次有效通道数为 `biasUse`，`validBytes=biasUse*4`。FullLoadBias 消费任务通道段，LoadBiasToBT 消费 N 子块；两者使用同一搬运规则。

| 参数 | 取值与单位 |
|---|---|
| DataCopyPad 的 blockCount | 1 |
| DataCopyPad 的 blockLen | validBytes，字节 |
| srcStride/dstStride | 0 |
| rightPadding | `alignUp(validBytes,32)/4-biasUse`，FP32 元素 |
| 填充值 | 0 |

BT 搬运按 64B 覆盖；GM→L1 的尾补齐只有 32B。当 `ceilDiv(validBytes,32)` 为奇数时，先用实现中的 `InitZeroValue` 将将被读取的 64B 对齐区域置零，再搬入有效 bias。Fill 与 DataCopyPad 之间建立所选流水的先后关系。检查实际访问范围与窄字段上限后再赋值。

```text
AllocTensor<float>
    |
    v
InitZeroValue (if needed)
    |
    v
DataCopyPad
    |
    v
biasL1Que_.EnQue
    |
    v
biasL1Que_.DeQue<float>
```

例如 biasUse=17：有效 68B，DataCopyPad 补 7 个 float 到 96B，BT 读取 128B；后 32B 也须已经初始化。补齐值只用于内部计算，不增加输出通道。

## 3 CopyIn MTE1：L1 到 BT

`DataCopyParams.blockCount=1`，`blockLen=ceilDiv(validBytes,64)*2`，此处 blockLen 的单位变为 **32B 块**；srcStride/dstStride 为 0。

```text
biasBTQue_.AllocTensor<float>
    |
    v
DataCopy(BT, L1, params)
    |
    v
biasBTQue_.EnQue
    |
    v
biasBTQue_.DeQue<float>
```

全量模式持有 L1 和 BT 句柄，在 bias 身份改变或全部任务结束时释放；分块模式在 L1→BT 复制后归还 L1，在本块 MMAD 读完后归还 BT。FreeTensor 和后续 AllocTensor 通过队列保证覆盖顺序，不能把设备指令提交等同于执行完成。

## 4 Compute：每个输出块加一次

`computeBiasOnce_` 在每个新输出块开始时为 false。公共 MmadLocal 的首次标志只在实际提交 MMAD 后改变；将首次有效贡献绑定到 BT，后续 K/kd 保留 L0C 部分和。

```cpp
// params 的 m/n/k 已按公共指令映射填写。
params.cmatrixSource = 0;
params.cmatrixInitVal = firstContribution;
if constexpr (hasBias) {
    if (firstContribution && !computeBiasOnce_) {
        params.cmatrixInitVal = false;
        params.cmatrixSource = 1;
        Mmad(useC1Buf, a2, b2, biasBTBuf_[biasOffset], params);
        computeBiasOnce_ = true;
    } else {
        Mmad(useC1Buf, a2, b2, params);
    }
} else {
    Mmad(useC1Buf, a2, b2, params);
}
params.cmatrixSource = 0;
```

带 bias 的首次 MMAD 用 BT 初始化结果，后续累加沿用 L0C。完成归约后一次 Fixpipe 转为 FP16/BF16；包装层使用此融合路径时，输出交付直接消费该结果。

## 5 ComputeForBias：无乘积贡献的输出块

`NoNeedComputeHandle<false>` 保持公共跳过行为；`NoNeedComputeHandle<true>` 申请 L0C，执行 ComputeForBias，再进入正常 ComputeEnd/写回。深度循环结束仍未应用 bias 时也执行同一步骤。

LoadL0Zero 使用全补零的 Load3D 窗口准备两路操作数，参数如下；其输入 GM/L1 真实值不参与结果：

| 参数 | L0A | L0B |
|---|---|---|
| filterW/filterH、strideW/strideH | 1、1 | 1、1 |
| padList，顺序左/右/上/下 | `{0,0,255,0}` | `{0,0,255,0}` |
| l1H/l1W | 1/32 | 1/32 |
| channelSize | C0 | alignUp(baseN,16) |
| mStartPt/kStartPt | 0/0 | 0/0 |
| mExtension | alignUp(baseM,16) | C0 |
| kExtension | C0 | alignUp(baseN,16) |

Repeat 设为 1，窗口填充值为零，Repeat/LoadData 重载按公共 SDK 适配成对选择。每个目的槽都满足相应容量；M/N 仍按当前输出块有效范围写回。

```cpp
ComputeBTBias();
WaitFlag<HardEvent::M_MTE1>(l0Slot);
LoadL0Zero();
SetFlag<HardEvent::MTE1_M>(l0Slot);
WaitFlag<HardEvent::MTE1_M>(l0Slot);
MmadLocal<true>(useC1Buf, C0, true);
SetFlag<HardEvent::M_MTE1>(l0Slot);
l0Slot = (l0Slot + 1) % inputSlotCount;
```

ComputeBTBias 在全量模式取得当前 BT 子段，在分块模式调用 LoadBiasToBT。本页阶段函数均由生成工程实现。纯 bias MMAD 的 K 临时取 C0，后续正常计算恢复 baseUseK_；不改变 Tiling 的 baseK 或完整 kValue。

## 6 生命周期与校验

bias 的生产依赖为 MTE2→MTE1→M，覆盖依赖为最后一次 M 读取→BT 可重用；全量 L1/BT 由持有状态控制一次释放，End 再完成队列事件收尾。L0C 继续沿 M→F→M 的公共协议。

检查无 bias、非零 bias、全无贡献、首次有效 kd 非零、多轮 K、多个 M/N、分组和尾通道。以 biasUse=1/8/9/16/17 覆盖 32B/64B 尾部；分别检查 Host 候选预算、最终物理槽和 API 访问区间。每个输出块只加一次 bias，重复执行不保留旧 L0C 或旧 BT 身份。
