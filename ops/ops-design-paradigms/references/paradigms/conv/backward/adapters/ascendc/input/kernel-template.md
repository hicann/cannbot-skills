# 卷积反向算子 Ascend C Kernel 实现模板

三个 Tiling 策略共用 `Conv3DDxKernel`。本页给出设备入口和阶段组织；完整数据流、任务与地址模型见 [compute-flow](../../../generic/input/compute-flow.md)，API 字段见 [instruction-mapping](instruction-mapping.md)。所有阶段及参数构造由生成工程实现。

该计算类也供 Conv3DTranspose 使用；其设备入口与绑定点见第 2.3 节。本页六 GM 参数入口保持 DX 接口，公共 Process 按 hasBias 分支。类末尾增加默认类型参数 `BiasT=float`，Init 的末尾 bias 参数默认 nullptr；仅在 bias 存在时分配对应资源。

## 1 Kernel 概述

### 1.1 数据流图

```text
+-------------------------+                      +-------------------------+
| dedy / GM               |                      | weight / GM             |
+------------+------------+                      +------------+------------+
             |                                                |
             | LoadToA1                                       | LoadToB1
             | DataCopy + Fill                                | DataCopy
             |                                                |
             v                                                v
+------------+------------+                      +------------+------------+
| L1A / TQue              |                      | L1B / TQue              |
+------------+------------+                      +------------+------------+
             |                                                |
             | LoadToA2                                       | LoadToB2 / Load2D
             | Load3D(img2col)                                |
             |                                                |
             v                                                v
+------------+------------+                      +------------+------------+
| L0A / TBuf              |                      | L0B / TBuf              |
+------------+------------+                      +------------+------------+
             |                                                |
             |                                                |
             +-----------------------+------------------------+
                                     |
                                     | MmadLocal / MMAD
                                     v
                        +------------+------------+
                        | L0C / FP32              |
                        +------------+------------+
                                     |
                                     |
                                     | LoadL0c2Gm / Fixpipe
                                     |
                                     v
                        +------------+------------+
                        | dedx / GM               |
                        +-------------------------+
```

L1A 保存插零视图 dedy_expand 的驻留片段；Load3D 提取边界视图 fmap 的窗口。L1B 保持原核位置，L0B 表达 `weight_rot`。输出初始化由 [前置处理协议](sync-implementation.md) 在主计算前完成。

可选支路为 `bias/GM → biasL1Que_ → biasBTQue_ → 首次 MMAD`，细节见 [bias 通路](bias-bt-path.md)。它不改变两路窗口与权重布局。

### 1.2 状态与术语

| 状态 | 含义与生产者 |
|---|---|
| taskId、TaskCoord | 输出任务号及起点；DecodeTask |
| mStart/cinStart、baseUseM_/baseUseN_ | 当前 dedx 空间/组内通道范围；任务内分块 |
| kd、curDoutIdx | 原核深度和配对的真实 dedy 深度；有符号采样检查 |
| kStart、baseUseK_ | 当前完整 K 起点和含通道补齐的长度 |
| a1/b1.segmentStart、segmentUse、curStep | 各父段实际范围；段定位 |
| kLocalA/kLocalB | 相对各自父段的 K 起点 |
| l0Slot、cSlot | L0 输入成对槽 / L0C 槽 |
| needComputeFlag_ | 当前输出块是否进入计算；UpdateL1ComputeInfo 更新行带与深度采样后确定 |
| firstContribution | 当前输出块是否尚无有效乘加；跨 K 和 kd 保留 |
| computeBiasOnce_ | 当前输出块是否已引入 bias，输出块开始时重置 |

`ctx.baseUseM_/N_/K_` 分别对应本轮有效 `baseUseM_/baseUseN_/baseUseK_`，`ctx.curLoadKbl1_` 对应 L1B 实际父跨度。M 的有效长度和指令对齐长度按以下口径传递：

| 消费者 | M 取值 |
|---|---|
| UpdateL1ComputeInfo、ctx.baseUseM_ | 当前有效 baseUseM_；有窗口裁剪时先更新有效范围 |
| Load3D.mExtension | baseUseM_ |
| MmadParams.m | isLastMBlock ? mCompute : baseUseM_ |
| Fixpipe.mSize / srcStride | baseUseM_ / mCompute |

例如任务末块有 27 个有效空间点：`baseUseM_=27、mExtension=27、MmadParams.m=32、Fixpipe.mSize=27`。对齐不改变窗口和写回的有效范围。

## 2 Kernel 骨架代码

### 2.1 入口与编译实例

注册入口保留五模板实参和六 GM 参数；直调 launch 按自己的 ABI 传入相同计算数据：

```cpp
template <uint8_t loadB2Condition, uint8_t kernelSplitMode,
          uint8_t groupConvMode, bool isBasicBlockTiling,
          uint8_t loadB1Condition>
__global__ __aicore__ void conv3ddx_custom(
    GM_ADDR input_size, GM_ADDR filter, GM_ADDR out_backprop,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA_WITH_STRUCT(Conv3DDxTilingData, td, tiling);
    InitOutput<DTYPE_Y>(y, td);
    if ASCEND_IS_AIC {
        Conv3DDxKernel<DTYPE_FILTER, FORMAT_FILTER,
                      DTYPE_OUT_BACKPROP, FORMAT_OUT_BACKPROP,
                      DTYPE_Y, FORMAT_Y, loadB2Condition> op;
        op.Init(filter, out_backprop, y, td);
        op.Process();
        op.End();
    }
}
```

算子包的入口分发与任务类型配置属工程封装知识，不在本范式内（注册形态参考 [example/conv3ddx_aclnn/](../example/conv3ddx_aclnn/README.md)，工程骨架见仓内技能 `ops/ascendc-registry-invoke-template`）；生成骨架不保留架构分发，宏形态与任务类型须与目标 SDK 核对。

这是生成骨架，`InitOutput` 与 `Conv3DDxKernel` 均须在本工程定义。支持的五参数组合由共享声明限制。Kernel 类型、SDK 宏和初始化方式按 [模板绑定](tilingkey-templates.md) 的核类型声明配对；先初始化、后允许无任务计算核返回。外部初始化版本的 `InitOutput` 不发起或等待融合通知。`Process` 负责计算及归还持有张量；入口只调用一次 `End`，等待并释放队列事件，随后由 TPipe 完成自身事件的收尾。

### 2.2 任务与归约骨架

下面省略类和模板前缀，阶段函数按后文实现。TaskCoord 保存解码后的任务起点及有效长度；L1 父段状态分别保存实际范围、形状、槽和持有权限。

```cpp
template <bool hasBias = false>
__aicore__ inline void Process() {
    for (uint64_t taskId = coreId; taskId < totalTasks; taskId += td.coreNum) {
        TaskCoord t = DecodeTask(taskId,td);
        CheckFullLoadEnable(t);
        if constexpr (hasBias) {
            SetBias(t.groupIdx*td.cinG+t.channelStart, t.channelUse);
            CalcBiasFullLoadFlag();
            if (td.isBiasFullLoad && fullLoadBiasFlag_) {
                FreeBiasTensor();
                FullLoadBias();
            }
        }
        for (uint32_t d = 0; d < t.depthUse; ++d) {
            curDinIdx_ = t.depthStart+d;
            for (uint32_t n = 0; n < t.channelUse; n += td.baseN) {
                cinStart = t.channelStart+n;
                baseUseN_ = min(td.baseN,t.channelUse-n);
                nCompute = alignUp(baseUseN_,16);
                for (uint64_t m = 0; m < t.spaceUse; m += td.baseM) {
                    mStart = t.spaceStart+m;
                    baseUseM_ = min(uint64_t(td.baseM),t.spaceUse-m);
                    computeBiasOnce_ = false;
                    UpdateL1ComputeInfo();
                    mCompute = alignUp(baseUseM_,16);
                    isLastMBlock = (m+baseUseM_ == t.spaceUse);
                    if (!needComputeFlag_) {
                        NoNeedComputeHandle<hasBias>();
                        continue;
                    }
                    auto &q = (cSlot == 0 ? l0cPing : l0cPong);
                    auto useC1Buf = q.AllocTensor<float>();
                    bool firstContribution = true;
                    for (uint32_t kd = 0; kd < td.dk; ++kd) {
                        int64_t dTmp = int64_t(curDinIdx_)+td.padFront-int64_t(kd)*td.dilationD;
                        if (dTmp < 0 || dTmp % td.strideD || dTmp / td.strideD >= td.dout) continue;
                        curDoutIdx = dTmp / td.strideD;
                        for (uint64_t kStart = 0; kStart < kValue; kStart += td.baseK) {
                            baseUseK_ = min(uint64_t(td.baseK),kValue-kStart);
                            LoadToA1(curDoutIdx,kStart);
                            LoadToB1(kd,kStart);
                            if constexpr (hasBias) {
                                if (firstContribution) ComputeBTBias();
                            }
                            WaitFlag<HardEvent::M_MTE1>(l0Slot);
                            LoadToA2(kStart,baseUseK_);
                            LoadToB2(kStart,baseUseK_);
                            SetFlag<HardEvent::MTE1_M>(l0Slot);
                            ReleaseConsumedParents();
                            WaitFlag<HardEvent::MTE1_M>(l0Slot);
                            MmadLocal<hasBias>(useC1Buf,baseUseK_,firstContribution);
                            firstContribution = false;
                            SetFlag<HardEvent::M_MTE1>(l0Slot);
                            l0Slot = (l0Slot+1) % inputSlotCount;
                        }
                    }
                    if constexpr (hasBias) {
                        if (!computeBiasOnce_) {
                            ComputeForBias(useC1Buf);
                            firstContribution = false;
                        }
                        if (!td.isBiasFullLoad) biasBTQue_.FreeTensor(biasBTBuf_);
                    }
                    if (!firstContribution) {
                        q.EnQue(useC1Buf);
                        useC1Buf = q.DeQue<float>();
                        LoadL0c2Gm(useC1Buf);
                    }
                    q.FreeTensor(useC1Buf);
                    cSlot = (cSlot+1) % td.cl0Pbuffer;
                }
            }
        }
        FinishTaskB1();
    }
    FreeB1Tensor();
    if constexpr (hasBias) FreeBiasTensor();
}
```

任务内顺序为深度 → dedx 通道块 → 空间块，与 AdvanceIterIndices 的 M 最快变化一致。`UpdateL1ComputeInfo` 确定有效 M/N、dedy_expand 行带交集及有效核深度。`NoNeedComputeHandle<false>` 只归还持有父段；`NoNeedComputeHandle<true>` 申请 L0C，执行 ComputeForBias，将 needComputeFlag_ 置 true，再按正常 M→F 协议写回、归还 L0C 并推进 cSlot。该分支归还本块的分块 bias，仍有效的全量 bias 继续持有；输入父段按公共规则归还。非空行带中只有 stride 插零时，仍可形成零窗口。

无 bias 时，未提交 MMAD 的块只归还已申请 L0C，跳过位置由输出初始化覆盖；有 bias 时 ComputeForBias 保证产生完整输出。输入行带裁剪不减少带 bias 输出的有效范围；指令范围不足时由 Tiling 调整块长。

SetBias 在本工程绑定 bias 全局起点和有效通道数；CalcBiasFullLoadFlag 比较持有身份。FreeBiasTensor/FreeB1Tensor 仅归还仍持有的张量。骨架中分块 bias 持有到当前输出块计算结束，首次贡献前搬入一次；每次实际归还后清除持有状态。全量 bias 与 L1B 分别管理身份和释放。具体 API、首次 MMAD 与纯 bias MMAD 参数见 [bias 通路](bias-bt-path.md)。

| 阶段 | 输入和状态 | 结果 |
|---|---|---|
| DecodeTask | taskId、td | TaskCoord 的各轴起点和有效长度 |
| CheckFullLoadEnable | 当前任务与前一权重身份 | 保留 L1B，或释放旧版本并关闭当前跨任务权限 |
| UpdateL1ComputeInfo / NoNeedComputeHandle | 当前有效输出块、dedy_expand 行带和核深度采样 | 更新 needComputeFlag_；跳过时按持有状态归还父段 |
| LoadToA1 | curDoutIdx、kStart、当前 dedx 空间块、L1A 父段 | 当前 dedy_expand 驻留片段及其就绪句柄 |
| LoadToB1 | 原 kd、kStart、dedx 通道块、L1B 状态 | 对应 weight 变换分支的实际父布局及就绪句柄 |
| LoadToA2/L0B | 当前 K 和各自父段/子块、可写 L0 槽 | 配对同一 K 区间的 NZ/ZN 操作数 |
| ReleaseConsumedParents | 两路最后消费者与完整驻留状态 | 在末次 MTE1 之后发布父段归还依赖 |
| MmadLocal | 当前 L0A/L0B、尺寸、firstContribution、useC1Buf | FP32 部分和 |
| ComputeBTBias / ComputeForBias | 可选 bias、isBiasFullLoad、当前输出通道；后者使用全补零操作数 | 首次 bias 或全无贡献块的 bias 结果 |
| LoadL0c2Gm | 完整 useC1Buf、当前 dedx 坐标和有效 M/N | Fixpipe 写回提交 |
| FinishTaskB1 / FreeB1Tensor | L1B 持有句柄、跨任务保留条件 | 任务末按条件保留；全部任务结束后归还剩余 L1B |
| End | 已归还的 L1A/L1B、L0C 队列 | 对队列调用 FreeAllEvent，等待归还并释放队列事件 |

这些阶段按后文参数和同步规则定义在生成工程中。`min/ceilDiv` 等标量辅助函数同样提供本地定义；它们与 SDK API 名称分开。

LoadToA1/L1B 内部按各自 step 判断装载、出队或复用；ReleaseConsumedParents 只对本轮末次读取的父段发布归还。FinishTaskB1 封装任务末的保留判断和 FreeFullLoadL1Tensor，不提前释放跨任务仍有效的 L1B。两个 L0 输入双 Buffer 时 inputSlotCount=2，否则为 1。固定事件号的初态与末次等待由所选 SDK 的 TPipe 管理，具体所有权见 [同步实现](sync-implementation.md#12-初态与收尾的所有权)。

### 2.3 转置卷积入口与绑定

任务循环、地址、MTE2/MTE1、MMAD、Fixpipe 和同步使用同一 `Conv3DDxKernel`；带 bias 的阶段按 [bias 通路](bias-bt-path.md) 接入公共实现。本节定义 Transpose 的外部设备入口。

五个模板参数继续控制 weight 变换与执行配置。注册 ABI 的 GM 参数依次为 input_size、x、filter、bias、offset_w、y、workspace、tiling：

```cpp
template <uint8_t loadB2Condition, uint8_t kernelSplitMode,
          uint8_t groupConvMode, bool isBasicBlockTiling,
          uint8_t loadB1Condition>
__global__ __aicore__ void conv3dtranspose_custom(
    GM_ADDR input_size, GM_ADDR x, GM_ADDR filter,
    GM_ADDR bias, GM_ADDR offset_w, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA_WITH_STRUCT(Conv3DDxTilingData, td, tiling);
    InitOutput<DTYPE_Y>(y, td);
    if ASCEND_IS_AIC {
        Conv3DDxKernel<DTYPE_FILTER, FORMAT_FILTER,
                      DTYPE_X, FORMAT_X, DTYPE_Y, FORMAT_Y,
                      loadB2Condition, DTYPE_BIAS> op;
        op.Init(filter, x, y, td, bias);
        if (bias != nullptr) {
            op.template Process<true>();
        } else {
            op.template Process<false>();
        }
        op.End();
    }
}
```

共享类的最后一个类型参数 `BiasT` 默认 float，DX 入口仍可省略。`Init` 的 bias 参数默认 nullptr。当前 FP16/BF16 生成实例将 DTYPE_BIAS 绑定 float；SDK 可选输入缺省时，也在构建配置中明确这个编译类型。

input_size 的值已由 Host 解析到几何，设备计算消费 td；offset_w 为接口保留槽，当前数值通路绑定缺省。工作流包装层统一空 bias 与 nullptr，Host 的 hasBiasFlag_、资源计划和设备分支保持一致。

公共 Kernel 的绑定点：

| 公共阶段 | Transpose 使用方式 |
|---|---|
| DecodeTask / 地址恢复 | 按解析后的 y 形状生成 dedx 任务；x/filter 采用公共 dedy/weight 轴序 |
| CheckFullLoadEnable | 只管理 L1B 权重身份，与 bias 身份独立 |
| SetBias / CalcBiasFullLoadFlag | 绑定 `groupIdx*cinG+channelStart`；决定当前任务 bias 的重载与释放 |
| FullLoadBias / ComputeBTBias | 按 isBiasFullLoad 取得任务通道段或当前 N 子块 |
| UpdateL1ComputeInfo | 使用真实核几何、最终输出范围与反向边界 |
| MmadLocal<true> | 首次贡献使用 BT，此后完整累加 K/kd |
| NoNeedComputeHandle<true> / ComputeForBias | 无乘积时仍产生完整 bias 输出 |
| LoadL0c2Gm | 完整乘积加 bias 后转为 y 类型，只写有效 M/N |
| FreeBiasTensor / End | 张量归还一次；等待并释放相应队列事件 |

output_padding 已折入目标形状及边界，内核不再给 dk/hk/wk、dilation 或 K 额外加 q。带 bias 和无 bias 使用相同的三种 weight 变换。

产出校验：检查八个 GM 槽、编译宏、共享字段、五参数 key 和核类型一致；实际加载实例应使用本入口，检查输入顺序，特别是 x/filter 与 DX 入口的差别。

## 3 地址偏移计算

### 3.1 核间偏移计算

`mCount=ceilDiv(hi*wi,singleCoreM)`，`nCount=ceilDiv(cinG,singleCoreCin)`，`dCount=ceilDiv(di,singleCoreDin)`；`totalTasks=batch*group*mCount*nCount*dCount`。批次和组在外层，内层从慢到快为：

| 条件 | 顺序 |
|---|---|
| dk>1 | M → D → N |
| dk=1 且 hk>1，或 mCount>nCount | D → M → N |
| 其余 | D → N → M |

起点为任务坐标乘任务长度，尾长取剩余总长与任务长度的较小值。DecodeTask 按 [任务恢复公式](../../../generic/input/compute-flow.md#6-多核任务与输出归属) 在工程内实现；设备侧函数使用 `__aicore__` 修饰。

### 3.2 核内循环间偏移计算

生成代码选择完整全局坐标寻址：每次先求 batch、group、depth、h/w、channel，再调用布局地址式。若优化为任务基址加增量，记录基址已含哪些轴，防止重复加深度、组或 K 起点。

| 对象 | 完整坐标来源 |
|---|---|
| dedy | batch、group*coutG+当前输出通道、curDoutIdx、实际 dedy 高宽 |
| weight | group*coutG+当前输出通道、组内 dedx 通道、原 kd/kh/kw |
| dedx | batch、group*cinG+cinStart、当前 dedx 深度、mStart 解码的高宽 |

`kLocalA=kStart-a1.segmentStart`，`kLocalB=kStart-b1.segmentStart`。矩阵中对应的 K 元素必须指向同一 dedy 通道和核高宽位置。Tensor 下标单位为元素，容量及 Buffer 基址单位为字节。

### 3.3 Buffer 分配

按 [Buffer 布局](../buffer-and-types.md#3-物理槽划分) 设置五路槽与最大访问范围；初始化队列、事件与归还协议见 [同步实现](sync-implementation.md)。本页后续阶段只消费已经确定的槽和父布局。

## 4 CopyIn MTE2 逻辑

### 4.1 LoadToA1：dedy 搬入

按父段 K 恢复 dedy 通道范围，按 dedx 窗口反推 dedy_expand 中的驻留高宽、stride 相位及原始 dedy 起点。NCDHW 使用 DN2NZ，NDHWC 使用 ND2NZ；目的布局为 `(Cout1,curHoSize,woExpand,Cout0)`。

先对本次需置零的可写 L1A 区域 Fill，再用 `PipeBarrier<PIPE_MTE2>()` 排序，随后 DataCopy 搬入真实值并 EnQue。窗口补边、stride 插零和存储尾部补零分别处理。完整字段见[dedy 指令参数](instruction-mapping.md#2-dedy-到-l1a)，行带与裁剪数学见[compute-flow](../../../generic/input/compute-flow.md#521-从-dedx-块反推-dedy-行带)。

### 4.2 LoadToB1：权重搬入

| weight 格式 / weight 变换分支 | DataCopy 参数 | 目的父布局 |
|---|---|---|
| 任一 / 仅转置 | Nd2NzParams | `(Cin1,Cout1,Cout0,Cin0)` |
| NCDHW / 转置加逆序 | Dn2NzParams | `(Cin1,Hk,Wk,Cout1,Cout0,Cin0)` |
| NDHWC / 转置加逆序 | Nd2NzParams | 同上 |
| NCDHW 或 NDHWC / 逆序 | Dn2NzParams | `(Cout1,Hk,Wk,Cin1,Cin0,Cout0)` |

仅转置时核体积为 1，两种 weight 格式都化为 Co×CiG 矩阵。按实际父段和通道尾部构造参数，具体字段见[权重指令参数](instruction-mapping.md#3-weight-到-l1b)；不能从源函数名称猜 ND/DN 朝向。

### 4.3 父段与预取

每路以 `stepK` 独立分段。第 j 个 K 迭代所属段起点为 `(j//stepK)*stepK*baseK`，末段 curStep 取剩余迭代数，segmentUse 取剩余实际 K 长度。下一段存在时才预取，段内消费结束后才能复用其槽。L1B 驻留命中时沿用持有的父段，不重复 Alloc/EnQue。

## 5 CopyIn MTE1 逻辑

### 5.1 LoadToA2：L1A 到 L0A

Load3D 的 H/W stride 固定为 1，保留卷积 dilation；原 stride 已由 L1A 插零表达。mStartPt 指向当前行内位置，kStartPt 使用 kLocalA。FMatrix 的驻留形状和补边必须与 L1A 一致。

SDK 版本选择 `SetLoadDataRepeatWithStride + LoadDataWithStride` 或 `SetLoadDataRepeat + LoadData` 成对使用。M 有效范围与物理对齐跨度分开：例如 72 个有效位置占 80 的存储跨度，写回仍为 72；具体合法字段按目标头文件核实。

### 5.2 LoadToB2：L1B 到 L0B

| 分支 | ifTranspose / 源地址组织 |
|---|---|
| 仅转置 | true；Cout1 正向递增，每个源分形为 256 个输入元素 |
| 转置加逆序 | true；每个 Cout1 内按 H/W 核位置逆序 |
| 逆序 | false；srcStride 为负的 N 分形数，每个 Cout1 的窗口分别提交 |

转置路径的源 stride 使用 L1B 实际父 K 跨度除 16；目的 stride 使用当前 N 分形数。K 块跨 Cout1 窗口时拆分提交，保持 Cout1 正序和每个窗口内部逆序。偏移、LoadData2DParamsV2 字段及跨窗例子见[L0B 指令参数](instruction-mapping.md#5-l1b-到-l0b)。

## 6 Compute 计算逻辑

### 6.1 深度采样与完整归约

对当前 dedx 深度 curDinIdx_ 和原核深度 kd，计算 `dTmp=curDinIdx_+padFront-kd*dilationD`。只有 dTmp 非负、能被 strideD 整除且商小于 dout 时，才进入该深度的 K 循环。原 kd 与逆序后的深度对应只表达一次。

L0C 保存当前输出块全部有效 kd 和 K 的部分和。`firstContribution` 在第一次实际 MMAD 后才变为 false；跳过的深度和父段切换都不能重置它。

### 6.2 MmadLocal

MmadParams 的 k 为 baseUseK_，n 为 baseUseN_；m 按所选模板的末 M 块规则设置。无 bias 时 `cmatrixSource=0、cmatrixInitVal=firstContribution`，调用 `Mmad(useC1Buf,a2,b2,params)`。带 bias 时首次实际乘加取 `cmatrixSource=1、cmatrixInitVal=false`，使用带 BT 的重载，此后恢复从 L0C 累加。两者均取 `unitFlag=0、kDirectionAlign=0`，输入 half/bfloat16_t，L0C float。

补齐 dedy 通道进入 baseUseK_，但其乘积为零；输出 M/N 尾部按有效范围写回。所有有效深度完成后，CO1 队列发布完整结果。

## 7 输出初始化与主计算通路

### 7.1 输出初始化

工程内 InitOutput 按前置处理协议清零 dedx，并建立初始化完成到计算写回的依赖。融合版本保留参与核、通知和等待；外部版本由调用序列保证依赖。MTE3 流水只出现在初始化的跨核通知旗语中（见 [同步实现](sync-implementation.md#32-两个编译分支)），不参与结果搬运。

### 7.2 主计算通路

主结果由 F 从 L0C 直接写 GM。当前主计算无需 UB 结果累加或 MTE3 中转；辅助初始化与主计算结果各自按自己的执行通路同步。

## 8 CopyOut 输出通路逻辑

### 8.1 LoadL0c2Gm：NZ 到 ND/DN

| dedx 格式 | Fixpipe 朝向 | dstStride（输出元素） |
|---|---|---|
| NCDHW | COLUMN_MAJOR，NZ2DN | di*hi*wi |
| NDHWC | ROW_MAJOR，NZ2ND | cin |

`mSize=baseUseM_、nSize=baseUseN_、srcStride=alignUp(baseUseM_,16)`；目标地址由第 3 节计算。FP32 转 FP16/BF16 分别设置 F322F16/F322BF16，按目标 SDK 对应的 Fixpipe 重载执行。

### 8.2 写回与归还

只写有效 M/N，保留全局通道、空间和组间步幅。输出块在完整归约后完成 CO1 的 EnQue/DeQue，LoadL0c2Gm 只提交 Fixpipe；输出块末尾统一执行一次 FreeTensor。全无贡献且无 bias 时省略写回；带 bias 时写回 ComputeForBias 的结果。下一轮复用等待 F 归还，不能仅轮换下标就覆盖。

## 9 流水同步

按 [同步实现](sync-implementation.md) 的队列/事件协议封装 ReleaseConsumedParents、FinishTaskB1、FreeB1Tensor 和 End。L1A/L1B 按各自最后一次 MTE1 读取归还，输入槽按实际 MMAD 轮转，L0C 按完整输出块交接。

输出块循环负责完整结果的 EnQue/DeQue，`LoadL0c2Gm` 只提交 Fixpipe；输出块末尾是正常路径的 L0C 唯一 FreeTensor 位置。NoNeedComputeHandle<true> 独立完成同一协议后立即返回，外层不再归还该块。无 bias 的全无贡献块只归还已取得的槽。Process 结束时归还 held L1B 和可选全量 bias，End 收尾相应队列事件。

## 10 Kernel 产出校验

1. 共享字段、dtype/format、五参数实例与启动类型一致；所有工程内阶段函数有定义。
2. 逐元素核对 GM 地址、K 父段和三种 weight 变换的布局，覆盖跨 Cout1、尾段和多组。
3. 分别核算 Host 预算、实际槽区间及 API 最大访问范围。
4. 验证空行带、无有效深度、首次有效深度、完整归约、L1B 命中/失效和最后一次 F；跳过路径不消费未产生的事件。
5. 用非零预填输出检查初始化与重复执行；设备编译、数值和性能分别记录。
6. 带 bias 时验证每个输出块恰好加一次，覆盖无贡献块、BT 尾部、身份变化及实际分配范围；DX 的无 bias 实例不产生 bias 搬运或队列等待。

完整设备编译、数值与再生成按 [验证方案](../../../generic/input/verification.md) 验收。
