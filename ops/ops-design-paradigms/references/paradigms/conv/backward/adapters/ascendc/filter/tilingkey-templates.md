# 权重梯度分支 Tiling 数据与模板绑定

Host 按 [Tiling](../../../generic/filter/tiling/tiling.md) 形成最终计划，再绑定编译实例和共享字段。Kernel 消费同一份结构；求解模板、窗口实现和运行时宽片分别发布。

## 1 当前执行配置

当前模板为 MN_StreamK，`streamkType=0、groupEnlarge=false、realGroup=group`，FP16/BF16 输入、FP32 累加与输出。isSplitKernelHW 取实际求解值，splitWo 为运行时字段。每个输出任务完成全部 batch×dout×ho×wo 归约，段间经 GM 原子累加合并。

### 1.1 MN_StreamK 与求解器的对应

| 环节 | 本地函数 | 产出 |
|---|---|---|
| 规格与能力 | 参数归一、形状检查、platformInfo/API 检查 | 统一问题、资源、可用指令 |
| 宽片 | InitSplitWOI | splitWo、派生 splitWi/tailWo/tailWi |
| 计算块 | InitBaseBlock / GetBaseK | baseM/N/K、L0C 份数 |
| 驻留 | SetStepK4SplitMN / UpdateStepMNK | stepKa/b、L1 份数 |
| 容量回调 | MultiCoreSplitMN / ShrinkBaseBlock | 普通/拆核/切宽与最终计算块 |
| 任务 | UpdateSingleCoreInfo | singleCoreM/N/K/Cin/Cout |
| 发布 | DoLibApiTiling / PostTiling 对应步骤 | Bound、字段、key、blockDim、调度 |

这些名字表示需要在生成工程实现的本地步骤，公式与返回值语义见通用 Tiling，不直接依赖外部求解器二进制或源码。

## 2 模板参数声明与 key

Host 与 Device 共用一份 TPL 声明。声明列表的取值和顺序参与编码，保留已选编码基线；SEL 只实例化当前两种窗口组合：

```cpp
#include "ascendc/host_api/tiling/template_argument.h"
#define TPL_MN_STREAM_K 2

ASCENDC_TPL_ARGS_DECL(Conv3DDwCustom,
    ASCENDC_TPL_UINT_DECL(conv3DDWTemplateId, ASCENDC_TPL_8_BW,
        ASCENDC_TPL_UI_LIST, 1, 2, 3),
    ASCENDC_TPL_BOOL_DECL(isSplitKernelHW, 0, 1),
    ASCENDC_TPL_BOOL_DECL(groupEnlarge, 0, 1),
    ASCENDC_TPL_UINT_DECL(winogradTilingFlag, ASCENDC_TPL_8_BW,
        ASCENDC_TPL_UI_LIST, 0, 1, 2),
    ASCENDC_TPL_BOOL_DECL(winogradResidentFlag, 0, 1));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
        ASCENDC_TPL_UINT_SEL(conv3DDWTemplateId, ASCENDC_TPL_UI_LIST, TPL_MN_STREAM_K),
        ASCENDC_TPL_BOOL_SEL(isSplitKernelHW, 0, 1),
        ASCENDC_TPL_BOOL_SEL(groupEnlarge, 0),
        ASCENDC_TPL_UINT_SEL(winogradTilingFlag, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_BOOL_SEL(winogradResidentFlag, 0)));
```

核类型来自模板选择元数据；注册入口不再叠加与之冲突的默认类型。直调按自己的编译/启动接口配置等价的混合核参与单元。目标能力从 platformInfo 取得，确认当前初始化所需 AIC/AIV 配比、MIX 启动、事件和 API 后启用。

使用 MODE0 汇合的直调入口声明 `__schedmode__(1)`；注册侧对应 `context->SetScheduleMode(1)`。两者都与 MIX 类型和实际 blockDim 配对，不只设置事件。若改用独立清零，须同时替换设备初始化等待，形成另一套完整协议。

```cpp
uint64_t tilingKey = GET_TPL_TILING_KEY(
    TPL_MN_STREAM_K, isSplitKernelHW, false, 0, false);
```

| 组合 | Kernel 行为 |
|---|---|
| `(2,false,false,0,false)` | 普通窗口，可有运行时宽片 |
| `(2,true,false,0,false)` | 原核坐标外移到搬运的拆核窗口 |

key 使用 SDK 宏计算，测试分别核对 Host 值、构建实例和设备解码。splitWo、stepKa/b、base* 不编码进 key。模板名称包含 StreamK 不等于 streamkType 非零。

## 3 共享字段

### 注册调用的上下文

手写 ACLNN 第一段使用 `L2_DFX_PHASE_1`，第二段使用 `L2_DFX_PHASE_2`；L0 装载使用 `L0_DFX`。这些标准入口同时初始化线程上下文，不能仅作为日志装饰删除。随后才创建 executor、装载 filter_size 常量并记录 AICore 启动。原始数组复制到 executor 持有的存储，保证第二段执行时仍有效。

平台信息在同一调用上下文中取得。若 MIX 核数为零，先检查入口上下文、平台资源与核类型，不填入固定核数掩盖接入错误。

### 3.1 字段类型与含义

字段按照当前计算路径的 TConv3DDwTiling 定义生成，Host 和 Device 共用同一头文件或注册数据定义：

| 类型 | 字段组 | 语义 |
|---|---|---|
| uint32_t | batch、cin、cout、cin1G、cout1G、group、realGroup | 批次、总通道、执行组通道跨度与分组 |
| uint32_t | dout/ho/wo、di/hi/wi、dk/hk/wk | 原始几何 |
| uint32_t | strideD/H/W、dilationD/H/W、六个 pad | 正向属性 |
| uint32_t | m0/n0/k0/channelSize | 当前分形粒度 16 |
| uint32_t | baseM/baseN/baseK、stepKa/stepKb | 计算块与驻留 |
| uint32_t | al0/bl0/cl0/al1/bl1Pbuffer | 实际槽份数 |
| uint32_t | al1Bound/bl1Bound | 单份 L1 元素数 |
| uint32_t | singleCoreM/N/K/Cout/Ho/Dk/Group | 任务几何与范围 |
| uint64_t | singleCoreBatch、singleCoreCin、singleCoreBatchDout | 宽整数的任务批次/通道/批深度范围 |
| uint32_t | splitWo、usedCoreNum、iterateOrder、streamkType、hf32Flag | 执行计划；当前后两项中的 streamkType/hf32Flag 均为 0 |

### 3.2 名字相近、单位不同的字段

| 名称 | 所在位置 | 单位 |
|---|---|---|
| cin1_g / cout1_g | Host 几何中间量 | C0 通道块数 |
| CinG / CoutG | 数学规格 | 实际组内通道数 |
| cin1G / cout1G | 共享 TilingData | 执行组通道跨度，**通道数** |
| al1Bound / bl1Bound | 共享 TilingData | 单份输入元素数 |
| aAligned / bAligned | Kernel 分配 | 对齐后字节数 |

当前原始分组配置发布 cin1G=cin/group、cout1G=cout/group；单组时分别等于 cin、cout。对齐只用于 mValue/nValue 和内部块，不能改变 GM 组起点，也不能把字段名中的 1G 解释为再乘 16 的块数。

### 3.3 序列化与派生量

普通 POD 使用相同字段顺序、类型、padding 和总长度；所有填充置零并校验 `sizeof(Data)%8==0`。注册数据类与手写 POD 不能在未核对偏移时解释同一段字节。

mValue/nValue/kValue、mCnt/nCnt/totalCnt、calRound、尾宽和精度归约段长按已发布几何与最终计划复算。Host 保存校验记录，Kernel 使用一致公式，避免两份可分别改动的派生真值。

## 4 Host 流程与发布

```text
      +-----------------------------------------------------+
      | parse x / filter_size / out_backprop and attributes |
      +--------------------------+--------------------------+
                                 |
                                 |
                                 v
     +---------------------------+---------------------------+
     | normalize rank / formats -> validate forward geometry |
     +---------------------------+---------------------------+
                                 |
                                 |
                                 v
       +-------------------------+-------------------------+
       | platformInfo + API / mixed-core capability record |
       +-------------------------+-------------------------+
                                 |
                                 |
                                 v
      +--------------------------+--------------------------+
      | MN_StreamK solver -> final window and resource plan |
      +--------------------------+--------------------------+
                                 |
                                 |
                                 v
 +-------------------------------+-------------------------------+
 | check fields / buffer addresses / ownership / initialization  |
 +-------------------------------+-------------------------------+
                                 |
                                 |
                                 v
+--------------------------------+--------------------------------+
| serialize shared data -> key / blockDim / schedule / workspace  |
+-----------------------------------------------------------------+
```

生成工程在最终 isSplitKernelHW、splitWo、base*、step*、singleCore* 形成后，重新检查 L0/L1、固定槽区间、字段范围、任务覆盖、输入输出地址关系和初始化协议。

```cpp
auto *raw = context->GetRawTilingData();
const size_t bytes = sizeof(Conv3DBackpropFilterV2TilingData);
if (raw == nullptr || raw->GetData() == nullptr || bytes > raw->GetCapacity()) {
    return ge::GRAPH_FAILED;
}
if (memcpy_s(raw->GetData(),raw->GetCapacity(),&td,bytes) != EOK) {
    return ge::GRAPH_FAILED;
}
context->SetTilingKey(tilingKey);
context->SetBlockDim(td.dwTiling.usedCoreNum);
context->SetScheduleMode(1);
raw->SetDataSize(bytes);
```

平台/系统 workspace 按工程框架处理；当前用户 workspace=0，不创建归约缓冲。直调发布相同字段与启动核数，注册发布上述框架元数据；异步执行结束前保持输入输出、Device Tiling 和必要系统区有效。

发布检查包括实际输入 dtype 相同、输出 float、format 编译实例命中、字段乘积未溢出、原子初始化覆盖实际输出。日志保存真实 key、模板参数和运行时字段，不能只凭接口精度通过认定分支覆盖。
