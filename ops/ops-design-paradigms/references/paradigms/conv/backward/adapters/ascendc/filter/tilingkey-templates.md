# 权重梯度分支 Tiling 数据与模板绑定

Host 按 [Tiling](../../../generic/filter/tiling/tiling.md) 形成最终计划，再绑定编译实例和共享字段。Kernel 消费同一份结构；求解模板、窗口实现和运行时宽片分别发布。

## 1 当前执行配置

在线顺序为 DLoad → Winograd → MN_StreamK。当前采用原始分组，groupEnlarge=false、realGroup=group；x/dedy 同为 FP16/BF16/FP32、dedw 为 FP32，最终类型、格式和 hf32 由所选模板准入。MN_StreamK 的 streamkType 有 0/1/2 三态；Winograd 由非零 winogradTilingFlag 选择独立变换流水。

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

Host 与 Device 共用一份声明。LIST 的**位置**参与编码，不直接把宏数值拼接成 key；TPL_MN_STREAM_K=2 为历史保留槽位，当前 Host 不发出，也不为它编译实例。不能从 LIST 删除它而移动 TPL_DLOAD 的位置。

ASCENDC_TPL_MIX_AIC_VALUE 由目标构建元数据绑定为已核实的 MIX 类型；AIC/AIV 配比与平台查询、初始化及尾轮通知一致，不在本页写死芯片型号。

```cpp
#include "ascendc/host_api/tiling/template_argument.h"
#define TPL_STREAM_K 1
#define TPL_MN_STREAM_K 2  // 保留 LIST 位置
#define TPL_DLOAD 3
#define TPL_WINOGRAD_DISABLE 0
#define TPL_WINOGRAD_SINGLE_SHAPE_TILE_1 1
#define TPL_WINOGRAD_SINGLE_SHAPE_TILE_2 2
#define TPL_WINOGRAD_RESIDENT_FMAP 0
#define TPL_WINOGRAD_RESIDENT_DY 1

ASCENDC_TPL_ARGS_DECL(Conv3DDwCustom,
    ASCENDC_TPL_UINT_DECL(conv3DDWTemplateId, ASCENDC_TPL_8_BW,
        ASCENDC_TPL_UI_LIST, TPL_STREAM_K, TPL_MN_STREAM_K, TPL_DLOAD),
    ASCENDC_TPL_BOOL_DECL(isSplitKernelHW, 0, 1),
    ASCENDC_TPL_BOOL_DECL(groupEnlarge, 0, 1),
    ASCENDC_TPL_UINT_DECL(winogradTilingFlag, ASCENDC_TPL_8_BW,
        ASCENDC_TPL_UI_LIST, 0, 1, 2),
    ASCENDC_TPL_BOOL_DECL(winogradResidentFlag, 0, 1));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_VALUE),
        ASCENDC_TPL_UINT_SEL(conv3DDWTemplateId, ASCENDC_TPL_UI_LIST, TPL_STREAM_K),
        ASCENDC_TPL_BOOL_SEL(isSplitKernelHW, 0),
        ASCENDC_TPL_BOOL_SEL(groupEnlarge, 0),
        ASCENDC_TPL_UINT_SEL(winogradTilingFlag, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_BOOL_SEL(winogradResidentFlag, 0)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_VALUE),
        ASCENDC_TPL_UINT_SEL(conv3DDWTemplateId, ASCENDC_TPL_UI_LIST, TPL_STREAM_K),
        ASCENDC_TPL_BOOL_SEL(isSplitKernelHW, 1),
        ASCENDC_TPL_BOOL_SEL(groupEnlarge, 0),
        ASCENDC_TPL_UINT_SEL(winogradTilingFlag, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_BOOL_SEL(winogradResidentFlag, 0)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_VALUE),
        ASCENDC_TPL_UINT_SEL(conv3DDWTemplateId, ASCENDC_TPL_UI_LIST, TPL_STREAM_K),
        ASCENDC_TPL_BOOL_SEL(isSplitKernelHW, 0),
        ASCENDC_TPL_BOOL_SEL(groupEnlarge, 0),
        ASCENDC_TPL_UINT_SEL(winogradTilingFlag, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_BOOL_SEL(winogradResidentFlag, 0)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_VALUE),
        ASCENDC_TPL_UINT_SEL(conv3DDWTemplateId, ASCENDC_TPL_UI_LIST, TPL_STREAM_K),
        ASCENDC_TPL_BOOL_SEL(isSplitKernelHW, 0),
        ASCENDC_TPL_BOOL_SEL(groupEnlarge, 0),
        ASCENDC_TPL_UINT_SEL(winogradTilingFlag, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_BOOL_SEL(winogradResidentFlag, 1)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_VALUE),
        ASCENDC_TPL_UINT_SEL(conv3DDWTemplateId, ASCENDC_TPL_UI_LIST, TPL_STREAM_K),
        ASCENDC_TPL_BOOL_SEL(isSplitKernelHW, 0),
        ASCENDC_TPL_BOOL_SEL(groupEnlarge, 0),
        ASCENDC_TPL_UINT_SEL(winogradTilingFlag, ASCENDC_TPL_UI_LIST, 2),
        ASCENDC_TPL_BOOL_SEL(winogradResidentFlag, 0)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_VALUE),
        ASCENDC_TPL_UINT_SEL(conv3DDWTemplateId, ASCENDC_TPL_UI_LIST, TPL_STREAM_K),
        ASCENDC_TPL_BOOL_SEL(isSplitKernelHW, 0),
        ASCENDC_TPL_BOOL_SEL(groupEnlarge, 0),
        ASCENDC_TPL_UINT_SEL(winogradTilingFlag, ASCENDC_TPL_UI_LIST, 2),
        ASCENDC_TPL_BOOL_SEL(winogradResidentFlag, 1)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_VALUE),
        ASCENDC_TPL_UINT_SEL(conv3DDWTemplateId, ASCENDC_TPL_UI_LIST, TPL_DLOAD),
        ASCENDC_TPL_BOOL_SEL(isSplitKernelHW, 0),
        ASCENDC_TPL_BOOL_SEL(groupEnlarge, 0),
        ASCENDC_TPL_UINT_SEL(winogradTilingFlag, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_BOOL_SEL(winogradResidentFlag, 0)));
```

上例只实例化当前原始分组域。源码还包含 groupEnlarge=true 的组合，新增生成范围时需连同重排、workspace、队列和验证知识一起接入，不能仅扩大 SEL。

| 模板 | GET_TPL_TILING_KEY 的五个参数 |
|---|---|
| MN_StreamK | `(TPL_STREAM_K,isSplitKernelHW,0,0,0)` |
| Winograd | `(TPL_STREAM_K,0,0,winogradTilingFlag,residentFlag)`，flag=1/2，resident=0/1 |
| DLoad | `(TPL_DLOAD,0,0,0,0)` |

DLoad 的几何、通道块和 hf32Flag 使用运行时字段；按[驻留与资源条件](../../../generic/filter/tiling/templates/dload/tiling.md)求得计划后发布 baseM/baseN/baseK、coreNum 和精度模式。生成入口据此组装 DLoadConfig，Cout/Cin 尾块的 MMAD 与 Fixpipe 共用实际计算跨度。

streamkType、splitWo、base*、step* 是运行时字段，不编码进 key。测试同时核对 Host 值、编译实例与设备分发；不能将 MN_StreamK 模板名直接当作编号 2。

使用 MODE0 汇合时，直调按 SDK 绑定 `__schedmode__(1)`，注册侧使用 `context->SetScheduleMode(1)`。核类型与参与单元由同一能力记录确定，入口不再叠加冲突的默认类型。

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
| uint32_t | m0/n0、k0/channelSize | m0=n0=16；k0=channelSize=32/sizeof(T)，FP32 为 8 |
| uint32_t | baseM/baseN/baseK、stepKa/stepKb | 计算块与驻留 |
| uint32_t | al0/bl0/cl0/al1/bl1Pbuffer | 实际槽份数 |
| uint32_t | al1Bound/bl1Bound | 单份 L1 元素数 |
| uint32_t | singleCoreM/N/K/Cout/Ho/Dk/Group | 任务几何与范围 |
| uint64_t | singleCoreBatch、singleCoreCin、singleCoreBatchDout | 宽整数的任务批次/通道/批深度范围 |
| uint32_t | splitWo、coreNum、iterateOrder、streamkType、hf32Flag | 调度基数、窗口与运行时分支；hf32 只用于 FP32 计算模式 |

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

usedCoreNum、coreStreamK、coreBindDirection 是 Host 工作变量，不在 TConv3DDwTiling 中；blockDim 独立发布。mValue/nValue/kValue、mCnt/nCnt/totalCnt、calRound、尾宽和精度归约段长按已发布几何与最终计划复算。Host 保存校验记录，Kernel 使用一致公式，避免两份可分别改动的派生真值。

## 4 Host 流程与发布

```text
          +---------------------------------------------+
          | 解析 x / filter_size / out_backprop 与属性  |
          +----------------------+----------------------+
                                 |
                                 v
                 +---------------+---------------+
                 | 归一化秩/格式 -> 校验正向几何 |
                 +---------------+---------------+
                                 |
                                 v
              +------------------+------------------+
              | platformInfo + API / 混合核能力记录 |
              +------------------+------------------+
                                 |
                                 v
             +-------------------+-------------------+
             | 模板在线求解 -> 最终任务与资源方案    |
             +-------------------+-------------------+
                                 |
                                 v
            +--------------------+--------------------+
            | 检查字段 / buffer 地址 / 归属 / 初始化  |
            +--------------------+--------------------+
                                 |
                                 v
+--------------------------------+--------------------------------+
| serialize shared data -> key / blockDim / schedule / workspace  |
+-----------------------------------------------------------------+
```

生成工程按所选模板检查最终参数、L0/L1/UB、固定槽区间、字段范围、任务覆盖、地址和初始化。MN_StreamK 校验 base*/step*/Bound；Winograd 校验变换档、rtTiling 和四段 workspace；两者不混用资源模型。

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
context->SetBlockDim(blockDim); // Host 按所选模板计算，与共享 coreNum 分别发布
context->SetScheduleMode(1);
raw->SetDataSize(bytes);
```

平台/系统 workspace 按工程框架处理。MN_StreamK type=0 用户区为 0，type≠0 为两份 coreNum×平台 L0C 容量；Winograd 使用预转换、驻留与尾轮四区；DLoad 无该用户归约区。直调发布相同字段与启动核数，注册发布上述框架元数据；异步执行结束前保持输入输出、Device Tiling 和必要系统区有效。

发布检查包括实际输入 dtype 相同、输出 float、format 编译实例命中、字段乘积未溢出、原子初始化覆盖实际输出。日志保存真实 key、模板参数和运行时字段，不能只凭接口精度通过认定分支覆盖。
