# 卷积反向算子 Tiling 数据与模板绑定

Host 按 [模板选择](../../../generic/input/tiling/tiling.md) 求解参数，再绑定编译分支、运行字段和启动配置。本文定义新生成工程的三实参声明；自带示例的私有 ABI 见第 2 节末。

## 1 执行配置与模板绑定

Tiling 模板决定求解策略，数据准备决定两路布局，编译声明决定可执行实例。准入见模板及 [指令映射](instruction-mapping.md)，本页只维护最终绑定。

| 执行路径 | kernelSplitMode | loadB1Condition | loadB2Condition | 运行字段 |
|---|---|---|---|---|
| 普通布局 | 0 | 0 | FP16/BF16 在最终块上选 1/2；FP32 为 2 | enlarge=1 |
| VecTranspose | 0 | 0 | 2 | enableVecTrans=1 |
| C04 | 0 | 1 | 0 | 入口设置 enableC04Flag=true |
| group enlarge | 0 | 0 | 2 | enlarge>1；group/oriGroup 按计算组/原始组发布 |
| kernel-split | 1 或 2 | 0 | 2 | kSCoutFullLoad、kSUseWorkSpace；可配合 enableVecTrans |

groupConvMode 是 Host 的分组状态，Kernel 从 enlarge 判断；它和 isBasicBlockTiling 都不是本声明的模板实参。C04、group enlarge、普通前置转置在当前浮点范围互斥；kernel-split 与 C04/group enlarge 不组合，前置转置按其模板条件判断。

standard 的 enableFullLoad=0，l1-resident 为 1；small-shape 发布全载收益复核后的实际值。singleIterateDk=dk、iterateOrder=1。kSCoutFullLoad 单独置位不要求 workspace；enableVecTrans 或 kSUseWorkSpace 置位时，入口必须取得相应用户空间。

| 模板 | 求解器 | 域内登记优先级 |
|---|---|---|
| kernel-split 的 cout 全载变体 | Conv3DDXV2KernelSplitFullLoadTiling | 97 |
| kernel-split | Conv3DDXV2KernelSplitTiling | 98 |
| small-shape | Conv3DDXV2SmallShapeTiling | 99 |
| l1-resident | Conv3DDXV2FullLoadTiling | 100 |
| standard | Conv3DDXV2InnerProductTiling | 101 |

同一 key 可服务不同求解模板；驻留与任务仍由最终运行字段决定。按本范式准入和上述域内顺序选择，不先固定 key 再修改准入标志。

dtype 由编译实例的 `DTYPE_FILTER/DTYPE_OUT_BACKPROP/DTYPE_Y` 绑定，key 保持三个实参。FP32 的 C0/K0=8、c0BitsA/B=3，按运行字段和 float 类型生成地址与计算参数。

## 2 loadB2Condition 判定与 key

普通求解器在最终块形状上选择 B2；kernel-split 固定为 REVERSE_ONLY。weightDtype 来自 weight 规格：

```cpp
if (enableC04Flag) return TPL_NO_TRANSPOSE_NO_REVERSE;
if (weightDtype == ge::DT_FLOAT || filterFormat == ge::FORMAT_DHWCN ||
    enlarge > 1 || enableVecTransFlag) {
    return TPL_REVERSE_ONLY;
}
const uint64_t kernelHW = uint64_t(hk)*wk;
const uint64_t kernelDHW = uint64_t(dk)*kernelHW;
if (kernelDHW == 1) return TPL_TRANSPOSE_AND_REVERSE;
if (filterFormat == ge::FORMAT_NDHWC && baseN >= kernelHW) return TPL_REVERSE_ONLY;
if (filterFormat == ge::FORMAT_NCDHW && kernelDHW <= 64/weightBytes/cin) {
    return TPL_REVERSE_ONLY;
}
return TPL_TRANSPOSE_AND_REVERSE;
```

cin 为总输入通道数。核体积为 1 时，FP16/BF16 取分支 1，在 B2 转置；FP32 取分支 2，在 GM→L1B 形成通道朝向，B2 直接取块。C04 的逆序由 AIV gather 完成。单位核搬运见 [指令映射 §5.4](instruction-mapping.md#54-单位核按-dtype-选择搬运)。

### 2.1 共享声明与合法组合

下面的 `CONV3DDX_CV_RATIO` 是工程编译常量，由已核实的目标执行配置绑定为 1 或 2，必须与 C/V 参与者和 Host 检查一致；它不是 platformInfo 的字段。SDK 核型名称保留准确拼写。

```cpp
#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_NO_TRANSPOSE_NO_REVERSE 0
#define TPL_TRANSPOSE_AND_REVERSE 1
#define TPL_REVERSE_ONLY 2
#define TPL_NO_SPLIT_KERNEL 0
#define TPL_SPLIT_KERNEL_HW 1
#define TPL_SPLIT_KERNEL_H 2
#define TPL_GM_TO_L1 0
#define TPL_VEC_TO_L1_C04 1

#if CONV3DDX_CV_RATIO == 2
#define CONV3D_DX_KERNEL_TYPE_SEL ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2)
#elif CONV3DDX_CV_RATIO == 1
#define CONV3D_DX_KERNEL_TYPE_SEL ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_1)
#else
#error "Bind CONV3DDX_CV_RATIO to the selected MIX execution contract"
#endif

ASCENDC_TPL_ARGS_DECL(Conv3DDxCustom,
    ASCENDC_TPL_UINT_DECL(loadB2Condition, ASCENDC_TPL_8_BW,
        ASCENDC_TPL_UI_LIST, TPL_NO_TRANSPOSE_NO_REVERSE,
        TPL_TRANSPOSE_AND_REVERSE, TPL_REVERSE_ONLY),
    ASCENDC_TPL_UINT_DECL(kernelSplitMode, ASCENDC_TPL_8_BW,
        ASCENDC_TPL_UI_LIST, TPL_NO_SPLIT_KERNEL,
        TPL_SPLIT_KERNEL_HW, TPL_SPLIT_KERNEL_H),
    ASCENDC_TPL_UINT_DECL(loadB1Condition, ASCENDC_TPL_8_BW,
        ASCENDC_TPL_UI_LIST, TPL_GM_TO_L1, TPL_VEC_TO_L1_C04));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(CONV3D_DX_KERNEL_TYPE_SEL,
        ASCENDC_TPL_UINT_SEL(loadB2Condition, ASCENDC_TPL_UI_LIST, TPL_TRANSPOSE_AND_REVERSE),
        ASCENDC_TPL_UINT_SEL(kernelSplitMode, ASCENDC_TPL_UI_LIST, TPL_NO_SPLIT_KERNEL),
        ASCENDC_TPL_UINT_SEL(loadB1Condition, ASCENDC_TPL_UI_LIST, TPL_GM_TO_L1)),
    ASCENDC_TPL_ARGS_SEL(CONV3D_DX_KERNEL_TYPE_SEL,
        ASCENDC_TPL_UINT_SEL(loadB2Condition, ASCENDC_TPL_UI_LIST, TPL_REVERSE_ONLY),
        ASCENDC_TPL_UINT_SEL(kernelSplitMode, ASCENDC_TPL_UI_LIST, TPL_NO_SPLIT_KERNEL),
        ASCENDC_TPL_UINT_SEL(loadB1Condition, ASCENDC_TPL_UI_LIST, TPL_GM_TO_L1)),
    ASCENDC_TPL_ARGS_SEL(CONV3D_DX_KERNEL_TYPE_SEL,
        ASCENDC_TPL_UINT_SEL(loadB2Condition, ASCENDC_TPL_UI_LIST, TPL_REVERSE_ONLY),
        ASCENDC_TPL_UINT_SEL(kernelSplitMode, ASCENDC_TPL_UI_LIST, TPL_SPLIT_KERNEL_HW),
        ASCENDC_TPL_UINT_SEL(loadB1Condition, ASCENDC_TPL_UI_LIST, TPL_GM_TO_L1)),
    ASCENDC_TPL_ARGS_SEL(CONV3D_DX_KERNEL_TYPE_SEL,
        ASCENDC_TPL_UINT_SEL(loadB2Condition, ASCENDC_TPL_UI_LIST, TPL_REVERSE_ONLY),
        ASCENDC_TPL_UINT_SEL(kernelSplitMode, ASCENDC_TPL_UI_LIST, TPL_SPLIT_KERNEL_H),
        ASCENDC_TPL_UINT_SEL(loadB1Condition, ASCENDC_TPL_UI_LIST, TPL_GM_TO_L1)),
    ASCENDC_TPL_ARGS_SEL(CONV3D_DX_KERNEL_TYPE_SEL,
        ASCENDC_TPL_UINT_SEL(loadB2Condition, ASCENDC_TPL_UI_LIST, TPL_NO_TRANSPOSE_NO_REVERSE),
        ASCENDC_TPL_UINT_SEL(kernelSplitMode, ASCENDC_TPL_UI_LIST, TPL_NO_SPLIT_KERNEL),
        ASCENDC_TPL_UINT_SEL(loadB1Condition, ASCENDC_TPL_UI_LIST, TPL_VEC_TO_L1_C04)));
```

本范围共五个合法组合，VecTranspose/group enlarge 与普通仅逆序复用同一编译实例，按运行字段改变来源和计算组。Host/Device 共用声明、实参顺序和位宽，保留 LIST 的语义顺序。

该声明使用融合 AIV 初始化的 MIX 入口。若生成普通 AIC-only 版本，须采用独立初始化或完整块写回协议，并为其声明匹配的 AIC-only 注册元数据；不能保留上述 MIX 元数据却删除 AIV 逻辑。核型不是第四个算子模板实参，详见 [C/V 协同](cv-collaboration.md)。

### 2.2 编码与运行字段

```cpp
const uint64_t tilingKey = GET_TPL_TILING_KEY(
    loadB2Condition, kernelSplitMode, loadB1Condition);
```

| 三实参语义值 | 数据路径 |
|---|---|
| `(1,0,0)` | 转置加逆序；单位核时仅转置 |
| `(2,0,0)` | 普通仅逆序，或 VecTranspose/group enlarge |
| `(2,1,0)` | split-HW |
| `(2,2,0)` | split-H |
| `(0,0,1)` | C04；B2 不再转置或逆序 |

FP32 使用 `(2,0,0)`、`(2,1,0)` 或 `(2,2,0)`；其余两种组合用于 FP16/BF16。

实际 key 由同一 SDK 的宏编码并在 Host/Device 分发中核对，不手写十进制值。group、oriGroup、enlarge、enableVecTrans、enableFullLoad 仍分别发布。

### 2.3 自带示例的私有 ABI

[conv3ddx](../example/conv3ddx/README.md) 与 [conv3dtranspose](../example/conv3dtranspose/README.md) 的 FP16/BF16 实现保留私有 loadB2=1/2/3 及配套 key/POD。私有 3 表示单位核仅转置；本文对同一 FP16/BF16 场景使用 1，FP32 使用 2。复用示例时成套使用其 Host、Kernel、共享头和调用分发；按本文声明生成时统一绑定三实参实例及共享字段。

## 3 共享字段

Host/Device 使用同一份定义。注册用 BEGIN_TILING_DATA_DEF/REGISTER_TILING_DATA_CLASS 生成对应算子的字段；直调可按同一字段定义共享 POD。两种传递方式各自验证编码和设备解码。

| 类型 | 字段组 | 用途 |
|---|---|---|
| uint64_t | coreNum、singleCoreBatch、singleCoreM | 启动计算核数、任务 batch/空间范围 |
| uint32_t | batch、cin/cout、cinG/coutG、group、oriGroup | batch/总通道/计算组通道/计算组数/原始组数；enlarge>1 时 cinG/coutG 已放大 |
| uint32_t | di/hi/wi、dout/ho/wo、dk/hk/wk | 输入输出与核几何 |
| uint32_t | strideD/H/W、dilationD/H/W、六侧 pad | 正向属性与采样 |
| int32_t | backpropPadTail、backpropPadUp/Down/Left/Right | 与裁剪原点共同使用的反向边界 |
| uint32_t | baseM/baseN/baseK、stepKa/stepKb | 计算块与驻留 |
| uint32_t | singleCoreCin/Cout/Din/Group、singleIterateDk | 任务通道/深度/组及核深度归约 |
| uint8_t | al1/bl1/al0/bl0/cl0Pbuffer、iterateOrder、c0 | Buffer 份数、遍历与粒度 |
| uint8_t | c0BitsA、c0BitsB、hf32Flag | FP16/BF16 为 4/4/0，FP32 为 3/3/0 |
| uint8_t | initOutputFlag、enableFullLoad | 初始化与驻留策略 |
| uint8_t | isBiasFullLoad | 可选 bias 的全量/分块通道准备方式 |
| uint32_t | kSCoutFullLoad、kSUseWorkSpace | kernel-split 的通道全载和重排 workspace 开关；全载不等于使用 workspace |
| uint8_t | enableVecTrans | 前置转置开关；见 [指令映射 §3.5](instruction-mapping.md) |
| 编译配置 | enableC04Flag | 由 loadB1Condition=TPL_VEC_TO_L1_C04 的入口实例化得到，不是原 TilingData 字段 |
| uint8_t | enlarge | 合并倍数；原始组数发布到 oriGroup，计算组数发布到 group |

其余派生量按公共 Tiling 的字段表在 Host 填写或在 Init 中复算，选择唯一生产方式。资源容量、L1A/L1B 槽大小可由同一目标的编译配置绑定，或随共享数据传入；它们必须与 Host 预算一致。结构尾部按序列化要求补齐，所有填充置零。

共享数据使用本工程读取的字段集合，字段名、单位和计算遵从对应实现。它不是原完整 ABI 的二进制替代；不包含未实现功能的读取者。使用普通 struct 时检查 padding、字段偏移与总长度；设备不得用另一份结构解释相同字节。

Transpose 复用三实参编码，将声明名和 TilingData 注册名绑定到自己的算子类型，使用 [八参数入口](kernel-template.md)。bias 存在性由外部指针与 Host 检查保持一致，不增加 key 位。isBiasFullLoad 与 enableFullLoad 分别发布；DX 不消费 bias 资源。

## 4 Host 流程与发布

```text
input_size / filter / out_backprop + attributes + three formats
    |
    v
Normalize spec; check shapes and groups
    |
    v
platformInfo + API capabilities
    |
    v
Template IsCapable / corresponding DoLibApiTiling solver
    |
    v
Final loadB2Condition; check config / resources / field ranges
    |
    v
Output init flag / shared data / key / blockDim / workspace
```

求解器中的 InitBaseMNK、SetSingleCoreInfo、CalStepK、LegalProtection 按模板文档实现为本地函数。调用封装只转换输入与发布结果；同一问题和能力下，两种调用方式应得到相同计算字段。loadB2Condition 使用本页判定，任务、K 父段和深度采样使用 [计算流程](../../../generic/input/compute-flow.md) 的公式；生成工程在自身编译单元内定义相应函数。

带 bias 时，最终 baseN、singleCoreCin 形成后重做 UpdateIsBiasFullLoad、实际 L1/BT 分配与能力检查。input_size/output_padding 的解析和初始化标志由相应算子的前置处理提供；公共求解不重新套用另一算子的形状准入条件。

注册发布片段：

```cpp
auto *raw = context->GetRawTilingData();
const auto bytes = tilingData.GetDataSize();
if (!raw || !raw->GetData() || bytes > raw->GetCapacity()) return ge::GRAPH_FAILED;
context->SetTilingKey(tilingKey);
context->SetBlockDim(coreNum);
tilingData.SaveToBuffer(raw->GetData(),raw->GetCapacity());
raw->SetDataSize(bytes);
```

大乘积与地址先用宽整数检查，反向边界保留符号。注册数据长度和对齐满足 SDK；Device Tiling 内存、输入输出和 workspace 保持到异步执行结束。融合初始化和数据准备按 [C/V 协同](cv-collaboration.md) 发布核类型及调度设置；使用 MODE0 的分离路径保留 SetScheduleMode(1)。
