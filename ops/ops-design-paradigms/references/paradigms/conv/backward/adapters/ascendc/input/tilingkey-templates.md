# 卷积反向算子 Tiling 数据与模板绑定

Host 按 [模板选择](../../../generic/input/tiling/tiling.md) 求解参数，再按最终块形状选择 loadB2Condition；Kernel 通过共享字段和编译实例消费结果。模板选择与 key 编码分别执行。

## 1 当前执行配置

采用原始分组寻址（按 G 组原始通道划分，groupConvMode=0）、GM→L1 权重搬入（loadB1Condition=0）、完整核深度归约，以及下表中的三种 weight 变换。运行时配置为 `singleIterateDk=dk、iterateOrder=1`；standard / small-shape / l1-resident 的 enableFullLoad 分别为 0 / 0 / 1。

普通通道路径同时满足 `c0=16、enableC04Flag=false、tilingHkWkMode=0`。发布时核对选中路径的 `kernelSplitMode=0、isBasicBlockTiling=true、loadB1Condition=0、groupConvMode=0`，以及 `enableVecTrans=enableLocalW=loadB1FractalZ=0`、`enableSplitK=useUbAccumForSplitK=false`、`kSCoutFullLoad=kSUseWorkSpace=0`（量化残留字段 offsetX 恒为 0，不参与本路径判定）。这些是执行配置检查，不能通过改写原选择结果扩大准入。

### 1.1 三模板与求解器的对应

三模板与算子包 arch35 求解器的对应关系：

| 范式模板 | arch35 求解器（op_host/op_tiling 下同名 *_tiling 单元） | 说明 |
|---|---|---|
| standard | `conv3d_backprop_input_v2_inner_product_tiling` | 经验选块与常规任务分配 |
| small-shape | `conv3d_backprop_input_v2_small_shape_tiling` | 小任务量 M/N 均衡 |
| l1-resident | `conv3d_backprop_input_v2_fullLoad_tiling` | 完整 K 权重驻留 |

算子包另含 `kernel_split` / `kernel_split_fullLoad`（kernelSplitMode=TPL_SPLIT_KERNEL_HW/H）、`small_kernel`（loadB1Condition=TPL_SMALL_KERNEL）、C04（TPL_VEC_TO_L1_C04）等扩展执行分支；本范式当前执行配置固定 `kernelSplitMode=0、loadB1Condition=0`（见上文"当前执行配置"），这些分支不在生成范围。arch22 的 basic_block/vec 求解器属 A2 平台路径，不在本范式范围。

## 2 loadB2Condition 判定与 key

在第 1 节配置内，`GetLoadB2ConditionByFormatAndKernel` 按以下顺序使用最终块选择 loadB2Condition。几何已通过正值检查，核体积乘积可由 uint64_t 表示：

```cpp
const uint64_t kernelHW = uint64_t(hk) * wk;
const uint64_t kernelDHW = uint64_t(dk) * kernelHW;
if (kernelDHW == 1) return TPL_TRANSPOSE_ONLY;
if (filterFormat == ge::FORMAT_NDHWC && baseN >= kernelHW) return TPL_REVERSE_ONLY;
if (filterFormat == ge::FORMAT_NCDHW && kernelDHW <= 64 / weightBytes / cin) {
    return TPL_REVERSE_ONLY;
}
return TPL_TRANSPOSE_AND_REVERSE;
```

片段中的几何和 weightBytes 来自归一化问题，filterFormat 使用 ge::Format；返回值使用下文共享头的 TPL 常量。`cin` 是总输入通道数；判定不使用 cinG。64B 条件用正整数的等价除法避免乘积溢出。核体积为 1 的判断优先于格式和 64 字节阈值；即使 `cin*weightBytes<=64`，也选择仅转置。loadB2Condition 同时决定 GM→L1B 的父布局，映射见 [布局说明](instruction-mapping.md)。

按这一确定性结果检查整条搬运链；所选路径不满足 API 或资源条件时返回具体原因。

共享头文件保留完整的声明列表顺序和位宽，`ASCENDC_TPL_SEL` 只选择本范式的三个组合。列表中的位置参与编码，不能通过缩短列表来裁剪实现范围。

```cpp
#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_TRANSPOSE_AND_REVERSE 1
#define TPL_REVERSE_ONLY 2
#define TPL_TRANSPOSE_ONLY 3
#define TPL_NO_SPLIT_KERNEL 0
#define TPL_GROUP_MODE_ORIGIN 0
#define TPL_GM_TO_L1 0

ASCENDC_TPL_ARGS_DECL(Conv3DDxCustom,
    ASCENDC_TPL_UINT_DECL(loadB2Condition, ASCENDC_TPL_8_BW,
        ASCENDC_TPL_UI_LIST, 0, 1, 2, 3),
    ASCENDC_TPL_UINT_DECL(kernelSplitMode, ASCENDC_TPL_8_BW,
        ASCENDC_TPL_UI_LIST, 0, 1, 2),
    ASCENDC_TPL_UINT_DECL(groupConvMode, ASCENDC_TPL_8_BW,
        ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_BOOL_DECL(isBasicBlockTiling, 0, 1),
    ASCENDC_TPL_UINT_DECL(loadB1Condition, ASCENDC_TPL_8_BW,
        ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(loadB2Condition, ASCENDC_TPL_UI_LIST, TPL_TRANSPOSE_AND_REVERSE),
        ASCENDC_TPL_UINT_SEL(kernelSplitMode, ASCENDC_TPL_UI_LIST, TPL_NO_SPLIT_KERNEL),
        ASCENDC_TPL_UINT_SEL(groupConvMode, ASCENDC_TPL_UI_LIST, TPL_GROUP_MODE_ORIGIN),
        ASCENDC_TPL_BOOL_SEL(isBasicBlockTiling, 1),
        ASCENDC_TPL_UINT_SEL(loadB1Condition, ASCENDC_TPL_UI_LIST, TPL_GM_TO_L1)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(loadB2Condition, ASCENDC_TPL_UI_LIST, TPL_REVERSE_ONLY),
        ASCENDC_TPL_UINT_SEL(kernelSplitMode, ASCENDC_TPL_UI_LIST, TPL_NO_SPLIT_KERNEL),
        ASCENDC_TPL_UINT_SEL(groupConvMode, ASCENDC_TPL_UI_LIST, TPL_GROUP_MODE_ORIGIN),
        ASCENDC_TPL_BOOL_SEL(isBasicBlockTiling, 1),
        ASCENDC_TPL_UINT_SEL(loadB1Condition, ASCENDC_TPL_UI_LIST, TPL_GM_TO_L1)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(loadB2Condition, ASCENDC_TPL_UI_LIST, TPL_TRANSPOSE_ONLY),
        ASCENDC_TPL_UINT_SEL(kernelSplitMode, ASCENDC_TPL_UI_LIST, TPL_NO_SPLIT_KERNEL),
        ASCENDC_TPL_UINT_SEL(groupConvMode, ASCENDC_TPL_UI_LIST, TPL_GROUP_MODE_ORIGIN),
        ASCENDC_TPL_BOOL_SEL(isBasicBlockTiling, 1),
        ASCENDC_TPL_UINT_SEL(loadB1Condition, ASCENDC_TPL_UI_LIST, TPL_GM_TO_L1)));
```

**核类型声明（registry/aclnn 编译必填，取值与初始化协议配对）**：Cube kernel 走注册编译时，在模板参数声明末位追加 `ASCENDC_TPL_KERNEL_TYPE_DECL(KT, <类型>)`，类型取值与所选初始化协议一致：主计算前独立初始化取 `ASCENDC_TPL_AIC_ONLY`；融合 AIV 清零取 `ASCENDC_TPL_MIX_AIC_1_2`（AIV 参与启动，等价于算子包在 Kernel 源内声明的 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`）。**配对必须一致：融合初始化误声明 AIC_ONLY 会使 AIV 清零缺失、AIC 等待通知永久挂起；独立初始化误声明混合类型则空跑 AIV 份**。直调不经注册元数据，核类型/调度由启动侧配置承载。

Host 在配置检查通过后调用共享声明生成 key：

```cpp
const uint64_t tilingKey = GET_TPL_TILING_KEY(
    loadB2Condition, 0, 0, true, 0);
```

| 五参数语义值 | `tilingKey` | L1B→L0B |
|---|---:|---|
| `(1,0,0,true,0)` | 16777217 | 转置加逆序 |
| `(2,0,0,true,0)` | 16777218 | 逆序 |
| `(3,0,0,true,0)` | 16777219 | 仅转置；核体积为 1 |

前面三个 UINT 字段分别占 8 位，BOOL 占 1 位，最后一个 UINT 接在第 25 位。当前三个组合均为 `2^24 + loadB2Condition`；实际代码通过 SDK 宏编码。构建实例和 Host 共用声明，切换 SDK 后重新核对结果。

## 3 共享字段

Host/Device 使用同一份定义。注册用 BEGIN_TILING_DATA_DEF/REGISTER_TILING_DATA_CLASS 生成对应算子的字段；直调可按同一字段定义共享 POD。两种传递方式各自验证编码和设备解码。

| 类型 | 字段组 | 用途 |
|---|---|---|
| uint64_t | coreNum、singleCoreBatch、singleCoreM | 启动计算核数、任务 batch/空间范围 |
| uint32_t | batch、cin/cout、cinG/coutG、group | batch/总通道/组内通道/分组 |
| uint32_t | di/hi/wi、dout/ho/wo、dk/hk/wk | 输入输出与核几何 |
| uint32_t | strideD/H/W、dilationD/H/W、六侧 pad | 正向属性与采样 |
| int32_t | backpropPadTail、backpropPadUp/Down/Left/Right | 与裁剪原点共同使用的反向边界 |
| uint32_t | baseM/baseN/baseK、stepKa/stepKb | 计算块与驻留 |
| uint32_t | singleCoreCin/Cout/Din/Group、singleIterateDk | 任务通道/深度/组及核深度归约 |
| uint8_t | al1/bl1/al0/bl0/cl0Pbuffer、iterateOrder、c0 | Buffer 份数、遍历与粒度 |
| uint8_t | initOutputFlag、enableFullLoad | 初始化与驻留策略 |
| uint8_t | isBiasFullLoad | 可选 bias 的全量/分块通道准备方式 |

其余派生量按公共 Tiling 的字段表在 Host 填写或在 Init 中复算，选择唯一生产方式。资源容量、L1A/L1B 槽大小可由同一目标的编译配置绑定，或随共享数据传入；它们必须与 Host 预算一致。结构尾部按序列化要求补齐，所有填充置零。

共享数据使用本工程读取的字段集合，字段名、单位和计算遵从对应实现。它不是原完整 ABI 的二进制替代；不包含未实现功能的读取者。使用普通 struct 时检查 padding、字段偏移与总长度；设备不得用另一份结构解释相同字节。

Transpose 复用五参数编码，将声明名和 TilingData 注册名绑定到自己的算子类型，使用 [八参数入口](kernel-template.md)。bias 存在性由外部指针与 Host 检查保持一致，不增加 key 位。isBiasFullLoad 与 enableFullLoad 分别发布；DX 不消费 bias 资源。

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

大乘积与地址先用宽整数检查，反向边界保留符号。注册数据长度和对齐满足 SDK；Device Tiling 内存、输入输出和 workspace 保持到异步执行结束。融合初始化按同步文档发布对应核类型及调度设置。
