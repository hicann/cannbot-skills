# D1: Mixed Precision Architecture (混合精度架构)
## Overview
baseline实现仅支持 float32 输入类型，这在实际 AI 推理场景中是不足的。现代大语言模型和推理框架通常使用 FP16 或 BF16 进行计算以减少内存带宽和存储需求。专家实现通过模板参数 typename T 支持 half (FP16) 和 bfloat16_t (BF16) 两种输入类型。关键技术细节包括：类型特化的计算精度（当输入为 half 时，计算类型 calcType 提升为 float，避免 FP16 精度损失；当输入为 bfloat16_t 时，直接以 BF16 计算）、条件类型推导使用 std::conditional 实现编译期类型选择、MicroAPI 自动处理中 FP16 输入会自动通过 Cast 转换为 FP32 进行计算而 BF16 保持原格式。

## When to Use
- Any op with FP16/BF16 input
- 支持FP16/BF16可显著减少内存带宽占用（50%），提升内存受限场景性能；编译期类型选择无运行时开销
- 支持BF16精度，代码复用率高，零运行时开销
- 运行时动态选择量化格式，提高算子灵活性

## Trade-off
- 代码复杂度增加，需要维护多类型模板实例
- 模板代码复杂度增加，编译时间可能增加
- 需要在运行时检查属性，略微增加开销

**Source operators**: adaptive_max_pool3d_grad, add_rms_norm_cast, add_rms_norm_dynamic_quant, apply_adagrad_d, batch_norm_v3, deep_norm, dynamic_mx_quant, fake_quant_affine_cachemask, foreach_abs, foreach_add_scalar, foreach_add_scalar_list, foreach_addcdiv_list, gather_elements_v2, gemma_rms_norm, grouped_dynamic_mx_quant, layer_norm_v3, layer_norm_v4, masked_scatter_with_position, max_pool_grad_with_argmax_common, max_pool_with_argmax_v3, modulate, multi_scale_deformable_attn_function, norm_common, rms_norm_grad, rms_norm_quant, scaled_masked_softmax_grad_v2, sparse_to_dense, trans_quant_param_v2

---

## Variant A: 丰富的数据类型矩阵支持
Source: foreach_add_scalar, layer_norm_v3, max_pool_grad_with_argmax_common, rms_norm_grad

专家实现支持4种主要数据类型（FP16、FP32、INT32、BF16），并且针对不同的硬件平台进行了差异化配置。在foreach_add_scalar_def.cpp中，通过OpAICoreConfig为不同芯片配置不同的数据类型支持矩阵。特别值得注意的是，BF16类型仅在__CCE_AICORE__ >= 220的架构上支持，通过条件编译实现架构适配。此外，scalar输入的数据类型也根据tensor数据类型进行了精细化配置，例如BF16和FP32类型的tensor可以接受FP64/FP32类型的scalar，而FP16类型的tensor在910D平台上可以接受更广泛的scalar类型。这种设计允许在不牺牲性能的前提下，提供更灵活的API接口。

**Expert implementation:**
```cpp
this->Input("x")
    .ParamType(DYNAMIC)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .AutoContiguous();
this->Input("scalar")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .AutoContiguous();
```

**vs. baseline:**
```cpp
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND});
this->Output("y")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND});
```

Benefit: 支持更广泛的业务场景，减少数据类型转换开销，提高端到端性能; 覆盖更广的精度需求，支持2-4倍内存带宽优化，适应不同硬件平台和精度要求; 一个算子支持多种数据类型组合，减少算子数量，简化用户使用；AutoContiguous自动处理非连续输入; FP16模式可减少50%内存带宽占用；在支持BF16的硬件上可进一步提升性能；梯度计算使用FP32避免精度损失
Trade-off: 增加了代码复杂度和测试矩阵规模; 增加代码复杂度，需要更多tiling key和模板实例化; 需要更多的tiling key来区分不同的数据类型组合; 代码复杂度增加（需要处理类型转换）；kernel需要模板化以支持不同数据类型

---

## Variant B: 动态输入列表支持
Source: foreach_abs, foreach_add_scalar_list

专家实现通过ParamType(DYNAMIC)支持动态输入列表（TensorList），这是PyTorch foreach系列算子的核心特性。与baseline仅支持单一Tensor不同，专家实现可以一次处理多个Tensor，通过索引访问每个Tensor的数据。KernelForeachBase::GetTensorAddr方法实现了通过索引获取Tensor地址的逻辑，利用64位指针数组存储每个Tensor的偏移地址。

**Expert implementation:**
```cpp
this->Input("x")
    .ParamType(DYNAMIC)  // 动态输入列表
    .DataType(tensor_dtype_list)
    .AutoContiguous();
```

**vs. baseline:**
```cpp
this->Input("x").ParamType(REQUIRED);  // 单一Tensor
```

Benefit: 支持PyTorch foreach语义，可同时处理多个Tensor，提高批量处理效率; 支持更灵活的输入配置，符合PyTorch foreach语义
Trade-off: 需要处理TensorList的索引和地址计算，增加逻辑复杂度; 需要更复杂的Tiling逻辑来支持多tensor负载均衡

---

## Variant C: 多数据类型统一模板设计
Source: adaptive_max_pool3d_grad

专家实现通过C++模板参数（TX, TGrad, TArgmax, TY）实现了对FP32、FP16、BF16的统一支持。这种设计允许在编译期确定数据类型，避免了运行时的类型判断开销。关键技巧在于通过is_same<TY, float>::value等编译期条件判断，为不同数据类型生成最优代码路径。对于非FP32类型（FP16/BF16）的重叠场景，专家实现使用workspace作为中间缓冲区，以FP32精度累加梯度，避免多次atomic add导致的精度损失。这是一种精度保持策略：在计算密集型操作中使用高精度，仅在最终输出时进行类型转换。

**Expert implementation:**
```cpp
this->Input("x")
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
    .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
    .AutoContiguous();

template <typename TX, typename TGrad, typename TArgmax, typename TY, bool IsOverlap>
class AdaptiveMaxPool3DGradNormal {
    if constexpr (is_same<TY, float>::value) {
        InitGlobalMemory(yGm, params_.ncDim * params_.diHiWiLen, 0.0f);
    } else {
        if constexpr (IsOverlap) {
            InitGlobalMemory(workspaceGm, params_.ncDim * params_.diHiWiLen, 0.0f);
        }
    }
};
```

**vs. baseline:**
```cpp
this->Input("grad_output")
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND});
this->Input("indices")
    .DataType({ge::DT_INT64})
```

Benefit: 支持FP16/BF16可显著减少内存带宽占用（50%），提升内存受限场景性能；编译期类型选择无运行时开销
Trade-off: 代码复杂度增加，需要维护多类型模板实例

---

## Variant D: 多数据类型统一模板实现
Source: add_rms_norm_cast

专家实现通过C++模板实现了对half和bfloat16_t两种数据类型的统一支持。在KernelAddRmsNormCast类中使用if constexpr进行编译期分支选择，FP16和BF16的处理流程有细微差异：BF16需要在加法前转换为FP32，加完后转回BF16；而FP16可以直接使用half类型的Add指令。输出y1固定为FP32类型，y2保持与输入相同的数据类型。这种设计带来高代码复用率、零运行时开销，并支持LLM训练中常用的BF16精度。

**Expert implementation:**
```cpp
template <typename T>
class KernelAddRmsNormCast {
private:
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<float> y1Gm;
    GlobalTensor<T> y2Gm;
};

if constexpr (is_same<T, half>::value) {
    Cast(x1_fp32, xLocal, RoundMode::CAST_NONE, numCol);
} else {
    Cast(x1_fp32, xLocal, RoundMode::CAST_NONE, numCol);
    Cast(xLocal, x1_fp32, RoundMode::CAST_RINT, numCol);
}
```

**vs. baseline:**
```cpp
class KernelAddRmsNormCast {
private:
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> residualGm;
    AscendC::GlobalTensor<half> weightGm;
    AscendC::GlobalTensor<half> outputGm;
};
```

Benefit: 支持BF16精度，代码复用率高，零运行时开销
Trade-off: 模板代码复杂度增加，编译时间可能增加

---

## Variant E: 动态输出数据类型推导
Source: add_rms_norm_dynamic_quant

专家实现在InferDataType阶段通过dst_type属性支持动态输出类型选择。当用户指定不同的dst_type时，算子会自动推导y1和y2的输出数据类型。这种设计提供了极大的灵活性，允许用户在不修改算子代码的情况下切换不同的量化格式。

**Expert implementation:**
```cpp
static const std::initializer_list<ge::DataType> OUT_TYPE_LIST = {
    DT_INT8, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN
};
const int32_t* pDstDtype = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_OF_DST_TYPE);
if (pDstDtype != nullptr) {
    yDtype = static_cast<ge::DataType>(*pDstDtype);
}
context->SetOutputDataType(Y1_IDX, yDtype);
```

**vs. baseline:**
```cpp
context->SetOutputDataType(0, ge::DT_INT8);
context->SetOutputDataType(1, ge::DT_FLOAT);
```

Benefit: 运行时动态选择量化格式，提高算子灵活性
Trade-off: 需要在运行时检查属性，略微增加开销

---

## Variant F: 双输出动态量化支持
Source: add_rms_norm_dynamic_quant

专家实现支持同时对输入进行两种不同的smooth scale量化，输出y1和y2。这是通过output_mask属性控制的，适用于MoE等需要多版本量化结果的场景。baseline实现仅支持单输出。

**Expert implementation:**
```cpp
// 双输出
GM_ADDR y1, GM_ADDR y2, GM_ADDR outScale1, GM_ADDR outScale2
const gert::ContinuousVector* outputMaskAttr = attrs->GetAttrPointer<gert::ContinuousVector>(OUT_QUANT_1_IDX);
this->outQuant1Flag = (scalesArray[0] == true) ? 1 : 0;
this->outQuant2Flag = (scalesArray[1] == true) ? 1 : 0;
```

**vs. baseline:**
```cpp
// 单输出
GM_ADDR output, GM_ADDR scale
```

Benefit: 支持MoE等复杂场景，一次计算输出两种量化结果
Trade-off: 增加内存占用和计算复杂度

---

## Variant G: 可选输入灵活处理
Source: add_rms_norm_dynamic_quant

专家实现支持smooth_scale1、smooth_scale2和beta等可选输入，通过CheckOptionalShapeExisting函数检查输入是否存在。baseline实现仅支持固定的x1、x2、weight三个输入。

**Expert implementation:**
```cpp
this->Input("smooth_scale1").ParamType(OPTIONAL);
this->Input("smooth_scale2").ParamType(OPTIONAL);
this->Input("beta").ParamType(OPTIONAL);
bool CheckOptionalShapeExisting(const gert::StorageShape* smoothShape) {
    if(nullptr == smoothShape) return false;
    int64_t smoothShapeSize = smoothShape->GetOriginShape().GetShapeSize();
    return smoothShapeSize > 0;
}
```

**vs. baseline:**
```cpp
this->Input("x1").ParamType(REQUIRED);
this->Input("x2").ParamType(REQUIRED);
this->Input("weight").ParamType(REQUIRED);
```

Benefit: 支持更灵活的调用方式，适配不同模型需求
Trade-off: 增加条件判断开销和代码复杂度

---

## Variant H: 统一的类型检查机制
Source: apply_adagrad_d

专家实现在Host端实现了严格的类型一致性检查。CheckDtype()函数确保所有输入(var, accum, lr, grad)和输出(var_out, accum_out)具有相同的数据类型，避免了隐式类型转换带来的性能损失和潜在的精度问题。CheckShape()函数确保输入张量shape匹配，防止运行时错误。这种设计提前发现问题，提高了算子的健壮性。

**Expert implementation:**
```cpp
ge::graphStatus ApplyAdagradDTiling::CheckDtype() {
    auto varDesc = tilingContext_->GetInputDesc(0);
    this->varDtype_ = varDesc->GetDataType();
    
    for (int32_t inputIdx = ACCUM_INDEX; inputIdx < INPUT_NUM; inputIdx++) {
        auto inputDesc = tilingContext_->GetInputDesc(inputIdx);
        auto curDtype = inputDesc->GetDataType();
        OP_CHECK_IF(curDtype != varDtype_,
                    OP_LOGE(tilingContext_->GetNodeName(), "Input %d dtype not match..."),
                    return ge::GRAPH_FAILED);
    }
    // ... 输出类型检查
}
```

**vs. baseline:**
```cpp
// baseline无显式类型检查
def apply_adagrad_d_host(var, accum, lr, grad, update_slots=True):
    total_elems = var.numel()
    # 直接进行计算
```

Benefit: 防止类型不匹配导致的运行时错误，避免隐式类型转换的开销，提高算子健壮性
Trade-off: 增加了Host端代码量，有额外的运行时检查开销（但相对kernel执行可忽略）

---

## Variant I: 混合精度计算架构
Source: batch_norm_v3

专家实现采用了输入低精度+计算高精度的混合精度架构。输入数据x、weight、bias支持FP16、BF16、FP32三种类型，但所有计算过程（均值、方差、归一化）都在FP32精度下完成。这种设计在保持计算精度的同时，减少了Global Memory的带宽压力。关键实现包括：1）数据类型检查与转换，确保weight/bias的数据类型一致，且running_mean/running_var必须是FP32；2）原地Cast优化，通过ReinterpretCast和Cast指令将低精度数据转换为FP32进行计算，避免额外的内存分配；3）输出精度控制，根据输出数据类型选择不同的RoundingMode，BF16使用CAST_ROUND而FP16使用CAST_NONE。

**Expert implementation:**
```cpp
// 专家实现支持FP16/BF16/FP32
static inline bool IsDtypeSupported(const ge::DataType dtype)
{
    return ((dtype == ge::DT_FLOAT16) || (dtype == ge::DT_BF16) || (dtype == ge::DT_FLOAT));
}
// 原地Cast
if constexpr (!IsSameType<T1, float>::value) {
    LocalTensor<T1> xTensorHalf = xTensor.template ReinterpretCast<T1>();
    Cast(xTensor, xTensorHalf[r0UbFactor], RoundMode::CAST_NONE, copyInSize);
}
```

**vs. baseline:**
```cpp
# baseline仅支持float32
data_ub = tl.alloc_ub(tile_size, dtype=tl.float32)
```

Benefit: 减少50%的Global Memory带宽占用，同时保持FP32计算精度；支持BF16新数据类型，提升新一代硬件利用率
Trade-off: 需要额外的Cast指令开销；需要处理不同RoundingMode的配置

---

## Variant J: 数据类型与精度控制分离
Source: deep_norm

专家实现将数据存储类型与计算精度分离。对于 FP16 和 BF16 输入，在计算过程中会转换为 FP32 进行中间计算，最后再转换回原始精度输出。这种设计利用了 Ascend C 的 Cast 指令和 RoundMode 参数，在保证数值稳定性的同时最大化计算吞吐量。特别是在 DeepNorm 这种涉及累加和方差计算的场景下，使用 FP32 中间结果可以有效避免低精度累加导致的数值溢出和精度损失。

**Expert implementation:**
```cpp
// 专家实现 - FP16 输入转 FP32 计算
Cast(local_y_fp32, x_local, RoundMode::CAST_NONE, stepSize);
PipeBarrier<PIPE_V>();
Cast(local_x_fp32, gx_local, RoundMode::CAST_NONE, stepSize);
PipeBarrier<PIPE_V>();
// FP32 精度计算
Axpy(local_x_fp32, local_y_fp32, alphaVal, stepSize);
// ... 计算完成后转回 FP16
Cast(z_local, local_y_fp32, RoundMode::CAST_NONE, stepSize);
```

**vs. baseline:**
```cpp
// baseline - 全程 FP32 计算
float rowMean = ComputeMean(rowIdx);
float rowStd = ComputeStd(rowIdx, rowMean);
NormalizeScaleShift(rowIdx, rowMean, rowStd);
```

Benefit: 保证低精度输入的数值稳定性，避免累加误差和溢出
Trade-off: 额外的类型转换开销，需要额外的 FP32 缓冲区

---

## Variant K: 输入数据类型扩展 (FP16/BF16)
Source: dynamic_mx_quant

baseline实现仅支持 float32 输入类型，这在实际 AI 推理场景中是不足的。现代大语言模型和推理框架通常使用 FP16 或 BF16 进行计算以减少内存带宽和存储需求。专家实现通过模板参数 typename T 支持 half (FP16) 和 bfloat16_t (BF16) 两种输入类型。关键技术细节包括：类型特化的计算精度（当输入为 half 时，计算类型 calcType 提升为 float，避免 FP16 精度损失；当输入为 bfloat16_t 时，直接以 BF16 计算）、条件类型推导使用 std::conditional 实现编译期类型选择、MicroAPI 自动处理中 FP16 输入会自动通过 Cast 转换为 FP32 进行计算而 BF16 保持原格式。

**Expert implementation:**
```cpp
this->Input("x")
    .ParamType(REQUIRED)
    .DataType(
        {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ...})
    .Format({ge::FORMAT_ND, ...})
    .AutoContiguous();

// Kernel 端动态计算类型选择
template <typename T, typename U, const bool ISTAIL>
class DynamicMxQuantNotTailAxis : public DynamicMxQuantBase<T, U, ISTAIL> {
private:
    using calcType = typename std::conditional<IsSame<T, half>::value, float, T>::type;
    using calcTypeInt = typename std::conditional<IsSame<T, half>::value, uint32_t, uint16_t>::type;
};
```

**vs. baseline:**
```cpp
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND});
```

Benefit: 支持现代 AI 模型常用的 FP16/BF16 格式，减少 50% 内存带宽，提升端到端推理性能 20-40%
Trade-off: 代码复杂度增加，需要模板特化和条件编译；当 FP16 转 FP32 计算时寄存器占用增加

---

## Variant L: 数据类型感知的Buffer分配
Source: fake_quant_affine_cachemask

专家实现在Buffer分配时充分考虑了不同数据类型的内存需求。在FP16实现中，由于需要在FP16和FP32之间进行转换，分配了更多的计算缓冲区（calcBuf, calcHf16Buf, calcInt32Buf）。而在FP32实现中，主要使用单一的calcBuf。这种差异化分配策略确保了内存使用效率的最大化。例如，FP16实现中的calcTemp使用float类型进行中间计算以保证精度，而FP32实现则可以直接使用原生类型。

**Expert implementation:**
```cpp
// FP16实现 - 多Buffer支持类型转换
pipe.InitBuffer(calcBuf, BUFFER_NUM * this->tileLength * sizeof(float));
pipe.InitBuffer(calcHf16Buf, BUFFER_NUM * this->tileLength * sizeof(yType));
pipe.InitBuffer(calcInt32Buf, BUFFER_NUM * this->tileLength * sizeof(int32_t));

// FP32实现 - 简化Buffer分配
pipe.InitBuffer(calcBuf, BUFFER_NUM * this->tileLength * sizeof(yType));
pipe.InitBuffer(calcInt32Buf, BUFFER_NUM * this->tileLength * sizeof(int32_t));
```

**vs. baseline:**
```cpp
// baseline统一使用float buffer
pipe.InitBuffer(xScaledBuf, this->tileSize * sizeof(float));
pipe.InitBuffer(xqRoundedBuf, this->tileSize * sizeof(float));
pipe.InitBuffer(xqBuf, this->tileSize * sizeof(float));
```

Benefit: 根据数据类型优化内存使用，FP16实现通过多Buffer支持精度转换，FP32实现简化Buffer分配
Trade-off: 代码复杂度增加，需要为不同数据类型维护不同的Buffer分配逻辑

---

## Variant M: Host端多类型Proto配置
Source: foreach_addcdiv_list

专家实现在Host端通过foreach_proto_utils.h中的宏实现了统一的多类型配置管理。使用std::vector<ge::DataType>定义支持的类型列表，并通过DYNAMIC ParamType支持动态TensorList输入。kirinx90平台不支持BF16，因此单独配置了类型列表。使用AutoContiguous()确保数据连续性。

**Expert implementation:**
```cpp
// 专家实现: 多平台多类型配置
std::vector<ge::DataType> tensor_dtype_list = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
this->AICore().AddConfig("ascend910_95");
this->AICore().AddConfig("ascend910_93");
this->AICore().AddConfig("ascend910b");
// Kirin平台特化（不支持BF16）
OpAICoreConfig config_kirin = GetKirinCoreConfig();
std::vector<ge::DataType> tensor_dtype_list_kirin = {ge::DT_FLOAT16, ge::DT_FLOAT};
this->AICore().AddConfig("kirinx90", config_kirin);
```

**vs. baseline:**
```cpp
// baseline: 单类型配置
this->Input("input")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND});
this->AICore().AddConfig("ascend910b");
```

Benefit: 跨平台兼容性，针对不同硬件能力配置不同数据类型支持
Trade-off: 增加了配置代码量

---

## Variant N: 丰富的数据类型支持
Source: gather_elements_v2

专家实现支持BF16、FP16、FP32、INT32四种输入数据类型，而baseline仅支持FP32。这种多数据类型支持不仅提供了更好的灵活性，更重要的是可以根据实际场景选择精度-性能最佳平衡点。BF16和FP16可以在保证足够精度的前提下提供更高的吞吐量和更低的内存带宽压力。实现上通过ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32在OpDef中声明，并在kernel端通过模板参数T_X和条件编译实现类型特化。特别值得注意的是，对于Kirin平台（x90），专家实现还做了特定的数据类型裁剪，移除了BF16支持，这可能是基于硬件能力和实际需求的考虑。

**Expert implementation:**
```cpp
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

// 类型特化
if constexpr (std::is_same<DTYPE_X, bfloat16_t>::value) {
    AscendC::GatherElementsV2TransposeKernel<half, int32_t> op(...);
}
```

**vs. baseline:**
```cpp
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND});
```

Benefit: 提供精度-性能权衡选择，BF16/FP16可获得更高吞吐量和更低内存带宽压力
Trade-off: 增加了类型处理的复杂度，需要模板和条件编译支持

---

## Variant O: 数据类型尺寸自适应
Source: gather_elements_v2

在LastDim模式下，专家实现针对不同数据类型尺寸采用了自适应的UB内存分配策略。对于INT8数据类型，通过将数据尺寸上取整到2字节来简化内存对齐处理；对于INT32索引，上取整到8字节以更好地利用向量指令。这种处理虽然增加了少量内存开销，但显著简化了向量化代码的编写，避免了复杂的非对齐访问处理。

**Expert implementation:**
```cpp
void GatherElementsV2LastDimTiling::GetDSize() {
    xDSize_ = ge::GetSizeByDataType(xDataType);
    if (xDSize_ == INT8_DSIZE) {
        xDSize_ = INT6_DSIZE;  // 将INT8提升到2字节对齐
        xDsizeRatio_ = DOUBLE_TIME;
    }
    if (indexDSize_ == INT32_DSIZE) {
        indexDSize_ = INT64_DSIZE;  // 将INT32提升到8字节对齐
    }
}
```

**vs. baseline:**
```cpp
uint32_t dtype_x_size = 4;   // float32
uint32_t dtype_idx_size = 4; // int32
```

Benefit: 简化向量化代码，避免复杂的非对齐访问处理
Trade-off: 增加少量内存开销

---

## Variant P: 混合精度计算路径优化
Source: gemma_rms_norm

专家实现针对不同数据类型组合设计了专门的计算路径。当输入为 FP16/BF16 时，会先将数据 Cast 到 FP32 进行平方和累加计算，然后再 Cast 回原始类型。这种设计在精度敏感的场景（如大模型推理）中至关重要，因为 FP16 直接累加容易导致精度损失。baseline实现缺少这种精度保护机制。此外，专家实现还区分了 IS_MIX_DTYPE 场景（输入和权重类型不同），通过条件编译选择最优的计算路径。

**Expert implementation:**
```cpp
// 专家实现 - 混合精度计算
if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
    LocalTensor<float> xBufFp32 = x_fp32_buf.Get<float>();
    Cast(xBufFp32, xLocal, RoundMode::CAST_NONE, num_col);
    PipeBarrier<PIPE_V>();
    inQueueX.FreeTensor(xLocal);
    Mul(sqx, xBufFp32, xBufFp32, num_col);
} else {
    Mul(sqx, xLocal, xLocal, num_col);
}
```

**vs. baseline:**
```cpp
// baseline - 直接计算
AscendC::Mul(tempLocal, inputLocal, inputLocal, this->tileLength);
float tileSqSum = sharedLocal.GetValue(0);
```

Benefit: 在保持低内存占用的同时，通过 FP32 中间计算保护关键路径的精度
Trade-off: 需要额外的 UB 内存缓冲区（x_fp32_buf），增加了内存占用

---

## Variant Q: 数据类型校验与转换策略
Source: grouped_dynamic_mx_quant

专家实现在Host端建立了完整的数据类型校验体系。使用std::set预定义支持的数据类型集合，便于扩展和维护。不仅校验单个张量的类型，还校验输入x与输出y的数据类型对应关系，以及dst_type属性与输出y的实际数据类型是否匹配。这种严格的类型校验机制确保了算子在不同数据类型组合下的正确性。

**Expert implementation:**
```cpp
const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = { ge::DT_FLOAT16, ge::DT_BF16 };
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = { ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2 };

OP_CHECK_IF((tilingParam.outDtype == ge::DT_FLOAT8_E4M3FN && checkDstType != 36) ||
    (tilingParam.outDtype == ge::DT_FLOAT8_E5M2 && checkDstType != 35),
    OP_LOGE(context->GetNodeName(), "y's data type and dst_type is not corresponded..."),
    return ge::GRAPH_FAILED);
```

**vs. baseline:**
```cpp
this->Input("x").DataType({ge::DT_FLOAT16});
this->Output("y").DataType({ge::DT_FLOAT8_E4M3FN});
```

Benefit: 完整的类型安全校验；易于扩展新数据类型；属性与输出类型一致性保证
Trade-off: 增加运行时校验开销（通常在可接受范围内）

---

## Variant R: 全数据类型组合支持
Source: layer_norm_v4

专家实现支持完整的数据类型矩阵：输入数据类型（FLOAT32/FLOAT16/BFLOAT16）× 参数数据类型（FLOAT32/FLOAT16/BFLOAT16）。通过模板参数Tfm和Tweight实现编译期类型多态，每个类型组合对应独立tiling key，确保kernel端完全特化优化。baseline仅支持FLOAT32，在现代深度学习部署中构成重大限制。

**Expert implementation:**
```cpp
enum class LayerNormV4TilingKey : int64_t {
    LAYER_NORM_SINGLE_READ_FLOAT32_FLOAT32 = 100,
    LAYER_NORM_SINGLE_READ_FLOAT16_FLOAT32 = 110,
    LAYER_NORM_SINGLE_READ_FLOAT16_FLOAT16 = 111,
    LAYER_NORM_SINGLE_READ_BFLOAT16_FLOAT32 = 120,
    LAYER_NORM_SINGLE_READ_BFLOAT16_BFLOAT16 = 122,
    // ... 完整类型矩阵
};
```

**vs. baseline:**
```cpp
this->Input("x").DataType({ge::DT_FLOAT});
this->Input("weight").DataType({ge::DT_FLOAT});
this->Input("bias").DataType({ge::DT_FLOAT});
this->Output("y").DataType({ge::DT_FLOAT});
```

Benefit: 允许用户在精度和性能间灵活权衡，训练用FP32保证精度，推理用FP16/BF16提升吞吐量，支持现代混合精度训练部署
Trade-off: 代码复杂度增加，需要维护多个模板特化版本

---

## Variant S: 输入数据类型严格校验
Source: masked_scatter_with_position

专家实现在Host端实现了完整的数据类型校验逻辑`CheckDataType()`函数，确保每个输入张量的数据类型符合预期。x和updates必须具有相同的数据类型，mask必须是BOOL类型，position必须是INT64类型。这种严格的类型检查在编译期就能发现错误，避免在Kernel端出现难以调试的类型不匹配问题。

**Expert implementation:**
```cpp
// 专家实现 - 完整的数据类型检查
ge::graphStatus MaskedScatterWithPositionTiling::CheckDataType() {
    OP_CHECK_IF(SUPPORT_DTYPE.count(dType_) == 0, OP_LOGE(opName_,
        "The dtype only support float32, float16..."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(maskDtype != ge::DT_BOOL, OP_LOGE(opName_,
        "The dtype of mask only support bool..."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(positionDtype != ge::DT_INT64, OP_LOGE(opName_,
        "The dtype of position only support int64..."), return ge::GRAPH_FAILED);
}
```

**vs. baseline:**
```cpp
// baseline没有显式的类型检查
namespace ge {
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
}
```

Benefit: 在编译期发现类型错误，避免运行时难以调试的问题，提升代码健壮性
Trade-off: 增加代码量，需要在Host端实现额外的检查逻辑

---

## Variant T: 完整的数据类型覆盖
Source: max_pool_with_argmax_v3

专家实现在算子定义层就考虑了多数据类型支持。输入支持 DT_FLOAT16、DT_FLOAT、DT_BF16 三种浮点类型，输出索引支持 DT_INT32 和 DT_INT64。通过模板参数在编译期生成不同的 kernel 实例，避免了运行时的类型判断开销。与 baseline 仅支持 DT_FLOAT 输入和 DT_INT32 输出相比，专家实现不仅提高了算子的通用性，还通过编译期优化确保了不同类型下的执行效率。

**Expert implementation:**
```cpp
// 专家实现支持多数据类型
this->Input("x").DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ...});
this->Output("argmax").DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64});

// Kernel 模板参数化
template <typename T1, typename T2, const uint32_t IS_PAD = 0>
class MaxPoolWithArgmaxV3SmallC {
    // T1: 输入数据类型, T2: 索引类型
};
```

**vs. baseline:**
```cpp
// baseline 仅支持 FLOAT
this->Input("x").DataType({ge::DT_FLOAT});
this->Output("indices").DataType({ge::DT_INT32});
```

Benefit: 支持更广泛的业务场景（混合精度训练、边缘推理），编译期类型特化带来额外性能优化
Trade-off: 编译产物体积增大（多模板实例），host端需要更复杂的tiling key选择逻辑

---

## Variant U: 可选参数状态管理
Source: modulate

专家实现支持scale和shift两个可选参数，设计了parameterStatus字段来管理四种状态。这种设计的优化价值在于UB资源优化（根据参数状态调整UB Tensor数量）、计算路径优化（Kernel端根据状态选择不同的计算函数）、内存访问优化（避免无效参数的GM访问）。在参数可选场景下，可减少15-20%的UB使用量和计算量。

**Expert implementation:**
```cpp
// 专家实现: 动态参数状态管理
if (scaleShape && shiftShape) {
    this->tilingData.parameterStatus = SCALE_AND_SHIFT;
} else if (!scaleShape) {
    this->tilingData.parameterStatus = NO_SCALE;
} else if (!shiftShape) {
    this->tilingData.parameterStatus = NO_SHIFT;
}
int64_t ubTensorNum = (this->tilingData.parameterStatus == SCALE_AND_SHIFT) ? UB_TENSOR_NUM_ALL : UB_TENSOR_NUM;
// Kernel端选择计算路径
switch (this->parameterStatus) {
    case SCALE_AND_SHIFT:
        Compute(...);
        break;
    case NO_SCALE:
        ComputeWithoutScale(...);
        break;
    case NO_SHIFT:
        ComputeWithoutShift(...);
        break;
}
```

**vs. baseline:**
```cpp
// baseline: 固定处理所有参数
void Compute(int64_t tileSize) {
    LocalTensor<float> scaleLocal = inQueueScale.DeQue<float>();
    LocalTensor<float> shiftLocal = inQueueShift.DeQue<float>();
    // 固定计算 y = x * (1 + scale) + shift
}
```

Benefit: 参数可选场景下减少15-20%的UB使用量和计算量
Trade-off: 代码复杂度增加，需要维护三种计算路径

---

## Variant V: 动态数据类型推导与Shape适配
Source: multi_scale_deformable_attn_function

Host端通过InferShape和InferDataType函数实现了输出Shape和数据类型的自动推导。通过检查输入的samplingLocations维度，自动判断是否为Transpose模式（isTranspose），并据此推导输出Shape。这种策略使得算子可以灵活处理不同数据布局，无需用户手动指定。同时，数据类型推导直接继承value输入的数据类型，确保类型一致性。

**Expert implementation:**
```cpp
static ge::graphStatus InferShapeForMultiScaleDeformableAttnFunction(gert::InferShapeContext *context)
{
    bool isTranspose = samplingLocationsShape->GetDim(INPUT_LOCAT_DIM_1) < 32;
    uint64_t numHeads = INPUT_LOCAT_DIM_2;
    uint64_t numQueries = INPUT_LOCAT_DIM_1;
    if (isTranspose) {
        numHeads = INPUT_LOCAT_DIM_1;
        numQueries = INPUT_LOCAT_DIM_5;
    }
    // ... 设置输出shape
}

static ge::graphStatus InferDataTypeForMultiScaleDeformableAttnFunction(gert::InferDataTypeContext* context)
{
    const ge::DataType value_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, value_dtype);
    return GRAPH_SUCCESS;
}
```

**vs. baseline:**
```cpp
// baseline实现不存在，无法提供对比代码
```

Benefit: 灵活处理不同数据布局，无需用户手动指定Shape和数据类型
Trade-off: 增加了Shape推导的复杂性

---

## Variant W: 数据类型自动推导与校验
Source: norm_common

专家实现在Host端通过GetDTypeKey函数建立数据类型到整型Key的映射机制，实现自动化的数据类型组合识别。该机制不仅支持类型推导，还包含完整的数据类型校验逻辑——检查输入数据类型是否为浮点类型（FP32/FP16/BF16）、权重和偏置类型是否匹配、当权重类型与输入类型不同时是否满足约束（如权重必须是FP32）。baseline实现缺乏任何数据类型校验，仅支持单一类型，无法适应多样化的生产环境需求。

**Expert implementation:**
```cpp
int64_t GetDTypeKey(ge::DataType tensorDtype, ge::DataType paramDtype)
{
    auto GetKeyForDType = [](ge::DataType dtype) -> int64_t {
        switch (dtype) {
            case ge::DT_FLOAT: return 0;
            case ge::DT_FLOAT16: return 1;
            case ge::DT_BF16: return 2;
            default: return -1;
        }
    };
    return GetKeyForDType(tensorDtype) * 10 + GetKeyForDType(paramDtype);
}

ge::graphStatus InputDtypeCheck(...)
{
    OP_CHECK_IF(!isFloatDtype(xDtype), 
        OP_LOGE(context->GetNodeName(), "x dtype must be in float32, float16, bfloat16."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(gammaDtype != betaDtype,
        OP_LOGE(context->GetNodeName(), "gamma dtype must be the same as beta dtype."),
        return ge::GRAPH_FAILED);
}
```

**vs. baseline:**
```cpp
// baseline无数据类型校验
this->Input("x").DataType({ge::DT_FLOAT});
this->Input("weight").DataType({ge::DT_FLOAT});
```

Benefit: 自动化的数据类型组合管理和严格的类型校验，提高代码健壮性和可维护性
Trade-off: 需要维护类型映射表和校验逻辑，增加代码量

---

## Variant X: 双精度输入支持（FP16 + BF16）
Source: rms_norm_quant

专家实现通过模板参数T支持half和bfloat16_t两种输入数据类型。BF16在保持与FP32相似动态范围的同时，减少了存储和计算开销。实现中使用#if条件编译针对不同AICORE架构进行适配——在910B/910_93上同时支持FP16和BF16，而在310P/310B/KirinX90上仅支持FP16。模板化的设计使得同一套计算逻辑可以无缝适配不同数据类型，避免了代码重复。

**Expert implementation:**
```cpp
template <typename T, typename yDtype, bool EN_BETA, bool FastComputeMode = false>
class RmsNormQuant {
    // ...
};

#if (defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220))
    if (TILING_KEY_IS(283)) {
        RmsNormQuant<bfloat16_t, DTYPE_Y, true, true> kernel(...);
    }
#endif
```

**vs. baseline:**
```cpp
this->Input("x")
    .DataType({ge::DT_FLOAT16})
    .Format({ge::FORMAT_ND});
```

Benefit: 支持更多模型格式，BF16可减少存储带宽同时保持数值稳定性
Trade-off: 增加了模板实例化数量，编译时间略有增加

---

## Variant Y: 中间计算类型提升策略
Source: scaled_masked_softmax_grad_v2

专家实现采用'输入输出保持原类型，中间计算提升为float'的策略。这是数值计算中的经典优化手段，能够保证精度、提高计算稳定性，并充分利用Ascend C的Vector计算单元对float类型的更好支持。实现上通过yGradTmpBuffer和yTmpBuffer两个VECCALC Buffer存储float类型的中间结果，在Compute阶段进行Cast转换。对于float输入类型，直接跳过Cast步骤，避免不必要的开销。

**Expert implementation:**
```cpp
template <typename T>
__aicore__ inline void ScaledMaskedSoftmaxGradV2NormHeadDim<T>::ComputeSoftmaxGrad() {
    if constexpr (IsSameType<T, float>::value) {
        DoSoftmaxGrad(xGradLocal, yGradLocal, yLocal);
    } else {
        LocalTensor<float> tmpBufYGrad = this->yGradTmpBuffer.template Get<float>();
        LocalTensor<float> tmpBufY = this->yTmpBuffer.template Get<float>();
        Cast(tmpBufYGrad, yGradLocal, RoundMode::CAST_NONE, this->calcNum);
        Cast(tmpBufY, yLocal, RoundMode::CAST_NONE, this->calcNum);
        DoSoftmaxGrad(tmpBufYGrad, tmpBufYGrad, tmpBufY);
        Cast(xGradLocal, tmpBufYGrad, RoundMode::CAST_RINT, this->calcNum);
    }
}
```

**vs. baseline:**
```cpp
AscendC::LocalTensor<float> gradOutputLocal = inQueueGradOutput.DeQue<float>();
AscendC::Muls(tmpLocal, softmaxOutputLocal, -1.0f, this->tileSize);
AscendC::Adds(gradSoftmaxLocal, tmpLocal, 1.0f, this->tileSize);
```

Benefit: 保证数值稳定性，避免FP16/BF16的溢出和精度丢失；中间计算使用float类型，硬件支持更好
Trade-off: 增加两次Cast操作的开销；需要额外的VECCALC Buffer存储中间结果

---

## Variant Z: 输入校验与边界保护
Source: sparse_to_dense

专家实现包含完整的输入校验机制，确保在各种边界条件下都能正确执行：1) 数据类型校验：检查indices、values、default_value、output的数据类型是否在支持范围内，并确保相互兼容；2) Shape维度校验：验证indices维度不超过2、output_shape维度必须为1、value维度不超过1等约束；3) 数值范围校验：检查default_value的大小必须为1（标量）。这种严格的校验机制可以在编译期和运行期捕获潜在的错误输入，避免因数据格式不正确导致的未定义行为或精度损失。

**Expert implementation:**
```cpp
ge::graphStatus SparseToDenseTiling::CheckInputDtype() {
    OP_CHECK_IF((INDICES_DTYPE.find(indicesDtype_) == INDICES_DTYPE.end()),
        OP_LOGE(opName_, "indices dtype only support int32 and int64"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((VALUE_DTYPE.find(valuesDtype_) == VALUE_DTYPE.end()),
        OP_LOGE(opName_, "values dtype not supported"),
        return ge::GRAPH_FAILED);
}

ge::graphStatus SparseToDenseTiling::CheckInputShape() {
    OP_CHECK_IF((indicesDimNum_ > DIM_NUM_2),
        OP_LOGE(opName_, "indicesDimNum cannnot be greater than 2"),
        return ge::GRAPH_FAILED);
}
```

Benefit: 提高算子鲁棒性，避免未定义行为
Trade-off: 增加了运行时校验开销（通常可忽略）

---

## Variant 27: 灵活的数据类型支持
Source: trans_quant_param_v2

专家实现明确定义了支持的数据类型集合，允许输入为DT_FLOAT，输出为DT_UINT64或DT_INT64。这种设计考虑了不同使用场景的需求：某些场景可能需要无符号解释（DT_UINT64），而其他场景可能需要有符号解释（DT_INT64）。通过initializer_list定义支持列表，代码清晰地表达了类型约束，同时也便于后续扩展。offset输入被标记为OPTIONAL，这使得算子可以灵活地处理只需要转换scale的场景。

**Expert implementation:**
```cpp
static const std::initializer_list<op::DataType> SCALE_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> OUT_TYPE_SUPPORT_LIST = {
    op::DataType::DT_UINT64, op::DataType::DT_INT64};

this->Input("offset").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
```

**vs. baseline:**
```cpp
"dtype": "float32"  # op_desc.json
output_scale = torch.empty_like(scale)
```

Benefit: 提供灵活的数据类型选择，支持optional输入，提高算子的通用性
Trade-off: 需要在运行时进行额外的类型检查
