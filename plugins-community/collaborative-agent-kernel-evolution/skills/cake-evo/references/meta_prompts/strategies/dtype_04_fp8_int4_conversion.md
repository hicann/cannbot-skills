# D4: FP8/INT4 Quantization Conversion (FP8/INT4量化输出类型)
## Overview
baseline 使用固定的 CAST_ROUND 模式进行类型转换，这可能在某些场景下引入系统性偏差。专家实现支持多种 rounding mode：MODE_RINT (默认，四舍五入到最近偶数/Banker's rounding)、MODE_ROUND (标准四舍五入)、MODE_FLOOR (向下取整)、MODE_TRUNC (向零取整)、MODE_CEIL (向上取整)。不同场景需要不同的 rounding mode：训练量化通常使用 rint 以减少累积误差、推理量化可能需要 floor 或 trunc 以满足特定硬件要求、FP8 输出强制使用 MODE_RINT 以符合标准。

## When to Use
- Quantization output operators
- 支持多种数据类型组合，适配不同硬件和场景需求，避免算子碎片化
- 支持多种精度要求和存储约束场景，INT4减少50%存储，FP16/BF16减少内存带宽压力
- 支持INT4量化，减少50%存储占用

## Trade-off
- 增加编译时间和代码复杂度，需要更多测试覆盖
- 代码复杂度增加，需要维护多个模板类实现
- 需要额外的条件判断和地址计算

**Source operators**: add_rms_norm_dynamic_quant, ascend_quant_v2, dynamic_block_quant, dynamic_mx_quant, dynamic_quant_update_scatter_v2, grouped_dynamic_mx_quant, rms_norm_quant

---

## Variant A: 多数据类型输入输出支持
Source: add_rms_norm_dynamic_quant

专家实现通过模板参数T和编译期类型判断is_same<T, half>::value实现了对FP16和BF16输入的统一支持。在OpDef层面，通过xDataType91095和yDataType91095数组定义了8种不同的数据类型组合，支持输入为FP16/BF16，输出量化为INT8/FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8。这种设计允许同一个算子内核根据编译时选择的不同数据类型实例化，避免了代码重复。

**Expert implementation:**
```cpp
static const std::vector<ge::DataType> xDataType91095 = {
    ge::DT_FLOAT16, ge::DT_BF16, ...
};
static const std::vector<ge::DataType> yDataType91095 = {
    ge::DT_INT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8
};
if constexpr (is_same<T, half>::value) {
    Cast(xOut, xLocalFp32, RoundMode::CAST_NONE, elementCount);
} else { // BF16
    Cast(xOut, xLocalFp32, RoundMode::CAST_RINT, elementCount);
}
```

**vs. baseline:**
```cpp
this->Input("x1").DataType({ge::DT_FLOAT16});
this->Output("output").DataType({ge::DT_INT8});
```

Benefit: 支持多种数据类型组合，适配不同硬件和场景需求，避免算子碎片化
Trade-off: 增加编译时间和代码复杂度，需要更多测试覆盖

---

## Variant B: 完整数据类型矩阵支持
Source: ascend_quant_v2

专家实现通过 OpDef 配置构建完整数据类型支持矩阵，输入支持 DT_FLOAT16、DT_FLOAT、DT_BF16；输出支持 DT_INT8、DT_INT4、DT_HIFLOAT8、DT_FLOAT8_E5M2、DT_FLOAT8_E4M3FN。针对不同芯片平台定义不同数据类型支持子集，Kernel端通过C++模板和if constexpr在编译期进行类型分发，实现零运行时开销的多类型支持。INT4输出可减少50%存储占用，HIFLOAT8和FLOAT8支持新一代AI加速器的高精度低比特计算。

**Expert implementation:**
```cpp
// 完整数据类型矩阵
this->Input("x").DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ...});
this->Output("y").DataType({ge::DT_INT8, ge::DT_INT4, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ...});

// 编译期类型分发
if constexpr (std::is_same<DTYPE_X, half>::value) {
    AscendQuantV2::AscendQuantV2PerChannelFP16<DTYPE_X> op;
}
```

**vs. baseline:**
```cpp
// 仅支持 FLOAT 输入，INT8 输出
this->Input("x").DataType({ge::DT_FLOAT});
this->Output("y").DataType({ge::DT_INT8});
```

Benefit: 支持多种精度要求和存储约束场景，INT4减少50%存储，FP16/BF16减少内存带宽压力
Trade-off: 代码复杂度增加，需要维护多个模板类实现

---

## Variant C: INT4特殊处理
Source: ascend_quant_v2

针对INT4输出格式（两个元素打包到一个字节），在地址计算和数据搬运上进行特殊处理。在GetOutCopyParams中自动调整yLenReal（长度减半）和yOutOffset（地址右移一位），确保INT4数据正确存储和访问。

**Expert implementation:**
```cpp
// INT4地址和长度特殊处理
if (ORIG_DTYPE_Y == DT_INT4) {
    yLenReal = yLenReal / INT4_NUMS_IN_INT8_SPACE;
    yOutOffset = yOutOffset >> 1;
}
```

Benefit: 支持INT4量化，减少50%存储占用
Trade-off: 需要额外的条件判断和地址计算

---

## Variant D: FP8/HIFLOAT8专用硬件支持
Source: dynamic_block_quant

专家实现针对新一代AI加速器支持的FP8/HIFLOAT8类型提供了专门优化。通过MicroAPI的CastTrait结构，可以精细配置类型转换行为：RegLayout指定寄存器布局，SatMode控制溢出行为（SAT表示饱和），MaskMergeMode处理mask合并，RoundMode选择舍入模式。特别是HIFLOAT8支持round和hybrid两种舍入模式，通过编译期常量RMode选择不同的CastTrait。这种设计充分利用了硬件提供的专门指令，避免了软件模拟FP8转换的开销，同时保证了量化精度的可配置性。

**Expert implementation:**
```cpp
static constexpr AscendC::MicroAPI::CastTrait castTrait32toh8 = []() {
    if constexpr (RMode == 1 || RMode == 4) {
        return AscendC::MicroAPI::CastTrait {
            AscendC::MicroAPI::RegLayout::ZERO,
            AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING,
            RoundMode::CAST_ROUND
        };
    }
}();
AscendC::MicroAPI::Cast<U, float, castTrait32toh8>(vreg6, vreg7, preg0);
```

**vs. baseline:**
```cpp
// 仅支持int8，无FP8支持
for (uint32_t i = 0; i < blockSize; i++) {
    float val = quantizedFloat.GetValue(i);
    int rounded = static_cast<int>(val + 0.5f);
    quantizedLocal.SetValue(i, static_cast<int8_t>(clamped));
}
```

Benefit: 利用硬件FP8指令，高性能高精度，支持多种舍入模式
Trade-off: 代码复杂度增加，需要理解硬件特性和MicroAPI

---

## Variant E: 差异化Round Mode支持
Source: dynamic_block_quant

专家实现支持rint、round、hybrid三种舍入模式，通过RoundModeList枚举定义。不同输出数据类型要求不同的舍入模式：HIFLOAT8支持round和hybrid，FP8_E4M3FN/FP8_E5M2仅支持rint。这种设计在Host端进行严格校验，在Tiling Key的千分位编码round mode信息，在Kernel端通过模板参数RMode编译期选择代码路径。相比baseline的固定round-half-up实现，专家实现提供了符合IEEE标准的多种舍入选项，满足不同应用对舍入行为的严格要求（如金融计算、科学计算）。

**Expert implementation:**
```cpp
enum class RoundModeList : int64_t {
    MODE_RINT = 1,
    MODE_ROUND = 4,
    MODE_HYBRID = 7,
};
Cast(xLocalTmpHalf.template ReinterpretCast<int16_t>(), xLocalTmp, RoundMode::CAST_RINT, calcNum);
```

**vs. baseline:**
```cpp
// 固定round-half-up实现
if (val >= 0.0f) {
    rounded = static_cast<int>(val + 0.5f);
} else {
    rounded = static_cast<int>(val - 0.5f);
}
```

Benefit: 支持标准舍入模式，满足不同应用需求，符合IEEE规范
Trade-off: 需要理解不同round mode的语义，增加测试覆盖范围

---

## Variant F: 输出数据类型多样化 (FP4/FP8 多格式支持)
Source: dynamic_mx_quant

baseline 只输出 int8 类型，这严重限制了应用场景。现代 AI 硬件支持更激进的量化格式如 FP4 (E2M1/E1M2) 和 FP8 (E4M3FN/E5M2)。专家实现完整支持这四种输出格式，每种格式都有不同的指数位/尾数位配置。关键技术包括：FP4 双格式支持（E2M1 和 E1M2 分别适用于不同精度需求）、FP8 标准格式（E4M3FN 无 Inf 和 E5M2 标准 IEEE）、Scale 格式统一使用 FLOAT8_E8M0、模板特化计算中 CalcElement 模板根据输出类型 U 选择不同的量化算法。

**Expert implementation:**
```cpp
this->Output("y")
    .DataType({
        ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1,  // FP4 E2M1
        ge::DT_FLOAT4_E1M2, ge::DT_FLOAT4_E1M2,  // FP4 E1M2
        ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,  // FP8 E4M3FN
        ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2   // FP8 E5M2
    });

// FP4 E2M1 的特化量化算法
if constexpr (IsSame<outType, fp4x2_e2m1_t>::value) {
    AscendC::MicroAPI::RegTensor<int32_t> exp1, exp2;
    AscendC::MicroAPI::And(exp1, (AscendC::MicroAPI::RegTensor<int32_t>&)in, maxEle, mask);
    AscendC::MicroAPI::ShiftRights(exp1, exp1, SHR_NUM_FOR_FP32, mask);
    AscendC::MicroAPI::Adds(exp1, exp1, FP32_BIAS_NEG, mask);
    // ... 复杂的指数调整
}
```

**vs. baseline:**
```cpp
this->Output("quantized_tensor")
    .ParamType(REQUIRED)
    .DataType({ge::DT_INT8})
    .Format({ge::FORMAT_ND});
```

Benefit: 支持前沿量化格式，FP4 相比 INT8 可减少 75% 存储，FP8 在保持精度的同时减少 50% 存储
Trade-off: 量化算法复杂度显著增加，特别是 FP4 需要处理指数裁剪和尾数打包；需要更多常量定义和特化代码

---

## Variant G: 多种 Rounding Mode 支持
Source: dynamic_mx_quant

baseline 使用固定的 CAST_ROUND 模式进行类型转换，这可能在某些场景下引入系统性偏差。专家实现支持多种 rounding mode：MODE_RINT (默认，四舍五入到最近偶数/Banker's rounding)、MODE_ROUND (标准四舍五入)、MODE_FLOOR (向下取整)、MODE_TRUNC (向零取整)、MODE_CEIL (向上取整)。不同场景需要不同的 rounding mode：训练量化通常使用 rint 以减少累积误差、推理量化可能需要 floor 或 trunc 以满足特定硬件要求、FP8 输出强制使用 MODE_RINT 以符合标准。

**Expert implementation:**
```cpp
// 专家实现：可配置 rounding mode
enum class RoundModeList {
    MODE_ROUND = 0, MODE_FLOOR = 1, MODE_CEIL = 2,
    MODE_TRUNC = 3, MODE_RINT = 4, MODE_HYBRID = 5
};

// Compute 模板中的 rounding mode 参数
template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
__aicore__ inline void Compute(...) {
    CalcElement<roundMode, U, calcType, calcTypeInt>(...);
}

// 运行时根据 attr 选择
if (this->roundMode_ == MODE_RINT) {
    Compute<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(...);
} else if (this->roundMode_ == MODE_ROUND) {
    Compute<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(...);
}
```

**vs. baseline:**
```cpp
// baseline：固定 rounding mode
AscendC::Cast(quantInt32Local, quantLocal, AscendC::RoundMode::CAST_ROUND, blockSize);
```

Benefit: 满足不同精度标准要求，训练推理场景都能获得最优精度
Trade-off: 代码体积增加（需要实例化多个模板）；用户需要理解不同 rounding mode 的含义

---

## Variant H: INT4输出量化支持
Source: dynamic_quant_update_scatter_v2

专家实现支持将FP16/BF16输入动态量化为INT4输出，这是大模型量化场景的关键需求。INT4数据类型的支持需要特殊处理，因为每个INT4值只占用4个bit，两个INT4值打包在一个字节中。实现中通过int4b_t类型和特殊的DataCopy参数设置(copyParams.blockLen = copyParams.blockLen >> 1)来处理这种打包存储格式。此外，实现还验证了当输出为INT4时，输入的最后一维必须是偶数，确保数据对齐。

**Expert implementation:**
```cpp
// INT4输出的DataCopy配置
DataCopyExtParams copyParams{1, (uint16_t)(tilingData_.rowLen * sizeof(int4b_t)), 0, 0, 0};
copyParams.blockLen = copyParams.blockLen >> 1;  // INT4打包: 长度减半
DataCopyPad(outputGm[dstOffset], outLocal, copyParams);

// Shape校验: INT4输出要求最后一维为偶数
if (varDtype == ge::DT_INT4) {
    OP_CHECK_IF(
        (xDimLast % EVEN_FACTOR),
        OP_LOGE(context, "if var datatype is int4, the last dim(%ld) of x should be an even number", xDimLast),
        return ge::GRAPH_FAILED);
}
```

**vs. baseline:**
```cpp
// baseline可能使用INT8量化
DataCopy(outGm[dstOffset], outLocal, rowLen);
```

Benefit: 显著降低模型存储和带宽需求(相比INT8再减半)，支持大模型量化部署
Trade-off: 需要处理INT4打包/解包，增加了代码复杂度和精度损失风险

---

## Variant I: 特殊值处理与溢出控制
Source: grouped_dynamic_mx_quant

专家实现对特殊值（NaN、Inf、零）进行了精细处理，确保量化结果的数值稳定性。通过FLOAT_OVERFLOW_MODE_CTRL寄存器控制溢出行为，处理FP8表示范围外的值。定义NAN_CUSTOMIZATION = 0x7f81作为特殊的NaN表示。通过Compare和Select指令实现条件化的指数裁剪，使用infMask、zeroMask、invalidDataMask等多个mask寄存器实现对不同特殊情况的处理。

**Expert implementation:**
```cpp
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;

#if (__NPU_ARCH__ == 3101)
int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL,FLOAT_OVERFLOW_MODE_CTRL>();
AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL,FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif

AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxRegTensor, maxEleRegTensor, p0);
AscendC::MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMaxRegTensor, fp8MaxExpRegTensor, p0);
AscendC::MicroAPI::Select<uint16_t>(mxScaleRegTensor, mxScaleRegTensor, fp8NanRegTensor, infMask);
AscendC::MicroAPI::Select<uint16_t>(mxScaleRegTensor, mxScaleRegTensor, zeroRegTensor, zeroMask);
```

**vs. baseline:**
```cpp
if (scaledVal > 448.0f) scaledVal = 448.0f;
if (scaledVal < -448.0f) scaledVal = -448.0f;
```

Benefit: 完整的特殊值处理保证数值稳定性；硬件级溢出控制；符合IEEE754标准
Trade-off: 增加指令数量；需要处理多种边界情况

---

## Variant J: 共享指数计算与逆量化因子
Source: grouped_dynamic_mx_quant

专家实现采用了标准的MX（Microscaling）量化算法，核心在于共享指数的计算和逆量化因子的推导。每个block（32个元素）共享一个最大指数expMaxRegTensor。通过Sub(expMaxRegTensor, expMaxRegTensor, fp8MaxExpRegTensor)计算与FP8最大指数的偏差。Sub(reversedShareExpRegTensor, biasRegTensor, expMaxRegTensor)计算1/scale的指数表示。这种设计确保了量化过程中的数值稳定性，同时通过共享指数减少了存储开销（每个block只需1字节的scale）。

**Expert implementation:**
```cpp
// 计算与FP8最大指数的偏差
AscendC::MicroAPI::Sub(expMaxRegTensor, expMaxRegTensor, fp8MaxExpRegTensor, p0);

// 计算scale（右移7位将BF16指数转换为FP8_E8M0格式）
AscendC::MicroAPI::ShiftRights(mxScaleRegTensor, expMaxRegTensor, SHR_NUM_FOR_BF16, p0);

// 计算1/scale的指数
AscendC::MicroAPI::Sub(reversedShareExpRegTensor, biasRegTensor, expMaxRegTensor, p0);

// 特殊情况处理
AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, expMaxRegTensor, biasRegTensor, p0);
AscendC::MicroAPI::Select<uint16_t>(reversedShareExpRegTensor, specialExpRegTensor,
    reversedShareExpRegTensor, specialDataMask);
```

**vs. baseline:**
```cpp
uint16_t maxExp = 0;
for (...) {
    if (exp > maxExp) maxExp = exp;
}
uint16_t scale = (maxExp - FP8_E4M3_MAX_EXP) >> SHR_NUM_FOR_BF16;
```

Benefit: 标准MX量化算法保证精度；共享指数减少存储；精确的反量化计算
Trade-off: 需要额外的指数计算指令

---

## Variant K: 向量化数据转换与量化
Source: grouped_dynamic_mx_quant

专家实现通过向量化指令实现了高效的FP32到FP8转换。FP16/BF16到FP32扩展通过Cast指令完成，保留精度。使用FP32精度进行Mul操作，避免中间结果精度损失。castTrait32to8定义了sat饱和模式和CAST_RINT舍入模式。Interleave指令的使用实现了数据的重排，配合Pack指令将寄存器数据打包到LocalTensor，形成了完整的向量化处理流程。

**Expert implementation:**
```cpp
static constexpr AscendC::MicroAPI::CastTrait castTrait32to8 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

AscendC::MicroAPI::Cast<float, T, castTraitZero>(yZero, xRegTensor, p0);
AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitZero>(reversedShareExpRegTensorFP32Zero, 
    (AscendC::MicroAPI::RegTensor<bfloat16_t>&)reversedShareExpRegTensor, p0);
AscendC::MicroAPI::Mul(yZero, yZero, reversedShareExpRegTensorFP32Zero, maskAll);

AscendC::MicroAPI::Interleave(yZero, yOne, yZero, yOne);
AscendC::MicroAPI::Cast<U, float, castTrait32to8>(yZeroFP8, yZero, maskAll);
AscendC::MicroAPI::Pack(yRegTensorZero, (AscendC::MicroAPI::RegTensor<uint32_t>&)yZeroFP8);
AscendC::MicroAPI::Pack(outZero, yRegTensorZero);
```

**vs. baseline:**
```cpp
T val = xLocal[j * vfLen + k];
float fp32Val = static_cast<float>(val);
float scaleFactor = *reinterpret_cast<float*>(&reverseExp);
float scaledVal = fp32Val * scaleFactor;
yAddr[...] = static_cast<uint8_t>(scaledVal);
```

Benefit: 完整的精度保持转换链；硬件级舍入和饱和控制；向量化提升性能
Trade-off: 需要理解CastTrait配置；数据重排增加指令数

---

## Variant L: 混合精度输出支持（INT8 + INT4）
Source: rms_norm_quant

专家实现通过yDtype模板参数支持INT8和INT4两种量化输出格式。INT4相比INT8可以进一步节省50%的显存带宽，在超大模型推理中具有重要意义。实现中针对INT4的特殊存储格式（每字节存储2个INT4值）进行了专门处理，通过SIZE_INT4常量和int4b_t类型正确处理内存访问。

**Expert implementation:**
```cpp
if constexpr(IsSameType<yDtype, int4b_t>::value) {
    AscendC::LocalTensor<int4b_t> int4_y = int4_y_que_.AllocTensor<int4b_t>();
    Cast(int4_y, tmpfp16, RoundMode::CAST_RINT, num_col_);
    int4_y_que_.EnQue(int4_y);
}

this->Output("y")
    .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT4, ge::DT_INT4})
```

**vs. baseline:**
```cpp
this->Output("y")
    .DataType({ge::DT_INT8})
    .Format({ge::FORMAT_ND});
```

Benefit: INT4可节省50%显存带宽，适合超大模型部署
Trade-off: 精度略有损失，需要校准确定最佳缩放因子
