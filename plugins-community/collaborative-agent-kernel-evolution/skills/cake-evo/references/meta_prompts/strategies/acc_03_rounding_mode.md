# A3: Rounding Mode Control (舍入模式控制)
## Overview
专家实现在FP16版本中采用了多层次类型转换策略来平衡精度与性能。计算流程为：1) 将FP16输入Cast到FP32进行除法运算（避免FP16除法精度损失）；2) 在FP32域进行Muls和Adds操作；3) 通过Cast到int32并指定RoundMode::CAST_RINT实现四舍五入；4) 将结果Cast回FP32进行后续计算；5) 最终输出时Cast到FP16。这种精细的类型转换链确保了关键计算步骤的精度，同时减少了不必要的精度转换开销。相比之下，baseline实现使用了Round指令（AscendC::Round(xqRoundedLocal, xScaledLocal, this->tileSize)），虽然简单但可能无法提供与Cast+RoundMode::CAST_RINT相同的精度和灵活性。

## When to Use
- Any Cast to lower precision
- 中间计算使用截断减少指令开销，最终输出使用四舍五入提高精度
- 转换精度更高，避免直接转换导致的精度损失
- 满足不同量化算法对舍入行为的要求，提高数值精度

## Trade-off
- 需要针对不同数据类型选择不同策略，增加代码复杂度
- 增加4条指令和同步开销
- 需要额外的条件分支，可能略微影响性能

**Source operators**: add_rms_norm_cast, add_rms_norm_dynamic_quant, ascend_quant_v2, dequant_bias, dynamic_quant_update_scatter_v2, fake_quant_affine_cachemask, foreach_abs, foreach_add_list, foreach_add_scalar, gemma_rms_norm, linear_index, max_pool_grad_with_argmax_common, rms_norm_grad, rms_norm_quant, scaled_masked_softmax_grad_v2, trans_quant_param_v2

---

## Variant A: 舍入模式精细化控制
Source: add_rms_norm_cast

专家实现在不同阶段使用不同的舍入模式（RoundMode::CAST_NONE vs RoundMode::CAST_RINT），根据场景需求在精度和性能之间取得平衡。舍入策略为：CAST_NONE直接截断，适用于中间计算；CAST_RINT四舍五入，适用于最终输出，减少精度损失；BF16转换时特别使用CAST_RINT以保持精度。

**Expert implementation:**
```cpp
// 精细化舍入控制
// BF16: 使用RINT保证精度
Cast(x1_fp32, x1Local, RoundMode::CAST_NONE, numCol);
Cast(x2_fp32, x2Local, RoundMode::CAST_NONE, numCol);
PipeBarrier<PIPE_V>();
Add(x1_fp32, x1_fp32, x2_fp32, numCol);
PipeBarrier<PIPE_V>();
Cast(xLocal, x1_fp32, RoundMode::CAST_RINT, numCol);  // 输出用RINT

// FP16: 输入输出都用NONE
Cast(yLocal, xFp32, RoundMode::CAST_NONE, numCol);
```

**vs. baseline:**
```cpp
// 统一使用CAST_NONE
AscendC::Cast(addedLocal, xLocal, AscendC::RoundMode::CAST_NONE, this->tileLength);
AscendC::Cast(outputLocal, addedLocal, AscendC::RoundMode::CAST_NONE, this->tileLength);
```

Benefit: 中间计算使用截断减少指令开销，最终输出使用四舍五入提高精度
Trade-off: 需要针对不同数据类型选择不同策略，增加代码复杂度

---

## Variant B: 整数转换的多阶段精度保持
Source: add_rms_norm_dynamic_quant

专家实现在Float32到Int8的转换过程中采用了多阶段转换策略：Cast到Int32并四舍五入(CAST_RINT)，SetDeqScale设置量化参数，Cast到Half(CAST_NONE)，最后Cast到Int8(CAST_TRUNC)。这种多阶段转换虽然增加了指令数，但确保了在转换过程中的数值精度。

**Expert implementation:**
```cpp
__aicore__ inline void RoundFloat2Int8(LocalTensor<int8_t>& dstTensor, LocalTensor<float>& srcTensor, int32_t size) {
    Cast(srcTensor.ReinterpretCast<int32_t>(), srcTensor, RoundMode::CAST_RINT, size);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    Cast(srcTensor.ReinterpretCast<half>(), srcTensor.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, size);
    PipeBarrier<PIPE_V>();
    Cast(dstTensor, srcTensor.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, size);
    PipeBarrier<PIPE_V>();
}
```

**vs. baseline:**
```cpp
AscendC::Cast(int32Local, tempLocal, AscendC::RoundMode::CAST_RINT, tileLength);
for (uint32_t i = 0; i < tileLength; ++i) {
    int32_t val = int32Local.GetValue(i);
    tempInt8.SetValue(i, static_cast<int8_t>(val));
}
```

Benefit: 转换精度更高，避免直接转换导致的精度损失
Trade-off: 增加4条指令和同步开销

---

## Variant C: 多模式舍入支持
Source: ascend_quant_v2

支持四种舍入模式：ROUND（四舍五入）、FLOOR（向下取整）、CEIL（向上取整）、TRUNC（向零截断）。通过GetRoundMode函数在Host端解析属性传递给Kernel。在CastOut函数中根据roundMode选择不同的RoundMode进行类型转换，float->int32和half->int8两个阶段可能使用不同舍入模式，提供精细数值控制。

**Expert implementation:**
```cpp
// Host端舍入模式解析
RoundMode GetRoundMode(std::string& roundMode) {
    if (roundMode == "round") return RoundMode::MODE_ROUND;
    else if (roundMode == "floor") return RoundMode::MODE_FLOOR;
    else if (roundMode == "ceil") return RoundMode::MODE_CEIL;
    else if (roundMode == "trunc") return RoundMode::MODE_TRUNC;
}

// Kernel端多模式舍入
if (runTilingData.roundMode == MODE_ROUND) {
    Cast(xLocal.ReinterpretCast<int32_t>(), xLocal, RoundMode::CAST_RINT, dataCount);
} else if (runTilingData.roundMode == MODE_FLOOR) {
    Cast(xLocal.ReinterpretCast<int32_t>(), xLocal, RoundMode::CAST_FLOOR, dataCount);
}
```

**vs. baseline:**
```cpp
// 固定RINT舍入
AscendC::Cast(tempInt32Local, tempDivLocal, AscendC::RoundMode::CAST_RINT, this->tileSize);
```

Benefit: 满足不同量化算法对舍入行为的要求，提高数值精度
Trade-off: 需要额外的条件分支，可能略微影响性能

---

## Variant D: Sqrt模式数值处理
Source: ascend_quant_v2

支持sqrt_mode属性，启用时在量化前对scale进行平方操作（x * scale * scale）。这对于考虑方差的量化算法非常有用。通过模板参数SQRT_MODE在编译期确定是否需要第二次乘法，避免运行时判断开销。

**Expert implementation:**
```cpp
// sqrt_mode支持
Mul(castXLocal, castXLocal, sLocal, dataCount);
PipeBarrier<PIPE_V>();
if constexpr (SQRT_MODE) {
    Mul(castXLocal, castXLocal, sLocal, dataCount);
    PipeBarrier<PIPE_V>();
}
```

Benefit: 支持更复杂的量化算法，如考虑方差的量化
Trade-off: 需要额外的模板实例化

---

## Variant E: 舍入模式的选择
Source: dequant_bias

专家实现在不同阶段的类型转换中使用了不同的舍入模式：CAST_NONE用于中间计算过程的转换，不进行舍入；CAST_RINT用于最终输出转换，使用最近整数舍入。这种差异化的舍入策略确保最终输出的精度最优。

**Expert implementation:**
```cpp
// 专家实现：差异化的舍入模式
Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, inRows * nAlign_);  // 中间转换：无舍入
// ... 计算 ...
Cast(yLocal, xLocalFp32, RoundMode::CAST_RINT, inRows * nAlign_);  // 最终输出：最近整数舍入
```

**vs. baseline:**
```cpp
// baseline：统一使用CAST_NONE
AscendC::Cast(inputHalfLocal, inputLocal, AscendC::RoundMode::CAST_NONE, tileSize);
AscendC::Cast(inputF32Local, inputHalfLocal, AscendC::RoundMode::CAST_NONE, tileSize);
```

Benefit: 最终输出精度最优，中间计算无精度损失
Trade-off: 需要关注不同舍入模式的选择

---

## Variant F: 四舍五入模式选择(Rounding Mode Selection)
Source: dynamic_quant_update_scatter_v2

专家实现在不同量化阶段使用不同的四舍五入模式以优化精度：(1) FP16→FP32使用CAST_NONE保持数值精确性；(2) FP32→INT32使用CAST_RINT实现最近偶数舍入；(3) INT32→half使用CAST_ROUND实现标准四舍五入；(4) half→目标格式使用CAST_TRUNC截断。这种差异化的舍入策略基于各阶段的数据特性：CAST_RINT在整数转换时具有更好的统计特性，CAST_TRUNC在最终输出时避免溢出。

**Expert implementation:**
```cpp
// 差异化舍入模式
Cast(tempFp32, inLocal[i * sizeHalfLen], RoundMode::CAST_NONE, tilingData_.rowLen);     // 保持精度
// ... 计算 ...
Cast(tempInt32, tempFp32, RoundMode::CAST_RINT, tilingData_.rowLen);                      // 最近偶数舍入
SetDeqScale(static_cast<half>(1.0));
Cast(tempHalf, tempInt32, RoundMode::CAST_ROUND, tilingData_.rowLen);                     // 标准四舍五入
Cast(outLocal, tempHalf, RoundMode::CAST_TRUNC, tilingData_.rowLen);                      // 截断
```

**vs. baseline:**
```cpp
// baseline可能使用统一舍入模式
Cast(out, in, RoundMode::CAST_ROUND, len);
```

Benefit: 各阶段使用最适合的舍入方式，整体量化误差最小化
Trade-off: 需要对舍入模式有深入理解，增加代码复杂度

---

## Variant G: 多层次类型转换与精度保持
Source: fake_quant_affine_cachemask

专家实现在FP16版本中采用了多层次类型转换策略来平衡精度与性能。计算流程为：1) 将FP16输入Cast到FP32进行除法运算（避免FP16除法精度损失）；2) 在FP32域进行Muls和Adds操作；3) 通过Cast到int32并指定RoundMode::CAST_RINT实现四舍五入；4) 将结果Cast回FP32进行后续计算；5) 最终输出时Cast到FP16。这种精细的类型转换链确保了关键计算步骤的精度，同时减少了不必要的精度转换开销。相比之下，baseline实现使用了Round指令（AscendC::Round(xqRoundedLocal, xScaledLocal, this->tileSize)），虽然简单但可能无法提供与Cast+RoundMode::CAST_RINT相同的精度和灵活性。

**Expert implementation:**
```cpp
// 专家实现 - 多层次类型转换
Cast(curTemp, xLocal, RoundMode::CAST_NONE, calCount);  // FP16 -> FP32
Muls(curTemp, curTemp, static_cast<float>(1.0f / scaleValue), calCount);
Cast(curInt32Temp, curTemp, RoundMode::CAST_RINT, calCount);  // 四舍五入
Cast(curTemp, curInt32Temp, RoundMode::CAST_NONE, calCount);  // int32 -> FP32
Adds(curTemp, curTemp, static_cast<float>(zeroPointValue), calCount);
Cast(curHf16Temp, curTemp, RoundMode::CAST_RINT, calCount);  // FP32 -> FP16
```

**vs. baseline:**
```cpp
// baseline - 简单Round指令
AscendC::Muls(xScaledLocal, inputLocal, invScale, this->tileSize);
AscendC::Round(xqRoundedLocal, xScaledLocal, this->tileSize);
AscendC::Adds(xqRoundedLocal, xqRoundedLocal, this->zeroPoint, this->tileSize);
```

Benefit: 关键计算步骤使用高精度（FP32），结果输出时转回低精度（FP16），平衡精度与性能；通过RoundMode精确控制舍入行为
Trade-off: 类型转换指令增加，可能有轻微性能损失；代码复杂度增加

---

## Variant H: RINT舍入模式
Source: foreach_abs

在BF16计算结果转回BF16时，专家实现使用了RoundMode::CAST_RINT（四舍五入）模式。RINT模式可以减少精度损失，特别是在数值接近0.5的情况下。这是因为BF16的尾数只有7位，相比FP32损失了大量精度，适当的舍入策略可以在一定程度上弥补这种损失。

**Expert implementation:**
```cpp
Cast(yLocal[index * maxCastDataCount], float32Tensor[offset], RoundMode::CAST_RINT, dataCount);
```

**vs. baseline:**
```cpp
// 不涉及
```

Benefit: 减少BF16精度损失，提高数值稳定性
Trade-off: RINT模式比CAST_NONE略慢

---

## Variant I: BF16高精度中间计算
Source: foreach_add_list

专家实现针对BF16类型采用FP32中间精度策略，避免BF16直接运算导致的精度损失。通过InnerComputer的模板特化实现：BF16数据在计算前转换为FP32，计算完成后再转回BF16。RoundMode使用CAST_RINT（四舍五入）进行输出转换，确保精度损失最小。

**Expert implementation:**
```cpp
// BF16 -> FP32 (CAST_NONE)
Cast(float32Tensor, inLocal_1[...], RoundMode::CAST_NONE, dataCount);
PipeBarrier<PIPE_V>();
// FP32计算
op(float32Tensor, float32Tensor, float32Tensor[maxCastDataCount], scalarVal, dataCount);
PipeBarrier<PIPE_V>();
// FP32 -> BF16 (CAST_RINT)
Cast(outLocal[...], float32Tensor, RoundMode::CAST_RINT, dataCount);
```

**vs. baseline:**
```cpp
// baseline: 不支持BF16
```

Benefit: 保证BF16运算的数值稳定性，精度损失最小化
Trade-off: 需要额外UB空间和Cast操作开销

---

## Variant J: BF16的FP32中间计算
Source: foreach_add_scalar

专家实现针对BF16（BFloat16）数据类型的精度限制，采用了FP32中间计算策略。在InnerComputer模板类中，针对bfloat16_t类型进行了特化，将数据从BF16转换为FP32后进行计算，最后再转换回BF16。这种策略避免了BF16在累加运算中的精度损失，特别是在涉及多次运算或较大数值范围的场景下。转换时使用RoundMode::CAST_NONE进行无损扩展，计算完成后使用RoundMode::CAST_RINT进行四舍五入，保证数值精度。maxCastDataCount参数控制每次转换的数据量，确保转换操作在UB容量限制内进行。

**Expert implementation:**
```cpp
template <OneScalarBinaryOp<float>* op, uint8_t paramsCount>
class InnerComputer<bfloat16_t, float, op, paramsCount> {
public:
    __aicore__ inline void Compute(
        const LocalTensor<bfloat16_t>& dataLocal, const LocalTensor<bfloat16_t>& outLocal,
        LocalTensor<float>& float32Tensor, float scalarVal, uint32_t maxCastDataCount, int64_t dataCount) {
        uint32_t castTimes = dataCount / maxCastDataCount;
        for (uint32_t i = 0; i < castTimes; i++) {
            PipeBarrier<PIPE_V>();
            Cast(float32Tensor, dataLocal[i * maxCastDataCount], RoundMode::CAST_NONE, maxCastDataCount);
            PipeBarrier<PIPE_V>();
            op(float32Tensor[offset], float32Tensor, scalarVal, maxCastDataCount);
            PipeBarrier<PIPE_V>();
            Cast(outLocal[i * maxCastDataCount], float32Tensor[offset], RoundMode::CAST_RINT, maxCastDataCount);
        }
    }
};
```

Benefit: 保证BF16类型计算精度，避免数值溢出和精度损失
Trade-off: 需要额外的Cast操作，增加计算开销和UB内存占用

---

## Variant K: 精度敏感的 Cast 模式选择
Source: gemma_rms_norm

专家实现在不同数据类型的 Cast 操作中使用了不同的 RoundMode。对于 FP16 使用 CAST_NONE（截断），对于 BF16 使用 CAST_RINT（四舍五入）。这是因为 BF16 的指数范围更大但尾数精度更低，需要更精确的舍入策略来避免累积误差。

**Expert implementation:**
```cpp
// 专家实现 - 差异化 Cast 模式
if constexpr (is_same<T, half>::value) {
    Cast(yLocal, sqx, RoundMode::CAST_NONE, elementNum);  // FP16: 截断
} else {
    Cast(yLocal, sqx, RoundMode::CAST_RINT, elementNum);  // BF16: 四舍五入
}
```

**vs. baseline:**
```cpp
// baseline - 无差异化 Cast 处理
// baseline 没有显式 Cast 操作
```

Benefit: 针对不同数据类型的最优舍入策略，减少精度损失
Trade-off: 增加了代码分支，但影响极小

---

## Variant L: 数据类型转换精度控制
Source: linear_index

专家实现在数据类型转换时使用了RoundMode::CAST_NONE（对于int64转int32）和RoundMode::CAST_FLOOR（对于float转int）。选择合适的round mode对于保证精度至关重要。CAST_NONE直接截断，适合已知范围的整数转换；CAST_FLOOR向下取整，适合将除法结果转换为索引。这种精细的round mode选择确保了计算结果的正确性。

**Expert implementation:**
```cpp
Cast<int, T>(indices32Local, indicesLocal, RoundMode::CAST_NONE, indicesAlign);
Cast(arangeIntLocal, arangeLocal, RoundMode::CAST_FLOOR, indicesAlign);
Cast(indicesTemp, arangeLocal, RoundMode::CAST_FLOOR, indicesAlign);
```

**vs. baseline:**
```cpp
indicesLocal.SetValue(i, static_cast<int32_t>(val));  // 默认截断
```

Benefit: 正确的round mode保证计算结果符合数学定义
Trade-off: 需要了解不同round mode的语义，增加了开发复杂度

---

## Variant M: 类型转换Trait配置
Source: max_pool_grad_with_argmax_common

专家实现为类型转换操作定义了详细的CastTrait配置，控制舍入模式、饱和模式等精度相关参数。例如，castTraitT1ComputeType用于将输入类型转换为计算类型，castTraitI64I32用于int64到int32的转换。这些trait配置确保了在不同数据类型之间转换时，数值精度和范围都能得到正确处理。

**Expert implementation:**
```cpp
// 专家实现：详细的CastTrait配置
constexpr AscendC::MicroAPI::CastTrait castTraitT1ComputeType = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::MicroAPI::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait castTraitI64I32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::MicroAPI::RoundMode::CAST_ROUND,
};
```

**vs. baseline:**
```cpp
// baseline实现：无显式类型转换配置
// 使用默认转换行为
```

Benefit: 精细控制类型转换行为，确保数值精度和范围正确性
Trade-off: 增加代码复杂度

---

## Variant N: 高精度中间计算
Source: rms_norm_grad

专家实现确保关键中间计算使用FP32精度：rstd（逆RMS）输入和计算始终使用FP32；dgamma输出固定为FP32；使用ReduceSumHalfInterval进行高精度reduce操作；在Cast操作时支持不同的舍入模式（CAST_NONE, CAST_RINT）。这种设计在保证精度的同时，通过低精度输入/输出来节省内存带宽。baseline实现虽然也使用FP32计算，但没有针对不同数据类型的特殊处理。

**Expert implementation:**
```cpp
float sumValue = ReduceSumHalfInterval(dxLocal, colVal_);
float meanValue = sumValue * avgFactor_;
if constexpr (is_same<T_DY, half>::value) {
    Cast(dxLocalB16, dxLocal, RoundMode::CAST_NONE, colValAlign_);
} else if constexpr (is_same<T_DY, bfloat16_t>::value) {
    Cast(dxLocalB16, dxLocal, RoundMode::CAST_RINT, colValAlign_);
}
```

**vs. baseline:**
```cpp
float sum_x2 = sumTempLocal.GetValue(0);
float mean_x2 = sum_x2 / N;
float rms_val = sqrt(mean_x2 + eps);
```

Benefit: 保证数值稳定性；减少精度损失；支持混合精度
Trade-off: 需要额外的类型转换操作

---

## Variant O: 量化范围灵活配置
Source: rms_norm_quant

专家实现支持通过环境变量ASDOPS_QUANT_MIN_NEG_127灵活配置INT8量化的最小值。默认模式范围是[-128, 127]（标准INT8），兼容模式范围是[-127, 127]（某些框架要求避免-128）。这种设计通过Host端的环境变量读取，在Tiling阶段确定，传递到Kernel端使用。

**Expert implementation:**
```cpp
const char* quantMinFlagPtr = std::getenv("ASDOPS_QUANT_MIN_NEG_127");
if (quantMinFlagPtr != nullptr && strcmp(quantMinFlagPtr, "1") == 0) {
    tilingDataPtr->set_quantMin(-127);
} else {
    tilingDataPtr->set_quantMin(std::numeric_limits<int8_t>::min());
}

__aicore__ inline void CastFromF16ToI8(...) {
    Maxs(in, in, quantMin, count);
    Mins(in, in, (half)127, count);
    Cast(out, in, AscendC::RoundMode::CAST_RINT, count);
}
```

**vs. baseline:**
```cpp
AscendC::Maxs(quantLocal, quantLocal, -128.0f, count);
AscendC::Mins(quantLocal, quantLocal, 127.0f, count);
```

Benefit: 支持不同的量化约定，与各种训练框架兼容
Trade-off: 需要在Tiling阶段读取环境变量

---

## Variant P: Round Mode差异化选择
Source: scaled_masked_softmax_grad_v2

专家实现根据芯片版本选择不同的Round Mode。CAST_NONE直接截断性能最好但精度较低，CAST_RINT四舍五入到最近偶数符合IEEE 754标准。对于CCE_AICORE >= 220（新架构芯片），使用CAST_RINT获得更好的精度；对于旧架构，使用CAST_NONE保证兼容性。

**Expert implementation:**
```cpp
#if __CCE_AICORE__ < 220
    Cast(xGradLocal, tmpBufYGrad, RoundMode::CAST_NONE, this->calcNum);
#else
    Cast(xGradLocal, tmpBufYGrad, RoundMode::CAST_RINT, this->calcNum);
#endif
```

**vs. baseline:**
```cpp
// baseline无类型转换
```

Benefit: 在保证兼容性的前提下，新芯片获得更好精度
Trade-off: 需要条件编译，代码可读性降低

---

## Variant Q: 双Round Mode支持
Source: trans_quant_param_v2

专家实现支持两种舍入模式，通过round_mode属性控制。Round Mode 0使用简单的截断（提取高19位），适用于对精度要求不高的场景；Round Mode 1使用'round to nearest, tie to even'（银行家舍入），符合IEEE 754标准，适用于高精度场景。这种灵活性允许用户根据具体应用的需求在精度和性能之间做出权衡。

**Expert implementation:**
```cpp
this->Attr("round_mode").AttrType(OPTIONAL).Int(0);

__aicore__ inline uint64_t CalQuantPreScaleValue(uint32_t uint32Scale) {
    if (roundMode_ == 0) {
        quantPre = (uint32Scale & DEQ_SCALE_MUL) | QUANT_SCALE;
    } else {
        // Round to nearest, tie to even
        if ((roundBit != 0) && ((stickyBits != 0) || ((fp19Mantissa & 0x1) != 0))) {
            fp19Mantissa += 1;
        }
    }
}
```

**vs. baseline:**
```cpp
# 无round mode选择
```

Benefit: 提供精度和性能的灵活权衡，符合IEEE 754标准
Trade-off: 增加了代码复杂度和测试负担
