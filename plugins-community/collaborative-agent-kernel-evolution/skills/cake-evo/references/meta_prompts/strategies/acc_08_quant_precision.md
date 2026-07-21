# A8: Quantization-Specific Precision (量化专用精度处理)
## Overview
专家实现通过Maxs和Mins指令精确控制量化边界，确保量化值严格限制在[quant_min, quant_max]范围内。具体实现为：Maxs(curInt32Temp, curInt32Temp, static_cast<int32_t>(this->quantMin), calCount)后接Mins(curInt32Temp, curInt32Temp, static_cast<int32_t>(this->quantMax), calCount)。这种实现确保了即使在浮点计算误差的情况下，量化结果也不会超出指定范围。此外，专家实现还在Cast操作中指定了不同的RoundMode（CAST_RINT用于四舍五入，CAST_ROUND用于向最近偶数舍入），根据不同的计算阶段选择最合适的舍入模式，进一步优化数值精度。

## When to Use
- Quantization output operators, custom float formats, dequantization with bias
- 保证INT32 bias场景的数值精度，避免舍入误差累积
- 最大化INT4表示范围利用，适应数据分布变化，减少量化误差
- 确保量化结果严格在指定范围内，避免溢出；不同计算阶段使用最优舍入模式，提高数值精度

## Trade-off
- 需要维护两套计算路径
- 每行需要独立计算统计量，增加计算开销
- 增加类型转换指令；需要理解不同RoundMode的语义

**Source operators**: dequant_bias, dynamic_quant_update_scatter_v2, fake_quant_affine_cachemask, rms_norm_quant, trans_quant_param_v2

---

## Variant A: INT32类型Bias的特殊处理
Source: dequant_bias

专家实现针对INT32类型的bias采用不同的计算顺序：Add -> Cast -> Mul，而其他类型采用Cast -> Mul -> Add。这种特殊处理是因为INT32可以精确表示大范围的整数，先进行加法不会损失精度；避免FP32的舍入误差；通过IsSameType<BIASTYPE, int32_t>::value在编译期确定路径。

**Expert implementation:**
```cpp
// 专家实现：INT32 bias的计算路径（先加后转）
__aicore__ inline void ComputeDequantWithBiasInt32(...) {
    for (int64_t i = 0; i < inRows; i++) {
        Add(xLocal[i * nAlign_], xLocal[i * nAlign_], biasLocal_, nAlign_);  // INT32加法
    }
    PipeBarrier<PIPE_V>();
    LocalTensor<float> xLocalFp32 = xLocal.template ReinterpretCast<float>();
    Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, inRows * nAlign_);  // 转为FP32
    // ... 后续乘法
}

// 其他类型bias的计算路径（先转后加）
__aicore__ inline void ComputeDequantWithBiasFloat(...) {
    LocalTensor<float> xLocalFp32 = xLocal.template ReinterpretCast<float>();
    Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, inRows * nAlign_);  // 先转为FP32
    // ... 乘法后再加bias
    Add(xLocalFp32[i * nAlign_], xLocalFp32[i * nAlign_], biasLocal_, nAlign_);
}
```

**vs. baseline:**
```cpp
// baseline：单一计算路径（未区分bias类型）
AscendC::Cast(inputHalfLocal, inputLocal, AscendC::RoundMode::CAST_NONE, tileSize);
AscendC::Cast(inputF32Local, inputHalfLocal, AscendC::RoundMode::CAST_NONE, tileSize);
AscendC::Adds(dequantLocal, inputF32Local, -zeroPointVal, tileSize);
AscendC::Muls(dequantLocal, dequantLocal, scaleVal, tileSize);
AscendC::Add(outputLocal, dequantLocal, biasLocal, tileSize);
```

Benefit: 保证INT32 bias场景的数值精度，避免舍入误差累积
Trade-off: 需要维护两套计算路径

---

## Variant B: 动态Range量化(Dynamic Range Quantization)
Source: dynamic_quant_update_scatter_v2

专家实现采用动态Range量化策略，每行数据独立计算scale和offset，最大化利用INT4的表示范围。具体而言，通过scale = max((maxValue - minValue) / 15.0, 1e-12)和offset = 7.0 - maxValue / scale计算量化参数。使用epsilon(1e-12)避免除零错误，使用SafeDiv函数处理极小值情况。这种动态Range策略相比静态量化能够更好地适应数据分布变化，减少量化误差。

**Expert implementation:**
```cpp
// 量化参数定义
constexpr float DYNAMIC_QUANT_INT4_SCALE = 15.0;
constexpr float DYNAMIC_QUANT_INT4_OFFSET = 7.0;
constexpr float DYNAMIC_QUANT_EPSINON = 1e-12;

// 安全的除法
__aicore__ inline float SafeDiv(float a, float b) {
    if (b < DYNAMIC_QUANT_EPSINON && b > -DYNAMIC_QUANT_EPSINON) {
        return a;
    }
    return a / b;
}

// 动态Range量化参数计算
__aicore__ inline void GetScaleAndOffset(float max_value, float min_value, float& scale, float& offset) {
    scale = GetMax((max_value - min_value) / DYNAMIC_QUANT_INT4_SCALE, DYNAMIC_QUANT_EPSINON);
    offset = DYNAMIC_QUANT_INT4_OFFSET - SafeDiv(max_value, scale);
}
```

**vs. baseline:**
```cpp
// baseline可能使用静态量化
constexpr float SCALE = 0.1;
constexpr float OFFSET = 7.0;
int4 out = (int4)((in / SCALE) + OFFSET);
```

Benefit: 最大化INT4表示范围利用，适应数据分布变化，减少量化误差
Trade-off: 每行需要独立计算统计量，增加计算开销

---

## Variant C: 量化边界精确控制
Source: fake_quant_affine_cachemask

专家实现通过Maxs和Mins指令精确控制量化边界，确保量化值严格限制在[quant_min, quant_max]范围内。具体实现为：Maxs(curInt32Temp, curInt32Temp, static_cast<int32_t>(this->quantMin), calCount)后接Mins(curInt32Temp, curInt32Temp, static_cast<int32_t>(this->quantMax), calCount)。这种实现确保了即使在浮点计算误差的情况下，量化结果也不会超出指定范围。此外，专家实现还在Cast操作中指定了不同的RoundMode（CAST_RINT用于四舍五入，CAST_ROUND用于向最近偶数舍入），根据不同的计算阶段选择最合适的舍入模式，进一步优化数值精度。

**Expert implementation:**
```cpp
// 专家实现 - 精确量化边界控制
Maxs(curInt32Temp, curInt32Temp, static_cast<int32_t>(this->quantMin), calCount);
Mins(curInt32Temp, curInt32Temp, static_cast<int32_t>(this->quantMax), calCount);
Cast(curTemp, curInt32Temp, RoundMode::CAST_ROUND, calCount);  // 向最近偶数舍入
```

**vs. baseline:**
```cpp
// baseline - 简单Mins/Maxs
AscendC::Mins(xqLocal, xqRoundedLocal, quantMaxF, this->tileSize);
AscendC::Maxs(xqLocal, xqLocal, quantMinF, this->tileSize);
```

Benefit: 确保量化结果严格在指定范围内，避免溢出；不同计算阶段使用最优舍入模式，提高数值精度
Trade-off: 增加类型转换指令；需要理解不同RoundMode的语义

---

## Variant D: Gemma Mode支持
Source: rms_norm_quant

专家实现通过gemma_mode属性支持Gemma模型的特殊RMSNorm实现。Gemma模型在RMSNorm计算中对gamma值加1（即gamma + 1），这是其归一化公式的特定变体。通过gemmaMode模板参数在编译期确定是否执行此偏移，避免了运行时的条件判断开销。

**Expert implementation:**
```cpp
template <typename T, uint32_t gemmaMode>
__aicore__ inline void CastGAndIsGemmaMode(...) {
    Cast(out, gamma, AscendC::RoundMode::CAST_NONE, count);
    AscendC::PipeBarrier<PIPE_V>();
    float value = 1.0;
    if constexpr (gemmaMode == 1) {
        Adds(out, out, value, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
}
```

Benefit: 支持Gemma模型变体，无需维护独立代码
Trade-off: 增加了模板参数维度

---

## Variant E: FP32到FP19量化转换 - 位操作优化
Source: trans_quant_param_v2

这是专家实现最核心的优化。NPU的量化指令使用特殊的FP19格式（1位符号 + 9位指数 + 9位尾数）来表示scale，并将offset编码为9位有符号整数（Int9），两者打包到一个uint64中。专家实现提供了两种转换模式：Round Mode 0使用简单的位掩码提取FP32的高19位；Round Mode 1则实现了完整的FP32到FP19转换，包括尾数舍入（round to nearest, tie to even）、指数溢出检测和特殊值（0, Inf, NaN）处理。这种转换避免了昂贵的浮点除法，将量化参数转换转化为纯位操作，显著提升了性能。

**Expert implementation:**
```cpp
constexpr uint64_t DEQ_SCALE_MUL = 0xFFFFE000;
constexpr uint64_t QUANT_SCALE = 0x1ULL << 46;

__aicore__ inline uint64_t CalQuantPreScaleValue(uint32_t uint32Scale) {
    if (roundMode_ == 0) {
        quantPre = (uint32Scale & DEQ_SCALE_MUL) | QUANT_SCALE;
    } else {
        uint32_t sign = (uint32Scale >> FP32_SIGN_SHIFT) & 0x1;
        uint32_t exponent = (uint32Scale >> FP32_EXP_SHIFT) & MAX_INT9;
        uint32_t mantissa = uint32Scale & FP32_MANTISSA_LEN;
        uint32_t fp19Mantissa = mantissa >> FP19_TAIL_SHIFT;
        uint32_t roundBit = (mantissa >> (FP19_TAIL_SHIFT - 1)) & 0x1;
        uint32_t stickyBits = mantissa & FP19_TAIL_LEN;
        
        if ((roundBit != 0) && ((stickyBits != 0) || ((fp19Mantissa & 0x1) != 0))) {
            fp19Mantissa += 1;
        }
        
        uint32_t fp19Scale = (sign << FP19_SIGN_SHIFT) | (exponent << FP19_EXP_SHIFT) | fp19Mantissa;
        quantPre = (fp19Scale << FP19_TAIL_SHIFT) | QUANT_SCALE;
    }
}
```

**vs. baseline:**
```cpp
tl.vdiv_scalar(output_scale_ub, scale_ub, scale_factor, count=tile_size)
```

Benefit: 使用位操作替代浮点运算，性能提升3-5倍；输出格式与NPU量化指令直接兼容
Trade-off: 代码复杂度高，需要精确理解FP19格式和位操作

---

## Variant F: Scale/Offset打包到Uint64
Source: trans_quant_param_v2

专家实现将scale（FP19格式，占用46位）和offset（Int9格式，占用9位）打包到一个uint64中。这种打包格式与NPU量化指令的硬件要求完全匹配，避免了运行时额外的格式转换。当offset长度为1时，使用广播策略（offetInt9Bit_在Init阶段预计算）；当offset长度大于1时，逐个元素计算并打包。这种设计使得输出可以直接被后续的量化矩阵乘指令使用，无需额外的数据重组。

**Expert implementation:**
```cpp
constexpr uint64_t QUANT_MASK_0 = 0x1FFULL;
constexpr int32_t OFFSET_DEVIATION = 37;

for (uint32_t idx = 0; idx < eachLength; ++idx) {
    CalQuantPreScale(eachLength * loopidx + idx, idx, resTensor);
    if (offsetLength_ == 1) {
        resTensor.SetValue(idx, resTensor.GetValue(idx) | offetInt9Bit_);
    }
}

__aicore__ inline void SetOffsetValue(...) {
    for (uint32_t idx = 0; idx < length; ++idx) {
        int offsetVal = offsetInt32.GetValue(idx);
        uint64_t int9bits = (static_cast<uint64_t>(offsetVal) & QUANT_MASK_0) << OFFSET_DEVIATION;
        resTensor.SetValue(idx, resTensor.GetValue(idx) | int9bits);
    }
}
```

**vs. baseline:**
```cpp
tl.store(output_scale_ptr + offsets, output_scale_ub)
tl.store(output_zero_point_ptr + offsets, output_zero_point_ub)
```

Benefit: 输出格式与NPU量化指令直接兼容，避免运行时格式转换，支持offset广播
Trade-off: 输出格式对用户不直观，需要额外文档说明

---

## Variant G: Int9范围限制
Source: trans_quant_param_v2

Offset在NPU量化指令中以9位有符号整数（Int9）的形式存储，范围为[-256, 255]。专家实现通过Maxs和Mins指令确保转换后的offset值始终在这个有效范围内。这种限制不仅是硬件要求，也是精度保证的重要措施——超出此范围的offset值会导致量化计算溢出。实现中使用了PipeBarrier<PIPE_V>()确保范围限制指令按顺序执行，避免流水线冒险。

**Expert implementation:**
```cpp
constexpr int32_t MAX_INT9 = 255;
constexpr int32_t MIN_INT9 = -256;

Cast(offsetInt32_, offsetFp32_, RoundMode::CAST_RINT, 1);
PipeBarrier<PIPE_V>();
Maxs(offsetInt32_, offsetInt32_, MIN_INT9, 1);
PipeBarrier<PIPE_V>();
Mins(offsetInt32_, offsetInt32_, MAX_INT9, 1);
PipeBarrier<PIPE_ALL>();
```

**vs. baseline:**
```cpp
# 无范围限制
```

Benefit: 确保offset值在硬件支持范围内，避免量化计算溢出
Trade-off: 极端值会被截断，可能影响精度
