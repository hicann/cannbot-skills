# A1: FP32 Intermediate Computation (FP32中间计算)
## Overview
专家实现在DAG中显式地进行了类型转换：输入数据类型U (BF16/FP16) 被转换为T (FP32) 后进行所有计算，最后转换回U输出。这种设计确保了累加精度（AdaGrad中accum的累加操作在FP32上进行，避免BF16/FP16的精度损失）、除法精度（sqrt(accum)和除法操作在FP32上进行，减少数值误差）、梯度精度（梯度平方和乘法在更高精度下进行）。baseline实现全程使用float32，虽然精度足够但失去了BF16/FP16的内存带宽优势。

## When to Use
- BF16/FP16 with precision requirement
- 有效避免FP16/BF16多次累加产生的精度损失，数值稳定性提升
- 精确的adaptive pooling平均因子，符合PyTorch语义，避免固定kernel_volume带来的误差
- 使用低精度存储节省带宽，同时通过FP32计算保持数值精度，特别适合训练场景

## Trade-off
- 需要额外的castBuf和sumBuf(FP32)存储空间
- 需要在Tiling阶段计算每个输出点的kernel大小
- 增加了类型转换操作的开销，但通常远小于内存带宽节省的收益

**Source operators**: adaptive_avg_pool3d, apply_adagrad_d, dynamic_block_quant, foreach_abs, gemma_rms_norm, norm_common, rms_norm_grad, rms_norm_quant, scatter_elements_v2

---

## Variant A: FP16/BF16升精度累加
Source: adaptive_avg_pool3d

**Expert implementation:**
```cpp
// 升精度累加
if constexpr (std::is_same_v<T, float>) {
    Add(sumBufLocal, sumBufLocal, inputLocal, len);
} else {
    LocalTensor<float> castBufLocal = castBuf.Get<float>();
    Cast(castBufLocal, inputLocal, RoundMode::CAST_NONE, len);
    Add(sumBufLocal, sumBufLocal, castBufLocal, len);
}

// 降精度输出
Cast(outputLocal, sumBufLocal, RoundMode::CAST_RINT, len);
```

**vs. baseline:**
```cpp
// baseline仅FP32，无需转换
AscendC::Add(accumLocal, accumLocal, inputLocal, C);
```

Benefit: 有效避免FP16/BF16多次累加产生的精度损失，数值稳定性提升
Trade-off: 需要额外的castBuf和sumBuf(FP32)存储空间

---

## Variant B: 平均因子Host端预计算
Source: adaptive_avg_pool3d

将平均因子计算(1.0f / kernel_volume)放在Host端完成，通过TilingData传递给Kernel。减少Kernel除法运算开销，同时Host端使用更高精度浮点运算计算平均因子，减少累积误差。

**Expert implementation:**
```cpp
// Host端精确计算每个输出点的平均因子
float factor = 1.0f / (static_cast<float>(index.dend - index.dstart) * 
                       (index.hend - index.hstart) * (index.wend - index.wstart));

// Kernel端直接使用
Muls(sumBufLocal, sumBufLocal, factor, count);
```

**vs. baseline:**
```cpp
// baseline预计算固定kernel volume
uint32_t kernel_volume = d_kernel * h_kernel * w_kernel;
float avg_scale = 1.0f / static_cast<float>(kernel_volume);
```

Benefit: 精确的adaptive pooling平均因子，符合PyTorch语义，避免固定kernel_volume带来的误差
Trade-off: 需要在Tiling阶段计算每个输出点的kernel大小

---

## Variant C: 中间计算精度提升
Source: apply_adagrad_d

专家实现在DAG中显式地进行了类型转换：输入数据类型U (BF16/FP16) 被转换为T (FP32) 后进行所有计算，最后转换回U输出。这种设计确保了累加精度（AdaGrad中accum的累加操作在FP32上进行，避免BF16/FP16的精度损失）、除法精度（sqrt(accum)和除法操作在FP32上进行，减少数值误差）、梯度精度（梯度平方和乘法在更高精度下进行）。baseline实现全程使用float32，虽然精度足够但失去了BF16/FP16的内存带宽优势。

**Expert implementation:**
```cpp
// 专家实现 - 类型转换链
using OpCopyInVar = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
using OpVarCast = Bind<Vec::Cast<T, U, 0>, OpCopyInVar>;  // U -> T
using OpGradPower = Bind<Vec::Mul<T>, OpGradCast, OpGradCast>;  // T类型计算
using OpAccumOut = Bind<Vec::Add<T>, OpAccumCast, OpGradPower>;
using OpVarOutCast = Bind<Vec::Cast<U, T, 1>, OpVarOut>;  // T -> U
using OpCopyOutVar = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpVarOutCast>;
```

**vs. baseline:**
```cpp
// baseline全程使用float32
AscendC::LocalTensor<float> varLocal = varQueue.DeQue<float>();
AscendC::LocalTensor<float> accumLocal = accumQueue.DeQue<float>();
AscendC::Mul(gradPowerLocal, gradLocal, gradLocal, this->tileSize);
AscendC::Add(accumOutLocal, accumLocal, gradPowerLocal, this->tileSize);
AscendC::Sqrt(accumSqrtLocal, accumOutLocal, this->tileSize);
AscendC::Div(varTLocal, lrMulGradLocal, accumSqrtLocal, this->tileSize);
```

Benefit: 使用低精度存储节省带宽，同时通过FP32计算保持数值精度，特别适合训练场景
Trade-off: 增加了类型转换操作的开销，但通常远小于内存带宽节省的收益

---

## Variant D: 高精度除法指令
Source: apply_adagrad_d

专家实现使用Vec::DivHighPrecision<T>代替普通的除法操作。在NPU架构中，高精度除法通常采用牛顿迭代法或其他数值稳定算法，相比硬件直接提供的除法指令具有更高的精度，特别是在处理接近零的分母时。这对于AdaGrad算法至关重要，因为sqrt(accum)可能产生非常小的值，导致数值不稳定。baseline使用普通除法指令，在极端情况下可能产生较大误差。

**Expert implementation:**
```cpp
// 专家实现 - 高精度除法
using OpVarT = Bind<Vec::DivHighPrecision<T>, OpLrMulGrad, OpAccumSqrt>;
```

**vs. baseline:**
```cpp
// baseline普通除法
AscendC::Div(varTLocal, lrMulGradLocal, accumSqrtLocal, this->tileSize);
```

Benefit: 在accum值较小时保持数值稳定性，提高训练收敛质量
Trade-off: 高精度除法通常需要更多的指令周期，但在数值敏感场景值得

---

## Variant E: BF16计算精度保护
Source: dynamic_block_quant

专家实现针对bfloat16的低精度特性（7位尾数）采取了特殊的精度保护措施。在ComputeReduceBf16路径中，首先将输入从bf16 cast到float（23位尾数），然后才进行Abs和ReduceMax计算。这是因为reduce操作涉及多次比较和累加，bf16的精度可能导致错误结果。相比之下，fp16有10位尾数，直接在fp16上reduce的精度足够。这种差异化处理确保了在不同输入精度下都能获得正确的量化scale。

**Expert implementation:**
```cpp
// BF16路径
Cast(xLocalTmp, xLocal, RoundMode::CAST_NONE, calcNum);
Abs(xLocalAbs, xLocalTmp, static_cast<int32_t>(calcNum));
// FP16路径
Abs(xLocalAbs, xLocal, calcNum);
```

Benefit: BF16场景下保护reduce精度，避免量化误差
Trade-off: BF16路径需要额外的cast操作，性能略低于FP16路径

---

## Variant F: BF16高精度中间计算
Source: foreach_abs

专家实现在BF16场景下使用FP32进行中间计算。通过InnerComputer<bfloat16_t, float, op, paramsCount>特化版本，将BF16数据先Cast到FP32，执行计算后再Cast回BF16。这是精度优化策略的核心实现。

**Expert implementation:**
```cpp
Cast(float32Tensor, x1Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
op(float32Tensor[offset], float32Tensor, dataCount);
Cast(yLocal[index * maxCastDataCount], float32Tensor[offset], RoundMode::CAST_RINT, dataCount);
```

**vs. baseline:**
```cpp
// 不支持BF16
```

Benefit: BF16场景下保证数值精度，避免低精度计算累积误差
Trade-off: 增加Cast操作开销

---

## Variant G: FP32 累加保护
Source: gemma_rms_norm

专家实现在计算平方和（x^2）时，即使输入是 FP16/BF16，也会先 Cast 到 FP32 再进行乘法累加。这是因为 FP16 的数值范围有限（最大 65504），对于大维度归一化，多个平方值累加容易溢出。baseline实现直接使用输入类型计算，在大数值场景下存在溢出风险。

**Expert implementation:**
```cpp
// 专家实现 - FP32 累加保护
if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
    LocalTensor<float> xBufFp32 = x_fp32_buf.Get<float>();
    Cast(xBufFp32, xLocal, RoundMode::CAST_NONE, num);
    PipeBarrier<PIPE_V>();
    Mul(sqx, xBufFp32, xBufFp32, num);  // FP32 乘法
} else {
    Mul(sqx, xLocal, xLocal, num);
}
```

**vs. baseline:**
```cpp
// baseline - 直接计算
AscendC::Mul(tempLocal, inputLocal, inputLocal, this->tileLength);
// 输入类型直接计算，FP16 可能溢出
```

Benefit: 避免 FP16/BF16 累加溢出，确保大维度归一化的数值稳定性
Trade-off: 需要额外的 FP32 UB 缓冲区，增加内存占用

---

## Variant H: 混合精度计算策略
Source: norm_common

专家实现支持混合精度计算模式——输入/输出使用低精度（FP16/BF16），但中间计算（mean/rstd）使用高精度（FP32）。这种策略在保持较低内存带宽的同时，确保归一化统计量的计算精度。具体实现中，meanGm和rstdGm强制使用float类型，无论输入类型如何。baseline实现全程使用float32，未利用混合精度优化。

**Expert implementation:**
```cpp
// 专家实现：混合精度设计
template <typename Tfm, typename Tweight>
class LayerNormV4SingleRead {
    GlobalTensor<Tfm> xGm;
    GlobalTensor<Tfm> yGm;
    GlobalTensor<Tweight> gammaGm;
    GlobalTensor<Tweight> betaGm;
    GlobalTensor<float> meanGm;  // 强制FP32
    GlobalTensor<float> rstdGm;  // 强制FP32
};

// OpDef定义：mean/rstd输出强制FP32
this->Output("mean").DataType({ge::DT_FLOAT, ge::DT_FLOAT, ...});
this->Output("rstd").DataType({ge::DT_FLOAT, ge::DT_FLOAT, ...});
```

**vs. baseline:**
```cpp
// baseline：全程FP32
GlobalTensor<float> x_gm;
GlobalTensor<float> y_gm;
```

Benefit: 平衡性能与精度，低精度输入/输出减少内存带宽，高精度中间计算保证数值稳定性
Trade-off: 需要额外的类型转换开销，增加代码复杂度

---

## Variant I: 高精度ReduceSum
Source: rms_norm_grad

专家实现提供了专门的ReduceSumFP32_V2函数，该函数使用ACC寄存器来获取reduce结果，支持分块处理大向量，并使用likely/unlikely优化分支预测。相比普通的ReduceSum，这种实现更加高效且精度更高。baseline实现使用普通的ReduceSum，可能在处理大数据量时效率较低。

**Expert implementation:**
```cpp
__aicore__ inline float ReduceSumFP32_V2(const LocalTensor<float>& src_local, int32_t count) {
    int32_t repeatTimes = count / elementNumPerRep;
    if (likely(repeatTimes > 0)) {
        AscendCUtils::SetMask<float>(elementNumPerRep);
        ReduceSum(src_local, src_local, src_local, elementNumPerRep);
        uint64_t acc_val = GetAccVal();
        value = *reinterpret_cast<float*>(&acc_val);
    }
}
```

**vs. baseline:**
```cpp
AscendC::ReduceSum(sumTempLocal, x2Local, sumTempLocal, cols);
```

Benefit: 更高的reduce精度；更好的性能；支持大数据量
Trade-off: 代码复杂度增加

---

## Variant J: FP32中间计算保证精度
Source: rms_norm_quant

专家实现在关键计算步骤中使用FP32作为中间精度，避免FP16累加导致的精度损失。方差计算的平方和累加、归一化除法、量化系数应用等关键步骤都在FP32中完成，仅在最后一步转换为FP16/INT8输出。这种设计遵循输入输出FP16、中间计算FP32的最佳实践。

**Expert implementation:**
```cpp
Cast(fp32_xy, fp16_x, AscendC::RoundMode::CAST_NONE, num_col_);
AscendC::PipeBarrier<PIPE_V>();
Mul(sqx, fp32_xy, fp32_xy, num_col_);
AscendC::PipeBarrier<PIPE_V>();
ReduceSum(sum, sqx, work, num_col_);
float factor = 1 / sum.GetValue(0);
Muls(fp32_xy, fp32_xy, factor, num_col_);
CastFrom32To16(tmpfp16, fp32_xy, num_col_);
```

**vs. baseline:**
```cpp
AscendC::Mul(accumLocal, xLocal, xLocal, count);
float tileSum = ReduceSum(count);
```

Benefit: 避免FP16累加精度损失，数值稳定性更好
Trade-off: UB内存占用增加（FP32是FP16的2倍）

---

## Variant K: 高精度模式支持
Source: rms_norm_quant

专家实现通过high_precision_mode属性支持两种精度模式。在高精度模式下，gamma的乘法和累加操作在FP32精度下进行，仅在最后一步转换为FP16。这对于大模型推理尤为重要——RMSNorm的累加操作涉及大量元素，低精度累加可能导致数值溢出或精度下降。

**Expert implementation:**
```cpp
if (precisionMode == 0) {
    CastFrom16To32(in, gamma, count);
    AscendC::PipeBarrier<PIPE_V>();
    Mul(in, in, tmp, count);
    AscendC::PipeBarrier<PIPE_V>();
    CastFrom32To16(out, in, count);
}
if constexpr (std::is_same<T, half>::value) {
    if (precisionMode_ == 1) {
        CastFrom32To16(out_buf, fp32_xy, numCol_);
        Mul(out_buf, out_buf, fp16_gamma, numColAlignFp16);
    }
}
```

Benefit: 高精度模式下数值稳定性更好，避免大模型推理精度损失
Trade-off: 高精度模式需要更多的类型转换操作

---

## Variant L: 低精度类型的浮点精度提升
Source: scatter_elements_v2

对于half（fp16）和bfloat16_t类型的add操作，专家实现采用'先升精、计算、后降精'的策略。在UB中分配float类型的临时缓冲区，将输入数据和updates数据转换为float后进行累加，最后将结果四舍五入转换回原类型。这种设计充分利用了Vector Unit的float计算能力，避免了fp16/bf16累加的精度损失问题。

**Expert implementation:**
```cpp
// Expert: 精度提升策略
if constexpr (IS_CAST_FLOAT) {
    pipe->InitBuffer(calcSelfBuf, inputAlign * sizeof(float));
    pipe->InitBuffer(calcUpdatesBuf, updatesAlign * sizeof(float));
}

// 类型转换
Cast(inputTemp, inputLocal, RoundMode::CAST_NONE, inputAlign);
Cast(updatesTemp, updatesLocal, RoundMode::CAST_NONE, updatesAlign);

// float累加
inputTemp.SetValue(kIndex, inputTemp.GetValue(kIndex) + updatesTemp.GetValue(k));

// 四舍五入转换回原类型
Cast(inputLocal, inputTemp, RoundMode::CAST_RINT, inputAlign);
```

**vs. baseline:**
```cpp
// Baseline: 直接累加
for (uint64_t j = 0; j < indicesOneTime; ++j) {
    U idx = indicesLocal.GetValue(j);
    T val = updatesLocal.GetValue(j);
    varLocal.SetValue(idx, val);
}
```

Benefit: 避免fp16/bf16累加精度损失，确保数值稳定性
Trade-off: 需要额外UB空间和类型转换开销
